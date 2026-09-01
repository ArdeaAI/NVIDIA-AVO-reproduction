"""
Model backends for Codex OAuth and the optional OpenAI Responses API.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ardea_avo.runtime._io import atomic_write_json, file_lock, sha256_json
from ardea_avo.runtime.budget import (
    CLAUDE_OPUS_5_PRICING,
    GPT_5_6_SOL_PRICING,
    BudgetExceeded,
    BudgetLedger,
    TokenUsage,
)

OPENAI_MODEL = "gpt-5.6-sol"
ANTHROPIC_MODEL = "claude-opus-5"
MODEL = OPENAI_MODEL
REASONING_EFFORT = "max"
OPENAI_MAX_OUTPUT_TOKENS = 65_536
OPENAI_SERVICE_TIER = "default"
ANTHROPIC_MAX_TOKENS = 65_536
SUPPORTED_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_CODEX_ENV_ALLOWLIST = {
    "CODEX_HOME",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
}
_FORBIDDEN_CODEX_CONFIG_KEYS = {
    "approval_policy",
    "model",
    "model_provider",
    "model_reasoning_effort",
    "sandbox",
    "sandbox_mode",
    "sandbox_permissions",
}


class BackendError(RuntimeError):
    """
    Report a provider, process, protocol, or tool execution failure.
    """


class AmbiguousProviderError(BackendError):
    """
    Report a request that may be billable but has no trustworthy usage receipt.

    The caller must retain the request's full reservation and stop automatic
    retries. A later continuation may use other capacity, but the ambiguous
    reservation remains charged against the run ceiling.
    """


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """
    Responses API function tool and its host-owned handler.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any]
    strict: bool = True
    sequential: bool = False


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """
    Backend-neutral input for one autonomous model episode.
    """

    prompt: str
    cwd: Path
    system_prompt: str | None = None
    session_id: str | None = None
    reservation_id: str | None = None
    role: str = "player"
    reasoning_effort: str = REASONING_EFFORT
    sandbox_mode: str = "workspace-write"
    tools: tuple[ToolDefinition, ...] = ()
    max_tool_rounds: int = 128
    estimated_cost_usd: Decimal = Decimal("0")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Reject ambiguous or unsafe request values before launching a backend.
        """

        if not self.prompt.strip():
            raise ValueError("agent prompt cannot be blank")
        if not self.cwd.is_dir():
            raise ValueError(f"agent working directory does not exist: {self.cwd}")
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be positive")
        if self.reservation_id is not None and not self.reservation_id.strip():
            raise ValueError("reservation_id cannot be blank")
        if self.reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of "
                + ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            )
        if self.sandbox_mode not in {"read-only", "workspace-write"}:
            raise ValueError("sandbox_mode must be read-only or workspace-write")
        if self.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd cannot be negative")


@dataclass(frozen=True, slots=True)
class AgentResult:
    """
    Normalized backend output and billable-equivalent usage.
    """

    text: str
    session_id: str | None
    usage: TokenUsage
    cost_usd: Decimal
    tool_rounds: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """
    Non-secret installation and login diagnostics for a backend.
    """

    installed: bool
    authenticated: bool
    version: str | None
    model: str
    messages: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """
        Return whether the backend is installed and authenticated.
        """

        return self.installed and self.authenticated


Runner = Callable[..., subprocess.CompletedProcess[str]]


class CodexOAuthBackend:
    """
    Run Codex non-interactively using its existing ChatGPT login.

    The backend never reads or copies Codex auth files. The child process uses
    the normal Codex authentication mechanism and inherits only the caller's
    process environment.
    """

    def __init__(
        self,
        budget: BudgetLedger,
        *,
        executable: str = "codex",
        timeout_seconds: float | None = None,
        config_overrides: Sequence[str] = (),
        runner: Runner = subprocess.run,
    ) -> None:
        """
        Configure an explicit model, reasoning, and sandboxed Codex invocation.
        """

        self.budget = budget
        self.budget.bind_pricing(GPT_5_6_SOL_PRICING)
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.config_overrides = tuple(config_overrides)
        for override in self.config_overrides:
            key, separator, _value = override.partition("=")
            normalized = key.strip().casefold()
            if not separator or not normalized:
                raise ValueError("Codex config overrides must use key=value syntax")
            if normalized in _FORBIDDEN_CODEX_CONFIG_KEYS:
                raise ValueError(f"Codex config cannot override protected key: {key}")
            if any(
                marker in normalized
                for marker in ("api_key", "access_token", "password", "secret")
            ):
                raise ValueError("Codex config cannot contain credential fields")
        self._runner = runner

    def status(self) -> BackendStatus:
        """
        Check Codex installation and saved-login status without a model call.
        """

        try:
            version_result = self._runner(
                [self.executable, "--version"],
                capture_output=True,
                env=_sanitized_codex_environment(),
                text=True,
                timeout=15,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as error:
            return BackendStatus(
                installed=False,
                authenticated=False,
                version=None,
                model=MODEL,
                messages=(str(error),),
            )
        if version_result.returncode != 0:
            message = (version_result.stderr or version_result.stdout).strip()
            return BackendStatus(
                installed=False,
                authenticated=False,
                version=None,
                model=MODEL,
                messages=(message or "codex --version failed",),
            )
        version = version_result.stdout.strip() or version_result.stderr.strip()
        try:
            login_result = self._runner(
                [self.executable, "login", "status"],
                capture_output=True,
                env=_sanitized_codex_environment(),
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.SubprocessError as error:
            return BackendStatus(
                installed=True,
                authenticated=False,
                version=version,
                model=MODEL,
                messages=(str(error),),
            )
        login_text = f"{login_result.stdout}\n{login_result.stderr}".strip()
        authenticated = login_result.returncode == 0 and bool(
            re.search(r"\bchatgpt\b", login_text, flags=re.IGNORECASE)
        )
        messages: tuple[str, ...] = ()
        if not authenticated:
            messages = (
                login_text
                or "Codex is not logged in with the required ChatGPT OAuth method",
            )
        return BackendStatus(
            installed=True,
            authenticated=authenticated,
            version=version,
            model=MODEL,
            messages=messages,
        )

    def doctor(
        self,
        *,
        probe_model: bool = False,
        cwd: Path | None = None,
    ) -> BackendStatus:
        """
        Run status checks and optionally make a budgeted model probe.
        """

        status = self.status()
        if not probe_model or not status.ok:
            return status
        if cwd is None:
            raise ValueError("cwd is required for a live model probe")
        try:
            result = self.run(
                AgentRequest(
                    prompt="Reply with exactly READY.",
                    cwd=cwd,
                    role="doctor",
                )
            )
        except (BackendError, RuntimeError) as error:
            return BackendStatus(
                installed=True,
                authenticated=True,
                version=status.version,
                model=MODEL,
                messages=(f"model probe failed: {error}",),
            )
        if result.text.strip() != "READY":
            return BackendStatus(
                installed=True,
                authenticated=True,
                version=status.version,
                model=MODEL,
                messages=("model probe returned an unexpected response",),
            )
        return status

    def run(self, request: AgentRequest) -> AgentResult:
        """
        Execute or resume one Codex turn and account for its JSONL usage.
        """

        if request.tools:
            raise ValueError(
                "Codex tools must be registered through explicit MCP config overrides"
            )
        self.budget.ensure_can_start(request.estimated_cost_usd)
        command = self._command(request)
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        try:
            completed = self._runner(
                command,
                input=prompt,
                cwd=request.cwd,
                capture_output=True,
                env=_sanitized_codex_environment(),
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            self._release_failed_reservation(request, "Codex executable not found")
            raise BackendError(f"Codex executable not found: {self.executable}") from error
        except subprocess.TimeoutExpired as error:
            self._release_failed_reservation(request, "Codex host timeout")
            raise BackendError("Codex turn exceeded its host timeout") from error
        if completed.returncode != 0:
            self._release_failed_reservation(request, "Codex process failed")
            detail = completed.stderr.strip() or "no stderr was returned"
            raise BackendError(
                f"Codex exited with status {completed.returncode}: {detail}"
            )
        try:
            text, session_id, usage, warnings = parse_codex_jsonl(completed.stdout)
        except BackendError:
            self._release_failed_reservation(request, "Codex event stream failed")
            raise
        if session_id is None and request.session_id is None:
            self._release_failed_reservation(request, "Codex session id missing")
            raise BackendError("Codex did not return a resumable thread id")
        cost = self.budget.record_usage(
            usage,
            backend="codex-oauth",
            role=request.role,
            session_id=session_id,
            reservation_id=request.reservation_id,
            metadata=dict(request.metadata),
        )
        return AgentResult(
            text=text,
            session_id=session_id or request.session_id,
            usage=usage,
            cost_usd=cost,
            warnings=warnings,
        )

    def _command(self, request: AgentRequest) -> list[str]:
        command = [
            self.executable,
            "--ask-for-approval",
            "never",
            "exec",
            "--json",
            "--color",
            "never",
            "--model",
            MODEL,
            "--sandbox",
            request.sandbox_mode,
            "--cd",
            str(request.cwd.resolve()),
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--config",
            f'model_reasoning_effort="{request.reasoning_effort}"',
        ]
        for override in self.config_overrides:
            command.extend(("--config", override))
        if request.session_id:
            command.extend(("resume", request.session_id, "-"))
        else:
            command.append("-")
        return command

    def _release_failed_reservation(self, request: AgentRequest, reason: str) -> None:
        if request.reservation_id is not None:
            self.budget.release(request.reservation_id, reason=reason)


def _sanitized_codex_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Retain only OS/auth-location essentials, excluding provider credentials.

    In particular, ARC_API_KEY, OPENAI_API_KEY, CODEX_API_KEY, bearer tokens,
    and conventionally named secret/password variables never reach the Codex
    process or the shell commands it launches.
    """

    environment = os.environ if source is None else source
    return {
        key: value
        for key, value in environment.items()
        if key in _CODEX_ENV_ALLOWLIST
    }


def parse_codex_jsonl(
    output: str,
) -> tuple[str, str | None, TokenUsage, tuple[str, ...]]:
    """
    Parse Codex JSONL across known event-shape variations.
    """

    messages: list[str] = []
    warnings: list[str] = []
    session_id: str | None = None
    recognized_usage: list[TokenUsage] = []
    fallback_usage: list[TokenUsage] = []
    valid_events = 0
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"ignored non-JSON Codex output line {line_number}")
            continue
        if not isinstance(event, dict):
            warnings.append(f"ignored non-object Codex event line {line_number}")
            continue
        valid_events += 1
        event_type = str(event.get("type", ""))
        candidate_session = _session_id_from_event(event)
        if candidate_session:
            session_id = candidate_session
        message = _message_from_event(event)
        if message:
            messages.append(message)
        usage = _usage_from_event(event)
        if usage is not None:
            if event_type in {"turn.completed", "response.completed"}:
                recognized_usage.append(usage)
            else:
                fallback_usage.append(usage)
        if event_type in {"error", "turn.failed", "response.failed"}:
            detail = event.get("message") or event.get("error") or event
            raise BackendError(f"Codex reported a failed event: {detail}")
    selected_usage = recognized_usage or fallback_usage[-1:]
    if valid_events == 0:
        raise BackendError("Codex returned no valid JSONL events")
    if not selected_usage:
        raise BackendError("Codex event stream did not contain token usage")
    total_usage = TokenUsage()
    for usage in selected_usage:
        total_usage += usage
    return "\n".join(messages).strip(), session_id, total_usage, tuple(warnings)


def _session_id_from_event(event: Mapping[str, Any]) -> str | None:
    for key in ("thread_id", "session_id", "conversation_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    thread = event.get("thread")
    if isinstance(thread, Mapping):
        value = thread.get("id")
        if isinstance(value, str) and value:
            return value
    return None


def _message_from_event(event: Mapping[str, Any]) -> str | None:
    item = event.get("item")
    if isinstance(item, Mapping) and item.get("type") in {
        "agent_message",
        "message",
    }:
        for key in ("text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    if event.get("type") in {"agent_message", "message"}:
        value = event.get("text") or event.get("content")
        if isinstance(value, str) and value:
            return value
    return None


def _usage_from_event(event: Mapping[str, Any]) -> TokenUsage | None:
    usage = event.get("usage")
    if not isinstance(usage, Mapping):
        response = event.get("response")
        if isinstance(response, Mapping):
            usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("input_tokens_details")
    cached = usage.get("cached_input_tokens", 0)
    if isinstance(details, Mapping):
        cached = details.get("cached_tokens", cached)
    try:
        return TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0)),
            cached_input_tokens=int(cached or 0),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
    except (TypeError, ValueError) as error:
        raise BackendError("Codex returned invalid token usage") from error


class OpenAIResponsesBackend:
    """
    Optional direct Responses API backend with host-owned function tools.

    Importing this module does not require the OpenAI package. The SDK is
    imported only when this backend is constructed without an injected client.
    """

    def __init__(
        self,
        budget: BudgetLedger,
        *,
        client: Any | None = None,
        model: str = MODEL,
        reasoning_effort: str = REASONING_EFFORT,
    ) -> None:
        """
        Configure the API backend without reading or persisting credentials.
        """

        if model != OPENAI_MODEL:
            raise ValueError(f"unsupported OpenAI model: {model}")
        self.budget = budget
        self.budget.bind_pricing(GPT_5_6_SOL_PRICING)
        self.model = model
        self.reasoning_effort = reasoning_effort
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise BackendError(
                    "the optional 'openai' package is required for --backend openai-api"
                ) from error
            client = OpenAI(max_retries=0)
        self.client = client

    def run(self, request: AgentRequest) -> AgentResult:
        """
        Execute a Responses API function-call loop under the shared budget.
        """

        tools_by_name = {tool.name: tool for tool in request.tools}
        if len(tools_by_name) != len(request.tools):
            raise ValueError("tool names must be unique")
        schemas = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
                "strict": tool.strict,
            }
            for tool in request.tools
        ]
        next_input: Any = request.prompt
        previous_response_id = request.session_id
        total_usage = TokenUsage()
        total_cost = Decimal("0")
        warnings: list[str] = []
        last_response_id = previous_response_id
        current_reservation, reservation_amount = _initial_reservation(
            self.budget,
            request.reservation_id,
        )
        for round_number in range(1, request.max_tool_rounds + 1):
            self.budget.ensure_can_start(request.estimated_cost_usd)
            arguments: dict[str, Any] = {
                "model": self.model,
                "input": next_input,
                "reasoning": {"effort": request.reasoning_effort},
                "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
                "service_tier": OPENAI_SERVICE_TIER,
            }
            if request.system_prompt:
                arguments["instructions"] = request.system_prompt
            if schemas:
                arguments["tools"] = schemas
                arguments["parallel_tool_calls"] = False
            if previous_response_id:
                arguments["previous_response_id"] = previous_response_id
            try:
                response = self.client.responses.create(**arguments)
            except Exception as error:
                raise AmbiguousProviderError(
                    "OpenAI Responses request outcome is ambiguous; "
                    "the full reservation remains held"
                ) from error
            response_id = _attribute(response, "id")
            response_model = _attribute(response, "model")
            last_response_id = str(response_id) if response_id else last_response_id
            try:
                usage = _responses_usage(_attribute(response, "usage"))
            except BackendError as error:
                raise AmbiguousProviderError(
                    "OpenAI response lacks trustworthy usage; "
                    "the full reservation remains held"
                ) from error
            total_usage += usage
            protocol_error: BackendError | None = None
            calls: list[tuple[str, str, str]] = []
            try:
                if not response_id:
                    raise BackendError("OpenAI Responses result lacks an id")
                if response_model != self.model:
                    raise BackendError("OpenAI Responses result came from an unexpected model")
                _validate_responses_status(response)
                calls = _responses_function_calls(response)
                _reject_parallel_sequential_calls(calls, tools_by_name)
            except BackendError as error:
                protocol_error = error
            try:
                total_cost += self.budget.record_usage(
                    usage,
                    backend="openai-api",
                    role=request.role,
                    session_id=(str(response_id) if response_id else None),
                    reservation_id=current_reservation,
                    metadata={
                        **dict(request.metadata),
                        "tool_round": round_number,
                        **(
                            {"terminal_error": True}
                            if protocol_error is not None
                            else {}
                        ),
                    },
                )
            except Exception as error:
                raise AmbiguousProviderError(
                    "OpenAI usage receipt could not be committed; "
                    "the full reservation remains held"
                ) from error
            current_reservation = None
            if protocol_error is not None:
                raise protocol_error
            if not calls:
                text = _responses_text(response)
                return AgentResult(
                    text=text,
                    session_id=(str(response_id) if response_id else previous_response_id),
                    usage=total_usage,
                    cost_usd=total_cost,
                    tool_rounds=round_number - 1,
                    warnings=tuple(warnings),
                )
            outputs: list[dict[str, str]] = []
            for call_id, name, raw_arguments in calls:
                tool = tools_by_name.get(name)
                if tool is None:
                    result: Any = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        parsed_arguments = json.loads(raw_arguments or "{}")
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError("tool arguments must be a JSON object")
                        result = tool.handler(**parsed_arguments)
                    except Exception as error:
                        result = {"error": f"{type(error).__name__}: {error}"}
                try:
                    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
                except (TypeError, ValueError):
                    serialized = json.dumps(
                        {"error": "tool returned a non-JSON-serializable value"}
                    )
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": serialized,
                    }
                )
            next_input = outputs
            previous_response_id = str(response_id) if response_id else previous_response_id
            if round_number < request.max_tool_rounds:
                current_reservation = _reserve_followup(
                    self.budget,
                    reservation_amount,
                    role=request.role,
                )
        raise BackendError(
            f"Responses tool loop exceeded {request.max_tool_rounds} rounds"
        )


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _initial_reservation(
    budget: BudgetLedger,
    reservation_id: str | None,
) -> tuple[str | None, Decimal | None]:
    if reservation_id is None:
        return None, None
    amount = budget.active_reservations().get(reservation_id)
    if amount is None:
        raise KeyError(f"unknown or closed reservation: {reservation_id}")
    return reservation_id, amount


def _reserve_followup(
    budget: BudgetLedger,
    amount: Decimal | None,
    *,
    role: str,
) -> str | None:
    if amount is None:
        return None
    available = budget.snapshot().available_usd
    if available < amount:
        raise BudgetExceeded(
            "insufficient budget for another provider round: "
            f"available=${available}, required=${amount}"
        )
    return budget.reserve(amount, role=role)


def _release_reservation(
    budget: BudgetLedger,
    reservation_id: str | None,
    *,
    reason: str,
) -> None:
    if reservation_id is not None:
        with suppress(KeyError):
            budget.release(reservation_id, reason=reason)


def _validate_responses_status(response: Any) -> None:
    status = _attribute(response, "status")
    error = _attribute(response, "error")
    incomplete = _attribute(response, "incomplete_details")
    if status != "completed" or error is not None or incomplete is not None:
        detail = error if error is not None else incomplete
        raise BackendError(
            f"OpenAI Responses result was not complete: status={status!r}, detail={detail!r}"
        )


def _responses_usage(value: Any) -> TokenUsage:
    if value is None:
        raise BackendError("Responses API result lacks token usage")

    def count(source: Any, name: str) -> int:
        raw = _attribute(source, name, 0) or 0
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise BackendError(f"Responses API returned invalid {name}")
        return raw

    input_tokens = count(value, "input_tokens")
    output_tokens = count(value, "output_tokens")
    details = _attribute(value, "input_tokens_details")
    cached_tokens = count(details, "cached_tokens")
    cache_write_tokens = count(details, "cache_write_tokens")
    try:
        return TokenUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            cache_creation_input_tokens=cache_write_tokens,
            output_tokens=output_tokens,
        )
    except (TypeError, ValueError) as error:
        raise BackendError("Responses API returned invalid token usage") from error


def _responses_function_calls(response: Any) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    for item in _attribute(response, "output", ()) or ():
        if _attribute(item, "type") != "function_call":
            continue
        call_id = _attribute(item, "call_id") or _attribute(item, "id")
        name = _attribute(item, "name")
        arguments = _attribute(item, "arguments", "{}")
        if not call_id or not name:
            raise BackendError("Responses API returned a malformed function call")
        calls.append((str(call_id), str(name), str(arguments)))
    return calls


def _responses_text(response: Any) -> str:
    output_text = _attribute(response, "output_text")
    if isinstance(output_text, str):
        return output_text
    fragments: list[str] = []
    for item in _attribute(response, "output", ()) or ():
        if _attribute(item, "type") != "message":
            continue
        for content in _attribute(item, "content", ()) or ():
            if _attribute(content, "type") in {"output_text", "text"}:
                text = _attribute(content, "text")
                if isinstance(text, str):
                    fragments.append(text)
    return "\n".join(fragments)


class AnthropicMessagesBackend:
    """
    Run Claude through the stateless Messages API with durable host transcripts.

    Anthropic message identifiers are not resumable conversations. This backend
    therefore returns an opaque local session identifier and stores the complete
    message history outside the model-writable workspace. Assistant content is
    round-tripped losslessly so signed thinking blocks remain valid.
    """

    def __init__(
        self,
        budget: BudgetLedger,
        *,
        client: Any | None = None,
        model: str = ANTHROPIC_MODEL,
        max_tokens: int = ANTHROPIC_MAX_TOKENS,
        session_dir: Path | None = None,
    ) -> None:
        """
        Configure a pinned Claude model and a private durable transcript store.
        """

        if model != ANTHROPIC_MODEL:
            raise ValueError(f"unsupported Anthropic model: {model}")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Anthropic max_tokens must be a positive integer")
        self.budget = budget
        self.budget.bind_pricing(CLAUDE_OPUS_5_PRICING)
        self.model = model
        self.max_tokens = max_tokens
        self.session_dir = (
            session_dir.resolve()
            if session_dir is not None
            else (budget.directory / "provider-sessions" / "anthropic").resolve()
        )
        self.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise BackendError(
                    "the 'anthropic' package is required for --backend anthropic-api"
                ) from error
            client = Anthropic(max_retries=0)
        self.client = client

    def run(self, request: AgentRequest) -> AgentResult:
        """
        Execute a serial Messages API tool loop and durably retain its context.
        """

        if request.reasoning_effort == "xhigh":
            raise ValueError("Anthropic reasoning effort does not support xhigh")
        tools_by_name = {tool.name: tool for tool in request.tools}
        if len(tools_by_name) != len(request.tools):
            raise ValueError("tool names must be unique")
        schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.parameters),
                "strict": tool.strict,
            }
            for tool in request.tools
        ]
        session_id = request.session_id or self._new_session_id()
        session_path, lock_path = self._session_paths(session_id)
        total_usage = TokenUsage()
        total_cost = Decimal("0")
        current_reservation, reservation_amount = _initial_reservation(
            self.budget,
            request.reservation_id,
        )
        try:
            with file_lock(lock_path):
                document = self._load_or_initialize_session(
                    session_path,
                    session_id=session_id,
                    request=request,
                    schemas=schemas,
                )
                messages = list(document["messages"])
                messages.append({"role": "user", "content": request.prompt})
                for round_number in range(1, request.max_tool_rounds + 1):
                    self.budget.ensure_can_start(request.estimated_cost_usd)
                    arguments: dict[str, Any] = {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "messages": messages,
                        "inference_geo": "global",
                        "thinking": {"type": "adaptive"},
                        "cache_control": {"type": "ephemeral"},
                        "output_config": {"effort": request.reasoning_effort},
                    }
                    if request.system_prompt:
                        arguments["system"] = request.system_prompt
                    if schemas:
                        arguments["tools"] = schemas
                        arguments["tool_choice"] = {
                            "type": "auto",
                            "disable_parallel_tool_use": True,
                        }
                    try:
                        with self.client.messages.stream(**arguments) as stream:
                            response = stream.get_final_message()
                    except Exception as error:
                        raise AmbiguousProviderError(
                            "Anthropic Messages request outcome is ambiguous; "
                            "the full reservation remains held"
                        ) from error

                    try:
                        usage = _anthropic_usage(_attribute(response, "usage"))
                    except BackendError as error:
                        raise AmbiguousProviderError(
                            "Anthropic response lacks trustworthy usage; "
                            "the full reservation remains held"
                        ) from error
                    total_usage += usage
                    provider_message_id = _attribute(response, "id")
                    provider_model = _attribute(response, "model")
                    protocol_error: BackendError | None = None
                    blocks: list[dict[str, Any]] = []
                    calls: list[tuple[str, str, Mapping[str, Any]]] = []
                    try:
                        if not isinstance(provider_message_id, str) or not provider_message_id:
                            raise BackendError("Anthropic response lacks a message id")
                        if provider_model != self.model:
                            raise BackendError("Anthropic response came from an unexpected model")
                        blocks = _anthropic_content_blocks(response)
                        calls = _anthropic_tool_calls(blocks)
                        stop_reason = _attribute(response, "stop_reason")
                        if calls and stop_reason != "tool_use":
                            raise BackendError(
                                "Anthropic returned tool calls without a tool_use stop reason"
                            )
                        if not calls and stop_reason != "end_turn":
                            raise BackendError(
                                "Anthropic turn ended with unsupported stop reason: "
                                f"{stop_reason!r}"
                            )
                        _reject_parallel_sequential_calls(calls, tools_by_name)
                    except BackendError as error:
                        protocol_error = error
                    try:
                        total_cost += self.budget.record_usage(
                            usage,
                            backend="anthropic-api",
                            role=request.role,
                            session_id=session_id,
                            reservation_id=current_reservation,
                            metadata={
                                **dict(request.metadata),
                                "tool_round": round_number,
                                **(
                                    {"provider_message_id": provider_message_id}
                                    if isinstance(provider_message_id, str)
                                    else {}
                                ),
                                **(
                                    {"terminal_error": True}
                                    if protocol_error is not None
                                    else {}
                                ),
                            },
                        )
                    except Exception as error:
                        raise AmbiguousProviderError(
                            "Anthropic usage receipt could not be committed; "
                            "the full reservation remains held"
                        ) from error
                    current_reservation = None
                    if protocol_error is not None:
                        raise protocol_error
                    assert isinstance(provider_message_id, str)
                    messages.append({"role": "assistant", "content": blocks})
                    if calls:
                        results = [
                            _anthropic_tool_result(call_id, name, raw_input, tools_by_name)
                            for call_id, name, raw_input in calls
                        ]
                        messages.append({"role": "user", "content": results})
                        document = self._save_session(
                            session_path,
                            document,
                            messages,
                            provider_message_id,
                        )
                        if round_number < request.max_tool_rounds:
                            current_reservation = _reserve_followup(
                                self.budget,
                                reservation_amount,
                                role=request.role,
                            )
                        continue
                    self._save_session(
                        session_path,
                        document,
                        messages,
                        provider_message_id,
                    )
                    return AgentResult(
                        text=_anthropic_text(blocks),
                        session_id=session_id,
                        usage=total_usage,
                        cost_usd=total_cost,
                        tool_rounds=round_number - 1,
                    )
                raise BackendError(
                    f"Anthropic tool loop exceeded {request.max_tool_rounds} rounds"
                )
        except Exception as error:
            if not isinstance(error, AmbiguousProviderError):
                _release_reservation(
                    self.budget,
                    current_reservation,
                    reason="Anthropic backend terminated before another response",
                )
            if isinstance(error, (BackendError, RuntimeError, ValueError)):
                raise
            raise BackendError(f"Anthropic session failed: {error}") from error

    def _new_session_id(self) -> str:
        while True:
            candidate = str(uuid4())
            path, _lock = self._session_paths(candidate)
            if not path.exists():
                return candidate

    def _session_paths(self, session_id: str) -> tuple[Path, Path]:
        try:
            parsed = UUID(session_id)
        except (TypeError, ValueError) as error:
            raise BackendError("Anthropic session identifier is invalid") from error
        if str(parsed) != session_id:
            raise BackendError("Anthropic session identifier is not canonical")
        path = self.session_dir / f"{session_id}.json"
        if path.is_symlink():
            raise BackendError("Anthropic session path cannot be a symbolic link")
        return path, self.session_dir / f".{session_id}.lock"

    def _load_or_initialize_session(
        self,
        path: Path,
        *,
        session_id: str,
        request: AgentRequest,
        schemas: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        contract = _anthropic_session_contract(
            session_id=session_id,
            model=self.model,
            request=request,
            schemas=schemas,
            max_tokens=self.max_tokens,
        )
        if request.session_id is None:
            if path.exists():
                raise BackendError("fresh Anthropic session path already exists")
            return {
                **contract,
                "revision": 0,
                "messages": [],
                "provider_message_ids": [],
            }
        if not path.is_file():
            raise BackendError("Anthropic session transcript is missing")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BackendError("Anthropic session transcript is unreadable") from error
        if not isinstance(document, dict):
            raise BackendError("Anthropic session transcript is malformed")
        expected_fields = {
            *contract,
            "revision",
            "messages",
            "provider_message_ids",
            "sha256",
        }
        if set(document) != expected_fields:
            raise BackendError("Anthropic session transcript fields are invalid")
        digest = document.pop("sha256")
        if not isinstance(digest, str) or digest != sha256_json(document):
            raise BackendError("Anthropic session transcript digest mismatch")
        if any(document.get(key) != value for key, value in contract.items()):
            raise BackendError("Anthropic session contract changed during resume")
        revision = document.get("revision")
        messages = document.get("messages")
        provider_message_ids = document.get("provider_message_ids")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
            or not isinstance(messages, list)
            or not isinstance(provider_message_ids, list)
            or len(provider_message_ids) != revision
            or not all(
                isinstance(value, str) and value for value in provider_message_ids
            )
        ):
            raise BackendError("Anthropic session transcript state is invalid")
        return document

    @staticmethod
    def _save_session(
        path: Path,
        document: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        provider_message_id: str,
    ) -> dict[str, Any]:
        revision = document.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise BackendError("Anthropic session revision is invalid")
        body = {
            key: value
            for key, value in document.items()
            if key not in {
                "revision",
                "messages",
                "provider_message_ids",
                "sha256",
            }
        }
        body["revision"] = revision + 1
        body["messages"] = list(messages)
        raw_ids = document.get("provider_message_ids", ())
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise BackendError("Anthropic provider message receipts are invalid")
        body["provider_message_ids"] = [*raw_ids, provider_message_id]
        stored = {**body, "sha256": sha256_json(body)}
        atomic_write_json(path, stored)
        return body


def _anthropic_session_contract(
    *,
    session_id: str,
    model: str,
    request: AgentRequest,
    schemas: Sequence[Mapping[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "schema": "ardea.anthropic-session.v1",
        "session_id": session_id,
        "model": model,
        "role": request.role,
        "reasoning_effort": request.reasoning_effort,
        "max_tokens": max_tokens,
        "thinking": "adaptive",
        "inference_geo": "global",
        "cache_ttl": "ephemeral-5m",
        "cwd": str(request.cwd.resolve()),
        "system_prompt_sha256": sha256_json(request.system_prompt or ""),
        "tool_schemas_sha256": sha256_json(list(schemas)),
    }


def _json_compatible(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise BackendError("Anthropic returned non-JSON message content") from error


def _anthropic_content_blocks(response: Any) -> list[dict[str, Any]]:
    raw_content = _attribute(response, "content")
    if not isinstance(raw_content, Sequence) or isinstance(raw_content, (str, bytes)):
        raise BackendError("Anthropic response lacks content blocks")
    blocks: list[dict[str, Any]] = []
    for value in raw_content:
        block = _json_compatible(value)
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            raise BackendError("Anthropic returned a malformed content block")
        blocks.append(block)
    if not blocks:
        raise BackendError("Anthropic returned no content blocks")
    return blocks


def _anthropic_usage(value: Any) -> TokenUsage:
    if value is None:
        raise BackendError("Anthropic response lacks token usage")

    def count(name: str) -> int:
        raw = _attribute(value, name, 0) or 0
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise BackendError(f"Anthropic returned invalid {name}")
        return raw

    base_input = count("input_tokens")
    cache_creation = count("cache_creation_input_tokens")
    cache_read = count("cache_read_input_tokens")
    return TokenUsage(
        input_tokens=base_input + cache_creation + cache_read,
        cached_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        output_tokens=count("output_tokens"),
    )


def _anthropic_tool_calls(
    blocks: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, Mapping[str, Any]]]:
    calls: list[tuple[str, str, Mapping[str, Any]]] = []
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        call_id = block.get("id")
        name = block.get("name")
        arguments = block.get("input")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(name, str)
            or not name
            or not isinstance(arguments, Mapping)
        ):
            raise BackendError("Anthropic returned a malformed tool call")
        calls.append((call_id, name, arguments))
    return calls


def _reject_parallel_sequential_calls(
    calls: Sequence[tuple[str, str, Any]],
    tools_by_name: Mapping[str, ToolDefinition],
) -> None:
    if len(calls) > 1 and any(
        (tool := tools_by_name.get(name)) is not None and tool.sequential
        for _call_id, name, _arguments in calls
    ):
        raise BackendError("provider returned parallel calls for a sequential tool")


def _anthropic_tool_result(
    call_id: str,
    name: str,
    raw_input: Mapping[str, Any],
    tools_by_name: Mapping[str, ToolDefinition],
) -> dict[str, Any]:
    tool = tools_by_name.get(name)
    is_error = False
    if tool is None:
        result: Any = {"error": f"unknown tool: {name}"}
        is_error = True
    else:
        try:
            result = tool.handler(**dict(raw_input))
            if isinstance(result, Mapping) and result.get("isError") is True:
                is_error = True
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
            is_error = True
    try:
        content = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        content = json.dumps({"error": "tool returned a non-JSON-serializable value"})
        is_error = True
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
        "is_error": is_error,
    }


def _anthropic_text(blocks: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        str(block["text"])
        for block in blocks
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )
