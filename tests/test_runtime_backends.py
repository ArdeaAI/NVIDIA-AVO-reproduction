"""
Tests for model backend command and protocol boundaries.
"""

import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from ardea_avo.runtime.backends import (
    AgentRequest,
    AmbiguousProviderError,
    AnthropicMessagesBackend,
    BackendError,
    CodexOAuthBackend,
    OpenAIResponsesBackend,
    ToolDefinition,
    _anthropic_tool_result,
    _sanitized_codex_environment,
    parse_codex_jsonl,
)
from ardea_avo.runtime.budget import CLAUDE_OPUS_5_PRICING, BudgetLedger, TokenUsage


def test_codex_jsonl_parser_handles_session_message_and_usage() -> None:
    """
    Normalized output is extracted from the documented event stream.
    """

    output = "\n".join(
        (
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
            '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":4,"output_tokens":3}}',
        )
    )
    text, session_id, usage, warnings = parse_codex_jsonl(output)
    assert text == "done"
    assert session_id == "thread-1"
    assert usage == TokenUsage(input_tokens=10, cached_input_tokens=4, output_tokens=3)
    assert warnings == ()


def test_codex_command_is_explicit_and_resume_uses_stdin(tmp_path) -> None:
    """
    OAuth execution fixes model, reasoning, sandbox, cwd, and resume identity.
    """

    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        stdout = "\n".join(
            (
                '{"type":"thread.started","thread_id":"session-new"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
                '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":2}}',
            )
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    ledger = BudgetLedger(tmp_path / "budget")
    reservation = ledger.reserve("1", role="player")
    backend = CodexOAuthBackend(ledger, runner=runner)
    result = backend.run(
        AgentRequest(
            prompt="continue",
            cwd=tmp_path,
            session_id="session-old",
            reservation_id=reservation,
        )
    )
    command, keyword_arguments = calls[0]
    assert command[:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert "--json" in command
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--ask-for-approval") + 1] == "never"
    assert 'model_reasoning_effort="max"' in command
    assert command[-3:] == ["resume", "session-old", "-"]
    assert keyword_arguments["input"] == "continue"
    assert result.session_id == "session-new"
    assert ledger.snapshot().reserved_usd == 0


@pytest.mark.parametrize(
    ("session_id", "suffix"),
    ((None, ["-"]), ("session-old", ["resume", "session-old", "-"])),
)
def test_codex_fresh_and_resume_argv_are_exact(
    tmp_path, session_id: str | None, suffix: list[str]
) -> None:
    """
    Global approval policy is placed before the exec subcommand for both modes.
    """

    backend = CodexOAuthBackend(BudgetLedger(tmp_path / f"budget-{session_id}"))
    request = AgentRequest(prompt="x", cwd=tmp_path, session_id=session_id)
    assert backend._command(request) == [
        "codex",
        "--ask-for-approval",
        "never",
        "exec",
        "--json",
        "--color",
        "never",
        "--model",
        "gpt-5.6-sol",
        "--sandbox",
        "workspace-write",
        "--cd",
        str(tmp_path.resolve()),
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--config",
        'model_reasoning_effort="max"',
        *suffix,
    ]


def test_request_reasoning_effort_is_honored_by_codex(tmp_path) -> None:
    """
    Supervisor calls can lower effort to high while players default to max.
    """

    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                (
                    '{"type":"thread.started","thread_id":"supervisor-thread"}',
                    '{"type":"turn.completed","usage":{}}',
                )
            ),
            stderr="",
        )

    backend = CodexOAuthBackend(BudgetLedger(tmp_path / "budget"), runner=runner)
    backend.run(
        AgentRequest(
            prompt="suggest directions",
            cwd=tmp_path,
            role="supervisor",
            reasoning_effort="high",
        )
    )
    assert 'model_reasoning_effort="high"' in calls[0]
    with pytest.raises(ValueError, match="reasoning_effort"):
        AgentRequest(prompt="x", cwd=tmp_path, reasoning_effort="unbounded")
    with pytest.raises(ValueError, match="sandbox_mode"):
        AgentRequest(prompt="x", cwd=tmp_path, sandbox_mode="danger-full-access")
    with pytest.raises(ValueError, match="reservation_id"):
        AgentRequest(prompt="x", cwd=tmp_path, reservation_id="  ")


def test_codex_environment_excludes_every_host_credential() -> None:
    """
    Agent-controlled shell commands receive only explicitly permitted variables.
    """

    source = {
        "HOME": "/home/user",
        "PATH": "/bin",
        "CODEX_HOME": "/codex",
        "LANG": "C.UTF-8",
        "SSL_CERT_FILE": "/cert.pem",
        "ARC_API_KEY": "arc-secret",
        "OPENAI_API_KEY": "openai-secret",
        "CODEX_API_KEY": "codex-secret",
        "CODEX_ACCESS_TOKEN": "token-secret",
        "SERVICE_TOKEN": "service-secret",
        "DB_PASSWORD": "password-secret",
        "OTHER_SECRET": "other-secret",
    }
    assert _sanitized_codex_environment(source) == {
        "HOME": "/home/user",
        "PATH": "/bin",
        "CODEX_HOME": "/codex",
        "LANG": "C.UTF-8",
        "SSL_CERT_FILE": "/cert.pem",
    }


def test_codex_parser_fails_closed_without_usage() -> None:
    """
    Successful process exit cannot silently become an unmetered model turn.
    """

    with pytest.raises(BackendError, match="token usage"):
        parse_codex_jsonl('{"type":"thread.started","thread_id":"thread-1"}')


def test_codex_status_requires_chatgpt_login_not_api_auth(tmp_path) -> None:
    """
    A successful API-key login status does not satisfy the OAuth backend.
    """

    responses = iter(
        (
            subprocess.CompletedProcess([], 0, stdout="codex-cli 1.0", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="Logged in using API key", stderr=""),
        )
    )

    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return next(responses)

    status = CodexOAuthBackend(
        BudgetLedger(tmp_path / "budget"), runner=runner
    ).status()
    assert status.installed
    assert not status.authenticated
    assert not status.ok


def test_codex_failure_raises_without_api_fallback(tmp_path) -> None:
    """
    Selecting OAuth never redirects a failed turn to the direct API backend.
    """

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="oauth failed")

    backend = CodexOAuthBackend(BudgetLedger(tmp_path / "budget"), runner=runner)
    with pytest.raises(BackendError, match="oauth failed"):
        backend.run(AgentRequest(prompt="x", cwd=tmp_path))


def test_codex_rejects_unwired_in_process_tools(tmp_path) -> None:
    """
    Tool handlers cannot be silently ignored by the Codex backend.
    """

    tool = ToolDefinition(
        name="x",
        description="x",
        parameters={"type": "object", "properties": {}},
        handler=lambda: {},
    )
    backend = CodexOAuthBackend(BudgetLedger(tmp_path / "budget"))
    with pytest.raises(ValueError, match="MCP"):
        backend.run(AgentRequest(prompt="x", cwd=tmp_path, tools=(tool,)))


def test_codex_rejects_config_that_weakens_backend_contract(tmp_path) -> None:
    """
    MCP overrides cannot switch model/provider or smuggle a credential field.
    """

    ledger = BudgetLedger(tmp_path / "budget")
    with pytest.raises(ValueError, match="protected key"):
        CodexOAuthBackend(ledger, config_overrides=('model="other"',))
    with pytest.raises(ValueError, match="credential"):
        CodexOAuthBackend(
            ledger, config_overrides=('mcp_servers.arc.env.ARC_API_KEY="secret"',)
        )


class _FakeResponses:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeMessages:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def stream(self, **kwargs: object) -> object:
        self.calls.append(copy.deepcopy(kwargs))
        response = self._responses.pop(0)

        class Stream:
            def __enter__(self) -> "Stream":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def get_final_message(self) -> object:
                return response

        return Stream()


def test_openai_provider_failure_retains_full_reservation(tmp_path) -> None:
    """
    A transport failure cannot restore capacity that the provider may have billed.
    """

    class FailingResponses:
        def create(self, **_kwargs: object) -> object:
            raise TimeoutError("response status is unknown")

    ledger = BudgetLedger(tmp_path / "ambiguous-openai")
    reservation = ledger.reserve("1", role="player")
    backend = OpenAIResponsesBackend(
        ledger,
        client=SimpleNamespace(responses=FailingResponses()),
    )
    with pytest.raises(AmbiguousProviderError, match="reservation remains held"):
        backend.run(
            AgentRequest(
                prompt="act",
                cwd=tmp_path,
                reservation_id=reservation,
            )
        )
    assert ledger.active_reservations() == {reservation: ledger.snapshot().reserved_usd}
    assert ledger.snapshot().reserved_usd == 1


def test_anthropic_missing_usage_retains_full_reservation(tmp_path) -> None:
    """
    A received message without trustworthy accounting remains budget-ambiguous.
    """

    response = SimpleNamespace(
        id="msg-without-usage",
        model="claude-opus-5",
        content=[{"type": "text", "text": "done"}],
        stop_reason="end_turn",
        usage=None,
    )
    ledger = BudgetLedger(
        tmp_path / "ambiguous-anthropic",
        pricing=CLAUDE_OPUS_5_PRICING,
    )
    reservation = ledger.reserve("1", role="player")
    backend = AnthropicMessagesBackend(
        ledger,
        client=SimpleNamespace(messages=_FakeMessages([response])),
    )
    with pytest.raises(AmbiguousProviderError, match="reservation remains held"):
        backend.run(
            AgentRequest(
                prompt="act",
                cwd=tmp_path,
                reservation_id=reservation,
            )
        )
    assert ledger.active_reservations() == {reservation: ledger.snapshot().reserved_usd}
    assert ledger.snapshot().reserved_usd == 1


def test_responses_backend_runs_function_loop_and_accounts_each_call(tmp_path) -> None:
    """
    Direct API mode returns tool output through previous-response continuity.
    """

    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=2,
        input_tokens_details=SimpleNamespace(cached_tokens=4),
    )
    first = SimpleNamespace(
        id="r1",
        model="gpt-5.6-sol",
        status="completed",
        usage=usage,
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call-1",
                name="add",
                arguments='{"left":2,"right":3}',
            )
        ],
        output_text="",
    )
    second = SimpleNamespace(
        id="r2",
        model="gpt-5.6-sol",
        status="completed",
        usage=usage,
        output=[],
        output_text="5",
    )
    responses = _FakeResponses([first, second])
    client = SimpleNamespace(responses=responses)
    tool = ToolDefinition(
        name="add",
        description="Add integers",
        parameters={
            "type": "object",
            "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        handler=lambda left, right: {"sum": left + right},
    )
    backend = OpenAIResponsesBackend(
        BudgetLedger(tmp_path / "budget"), client=client
    )
    reservation = backend.budget.reserve("1", role="supervisor")
    result = backend.run(
        AgentRequest(
            prompt="add",
            cwd=tmp_path,
            tools=(tool,),
            role="supervisor",
            reasoning_effort="high",
            reservation_id=reservation,
        )
    )
    assert result.text == "5"
    assert result.session_id == "r2"
    assert result.tool_rounds == 1
    assert result.usage == TokenUsage(
        input_tokens=20, cached_input_tokens=8, output_tokens=4
    )
    assert responses.calls[1]["previous_response_id"] == "r1"
    assert responses.calls[0]["reasoning"] == {"effort": "high"}
    assert responses.calls[0]["max_output_tokens"] == 65_536
    assert responses.calls[0]["service_tier"] == "default"
    assert responses.calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"sum": 5}',
        }
    ]
    assert backend.budget.snapshot().reserved_usd == 0


def test_openai_client_disables_automatic_retries(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ambiguous SDK retries cannot create provider calls absent from the ledger.
    """

    constructor_calls: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructor_calls.append(kwargs)
            self.responses = _FakeResponses([])

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    OpenAIResponsesBackend(BudgetLedger(tmp_path / "retry-budget"))
    assert constructor_calls == [{"max_retries": 0}]


def test_responses_backend_disables_parallel_sequential_calls(tmp_path) -> None:
    """
    Multiple precomputed mutating actions fail before any handler is executed.
    """

    invoked: list[int] = []
    usage = SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    response = SimpleNamespace(
        id="parallel",
        model="gpt-5.6-sol",
        status="completed",
        usage=usage,
        output=[
            SimpleNamespace(
                type="function_call",
                call_id=f"call-{index}",
                name="play",
                arguments="{}",
            )
            for index in (1, 2)
        ],
        output_text="",
    )
    responses = _FakeResponses([response])
    backend = OpenAIResponsesBackend(
        BudgetLedger(tmp_path / "parallel-budget"),
        client=SimpleNamespace(responses=responses),
    )
    tool = ToolDefinition(
        "play",
        "mutate once",
        {"type": "object", "properties": {}, "required": []},
        lambda: invoked.append(1),
        sequential=True,
    )
    with pytest.raises(BackendError, match="parallel"):
        backend.run(AgentRequest(prompt="act", cwd=tmp_path, tools=(tool,)))
    assert invoked == []
    assert responses.calls[0]["parallel_tool_calls"] is False


def test_anthropic_backend_preserves_thinking_and_resumes_durable_tools(tmp_path) -> None:
    """
    Claude sessions replay signed blocks verbatim and survive backend reconstruction.
    """

    usage = SimpleNamespace(
        input_tokens=10,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=3,
        output_tokens=2,
    )
    thinking = {"type": "thinking", "thinking": "inspect", "signature": "signed-value"}
    first = SimpleNamespace(
        id="msg-1",
        model="claude-opus-5",
        content=[
            thinking,
            {"type": "tool_use", "id": "tool-1", "name": "add", "input": {"value": 2}},
        ],
        stop_reason="tool_use",
        usage=usage,
    )
    second = SimpleNamespace(
        id="msg-2",
        model="claude-opus-5",
        content=[{"type": "text", "text": "done"}],
        stop_reason="end_turn",
        usage=usage,
    )
    messages = _FakeMessages([first, second])
    ledger = BudgetLedger(tmp_path / "claude-budget", pricing=CLAUDE_OPUS_5_PRICING)
    backend = AnthropicMessagesBackend(
        ledger,
        client=SimpleNamespace(messages=messages),
    )
    tool = ToolDefinition(
        "add",
        "Add one",
        {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        lambda value: {"value": value + 1},
        sequential=True,
    )
    reservation = ledger.reserve("1", role="player")
    result = backend.run(
        AgentRequest(
            prompt="start",
            cwd=tmp_path,
            system_prompt="system",
            tools=(tool,),
            reservation_id=reservation,
        )
    )
    assert result.text == "done"
    assert result.tool_rounds == 1
    assert result.usage == TokenUsage(
        input_tokens=36,
        cached_input_tokens=6,
        cache_creation_input_tokens=10,
        output_tokens=4,
    )
    assert messages.calls[0]["model"] == "claude-opus-5"
    assert messages.calls[0]["max_tokens"] == 65_536
    assert messages.calls[0]["thinking"] == {"type": "adaptive"}
    assert messages.calls[0]["output_config"] == {"effort": "max"}
    assert messages.calls[0]["inference_geo"] == "global"
    assert messages.calls[0]["cache_control"] == {"type": "ephemeral"}
    assert messages.calls[0]["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    second_messages = messages.calls[1]["messages"]
    assert isinstance(second_messages, list)
    assert second_messages[1]["content"][0] == thinking
    assert second_messages[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tool-1",
            "content": '{"value": 3}',
            "is_error": False,
        }
    ]

    resumed_messages = _FakeMessages(
        [
            SimpleNamespace(
                id="msg-3",
                model="claude-opus-5",
                content=[{"type": "text", "text": "resumed"}],
                stop_reason="end_turn",
                usage=usage,
            )
        ]
    )
    resumed = AnthropicMessagesBackend(
        BudgetLedger(tmp_path / "claude-budget"),
        client=SimpleNamespace(messages=resumed_messages),
    ).run(
        AgentRequest(
            prompt="continue",
            cwd=tmp_path,
            system_prompt="system",
            session_id=result.session_id,
            tools=(tool,),
        )
    )
    assert resumed.text == "resumed"
    resumed_history = resumed_messages.calls[0]["messages"]
    assert isinstance(resumed_history, list)
    assert resumed_history[-1] == {"role": "user", "content": "continue"}
    assert resumed_history[1]["content"][0] == thinking

    assert result.session_id is not None
    session_path = backend.session_dir / f"{result.session_id}.json"
    document = json.loads(session_path.read_text(encoding="utf-8"))
    document["messages"].append({"role": "user", "content": "tampered"})
    session_path.write_text(json.dumps(document), encoding="utf-8")
    unused = _FakeMessages([])
    with pytest.raises(BackendError, match="digest mismatch"):
        AnthropicMessagesBackend(
            BudgetLedger(tmp_path / "claude-budget"),
            client=SimpleNamespace(messages=unused),
        ).run(
            AgentRequest(
                prompt="continue again",
                cwd=tmp_path,
                system_prompt="system",
                session_id=result.session_id,
                tools=(tool,),
            )
        )
    assert unused.calls == []


def test_anthropic_tool_result_propagates_host_reported_errors() -> None:
    """
    ARC validation failures reach Claude as protocol-level tool errors.
    """

    tool = ToolDefinition(
        "play",
        "act",
        {"type": "object", "properties": {}},
        lambda: {"content": [{"type": "text", "text": "invalid"}], "isError": True},
    )
    result = _anthropic_tool_result("tool-1", "play", {}, {"play": tool})
    assert result["is_error"] is True
    assert '"isError": true' in result["content"]
