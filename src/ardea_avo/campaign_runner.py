"""
Long-horizon ARC campaign orchestration over durable model episodes.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from ardea_avo.arc import (
    TOOL_DEFINITIONS,
    ArcEnvironmentFactory,
    ArcToolRuntime,
    CampaignBank,
    GameStatus,
    LocalArcMcpSurface,
    OfficialArcadeFactory,
    OfficialGameDescriptor,
    load_trace,
    validate_replay,
)
from ardea_avo.runtime import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    AgentRequest,
    AgentResult,
    AmbiguousProviderError,
    AnthropicMessagesBackend,
    BackendError,
    BudgetExceeded,
    BudgetLedger,
    CodexOAuthBackend,
    MemoryStatus,
    MemoryStore,
    OpenAIResponsesBackend,
    ResultsManager,
    RunContext,
    Supervisor,
    SupervisorRedirect,
    ToolDefinition,
)
from ardea_avo.runtime._io import append_jsonl

# Covers the maximum configured output plus worst-case supported context at the
# pinned standard/global token rates. Each provider round settles separately.
TURN_RESERVATION_USD = Decimal("12.00")
QUALIFYING_RHAE_PERCENT = 100.0


class _WinDisposition(StrEnum):
    """
    Host disposition of a freshly replay-verified winning trace.
    """

    NONE = "none"
    FALLBACK = "fallback"
    QUALIFIED = "qualified"


class EpisodeDriver(Protocol):
    """
    Execute player and supervisor turns without owning campaign policy.
    """

    def play(
        self,
        *,
        game_id: str,
        trace_path: Path,
        scratch: Path,
        memory_path: Path,
        run_id: str,
        system_prompt: str,
        prompt: str,
        session_id: str | None,
    ) -> AgentResult:
        """
        Run one player episode against a replayable environment.
        """
        ...

    def supervise(self, *, scratch: Path, system_prompt: str, prompt: str) -> AgentResult:
        """
        Run one tool-free supervisor turn.
        """
        ...


def _toml(value: str | list[str]) -> str:
    """
    Encode strings and arrays using the JSON-compatible TOML subset.
    """
    return json.dumps(value, ensure_ascii=False)


class _BudgetedDriver:
    """
    Shared reservation and backend-construction helpers.
    """

    def __init__(self, ledger: BudgetLedger) -> None:
        self.ledger = ledger

    def _reserve(self, role: str) -> str:
        snapshot = self.ledger.snapshot()
        if snapshot.available_usd < TURN_RESERVATION_USD:
            raise BudgetExceeded(
                "insufficient budget for another worst-case provider response: "
                f"available=${snapshot.available_usd}, "
                f"required=${TURN_RESERVATION_USD}"
            )
        return self.ledger.reserve(TURN_RESERVATION_USD, role=role)

    def _release_failed(self, reservation_id: str, error: Exception) -> None:
        if isinstance(error, AmbiguousProviderError):
            return
        with suppress(KeyError):
            self.ledger.release(reservation_id, reason=f"backend failed: {type(error).__name__}")


class CodexEpisodeDriver(_BudgetedDriver):
    """
    Launch Codex with a per-attempt offline ARC MCP server.
    """

    def __init__(self, ledger: BudgetLedger, cache_dir: Path) -> None:
        super().__init__(ledger)
        self.cache_dir = cache_dir.resolve()

    def _player_backend(
        self,
        *,
        game_id: str,
        trace_path: Path,
        memory_path: Path,
        run_id: str,
    ) -> CodexOAuthBackend:
        arguments = [
            "-m",
            "ardea_avo.arc.server",
            "--game-id",
            game_id,
            "--cache-dir",
            str(self.cache_dir),
            "--trace",
            str(trace_path.resolve()),
            "--memory-db",
            str(memory_path.resolve()),
            "--run-id",
            run_id,
        ]
        overrides = (
            f"mcp_servers.arc.command={_toml(sys.executable)}",
            f"mcp_servers.arc.args={_toml(arguments)}",
        )
        return CodexOAuthBackend(self.ledger, config_overrides=overrides)

    def play(
        self,
        *,
        game_id: str,
        trace_path: Path,
        scratch: Path,
        memory_path: Path,
        run_id: str,
        system_prompt: str,
        prompt: str,
        session_id: str | None,
    ) -> AgentResult:
        """
        Execute a maximum-reasoning Codex player turn.
        """
        reservation = self._reserve("player")
        try:
            return self._player_backend(
                game_id=game_id,
                trace_path=trace_path,
                memory_path=memory_path,
                run_id=run_id,
            ).run(
                AgentRequest(
                    prompt=prompt,
                    cwd=scratch,
                    system_prompt=system_prompt,
                    session_id=session_id,
                    role="player",
                    reasoning_effort="max",
                    sandbox_mode="workspace-write",
                    reservation_id=reservation,
                    metadata={"game_id": game_id},
                )
            )
        except Exception as error:
            self._release_failed(reservation, error)
            raise

    def supervise(self, *, scratch: Path, system_prompt: str, prompt: str) -> AgentResult:
        """
        Execute a tool-free high-reasoning Codex supervisor turn.
        """
        reservation = self._reserve("supervisor")
        try:
            return CodexOAuthBackend(self.ledger).run(
                AgentRequest(
                    prompt=prompt,
                    cwd=scratch,
                    system_prompt=system_prompt,
                    role="supervisor",
                    reasoning_effort="high",
                    sandbox_mode="read-only",
                    reservation_id=reservation,
                )
            )
        except Exception as error:
            self._release_failed(reservation, error)
            raise


class OpenAIEpisodeDriver(_BudgetedDriver):
    """
    Run the same campaign through explicit Responses API function tools.
    """

    def __init__(self, ledger: BudgetLedger, cache_dir: Path, *, client: Any | None = None) -> None:
        super().__init__(ledger)
        self.factory = OfficialArcadeFactory(cache_dir)
        self.client = client

    memory_origin_model = OPENAI_MODEL

    @classmethod
    def _memory_tool(
        cls,
        runtime: ArcToolRuntime,
        store: MemoryStore,
    ) -> ToolDefinition:
        def propose_memory(
            claim: str,
            status: str = "hypothesis",
            confidence: float = 0.5,
            scope: str = "game",
            evidence: list[str] | None = None,
            contradictions: list[str] | None = None,
        ) -> dict[str, Any]:
            evidence_values = tuple(evidence or ())
            contradiction_values = tuple(contradictions or ())
            unknown = (set(evidence_values) | set(contradiction_values)) - runtime.evidence_hashes
            if unknown:
                raise ValueError("memory cites evidence absent from the current trace")
            scope_id = {
                "run": None,
                "game": runtime.game_id,
                "level": f"{runtime.game_id}:level:{runtime.frame.levels_completed + 1}",
            }.get(scope)
            if scope not in {"run", "game", "level"}:
                raise ValueError("scope must be run, game, or level")
            record = store.add(
                scope=scope,
                scope_id=scope_id,
                claim=claim,
                status=status,
                confidence=confidence,
                evidence=evidence_values,
                contradictions=contradiction_values,
                origin_model=cls.memory_origin_model,
            )
            if record.status in {MemoryStatus.VERIFIED, MemoryStatus.FALSIFIED}:
                record = store.approve_for_warm(record.id)
            return {
                "id": record.id,
                "status": record.status.value,
                "claim": record.claim,
                "confidence": record.confidence,
                "evidence": list(record.evidence),
                "contradictions": list(record.contradictions),
                "approved_for_warm": record.approved_for_warm,
            }

        return ToolDefinition(
            name="propose_memory",
            description="Store an evidence-linked hypothesis, verified fact, or falsification without acting.",
            parameters={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "enum": ["hypothesis", "verified", "falsified"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "scope": {"type": "string", "enum": ["run", "game", "level"]},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "contradictions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim"],
                "additionalProperties": False,
            },
            handler=lambda **arguments: propose_memory(**arguments),
            strict=False,
            sequential=True,
        )

    @classmethod
    def _tools(cls, runtime: ArcToolRuntime, store: MemoryStore) -> tuple[ToolDefinition, ...]:
        surface = LocalArcMcpSurface(runtime)

        def handler(name: str) -> Any:
            def call(**arguments: Any) -> dict[str, Any]:
                return surface.call_tool(name, arguments)

            return call

        tools = [
            ToolDefinition(
                name=str(definition["name"]),
                description=str(definition["description"]),
                parameters=dict(definition["inputSchema"]),
                handler=handler(str(definition["name"])),
                strict=False,
                sequential=definition["name"] == "play",
            )
            for definition in TOOL_DEFINITIONS
        ]
        tools.append(cls._memory_tool(runtime, store))
        return tuple(tools)

    def play(
        self,
        *,
        game_id: str,
        trace_path: Path,
        scratch: Path,
        memory_path: Path,
        run_id: str,
        system_prompt: str,
        prompt: str,
        session_id: str | None,
    ) -> AgentResult:
        """
        Execute one Responses player turn while the host owns the live engine.
        """
        environment = self.factory(game_id)
        runtime = (
            ArcToolRuntime.resume(environment, trace_path)
            if trace_path.exists()
            else ArcToolRuntime(environment, trace_path=trace_path)
        )
        store = MemoryStore(memory_path, run_id=run_id)
        reservation = self._reserve("player")
        try:
            backend = OpenAIResponsesBackend(self.ledger, client=self.client)
            return backend.run(
                AgentRequest(
                    prompt=prompt,
                    cwd=scratch,
                    system_prompt=system_prompt,
                    session_id=session_id,
                    role="player",
                    reasoning_effort="max",
                    tools=self._tools(runtime, store),
                    reservation_id=reservation,
                    metadata={"game_id": game_id},
                )
            )
        except Exception as error:
            self._release_failed(reservation, error)
            raise
        finally:
            store.close()
            runtime.close()

    def supervise(self, *, scratch: Path, system_prompt: str, prompt: str) -> AgentResult:
        """
        Execute a tool-free Responses supervisor turn.
        """

        reservation = self._reserve("supervisor")
        try:
            return OpenAIResponsesBackend(self.ledger, client=self.client).run(
                AgentRequest(
                    prompt=prompt,
                    cwd=scratch,
                    system_prompt=system_prompt,
                    role="supervisor",
                    reasoning_effort="high",
                    reservation_id=reservation,
                )
            )
        except Exception as error:
            self._release_failed(reservation, error)
            raise


class AnthropicEpisodeDriver(OpenAIEpisodeDriver):
    """
    Run the campaign through Claude Opus 5 and host-owned function tools.
    """

    memory_origin_model = ANTHROPIC_MODEL

    def play(
        self,
        *,
        game_id: str,
        trace_path: Path,
        scratch: Path,
        memory_path: Path,
        run_id: str,
        system_prompt: str,
        prompt: str,
        session_id: str | None,
    ) -> AgentResult:
        """
        Execute one Claude player turn with durable host-side conversation state.
        """

        environment = self.factory(game_id)
        runtime = (
            ArcToolRuntime.resume(environment, trace_path)
            if trace_path.exists()
            else ArcToolRuntime(environment, trace_path=trace_path)
        )
        store = MemoryStore(memory_path, run_id=run_id)
        reservation = self._reserve("player")
        try:
            backend = AnthropicMessagesBackend(self.ledger, client=self.client)
            return backend.run(
                AgentRequest(
                    prompt=prompt,
                    cwd=scratch,
                    system_prompt=system_prompt,
                    session_id=session_id,
                    role="player",
                    reasoning_effort="max",
                    tools=self._tools(runtime, store),
                    reservation_id=reservation,
                    metadata={"game_id": game_id},
                )
            )
        except Exception as error:
            self._release_failed(reservation, error)
            raise
        finally:
            store.close()
            runtime.close()

    def supervise(self, *, scratch: Path, system_prompt: str, prompt: str) -> AgentResult:
        """
        Execute a tool-free high-effort Claude supervisor turn.
        """

        reservation = self._reserve("supervisor")
        try:
            return AnthropicMessagesBackend(self.ledger, client=self.client).run(
                AgentRequest(
                    prompt=prompt,
                    cwd=scratch,
                    system_prompt=system_prompt,
                    role="supervisor",
                    reasoning_effort="high",
                    reservation_id=reservation,
                )
            )
        except Exception as error:
            self._release_failed(reservation, error)
            raise


@dataclass(slots=True)
class GameRunState:
    """
    Recovery state for one game across attempts and model episodes.
    """

    attempt: int = 1
    episodes: int = 0
    session_id: str | None = None
    trace_actions: int = 0
    levels_completed: int = 0
    consecutive_zero_action_episodes: int = 0
    consecutive_backend_errors: int = 0
    supervisor_state: dict[str, Any] = field(default_factory=lambda: Supervisor().checkpoint())
    pending_redirect: tuple[str, ...] = ()
    optimization_required: bool = False
    banked: bool = False


@dataclass(slots=True)
class CampaignState:
    """
    Complete checkpoint payload for campaign recovery.
    """

    status: str
    games: dict[str, GameRunState]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible checkpoint payload.
        """
        return {"status": self.status, "games": {key: asdict(value) for key, value in self.games.items()}}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], game_ids: Sequence[str]) -> CampaignState:
        """
        Restore exact game state while rejecting roster drift.
        """
        raw_games = value.get("games")
        if not isinstance(raw_games, Mapping) or set(raw_games) != set(game_ids):
            raise ValueError("campaign checkpoint game roster does not match the manifest")
        games: dict[str, GameRunState] = {}
        for game_id in game_ids:
            raw = dict(raw_games[game_id])
            if "pending_redirect" in raw:
                raw["pending_redirect"] = tuple(raw["pending_redirect"])
            games[game_id] = GameRunState(**raw)
        return cls(status=str(value.get("status", "interrupted")), games=games)


class CampaignRunner:
    """
    Play the complete pinned public roster with replayable, resumable evidence.
    """

    def __init__(
        self,
        context: RunContext,
        descriptors: Sequence[OfficialGameDescriptor],
        factory: ArcEnvironmentFactory,
        driver: EpisodeDriver,
        *,
        bundle_dir: Path,
        attempts_per_game: int = 3,
        episodes_per_attempt: int = 12,
        jobs: int = 1,
    ) -> None:
        """
        Bind a run, frozen game roster, model driver, and scheduling limits.
        """
        if not descriptors:
            raise ValueError("campaign requires at least one game")
        if attempts_per_game < 1 or episodes_per_attempt < 1 or jobs < 1:
            raise ValueError("campaign attempts, episodes, and jobs must be positive")
        game_ids = [descriptor.game_id for descriptor in descriptors]
        if len(game_ids) != len(set(game_ids)):
            raise ValueError("campaign game identifiers must be unique")
        self.context = context
        self.descriptors = tuple(sorted(descriptors, key=lambda item: item.game_id))
        self._slot_by_game = {
            descriptor.game_id: f"game-{index:03d}"
            for index, descriptor in enumerate(self.descriptors, start=1)
        }
        self.factory = factory
        self.driver = driver
        self.bundle_dir = bundle_dir.resolve()
        self.attempts_per_game = attempts_per_game
        self.episodes_per_attempt = episodes_per_attempt
        self.jobs = jobs
        self.memory_path = context.directory / "memory.sqlite"
        self.bank = CampaignBank(context.directory / "bank.json")
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._provider_ambiguous = threading.Event()
        self.state = self._load_state()
        self._reconcile_bank_flags()
        self.system_prompt = (self.bundle_dir / "system.md").read_text(encoding="utf-8")
        self.supervisor_prompt = (self.bundle_dir / "supervisor.md").read_text(encoding="utf-8")

    def _load_state(self) -> CampaignState:
        game_ids = [item.game_id for item in self.descriptors]
        checkpoint = self.context.read_checkpoint()
        if checkpoint is None:
            return CampaignState(status="created", games={game_id: GameRunState() for game_id in game_ids})
        payload = checkpoint.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("campaign checkpoint payload is malformed")
        return CampaignState.from_dict(payload, game_ids)

    def _checkpoint(self) -> None:
        with self._state_lock:
            self.context.write_checkpoint(self.state.to_dict())

    def _reconcile_bank_flags(self) -> None:
        """
        Demote legacy completion flags whose selected fallback is below 100.
        """

        for game_id, game in self.state.games.items():
            selected = self.bank.selected(game_id)
            if game.banked and (
                selected is None or selected.rhae_percent != QUALIFYING_RHAE_PERCENT
            ):
                game.banked = False

    def run(self) -> CampaignState:
        """
        Run unbanked games, stopping cleanly when the shared model budget ends.
        """
        self.state.status = "running"
        self.context.append_event("campaign.started", {"jobs": self.jobs, "games": len(self.descriptors)})
        self._checkpoint()
        pending = [item for item in self.descriptors if not self.state.games[item.game_id].banked]
        with ThreadPoolExecutor(max_workers=self.jobs, thread_name_prefix="ardea-game") as pool:
            futures = {pool.submit(self._run_game, descriptor): descriptor.game_id for descriptor in pending}
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    future.result()
                except BudgetExceeded as error:
                    self._stop.set()
                    self.context.append_event("campaign.budget_exhausted", {"game_id": game_id, "error": str(error)})
                except AmbiguousProviderError:
                    self._provider_ambiguous.set()
                    self._stop.set()
                except Exception as error:
                    self.context.append_event(
                        "game.worker_failed",
                        {"game_id": game_id, "error": f"{type(error).__name__}: {error}"},
                    )
        banked = sum(game.banked for game in self.state.games.values())
        if self._provider_ambiguous.is_set():
            self.state.status = "provider_ambiguous"
        elif banked == len(self.state.games):
            self.state.status = "complete"
        elif self._stop.is_set():
            self.state.status = "budget_exhausted"
        else:
            self.state.status = "partial"
        self.context.append_event("campaign.finished", {"status": self.state.status, "banked_games": banked})
        self._checkpoint()
        return self.state

    def _run_game(self, descriptor: OfficialGameDescriptor) -> None:
        game_id = descriptor.game_id
        game = self.state.games[game_id]
        while game.attempt <= self.attempts_per_game and not game.banked and not self._stop.is_set():
            attempt_dir = (
                self.context.directory
                / "games"
                / self._slot_by_game[game_id]
                / f"attempt-{game.attempt:03d}"
            )
            scratch = attempt_dir / "workspace"
            scratch.mkdir(parents=True, exist_ok=True)
            trace_path = attempt_dir / "trace.jsonl"
            _actions, _levels, _verified, reconciled = self._reconcile_trace_state(
                game,
                trace_path,
            )
            if reconciled:
                self._checkpoint()
            disposition = self._bank_if_complete(game_id, game, trace_path)
            if disposition is _WinDisposition.QUALIFIED:
                return
            if disposition is _WinDisposition.FALLBACK:
                self._advance_attempt(game_id, game)
                continue
            supervisor = Supervisor.from_checkpoint(game.supervisor_state)
            for _ in range(self.episodes_per_attempt - game.episodes):
                if self._stop.is_set():
                    return
                before_actions, before_levels, before_verified, reconciled = (
                    self._reconcile_trace_state(game, trace_path)
                )
                if reconciled:
                    self._checkpoint()
                prompt = self._player_prompt(game_id, game, trace_path)
                try:
                    result = self.driver.play(
                        game_id=game_id,
                        trace_path=trace_path,
                        scratch=scratch,
                        memory_path=self.memory_path,
                        run_id=self.context.manifest.run_id,
                        system_prompt=self.system_prompt,
                        prompt=prompt,
                        session_id=game.session_id,
                    )
                    game.consecutive_backend_errors = 0
                    game.pending_redirect = ()
                except BudgetExceeded:
                    self._stop.set()
                    raise
                except AmbiguousProviderError as error:
                    self._mark_provider_ambiguity(
                        error,
                        role="player",
                        game_id=game_id,
                        attempt=game.attempt,
                    )
                    game.session_id = None
                    self._reconcile_trace_state(game, trace_path)
                    self._checkpoint()
                    raise
                except BackendError as error:
                    game.consecutive_backend_errors += 1
                    game.session_id = None
                    self.context.append_event(
                        "model.failed",
                        {"game_id": game_id, "attempt": game.attempt, "error": str(error)},
                    )
                    self._reconcile_trace_state(game, trace_path)
                    disposition = self._bank_if_complete(game_id, game, trace_path)
                    self._checkpoint()
                    if disposition is _WinDisposition.QUALIFIED:
                        return
                    if disposition is _WinDisposition.FALLBACK:
                        break
                    if game.consecutive_backend_errors >= 2:
                        break
                    continue

                game.episodes += 1
                append_jsonl(
                    attempt_dir / "model-episodes.jsonl",
                    {
                        "episode": game.episodes,
                        "session_id": result.session_id,
                        "prompt": prompt,
                        "text": result.text,
                        "usage": asdict(result.usage),
                        "cost_usd": str(result.cost_usd),
                        "tool_rounds": result.tool_rounds,
                        "warnings": list(result.warnings),
                    },
                )
                after_actions, after_levels, after_verified = self._trace_progress(trace_path)
                if after_actions < before_actions or after_levels < before_levels:
                    raise ValueError("campaign trace moved behind its pre-episode state")
                delta = after_actions - before_actions
                game.session_id = result.session_id
                game.trace_actions = after_actions
                if after_levels > before_levels:
                    game.session_id = None
                game.levels_completed = after_levels
                game.consecutive_zero_action_episodes = (
                    game.consecutive_zero_action_episodes + 1 if delta == 0 else 0
                )
                redirect = self._observe_supervisor(
                    supervisor,
                    trace_path,
                    start=before_actions,
                    new_verified=after_verified > before_verified,
                )
                episode_redirect = supervisor.record_episode(action_count=delta)
                redirect = redirect or episode_redirect
                if redirect is not None:
                    game.pending_redirect = self._supervisor_directions(redirect, scratch, trace_path)
                game.supervisor_state = supervisor.checkpoint()
                self.context.append_event(
                    "model.completed",
                    {
                        "game_id": game_id,
                        "attempt": game.attempt,
                        "episode": game.episodes,
                        "actions": delta,
                        "levels_completed": after_levels,
                        "session_id": result.session_id,
                        "cost_usd": str(result.cost_usd),
                    },
                )
                self._checkpoint()
                disposition = self._bank_if_complete(game_id, game, trace_path)
                if disposition is _WinDisposition.QUALIFIED:
                    return
                if disposition is _WinDisposition.FALLBACK:
                    break
                if game.consecutive_zero_action_episodes >= 5:
                    break
            self._advance_attempt(game_id, game)

    def _advance_attempt(self, game_id: str, game: GameRunState) -> None:
        """
        Start a fresh environment attempt without carrying trace progress or session state.
        """

        game.attempt += 1
        game.episodes = 0
        game.session_id = None
        game.trace_actions = 0
        game.levels_completed = 0
        game.consecutive_zero_action_episodes = 0
        game.consecutive_backend_errors = 0
        game.supervisor_state = Supervisor().checkpoint()
        game.pending_redirect = ()
        self.context.append_event("game.retry", {"game_id": game_id, "next_attempt": game.attempt})
        self._checkpoint()

    def _bank_if_complete(
        self,
        game_id: str,
        game: GameRunState,
        trace_path: Path,
    ) -> _WinDisposition:
        """
        Retain every verified WIN as a fallback and complete only at exact 100 RHAE.
        """

        if not trace_path.exists():
            return _WinDisposition.NONE
        data = load_trace(trace_path)
        if not data.steps or data.steps[-1].status is not GameStatus.WIN:
            return _WinDisposition.NONE
        replay = validate_replay(trace_path, self.factory)
        with self._state_lock:
            selected = self.bank.consider(trace_path, replay, optimize_actions=True)
            qualified = replay.rhae_percent == QUALIFYING_RHAE_PERCENT
            game.banked = qualified
            game.optimization_required = not qualified
            game.trace_actions = replay.actions
            game.levels_completed = replay.levels_completed
            game.session_id = None
            self.context.append_event(
                "game.banked" if qualified else "game.fallback_banked",
                {
                    "game_id": game_id,
                    "attempt": game.attempt,
                    "actions": replay.actions,
                    "rhae_percent": replay.rhae_percent,
                    "selected": selected,
                },
            )
            self._checkpoint()
        return (
            _WinDisposition.QUALIFIED
            if qualified
            else _WinDisposition.FALLBACK
        )

    def _reconcile_trace_state(
        self,
        game: GameRunState,
        trace_path: Path,
    ) -> tuple[int, int, int, bool]:
        """
        Make replayable trace progress authoritative before another model prompt.

        A trace may be ahead of its checkpoint when a model or MCP process committed
        actions immediately before interruption. Its old provider session is then
        unsafe to resume. A checkpoint ahead of an extant trace instead indicates
        rollback or truncation and fails closed.
        """

        actions, levels, verified = self._trace_progress(trace_path)
        before: tuple[int, int, str | None] = (
            game.trace_actions,
            game.levels_completed,
            game.session_id,
        )
        if not trace_path.exists():
            game.session_id = None
            game.trace_actions = 0
            game.levels_completed = 0
            after: tuple[int, int, str | None] = (
                game.trace_actions,
                game.levels_completed,
                game.session_id,
            )
            return actions, levels, verified, after != before
        if actions < game.trace_actions:
            raise ValueError("campaign trace action count moved behind its checkpoint")
        trace_ahead = actions > game.trace_actions
        level_changed = levels != game.levels_completed
        if trace_ahead or level_changed:
            game.session_id = None
        game.trace_actions = actions
        game.levels_completed = levels
        after = (game.trace_actions, game.levels_completed, game.session_id)
        return actions, levels, verified, after != before

    def _trace_progress(self, trace_path: Path) -> tuple[int, int, int]:
        if not trace_path.exists():
            return 0, 0, self._verified_memory_count()
        data = load_trace(trace_path)
        levels = data.steps[-1].levels_completed if data.steps else data.header.initial_levels_completed
        return len(data.steps), levels, self._verified_memory_count()

    def _verified_memory_count(self) -> int:
        if not self.memory_path.exists():
            return 0
        with MemoryStore(self.memory_path, run_id=self.context.manifest.run_id) as store:
            return len(store.list(status=MemoryStatus.VERIFIED)) + len(store.list(status=MemoryStatus.FALSIFIED))

    def _memory_context(self, game_id: str) -> str:
        if not self.memory_path.exists():
            return "No evidence-backed memory exists yet."
        with MemoryStore(self.memory_path, run_id=self.context.manifest.run_id) as store:
            records = [
                record
                for record in store.list()
                if record.scope == "run" or record.scope_id is None or record.scope_id.startswith(game_id)
            ]
        if not records:
            return "No relevant memory exists yet."
        return "\n".join(
            f"- [{record.status.value}; confidence={record.confidence:.2f}] {record.claim} "
            f"(evidence={','.join(record.evidence + record.contradictions) or 'none'})"
            for record in records[-100:]
        )

    def _player_prompt(self, game_id: str, game: GameRunState, trace_path: Path) -> str:
        continuation = (
            "Continue the current level from its replayed trace."
            if trace_path.exists()
            else "This is a fresh attempt; call observe before choosing an action."
        )
        redirect = ""
        if game.pending_redirect:
            redirect = "\nOne-time supervisor suggestions:\n" + "\n".join(
                f"{index}. {direction}" for index, direction in enumerate(game.pending_redirect, start=1)
            )
        optimization = (
            " A prior verified WIN missed the efficiency target; find a shorter solution."
            if game.optimization_required
            else ""
        )
        return (
            f"{continuation} Progress recorded by the host: {game.levels_completed} completed levels. "
            f"Reach WIN and do not stop merely to explain.{optimization} Human baselines and the private action limit "
            "are intentionally unavailable.\n\nRelevant durable memory:\n"
            f"{self._memory_context(game_id)}{redirect}"
        )

    @staticmethod
    def _observe_supervisor(
        supervisor: Supervisor,
        trace_path: Path,
        *,
        start: int,
        new_verified: bool,
    ) -> SupervisorRedirect | None:
        if not trace_path.exists():
            return None
        data = load_trace(trace_path)
        before_hash = data.header.initial_frame_hash if start == 0 else data.steps[start - 1].frame_hash
        previous_levels = data.header.initial_levels_completed if start == 0 else data.steps[start - 1].levels_completed
        redirect: SupervisorRedirect | None = None
        for step in data.steps[start:]:
            death_signature = None
            if step.status is GameStatus.GAME_OVER:
                recent = "".join(item.record_hash for item in data.steps[max(0, step.number - 10) : step.number])
                death_signature = sha256(recent.encode("ascii")).hexdigest()
            redirect = redirect or supervisor.record_transition(
                before_hash=before_hash,
                after_hash=step.frame_hash,
                outcome=step.status.value,
                death_path_signature=death_signature,
                level_progress=step.levels_completed > previous_levels,
                new_verified_evidence=new_verified,
            )
            before_hash = step.frame_hash
            previous_levels = step.levels_completed
            new_verified = False
        return redirect

    def _supervisor_directions(
        self,
        redirect: SupervisorRedirect,
        scratch: Path,
        trace_path: Path,
    ) -> tuple[str, ...]:
        fallback = tuple(redirect.directions)
        summary = "no trace"
        if trace_path.exists():
            data = load_trace(trace_path)
            summary = json.dumps(
                [
                    {
                        "n": item.number,
                        "action": item.action.value,
                        "status": item.status.value,
                        "levels": item.levels_completed,
                        "frame_hash": item.frame_hash,
                    }
                    for item in data.steps[-20:]
                ],
                separators=(",", ":"),
            )
        try:
            result = self.driver.supervise(
                scratch=scratch,
                system_prompt=self.supervisor_prompt,
                prompt=(
                    f"Trigger: {redirect.trigger.value}.\nRecent trace: {summary}\n"
                    "Return exactly a JSON array of three distinct strategy strings and nothing else."
                ),
            )
            value = json.loads(result.text)
            if (
                isinstance(value, list)
                and len(value) == 3
                and all(isinstance(item, str) and item.strip() for item in value)
                and len({item.strip().casefold() for item in value}) == 3
            ):
                directions = tuple(item.strip() for item in value)
                self.context.append_event(
                    "supervisor.redirect",
                    {"trigger": redirect.trigger.value, "directions": list(directions)},
                )
                return directions
        except AmbiguousProviderError as error:
            self._mark_provider_ambiguity(error, role="supervisor")
            raise
        except (BackendError, BudgetExceeded, json.JSONDecodeError, TypeError, ValueError) as error:
            self.context.append_event(
                "supervisor.fallback",
                {"trigger": redirect.trigger.value, "error": f"{type(error).__name__}: {error}"},
            )
        return fallback

    def _mark_provider_ambiguity(
        self,
        error: AmbiguousProviderError,
        *,
        role: str,
        game_id: str | None = None,
        attempt: int | None = None,
    ) -> None:
        """
        Stop new model work and durably mark an unreceipted provider outcome.
        """

        self._provider_ambiguous.set()
        self._stop.set()
        payload: dict[str, Any] = {
            "role": role,
            "error": str(error),
            "full_reservation_retained": True,
        }
        if game_id is not None:
            payload["game_id"] = game_id
        if attempt is not None:
            payload["attempt"] = attempt
        self.context.append_event("provider.usage_ambiguous", payload)


def collect_parent_evidence(parent_directory: Path) -> set[str]:
    """
    Collect validated frame and receipt hashes eligible to support warm memory.
    """
    evidence: set[str] = set()
    for trace_path in parent_directory.glob("games/*/attempt-*/trace.jsonl"):
        data = load_trace(trace_path)
        evidence.add(data.header.initial_frame_hash)
        evidence.add(data.header.record_hash)
        for step in data.steps:
            evidence.add(step.frame_hash)
            evidence.add(step.record_hash)
    return evidence


def import_warm_memory(context: RunContext, results_root: Path) -> int:
    """
    Idempotently import approved parent claims and anchor warm initialization.
    """
    parent_id = context.manifest.parent_run_id
    if parent_id is None:
        return 0
    parent_context = ResultsManager(results_root).open(parent_id)
    context.assert_parent_unchanged(parent_context)
    prior = [
        event
        for event in context.events()
        if event["kind"] == "memory.imported"
        and event["payload"].get("parent_run_id") == parent_id
    ]
    if prior:
        return int(prior[-1]["payload"]["records"])
    parent_directory = results_root / parent_id
    parent_path = parent_directory / "memory.sqlite"
    if not parent_path.exists():
        context.append_event(
            "memory.imported",
            {
                "parent_run_id": parent_id,
                "records": 0,
                "new_records": 0,
                "parent_memory_present": False,
            },
        )
        return 0
    evidence = collect_parent_evidence(parent_directory)
    with (
        MemoryStore(parent_path, run_id=parent_id, read_only=True) as parent,
        MemoryStore(context.directory / "memory.sqlite", run_id=context.manifest.run_id) as child,
    ):
        imported = child.import_from_parent(parent, evidence_validator=evidence.__contains__)
        total = sum(record.imported_from_run == parent_id for record in child.list())
    context.assert_parent_unchanged(parent_context)
    context.append_event(
        "memory.imported",
        {
            "parent_run_id": parent_id,
            "records": total,
            "new_records": len(imported),
            "parent_memory_present": True,
        },
    )
    return total
