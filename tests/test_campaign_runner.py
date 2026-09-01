"""
Integration tests for episode boundaries, transcripts, replay, and banking.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from test_arc_runtime import fake_environment

from ardea_avo.arc import ArcToolRuntime, CampaignBank, OfficialGameDescriptor
from ardea_avo.campaign_runner import (
    CampaignRunner,
    CampaignState,
    GameRunState,
    OpenAIEpisodeDriver,
    import_warm_memory,
)
from ardea_avo.runtime import (
    AgentResult,
    AmbiguousProviderError,
    BackendError,
    MemoryStore,
    ResultsManager,
    TokenUsage,
)

NOW = datetime(2026, 9, 1, 13, 20, 26, tzinfo=UTC)
GAME_ID = "synthetic-00000001"


class PlayingDriver:
    """
    Use the real ARC runtime while standing in for a model backend.
    """

    def __init__(self) -> None:
        self.calls = 0

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
        Complete one level per episode through the durable trace.
        """

        del scratch, memory_path, run_id, system_prompt, prompt
        environment = fake_environment(game_id)
        runtime = (
            ArcToolRuntime.resume(environment, trace_path)
            if trace_path.exists()
            else ArcToolRuntime(environment, trace_path=trace_path)
        )
        runtime.play("ACTION1")
        runtime.close()
        self.calls += 1
        return AgentResult(
            text=f"episode {self.calls}",
            session_id=session_id or "synthetic-session",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            cost_usd=Decimal("0.00008"),
        )

    def supervise(self, *, scratch: Path, system_prompt: str, prompt: str) -> AgentResult:
        """
        Return a valid redirect if the deterministic supervisor asks.
        """

        del scratch, system_prompt, prompt
        return AgentResult(
            text='["inspect geometry","probe alternative","replay verified prefix"]',
            session_id="supervisor-session",
            usage=TokenUsage(),
            cost_usd=Decimal("0"),
        )


class ScriptedWinDriver(PlayingDriver):
    """
    Produce winning traces with an attempt-specific number of wasted actions.
    """

    def __init__(self, extra_actions: dict[int, int]) -> None:
        super().__init__()
        self.extra_actions = extra_actions
        self.prompts: list[str] = []
        self.sessions: list[str | None] = []

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
        Finish both levels in one episode after the configured exploratory waste.
        """

        del scratch, memory_path, run_id, system_prompt
        attempt = int(trace_path.parent.name.removeprefix("attempt-"))
        runtime = ArcToolRuntime(
            fake_environment(game_id),
            trace_path=trace_path,
            episode_per_level=False,
        )
        for _ in range(self.extra_actions[attempt]):
            runtime.play("ACTION2")
        runtime.play("ACTION1")
        runtime.play("ACTION1")
        runtime.close()
        self.calls += 1
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        return AgentResult(
            text=f"attempt {attempt}",
            session_id=f"session-{attempt}",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            cost_usd=Decimal("0.00008"),
        )


class ResumeDriver(PlayingDriver):
    """
    Continue a host-created trace while recording the recovered prompt inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt = ""
        self.received_session: str | None = "unobserved"

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
        Resume the replayed level and finish it with one action.
        """

        del scratch, memory_path, run_id, system_prompt
        runtime = ArcToolRuntime.resume(fake_environment(game_id), trace_path)
        runtime.play("ACTION1")
        runtime.close()
        self.calls += 1
        self.prompt = prompt
        self.received_session = session_id
        return AgentResult(
            text="resumed",
            session_id="replacement-session",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            cost_usd=Decimal("0.00008"),
        )


class WinningBackendErrorDriver(PlayingDriver):
    """
    Commit a complete trace before simulating loss of the backend response.
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
        Persist an optimal WIN and then raise the recoverable backend error.
        """

        del scratch, memory_path, run_id, system_prompt, prompt, session_id
        runtime = ArcToolRuntime(
            fake_environment(game_id),
            trace_path=trace_path,
            episode_per_level=False,
        )
        runtime.play("ACTION1")
        runtime.play("ACTION1")
        runtime.close()
        self.calls += 1
        raise BackendError("provider response was lost after tool execution")


class AmbiguousDriver(PlayingDriver):
    """
    Simulate a provider call whose billable outcome cannot be determined.
    """

    def play(self, **_kwargs: object) -> AgentResult:
        """
        Count one call and require the host to stop rather than retry it.
        """

        self.calls += 1
        raise AmbiguousProviderError("usage receipt unavailable")


class AmbiguousSupervisorDriver(PlayingDriver):
    """
    Reach a deterministic redirect and lose only the supervisor usage receipt.
    """

    def __init__(self) -> None:
        super().__init__()
        self.supervisor_calls = 0

    def play(self, **_kwargs: object) -> AgentResult:
        """
        Complete a zero-action episode so the second turn requests a redirect.
        """

        self.calls += 1
        return AgentResult(
            text="no action",
            session_id="zero-action-session",
            usage=TokenUsage(),
            cost_usd=Decimal("0"),
        )

    def supervise(self, **_kwargs: object) -> AgentResult:
        """
        Require the same stop semantics for a supervisor provider call.
        """

        self.supervisor_calls += 1
        raise AmbiguousProviderError("supervisor usage receipt unavailable")


def _bundle(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "system.md").write_text("player contract\n", encoding="utf-8")
    (path / "supervisor.md").write_text("supervisor contract\n", encoding="utf-8")
    return path


def _descriptor() -> tuple[OfficialGameDescriptor, ...]:
    """
    Return the single deterministic public-game fixture descriptor.
    """

    return (OfficialGameDescriptor(GAME_ID, None, 2),)


def test_runner_crosses_level_episode_boundary_and_banks_win(tmp_path: Path) -> None:
    """
    A fresh session per level still resumes one exact game trace.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    context = manager.create_cold("campaign")
    driver = PlayingDriver()
    runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=1,
        episodes_per_attempt=3,
    )

    state = runner.run()

    assert state.status == "complete"
    assert state.games[GAME_ID].banked
    assert driver.calls == 2
    entry = CampaignBank(context.directory / "bank.json").validate_selected(GAME_ID)
    assert entry.levels_completed == 2
    assert entry.actions == 2
    transcript = context.directory / "games" / "game-001" / "attempt-001" / "model-episodes.jsonl"
    assert len(transcript.read_text(encoding="utf-8").splitlines()) == 2


def test_api_memory_tool_hides_host_game_identity(tmp_path: Path) -> None:
    """
    A memory receipt never returns the host-only full game identifier.
    """

    runtime = ArcToolRuntime(fake_environment(GAME_ID))
    with MemoryStore(tmp_path / "memory.sqlite", run_id="run-1") as store:
        memory_tool = OpenAIEpisodeDriver._memory_tool(runtime, store)
        payload = memory_tool.handler(claim="directional probe may move the avatar")
    runtime.close()

    assert GAME_ID not in str(payload)
    assert "scope_id" not in payload
    assert payload["status"] == "hypothesis"


def test_runner_keeps_fallback_then_uses_fresh_attempt_for_exact_100(tmp_path: Path) -> None:
    """
    A sub-100 WIN is evidence, but only the later exact score completes the game.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    context = manager.create_cold("campaign")
    driver = ScriptedWinDriver({1: 5, 2: 0})
    runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=2,
        episodes_per_attempt=1,
    )

    state = runner.run()

    assert state.status == "complete"
    assert state.games[GAME_ID].banked is True
    assert driver.calls == 2
    assert driver.sessions == [None, None]
    assert "0 completed levels" in driver.prompts[1]
    assert "find a shorter solution" in driver.prompts[1]
    entry = CampaignBank(context.directory / "bank.json").validate_selected(GAME_ID)
    assert entry.rhae_percent == 100.0
    assert Path(entry.trace_path).parent.name == "attempt-002"
    kinds = [event["kind"] for event in context.events()]
    assert "game.fallback_banked" in kinds
    assert "game.retry" in kinds
    assert "game.banked" in kinds


def test_exhausted_attempts_retain_highest_scoring_fallback(tmp_path: Path) -> None:
    """
    Optimization replaces a first fallback while leaving the exhausted game partial.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    context = manager.create_cold("campaign")
    driver = ScriptedWinDriver({1: 5, 2: 2})
    runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=2,
        episodes_per_attempt=1,
    )

    state = runner.run()

    entry = CampaignBank(context.directory / "bank.json").validate_selected(GAME_ID)
    assert state.status == "partial"
    assert state.games[GAME_ID].banked is False
    assert state.games[GAME_ID].attempt == 3
    assert entry.rhae_percent < 100.0
    assert entry.actions == 4
    assert Path(entry.trace_path).parent.name == "attempt-002"


def test_resume_demotes_legacy_sub_100_completion_and_continues(tmp_path: Path) -> None:
    """
    A legacy banked flag cannot turn a selected fallback into campaign completion.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    context = manager.create_cold("campaign")
    bundle = _bundle(tmp_path / "bundle")
    first_driver = ScriptedWinDriver({1: 5})
    first_runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        first_driver,
        bundle_dir=bundle,
        attempts_per_game=1,
        episodes_per_attempt=1,
    )
    first_state = first_runner.run()
    legacy_payload = first_state.to_dict()
    legacy_payload["games"][GAME_ID]["banked"] = True
    context.write_checkpoint(legacy_payload)

    resumed_driver = ScriptedWinDriver({2: 0})
    resumed_runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        resumed_driver,
        bundle_dir=bundle,
        attempts_per_game=2,
        episodes_per_attempt=1,
    )
    resumed_state = resumed_runner.run()

    assert resumed_state.status == "complete"
    assert resumed_state.games[GAME_ID].banked is True
    assert resumed_driver.calls == 1
    entry = CampaignBank(context.directory / "bank.json").validate_selected(GAME_ID)
    assert entry.rhae_percent == 100.0
    assert Path(entry.trace_path).parent.name == "attempt-002"


def test_resume_reconciles_trace_ahead_and_clears_provider_session(tmp_path: Path) -> None:
    """
    Durable trace progress outranks a stale checkpoint before the next prompt.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    context = manager.create_cold("campaign")
    trace_path = context.directory / "games" / "game-001" / "attempt-001" / "trace.jsonl"
    runtime = ArcToolRuntime(fake_environment(GAME_ID), trace_path=trace_path)
    runtime.play("ACTION1")
    runtime.close()
    context.write_checkpoint(
        CampaignState(
            status="running",
            games={GAME_ID: GameRunState(session_id="stale-provider-session")},
        ).to_dict()
    )
    driver = ResumeDriver()
    runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=1,
        episodes_per_attempt=2,
    )

    state = runner.run()

    assert state.status == "complete"
    assert driver.received_session is None
    assert "1 completed levels" in driver.prompt
    assert state.games[GAME_ID].trace_actions == 2
    assert state.games[GAME_ID].levels_completed == 2


def test_resume_resets_stale_checkpoint_progress_when_attempt_trace_is_absent(
    tmp_path: Path,
) -> None:
    """
    A fresh attempt never inherits session or progress counters from a missing trace.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    context = manager.create_cold("campaign")
    context.write_checkpoint(
        CampaignState(
            status="running",
            games={
                GAME_ID: GameRunState(
                    session_id="stale-provider-session",
                    trace_actions=3,
                    levels_completed=1,
                )
            },
        ).to_dict()
    )
    driver = ScriptedWinDriver({1: 0})
    runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=1,
        episodes_per_attempt=1,
    )

    state = runner.run()

    assert state.status == "complete"
    assert driver.sessions == [None]
    assert "0 completed levels" in driver.prompts[0]


def test_backend_error_banks_committed_exact_win_before_rollover(tmp_path: Path) -> None:
    """
    A lost model response cannot discard an already durable winning trace.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    context = manager.create_cold("campaign")
    driver = WinningBackendErrorDriver()
    runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=1,
        episodes_per_attempt=1,
    )

    state = runner.run()

    assert driver.calls == 1
    assert state.status == "complete"
    assert state.games[GAME_ID].banked is True
    entry = CampaignBank(context.directory / "bank.json").validate_selected(GAME_ID)
    assert entry.rhae_percent == 100.0
    kinds = [event["kind"] for event in context.events()]
    assert "model.failed" in kinds
    assert "game.banked" in kinds
    assert "game.retry" not in kinds


def test_ambiguous_provider_outcome_stops_campaign_without_retry(tmp_path: Path) -> None:
    """
    No later episode or attempt launches while provider spend is unreconciled.
    """

    context = ResultsManager(tmp_path / "results", clock=lambda: NOW).create_cold(
        "campaign"
    )
    driver = AmbiguousDriver()
    runner = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=3,
        episodes_per_attempt=4,
    )

    state = runner.run()

    assert driver.calls == 1
    assert state.status == "provider_ambiguous"
    ambiguity = [
        event
        for event in context.events()
        if event["kind"] == "provider.usage_ambiguous"
    ]
    assert len(ambiguity) == 1
    assert ambiguity[0]["payload"]["full_reservation_retained"] is True


def test_ambiguous_supervisor_outcome_stops_later_player_calls(tmp_path: Path) -> None:
    """
    A lost supervisor receipt cannot fall back and silently continue spending.
    """

    context = ResultsManager(tmp_path / "results", clock=lambda: NOW).create_cold(
        "campaign"
    )
    driver = AmbiguousSupervisorDriver()
    state = CampaignRunner(
        context,
        _descriptor(),
        lambda game_id: fake_environment(game_id),
        driver,
        bundle_dir=_bundle(tmp_path / "bundle"),
        attempts_per_game=2,
        episodes_per_attempt=5,
    ).run()

    assert state.status == "provider_ambiguous"
    assert driver.calls == 2
    assert driver.supervisor_calls == 1
    events = [
        event
        for event in context.events()
        if event["kind"] == "provider.usage_ambiguous"
    ]
    assert len(events) == 1
    assert events[0]["payload"]["role"] == "supervisor"


def test_warm_memory_initialization_is_idempotently_anchored_without_parent_db(
    tmp_path: Path,
) -> None:
    """
    Resume can prove that a zero-record warm import was intentionally completed.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    parent = manager.create_cold("parent")
    child = manager.create_warm(parent.manifest.run_id, "child")

    assert import_warm_memory(child, manager.root) == 0
    assert import_warm_memory(child, manager.root) == 0

    imports = [event for event in child.events() if event["kind"] == "memory.imported"]
    assert len(imports) == 1
    assert imports[0]["payload"]["parent_memory_present"] is False
