"""
Tests for optional official package and safety gates.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_arc_runtime import fake_environment, grid_with

from ardea_avo.arc import (
    ArcAction,
    ArcToolRuntime,
    HumanConfirmationRequired,
    OfficialArcadeFactory,
    OfficialVersionError,
    ScorecardMode,
    ScorecardSubmissionError,
    check_official_versions,
    inspect_environment_cache,
    setup_public_games,
    submit_scorecard,
)
from ardea_avo.arc.adapter import OfficialArcEnvironment
from ardea_avo.arc.official import _validate_official_response
from ardea_avo.arc.server import create_mcp_server, parse_server_args
from ardea_avo.runtime.memory import MemoryStore


def test_version_check_requires_exact_pins() -> None:
    versions = {"arc-agi": "0.9.9", "arcengine": "0.9.3"}
    assert check_official_versions(version_reader=versions.__getitem__) == versions
    versions["arcengine"] = "0.9.4"
    with pytest.raises(OfficialVersionError, match="version drift"):
        check_official_versions(version_reader=versions.__getitem__)


def test_cache_digest_is_path_and_content_stable(tmp_path: Path) -> None:
    first = tmp_path / "a"
    first.mkdir()
    (first / "x.dat").write_bytes(b"one")
    (first / "y.dat").write_bytes(b"two")
    files, digest = inspect_environment_cache(first)
    assert [item.relative_path for item in files] == ["x.dat", "y.dat"]
    assert len(digest) == 64
    (first / "y.dat").write_bytes(b"three")
    _, changed = inspect_environment_cache(first)
    assert changed != digest


def test_competition_gate_precedes_dependencies_or_scorecard_creation(tmp_path: Path) -> None:
    called = False

    def forbidden_factory(game_id: str):
        nonlocal called
        called = True
        raise AssertionError(game_id)

    with pytest.raises(HumanConfirmationRequired):
        submit_scorecard(
            [],
            tmp_path,
            mode=ScorecardMode.COMPETITION,
            local_factory=forbidden_factory,
            human_confirmed=False,
        )
    assert called is False


@pytest.mark.parametrize("competition_mode", ("false", 1))
def test_official_response_rejects_truthy_or_falsey_non_boolean_modes(
    competition_mode: object,
) -> None:
    """
    Scorecard mode validation requires the provider's literal JSON boolean.
    """

    response = {
        "card_id": "card-1",
        "total_environments": 0,
        "total_environments_completed": 0,
        "total_levels": 0,
        "total_levels_completed": 0,
        "total_actions": 0,
        "competition_mode": competition_mode,
        "environments": [],
    }
    with pytest.raises(ScorecardSubmissionError, match="mode"):
        _validate_official_response(
            response,
            scorecard_id="card-1",
            mode=ScorecardMode.DRY_RUN,
            replays=(),
            expectations=(),
        )


def test_official_factory_defaults_to_explicit_offline_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    offline = object()

    class FakeArcade:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    modules = {
        "arc_agi": SimpleNamespace(Arcade=FakeArcade),
        "arc_agi.base": SimpleNamespace(OperationMode=SimpleNamespace(OFFLINE=offline)),
        "arcengine": SimpleNamespace(GameAction=object()),
    }
    monkeypatch.setattr("ardea_avo.arc.adapter.import_module", modules.__getitem__)
    factory = OfficialArcadeFactory(tmp_path, verify_versions=False)
    factory._load()
    assert calls == [{"environments_dir": str(tmp_path.resolve()), "operation_mode": offline}]


def test_official_factory_masks_ambient_mode_and_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Poisoned SDK control variables cannot turn offline replay into Competition.
    """

    offline = SimpleNamespace(name="OFFLINE")
    observed: list[tuple[str | None, str | None]] = []

    class FakeArcade:
        def __init__(self, **kwargs: object) -> None:
            self.operation_mode = kwargs["operation_mode"]
            self.arc_base_url = os.environ.get("ARC_BASE_URL")
            observed.append(
                (os.environ.get("OPERATION_MODE"), os.environ.get("ARC_BASE_URL"))
            )

    modules = {
        "arc_agi": SimpleNamespace(Arcade=FakeArcade),
        "arc_agi.base": SimpleNamespace(OperationMode=SimpleNamespace(OFFLINE=offline)),
        "arcengine": SimpleNamespace(GameAction=object()),
    }
    monkeypatch.setenv("OPERATION_MODE", "competition")
    monkeypatch.setenv("ARC_BASE_URL", "https://evil.invalid")
    monkeypatch.setattr("ardea_avo.arc.adapter.import_module", modules.__getitem__)

    OfficialArcadeFactory(tmp_path, verify_versions=False)._load()

    assert observed == [("offline", "https://arcprize.org")]
    assert os.environ["OPERATION_MODE"] == "competition"
    assert os.environ["ARC_BASE_URL"] == "https://evil.invalid"


def test_official_factory_serializes_load_and_make_and_scopes_reset_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Concurrent game workers share one safely published official SDK client.
    """

    offline = SimpleNamespace(name="OFFLINE")
    guard = threading.Lock()
    constructed = 0
    active = 0
    maximum_active = 0
    observed_reset_modes: list[str | None] = []

    def frame() -> object:
        return SimpleNamespace(
            frame=[grid_with()],
            state=SimpleNamespace(value="NOT_FINISHED"),
            levels_completed=0,
            win_levels=1,
            available_actions=[],
        )

    class FakeArcade:
        def __init__(self, **kwargs: object) -> None:
            nonlocal constructed
            constructed += 1
            self.operation_mode = kwargs["operation_mode"]
            self.arc_base_url = os.environ["ARC_BASE_URL"]

        def make(self, game_id: str) -> object:
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            observed_reset_modes.append(os.environ.get("ONLY_RESET_LEVELS"))
            time.sleep(0.01)
            with guard:
                active -= 1
            return SimpleNamespace(
                info=SimpleNamespace(game_id=game_id, baseline_actions=[1]),
                observation_space=frame(),
            )

    modules = {
        "arc_agi": SimpleNamespace(Arcade=FakeArcade),
        "arc_agi.base": SimpleNamespace(OperationMode=SimpleNamespace(OFFLINE=offline)),
        "arcengine": SimpleNamespace(GameAction=object()),
    }
    monkeypatch.setattr("ardea_avo.arc.adapter.import_module", modules.__getitem__)
    monkeypatch.setenv("ONLY_RESET_LEVELS", "ambient")
    factory = OfficialArcadeFactory(tmp_path, verify_versions=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        environments = tuple(
            pool.map(factory, ("game-a-version", "game-b-version", "game-c-version"))
        )
    assert len(environments) == 3
    assert constructed == 1
    assert maximum_active == 1
    assert observed_reset_modes == ["true", "true", "true"]
    assert os.environ["ONLY_RESET_LEVELS"] == "ambient"


def test_official_reset_preserves_levels_and_restores_ambient_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A later-level RESET uses the SDK's level-only mode for exactly one action.
    """

    class RawEnvironment:
        info = SimpleNamespace(game_id="fixture-version", baseline_actions=[1, 1])
        observation_space = SimpleNamespace(
            frame=[grid_with()],
            state=SimpleNamespace(value="NOT_FINISHED"),
            levels_completed=1,
            win_levels=2,
            available_actions=[],
        )

        def step(self, _action: object, *, data: object) -> object:
            levels = 1 if os.environ.get("ONLY_RESET_LEVELS") == "true" else 0
            return SimpleNamespace(
                frame=[grid_with()],
                state=SimpleNamespace(value="NOT_FINISHED"),
                levels_completed=levels,
                win_levels=2,
                available_actions=[],
            )

    monkeypatch.setenv("ONLY_RESET_LEVELS", "ambient")
    environment = OfficialArcEnvironment(
        RawEnvironment(),
        {"RESET": object()},
        engine_version="0.9.3",
    )
    assert environment.step(ArcAction.RESET).levels_completed == 1
    assert os.environ["ONLY_RESET_LEVELS"] == "ambient"


def test_official_reset_restores_ambient_state_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    SDK failures cannot leak the process-global reset switch.
    """

    class RawEnvironment:
        info = SimpleNamespace(game_id="fixture-version", baseline_actions=[1])
        observation_space = SimpleNamespace(
            frame=[grid_with()],
            state=SimpleNamespace(value="NOT_FINISHED"),
            levels_completed=0,
            win_levels=1,
            available_actions=[],
        )

        def step(self, _action: object, *, data: object) -> object:
            assert os.environ.get("ONLY_RESET_LEVELS") == "true"
            raise RuntimeError("engine failed")

    monkeypatch.delenv("ONLY_RESET_LEVELS", raising=False)
    environment = OfficialArcEnvironment(
        RawEnvironment(),
        {"RESET": object()},
        engine_version="0.9.3",
    )
    with pytest.raises(RuntimeError, match="engine failed"):
        environment.step(ArcAction.RESET)
    assert "ONLY_RESET_LEVELS" not in os.environ


def test_official_environment_uses_full_game_id_as_version_fallback() -> None:
    game_id = "fixture-89abcdef"
    raw = SimpleNamespace(
        info=SimpleNamespace(game_id=game_id, baseline_actions=[2, 4]),
        observation_space=SimpleNamespace(
            frame=[grid_with()],
            state=SimpleNamespace(value="NOT_FINISHED"),
            levels_completed=0,
            win_levels=2,
            available_actions=[],
        ),
    )
    environment = OfficialArcEnvironment(raw, object(), engine_version="0.9.3")
    assert environment.info.environment_version == game_id


def test_stdio_server_parser_is_side_effect_free(tmp_path: Path) -> None:
    config = parse_server_args(
        [
            "--game-id",
            "fixture-89abcdef",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--trace",
            str(tmp_path / "trace.jsonl"),
        ]
    )
    assert config.game_id == "fixture-89abcdef"
    assert config.cache_dir == (tmp_path / "cache").resolve()
    assert config.trace == (tmp_path / "trace.jsonl").resolve()
    assert not config.trace.exists()
    with pytest.raises(SystemExit):
        parse_server_args(
            [
                "--game-id",
                "fixture-89abcdef",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--trace",
                str(tmp_path / "trace.jsonl"),
                "--memory-db",
                str(tmp_path / "memory.sqlite"),
            ]
        )


def test_stdio_server_registers_the_expected_real_mcp_tools() -> None:
    runtime = ArcToolRuntime(fake_environment())
    try:
        server = create_mcp_server(runtime)
        tools = asyncio.run(server.list_tools())
    finally:
        runtime.close()
    assert {tool.name for tool in tools} == {
        "diff",
        "history",
        "inspect",
        "observe",
        "play",
        "propose_memory",
        "read_pixels",
        "segments",
    }


def test_stdio_server_approves_only_resolved_evidence_linked_memory(tmp_path: Path) -> None:
    runtime = ArcToolRuntime(fake_environment())
    memory = MemoryStore(tmp_path / "memory.sqlite", run_id="260901-000000_test")
    try:
        server = create_mcp_server(runtime, memory)
        evidence = next(iter(runtime.evidence_hashes))
        asyncio.run(
            server.call_tool(
                "propose_memory",
                {
                    "claim": "the initial frame contains no colored cells",
                    "status": "verified",
                    "confidence": 1.0,
                    "evidence": [evidence],
                },
            )
        )
        asyncio.run(
            server.call_tool(
                "propose_memory",
                {
                    "claim": "ACTION2 may cycle the bottom-right cell",
                    "status": "hypothesis",
                },
            )
        )
        records = memory.list()
    finally:
        memory.close()
        runtime.close()
    assert records[0].approved_for_warm is True
    assert records[1].approved_for_warm is False


def test_setup_downloads_complete_catalog_and_hashes_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[object] = []

    class SetupArcade:
        def __init__(self, *, environments_dir: str, operation_mode: object) -> None:
            self.root = Path(environments_dir)
            self.operation_mode = operation_mode
            self.available_environments = [
                SimpleNamespace(game_id="game-a-11111111"),
                SimpleNamespace(game_id="game-b-22222222"),
            ]
            self._default_scorecard_id = "empty-card"
            self.closed: list[str] = []
            instances.append(self)

        def make(self, game_id: str) -> object:
            (self.root / f"{game_id}.dat").write_text(game_id, encoding="utf-8")
            return object()

        def close_scorecard(self, scorecard_id: str) -> None:
            self.closed.append(scorecard_id)

    monkeypatch.setenv("ARC_API_KEY", "test-only-secret")
    root = tmp_path / "cache"
    manifest = setup_public_games(
        root,
        expected_game_count=2,
        arcade_builder=SetupArcade,
        verify_versions=False,
    )
    assert manifest.game_ids == ("game-a-11111111", "game-b-22222222")
    assert len(manifest.files) == 2
    assert len(manifest.aggregate_sha256) == 64
    assert "test-only-secret" not in "".join(path.read_text() for path in root.iterdir())
    assert instances[0].closed == ["empty-card"]


def test_dry_run_preflights_then_uses_exactly_one_official_scorecard(tmp_path: Path) -> None:
    game_id = "synthetic-00000001"
    trace = tmp_path / "win.jsonl"
    runtime = ArcToolRuntime(
        fake_environment(game_id, engine_version="0.9.3"),
        trace_path=trace,
        episode_per_level=False,
    )
    runtime.play("ACTION1")
    runtime.play("ACTION1")
    runtime.close()
    created: list[object] = []

    class RawEnvironment:
        def __init__(self) -> None:
            self.environment = fake_environment(game_id, engine_version="0.9.3")
            self.info = SimpleNamespace(
                game_id=game_id,
                baseline_actions=[2, 4],
                version="fixture-1",
            )
            self.observation_space = self._raw_frame()

        def _raw_frame(self) -> object:
            frame = self.environment.frame
            return SimpleNamespace(
                frame=[frame.grid],
                state=SimpleNamespace(value=frame.status.value),
                levels_completed=frame.levels_completed,
                win_levels=frame.win_levels,
                available_actions=[
                    SimpleNamespace(name=action.value)
                    for action in frame.available_actions
                ],
            )

        def step(self, action: object, *, data: dict[str, int] | None = None) -> object:
            row = None if data is None else data["y"]
            col = None if data is None else data["x"]
            self.environment.step(ArcAction[action.name], row=row, col=col)
            return self._raw_frame()

        def close(self) -> None:
            self.environment.close()

    class ScorecardArcade:
        def __init__(self, *, environments_dir: str, operation_mode: object) -> None:
            self.environments_dir = environments_dir
            self.operation_mode = operation_mode
            self.opened: list[list[str]] = []
            self.closed: list[str] = []
            self.made: list[tuple[str, str]] = []
            created.append(self)

        def open_scorecard(self, *, tags: list[str]) -> str:
            self.opened.append(tags)
            return "one-card"

        def make(self, requested_game: str, *, scorecard_id: str) -> RawEnvironment:
            self.made.append((requested_game, scorecard_id))
            return RawEnvironment()

        def close_scorecard(self, scorecard_id: str) -> dict[str, object]:
            self.closed.append(scorecard_id)
            return {
                "card_id": scorecard_id,
                "score": 100.0,
                "competition_mode": False,
                "environments": [
                    {
                        "id": game_id,
                        "score": 100.0,
                        "runs": [
                            {
                                "guid": "fixture-guid",
                                "score": 100.0,
                                "levels_completed": 2,
                                "actions": 2,
                                "resets": 0,
                                "state": "WIN",
                                "completed": True,
                                "level_actions": [1, 1],
                                "level_baseline_actions": [2, 4],
                                "level_scores": [115.0, 115.0],
                            }
                        ],
                    }
                ],
                "total_environments": 1,
                "total_environments_completed": 1,
                "total_levels": 2,
                "total_levels_completed": 2,
                "total_actions": 2,
            }

    report = submit_scorecard(
        [trace],
        tmp_path / "cache",
        mode=ScorecardMode.DRY_RUN,
        local_factory=lambda requested: fake_environment(requested, engine_version="0.9.3"),
        expected_game_count=1,
        arcade_builder=ScorecardArcade,
    )
    scorecard = created[0]
    assert scorecard.opened == [["ardea-avo"]]
    assert scorecard.made == [(game_id, "one-card")]
    assert scorecard.closed == ["one-card"]
    assert report.scorecard_id == "one-card"
    assert report.official_response["score"] == 100.0
    assert report.replays[0].levels_completed == 2


def test_scorecard_close_response_must_match_opened_card_and_replays(tmp_path: Path) -> None:
    """
    A missing response cannot be promoted into official submission evidence.
    """

    game_id = "synthetic-00000001"
    trace = tmp_path / "win.jsonl"
    runtime = ArcToolRuntime(
        fake_environment(game_id, engine_version="0.9.3"),
        trace_path=trace,
        episode_per_level=False,
    )
    runtime.play("ACTION1")
    runtime.play("ACTION1")
    runtime.close()

    class RawEnvironment:
        def __init__(self) -> None:
            self.environment = fake_environment(game_id, engine_version="0.9.3")
            self.info = SimpleNamespace(
                game_id=game_id,
                baseline_actions=[2, 4],
                version="fixture-1",
            )
            self.observation_space = self._raw_frame()

        def _raw_frame(self) -> object:
            frame = self.environment.frame
            return SimpleNamespace(
                frame=[frame.grid],
                state=SimpleNamespace(value=frame.status.value),
                levels_completed=frame.levels_completed,
                win_levels=frame.win_levels,
                available_actions=[
                    SimpleNamespace(name=action.value)
                    for action in frame.available_actions
                ],
            )

        def step(self, action: object, *, data: dict[str, int] | None = None) -> object:
            row = None if data is None else data["y"]
            col = None if data is None else data["x"]
            self.environment.step(ArcAction[action.name], row=row, col=col)
            return self._raw_frame()

        def close(self) -> None:
            self.environment.close()

    class MissingResponseArcade:
        def __init__(self, *, environments_dir: str, operation_mode: object) -> None:
            self.operation_mode = operation_mode

        def open_scorecard(self, *, tags: list[str]) -> str:
            return "one-card"

        def make(self, requested_game: str, *, scorecard_id: str) -> RawEnvironment:
            assert requested_game == game_id
            assert scorecard_id == "one-card"
            return RawEnvironment()

        def close_scorecard(self, scorecard_id: str) -> None:
            return None

    with pytest.raises(ScorecardSubmissionError):
        submit_scorecard(
            [trace],
            tmp_path / "cache",
            mode=ScorecardMode.DRY_RUN,
            local_factory=lambda requested: fake_environment(
                requested,
                engine_version="0.9.3",
            ),
            expected_game_count=1,
            arcade_builder=MissingResponseArcade,
        )
