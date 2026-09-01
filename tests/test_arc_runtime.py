"""
Deterministic mandatory tests for the ARC domain runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ardea_avo.arc import (
    ArcAction,
    ArcContractError,
    ArcFrame,
    ArcGameInfo,
    ArcToolRuntime,
    GameStatus,
    LocalArcMcpSurface,
    ReplayDivergence,
    TraceIntegrityError,
    action_budget,
    connected_segments,
    grid_sha256,
    load_trace,
    normalize_grid,
    validate_replay,
)


def grid_with(*cells: tuple[int, int, int]) -> tuple[tuple[int, ...], ...]:
    rows = [[0 for _ in range(64)] for _ in range(64)]
    for row, col, color in cells:
        rows[row][col] = color
    return tuple(tuple(row) for row in rows)


@dataclass
class FakeArcEnvironment:
    info: ArcGameInfo

    def __post_init__(self) -> None:
        self._grid = [list(row) for row in grid_with()]
        self._levels = 0
        self._status = GameStatus.NOT_FINISHED
        self._action1_count = 0
        self.closed = False

    @property
    def frame(self) -> ArcFrame:
        return ArcFrame(
            grid=tuple(tuple(row) for row in self._grid),
            status=self._status,
            levels_completed=self._levels,
            win_levels=2,
            available_actions=(ArcAction.ACTION1, ArcAction.ACTION2, ArcAction.ACTION6),
        )

    def step(
        self,
        action: ArcAction,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> ArcFrame:
        action = ArcAction.coerce(action)
        if action not in self.frame.legal_actions:
            raise ValueError("illegal fake action")
        if action is ArcAction.ACTION1:
            self._action1_count += 1
            self._levels = min(2, self._action1_count)
            self._grid[self._levels - 1][self._action1_count - 1] = self._levels
            if self._levels == 2:
                self._status = GameStatus.WIN
        elif action is ArcAction.ACTION2:
            self._grid[63][63] = (self._grid[63][63] + 1) % 16
        elif action is ArcAction.ACTION6:
            if row is None or col is None:
                raise ValueError("coordinates required")
            self._grid[row][col] = 6
        elif action is ArcAction.RESET:
            self._status = GameStatus.NOT_FINISHED
        return self.frame

    def close(self) -> None:
        self.closed = True


def fake_environment(
    game_id: str = "synthetic-00000001",
    *,
    baselines: tuple[int, ...] = (2, 4),
    engine_version: str = "test-engine-1",
) -> FakeArcEnvironment:
    return FakeArcEnvironment(
        ArcGameInfo(
            game_id=game_id,
            baseline_actions=baselines,
            engine_version=engine_version,
            environment_version="fixture-1",
        )
    )


def test_exact_grid_validation_and_hashing() -> None:
    grid = grid_with((0, 0, 15), (63, 63, 7))
    assert normalize_grid(grid) == grid
    assert grid_sha256(grid) == grid_sha256([list(row) for row in grid])
    assert len(grid_sha256(grid)) == 64
    with pytest.raises(ArcContractError, match="64 rows"):
        normalize_grid(grid[:-1])
    bad = [list(row) for row in grid]
    bad[4][5] = 16
    with pytest.raises(ArcContractError, match="0 through 15"):
        normalize_grid(bad)
    bad[4][5] = 1.0
    with pytest.raises(ArcContractError, match="not an integer"):
        normalize_grid(bad)


def test_observe_is_exact_text_and_hides_baselines() -> None:
    runtime = ArcToolRuntime(fake_environment(baselines=(13, 17)))
    observation = runtime.observe()
    rows = [line for line in observation.splitlines() if line.startswith("r")]
    assert len(rows) == 64
    assert rows[0] == "r00 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000"
    assert rows[-1].startswith("r63 ")
    assert "grid=64x64" in observation
    assert "baseline" not in observation.lower()
    assert "150" not in observation
    assert "actions=0" in observation
    runtime.close()


def test_agent_visible_trace_does_not_contain_private_baseline_values(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    runtime = ArcToolRuntime(
        fake_environment(baselines=(1_234_567, 7_654_321)),
        trace_path=path,
    )
    runtime.close()
    raw = path.read_text(encoding="utf-8")
    assert "1234567" not in raw
    assert "7654321" not in raw
    assert "baseline_actions" not in raw
    assert "baseline_digest" in raw


def test_play_is_only_counted_operation_and_validates_before_step(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    runtime = ArcToolRuntime(fake_environment(), trace_path=path, episode_per_level=False)
    surface = LocalArcMcpSurface(runtime)

    surface.call_tool("observe")
    surface.call_tool("inspect", {"top": 0, "left": 0, "bottom": 2, "right": 2})
    surface.call_tool("read_pixels", {"coordinates": [{"row": 0, "col": 0}]})
    surface.call_tool("history")
    surface.call_tool("diff")
    surface.call_tool("segments")
    assert runtime.action_count == 0

    invalid = surface.call_tool("play", {"action": "ACTION6", "row": 0})
    assert invalid["isError"] is True
    assert runtime.action_count == 0
    extra = surface.call_tool("play", {"action": "ACTION2", "surprise": True})
    assert extra["isError"] is True
    assert runtime.action_count == 0

    valid = surface.call_tool(
        "play",
        {"action": "ACTION6", "row": 4, "col": 5, "expected_outcome": "mark target"},
    )
    assert valid["isError"] is False
    assert valid["_meta"]["counted"] is True
    assert runtime.action_count == 1
    assert "r04c05:0>6" in valid["content"][0]["text"]
    assert "r04c05=6" in runtime.read_pixels(((4, 5),))
    runtime.close()
    assert len(load_trace(path).steps) == 1


def test_action_budget_is_private_and_enforced() -> None:
    runtime = ArcToolRuntime(fake_environment(baselines=(1, 1)), episode_per_level=False)
    assert runtime.action_limit == 10
    for _ in range(10):
        assert runtime.play("ACTION2").counted
    refused = runtime.play("ACTION2")
    assert refused.counted is False
    assert refused.terminal is True
    assert "exhausted" in refused.content
    assert "10" not in runtime.observe().splitlines()[0].split("legal=")[1]
    runtime.close()
    assert action_budget((1_000, 1_000)) == 2_500


def test_level_completion_blocks_more_actions_until_fresh_server_resume(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    runtime = ArcToolRuntime(fake_environment(), trace_path=path)
    completed = runtime.play("ACTION1")
    assert completed.counted is True
    assert completed.terminal is True
    assert "episode_boundary=level_complete" in completed.content
    blocked = runtime.play("ACTION2")
    assert blocked.counted is False
    assert blocked.terminal is True
    assert runtime.action_count == 1
    runtime.close()

    resumed = ArcToolRuntime.resume(fake_environment(), path)
    continued = resumed.play("ACTION1")
    assert continued.counted is True
    assert resumed.frame.status is GameStatus.WIN
    resumed.close()


def test_inspection_diff_history_and_segments_do_not_act() -> None:
    runtime = ArcToolRuntime(fake_environment(), episode_per_level=False)
    runtime.play("ACTION6", row=4, col=5)
    runtime.play("ACTION6", row=4, col=6)
    assert "grid=2x2" in runtime.inspect(turn=2, top=4, left=5, bottom=6, right=7)
    assert "2 cells changed" in runtime.diff(before_turn=0, after_turn=2)
    history = runtime.history()
    assert "turn=1; action=ACTION6(r04,c05)" in history
    segments = json.loads(runtime.segments())
    assert segments["count"] == 1
    assert segments["segments"][0] == {
        "bottom": 4,
        "color": 6,
        "left": 5,
        "right": 6,
        "size": 2,
        "top": 4,
    }
    assert runtime.action_count == 2
    runtime.close()


def test_connected_segments_are_four_connected() -> None:
    segments = connected_segments(grid_with((1, 1, 3), (1, 2, 3), (2, 2, 3), (3, 3, 3)))
    assert [(segment.color, segment.size) for segment in segments] == [(3, 3), (3, 1)]


def test_trace_replays_exactly_and_resume_appends(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    runtime = ArcToolRuntime(fake_environment(), trace_path=path, episode_per_level=False)
    runtime.play("ACTION2")
    runtime.play("ACTION6", row=7, col=8)
    runtime.close()

    result = validate_replay(path, lambda game_id: fake_environment(game_id))
    assert result.verified
    assert result.actions == 2
    assert result.status is GameStatus.NOT_FINISHED

    resumed = ArcToolRuntime.resume(fake_environment(), path, episode_per_level=False)
    assert resumed.action_count == 2
    assert "r07c08=6" in resumed.read_pixels(((7, 8),))
    resumed.play("ACTION1")
    resumed.close()
    data = load_trace(path)
    assert [step.number for step in data.steps] == [1, 2, 3]
    assert validate_replay(path, lambda game_id: fake_environment(game_id)).actions == 3


def test_trace_tamper_and_replay_drift_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    runtime = ArcToolRuntime(fake_environment(), trace_path=path)
    runtime.play("ACTION2")
    runtime.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    altered = json.loads(lines[1])
    altered["levels_completed"] = 1
    lines[1] = json.dumps(altered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="hash"):
        load_trace(path)

    clean = tmp_path / "clean.jsonl"
    runtime = ArcToolRuntime(fake_environment(), trace_path=clean)
    runtime.play("ACTION2")
    runtime.close()
    with pytest.raises(ReplayDivergence, match="engine version"):
        validate_replay(clean, lambda game_id: fake_environment(game_id, engine_version="drifted"))


def test_torn_trace_is_never_silently_recovered(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    runtime = ArcToolRuntime(fake_environment(), trace_path=path)
    runtime.play("ACTION2")
    runtime.close()
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(TraceIntegrityError, match="torn"):
        load_trace(path)
