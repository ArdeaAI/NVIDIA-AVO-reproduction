"""
Synthetic full-board tests for the offline campaign acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ardea_avo.arc import (
    ArcAction,
    ArcFrame,
    ArcGameInfo,
    ArcToolRuntime,
    CampaignBank,
    GameStatus,
    OfficialGameDescriptor,
    validate_replay,
)
from ardea_avo.runtime import ResultsManager
from ardea_avo.validation import validate_campaign

NOW = datetime(2026, 9, 1, 13, 20, 26, tzinfo=UTC)
GRID = tuple(tuple(0 for _ in range(64)) for _ in range(64))


@dataclass
class SequentialEnvironment:
    """
    Complete exactly one synthetic level per legal action.
    """

    game_id: str
    levels: int
    completed: int = 0

    @property
    def info(self) -> ArcGameInfo:
        """
        Return pinned synthetic metadata.
        """

        return ArcGameInfo(
            game_id=self.game_id,
            baseline_actions=(1,) * self.levels,
            engine_version="test-engine",
            environment_version="test-environment",
        )

    @property
    def frame(self) -> ArcFrame:
        """
        Return the current deterministic frame.
        """

        return ArcFrame(
            grid=GRID,
            status=(
                GameStatus.WIN
                if self.completed == self.levels
                else GameStatus.NOT_FINISHED
            ),
            levels_completed=self.completed,
            win_levels=self.levels,
            available_actions=(
                () if self.completed == self.levels else (ArcAction.ACTION1,)
            ),
        )

    def step(
        self,
        action: ArcAction,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> ArcFrame:
        """
        Complete the next level on ACTION1.
        """

        assert action is ArcAction.ACTION1
        assert row is None and col is None
        self.completed += 1
        return self.frame

    def close(self) -> None:
        """
        Close the no-resource fixture.
        """


def _descriptors() -> tuple[OfficialGameDescriptor, ...]:
    return tuple(
        OfficialGameDescriptor(
            game_id=f"synthetic-{index:08d}",
            title=None,
            levels=8 if index < 8 else 7,
        )
        for index in range(25)
    )


def _factory(levels: dict[str, int]):
    def create(game_id: str) -> SequentialEnvironment:
        return SequentialEnvironment(game_id, levels[game_id])

    return create


def _context(tmp_path: Path, descriptors: tuple[OfficialGameDescriptor, ...]):
    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    roster = [
        {"game_id": item.game_id, "levels": item.levels}
        for item in sorted(descriptors, key=lambda item: item.game_id)
    ]
    return manager.create_cold("validation", config={"game_roster": roster})


def test_partial_board_is_incomplete_without_being_corrupt(tmp_path: Path) -> None:
    """
    Missing bank entries score zero but do not masquerade as integrity errors.
    """

    descriptors = _descriptors()
    context = _context(tmp_path, descriptors)
    levels = {item.game_id: item.levels for item in descriptors}

    result = validate_campaign(context, descriptors, _factory(levels))

    assert not result.valid
    assert not result.eligible_for_competition
    assert result.solved_games == 0
    assert result.solved_levels == 0
    assert result.board_rhae_percent == 0.0
    assert result.errors == ()


def test_exact_25_game_183_level_board_qualifies_at_100(tmp_path: Path) -> None:
    """
    Every selected trace is freshly replayed and occupies one board slot.
    """

    descriptors = _descriptors()
    context = _context(tmp_path, descriptors)
    levels = {item.game_id: item.levels for item in descriptors}
    factory = _factory(levels)
    bank = CampaignBank(context.directory / "bank.json")
    for descriptor in descriptors:
        path = context.directory / "games" / descriptor.game_id / "attempt-001" / "trace.jsonl"
        runtime = ArcToolRuntime(factory(descriptor.game_id), trace_path=path, episode_per_level=False)
        for _ in range(descriptor.levels):
            runtime.play("ACTION1", expected_outcome="complete the next synthetic level")
        runtime.close()
        assert bank.consider(path, validate_replay(path, factory))

    result = validate_campaign(context, descriptors, factory)

    assert result.valid
    assert result.eligible_for_competition
    assert result.solved_games == 25
    assert result.solved_levels == 183
    assert result.submitted_actions == 183
    assert result.board_rhae_percent == 100.0


def test_contamination_disqualifies_without_erasing_valid_replays(tmp_path: Path) -> None:
    """
    Provenance contamination remains distinct from replay validity.
    """

    descriptors = _descriptors()
    context = _context(tmp_path, descriptors)
    levels = {item.game_id: item.levels for item in descriptors}

    result = validate_campaign(
        context,
        descriptors,
        _factory(levels),
        contamination=("source repository was dirty",),
    )

    assert not result.valid
    assert not result.eligible_for_competition
    assert result.errors == ()
    assert result.contamination == ("source repository was dirty",)
