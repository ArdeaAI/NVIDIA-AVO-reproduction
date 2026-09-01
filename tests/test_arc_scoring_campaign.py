"""
Scoring and verified campaign-bank tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_arc_runtime import fake_environment

from ardea_avo.arc import (
    ArcToolRuntime,
    BankValidationError,
    CampaignBank,
    GameStatus,
    ReplayResult,
    actions_per_completed_level,
    board_rhae,
    level_rhae,
    score_game,
    trace_sha256,
    validate_replay,
)


def make_winning_trace(path: Path, *, extra_actions: int = 0) -> ReplayResult:
    runtime = ArcToolRuntime(fake_environment(), trace_path=path, episode_per_level=False)
    for _ in range(extra_actions):
        runtime.play("ACTION2")
    runtime.play("ACTION1")
    runtime.play("ACTION1")
    runtime.close()
    return validate_replay(path, lambda game_id: fake_environment(game_id))


def test_actions_are_charged_to_completed_levels() -> None:
    records = [
        {"levels_completed": 0},
        {"levels_completed": 1},
        {"levels_completed": 1},
        {"levels_completed": 2},
        {"levels_completed": 2},
    ]
    assert actions_per_completed_level(records) == (2, 2)
    assert actions_per_completed_level([{"levels_completed": 2}]) == (1, 0)
    with pytest.raises(ValueError, match="decrease"):
        actions_per_completed_level([{"levels_completed": 1}, {"levels_completed": 0}])


def test_official_level_game_and_board_scoring() -> None:
    assert level_rhae(10, 10) == 1.0
    assert level_rhae(10, 1) == 1.15
    assert level_rhae(10, 0) == 0.0
    partial = score_game((10, 10), (10,))
    assert partial.score == pytest.approx(1 / 3)
    complete = score_game((10, 10), (1, 1))
    assert complete.score == 1.0
    assert board_rhae((partial, complete)) == pytest.approx(200 / 3)


def test_local_level_scoring_matches_pinned_official_calculator() -> None:
    """
    Guard the subtle zero-action rule against drift from arc-agi 0.9.9.
    """

    from arc_agi.scorecard import EnvironmentScoreCalculator

    for actions in (0, 1, 5, 10, 20):
        calculator = EnvironmentScoreCalculator()
        calculator.add_level(
            level_index=1,
            completed=True,
            actions_taken=actions,
            baseline_actions=10,
        )
        official = calculator.level_scores[0] / 100.0
        assert level_rhae(10, actions) == pytest.approx(official)
        assert score_game((10,), (actions,)).score == pytest.approx(
            calculator.to_score().score / 100.0
        )


@pytest.mark.parametrize("actions", ((1, 0), (0, 0), (1, 0, 0)))
def test_multi_level_jump_ceiling_matches_pinned_official_calculator(
    actions: tuple[int, ...],
) -> None:
    """
    Zero-action levels completed by a jump do not enlarge the score ceiling.
    """

    from arc_agi.scorecard import EnvironmentScoreCalculator

    baselines = (10,) * len(actions)
    calculator = EnvironmentScoreCalculator()
    for level_index, actions_taken in enumerate(actions, start=1):
        calculator.add_level(
            level_index=level_index,
            completed=True,
            actions_taken=actions_taken,
            baseline_actions=10,
        )
    assert score_game(baselines, actions).score == pytest.approx(
        calculator.to_score().score / 100.0
    )


def test_campaign_banks_first_verified_win(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_replay = make_winning_trace(first, extra_actions=2)
    second_replay = make_winning_trace(second)
    manifest = tmp_path / "bank.json"
    bank = CampaignBank(manifest)
    assert bank.consider(first, first_replay)
    assert bank.consider(second, second_replay) is False
    assert bank.selected(first_replay.game_id).trace_path == str(first.resolve())

    assert bank.consider(second, second_replay, optimize_actions=True)
    selected = bank.selected(second_replay.game_id)
    assert selected is not None
    assert selected.actions == 2
    assert selected.rhae_percent == 100.0
    assert CampaignBank(manifest).validate_selected(second_replay.game_id) == selected
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["schema"] == "ardea.arc.campaign-bank.v1"


def test_campaign_rejects_unverified_or_changed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "win.jsonl"
    replay = make_winning_trace(path)
    bank = CampaignBank()
    fabricated = ReplayResult(
        verified=False,
        game_id=replay.game_id,
        status=GameStatus.WIN,
        levels_completed=2,
        win_levels=2,
        actions=2,
        per_level_actions=(1, 1),
        rhae_percent=100.0,
        final_frame_hash=replay.final_frame_hash,
        trace_sha256=trace_sha256(path),
    )
    with pytest.raises(BankValidationError, match="replay evidence"):
        bank.consider(path, fabricated)

    persistent = CampaignBank(tmp_path / "bank.json")
    persistent.consider(path, replay)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BankValidationError, match="changed"):
        persistent.validate_selected(replay.game_id)
