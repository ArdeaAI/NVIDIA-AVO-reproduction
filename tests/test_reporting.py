"""
Tests for offline run reports and atomic report persistence.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_arc_runtime import fake_environment

from ardea_avo.arc import ArcToolRuntime, CampaignBank, validate_replay
from ardea_avo.reporting import (
    ReportStatus,
    RunReport,
    SubmissionSummary,
    build_run_report,
    read_report,
    write_report,
)
from ardea_avo.runtime import BudgetLedger, ResultsManager, TokenUsage

NOW = datetime(2026, 9, 1, 13, 20, 26, tzinfo=UTC)
GAME_ID = "synthetic-00000001"


def _winning_trace(path: Path) -> None:
    runtime = ArcToolRuntime(fake_environment(), trace_path=path, episode_per_level=False)
    runtime.play("ACTION1")
    runtime.play("ACTION1")
    runtime.close()


def _exploratory_trace(path: Path) -> None:
    runtime = ArcToolRuntime(fake_environment(), trace_path=path, episode_per_level=False)
    runtime.play("ACTION2")
    runtime.close()


def _run_and_bank(tmp_path: Path):
    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    run = manager.create_cold("report")
    winning_path = run.directory / "games" / "fake" / "winning.jsonl"
    winning_path.parent.mkdir(parents=True)
    _winning_trace(winning_path)
    replay = validate_replay(winning_path, lambda game_id: fake_environment(game_id))
    bank = CampaignBank(run.directory / "bank.json")
    assert bank.consider(winning_path, replay)
    budget = BudgetLedger(run.directory)
    budget.record_usage(
        TokenUsage(input_tokens=100, cached_input_tokens=40, output_tokens=20),
        backend="codex-oauth",
        role="player",
    )
    return run, bank, budget, winning_path


def test_report_has_complete_board_with_zero_for_missing_game(tmp_path: Path) -> None:
    """
    Missing games lower the official board average instead of disappearing.
    """

    run, bank, budget, winning_path = _run_and_bank(tmp_path)
    exploration = run.directory / "games" / "fake" / "exploration.jsonl"
    _exploratory_trace(exploration)
    report = build_run_report(
        run,
        bank,
        budget,
        expected_games={GAME_ID: 2, "missing-game": 3},
        trace_paths=(winning_path, exploration, exploration),
        generated_at="2026-09-01T13:30:00Z",
    )
    assert report.status is ReportStatus.IN_PROGRESS
    assert report.expected_games == 2
    assert report.solved_games == 1
    assert report.expected_levels == 5
    assert report.solved_levels == 2
    assert report.rhae_percent == pytest.approx(50.0)
    assert report.submitted_actions == 2
    assert report.exploratory_actions == 1
    assert report.total_environment_actions == 3
    missing = next(game for game in report.games if game.game_id == "missing-game")
    assert missing.rhae_percent == 0.0
    assert report.usage.input_tokens == 100
    assert report.usage.cached_input_tokens == 40
    assert report.usage.output_tokens == 20


def test_full_replay_validation_and_competition_acceptance_are_distinct(tmp_path: Path) -> None:
    """
    Local completeness is not the official acceptance gate.
    """

    run, bank, budget, winning_path = _run_and_bank(tmp_path)
    local = build_run_report(
        run,
        bank,
        budget,
        expected_games={GAME_ID: 2},
        trace_paths=(winning_path,),
        fresh_replay_validated=True,
    )
    assert local.status is ReportStatus.LOCALLY_VALIDATED
    assert not local.submission.acceptance_met

    submission = SubmissionSummary(
        mode="competition",
        completed=True,
        scorecard_id="scorecard-1",
        submitted_at="2026-09-01T14:00:00Z",
        official_rhae_percent=100.0,
        official_games_solved=1,
        official_levels_solved=2,
        official_submitted_actions=2,
        official_response_sha256="c" * 64,
    )
    official = build_run_report(
        run,
        bank,
        budget,
        expected_games={GAME_ID: 2},
        trace_paths=(winning_path,),
        fresh_replay_validated=True,
        submission=submission,
    )
    assert official.status is ReportStatus.COMPETITION_SUBMITTED
    assert official.submission.acceptance_met


def test_contamination_is_deduplicated_and_disqualifies_status(tmp_path: Path) -> None:
    """
    A complete score remains visibly contaminated rather than locally validated.
    """

    run, bank, budget, winning_path = _run_and_bank(tmp_path)
    report = build_run_report(
        run,
        bank,
        budget,
        expected_games={GAME_ID: 2},
        trace_paths=(winning_path,),
        fresh_replay_validated=True,
        contamination=("manual inherited", "manual inherited"),
    )
    assert report.status is ReportStatus.CONTAMINATED
    assert report.contamination == ("manual inherited",)


def test_report_atomic_json_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    """
    Persisted reports validate their nested totals when read back.
    """

    run, bank, budget, winning_path = _run_and_bank(tmp_path)
    report = build_run_report(
        run,
        bank,
        budget,
        expected_games={GAME_ID: 2},
        trace_paths=(winning_path,),
    )
    path = write_report(tmp_path / "report.json", report)
    assert read_report(path) == report
    assert RunReport.read(path) == report
    assert not tuple(tmp_path.glob(".report.json.*.tmp"))

    document = json.loads(path.read_text(encoding="utf-8"))
    document["submitted_actions"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="submitted"):
        read_report(path)


def test_report_rejects_unexpected_bank_or_trace_game(tmp_path: Path) -> None:
    """
    Reporting fails closed when artifacts are outside the pinned catalog.
    """

    run, bank, budget, winning_path = _run_and_bank(tmp_path)
    with pytest.raises(ValueError, match="unexpected games"):
        build_run_report(
            run,
            bank,
            budget,
            expected_games={"different-game": 2},
            trace_paths=(winning_path,),
        )


def test_submission_summary_reads_official_scorecard_without_persisting_response() -> None:
    """
    Pydantic-like official responses are reduced to metrics and a sanitized digest.
    """

    response = SimpleNamespace(
        score=100.0,
        total_environments_completed=25,
        total_levels_completed=183,
        total_actions=6624,
        model_dump=lambda **_kwargs: {
            "score": 100.0,
            "total_environments_completed": 25,
            "total_levels_completed": 183,
            "total_actions": 6624,
            "api_key": "must-not-be-hashed",
        },
    )
    scorecard = SimpleNamespace(
        mode=SimpleNamespace(value="competition"),
        scorecard_id="official-card",
        replays=(),
        official_response=response,
    )
    summary = SubmissionSummary.from_scorecard(
        scorecard, submitted_at="2026-09-01T14:00:00Z"
    )
    assert summary.official_rhae_percent == 100.0
    assert summary.official_games_solved == 25
    assert summary.official_levels_solved == 183
    assert summary.official_submitted_actions == 6624
    assert summary.official_response_sha256 is not None
    response.model_dump = lambda **_kwargs: {
        "score": 100.0,
        "total_environments_completed": 25,
        "total_levels_completed": 183,
        "total_actions": 6624,
        "api_key": "a-different-secret",
    }
    assert (
        SubmissionSummary.from_scorecard(
            scorecard, submitted_at="2026-09-01T14:00:00Z"
        ).official_response_sha256
        == summary.official_response_sha256
    )
