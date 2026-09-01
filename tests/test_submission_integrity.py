"""
Submission artifacts remain bound to one run, bank, claim, and event chain.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_arc_runtime import fake_environment

from ardea_avo.arc import ArcToolRuntime, CampaignBank, validate_replay
from ardea_avo.cli import (
    OFFICIAL_ARC_BASE_URL,
    _read_submission,
    _submission_replay_roster,
)
from ardea_avo.reporting import SubmissionSummary
from ardea_avo.runtime import ResultsManager
from ardea_avo.runtime._io import atomic_write_json, sha256_json

NOW = datetime(2026, 9, 1, 13, 20, 26, tzinfo=UTC)


def _completed_submission(tmp_path: Path):
    """
    Construct a fully anchored local submission-evidence fixture.
    """

    context = ResultsManager(tmp_path / "results", clock=lambda: NOW).create_cold(
        "submission"
    )
    trace = context.directory / "games" / "game-001" / "attempt-001" / "trace.jsonl"
    runtime = ArcToolRuntime(
        fake_environment(),
        trace_path=trace,
        episode_per_level=False,
    )
    runtime.play("ACTION1")
    runtime.play("ACTION1")
    runtime.close()
    replay = validate_replay(trace, lambda game_id: fake_environment(game_id))
    CampaignBank(context.directory / "bank.json").consider(trace, replay)
    summary = SubmissionSummary(
        mode="dry-run",
        completed=True,
        scorecard_id="scorecard-1",
        submitted_at="2026-09-01T13:20:26Z",
        official_rhae_percent=100.0,
        official_games_solved=1,
        official_levels_solved=2,
        official_submitted_actions=2,
    )
    document = {
        "schema": "ardea.arc.submission.v2",
        "run_id": context.manifest.run_id,
        "manifest_sha256": sha256_json(context.manifest.to_dict()),
        "official_arc_base_url": OFFICIAL_ARC_BASE_URL,
        "mode": "dry-run",
        "completed_at": "2026-09-01T13:20:26Z",
        "summary": asdict(summary),
        "replays": _submission_replay_roster(context),
    }
    artifact = context.directory / "dry-run-submission.json"
    atomic_write_json(artifact, document)
    artifact_sha256 = sha256_json(document)
    atomic_write_json(
        context.directory / "dry-run-submission.claim.json",
        {
            "status": "completed",
            "run_id": context.manifest.run_id,
            "mode": "dry-run",
            "scorecard_id": "scorecard-1",
            "artifact_sha256": artifact_sha256,
        },
    )
    context.append_event(
        "scorecard.completed",
        {
            "mode": "dry-run",
            "scorecard_id": "scorecard-1",
            "artifact_sha256": artifact_sha256,
        },
    )
    return context, artifact, document


def test_submission_artifact_requires_matching_claim_and_event(tmp_path: Path) -> None:
    """
    A valid anchored artifact can be read without retaining its raw response.
    """

    context, _artifact, _document = _completed_submission(tmp_path)

    summary = _read_submission(context)

    assert summary is not None
    assert summary.scorecard_id == "scorecard-1"


def test_submission_artifact_tampering_breaks_event_binding(tmp_path: Path) -> None:
    """
    Replacing metrics after completion cannot inherit the original event anchor.
    """

    context, artifact, document = _completed_submission(tmp_path)
    document["summary"]["official_rhae_percent"] = 99.0
    atomic_write_json(artifact, document)

    with pytest.raises(ValueError, match=r"claim|anchored"):
        _read_submission(context)
