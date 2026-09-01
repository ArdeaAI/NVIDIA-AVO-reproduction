"""
Tests for run identity, immutable provenance, and recovery artifacts.
"""

import json
import sqlite3
from datetime import UTC, datetime

import pytest

from ardea_avo.runtime.budget import BudgetLedger
from ardea_avo.runtime.memory import MemoryStore
from ardea_avo.runtime.results import EventChainError, ResultsManager, RunMode

NOW = datetime(2026, 9, 1, 13, 20, 26, tzinfo=UTC)


def test_cold_run_id_manifest_events_and_checkpoint_round_trip(tmp_path) -> None:
    """
    A cold run is canonical, parentless, and recoverable from atomic state.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    run = manager.create_cold(
        "My ARC Run!",
        config={"games": 25},
        provenance={"source_revision": "abc123"},
    )
    assert run.manifest.run_id == "260901-132026_my_arc_run"
    assert run.manifest.mode is RunMode.COLD
    assert run.manifest.parent_run_id is None
    event = run.append_event("game.started", {"game_id": "game-1"})
    checkpoint = run.write_checkpoint({"next_game": 1})
    assert checkpoint["event_hash"] == event["hash"]

    reopened = manager.resume(
        run.manifest.run_id,
        expected={"model": "gpt-5.6-sol", "config": {"games": 25}},
    )
    assert reopened.read_checkpoint()["payload"] == {"next_game": 1}


def test_event_tampering_is_detected(tmp_path) -> None:
    """
    Editing an earlier event invalidates the full run on reopen.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    run = manager.create_cold("tamper")
    records = [json.loads(line) for line in run.events_path.read_text().splitlines()]
    records[0]["payload"]["mode"] = "warm"
    run.events_path.write_text("\n".join(json.dumps(item) for item in records) + "\n")
    with pytest.raises(EventChainError, match="hash mismatch"):
        manager.open(run.manifest.run_id)


def test_warm_child_seals_parent_and_binds_snapshot(tmp_path) -> None:
    """
    Creating a warm child freezes its parent and records the exact event head.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    parent = manager.create_cold("parent")
    parent.append_event("memory.approved", {"record_id": "memory-1"})
    child = manager.create_warm(parent.manifest.run_id, "child")
    assert parent.is_sealed
    assert child.manifest.mode is RunMode.WARM
    assert child.manifest.parent_snapshot is not None
    with pytest.raises(RuntimeError, match="immutable"):
        parent.append_event("late.event", {})
    resumed = manager.resume(child.manifest.run_id)
    resumed.assert_parent_unchanged(manager.open(parent.manifest.run_id))


def test_warm_child_refuses_parent_with_inflight_budget_reservation(tmp_path) -> None:
    """
    Parent sealing cannot strand a turn that may still consume shared spend.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    parent = manager.create_cold("reserved-parent")
    ledger = BudgetLedger(parent.directory)
    reservation = ledger.reserve("1", role="player")
    with pytest.raises(RuntimeError, match="active model reservations"):
        manager.create_warm(parent.manifest.run_id, "child")
    ledger.release(reservation)
    child = manager.create_warm(parent.manifest.run_id, "child")
    assert child.manifest.parent_run_id == parent.manifest.run_id


def test_warm_child_detects_mutation_of_a_sealed_parent_artifact(tmp_path) -> None:
    """
    The parent snapshot covers durable run files, not only its event chain.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    parent = manager.create_cold("parent-artifacts")
    artifact = parent.directory / "bank.json"
    artifact.write_text('{"before":true}\n', encoding="utf-8")
    child = manager.create_warm(parent.manifest.run_id, "artifact-child")
    artifact.write_text('{"after":true}\n', encoding="utf-8")
    with pytest.raises(EventChainError, match="artifacts changed"):
        manager.resume(child.manifest.run_id)


def test_warm_child_detects_post_seal_sqlite_wal_mutation(tmp_path) -> None:
    """
    A writer kept open across sealing cannot change parent memory undetected.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    parent = manager.create_cold("parent-wal")
    store = MemoryStore(parent.directory / "memory.sqlite", run_id=parent.manifest.run_id)
    child = manager.create_warm(parent.manifest.run_id, "child-wal")
    with pytest.raises(RuntimeError, match="sealed parent"):
        store.add(scope="run", claim="late mutation")
    store.close()

    connection = sqlite3.connect(parent.directory / "memory.sqlite")
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("post_seal_tamper", "true"),
        )
        connection.commit()
        with pytest.raises(EventChainError, match="artifacts changed"):
            manager.resume(child.manifest.run_id)
    finally:
        connection.close()


def test_resume_rejects_provenance_drift(tmp_path) -> None:
    """
    Callers can bind resume to exact dependency and game versions.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    run = manager.create_cold(
        "versions",
        provenance={"dependencies": {"arc-agi": "0.9.9"}},
    )
    with pytest.raises(EventChainError, match="provenance mismatch"):
        manager.resume(
            run.manifest.run_id,
            expected={"provenance": {"dependencies": {"arc-agi": "1.0"}}},
        )


def test_secret_bearing_manifest_and_events_are_refused(tmp_path) -> None:
    """
    Results storage never accepts auth material even in flexible metadata.
    """

    manager = ResultsManager(tmp_path / "results", clock=lambda: NOW)
    with pytest.raises(ValueError, match="secret-bearing"):
        manager.create_cold("secret", config={"openai_api_key": "do-not-store"})
    run = manager.create_cold("safe")
    with pytest.raises(ValueError, match="secret-bearing"):
        run.append_event("bad", {"access_token": "do-not-store"})
