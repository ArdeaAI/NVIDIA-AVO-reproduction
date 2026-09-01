"""
Tests for the public command grammar and safe local helpers.
"""

import argparse
from decimal import Decimal
from pathlib import Path

import pytest

from ardea_avo.arc import OfficialGameDescriptor
from ardea_avo.cli import (
    EXPECTED_PUBLIC_ROSTER_SHA256,
    _budget_ambiguity_contamination,
    _campaign_limits,
    _canonical_public_roster,
    _command_setup,
    _positive_decimal,
    _public_roster_sha256,
    _record_unreconciled_reservations,
    _resume_arc_context,
    _setup_manifest_path,
    _unreconciled_reservation_payload,
    _uses_disclosed_model_lane,
    _validate_campaign_jobs,
    _write_exclusive_json,
    build_parser,
)
from ardea_avo.runtime import CLAUDE_OPUS_5_PRICING, BudgetLedger
from ardea_avo.runtime.results import ResultsManager


def test_arc_and_generic_modes_parse_as_documented() -> None:
    """
    ARC modes are top-level while generic modes follow the target path.
    """

    cold = build_parser().parse_args(["--cold", "--max-cost-usd", "25"])
    assert cold.cold
    assert str(cold.max_cost_usd) == "25"
    assert not hasattr(cold, "attempts")
    assert not hasattr(cold, "episodes_per_attempt")

    generic = build_parser().parse_args(
        ["evolve", "target.yaml", "--resume", "260901-120000_run", "--attempts", "7"]
    )
    assert generic.command == "evolve"
    assert generic.target == Path("target.yaml")
    assert generic.resume == "260901-120000_run"
    assert generic.attempts == 7
    assert generic.slug == "evolve"

    anthropic = build_parser().parse_args(["--cold", "--backend", "anthropic-api"])
    assert anthropic.backend == "anthropic-api"


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf", "not-money"))
def test_budget_parser_rejects_nonpositive_or_nonfinite_values(value: str) -> None:
    """
    CLI budget values cannot disable the hard start gate accidentally.
    """

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_decimal(value)


def test_public_roster_digest_is_order_stable_and_identity_sensitive() -> None:
    """
    Equal catalogs commit equally while a same-size ID substitution is detected.
    """

    versions = {"arc-agi": "0.9.9", "arcengine": "0.9.3"}
    roster = [
        {"game_id": "game-b-v1", "levels": 7},
        {"game_id": "game-a-v1", "levels": 8},
    ]
    digest = _public_roster_sha256(roster, versions)
    assert digest == _public_roster_sha256(list(reversed(roster)), versions)
    changed = [
        {"game_id": "game-c-v1", "levels": 7},
        {"game_id": "game-a-v1", "levels": 8},
    ]
    assert digest != _public_roster_sha256(changed, versions)
    with pytest.raises(ValueError, match="duplicate"):
        _public_roster_sha256([roster[0], roster[0]], versions)


def test_frozen_public_roster_has_exact_primary_identity() -> None:
    """
    The committed protocol snapshot fixes all 25 versioned games and 183 levels.
    """

    roster = _canonical_public_roster()
    versions = {"arc-agi": "0.9.9", "arcengine": "0.9.3"}
    assert len(roster) == 25
    assert sum(item["levels"] for item in roster) == 183
    assert _public_roster_sha256(roster, versions) == EXPECTED_PUBLIC_ROSTER_SHA256


def test_resume_campaign_limits_are_append_only_and_monotonic(tmp_path: Path) -> None:
    """
    An exhausted run can continue after explicit attempt or episode increases.
    """

    manager = ResultsManager(tmp_path / "results")
    run = manager.create_cold(
        "limits",
        config={
            "target": "arc-agi-3-public",
            "attempts_per_game": 1,
            "episodes_per_attempt": 2,
        },
    )
    args = argparse.Namespace(
        resume=run.manifest.run_id,
        backend=None,
        max_cost_usd=None,
        attempts=4,
        episodes_per_attempt=6,
    )
    resumed = _resume_arc_context(args, manager)
    assert _campaign_limits(resumed) == (4, 6)
    assert resumed.manifest.config["attempts_per_game"] == 1
    assert [event["kind"] for event in resumed.events()].count(
        "campaign.limits_revised"
    ) == 1

    args.attempts = 3
    with pytest.raises(ValueError, match="cannot lower"):
        _resume_arc_context(args, manager)


def test_primary_lane_requires_both_anthropic_backend_and_opus_model(tmp_path: Path) -> None:
    """
    A backend label alone cannot qualify a manifest using a different model.
    """

    manager = ResultsManager(tmp_path / "results")
    disclosed = manager.create_cold(
        "disclosed",
        backend="anthropic-api",
        auth_method="api-key",
        model="claude-opus-5",
    )
    mismatched = manager.create_cold(
        "mismatched",
        backend="anthropic-api",
        auth_method="api-key",
        model="gpt-5.6-sol",
    )
    assert _uses_disclosed_model_lane(disclosed)
    assert not _uses_disclosed_model_lane(mismatched)

    _validate_campaign_jobs("anthropic-api", 1)
    with pytest.raises(ValueError, match="requires --jobs 1"):
        _validate_campaign_jobs("anthropic-api", 2)
    _validate_campaign_jobs("openai-api", 2)


def test_ambiguous_budget_hold_remains_conservative_and_permanent(tmp_path: Path) -> None:
    """
    Recovery receipts retain capacity and remain contaminating after manual release.
    """

    context = ResultsManager(tmp_path / "results").create_cold(
        "ambiguous",
        backend="anthropic-api",
        auth_method="api-key",
        model="claude-opus-5",
    )
    ledger = BudgetLedger(context.directory, pricing=CLAUDE_OPUS_5_PRICING)
    reservation = ledger.reserve("12", role="player")
    active = ledger.active_reservations()
    payload = _unreconciled_reservation_payload(active)
    assert _record_unreconciled_reservations(context, ledger) == payload
    assert _record_unreconciled_reservations(context, ledger) == payload

    assert payload["reservations"] == [
        {"reservation_id": reservation, "max_unreported_usd": "12"}
    ]
    assert payload["max_unreported_usd"] == "12"
    assert ledger.snapshot().available_usd == Decimal("8.00")
    assert len(_budget_ambiguity_contamination(context)) == 2
    assert [
        event["kind"] for event in context.events()
    ].count("budget.reservations_unreconciled") == 1

    ledger.release(reservation, reason="manual forensic reconciliation")
    assert _budget_ambiguity_contamination(context) == (
        "provider usage may be unreported after an ambiguous or interrupted request",
    )


def test_setup_rejects_same_size_roster_substitution_before_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An internally consistent 25/183 endpoint substitution is not self-attested.
    """

    cache_dir = tmp_path / "cache"
    canonical = _canonical_public_roster()
    altered = [
        OfficialGameDescriptor(
            "substitute-00000000" if index == 0 else item["game_id"],
            None,
            item["levels"],
        )
        for index, item in enumerate(canonical)
    ]
    manifest = argparse.Namespace(
        environments_dir=str(cache_dir),
        game_ids=tuple(item.game_id for item in altered),
        files=(object(),),
        aggregate_sha256="a" * 64,
        package_versions=(("arc-agi", "0.9.9"), ("arcengine", "0.9.3")),
    )
    monkeypatch.setattr("ardea_avo.cli.setup_public_games", lambda _path: manifest)
    monkeypatch.setattr(
        "ardea_avo.cli.list_official_games",
        lambda _path, online_catalog=False: tuple(altered),
    )
    with pytest.raises(RuntimeError, match="frozen public benchmark roster"):
        _command_setup(argparse.Namespace(cache_dir=cache_dir))
    assert not _setup_manifest_path(cache_dir).exists()


def test_submission_claim_creation_is_exclusive(tmp_path: Path) -> None:
    """
    The durable intent marker cannot be overwritten by a racing submitter.
    """

    path = tmp_path / "competition-submission.claim.json"
    _write_exclusive_json(path, {"status": "claimed"})
    with pytest.raises(FileExistsError):
        _write_exclusive_json(path, {"status": "replacement"})
    assert "claimed" in path.read_text(encoding="utf-8")
