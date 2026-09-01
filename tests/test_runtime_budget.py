"""
Tests for durable shared budget accounting.
"""

import json
from decimal import Decimal

import pytest

from ardea_avo.runtime.budget import (
    CLAUDE_OPUS_5_PRICING,
    GPT_5_6_SOL_PRICING,
    BudgetExceeded,
    BudgetLedger,
    TokenUsage,
)


def test_sol_pricing_separates_cached_input() -> None:
    """
    Cached input must not also be charged at the uncached rate.
    """

    usage = TokenUsage(
        input_tokens=100_000,
        cached_input_tokens=25_000,
        output_tokens=10_000,
    )
    assert GPT_5_6_SOL_PRICING.cost(usage) == Decimal("0.510")


def test_sol_pricing_applies_long_context_multipliers_per_response() -> None:
    """
    Requests above 272K input use the pinned long-context input/output rates.
    """

    usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000)
    assert GPT_5_6_SOL_PRICING.cost(usage) == Decimal("11.000")


def test_claude_pricing_separates_base_cache_write_read_and_output() -> None:
    """
    Claude's normalized total input retains each differently priced category.
    """

    usage = TokenUsage(
        input_tokens=1_000_000,
        cached_input_tokens=100_000,
        cache_creation_input_tokens=200_000,
        output_tokens=100_000,
    )
    assert CLAUDE_OPUS_5_PRICING.cost(usage) == Decimal("7.300")


def test_pricing_profile_is_immutable_and_reloads_automatically(tmp_path) -> None:
    """
    Reopening a Claude ledger cannot silently fall back to GPT accounting.
    """

    BudgetLedger(tmp_path, pricing=CLAUDE_OPUS_5_PRICING)
    reopened = BudgetLedger(tmp_path)
    assert reopened.pricing == CLAUDE_OPUS_5_PRICING
    with pytest.raises(ValueError, match="pricing differs"):
        BudgetLedger(tmp_path, pricing=GPT_5_6_SOL_PRICING)


def test_ledger_reloads_usage_and_raised_cap(tmp_path) -> None:
    """
    A new ledger instance sees usage and append-only cap revisions.
    """

    ledger = BudgetLedger(tmp_path)
    cost = ledger.record_usage(
        TokenUsage(input_tokens=1_000_000, output_tokens=100_000),
        backend="codex-oauth",
        role="player",
    )
    assert cost == Decimal("11.000")
    ledger.revise_cap("25", reason="continue campaign")

    reopened = BudgetLedger(tmp_path, max_cost_usd="999")
    snapshot = reopened.snapshot()
    assert snapshot.cap_usd == Decimal("25")
    assert snapshot.spent_usd == Decimal("11.000")
    assert snapshot.available_usd == Decimal("14.000")
    assert reopened.total_usage() == TokenUsage(
        input_tokens=1_000_000, output_tokens=100_000
    )


def test_reservations_gate_parallel_turns_and_close_on_usage(tmp_path) -> None:
    """
    Reservations count against the shared cap until released or consumed.
    """

    ledger = BudgetLedger(tmp_path, max_cost_usd="1")
    reservation = ledger.reserve("0.75", role="player")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("0.26", role="supervisor")
    ledger.record_usage(
        TokenUsage(input_tokens=25_000),
        backend="codex-oauth",
        role="player",
        reservation_id=reservation,
    )
    snapshot = ledger.snapshot()
    assert snapshot.reserved_usd == 0
    assert snapshot.spent_usd == Decimal("0.1")


def test_spend_at_cap_blocks_subsequent_turn(tmp_path) -> None:
    """
    An in-flight overage remains recorded and blocks later model turns.
    """

    ledger = BudgetLedger(tmp_path, max_cost_usd="0.01")
    ledger.record_usage(
        TokenUsage(output_tokens=1_000),
        backend="openai-api",
        role="player",
    )
    with pytest.raises(BudgetExceeded):
        ledger.ensure_can_start()


def test_resume_can_explicitly_release_stranded_reservations(tmp_path) -> None:
    """
    Crash recovery is explicit and leaves an append-only release receipt.
    """

    ledger = BudgetLedger(tmp_path)
    reservation = ledger.reserve("2", role="player")
    assert ledger.active_reservations() == {reservation: Decimal("2")}
    assert ledger.release_all_active() == (reservation,)
    assert ledger.active_reservations() == {}


def test_invalid_token_relationship_is_rejected() -> None:
    """
    Cached tokens are constrained to the provider's total input count.
    """

    with pytest.raises(ValueError):
        TokenUsage(input_tokens=2, cached_input_tokens=3)
    with pytest.raises(ValueError):
        TokenUsage(
            input_tokens=2,
            cached_input_tokens=1,
            cache_creation_input_tokens=2,
        )


def test_usage_ledger_tampering_is_detected_on_reload(tmp_path) -> None:
    """
    Cost records are hash-chained rather than trusted as mutable JSONL.
    """

    ledger = BudgetLedger(tmp_path)
    ledger.record_usage(
        TokenUsage(input_tokens=10), backend="codex-oauth", role="player"
    )
    records = [json.loads(line) for line in ledger.usage_path.read_text().splitlines()]
    records[0]["cost_usd"] = "0"
    ledger.usage_path.write_text("\n".join(json.dumps(item) for item in records) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        BudgetLedger(tmp_path).snapshot()


def test_usage_metadata_refuses_credentials(tmp_path) -> None:
    """
    Flexible usage metadata cannot become an accidental auth store.
    """

    ledger = BudgetLedger(tmp_path)
    with pytest.raises(ValueError, match="sensitive field"):
        ledger.record_usage(
            TokenUsage(),
            backend="codex-oauth",
            role="player",
            metadata={"service_token": "do-not-store"},
        )


def test_sealed_parent_budget_remains_readable_but_not_mutable(tmp_path) -> None:
    """
    Warm-parent sealing prevents late model usage while preserving reporting.
    """

    ledger = BudgetLedger(tmp_path)
    (tmp_path / "sealed.json").write_text("{}", encoding="utf-8")
    assert ledger.snapshot().cap_usd == Decimal("20.00")
    with pytest.raises(RuntimeError, match="immutable"):
        ledger.record_usage(
            TokenUsage(), backend="codex-oauth", role="player"
        )
