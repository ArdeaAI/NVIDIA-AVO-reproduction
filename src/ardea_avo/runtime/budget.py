"""
Shared token-cost accounting with a durable hard start gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from ardea_avo.runtime._io import (
    append_jsonl,
    file_lock,
    load_jsonl,
    sha256_json,
    utc_now,
)

DEFAULT_MAX_COST_USD = Decimal("20.00")
_GENESIS_HASH = "0" * 64


class BudgetExceeded(RuntimeError):
    """
    Raised before a model turn that would exceed the configured ceiling.
    """


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """
    Per-million-token pricing in US dollars.
    """

    model: str
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal
    version: str
    cache_creation_input_per_million: Decimal | None = None
    long_context_threshold_tokens: int | None = None
    long_context_input_multiplier: Decimal = Decimal("1")
    long_context_output_multiplier: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        """
        Reject incomplete identities and invalid per-token rates.
        """

        if not self.model.strip() or not self.version.strip():
            raise ValueError("pricing model and version cannot be blank")
        rates = (
            self.input_per_million,
            self.cached_input_per_million,
            self.output_per_million,
        )
        if any(not rate.is_finite() or rate < 0 for rate in rates):
            raise ValueError("pricing rates must be finite and non-negative")
        cache_creation = self.cache_creation_input_per_million
        if cache_creation is not None and (
            not cache_creation.is_finite() or cache_creation < 0
        ):
            raise ValueError("cache-creation pricing must be finite and non-negative")
        threshold = self.long_context_threshold_tokens
        if threshold is not None and (
            isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1
        ):
            raise ValueError("long-context threshold must be a positive integer")
        multipliers = (
            self.long_context_input_multiplier,
            self.long_context_output_multiplier,
        )
        if any(not value.is_finite() or value < 1 for value in multipliers):
            raise ValueError("long-context multipliers must be finite and at least one")

    def cost(self, usage: TokenUsage) -> Decimal:
        """
        Calculate cost while charging cached tokens only at the cached rate.
        """

        cache_creation_rate = (
            self.input_per_million
            if self.cache_creation_input_per_million is None
            else self.cache_creation_input_per_million
        )
        uncached = (
            usage.input_tokens
            - usage.cached_input_tokens
            - usage.cache_creation_input_tokens
        )
        long_context = (
            self.long_context_threshold_tokens is not None
            and usage.input_tokens > self.long_context_threshold_tokens
        )
        input_multiplier = (
            self.long_context_input_multiplier if long_context else Decimal("1")
        )
        output_multiplier = (
            self.long_context_output_multiplier if long_context else Decimal("1")
        )
        million = Decimal(1_000_000)
        return (
            input_multiplier
            * (
                Decimal(uncached) * self.input_per_million
                + Decimal(usage.cached_input_tokens) * self.cached_input_per_million
                + Decimal(usage.cache_creation_input_tokens) * cache_creation_rate
            )
            + output_multiplier
            * Decimal(usage.output_tokens)
            * self.output_per_million
        ) / million


GPT_5_6_SOL_PRICING = ModelPricing(
    model="gpt-5.6-sol",
    input_per_million=Decimal("4.00"),
    cached_input_per_million=Decimal("0.40"),
    output_per_million=Decimal("20.00"),
    version="2026-09-01",
    cache_creation_input_per_million=Decimal("5.00"),
    long_context_threshold_tokens=272_000,
    long_context_input_multiplier=Decimal("2"),
    long_context_output_multiplier=Decimal("1.5"),
)

CLAUDE_OPUS_5_PRICING = ModelPricing(
    model="claude-opus-5",
    input_per_million=Decimal("5.00"),
    cached_input_per_million=Decimal("0.50"),
    output_per_million=Decimal("25.00"),
    version="2026-09-01",
    cache_creation_input_per_million=Decimal("6.25"),
)


def pricing_for_model(model: str) -> ModelPricing:
    """
    Return the pinned accounting schedule for a supported model.
    """

    by_model = {
        GPT_5_6_SOL_PRICING.model: GPT_5_6_SOL_PRICING,
        CLAUDE_OPUS_5_PRICING.model: CLAUDE_OPUS_5_PRICING,
    }
    try:
        return by_model[model]
    except KeyError as error:
        raise ValueError(f"no pinned pricing schedule for model: {model}") from error


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """
    Normalized token counts from either supported backend.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        """
        Validate counts and the cached-input subset invariant.
        """

        values = (
            self.input_tokens,
            self.cached_input_tokens,
            self.cache_creation_input_tokens,
            self.output_tokens,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("token counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("token counts must be non-negative integers")
        if self.cached_input_tokens + self.cache_creation_input_tokens > self.input_tokens:
            raise ValueError("cache read and creation tokens cannot exceed input tokens")

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """
        Combine usage from multiple model calls.
        """

        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=(
                self.cached_input_tokens + other.cached_input_tokens
            ),
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens
                + other.cache_creation_input_tokens
            ),
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """
    Reloaded state of a shared budget ledger.
    """

    cap_usd: Decimal
    spent_usd: Decimal
    reserved_usd: Decimal

    @property
    def available_usd(self) -> Decimal:
        """
        Return remaining unspent and unreserved capacity.
        """

        return max(Decimal("0"), self.cap_usd - self.spent_usd - self.reserved_usd)


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid decimal for {field}") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field} must be a finite non-negative amount")
    return result


def _pricing_payload(pricing: ModelPricing) -> dict[str, str | None]:
    return {
        "model": pricing.model,
        "input_per_million": str(pricing.input_per_million),
        "cached_input_per_million": str(pricing.cached_input_per_million),
        "cache_creation_input_per_million": (
            str(pricing.cache_creation_input_per_million)
            if pricing.cache_creation_input_per_million is not None
            else None
        ),
        "output_per_million": str(pricing.output_per_million),
        "version": pricing.version,
        "long_context_threshold_tokens": (
            str(pricing.long_context_threshold_tokens)
            if pricing.long_context_threshold_tokens is not None
            else None
        ),
        "long_context_input_multiplier": str(
            pricing.long_context_input_multiplier
        ),
        "long_context_output_multiplier": str(
            pricing.long_context_output_multiplier
        ),
    }


def _pricing_from_payload(value: Any) -> ModelPricing:
    if not isinstance(value, dict) or set(value) != {
        "model",
        "input_per_million",
        "cached_input_per_million",
        "cache_creation_input_per_million",
        "output_per_million",
        "version",
        "long_context_threshold_tokens",
        "long_context_input_multiplier",
        "long_context_output_multiplier",
    }:
        raise ValueError("budget ledger pricing profile is malformed")
    model = value["model"]
    version = value["version"]
    if not isinstance(model, str) or not isinstance(version, str):
        raise ValueError("budget ledger pricing identity is malformed")
    raw_cache_creation = value["cache_creation_input_per_million"]
    raw_threshold = value["long_context_threshold_tokens"]
    if raw_threshold is not None:
        try:
            threshold = int(str(raw_threshold))
        except ValueError as error:
            raise ValueError("budget ledger long-context threshold is malformed") from error
        if str(threshold) != str(raw_threshold):
            raise ValueError("budget ledger long-context threshold is malformed")
    else:
        threshold = None
    return ModelPricing(
        model=model,
        input_per_million=_decimal(value["input_per_million"], "input_per_million"),
        cached_input_per_million=_decimal(
            value["cached_input_per_million"], "cached_input_per_million"
        ),
        cache_creation_input_per_million=(
            None
            if raw_cache_creation is None
            else _decimal(
                raw_cache_creation,
                "cache_creation_input_per_million",
            )
        ),
        output_per_million=_decimal(value["output_per_million"], "output_per_million"),
        version=version,
        long_context_threshold_tokens=threshold,
        long_context_input_multiplier=_decimal(
            value["long_context_input_multiplier"],
            "long_context_input_multiplier",
        ),
        long_context_output_multiplier=_decimal(
            value["long_context_output_multiplier"],
            "long_context_output_multiplier",
        ),
    )


class BudgetLedger:
    """
    Coordinate one aggregate budget across processes and model roles.

    Usage, cap revisions, and reservations are separate append-only JSONL
    ledgers. Every decision reloads them while holding a shared lock so game
    workers cannot independently spend the same remaining allowance.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        max_cost_usd: Decimal | str | float = DEFAULT_MAX_COST_USD,
        pricing: ModelPricing | None = None,
    ) -> None:
        """
        Open a ledger or initialize its first cap revision.
        """

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.usage_path = self.directory / "usage.jsonl"
        self.revisions_path = self.directory / "config-revisions.jsonl"
        self.reservations_path = self.directory / "budget-reservations.jsonl"
        self.lock_path = self.directory / ".budget.lock"
        requested_pricing = pricing
        self.pricing = pricing or GPT_5_6_SOL_PRICING
        initial_cap = _decimal(max_cost_usd, "max_cost_usd")
        if initial_cap <= 0:
            raise ValueError("max_cost_usd must be greater than zero")
        with file_lock(self.lock_path):
            revisions = self._load_chain(self.revisions_path)
            if not revisions:
                self._ensure_mutable()
                self._append_chain(
                    self.revisions_path,
                    {
                        "kind": "initial_cap",
                        "max_cost_usd": str(initial_cap),
                        "pricing": _pricing_payload(self.pricing),
                        "timestamp": utc_now(),
                    },
                )
            else:
                self._validate_revisions(revisions)
                stored_pricing = revisions[0].get("pricing")
                if stored_pricing is not None:
                    restored = _pricing_from_payload(stored_pricing)
                    if requested_pricing is not None and requested_pricing != restored:
                        raise ValueError(
                            "requested pricing differs from the ledger's immutable profile"
                        )
                    self.pricing = restored

    @property
    def max_cost_usd(self) -> Decimal:
        """
        Reload and return the current lifetime ceiling.
        """

        return self.snapshot().cap_usd

    def snapshot(self) -> BudgetSnapshot:
        """
        Return a consistent view of cap, spend, and active reservations.
        """

        with file_lock(self.lock_path):
            return self._snapshot_unlocked()

    def total_usage(self) -> TokenUsage:
        """
        Reload and sum normalized tokens across every recorded model call.
        """

        with file_lock(self.lock_path):
            total = TokenUsage()
            for record in self._load_chain(self.usage_path):
                raw = record.get("usage")
                if not isinstance(raw, dict):
                    raise ValueError("usage ledger record lacks normalized token counts")
                try:
                    total += TokenUsage(
                        input_tokens=raw["input_tokens"],
                        cached_input_tokens=raw["cached_input_tokens"],
                        cache_creation_input_tokens=raw.get(
                            "cache_creation_input_tokens", 0
                        ),
                        output_tokens=raw["output_tokens"],
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("usage ledger contains invalid token counts") from error
            return total

    def bind_pricing(self, pricing: ModelPricing) -> None:
        """
        Select a model schedule after rejecting an existing mixed-model ledger.
        """

        with file_lock(self.lock_path):
            revisions = self._load_chain(self.revisions_path)
            stored = revisions[0].get("pricing") if revisions else None
            if stored is not None and _pricing_from_payload(stored) != pricing:
                raise ValueError(
                    "budget ledger pricing profile differs from the requested model"
                )
            models = {
                str(record.get("model", ""))
                for record in self._load_chain(self.usage_path)
            }
            if models and models != {pricing.model}:
                raise ValueError(
                    "existing usage ledger model differs from the requested pricing schedule"
                )
            self.pricing = pricing

    def can_start(self, estimated_cost_usd: Decimal | str | float = 0) -> bool:
        """
        Report whether a new turn may reserve the requested estimate.
        """

        estimate = _decimal(estimated_cost_usd, "estimated_cost_usd")
        snapshot = self.snapshot()
        return snapshot.spent_usd + snapshot.reserved_usd + estimate <= snapshot.cap_usd

    def ensure_can_start(
        self, estimated_cost_usd: Decimal | str | float = 0
    ) -> None:
        """
        Raise when launching another model turn is not budget-safe.
        """

        estimate = _decimal(estimated_cost_usd, "estimated_cost_usd")
        snapshot = self.snapshot()
        projected = snapshot.spent_usd + snapshot.reserved_usd + estimate
        if projected > snapshot.cap_usd or snapshot.spent_usd >= snapshot.cap_usd:
            raise BudgetExceeded(
                "model turn blocked by run budget: "
                f"spent=${snapshot.spent_usd}, reserved=${snapshot.reserved_usd}, "
                f"requested=${estimate}, cap=${snapshot.cap_usd}"
            )

    def reserve(
        self,
        estimated_cost_usd: Decimal | str | float,
        *,
        role: str,
    ) -> str:
        """
        Atomically reserve capacity for a concurrent model turn.
        """

        estimate = _decimal(estimated_cost_usd, "estimated_cost_usd")
        if estimate <= 0:
            raise ValueError("a reservation must be greater than zero")
        reservation_id = str(uuid4())
        with file_lock(self.lock_path):
            self._ensure_mutable()
            snapshot = self._snapshot_unlocked()
            if snapshot.spent_usd + snapshot.reserved_usd + estimate > snapshot.cap_usd:
                raise BudgetExceeded(
                    f"cannot reserve ${estimate}; only ${snapshot.available_usd} remains"
                )
            self._append_chain(
                self.reservations_path,
                {
                    "kind": "reserve",
                    "reservation_id": reservation_id,
                    "estimated_cost_usd": str(estimate),
                    "role": role,
                    "timestamp": utc_now(),
                },
            )
        return reservation_id

    def release(self, reservation_id: str, *, reason: str = "cancelled") -> None:
        """
        Release an active reservation without recording usage.
        """

        with file_lock(self.lock_path):
            self._ensure_mutable()
            active = self._active_reservations_unlocked()
            if reservation_id not in active:
                raise KeyError(f"unknown or closed reservation: {reservation_id}")
            self._append_chain(
                self.reservations_path,
                {
                    "kind": "release",
                    "reservation_id": reservation_id,
                    "reason": reason,
                    "timestamp": utc_now(),
                },
            )

    def active_reservations(self) -> dict[str, Decimal]:
        """
        Return a copied map of durable reservations currently blocking spend.
        """

        with file_lock(self.lock_path):
            return dict(self._active_reservations_unlocked())

    def release_all_active(self, *, reason: str = "resume recovery") -> tuple[str, ...]:
        """
        Release stranded reservations after proving no model calls are in flight.

        Callers should use this only during exclusive run recovery. It is not a
        substitute for normal backend consumption of a reservation.
        """

        with file_lock(self.lock_path):
            self._ensure_mutable()
            reservation_ids = tuple(sorted(self._active_reservations_unlocked()))
            for reservation_id in reservation_ids:
                self._append_chain(
                    self.reservations_path,
                    {
                        "kind": "release",
                        "reservation_id": reservation_id,
                        "reason": reason,
                        "timestamp": utc_now(),
                    },
                )
            return reservation_ids

    def record_usage(
        self,
        usage: TokenUsage,
        *,
        backend: str,
        role: str,
        session_id: str | None = None,
        reservation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Decimal:
        """
        Append normalized usage and close its reservation, if present.

        Actual cost may exceed the ceiling because an in-flight provider call
        cannot be interrupted at the exact token that consumes the estimate.
        The overage remains visible and blocks every subsequent turn.
        """

        cost = self.pricing.cost(usage)
        normalized_metadata = metadata or {}
        _reject_sensitive_metadata(normalized_metadata)
        with file_lock(self.lock_path):
            self._ensure_mutable()
            if reservation_id is not None:
                active = self._active_reservations_unlocked()
                if reservation_id not in active:
                    raise KeyError(f"unknown or closed reservation: {reservation_id}")
            self._append_chain(
                self.usage_path,
                {
                    "kind": "model_usage",
                    "backend": backend,
                    "role": role,
                    "model": self.pricing.model,
                    "pricing_version": self.pricing.version,
                    "usage": asdict(usage),
                    "cost_usd": str(cost),
                    "session_id": session_id,
                    "reservation_id": reservation_id,
                    "metadata": normalized_metadata,
                    "timestamp": utc_now(),
                },
            )
        return cost

    def revise_cap(
        self,
        new_max_cost_usd: Decimal | str | float,
        *,
        reason: str = "resume override",
    ) -> None:
        """
        Append a strictly higher lifetime ceiling revision.
        """

        new_cap = _decimal(new_max_cost_usd, "new_max_cost_usd")
        with file_lock(self.lock_path):
            self._ensure_mutable()
            snapshot = self._snapshot_unlocked()
            if new_cap <= snapshot.cap_usd:
                raise ValueError("budget revisions may only raise the lifetime ceiling")
            self._append_chain(
                self.revisions_path,
                {
                    "kind": "cap_revision",
                    "previous_max_cost_usd": str(snapshot.cap_usd),
                    "max_cost_usd": str(new_cap),
                    "reason": reason,
                    "timestamp": utc_now(),
                },
            )

    def _snapshot_unlocked(self) -> BudgetSnapshot:
        revisions = self._load_chain(self.revisions_path)
        self._validate_revisions(revisions)
        cap = _decimal(revisions[-1]["max_cost_usd"], "max_cost_usd")
        spent = Decimal("0")
        for record in self._load_chain(self.usage_path):
            if record.get("kind") != "model_usage":
                raise ValueError("unknown usage ledger record")
            spent += _decimal(record.get("cost_usd"), "cost_usd")
        reserved = sum(self._active_reservations_unlocked().values(), Decimal("0"))
        return BudgetSnapshot(cap_usd=cap, spent_usd=spent, reserved_usd=reserved)

    def _ensure_mutable(self) -> None:
        candidates = (self.directory, *self.directory.parents)
        if any(
            (candidate / "sealed.json").exists()
            and (
                candidate == self.directory
                or (candidate / "manifest.json").is_file()
            )
            for candidate in candidates
        ):
            raise RuntimeError("sealed parent run budgets are immutable")

    def _active_reservations_unlocked(self) -> dict[str, Decimal]:
        active: dict[str, Decimal] = {}
        consumed_ids = [
            str(record["reservation_id"])
            for record in self._load_chain(self.usage_path)
            if record.get("reservation_id")
        ]
        if len(consumed_ids) != len(set(consumed_ids)):
            raise ValueError("a reservation was consumed more than once")
        consumed = set(consumed_ids)
        for record in self._load_chain(self.reservations_path):
            reservation_id = str(record.get("reservation_id", ""))
            if not reservation_id:
                raise ValueError("reservation record lacks an id")
            kind = record.get("kind")
            if kind == "reserve":
                if reservation_id in active:
                    raise ValueError(f"duplicate reservation id: {reservation_id}")
                active[reservation_id] = _decimal(
                    record.get("estimated_cost_usd"), "estimated_cost_usd"
                )
            elif kind == "release":
                if reservation_id not in active:
                    raise ValueError(f"release of inactive reservation: {reservation_id}")
                active.pop(reservation_id)
            else:
                raise ValueError("unknown reservation ledger record")
        for reservation_id in consumed:
            active.pop(reservation_id, None)
        return active

    @staticmethod
    def _load_chain(path: Path) -> list[dict[str, Any]]:
        records = load_jsonl(path)
        previous_hash = _GENESIS_HASH
        for sequence, record in enumerate(records):
            if record.get("sequence") != sequence:
                raise ValueError(f"ledger sequence mismatch at {path}:{sequence + 1}")
            if record.get("previous_hash") != previous_hash:
                raise ValueError(f"ledger predecessor mismatch at {path}:{sequence + 1}")
            digest = record.get("hash")
            body = {key: value for key, value in record.items() if key != "hash"}
            if digest != sha256_json(body):
                raise ValueError(f"ledger hash mismatch at {path}:{sequence + 1}")
            previous_hash = str(digest)
        return records

    @classmethod
    def _append_chain(cls, path: Path, value: dict[str, Any]) -> None:
        records = cls._load_chain(path)
        body = {
            **value,
            "sequence": len(records),
            "previous_hash": records[-1]["hash"] if records else _GENESIS_HASH,
        }
        append_jsonl(path, {**body, "hash": sha256_json(body)})

    @staticmethod
    def _validate_revisions(records: list[dict[str, Any]]) -> None:
        if not records or records[0].get("kind") != "initial_cap":
            raise ValueError("budget ledger lacks an initial cap")
        previous = Decimal("0")
        for index, record in enumerate(records):
            expected_kind = "initial_cap" if index == 0 else "cap_revision"
            if record.get("kind") != expected_kind:
                raise ValueError("invalid budget revision sequence")
            cap = _decimal(record.get("max_cost_usd"), "max_cost_usd")
            if cap <= previous:
                raise ValueError("budget cap revisions must be strictly increasing")
            previous = cap


def _reject_sensitive_metadata(value: Any, *, path: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if (
                normalized
                in {
                    "api_key",
                    "access_token",
                    "refresh_token",
                    "authorization",
                    "password",
                    "secret",
                    "credentials",
                }
                or normalized.endswith("_api_key")
                or normalized.endswith("_token")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
            ):
                raise ValueError(f"refusing to persist sensitive field: {path}.{key}")
            _reject_sensitive_metadata(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive_metadata(nested, path=f"{path}[{index}]")
