"""
Offline construction and durable storage of auditable run reports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from ardea_avo.arc.campaign import CampaignBank
from ardea_avo.arc.scoring import board_rhae
from ardea_avo.arc.trace import load_trace, trace_sha256
from ardea_avo.runtime._io import atomic_write_json, canonical_json, utc_now
from ardea_avo.runtime.budget import BudgetLedger
from ardea_avo.runtime.results import RunContext

REPORT_SCHEMA = "ardea.avo.run-report.v2"


class ExpectedGameLike(Protocol):
    """
    Structural public catalog item accepted by report construction.
    """

    game_id: str
    levels: int


class ReportStatus(StrEnum):
    """
    Highest evidence-backed state reached by a run.
    """

    IN_PROGRESS = "in-progress"
    LOCALLY_COMPLETE = "locally-complete"
    LOCALLY_VALIDATED = "locally-validated"
    DRY_RUN_SUBMITTED = "dry-run-submitted"
    COMPETITION_SUBMITTED = "competition-submitted"
    CONTAMINATED = "contaminated"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ExpectedGame:
    """
    Public game identity and its expected level count.
    """

    game_id: str
    levels: int

    def __post_init__(self) -> None:
        """
        Validate non-secret public catalog metadata.
        """

        if not self.game_id.strip():
            raise ValueError("expected game id cannot be blank")
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or self.levels <= 0:
            raise ValueError("expected game levels must be a positive integer")


@dataclass(frozen=True, slots=True)
class GameReport:
    """
    One fixed board slot, including a zero for an unsolved game.
    """

    game_id: str
    expected_levels: int
    solved_levels: int
    solved: bool
    rhae_percent: float
    submitted_actions: int
    trace_sha256: str | None


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """
    Aggregate model tokens, estimated spend, and remaining run capacity.
    """

    input_tokens: int
    cached_input_tokens: int
    cache_creation_input_tokens: int
    output_tokens: int
    estimated_cost_usd: str
    cap_usd: str
    reserved_usd: str
    available_usd: str
    pricing_model: str
    pricing_version: str


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """
    Distinguish storage integrity from fresh-engine replay validation.
    """

    run_integrity_valid: bool
    bank_integrity_valid: bool
    fresh_replay_validated: bool
    errors: tuple[str, ...] = ()
    validated_at: str | None = None


@dataclass(frozen=True, slots=True)
class SubmissionSummary:
    """
    Sanitized evidence from an optional official scorecard submission.
    """

    mode: str = "none"
    completed: bool = False
    scorecard_id: str | None = None
    submitted_at: str | None = None
    official_rhae_percent: float | None = None
    official_games_solved: int | None = None
    official_levels_solved: int | None = None
    official_submitted_actions: int | None = None
    official_response_sha256: str | None = None
    acceptance_met: bool = False

    def __post_init__(self) -> None:
        """
        Reject partial or internally inconsistent submission claims.
        """

        if self.mode not in {"none", "dry-run", "competition"}:
            raise ValueError("submission mode must be none, dry-run, or competition")
        if self.completed and self.mode == "none":
            raise ValueError("a completed submission requires an online mode")
        if self.completed and not self.scorecard_id:
            raise ValueError("a completed submission requires a scorecard id")
        if self.completed and not self.submitted_at:
            raise ValueError("a completed submission requires a timestamp")
        if not self.completed and self.scorecard_id is not None:
            raise ValueError("an incomplete submission cannot claim a scorecard id")
        if self.acceptance_met and (not self.completed or self.mode != "competition"):
            raise ValueError("acceptance can be met only by a completed Competition scorecard")
        _optional_percent(self.official_rhae_percent, "official_rhae_percent")
        for name, value in (
            ("official_games_solved", self.official_games_solved),
            ("official_levels_solved", self.official_levels_solved),
            ("official_submitted_actions", self.official_submitted_actions),
        ):
            _optional_count(value, name)
        if self.official_response_sha256 is not None and (
            len(self.official_response_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.official_response_sha256
            )
        ):
            raise ValueError("official response digest must be SHA-256")

    @classmethod
    def from_scorecard(
        cls,
        scorecard: Any,
        *,
        official_rhae_percent: float | None = None,
        official_games_solved: int | None = None,
        official_levels_solved: int | None = None,
        submitted_at: str | None = None,
    ) -> SubmissionSummary:
        """
        Summarize a ``ScorecardReport`` without storing its raw response.
        """

        mode = str(getattr(getattr(scorecard, "mode", None), "value", "none"))
        replays = tuple(getattr(scorecard, "replays", ()))
        response = getattr(scorecard, "official_response", None)
        response_payload = _scorecard_response_payload(response)
        response_digest = sha256(
            canonical_json(response_payload).encode("utf-8")
        ).hexdigest()
        return cls(
            mode=mode,
            completed=True,
            scorecard_id=str(scorecard.scorecard_id),
            submitted_at=submitted_at or utc_now(),
            official_rhae_percent=(
                official_rhae_percent
                if official_rhae_percent is not None
                else _optional_response_number(response, "score")
            ),
            official_games_solved=(
                official_games_solved
                if official_games_solved is not None
                else _response_count(
                    response, "total_environments_completed", default=len(replays)
                )
            ),
            official_levels_solved=(
                official_levels_solved
                if official_levels_solved is not None
                else _response_count(
                    response,
                    "total_levels_completed",
                    default=sum(int(item.levels_completed) for item in replays),
                )
            ),
            official_submitted_actions=_response_count(
                response,
                "total_actions",
                default=sum(int(item.actions) for item in replays),
            ),
            official_response_sha256=response_digest,
        )


@dataclass(frozen=True, slots=True)
class RunReport:
    """
    Complete offline summary of one cold or warm ARC campaign.
    """

    schema: str
    generated_at: str
    run_id: str
    mode: str
    parent_run_id: str | None
    backend: str
    auth_method: str
    model: str
    reasoning_effort: str
    observation_mode: str
    status: ReportStatus
    expected_games: int
    solved_games: int
    expected_levels: int
    solved_levels: int
    rhae_percent: float
    submitted_actions: int
    exploratory_actions: int
    total_environment_actions: int
    games: tuple[GameReport, ...]
    usage: UsageSummary
    contamination: tuple[str, ...]
    validation: ValidationSummary
    submission: SubmissionSummary

    def to_dict(self) -> dict[str, Any]:
        """
        Return a canonical JSON-compatible report object.
        """

        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunReport:
        """
        Parse and validate the exact current report schema.
        """

        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("run report fields do not match the current schema")
        data = dict(value)
        try:
            data["status"] = ReportStatus(data["status"])
            data["games"] = tuple(GameReport(**item) for item in data["games"])
            data["usage"] = UsageSummary(**data["usage"])
            validation = dict(data["validation"])
            validation["errors"] = tuple(validation.get("errors", ()))
            data["validation"] = ValidationSummary(**validation)
            data["submission"] = SubmissionSummary(**data["submission"])
            data["contamination"] = tuple(data["contamination"])
            report = cls(**data)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("run report contains invalid nested fields") from error
        report.validate()
        return report

    @classmethod
    def read(cls, path: str | Path) -> RunReport:
        """
        Read a complete report from disk and fail closed on malformed JSON.
        """

        report_path = Path(path)
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"run report is unreadable: {report_path}") from error
        if not isinstance(value, dict):
            raise ValueError("run report must be a JSON object")
        return cls.from_dict(value)

    def write(self, path: str | Path) -> Path:
        """
        Atomically replace a report file with this validated snapshot.
        """

        self.validate()
        report_path = Path(path)
        atomic_write_json(report_path, self.to_dict())
        return report_path

    def validate(self) -> None:
        """
        Validate totals, the complete board, and acceptance semantics.
        """

        if self.schema != REPORT_SCHEMA:
            raise ValueError("unsupported run report schema")
        identity = (
            self.generated_at,
            self.run_id,
            self.backend,
            self.auth_method,
            self.model,
            self.reasoning_effort,
            self.observation_mode,
        )
        if not all(isinstance(item, str) and item.strip() for item in identity):
            raise ValueError("run report identity fields cannot be blank")
        if self.mode not in {"cold", "warm"}:
            raise ValueError("run report mode must be cold or warm")
        if self.mode == "cold" and self.parent_run_id is not None:
            raise ValueError("cold run reports cannot have a parent")
        if self.mode == "warm" and not self.parent_run_id:
            raise ValueError("warm run reports require a parent")
        counts = (
            self.expected_games,
            self.solved_games,
            self.expected_levels,
            self.solved_levels,
            self.submitted_actions,
            self.exploratory_actions,
            self.total_environment_actions,
        )
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
            raise ValueError("run report counts must be non-negative integers")
        if self.expected_games < 1 or self.expected_levels < 1:
            raise ValueError("run report requires at least one expected game and level")
        if self.solved_games > self.expected_games or self.solved_levels > self.expected_levels:
            raise ValueError("solved totals cannot exceed expected totals")
        if self.expected_games != len(self.games):
            raise ValueError("game board length differs from expected game count")
        if len({game.game_id for game in self.games}) != len(self.games):
            raise ValueError("game board contains duplicate ids")
        if tuple(game.game_id for game in self.games) != tuple(
            sorted(game.game_id for game in self.games)
        ):
            raise ValueError("game board must use deterministic id order")
        _percent(self.rhae_percent, "rhae_percent")
        for game in self.games:
            _validate_game(game)
        if self.expected_levels != sum(game.expected_levels for game in self.games):
            raise ValueError("expected level total differs from the game board")
        if self.solved_games != sum(game.solved for game in self.games):
            raise ValueError("solved game total differs from the game board")
        if self.solved_levels != sum(game.solved_levels for game in self.games):
            raise ValueError("solved level total differs from the game board")
        if self.submitted_actions != sum(game.submitted_actions for game in self.games):
            raise ValueError("submitted action total differs from selected game traces")
        if self.total_environment_actions != self.submitted_actions + self.exploratory_actions:
            raise ValueError("environment action totals are inconsistent")
        _validate_usage(self.usage)
        if any(
            not isinstance(item, str) or not item.strip() for item in self.contamination
        ) or len(set(self.contamination)) != len(self.contamination):
            raise ValueError("contamination findings must be unique non-empty strings")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.validation.errors
        ):
            raise ValueError("validation errors must be non-empty strings")
        validation_flags = (
            self.validation.run_integrity_valid,
            self.validation.bank_integrity_valid,
            self.validation.fresh_replay_validated,
        )
        if any(not isinstance(item, bool) for item in validation_flags):
            raise ValueError("validation flags must be booleans")
        if self.validation.fresh_replay_validated and not self.validation.validated_at:
            raise ValueError("fresh replay validation requires a timestamp")
        expected_board = board_rhae(
            tuple(game.rhae_percent / 100.0 for game in self.games)
        )
        if not math.isclose(self.rhae_percent, expected_board, abs_tol=1e-9):
            raise ValueError("board RHAE differs from its per-game slots")
        if self.submission.acceptance_met and (
            self.submission.official_rhae_percent != 100.0
            or self.submission.official_games_solved != self.expected_games
            or self.submission.official_levels_solved != self.expected_levels
        ):
            raise ValueError("submission acceptance fields do not satisfy the benchmark gate")
        complete = (
            self.solved_games == self.expected_games
            and self.solved_levels == self.expected_levels
        )
        if self.status is not _status(
            complete=complete,
            contamination=self.contamination,
            validation=self.validation,
            submission=self.submission,
        ):
            raise ValueError("run report status differs from its evidence")


def build_run_report(
    run: RunContext,
    bank: CampaignBank,
    budget: BudgetLedger,
    *,
    expected_games: Mapping[str, int] | Sequence[ExpectedGameLike],
    trace_paths: Iterable[str | Path] = (),
    contamination: Iterable[str] = (),
    fresh_replay_validated: bool = False,
    validation_errors: Iterable[str] = (),
    submission: SubmissionSummary | None = None,
    generated_at: str | None = None,
) -> RunReport:
    """
    Build an offline report from validated run, bank, budget, and trace state.

    Missing expected games occupy zero-valued RHAE board slots. Submitted
    actions come only from selected winning traces; exploratory actions come
    from every distinct supplied trace whose bytes are not a selected win.
    """

    run.validate()
    catalog = _expected_catalog(expected_games)
    selected = {entry.game_id: entry for entry in bank.entries()}
    unexpected = sorted(set(selected) - set(catalog))
    if unexpected:
        raise ValueError(f"campaign bank contains unexpected games: {', '.join(unexpected)}")

    games: list[GameReport] = []
    selected_digests: set[str] = set()
    for game_id, expected_levels in catalog.items():
        entry = selected.get(game_id)
        if entry is None:
            games.append(
                GameReport(
                    game_id=game_id,
                    expected_levels=expected_levels,
                    solved_levels=0,
                    solved=False,
                    rhae_percent=0.0,
                    submitted_actions=0,
                    trace_sha256=None,
                )
            )
            continue
        verified = bank.validate_selected(game_id)
        if verified.win_levels != expected_levels:
            raise ValueError(f"banked game {game_id!r} has unexpected level metadata")
        _percent(verified.rhae_percent, f"{game_id} rhae_percent")
        selected_digests.add(verified.trace_sha256)
        games.append(
            GameReport(
                game_id=game_id,
                expected_levels=expected_levels,
                solved_levels=verified.levels_completed,
                solved=verified.levels_completed == expected_levels,
                rhae_percent=verified.rhae_percent,
                submitted_actions=verified.actions,
                trace_sha256=verified.trace_sha256,
            )
        )

    exploratory_actions = 0
    seen_paths: set[Path] = set()
    for raw_path in trace_paths:
        path = Path(raw_path).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        data = load_trace(path)
        if data.header.game_id not in catalog:
            raise ValueError(f"trace contains unexpected game {data.header.game_id!r}")
        if trace_sha256(path) not in selected_digests:
            exploratory_actions += len(data.steps)

    game_tuple = tuple(games)
    solved_games = sum(game.solved for game in game_tuple)
    solved_levels = sum(game.solved_levels for game in game_tuple)
    submitted_actions = sum(game.submitted_actions for game in game_tuple)
    local_rhae = board_rhae(
        tuple(game.rhae_percent / 100.0 for game in game_tuple)
    )
    usage = budget.total_usage()
    budget_snapshot = budget.snapshot()
    usage_summary = UsageSummary(
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=str(budget_snapshot.spent_usd),
        cap_usd=str(budget_snapshot.cap_usd),
        reserved_usd=str(budget_snapshot.reserved_usd),
        available_usd=str(budget_snapshot.available_usd),
        pricing_model=budget.pricing.model,
        pricing_version=budget.pricing.version,
    )
    errors = tuple(dict.fromkeys(item.strip() for item in validation_errors if item.strip()))
    validation = ValidationSummary(
        run_integrity_valid=True,
        bank_integrity_valid=True,
        fresh_replay_validated=fresh_replay_validated,
        errors=errors,
        validated_at=(generated_at or utc_now()) if fresh_replay_validated else None,
    )
    contamination_tuple = tuple(
        dict.fromkeys(item.strip() for item in contamination if item.strip())
    )
    normalized_submission = submission or SubmissionSummary()
    acceptance_met = bool(
        normalized_submission.completed
        and normalized_submission.mode == "competition"
        and normalized_submission.official_rhae_percent == 100.0
        and normalized_submission.official_games_solved == len(catalog)
        and normalized_submission.official_levels_solved == sum(catalog.values())
    )
    normalized_submission = replace(
        normalized_submission,
        acceptance_met=acceptance_met,
    )
    status = _status(
        complete=solved_games == len(catalog) and solved_levels == sum(catalog.values()),
        contamination=contamination_tuple,
        validation=validation,
        submission=normalized_submission,
    )
    report = RunReport(
        schema=REPORT_SCHEMA,
        generated_at=generated_at or utc_now(),
        run_id=run.manifest.run_id,
        mode=run.manifest.mode.value,
        parent_run_id=run.manifest.parent_run_id,
        backend=run.manifest.backend,
        auth_method=run.manifest.auth_method,
        model=run.manifest.model,
        reasoning_effort=run.manifest.reasoning_effort,
        observation_mode=run.manifest.observation_mode,
        status=status,
        expected_games=len(catalog),
        solved_games=solved_games,
        expected_levels=sum(catalog.values()),
        solved_levels=solved_levels,
        rhae_percent=local_rhae,
        submitted_actions=submitted_actions,
        exploratory_actions=exploratory_actions,
        total_environment_actions=submitted_actions + exploratory_actions,
        games=game_tuple,
        usage=usage_summary,
        contamination=contamination_tuple,
        validation=validation,
        submission=normalized_submission,
    )
    report.validate()
    return report


def write_report(path: str | Path, report: RunReport) -> Path:
    """
    Write a report atomically for CLI and library callers.
    """

    return report.write(path)


def read_report(path: str | Path) -> RunReport:
    """
    Read and validate a stored report.
    """

    return RunReport.read(path)


def _expected_catalog(
    expected: Mapping[str, int] | Sequence[ExpectedGameLike],
) -> dict[str, int]:
    if isinstance(expected, Mapping):
        items = tuple(ExpectedGame(str(game_id), levels) for game_id, levels in expected.items())
    else:
        items = tuple(ExpectedGame(item.game_id, item.levels) for item in expected)
    if not items:
        raise ValueError("at least one expected game is required")
    if len({item.game_id for item in items}) != len(items):
        raise ValueError("expected game catalog contains duplicates")
    return {item.game_id: item.levels for item in sorted(items, key=lambda item: item.game_id)}


def _status(
    *,
    complete: bool,
    contamination: tuple[str, ...],
    validation: ValidationSummary,
    submission: SubmissionSummary,
) -> ReportStatus:
    if validation.errors or not validation.run_integrity_valid or not validation.bank_integrity_valid:
        return ReportStatus.INVALID
    if contamination:
        return ReportStatus.CONTAMINATED
    if submission.completed and submission.mode == "competition":
        return ReportStatus.COMPETITION_SUBMITTED
    if submission.completed and submission.mode == "dry-run":
        return ReportStatus.DRY_RUN_SUBMITTED
    if complete and validation.fresh_replay_validated:
        return ReportStatus.LOCALLY_VALIDATED
    if complete:
        return ReportStatus.LOCALLY_COMPLETE
    return ReportStatus.IN_PROGRESS


def _percent(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0:
        raise ValueError(f"{name} must be between zero and 100")


def _optional_percent(value: float | None, name: str) -> None:
    if value is not None:
        _percent(value, name)


def _optional_count(value: int | None, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or null")


def _validate_game(game: GameReport) -> None:
    if not game.game_id.strip():
        raise ValueError("game report id cannot be blank")
    for name, value in (
        ("expected_levels", game.expected_levels),
        ("solved_levels", game.solved_levels),
        ("submitted_actions", game.submitted_actions),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"game {name} must be a non-negative integer")
    if game.expected_levels < 1 or game.solved_levels > game.expected_levels:
        raise ValueError("game level totals are inconsistent")
    if game.solved != (game.solved_levels == game.expected_levels):
        raise ValueError("game solved flag differs from level totals")
    _percent(game.rhae_percent, "game rhae_percent")
    if not game.solved and game.rhae_percent != 0.0 and game.trace_sha256 is None:
        raise ValueError("missing games must occupy a zero-valued board slot")
    if game.trace_sha256 is None:
        if game.solved_levels != 0 or game.submitted_actions != 0 or game.solved:
            raise ValueError("missing games cannot have solved levels or submitted actions")
    elif len(game.trace_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in game.trace_sha256
    ):
        raise ValueError("game trace digest must be SHA-256")
    elif not game.solved or game.submitted_actions < 1:
        raise ValueError("selected traces must be complete wins with counted actions")


def _validate_usage(usage: UsageSummary) -> None:
    counts = (
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.cache_creation_input_tokens,
        usage.output_tokens,
    )
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
        raise ValueError("usage token counts must be non-negative integers")
    if usage.cached_input_tokens + usage.cache_creation_input_tokens > usage.input_tokens:
        raise ValueError("cached and cache-creation input cannot exceed total input")
    amounts: list[Decimal] = []
    for name, value in (
        ("estimated_cost_usd", usage.estimated_cost_usd),
        ("cap_usd", usage.cap_usd),
        ("reserved_usd", usage.reserved_usd),
        ("available_usd", usage.available_usd),
    ):
        try:
            amount = Decimal(value)
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError(f"usage {name} is not a decimal amount") from error
        if not amount.is_finite() or amount < 0:
            raise ValueError(f"usage {name} must be finite and non-negative")
        amounts.append(amount)
    spent, cap, reserved, available = amounts
    if available != max(Decimal("0"), cap - spent - reserved):
        raise ValueError("usage available budget is inconsistent")
    if not usage.pricing_model.strip() or not usage.pricing_version.strip():
        raise ValueError("usage pricing identity cannot be blank")


def _scorecard_response_payload(response: Any) -> Any:
    if response is None:
        return None
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            response = model_dump(mode="json", exclude={"api_key"})
        except TypeError:
            response = model_dump()
    return _without_credentials(response)


def _without_credentials(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if (
                normalized in {"api_key", "authorization", "password", "secret"}
                or normalized.endswith("_token")
                or normalized.endswith("_api_key")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
            ):
                continue
            cleaned[str(key)] = _without_credentials(nested)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_without_credentials(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("official scorecard response is not JSON-compatible")


def _response_field(response: Any, name: str) -> Any:
    if isinstance(response, Mapping):
        return response.get(name)
    return getattr(response, name, None)


def _response_count(response: Any, name: str, *, default: int) -> int:
    value = _response_field(response, name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"official scorecard {name} is invalid")
    return value


def _optional_response_number(response: Any, name: str) -> float | None:
    value = _response_field(response, name)
    if value is None:
        return None
    _percent(value, f"official scorecard {name}")
    return float(value)
