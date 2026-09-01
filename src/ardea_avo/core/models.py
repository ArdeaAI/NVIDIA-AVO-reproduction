"""
Immutable contracts shared by the AVO engine and its adapters.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from ardea_avo.core.exceptions import ScoreValidationError

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> datetime:
    """
    Return an aware UTC timestamp.
    """
    return datetime.now(UTC)


def _validate_name(value: str, *, label: str) -> str:
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter and contain only letters, digits, '.', '_', or '-'"
        )
    return value


def _validate_json_object(value: dict[str, Any], *, label: str) -> dict[str, Any]:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON-compatible values") from exc
    return value


class CoreModel(BaseModel):
    """
    Base model with strict, immutable, forward-compatible validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ObjectiveDirection(StrEnum):
    """
    Optimization direction for one ordered score metric.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ScoreComparison(StrEnum):
    """
    Lexicographic relationship between a proposal and the incumbent.
    """

    IMPROVED = "improved"
    EQUAL = "equal"
    REGRESSED = "regressed"


class StepDecision(StrEnum):
    """
    Durable disposition of one variation attempt.
    """

    ACCEPTED = "accepted"
    REJECTED_INCORRECT = "rejected_incorrect"
    REJECTED_REGRESSION = "rejected_regression"
    AGENT_FAILED = "agent_failed"
    EVALUATION_FAILED = "evaluation_failed"


class MetricObjective(CoreModel):
    """
    One metric in the target's ordered lexicographic objective.
    """

    name: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """
        Validate a metric name used in evaluator JSON.
        """
        return _validate_name(value, label="metric name")


class Score(CoreModel):
    """
    A correctness gate plus the evaluator's complete finite metric roster.
    """

    correct: StrictBool
    metrics: dict[str, float]

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_finite_metrics(cls, value: Any) -> dict[str, float]:
        """
        Reject non-numeric, Boolean, non-finite, or malformed metric values.
        """
        if not isinstance(value, dict) or not value:
            raise ValueError("metrics must be a non-empty object")

        normalized: dict[str, float] = {}
        for raw_name, raw_value in value.items():
            if not isinstance(raw_name, str):
                raise ValueError("metric names must be strings")
            name = _validate_name(raw_name, label="metric name")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"metric {name!r} must be a number")
            numeric = float(raw_value)
            if not math.isfinite(numeric):
                raise ValueError(f"metric {name!r} must be finite")
            normalized[name] = numeric
        return normalized


class TargetSpec(CoreModel):
    """
    Domain-neutral evaluation and comparison contract for one search target.
    """

    name: str
    objectives: tuple[MetricObjective, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """
        Validate the stable target identifier.
        """
        return _validate_name(value, label="target name")

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure target metadata can be persisted in a checkpoint.
        """
        return _validate_json_object(value, label="target metadata")

    @model_validator(mode="after")
    def validate_objectives(self) -> Self:
        """
        Require a non-empty objective with unique metric names.
        """
        if not self.objectives:
            raise ValueError("a target must define at least one objective")
        names = [objective.name for objective in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError("objective metric names must be unique")
        return self

    @property
    def metric_names(self) -> tuple[str, ...]:
        """
        Return the exact evaluator metric roster in comparison order.
        """
        return tuple(objective.name for objective in self.objectives)

    def validate_score(self, score: Score) -> None:
        """
        Enforce exact metric names in addition to Score's finite-value checks.
        """
        expected = set(self.metric_names)
        actual = set(score.metrics)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ScoreValidationError("metric roster mismatch: " + ", ".join(details))

    def compare(self, proposal: Score, incumbent: Score) -> ScoreComparison:
        """
        Compare two valid scores lexicographically; exact ties are equal.
        """
        self.validate_score(proposal)
        self.validate_score(incumbent)
        for objective in self.objectives:
            proposed_value = proposal.metrics[objective.name]
            incumbent_value = incumbent.metrics[objective.name]
            if proposed_value == incumbent_value:
                continue
            improves = proposed_value > incumbent_value
            if objective.direction is ObjectiveDirection.MINIMIZE:
                improves = not improves
            return ScoreComparison.IMPROVED if improves else ScoreComparison.REGRESSED
        return ScoreComparison.EQUAL


class Candidate(CoreModel):
    """
    Identity and immutable content digest of one candidate tree.
    """

    candidate_id: str
    parent_id: str | None
    generation: int = Field(ge=0)
    artifact_digest: str
    created_at: datetime = Field(default_factory=utc_now)
    commit_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        """
        Validate an identifier safe for records and archive filenames.
        """
        return _validate_name(value, label="candidate id")

    @field_validator("artifact_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        """
        Require a lowercase SHA-256 digest.
        """
        if not _DIGEST_PATTERN.fullmatch(value):
            raise ValueError("artifact_digest must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("commit_hash")
    @classmethod
    def validate_commit_hash(cls, value: str | None) -> str | None:
        """
        Validate a full Git SHA-1 or SHA-256 object identifier when present.
        """
        if value is not None and not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
            raise ValueError("commit_hash must be a full lowercase Git object identifier")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure candidate metadata can be persisted in a checkpoint.
        """
        return _validate_json_object(value, label="candidate metadata")


class Evaluation(CoreModel):
    """
    Evaluator result bound to one candidate identity and artifact digest.
    """

    candidate_id: str
    artifact_digest: str
    score: Score
    evaluator: str
    evidence: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=utc_now)

    @field_validator("candidate_id", "evaluator")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        """
        Validate evaluator-facing stable identifiers.
        """
        return _validate_name(value, label=str(info.field_name).replace("_", " "))

    @field_validator("artifact_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        """
        Require a lowercase SHA-256 artifact digest.
        """
        if not _DIGEST_PATTERN.fullmatch(value):
            raise ValueError("artifact_digest must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        """
        Ensure each evidence item can be persisted without lossy conversion.
        """
        for index, item in enumerate(value):
            _validate_json_object(item, label=f"evidence item {index}")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure evaluation metadata is checkpoint-safe.
        """
        return _validate_json_object(value, label="evaluation metadata")


class VariationRequest(CoreModel):
    """
    Complete context supplied to an agent for one candidate variation.
    """

    candidate: Candidate
    target: TargetSpec
    prior_evaluation: Evaluation
    knowledge: tuple[dict[str, Any], ...] = ()
    attempt: int = Field(ge=1)

    @field_validator("knowledge")
    @classmethod
    def validate_knowledge(cls, value: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        """
        Keep supplied knowledge serializable for audit and replay.
        """
        for index, item in enumerate(value):
            _validate_json_object(item, label=f"knowledge item {index}")
        return value


class VariationResult(CoreModel):
    """
    Agent-supplied description of a completed candidate mutation.
    """

    summary: str = Field(min_length=1, max_length=10_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure variation metadata can be included in durable records.
        """
        return _validate_json_object(value, label="variation metadata")


class RejectionArchive(CoreModel):
    """
    Paths and digest for an archived rejected worktree patch.
    """

    record_path: str
    patch_path: str
    patch_sha256: str


class StepRecord(CoreModel):
    """
    Checkpoint-friendly record of one completed variation attempt.
    """

    attempt: int = Field(ge=1)
    candidate: Candidate
    evaluation: Evaluation | None
    variation: VariationResult | None
    decision: StepDecision
    comparison: ScoreComparison | None
    accepted_commit: str | None = None
    rejection: RejectionArchive | None = None
    error: str | None = Field(default=None, max_length=10_000)
    completed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        """
        Require commit metadata for acceptance and archive metadata for rejection.
        """
        if self.decision is StepDecision.ACCEPTED:
            if (
                self.accepted_commit is None
                or self.rejection is not None
                or self.evaluation is None
                or self.variation is None
                or self.error is not None
            ):
                raise ValueError(
                    "accepted records require evaluation, variation, and commit without rejection or error"
                )
        elif self.accepted_commit is not None or self.rejection is None:
            raise ValueError("rejected records require an archive and no accepted commit")

        if self.decision in {StepDecision.AGENT_FAILED, StepDecision.EVALUATION_FAILED}:
            if not self.error:
                raise ValueError("failed-attempt records require an error")
            if self.evaluation is not None or self.comparison is not None:
                raise ValueError("failed-attempt records cannot contain an accepted evaluation or comparison")
        elif self.error is not None or self.evaluation is None or self.variation is None:
            raise ValueError("evaluated records require evaluation and variation without an error")

        if self.evaluation is not None:
            if self.evaluation.candidate_id != self.candidate.candidate_id:
                raise ValueError("step candidate and evaluation identities differ")
            if self.evaluation.artifact_digest != self.candidate.artifact_digest:
                raise ValueError("step candidate and evaluation digests differ")

        if self.decision is StepDecision.ACCEPTED:
            if not self.evaluation or not self.evaluation.score.correct:
                raise ValueError("accepted records require a correctness-passing evaluation")
            if self.comparison not in {ScoreComparison.IMPROVED, ScoreComparison.EQUAL}:
                raise ValueError("accepted records require an improved or equal comparison")
            if self.candidate.commit_hash != self.accepted_commit:
                raise ValueError("accepted record candidate commit does not match accepted_commit")
        elif self.decision is StepDecision.REJECTED_INCORRECT:
            if not self.evaluation or self.evaluation.score.correct or self.comparison is not None:
                raise ValueError("incorrect rejection requires a failing evaluation and no comparison")
        elif self.decision is StepDecision.REJECTED_REGRESSION:
            if (
                not self.evaluation
                or not self.evaluation.score.correct
                or self.comparison is not ScoreComparison.REGRESSED
            ):
                raise ValueError("regression rejection requires a correct regressing evaluation")
        return self


class EngineState(CoreModel):
    """
    Complete serializable state required to resume an evolution lineage.
    """

    schema_version: int = 1
    target_name: str
    accepted_candidate: Candidate
    accepted_evaluation: Evaluation
    accepted_commit: str
    attempts: int = Field(default=0, ge=0)
    records: tuple[StepRecord, ...] = ()

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """
        Check identity and history invariants before a state can be restored.
        """
        if self.accepted_candidate.commit_hash != self.accepted_commit:
            raise ValueError("accepted candidate commit does not match accepted_commit")
        if self.accepted_candidate.candidate_id != self.accepted_evaluation.candidate_id:
            raise ValueError("accepted candidate and evaluation identities differ")
        if self.accepted_candidate.artifact_digest != self.accepted_evaluation.artifact_digest:
            raise ValueError("accepted candidate and evaluation digests differ")
        if self.attempts != len(self.records):
            raise ValueError("attempt count must equal the number of step records")
        if any(record.attempt != index for index, record in enumerate(self.records, start=1)):
            raise ValueError("step record attempts must form a contiguous one-based sequence")
        return self
