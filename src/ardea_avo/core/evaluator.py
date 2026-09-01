"""
Strict JSON subprocess adapter for external candidate evaluators.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from ardea_avo.core.digest import tree_digest
from ardea_avo.core.exceptions import ExternalEvaluatorError, StaleArtifactError
from ardea_avo.core.models import Candidate, Evaluation, Score

_SAFE_INHERITED_ENVIRONMENT = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")


class _EvaluatorPayload(BaseModel):
    """
    Exact stdout JSON contract implemented by external evaluators.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    artifact_digest: str
    correct: StrictBool
    metrics: dict[str, Any]
    evaluator: str = "external-evaluator"
    evidence: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metrics", mode="before")
    @classmethod
    def preserve_metric_input_for_score(cls, value: Any) -> Any:
        """
        Leave strict numeric and finite validation to the public Score model.
        """
        if not isinstance(value, dict):
            raise ValueError("metrics must be an object")
        return value


class ExternalEvaluator:
    """
    Run an evaluator command without a shell and parse its sole JSON result.

    The command may use the literal placeholders ``{candidate_root}``,
    ``{candidate_id}``, and ``{artifact_digest}``. Those values are also exposed
    as ``AVO_CANDIDATE_ROOT``, ``AVO_CANDIDATE_ID``, and
    ``AVO_ARTIFACT_DIGEST`` environment variables.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 300.0,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        inherit_environment: bool = False,
        immutable_cwd: bool = False,
    ) -> None:
        """
        Configure an exact argv evaluator invocation.
        """
        if not command or any(not isinstance(argument, str) or not argument for argument in command):
            raise ValueError("external evaluator command must contain non-empty argv strings")
        if timeout_seconds <= 0:
            raise ValueError("external evaluator timeout must be positive")
        self._command = tuple(command)
        self._timeout_seconds = timeout_seconds
        self._cwd = cwd.resolve() if cwd is not None else None
        self._environment = dict(environment or {})
        self._inherit_environment = inherit_environment
        if immutable_cwd and self._cwd is None:
            raise ValueError("immutable_cwd requires an evaluator working directory")
        self._immutable_cwd_digest = (
            tree_digest(self._cwd) if immutable_cwd and self._cwd is not None else None
        )

    def evaluate(self, candidate: Candidate, workspace: Path) -> Evaluation:
        """
        Evaluate a digest-bound candidate and reject process or artifact drift.
        """
        workspace = workspace.resolve()
        self._ensure_evaluator_unchanged("before")
        before = tree_digest(workspace)
        if before != candidate.artifact_digest:
            raise StaleArtifactError(
                f"candidate {candidate.candidate_id!r} digest is stale before evaluation: "
                f"recorded={candidate.artifact_digest}, actual={before}"
            )

        replacements = {
            "{candidate_root}": str(workspace),
            "{candidate_id}": candidate.candidate_id,
            "{artifact_digest}": candidate.artifact_digest,
        }
        command = [self._replace_placeholders(argument, replacements) for argument in self._command]
        environment = self._build_environment(candidate, workspace)
        try:
            completed = subprocess.run(
                command,
                cwd=self._cwd or workspace,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalEvaluatorError(
                f"external evaluator timed out after {self._timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise ExternalEvaluatorError(f"could not start external evaluator: {exc}") from exc

        after = tree_digest(workspace)
        self._ensure_evaluator_unchanged("after")
        if after != before:
            raise StaleArtifactError(
                "external evaluator mutated the candidate tree: " f"before={before}, after={after}"
            )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            detail = stderr[-2_000:] if stderr else "no stderr"
            raise ExternalEvaluatorError(
                f"external evaluator exited with status {completed.returncode}: {detail}"
            )

        try:
            raw_payload = json.loads(
                completed.stdout,
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_nonstandard_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ExternalEvaluatorError("external evaluator stdout must be one JSON object") from exc
        if not isinstance(raw_payload, dict):
            raise ExternalEvaluatorError("external evaluator stdout must be a JSON object")
        try:
            payload = _EvaluatorPayload.model_validate(raw_payload)
            score = Score(correct=payload.correct, metrics=payload.metrics)
            evaluation = Evaluation(
                candidate_id=payload.candidate_id,
                artifact_digest=payload.artifact_digest,
                score=score,
                evaluator=payload.evaluator,
                evidence=payload.evidence,
                metadata=payload.metadata,
            )
        except (ValidationError, ValueError) as exc:
            raise ExternalEvaluatorError(f"invalid external evaluator payload: {exc}") from exc

        if evaluation.candidate_id != candidate.candidate_id:
            raise StaleArtifactError(
                "external evaluator returned the wrong candidate id: "
                f"expected={candidate.candidate_id}, actual={evaluation.candidate_id}"
            )
        if evaluation.artifact_digest != candidate.artifact_digest:
            raise StaleArtifactError(
                "external evaluator returned a stale artifact digest: "
                f"expected={candidate.artifact_digest}, actual={evaluation.artifact_digest}"
            )
        return evaluation

    def _ensure_evaluator_unchanged(self, phase: str) -> None:
        if self._immutable_cwd_digest is None or self._cwd is None:
            return
        actual = tree_digest(self._cwd)
        if actual != self._immutable_cwd_digest:
            raise StaleArtifactError(
                f"immutable evaluator changed {phase} execution: "
                f"expected={self._immutable_cwd_digest}, actual={actual}"
            )

    @staticmethod
    def _replace_placeholders(argument: str, replacements: Mapping[str, str]) -> str:
        result = argument
        for marker, replacement in replacements.items():
            result = result.replace(marker, replacement)
        return result

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    @staticmethod
    def _reject_nonstandard_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    def _build_environment(self, candidate: Candidate, workspace: Path) -> dict[str, str]:
        if self._inherit_environment:
            environment = dict(os.environ)
        else:
            environment = {
                name: os.environ[name] for name in _SAFE_INHERITED_ENVIRONMENT if name in os.environ
            }
        environment.update(self._environment)
        environment.update(
            {
                "AVO_CANDIDATE_ROOT": str(workspace),
                "AVO_CANDIDATE_ID": candidate.candidate_id,
                "AVO_ARTIFACT_DIGEST": candidate.artifact_digest,
            }
        )
        return environment
