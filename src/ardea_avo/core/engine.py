"""
Paper-faithful single-lineage AVO orchestration.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ardea_avo.core.digest import archive_tree_digest, tree_digest
from ardea_avo.core.exceptions import (
    EngineStateError,
    EvaluationError,
    StaleArtifactError,
)
from ardea_avo.core.models import (
    Candidate,
    EngineState,
    Evaluation,
    ScoreComparison,
    StepDecision,
    StepRecord,
    TargetSpec,
    VariationRequest,
    VariationResult,
)
from ardea_avo.core.protocols import (
    CandidateEvaluator,
    CheckpointCallback,
    CheckpointStore,
    LineageStore,
    VariationAgent,
)


class EvolutionEngine:
    """
    Coordinate autonomous variation with host-authoritative evaluation and Git promotion.
    """

    def __init__(
        self,
        target: TargetSpec,
        agent: VariationAgent,
        evaluator: CandidateEvaluator,
        lineage: LineageStore,
        *,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> None:
        """
        Bind one target, agent, evaluator, and accepted lineage.
        """
        self.target = target
        self.agent = agent
        self.evaluator = evaluator
        self.lineage = lineage
        self.checkpoint_store = checkpoint_store
        self.checkpoint_callback = checkpoint_callback

    @property
    def workspace(self) -> Path:
        """
        Return the mutable candidate workspace owned by the lineage store.
        """
        return self.lineage.workspace

    def initialize(
        self,
        *,
        seed_id: str = "v0",
        metadata: dict[str, Any] | None = None,
    ) -> EngineState:
        """
        Evaluate the complete seed tree and commit it only if correctness passes.
        """
        seed = Candidate(
            candidate_id=seed_id,
            parent_id=None,
            generation=0,
            artifact_digest=tree_digest(self.workspace),
            metadata=metadata or {},
        )
        evaluation = self.evaluator.evaluate(seed, self.workspace)
        self._validate_evaluation(seed, evaluation)
        if not evaluation.score.correct:
            raise EvaluationError("evaluated seed failed the target correctness gate")

        committed_seed = self.lineage.commit_seed(seed, evaluation)
        state = EngineState(
            target_name=self.target.name,
            accepted_candidate=committed_seed,
            accepted_evaluation=evaluation,
            accepted_commit=self._required_commit(committed_seed),
        )
        if self.checkpoint_store is not None:
            self.checkpoint_store.save(state)
        return state

    def restore(self, state: EngineState) -> EngineState:
        """
        Validate a deserialized checkpoint against target and Git worktree state.
        """
        if state.target_name != self.target.name:
            raise EngineStateError(
                f"checkpoint target mismatch: expected={self.target.name}, actual={state.target_name}"
            )
        self._validate_evaluation(state.accepted_candidate, state.accepted_evaluation)
        if not state.accepted_evaluation.score.correct:
            raise EngineStateError("checkpoint accepted evaluation fails the correctness gate")
        self.lineage.ensure_matches(state.accepted_candidate)
        return state

    def restore_from_store(self) -> EngineState:
        """
        Load and validate the configured checkpoint store.
        """
        if self.checkpoint_store is None:
            raise EngineStateError("no checkpoint store is configured")
        state = self.checkpoint_store.load()
        if state is None:
            raise EngineStateError("checkpoint store is uninitialized")
        return self.restore(state)

    def step(
        self,
        state: EngineState,
        *,
        knowledge: Iterable[dict[str, Any]] = (),
    ) -> EngineState:
        """
        Run one inspect/edit/evaluate decision and publish a resumable state.
        """
        self.restore(state)
        attempt = state.attempts + 1
        request = VariationRequest(
            candidate=state.accepted_candidate,
            target=self.target,
            prior_evaluation=state.accepted_evaluation,
            knowledge=tuple(knowledge),
            attempt=attempt,
        )

        variation: VariationResult | None = None
        try:
            variation = self.agent.vary(request, self.workspace)
            if not isinstance(variation, VariationResult):
                raise TypeError("variation agent must return VariationResult")
            candidate = self._proposal_candidate(state, attempt, metadata=variation.metadata)
        except Exception as exc:
            error = self._error_text(exc)
            failed_candidate = self._proposal_candidate(
                state,
                attempt,
                metadata={"agent_failed": True},
                archival=True,
            )
            archive = self.lineage.reject(
                failed_candidate,
                None,
                reason=f"agent variation failed: {error}",
            )
            record = StepRecord(
                attempt=attempt,
                candidate=failed_candidate,
                evaluation=None,
                variation=variation,
                decision=StepDecision.AGENT_FAILED,
                comparison=None,
                rejection=archive,
                error=error,
            )
            return self._publish(state, record)

        try:
            evaluation = self.evaluator.evaluate(candidate, self.workspace)
            self._validate_evaluation(candidate, evaluation)
        except Exception as exc:
            error = self._error_text(exc)
            actual_digest = archive_tree_digest(self.workspace)
            archive_candidate = candidate
            if actual_digest != candidate.artifact_digest:
                archive_candidate = candidate.model_copy(update={"artifact_digest": actual_digest})
            archive = self.lineage.reject(
                archive_candidate,
                None,
                reason=f"evaluation failed: {error}",
            )
            record = StepRecord(
                attempt=attempt,
                candidate=archive_candidate,
                evaluation=None,
                variation=variation,
                decision=StepDecision.EVALUATION_FAILED,
                comparison=None,
                rejection=archive,
                error=error,
            )
            return self._publish(state, record)

        if not evaluation.score.correct:
            rejection = self.lineage.reject(
                candidate,
                evaluation,
                reason="candidate failed the target correctness gate",
            )
            record = StepRecord(
                attempt=attempt,
                candidate=candidate,
                evaluation=evaluation,
                variation=variation,
                decision=StepDecision.REJECTED_INCORRECT,
                comparison=None,
                rejection=rejection,
            )
            return self._publish(state, record)

        comparison = self.target.compare(evaluation.score, state.accepted_evaluation.score)
        if comparison is ScoreComparison.REGRESSED:
            rejection = self.lineage.reject(
                candidate,
                evaluation,
                reason="candidate regressed under the target's lexicographic comparator",
            )
            record = StepRecord(
                attempt=attempt,
                candidate=candidate,
                evaluation=evaluation,
                variation=variation,
                decision=StepDecision.REJECTED_REGRESSION,
                comparison=comparison,
                rejection=rejection,
            )
            return self._publish(state, record)

        committed = self.lineage.accept(candidate, evaluation)
        record = StepRecord(
            attempt=attempt,
            candidate=committed,
            evaluation=evaluation,
            variation=variation,
            decision=StepDecision.ACCEPTED,
            comparison=comparison,
            accepted_commit=self._required_commit(committed),
        )
        return self._publish(state, record)

    def run(
        self,
        state: EngineState,
        attempts: int,
        *,
        knowledge: Iterable[dict[str, Any]] = (),
    ) -> EngineState:
        """
        Run a bounded sequence of variation attempts from a validated state.
        """
        if attempts < 0:
            raise ValueError("attempts must be non-negative")
        current = state
        stable_knowledge = tuple(knowledge)
        for _ in range(attempts):
            current = self.step(current, knowledge=stable_knowledge)
        return current

    def _proposal_candidate(
        self,
        state: EngineState,
        attempt: int,
        *,
        metadata: dict[str, Any],
        archival: bool = False,
    ) -> Candidate:
        digest = archive_tree_digest(self.workspace) if archival else tree_digest(self.workspace)
        return Candidate(
            candidate_id=f"candidate-{attempt:06d}",
            parent_id=state.accepted_candidate.candidate_id,
            generation=state.accepted_candidate.generation + 1,
            artifact_digest=digest,
            metadata=metadata,
        )

    def _validate_evaluation(self, candidate: Candidate, evaluation: Evaluation) -> None:
        if evaluation.candidate_id != candidate.candidate_id:
            raise StaleArtifactError(
                f"evaluation candidate mismatch: expected={candidate.candidate_id}, "
                f"actual={evaluation.candidate_id}"
            )
        if evaluation.artifact_digest != candidate.artifact_digest:
            raise StaleArtifactError(
                f"evaluation digest mismatch: expected={candidate.artifact_digest}, "
                f"actual={evaluation.artifact_digest}"
            )
        current_digest = tree_digest(self.workspace)
        if current_digest != candidate.artifact_digest:
            raise StaleArtifactError(
                f"candidate changed during evaluation: expected={candidate.artifact_digest}, "
                f"actual={current_digest}"
            )
        self.target.validate_score(evaluation.score)

    def _publish(self, prior: EngineState, record: StepRecord) -> EngineState:
        if record.decision is StepDecision.ACCEPTED:
            if record.evaluation is None:
                raise EngineStateError("accepted step record is missing its evaluation")
            accepted_candidate = record.candidate
            accepted_evaluation = record.evaluation
            accepted_commit = self._required_commit(record.candidate)
        else:
            accepted_candidate = prior.accepted_candidate
            accepted_evaluation = prior.accepted_evaluation
            accepted_commit = prior.accepted_commit

        state = EngineState(
            target_name=prior.target_name,
            accepted_candidate=accepted_candidate,
            accepted_evaluation=accepted_evaluation,
            accepted_commit=accepted_commit,
            attempts=record.attempt,
            records=(*prior.records, record),
        )
        if self.checkpoint_store is not None:
            self.checkpoint_store.append(record)
            self.checkpoint_store.save(state)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(state, record)
        return state

    @staticmethod
    def _required_commit(candidate: Candidate) -> str:
        if candidate.commit_hash is None:
            raise EngineStateError(f"accepted candidate {candidate.candidate_id!r} has no commit hash")
        return candidate.commit_hash

    @staticmethod
    def _error_text(error: Exception) -> str:
        detail = f"{type(error).__name__}: {error}"
        if len(detail) <= 10_000:
            return detail
        return detail[:9_970] + "…[truncated]"
