"""
Structural interfaces used by the domain-neutral evolution engine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ardea_avo.core.models import (
    Candidate,
    EngineState,
    Evaluation,
    RejectionArchive,
    StepRecord,
    VariationRequest,
    VariationResult,
)


@runtime_checkable
class VariationAgent(Protocol):
    """
    Agent capable of mutating a candidate worktree for one attempt.
    """

    def vary(self, request: VariationRequest, workspace: Path) -> VariationResult:
        """
        Mutate workspace and return a durable summary of the attempted variation.
        """
        ...


@runtime_checkable
class CandidateEvaluator(Protocol):
    """
    External correctness and performance authority for candidates.
    """

    def evaluate(self, candidate: Candidate, workspace: Path) -> Evaluation:
        """
        Evaluate exactly the supplied candidate tree.
        """
        ...


@runtime_checkable
class LineageStore(Protocol):
    """
    Accepted-candidate lineage and rejected-attempt archive.
    """

    @property
    def workspace(self) -> Path:
        """
        Return the mutable candidate worktree.
        """
        ...

    def commit_seed(self, candidate: Candidate, evaluation: Evaluation) -> Candidate:
        """
        Commit the evaluated seed as lineage version zero.
        """
        ...

    def accept(self, candidate: Candidate, evaluation: Evaluation) -> Candidate:
        """
        Commit a correctness-passing, non-regressing candidate tree.
        """
        ...

    def reject(
        self,
        candidate: Candidate,
        evaluation: Evaluation | None,
        *,
        reason: str,
    ) -> RejectionArchive:
        """
        Archive the current patch and restore the accepted worktree.
        """
        ...

    def ensure_matches(self, candidate: Candidate) -> None:
        """
        Verify HEAD, cleanliness, and content against an accepted candidate.
        """
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """
    Durable storage boundary for resumable engine state and step records.
    """

    def load(self) -> EngineState | None:
        """
        Load the latest complete state, or return None when uninitialized.
        """
        ...

    def save(self, state: EngineState) -> None:
        """
        Atomically persist a complete engine state.
        """
        ...

    def append(self, record: StepRecord) -> None:
        """
        Append a durable attempt record before publishing a checkpoint.
        """
        ...


CheckpointCallback = Callable[[EngineState, StepRecord], None]
KnowledgeItems = Sequence[dict[str, object]]
