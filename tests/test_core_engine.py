from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ardea_avo.core import (
    Candidate,
    EngineState,
    EngineStateError,
    Evaluation,
    EvolutionEngine,
    GitLineage,
    MetricObjective,
    Score,
    ScoreComparison,
    StaleArtifactError,
    StepDecision,
    StepRecord,
    TargetSpec,
    VariationRequest,
    VariationResult,
    tree_digest,
)


class WritingAgent:
    def __init__(self, changes: list[tuple[str, float]]) -> None:
        self.changes = changes

    def vary(self, request: VariationRequest, workspace: Path) -> VariationResult:
        name, value = self.changes.pop(0)
        (workspace / "score.txt").write_text(str(value), encoding="utf-8")
        (workspace / name).write_text(f"attempt {request.attempt}", encoding="utf-8")
        return VariationResult(summary=f"set score to {value}", metadata={"file": name})


class ScoreFileEvaluator:
    def evaluate(self, candidate: Candidate, workspace: Path) -> Evaluation:
        value = float((workspace / "score.txt").read_text(encoding="utf-8"))
        return Evaluation(
            candidate_id=candidate.candidate_id,
            artifact_digest=tree_digest(workspace),
            score=Score(correct=value >= 0, metrics={"score": value}),
            evaluator="score-file",
        )


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self.state: EngineState | None = None
        self.records: list[StepRecord] = []

    def load(self) -> EngineState | None:
        return self.state

    def save(self, state: EngineState) -> None:
        self.state = EngineState.model_validate_json(state.model_dump_json())

    def append(self, record: StepRecord) -> None:
        self.records.append(StepRecord.model_validate_json(record.model_dump_json()))


def _engine(
    tmp_path: Path,
    agent: Any,
    evaluator: Any | None = None,
    checkpoint_store: MemoryCheckpointStore | None = None,
) -> EvolutionEngine:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "score.txt").write_text("1", encoding="utf-8")
    return EvolutionEngine(
        TargetSpec(name="fixture", objectives=(MetricObjective(name="score"),)),
        agent,
        evaluator or ScoreFileEvaluator(),
        GitLineage(workspace, tmp_path / "rejected"),
        checkpoint_store=checkpoint_store,
    )


def test_engine_evaluates_seed_accepts_improvement_and_tie_then_rejects_failures(
    tmp_path: Path,
) -> None:
    store = MemoryCheckpointStore()
    agent = WritingAgent(
        [
            ("improvement.txt", 2),
            ("different-but-equal.txt", 2),
            ("regression.txt", 0),
            ("incorrect.txt", -1),
        ]
    )
    engine = _engine(tmp_path, agent, checkpoint_store=store)
    state = engine.initialize(metadata={"seed": True})

    assert state.accepted_candidate.candidate_id == "v0"
    assert state.accepted_evaluation.score.metrics == {"score": 1.0}
    assert store.state == state

    state = engine.step(state, knowledge=({"claim": "try higher values"},))
    assert state.records[-1].decision is StepDecision.ACCEPTED
    assert state.records[-1].comparison is ScoreComparison.IMPROVED
    first_commit = state.accepted_commit

    state = engine.step(state)
    assert state.records[-1].decision is StepDecision.ACCEPTED
    assert state.records[-1].comparison is ScoreComparison.EQUAL
    assert state.accepted_commit != first_commit
    accepted_digest = state.accepted_candidate.artifact_digest

    state = engine.step(state)
    assert state.records[-1].decision is StepDecision.REJECTED_REGRESSION
    assert state.records[-1].rejection is not None
    assert state.accepted_candidate.artifact_digest == accepted_digest
    assert not (engine.workspace / "regression.txt").exists()

    state = engine.step(state)
    assert state.records[-1].decision is StepDecision.REJECTED_INCORRECT
    assert state.records[-1].comparison is None
    assert not (engine.workspace / "incorrect.txt").exists()
    assert state.attempts == 4
    assert len(store.records) == 4
    assert EngineState.model_validate_json(state.model_dump_json()) == state
    assert engine.restore_from_store() == state


class MutatingEvaluator(ScoreFileEvaluator):
    def evaluate(self, candidate: Candidate, workspace: Path) -> Evaluation:
        evaluation = super().evaluate(candidate, workspace)
        if candidate.generation > 0:
            (workspace / "after-evaluation.txt").write_text("stale", encoding="utf-8")
        return evaluation


def test_engine_archives_and_rolls_back_stale_evaluation(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        WritingAgent([("proposal.txt", 2)]),
        evaluator=MutatingEvaluator(),
    )
    state = engine.initialize()

    state = engine.step(state)

    assert state.attempts == 1
    assert state.records[-1].decision is StepDecision.EVALUATION_FAILED
    assert "changed during evaluation" in (state.records[-1].error or "")
    assert (engine.workspace / "score.txt").read_text(encoding="utf-8") == "1"
    assert not (engine.workspace / "proposal.txt").exists()
    assert not (engine.workspace / "after-evaluation.txt").exists()
    assert list((tmp_path / "rejected").glob("*.patch"))
    engine.restore(state)


class FailingAgent:
    def vary(self, request: VariationRequest, workspace: Path) -> VariationResult:
        (workspace / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError(f"failure at {request.attempt}")


def test_engine_archives_and_rolls_back_agent_failure(tmp_path: Path) -> None:
    store = MemoryCheckpointStore()
    engine = _engine(tmp_path, FailingAgent(), checkpoint_store=store)
    state = engine.initialize()

    state = engine.step(state)

    assert state.attempts == 1
    assert state.records[-1].decision is StepDecision.AGENT_FAILED
    assert "failure at 1" in (state.records[-1].error or "")
    assert not (engine.workspace / "partial.txt").exists()
    assert list((tmp_path / "rejected").glob("*.patch"))
    assert store.state == state
    assert store.records[-1] == state.records[-1]

    state = engine.step(state)
    assert state.attempts == 2
    assert state.records[-1].candidate.candidate_id == "candidate-000002"
    assert store.state == state
    engine.restore(state)


class GitRedirectAgent:
    def vary(self, request: VariationRequest, workspace: Path) -> VariationResult:
        (workspace / ".git").write_text("gitdir: /tmp/attacker", encoding="utf-8")
        return VariationResult(summary=f"redirect Git on attempt {request.attempt}")


def test_engine_records_and_quarantines_forbidden_git_entry(tmp_path: Path) -> None:
    engine = _engine(tmp_path, GitRedirectAgent())
    initial = engine.initialize()

    state = engine.step(initial)

    assert state.attempts == 1
    assert state.records[-1].decision is StepDecision.AGENT_FAILED
    assert "forbidden .git entry" in (state.records[-1].error or "")
    assert not (engine.workspace / ".git").exists()
    assert list((tmp_path / "rejected" / "invalid").rglob(".git"))
    engine.restore(state)


def test_engine_restore_detects_checkpoint_or_worktree_drift(tmp_path: Path) -> None:
    engine = _engine(tmp_path, WritingAgent([]))
    state = engine.initialize()
    (engine.workspace / "score.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(StaleArtifactError):
        engine.restore(state)

    wrong_target = state.model_copy(update={"target_name": "different"})
    with pytest.raises(EngineStateError, match="target mismatch"):
        engine.restore(wrong_target)
