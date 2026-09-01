"""
Tests for generic evolution composition, isolation, and recovery.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from ardea_avo.core import EvaluationError, MetricObjective, StepDecision
from ardea_avo.evolve import EvolutionRun, resume_evolution, start_evolution
from ardea_avo.runtime import AgentRequest, AgentResult, BudgetExceeded, BudgetLedger, TokenUsage
from ardea_avo.target_config import TargetFile


@dataclass(frozen=True, slots=True)
class Mutation:
    score: float
    filename: str


class FakeBackend:
    def __init__(
        self,
        budget: BudgetLedger,
        mutations: list[Mutation],
        *,
        usage: TokenUsage | None = None,
    ) -> None:
        self.budget = budget
        self.mutations = mutations
        self.usage = usage or TokenUsage()
        self.requests: list[AgentRequest] = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        mutation = self.mutations.pop(0)
        (request.cwd / "score.txt").write_text(str(mutation.score), encoding="utf-8")
        (request.cwd / mutation.filename).write_text("candidate change\n", encoding="utf-8")
        cost = self.budget.record_usage(
            self.usage,
            backend="fake",
            role=request.role,
            session_id=f"session-{mutation.filename}",
        )
        return AgentResult(
            text=f"changed score to {mutation.score}",
            session_id=f"session-{mutation.filename}",
            usage=self.usage,
            cost_usd=cost,
        )


class FactoryHarness:
    def __init__(
        self,
        batches: Sequence[Sequence[Mutation]],
        *,
        usage: TokenUsage | None = None,
    ) -> None:
        self.batches = [list(batch) for batch in batches]
        self.usage = usage or TokenUsage()
        self.instances: list[FakeBackend] = []

    def __call__(self, budget: BudgetLedger) -> FakeBackend:
        backend = FakeBackend(budget, self.batches.pop(0), usage=self.usage)
        self.instances.append(backend)
        return backend


def _target(tmp_path: Path, *, score: float = 1, knowledge: bool = False) -> TargetFile:
    seed = tmp_path / "seed"
    seed.mkdir(parents=True)
    (seed / "score.txt").write_text(str(score), encoding="utf-8")
    (seed / ".git").mkdir()
    (seed / ".git" / "config").write_text("untrusted source metadata", encoding="utf-8")
    (seed / "evaluator.py").write_text(
        """
import json
import os
from pathlib import Path

root = Path(os.environ["AVO_CANDIDATE_ROOT"])
score = float((root / "score.txt").read_text(encoding="utf-8"))
print(json.dumps({
    "candidate_id": os.environ["AVO_CANDIDATE_ID"],
    "artifact_digest": os.environ["AVO_ARTIFACT_DIGEST"],
    "correct": score >= 0,
    "metrics": {"score": score},
    "evaluator": "fixture-evaluator",
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    knowledge_paths: tuple[Path, ...] = ()
    if knowledge:
        facts = tmp_path / "facts.md"
        facts.write_text("Prefer monotonically larger fixture scores.\n", encoding="utf-8")
        knowledge_paths = (facts,)
    return TargetFile(
        name="fixture",
        seed=seed,
        evaluator=(sys.executable, "evaluator.py"),
        objectives=(MetricObjective(name="score"),),
        knowledge=knowledge_paths,
    )


def test_start_copies_seed_and_initializes_external_host_state(tmp_path: Path) -> None:
    """
    Score v0 from a copied tree while keeping Git, evaluator, and budget outside it.
    """
    target = _target(tmp_path)
    factory = FactoryHarness([[]])

    run = start_evolution(target, tmp_path / "run", backend_factory=factory)

    assert isinstance(run, EvolutionRun)
    assert run.state.accepted_candidate.candidate_id == "v0"
    assert run.state.accepted_evaluation.score.metrics == {"score": 1.0}
    assert run.layout.workspace != target.seed
    assert (run.layout.workspace / "score.txt").read_text(encoding="utf-8") == "1"
    assert not (run.layout.workspace / ".git").exists()
    assert run.layout.git_dir.is_dir()
    assert run.layout.git_dir.is_relative_to(run.layout.host)
    assert run.layout.evaluator_snapshot.is_dir()
    assert run.checkpoints.state_path.is_file()
    assert run.backend is factory.instances[0]
    assert run.backend.budget is run.budget
    assert run.budget.max_cost_usd == Decimal("20.00")
    assert factory.instances[0].requests == []


def test_advance_runs_agent_with_knowledge_and_keeps_only_non_regressing_lineage(
    tmp_path: Path,
) -> None:
    """
    Preserve improvement and tie commits while archiving regressions and failures.
    """
    target = _target(tmp_path, knowledge=True)
    factory = FactoryHarness(
        [
            [
                Mutation(2, "improvement.txt"),
                Mutation(2, "tie.txt"),
                Mutation(0, "regression.txt"),
                Mutation(-1, "incorrect.txt"),
            ]
        ]
    )
    run = start_evolution(target, tmp_path / "run", backend_factory=factory)

    state = run.advance(4)

    assert [record.decision for record in state.records] == [
        StepDecision.ACCEPTED,
        StepDecision.ACCEPTED,
        StepDecision.REJECTED_REGRESSION,
        StepDecision.REJECTED_INCORRECT,
    ]
    assert state.accepted_evaluation.score.metrics == {"score": 2.0}
    assert (run.layout.workspace / "improvement.txt").is_file()
    assert (run.layout.workspace / "tie.txt").is_file()
    assert not (run.layout.workspace / "regression.txt").exists()
    assert not (run.layout.workspace / "incorrect.txt").exists()
    assert "Prefer monotonically larger" in factory.instances[0].requests[0].prompt
    assert "host-evaluation-attempt-1" in factory.instances[0].requests[1].prompt
    assert "accepted" in factory.instances[0].requests[1].prompt
    assert len(list(run.layout.rejected.glob("*.patch"))) == 2


def test_resume_restores_checkpoint_session_budget_and_evaluator_snapshot(tmp_path: Path) -> None:
    """
    Continue the exact workspace without recopying a changed source seed.
    """
    target = _target(tmp_path)
    factory = FactoryHarness(
        [
            [Mutation(2, "first.txt")],
            [Mutation(3, "resumed.txt")],
        ]
    )
    original = start_evolution(target, tmp_path / "run", backend_factory=factory)
    original.advance()
    reservation = original.budget.reserve("1", role="variation")
    (target.seed / "score.txt").write_text("999", encoding="utf-8")
    (target.seed / "evaluator.py").write_text("raise RuntimeError('changed')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot lower"):
        resume_evolution(
            target,
            tmp_path / "run",
            backend_factory=factory,
            max_cost_usd="10",
        )
    assert len(factory.instances) == 1

    resumed = resume_evolution(
        target,
        tmp_path / "run",
        backend_factory=factory,
        max_cost_usd="25",
    )

    assert resumed.state == original.state
    assert resumed.released_reservations == (reservation,)
    assert resumed.budget.max_cost_usd == Decimal("25")
    assert (resumed.layout.workspace / "score.txt").read_text(encoding="utf-8") == "2"
    assert "fixture-evaluator" in (
        resumed.layout.evaluator_snapshot / "evaluator.py"
    ).read_text(encoding="utf-8")

    state = resumed.advance()
    assert state.accepted_evaluation.score.metrics == {"score": 3.0}
    assert factory.instances[1].requests[0].session_id == "session-first.txt"


def test_resume_rejects_target_or_snapshot_drift_before_backend_creation(tmp_path: Path) -> None:
    """
    Bind recovery to its original target and immutable knowledge snapshot.
    """
    target = _target(tmp_path, knowledge=True)
    factory = FactoryHarness([[], []])
    run = start_evolution(target, tmp_path / "run", backend_factory=factory)
    different = target.model_copy(update={"name": "different"})

    with pytest.raises(ValueError, match="target does not match"):
        resume_evolution(different, tmp_path / "run", backend_factory=factory)
    assert len(factory.instances) == 1

    document = json.loads(run.layout.knowledge.read_text(encoding="utf-8"))
    document["items"][0]["content"] = "tampered"
    run.layout.knowledge.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="knowledge snapshot digest"):
        resume_evolution(target, tmp_path / "run", backend_factory=factory)
    assert len(factory.instances) == 1


def test_resume_uses_run_owned_snapshots_after_original_inputs_are_deleted(
    tmp_path: Path,
) -> None:
    """
    Recovery does not depend on mutable seed or knowledge source availability.
    """

    target = _target(tmp_path, knowledge=True)
    factory = FactoryHarness([[], []])
    original = start_evolution(target, tmp_path / "run", backend_factory=factory)
    shutil.rmtree(target.seed)
    for path in target.knowledge:
        path.unlink()

    resumed = resume_evolution(
        target,
        tmp_path / "run",
        backend_factory=factory,
    )

    assert resumed.state == original.state
    assert resumed.layout.workspace.is_dir()
    assert resumed.knowledge_items()[0]["content"].startswith("Prefer monotonically")


def test_budget_gate_stops_before_creating_an_unrecorded_attempt(tmp_path: Path) -> None:
    """
    Check the durable ledger before every new autonomous turn.
    """
    target = _target(tmp_path)
    factory = FactoryHarness(
        [[Mutation(2, "first.txt"), Mutation(3, "blocked.txt")]],
        usage=TokenUsage(output_tokens=1),
    )
    run = start_evolution(
        target,
        tmp_path / "run",
        backend_factory=factory,
        max_cost_usd="0.00001",
    )

    run.advance()
    assert run.state.attempts == 1
    with pytest.raises(BudgetExceeded):
        run.advance()
    assert run.state.attempts == 1
    assert len(factory.instances[0].requests) == 1


def test_start_is_non_overwriting_and_cleans_failed_seed_evaluation(tmp_path: Path) -> None:
    """
    Never replace an existing lineage and remove only a failed new evolution subtree.
    """
    target = _target(tmp_path)
    factory = FactoryHarness([[]])
    run = start_evolution(target, tmp_path / "run", backend_factory=factory)

    with pytest.raises(FileExistsError):
        start_evolution(target, tmp_path / "run", backend_factory=FactoryHarness([[]]))
    assert run.checkpoints.state_path.is_file()

    failing_root = tmp_path / "failing"
    failing_target = _target(failing_root, score=-1)
    failing_factory = FactoryHarness([[]])
    with pytest.raises(EvaluationError, match="correctness gate"):
        start_evolution(
            failing_target,
            failing_root / "run",
            backend_factory=failing_factory,
        )
    assert not (failing_root / "run" / "evolution").exists()


def test_backend_factory_must_use_the_orchestrator_ledger(tmp_path: Path) -> None:
    """
    Reject a backend that would account usage against a different run budget.
    """
    target = _target(tmp_path)

    def wrong_factory(_budget: BudgetLedger) -> FakeBackend:
        return FakeBackend(BudgetLedger(tmp_path / "wrong-budget"), [])

    with pytest.raises(ValueError, match="shared BudgetLedger"):
        start_evolution(target, tmp_path / "run", backend_factory=wrong_factory)
    assert not (tmp_path / "run" / "evolution").exists()


def test_sealed_generic_parent_cannot_advance(tmp_path: Path) -> None:
    """
    A live object cannot bypass the cold/warm parent immutability boundary.
    """

    target = _target(tmp_path)
    run = start_evolution(
        target,
        tmp_path / "run",
        backend_factory=FactoryHarness([[Mutation(2, "forbidden.txt")]]),
    )
    (run.layout.run_directory / "sealed.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sealed"):
        run.advance()
    assert run.state.attempts == 0
