"""
Composition and recovery boundary for generic AVO evolution runs.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from ardea_avo.checkpoints import JsonCheckpointStore
from ardea_avo.core import (
    EngineState,
    EvolutionEngine,
    ExternalEvaluator,
    GitLineage,
    tree_digest,
)
from ardea_avo.runtime import (
    DEFAULT_MAX_COST_USD,
    AgentRequest,
    AgentResult,
    BudgetLedger,
    ModelPricing,
)
from ardea_avo.runtime._io import atomic_write_json, sha256_json
from ardea_avo.target_config import TargetFile
from ardea_avo.variation import BackendVariationAgent

_DEFINITION_FIELDS = {
    "schema_version",
    "target",
    "backend_type",
    "initial_max_cost_usd",
    "seed_digest",
    "knowledge_digest",
    "pricing_model",
    "pricing_version",
    "evaluator_cwd",
    "evaluator_cwd_digest",
    "evaluator_snapshot_digest",
}


class AgentBackend(Protocol):
    """
    Dependency-injected model backend used by the generic variation agent.
    """

    budget: BudgetLedger

    def run(self, request: AgentRequest) -> AgentResult:
        """
        Execute one budgeted autonomous variation turn.
        """
        ...


BackendFactory = Callable[[BudgetLedger], AgentBackend]


@dataclass(frozen=True, slots=True)
class EvolutionLayout:
    """
    Fixed candidate and host-owned filesystem layout for one generic run.
    """

    run_directory: Path
    root: Path
    workspace: Path
    host: Path
    evaluator_snapshot: Path
    rejected: Path
    git_dir: Path
    budget: Path
    definition: Path
    knowledge: Path

    @classmethod
    def for_run(cls, run_directory: str | Path) -> EvolutionLayout:
        """
        Derive all evolution paths from a caller-owned run directory.
        """
        run = Path(run_directory).resolve()
        root = run / "evolution"
        host = root / "host"
        return cls(
            run_directory=run,
            root=root,
            workspace=root / "workspace",
            host=host,
            evaluator_snapshot=host / "evaluator-source",
            rejected=host / "rejected",
            git_dir=host / "lineage.git",
            budget=host / "budget",
            definition=host / "definition.json",
            knowledge=host / "knowledge.json",
        )


@dataclass(slots=True)
class EvolutionRun:
    """
    Live generic evolution composition with mutable resumable engine state.
    """

    target: TargetFile
    layout: EvolutionLayout
    budget: BudgetLedger
    backend: AgentBackend
    agent: BackendVariationAgent
    evaluator: ExternalEvaluator
    lineage: GitLineage
    checkpoints: JsonCheckpointStore
    engine: EvolutionEngine
    state: EngineState = field(init=False)
    released_reservations: tuple[str, ...] = ()

    def advance(self, attempts: int = 1) -> EngineState:
        """
        Execute a bounded number of attempts while honoring the shared hard gate.
        """
        if attempts < 0:
            raise ValueError("attempts must be non-negative")
        if (self.layout.run_directory / "sealed.json").exists():
            raise RuntimeError("sealed parent evolutions are immutable")
        for _ in range(attempts):
            self.budget.ensure_can_start()
            knowledge = (*self.knowledge_items(), *self.history_items())
            self.state = self.engine.step(self.state, knowledge=knowledge)
        return self.state

    def knowledge_items(self) -> tuple[dict[str, object], ...]:
        """
        Load and validate the immutable knowledge snapshot supplied to each turn.
        """
        try:
            document = json.loads(self.layout.knowledge.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("evolution knowledge snapshot is unreadable") from error
        if not isinstance(document, dict) or set(document) != {"schema_version", "items"}:
            raise ValueError("evolution knowledge snapshot has an invalid schema")
        if document["schema_version"] != 1 or not isinstance(document["items"], list):
            raise ValueError("evolution knowledge snapshot has an invalid schema")
        items: list[dict[str, object]] = []
        for item in document["items"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"source", "content"}
                or not isinstance(item["source"], str)
                or not isinstance(item["content"], str)
            ):
                raise ValueError("evolution knowledge snapshot contains an invalid item")
            items.append({"source": item["source"], "content": item["content"]})
        return tuple(items)

    def history_items(self, *, limit: int = 20) -> tuple[dict[str, object], ...]:
        """
        Summarize recent host evaluations so later variations learn from failures.
        """

        if limit < 1:
            raise ValueError("history limit must be positive")
        items: list[dict[str, object]] = []
        for record in self.state.records[-limit:]:
            evaluation = record.evaluation
            payload = {
                "attempt": record.attempt,
                "decision": record.decision.value,
                "variation_summary": record.variation.summary if record.variation else None,
                "correct": evaluation.score.correct if evaluation else None,
                "metrics": evaluation.score.metrics if evaluation else None,
                "evidence": evaluation.evidence if evaluation else (),
                "error": record.error,
            }
            items.append(
                {
                    "source": f"host-evaluation-attempt-{record.attempt}",
                    "content": json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                }
            )
        return tuple(items)


def start_evolution(
    target: TargetFile,
    run_directory: str | Path,
    *,
    backend_factory: BackendFactory,
    max_cost_usd: Decimal | str | float = DEFAULT_MAX_COST_USD,
    pricing: ModelPricing | None = None,
    evaluator_cwd: str | Path | None = None,
) -> EvolutionRun:
    """
    Copy a seed, score and commit ``v0``, and return a ready generic run.

    ``run_directory/evolution`` must not already exist. If initialization fails,
    only that newly created evolution subtree is removed; unrelated run artifacts
    remain untouched.
    """
    normalized_target = _normalize_target(target)
    layout = EvolutionLayout.for_run(run_directory)
    if layout.root.exists():
        raise FileExistsError(f"evolution run already exists: {layout.root}")
    if layout.run_directory.is_relative_to(normalized_target.seed):
        raise ValueError("run directory cannot be nested inside the seed tree")

    layout.run_directory.mkdir(parents=True, exist_ok=True)
    layout.root.mkdir(parents=False)
    try:
        _copy_tree(normalized_target.seed, layout.workspace)
        _copy_tree(normalized_target.seed, layout.evaluator_snapshot)
        seed_digest = tree_digest(layout.workspace)
        evaluator_snapshot_digest = tree_digest(layout.evaluator_snapshot)

        knowledge_items = _collect_knowledge(normalized_target.knowledge)
        knowledge_document: dict[str, object] = {
            "schema_version": 1,
            "items": knowledge_items,
        }
        atomic_write_json(layout.knowledge, knowledge_document)

        budget = BudgetLedger(
            layout.budget,
            max_cost_usd=max_cost_usd,
            pricing=pricing,
        )
        backend = _make_backend(backend_factory, budget)
        selected_evaluator_cwd = _select_start_evaluator_cwd(
            evaluator_cwd,
            layout,
        )
        definition = {
            "schema_version": 1,
            "target": normalized_target.model_dump(mode="json"),
            "backend_type": _backend_type(backend),
            "initial_max_cost_usd": str(budget.max_cost_usd),
            "pricing_model": budget.pricing.model,
            "pricing_version": budget.pricing.version,
            "seed_digest": seed_digest,
            "knowledge_digest": sha256_json(knowledge_document),
            "evaluator_cwd": (
                "snapshot"
                if selected_evaluator_cwd == layout.evaluator_snapshot
                else str(selected_evaluator_cwd)
            ),
            "evaluator_cwd_digest": tree_digest(selected_evaluator_cwd),
            "evaluator_snapshot_digest": evaluator_snapshot_digest,
        }
        atomic_write_json(layout.definition, definition)

        run = _compose_run(
            normalized_target,
            layout,
            budget,
            backend,
            selected_evaluator_cwd,
            session_id=None,
        )
        run.state = run.engine.initialize(
            metadata={
                "seed_digest": seed_digest,
                "definition_digest": sha256_json(definition),
            }
        )
        return run
    except Exception as error:
        try:
            shutil.rmtree(layout.root)
        except OSError as cleanup_error:
            error.add_note(f"could not clean failed evolution start: {cleanup_error}")
        raise


def resume_evolution(
    target: TargetFile,
    run_directory: str | Path,
    *,
    backend_factory: BackendFactory,
    max_cost_usd: Decimal | str | float | None = None,
    pricing: ModelPricing | None = None,
    evaluator_cwd: str | Path | None = None,
) -> EvolutionRun:
    """
    Validate and resume an existing generic lineage without recopying its seed.

    A supplied ``max_cost_usd`` is a new lifetime ceiling and may only raise the
    durable cap. Stranded reservations are released because resume assumes the
    caller has exclusive ownership of the stopped run.
    """
    layout = EvolutionLayout.for_run(run_directory)
    definition = _load_definition(layout)
    normalized_target = _resolve_target_identity(target)
    expected_target = normalized_target.model_dump(mode="json")
    if definition["target"] != expected_target:
        raise ValueError("resume target does not match the frozen evolution definition")

    knowledge_document = _load_json_object(layout.knowledge, "knowledge snapshot")
    if sha256_json(knowledge_document) != definition["knowledge_digest"]:
        raise ValueError("evolution knowledge snapshot digest mismatch")
    if tree_digest(layout.evaluator_snapshot) != definition["evaluator_snapshot_digest"]:
        raise ValueError("evaluator snapshot digest mismatch")

    selected_evaluator_cwd = _select_resume_evaluator_cwd(
        evaluator_cwd,
        layout,
        definition,
    )
    if tree_digest(selected_evaluator_cwd) != definition["evaluator_cwd_digest"]:
        raise ValueError("evaluator working directory digest mismatch")
    runtime_target = normalized_target.model_copy(
        update={"seed": layout.workspace, "knowledge": (layout.knowledge,)}
    )
    budget = BudgetLedger(
        layout.budget,
        max_cost_usd=str(definition["initial_max_cost_usd"]),
        pricing=pricing,
    )
    if (
        budget.pricing.model != definition["pricing_model"]
        or budget.pricing.version != definition["pricing_version"]
    ):
        raise ValueError("resume pricing does not match the frozen evolution definition")
    requested_cap: Decimal | None = None
    if max_cost_usd is not None:
        try:
            requested_cap = Decimal(str(max_cost_usd))
        except Exception as error:
            raise ValueError("resume max_cost_usd must be a finite positive amount") from error
        if not requested_cap.is_finite() or requested_cap <= 0:
            raise ValueError("resume max_cost_usd must be a finite positive amount")
        current = budget.max_cost_usd
        if requested_cap < current:
            raise ValueError("resume max_cost_usd cannot lower the lifetime ceiling")

    backend = _make_backend(backend_factory, budget)
    if _backend_type(backend) != definition["backend_type"]:
        raise ValueError("resume backend type does not match the frozen evolution definition")

    checkpoints = JsonCheckpointStore(layout.host)
    checkpoint = checkpoints.load()
    if checkpoint is None:
        raise ValueError("evolution checkpoint is missing")
    session_id = _latest_session_id(checkpoint)
    run = _compose_run(
        runtime_target,
        layout,
        budget,
        backend,
        selected_evaluator_cwd,
        session_id=session_id,
        checkpoints=checkpoints,
    )
    run.state = run.engine.restore(checkpoint)
    if requested_cap is not None and requested_cap > budget.max_cost_usd:
        budget.revise_cap(requested_cap, reason="generic evolution resume override")
    released = budget.release_all_active(reason="generic evolution resume recovery")
    run.released_reservations = released
    return run


def _compose_run(
    target: TargetFile,
    layout: EvolutionLayout,
    budget: BudgetLedger,
    backend: AgentBackend,
    evaluator_cwd: Path,
    *,
    session_id: str | None,
    checkpoints: JsonCheckpointStore | None = None,
) -> EvolutionRun:
    lineage = GitLineage(
        layout.workspace,
        layout.rejected,
        git_dir=layout.git_dir,
    )
    checkpoint_store = checkpoints or JsonCheckpointStore(layout.host)
    evaluator = ExternalEvaluator(
        target.evaluator,
        timeout_seconds=target.evaluator_timeout_seconds,
        cwd=evaluator_cwd,
        environment={"PYTHONDONTWRITEBYTECODE": "1"},
        immutable_cwd=True,
    )
    agent = BackendVariationAgent(
        backend,
        check_argv=target.check,
        session_id=session_id,
    )
    engine = EvolutionEngine(
        target.core_spec(),
        agent,
        evaluator,
        lineage,
        checkpoint_store=checkpoint_store,
    )
    return EvolutionRun(
        target=target,
        layout=layout,
        budget=budget,
        backend=backend,
        agent=agent,
        evaluator=evaluator,
        lineage=lineage,
        checkpoints=checkpoint_store,
        engine=engine,
    )


def _normalize_target(target: TargetFile) -> TargetFile:
    normalized = _resolve_target_identity(target)
    seed = normalized.seed
    knowledge = normalized.knowledge
    if not seed.is_dir():
        raise ValueError(f"target seed directory does not exist: {seed}")
    missing = [path for path in knowledge if not path.exists()]
    if missing:
        raise ValueError("target knowledge path does not exist: " + ", ".join(map(str, missing)))
    return normalized


def _resolve_target_identity(target: TargetFile) -> TargetFile:
    """
    Canonicalize target paths without requiring original inputs during resume.
    """

    return target.model_copy(
        update={
            "seed": target.seed.resolve(),
            "knowledge": tuple(path.resolve() for path in target.knowledge),
        }
    )


def _copy_tree(source: Path, destination: Path) -> None:
    def ignore_git(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name.casefold() == ".git"}

    shutil.copytree(source, destination, symlinks=True, ignore=ignore_git)
    tree_digest(destination)


def _collect_knowledge(sources: tuple[Path, ...]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for index, source in enumerate(sources):
        if source.is_symlink():
            raise ValueError(f"knowledge sources cannot be symlinks: {source}")
        if source.is_file():
            items.append(
                {
                    "source": f"{index}:{source.name}",
                    "content": _read_knowledge_file(source),
                }
            )
            continue
        if not source.is_dir():
            raise ValueError(f"knowledge source is not a regular file or directory: {source}")
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part.casefold() == ".git" for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"knowledge sources cannot contain symlinks: {path}")
            if path.is_file():
                items.append(
                    {
                        "source": f"{index}:{source.name}/{relative.as_posix()}",
                        "content": _read_knowledge_file(path),
                    }
                )
    return items


def _read_knowledge_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"knowledge source must be readable UTF-8 text: {path}") from error


def _select_start_evaluator_cwd(
    requested: str | Path | None,
    layout: EvolutionLayout,
) -> Path:
    selected = layout.evaluator_snapshot if requested is None else Path(requested).resolve()
    if not selected.is_dir():
        raise ValueError(f"evaluator working directory does not exist: {selected}")
    if selected == layout.workspace or selected.is_relative_to(layout.workspace):
        raise ValueError("evaluator working directory cannot be inside the mutable candidate workspace")
    return selected


def _select_resume_evaluator_cwd(
    requested: str | Path | None,
    layout: EvolutionLayout,
    definition: Mapping[str, object],
) -> Path:
    frozen = definition["evaluator_cwd"]
    if frozen == "snapshot":
        selected = layout.evaluator_snapshot
        if requested is not None and Path(requested).resolve() != selected:
            raise ValueError("resume evaluator_cwd does not match the frozen snapshot")
    else:
        selected = Path(str(frozen)).resolve()
        if requested is not None and Path(requested).resolve() != selected:
            raise ValueError("resume evaluator_cwd does not match the frozen definition")
    if not selected.is_dir():
        raise ValueError(f"evaluator working directory does not exist: {selected}")
    if selected == layout.workspace or selected.is_relative_to(layout.workspace):
        raise ValueError("evaluator working directory cannot be inside the mutable candidate workspace")
    return selected


def _make_backend(factory: BackendFactory, budget: BudgetLedger) -> AgentBackend:
    backend = factory(budget)
    if getattr(backend, "budget", None) is not budget:
        raise ValueError("backend factory must bind the supplied shared BudgetLedger instance")
    if not callable(getattr(backend, "run", None)):
        raise TypeError("backend factory must return an AgentBackend")
    return backend


def _backend_type(backend: AgentBackend) -> str:
    backend_class = type(backend)
    return f"{backend_class.__module__}.{backend_class.__qualname__}"


def _latest_session_id(state: EngineState) -> str | None:
    for record in reversed(state.records):
        if record.variation is None:
            continue
        value = record.variation.metadata.get("backend_session_id")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _load_definition(layout: EvolutionLayout) -> dict[str, object]:
    if not layout.root.is_dir() or not layout.workspace.is_dir() or not layout.host.is_dir():
        raise FileNotFoundError(f"evolution run is incomplete: {layout.root}")
    definition = _load_json_object(layout.definition, "definition")
    if set(definition) != _DEFINITION_FIELDS or definition.get("schema_version") != 1:
        raise ValueError("evolution definition has an invalid schema")
    string_fields = (
        "backend_type",
        "initial_max_cost_usd",
        "seed_digest",
        "knowledge_digest",
        "pricing_model",
        "pricing_version",
        "evaluator_cwd",
        "evaluator_cwd_digest",
        "evaluator_snapshot_digest",
    )
    if not isinstance(definition.get("target"), dict) or not all(
        isinstance(definition.get(field), str) and definition[field]
        for field in string_fields
    ):
        raise ValueError("evolution definition contains invalid values")
    return definition


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"evolution {label} is unreadable") from error
    if not isinstance(document, dict):
        raise ValueError(f"evolution {label} must be a JSON object")
    return document


__all__ = [
    "AgentBackend",
    "BackendFactory",
    "EvolutionLayout",
    "EvolutionRun",
    "resume_evolution",
    "start_evolution",
]
