"""
Domain-neutral, paper-faithful Agentic Variation Operator primitives.
"""

from ardea_avo.core.digest import tree_digest
from ardea_avo.core.engine import EvolutionEngine
from ardea_avo.core.evaluator import ExternalEvaluator
from ardea_avo.core.exceptions import (
    AgentVariationError,
    AVOError,
    ConfigurationError,
    EngineStateError,
    EvaluationError,
    ExternalEvaluatorError,
    LineageError,
    LineageStateError,
    ScoreValidationError,
    StaleArtifactError,
)
from ardea_avo.core.lineage import GitLineage
from ardea_avo.core.models import (
    Candidate,
    EngineState,
    Evaluation,
    MetricObjective,
    ObjectiveDirection,
    RejectionArchive,
    Score,
    ScoreComparison,
    StepDecision,
    StepRecord,
    TargetSpec,
    VariationRequest,
    VariationResult,
)
from ardea_avo.core.protocols import (
    CandidateEvaluator,
    CheckpointStore,
    LineageStore,
    VariationAgent,
)

__all__ = [
    "AVOError",
    "AgentVariationError",
    "Candidate",
    "CandidateEvaluator",
    "CheckpointStore",
    "ConfigurationError",
    "EngineState",
    "EngineStateError",
    "Evaluation",
    "EvaluationError",
    "EvolutionEngine",
    "ExternalEvaluator",
    "ExternalEvaluatorError",
    "GitLineage",
    "LineageError",
    "LineageStateError",
    "LineageStore",
    "MetricObjective",
    "ObjectiveDirection",
    "RejectionArchive",
    "Score",
    "ScoreComparison",
    "ScoreValidationError",
    "StaleArtifactError",
    "StepDecision",
    "StepRecord",
    "TargetSpec",
    "VariationAgent",
    "VariationRequest",
    "VariationResult",
    "tree_digest",
]
