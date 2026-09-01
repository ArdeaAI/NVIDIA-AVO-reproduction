"""
Exceptions raised by the domain-neutral AVO core.
"""


class AVOError(Exception):
    """
    Base class for core AVO failures.
    """


class ConfigurationError(AVOError):
    """
    Raised when a target or component is configured inconsistently.
    """


class ScoreValidationError(AVOError, ValueError):
    """
    Raised when evaluator metrics do not match the target contract.
    """


class EvaluationError(AVOError):
    """
    Raised when an evaluator cannot produce a trustworthy evaluation.
    """


class ExternalEvaluatorError(EvaluationError):
    """
    Raised when an external evaluator process fails or returns invalid data.
    """


class StaleArtifactError(EvaluationError):
    """
    Raised when an evaluation does not describe the current candidate tree.
    """


class LineageError(AVOError):
    """
    Raised for Git lineage setup, commit, archive, or rollback failures.
    """


class LineageStateError(LineageError):
    """
    Raised when the worktree does not match the recorded accepted candidate.
    """


class AgentVariationError(AVOError):
    """
    Raised after an agent failure has been archived and rolled back.
    """


class EngineStateError(AVOError):
    """
    Raised when an evolution operation is incompatible with engine state.
    """
