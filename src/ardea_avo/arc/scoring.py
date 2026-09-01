"""
Official ARC-AGI-3 RHAE scoring helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

LEVEL_SCORE_CAP = 1.15


class ProgressRecord(Protocol):
    """
    Minimal action record needed to assign actions to completed levels.
    """

    @property
    def levels_completed(self) -> int:
        """
        Return cumulative completed levels after this action.
        """

        ...


@dataclass(frozen=True, slots=True)
class GameScore:
    """
    Auditable components of one official game score.
    """

    score: float
    completed_levels: int
    total_levels: int
    per_level_actions: tuple[int, ...]
    per_level_scores: tuple[float, ...]

    @property
    def percent(self) -> float:
        """
        Return this game score on the scorecard's 0 through 100 scale.
        """

        return self.score * 100.0


def action_budget(
    baselines: Sequence[int],
    *,
    multiplier: int = 5,
    hard_cap: int = 2_500,
) -> int:
    """
    Calculate the private per-game action limit.
    """

    _validate_baselines(baselines)
    if isinstance(multiplier, bool) or multiplier <= 0:
        raise ValueError("action-budget multiplier must be positive")
    if isinstance(hard_cap, bool) or hard_cap <= 0:
        raise ValueError("action-budget hard cap must be positive")
    return min(multiplier * sum(baselines), hard_cap)


def baseline_sha256(baselines: Sequence[int]) -> str:
    """
    Bind replay to private baseline metadata without recording the values.
    """

    _validate_baselines(baselines)
    payload = "ARDEA-ARC-BASELINES-v1\0" + ",".join(str(value) for value in baselines)
    return sha256(payload.encode("ascii")).hexdigest()


def _validate_baselines(baselines: Sequence[int]) -> None:
    if not baselines:
        raise ValueError("at least one level baseline is required")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in baselines):
        raise ValueError("all level baselines must be positive integers")


def _read_levels(record: ProgressRecord | Mapping[str, object]) -> int:
    raw = record["levels_completed"] if isinstance(record, Mapping) else record.levels_completed
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError("trace levels_completed values must be non-negative integers")
    return raw


def actions_per_completed_level(
    records: Iterable[ProgressRecord | Mapping[str, object]],
) -> tuple[int, ...]:
    """
    Charge actions to each completed level and omit an unfinished tail.

    A single action may complete multiple levels. The first receives the action
    count accumulated since the prior completion and subsequent levels receive
    zero. The official scorer assigns those zero-action levels a score of zero.
    """

    completed: list[int] = []
    actions_since_completion = 0
    previous_levels = 0
    for record in records:
        actions_since_completion += 1
        levels = _read_levels(record)
        if levels < previous_levels:
            raise ValueError("levels_completed cannot decrease within a trace")
        while len(completed) < levels:
            completed.append(actions_since_completion)
            actions_since_completion = 0
        previous_levels = levels
    return tuple(completed)


def level_rhae(baseline_actions: int, agent_actions: int) -> float:
    """
    Score one completed level with the pinned official zero-action semantics.
    """

    if (
        isinstance(baseline_actions, bool)
        or not isinstance(baseline_actions, int)
        or baseline_actions <= 0
    ):
        raise ValueError("baseline actions must be a positive integer")
    if isinstance(agent_actions, bool) or not isinstance(agent_actions, int) or agent_actions < 0:
        raise ValueError("agent actions must be a non-negative integer")
    if agent_actions == 0:
        return 0.0
    return min((baseline_actions / agent_actions) ** 2, LEVEL_SCORE_CAP)


def score_game(baselines: Sequence[int], per_level_actions: Sequence[int]) -> GameScore:
    """
    Score a game with level-number weights and the completion ceiling.
    """

    _validate_baselines(baselines)
    if len(per_level_actions) > len(baselines):
        raise ValueError("completed action counts cannot exceed the number of levels")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in per_level_actions):
        raise ValueError("per-level action counts must be non-negative integers")

    scores = tuple(
        level_rhae(baseline, actions)
        for baseline, actions in zip(baselines, per_level_actions, strict=False)
    )
    total_weight = sum(range(1, len(baselines) + 1))
    earned = sum(weight * score for weight, score in enumerate(scores, start=1))
    positive_score_weight = sum(
        weight
        for weight, score in enumerate(scores, start=1)
        if score > 0.0
    )
    completion_ceiling = positive_score_weight / total_weight
    normalized = earned / total_weight
    return GameScore(
        score=min(normalized, completion_ceiling),
        completed_levels=len(scores),
        total_levels=len(baselines),
        per_level_actions=tuple(per_level_actions),
        per_level_scores=scores,
    )


def board_rhae(games: Sequence[GameScore | float]) -> float:
    """
    Average game scores and return the official 0 through 100 board value.
    """

    if not games:
        raise ValueError("at least one game score is required")
    values = tuple(game.score if isinstance(game, GameScore) else float(game) for game in games)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("game scores must be between zero and one")
    return 100.0 * sum(values) / len(values)
