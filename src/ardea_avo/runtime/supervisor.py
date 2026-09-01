"""
Deterministic stagnation detection and bounded supervisor redirects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class SupervisorTrigger(StrEnum):
    """
    Host-observable conditions that justify a strategy redirect.
    """

    NO_STATE_CHANGE = "no_state_change"
    REPEATED_DEATH = "repeated_death"
    PROGRESS_PLATEAU = "progress_plateau"
    ZERO_ACTION_EPISODES = "zero_action_episodes"


@dataclass(frozen=True, slots=True)
class SupervisorRedirect:
    """
    One trigger and exactly three causally distinct strategy directions.
    """

    trigger: SupervisorTrigger
    directions: tuple[str, str, str]

    def __post_init__(self) -> None:
        """
        Enforce the supervisor's deliberately narrow authority.
        """

        if len(self.directions) != 3:
            raise ValueError("a supervisor redirect must contain exactly three directions")
        normalized = {direction.strip().casefold() for direction in self.directions}
        if "" in normalized or len(normalized) != 3:
            raise ValueError("supervisor directions must be non-empty and distinct")

    def as_prompt(self) -> str:
        """
        Render suggestions without transferring decision authority.
        """

        options = "\n".join(
            f"{index}. {direction}"
            for index, direction in enumerate(self.directions, start=1)
        )
        return (
            f"Supervisor trigger: {self.trigger.value}. Consider these three "
            f"directions; choose, adapt, or reject them yourself:\n{options}"
        )


@dataclass(slots=True)
class _SupervisorState:
    consecutive_no_change: int = 0
    no_change_fired: bool = False
    last_death_signature: str | None = None
    repeated_deaths: int = 0
    death_fired: bool = False
    actions_since_progress: int = 0
    actions_since_evidence: int = 0
    plateau_fired: bool = False
    consecutive_zero_action_episodes: int = 0
    zero_action_fired: bool = False


class Supervisor:
    """
    Detect four documented stall conditions from authoritative run events.

    This component can only emit suggestions. It has no references to domain
    tools, evaluators, backends, budgets, or mutable run configuration.
    """

    NO_CHANGE_THRESHOLD = 3
    REPEATED_DEATH_THRESHOLD = 2
    ACTION_PLATEAU_THRESHOLD = 20
    EVIDENCE_PLATEAU_THRESHOLD = 10
    ZERO_ACTION_EPISODE_THRESHOLD = 2

    def __init__(self) -> None:
        """
        Initialize counters for a new agent scope.
        """

        self._state = _SupervisorState()

    def record_transition(
        self,
        *,
        before_hash: str,
        after_hash: str,
        outcome: str,
        death_path_signature: str | None = None,
        level_progress: bool = False,
        new_verified_evidence: bool = False,
    ) -> SupervisorRedirect | None:
        """
        Observe one counted environment transition and emit at most one redirect.
        """

        state = self._state
        if level_progress:
            state.actions_since_progress = 0
            state.plateau_fired = False
        else:
            state.actions_since_progress += 1
        if new_verified_evidence:
            state.actions_since_evidence = 0
            state.plateau_fired = False
        else:
            state.actions_since_evidence += 1

        if before_hash == after_hash:
            state.consecutive_no_change += 1
        else:
            state.consecutive_no_change = 0
            state.no_change_fired = False

        is_death = outcome.strip().upper() in {"GAME_OVER", "DEAD", "DEATH"}
        if is_death and death_path_signature:
            if death_path_signature == state.last_death_signature:
                state.repeated_deaths += 1
            else:
                state.last_death_signature = death_path_signature
                state.repeated_deaths = 1
                state.death_fired = False

        if (
            state.consecutive_no_change >= self.NO_CHANGE_THRESHOLD
            and not state.no_change_fired
        ):
            state.no_change_fired = True
            return self._redirect(SupervisorTrigger.NO_STATE_CHANGE)
        if (
            state.repeated_deaths >= self.REPEATED_DEATH_THRESHOLD
            and not state.death_fired
        ):
            state.death_fired = True
            return self._redirect(SupervisorTrigger.REPEATED_DEATH)
        if (
            state.actions_since_progress >= self.ACTION_PLATEAU_THRESHOLD
            and state.actions_since_evidence >= self.EVIDENCE_PLATEAU_THRESHOLD
            and not state.plateau_fired
        ):
            state.plateau_fired = True
            return self._redirect(SupervisorTrigger.PROGRESS_PLATEAU)
        return None

    def record_episode(self, *, action_count: int) -> SupervisorRedirect | None:
        """
        Observe a completed model episode for the zero-action stall trigger.
        """

        if (
            isinstance(action_count, bool)
            or not isinstance(action_count, int)
            or action_count < 0
        ):
            raise ValueError("episode action count must be a non-negative integer")
        state = self._state
        if action_count == 0:
            state.consecutive_zero_action_episodes += 1
        else:
            state.consecutive_zero_action_episodes = 0
            state.zero_action_fired = False
        if (
            state.consecutive_zero_action_episodes
            >= self.ZERO_ACTION_EPISODE_THRESHOLD
            and not state.zero_action_fired
        ):
            state.zero_action_fired = True
            return self._redirect(SupervisorTrigger.ZERO_ACTION_EPISODES)
        return None

    def checkpoint(self) -> dict[str, Any]:
        """
        Return JSON-compatible counter state for an atomic run checkpoint.
        """

        return asdict(self._state)

    @classmethod
    def from_checkpoint(cls, value: dict[str, Any]) -> Supervisor:
        """
        Restore counters while rejecting missing, extra, or invalid fields.
        """

        expected = set(_SupervisorState.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("supervisor checkpoint fields do not match the schema")
        instance = cls()
        state = _SupervisorState(**value)
        integer_fields = (
            state.consecutive_no_change,
            state.repeated_deaths,
            state.actions_since_progress,
            state.actions_since_evidence,
            state.consecutive_zero_action_episodes,
        )
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in integer_fields):
            raise ValueError("supervisor counters must be non-negative integers")
        boolean_fields = (
            state.no_change_fired,
            state.death_fired,
            state.plateau_fired,
            state.zero_action_fired,
        )
        if any(not isinstance(item, bool) for item in boolean_fields):
            raise ValueError("supervisor fired flags must be booleans")
        if state.last_death_signature is not None and not isinstance(
            state.last_death_signature, str
        ):
            raise ValueError("supervisor death signature must be text or null")
        instance._state = state
        return instance

    @staticmethod
    def _redirect(trigger: SupervisorTrigger) -> SupervisorRedirect:
        directions = {
            SupervisorTrigger.NO_STATE_CHANGE: (
                "Challenge the action-semantics hypothesis by probing a different legal action class.",
                "Inspect a different board region and derive a state-change invariant before acting.",
                "Replace the current repeated sequence with a minimal one-step controlled experiment.",
            ),
            SupervisorTrigger.REPEATED_DEATH: (
                "Identify the earliest shared precursor in the two death paths and avoid it.",
                "Test a defensive or timing-based action before entering the hazardous state.",
                "Seek an alternate route whose intermediate frame hashes differ from the failed path.",
            ),
            SupervisorTrigger.PROGRESS_PLATEAU: (
                "Re-evaluate the current goal hypothesis using only observed progress signals.",
                "Explore an untested interaction target rather than extending the current trajectory.",
                "Form one falsifiable rule, run its cheapest discriminating experiment, and record evidence.",
            ),
            SupervisorTrigger.ZERO_ACTION_EPISODES: (
                "Choose the safest legal information-gathering action and execute it immediately.",
                "Reduce the plan to one observable prediction followed by one action.",
                "Inspect the action schema and correct any uncertainty that prevented tool use.",
            ),
        }[trigger]
        return SupervisorRedirect(trigger=trigger, directions=directions)
