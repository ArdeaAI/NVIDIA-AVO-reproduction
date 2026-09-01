"""
Tests for deterministic supervisor trigger thresholds.
"""

from ardea_avo.runtime.supervisor import Supervisor, SupervisorTrigger


def _assert_redirect(redirect, trigger: SupervisorTrigger) -> None:
    assert redirect is not None
    assert redirect.trigger is trigger
    assert len(redirect.directions) == 3
    assert len(set(redirect.directions)) == 3


def test_three_no_change_transitions_trigger_once_until_change() -> None:
    """
    Repeated no-op states emit one redirect per uninterrupted sequence.
    """

    supervisor = Supervisor()
    for _ in range(2):
        assert supervisor.record_transition(
            before_hash="same", after_hash="same", outcome="PLAYING"
        ) is None
    redirect = supervisor.record_transition(
        before_hash="same", after_hash="same", outcome="PLAYING"
    )
    _assert_redirect(redirect, SupervisorTrigger.NO_STATE_CHANGE)
    assert supervisor.record_transition(
        before_hash="same", after_hash="same", outcome="PLAYING"
    ) is None


def test_identical_death_path_triggers_on_second_death() -> None:
    """
    A repeated death signature redirects without inspecting game semantics.
    """

    supervisor = Supervisor()
    assert supervisor.record_transition(
        before_hash="a",
        after_hash="b",
        outcome="GAME_OVER",
        death_path_signature="path",
    ) is None
    redirect = supervisor.record_transition(
        before_hash="c",
        after_hash="d",
        outcome="GAME_OVER",
        death_path_signature="path",
    )
    _assert_redirect(redirect, SupervisorTrigger.REPEATED_DEATH)


def test_plateau_requires_both_action_and_evidence_thresholds() -> None:
    """
    Twenty non-progress actions trigger only when evidence also stagnates.
    """

    supervisor = Supervisor()
    redirect = None
    for index in range(20):
        redirect = supervisor.record_transition(
            before_hash=f"before-{index}",
            after_hash=f"after-{index}",
            outcome="PLAYING",
        )
    _assert_redirect(redirect, SupervisorTrigger.PROGRESS_PLATEAU)


def test_two_zero_action_episodes_trigger_and_state_round_trips() -> None:
    """
    Checkpoint restoration preserves deterministic episode counters.
    """

    supervisor = Supervisor()
    assert supervisor.record_episode(action_count=0) is None
    restored = Supervisor.from_checkpoint(supervisor.checkpoint())
    redirect = restored.record_episode(action_count=0)
    _assert_redirect(redirect, SupervisorTrigger.ZERO_ACTION_EPISODES)
    assert "choose, adapt, or reject" in redirect.as_prompt()
