from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ardea_avo.core import (
    LineageStateError,
    MetricObjective,
    ObjectiveDirection,
    Score,
    ScoreComparison,
    ScoreValidationError,
    TargetSpec,
    tree_digest,
)


def test_target_validates_exact_roster_and_compares_lexicographically() -> None:
    target = TargetSpec(
        name="arc-score",
        objectives=(
            MetricObjective(name="rhae", direction=ObjectiveDirection.MAXIMIZE),
            MetricObjective(name="actions", direction=ObjectiveDirection.MINIMIZE),
        ),
    )
    incumbent = Score(correct=True, metrics={"rhae": 100, "actions": 20})

    assert target.metric_names == ("rhae", "actions")
    assert (
        target.compare(Score(correct=True, metrics={"rhae": 100, "actions": 19}), incumbent)
        is ScoreComparison.IMPROVED
    )
    assert (
        target.compare(Score(correct=True, metrics={"rhae": 100, "actions": 21}), incumbent)
        is ScoreComparison.REGRESSED
    )
    assert target.compare(incumbent, incumbent) is ScoreComparison.EQUAL

    with pytest.raises(ScoreValidationError, match=r"missing=.*actions"):
        target.validate_score(Score(correct=True, metrics={"rhae": 100}))
    with pytest.raises(ScoreValidationError, match=r"extra=.*debug"):
        target.validate_score(
            Score(correct=True, metrics={"rhae": 100, "actions": 20, "debug": 1})
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1.0", None])
def test_score_rejects_non_finite_or_non_numeric_metric_values(value: object) -> None:
    with pytest.raises(ValidationError):
        Score(correct=True, metrics={"score": value})  # type: ignore[dict-item]


def test_models_forbid_unknown_fields_and_duplicate_objectives() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Score(correct=True, metrics={"score": 1}, surprise=True)  # type: ignore[call-arg]

    with pytest.raises(ValidationError, match="unique"):
        TargetSpec(
            name="duplicate",
            objectives=(MetricObjective(name="score"), MetricObjective(name="score")),
        )


def test_tree_digest_is_order_independent_and_rejects_git_entries(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "nested").mkdir()
    (first / "nested" / "b.bin").write_bytes(b"\x00\xff")
    (first / "a.txt").write_text("alpha\n", encoding="utf-8")
    (second / "a.txt").write_text("alpha\n", encoding="utf-8")
    (second / "nested").mkdir()
    (second / "nested" / "b.bin").write_bytes(b"\x00\xff")

    assert tree_digest(first) == tree_digest(second)

    (first / ".git").mkdir()
    with pytest.raises(LineageStateError, match=r"forbidden \.git entry"):
        tree_digest(first)
    (first / ".git").rmdir()

    (second / "a.txt").write_text("changed\n", encoding="utf-8")
    assert tree_digest(first) != tree_digest(second)


def test_tree_digest_records_executable_bit_and_symlink_target(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    script = root / "run"
    script.write_text("exit 0\n", encoding="utf-8")
    plain_digest = tree_digest(root)

    script.chmod(script.stat().st_mode | 0o100)
    executable_digest = tree_digest(root)
    assert executable_digest != plain_digest

    link = root / "alias"
    try:
        link.symlink_to("run")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    linked_digest = tree_digest(root)
    assert linked_digest != executable_digest

    link.unlink()
    link.symlink_to("missing")
    assert tree_digest(root) != linked_digest
    assert not os.path.exists(link)
