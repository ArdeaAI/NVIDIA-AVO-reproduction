"""
Tests for generic target YAML loading.
"""

from pathlib import Path

import pytest

from ardea_avo.target_config import load_target


def test_target_paths_resolve_from_yaml(tmp_path: Path) -> None:
    """
    Resolve resources independently of the caller's working directory.
    """
    seed = tmp_path / "seed"
    seed.mkdir()
    knowledge = tmp_path / "facts.md"
    knowledge.write_text("facts", encoding="utf-8")
    config = tmp_path / "target.yaml"
    config.write_text(
        """schema_version: 1
name: toy
seed: seed
knowledge: [facts.md]
evaluator: [python, evaluator.py]
objectives:
  - name: score
    direction: maximize
""",
        encoding="utf-8",
    )

    loaded = load_target(config)

    assert loaded.seed == seed.resolve()
    assert loaded.knowledge == (knowledge.resolve(),)
    assert loaded.core_spec().metric_names == ("score",)


def test_target_rejects_missing_seed(tmp_path: Path) -> None:
    """
    Fail before starting a run when its seed is absent.
    """
    config = tmp_path / "target.yaml"
    config.write_text(
        """name: toy
seed: absent
evaluator: python evaluator.py
objectives: [{name: score}]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seed directory"):
        load_target(config)

