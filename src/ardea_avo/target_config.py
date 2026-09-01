"""
File-backed configuration for generic AVO evolution targets.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ardea_avo.core import MetricObjective, TargetSpec


class TargetFile(BaseModel):
    """
    Complete on-disk contract for a generic evolution target.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    name: str
    seed: Path
    evaluator: tuple[str, ...]
    check: tuple[str, ...] = ()
    objectives: tuple[MetricObjective, ...]
    knowledge: tuple[Path, ...] = ()
    evaluator_timeout_seconds: float = Field(default=300.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evaluator", mode="before")
    @classmethod
    def parse_evaluator(cls, value: Any) -> Any:
        """
        Accept either explicit argv or a shell-like string parsed without a shell.
        """
        if isinstance(value, str):
            return tuple(shlex.split(value))
        return value

    @field_validator("check", mode="before")
    @classmethod
    def parse_check(cls, value: Any) -> Any:
        """
        Parse an optional fixed candidate-side validation command.
        """
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(shlex.split(value))
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> TargetFile:
        """
        Reject target files that cannot safely initialize a run.
        """
        if self.schema_version != 1:
            raise ValueError("unsupported target schema_version")
        if not self.evaluator or any(not item for item in self.evaluator):
            raise ValueError("evaluator must contain non-empty argv strings")
        if any(not item for item in self.check):
            raise ValueError("check must contain only non-empty argv strings")
        if not self.objectives:
            raise ValueError("at least one objective is required")
        return self

    def resolve(self, source: Path) -> TargetFile:
        """
        Resolve relative resource paths against the YAML file directory.
        """
        base = source.resolve().parent

        def absolute(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "seed": absolute(self.seed),
                "knowledge": tuple(absolute(path) for path in self.knowledge),
            }
        )

    def core_spec(self) -> TargetSpec:
        """
        Return the immutable comparison portion consumed by EvolutionEngine.
        """
        return TargetSpec(name=self.name, objectives=self.objectives, metadata=self.metadata)


def load_target(
    path: str | Path,
    *,
    require_inputs: bool = True,
) -> TargetFile:
    """
    Load one strict YAML target and optionally validate its external inputs.
    """
    source = Path(path).resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read target file {source}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("target YAML must contain an object")
    target = TargetFile.model_validate(document).resolve(source)
    if not require_inputs:
        return target
    if not target.seed.is_dir():
        raise ValueError(f"target seed directory does not exist: {target.seed}")
    missing = [path for path in target.knowledge if not path.exists()]
    if missing:
        raise ValueError("target knowledge path does not exist: " + ", ".join(map(str, missing)))
    return target
