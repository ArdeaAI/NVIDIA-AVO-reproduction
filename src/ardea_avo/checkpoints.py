"""
Atomic checkpoint adapter for the generic evolution engine.
"""

from __future__ import annotations

import json
from pathlib import Path

from ardea_avo.core import EngineState, StepRecord
from ardea_avo.runtime._io import append_jsonl, atomic_write_json, file_lock


class JsonCheckpointStore:
    """
    Persist a complete engine state plus append-only step records.
    """

    def __init__(self, directory: str | Path) -> None:
        """
        Bind checkpoint files to a run-owned directory.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "evolution-checkpoint.json"
        self.steps_path = self.directory / "evolution-steps.jsonl"
        self.lock_path = self.directory / ".evolution.lock"

    def load(self) -> EngineState | None:
        """
        Load the latest state or return None before seed initialization.
        """
        with file_lock(self.lock_path):
            if not self.state_path.exists():
                return None
            try:
                value = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("evolution checkpoint is unreadable") from error
            return EngineState.model_validate(value)

    def save(self, state: EngineState) -> None:
        """
        Atomically replace the complete recovery state.
        """
        with file_lock(self.lock_path):
            atomic_write_json(self.state_path, state.model_dump(mode="json"))

    def append(self, record: StepRecord) -> None:
        """
        Append one durable attempt record before publishing state.
        """
        with file_lock(self.lock_path):
            append_jsonl(self.steps_path, record.model_dump(mode="json"))

