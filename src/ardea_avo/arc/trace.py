"""
Append-only action traces, hash chains, and fresh-engine replay.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from .observation import grid_sha256
from .scoring import action_budget, actions_per_completed_level, baseline_sha256, score_game
from .types import ArcAction, ArcEnvironment, ArcEnvironmentFactory, GameStatus

TRACE_SCHEMA = "ardea.arc.action-trace.v1"
_HEX_DIGITS = frozenset("0123456789abcdef")


class TraceIntegrityError(RuntimeError):
    """
    Raised when a trace is malformed, truncated, reordered, or altered.
    """


class ReplayDivergence(RuntimeError):
    """
    Raised when a fresh engine does not reproduce a recorded transition.
    """


def _canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _record_digest(record: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_hash"}
    return sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= _HEX_DIGITS


@dataclass(frozen=True, slots=True)
class TraceHeader:
    """
    Immutable provenance and private scoring data for one action trace.
    """

    game_id: str
    engine_version: str
    environment_version: str
    win_levels: int
    baseline_digest: str
    initial_status: GameStatus
    initial_levels_completed: int
    initial_frame_hash: str
    schema: str = TRACE_SCHEMA
    record_hash: str = ""

    def to_record(self) -> dict[str, Any]:
        """
        Return the canonical JSON record, including its integrity digest.
        """

        record: dict[str, Any] = {
            "kind": "header",
            "schema": self.schema,
            "game_id": self.game_id,
            "engine_version": self.engine_version,
            "environment_version": self.environment_version,
            "win_levels": self.win_levels,
            "baseline_digest": self.baseline_digest,
            "initial_status": self.initial_status.value,
            "initial_levels_completed": self.initial_levels_completed,
            "initial_frame_hash": self.initial_frame_hash,
        }
        record["record_hash"] = _record_digest(record)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TraceHeader:
        """
        Validate and parse a trace header.
        """

        try:
            header = cls(
                schema=str(record["schema"]),
                game_id=str(record["game_id"]),
                engine_version=str(record["engine_version"]),
                environment_version=str(record["environment_version"]),
                win_levels=int(record["win_levels"]),
                baseline_digest=str(record["baseline_digest"]),
                initial_status=GameStatus.coerce(record["initial_status"]),
                initial_levels_completed=int(record["initial_levels_completed"]),
                initial_frame_hash=str(record["initial_frame_hash"]),
                record_hash=str(record["record_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceIntegrityError("trace header is missing or contains invalid fields") from exc
        if record.get("kind") != "header" or header.schema != TRACE_SCHEMA:
            raise TraceIntegrityError("unsupported trace header or schema")
        if _record_digest(record) != header.record_hash:
            raise TraceIntegrityError("trace header hash does not match its content")
        if header.win_levels <= 0:
            raise TraceIntegrityError("trace header level metadata is inconsistent")
        if not _is_sha256(header.baseline_digest):
            raise TraceIntegrityError("trace baseline metadata digest is invalid")
        if header.initial_levels_completed != 0 or header.initial_status is not GameStatus.NOT_FINISHED:
            raise TraceIntegrityError("a qualifying trace must begin at fresh game progress")
        if not _is_sha256(header.initial_frame_hash):
            raise TraceIntegrityError("trace initial frame digest is invalid")
        return header


@dataclass(frozen=True, slots=True)
class TraceStep:
    """
    One committed action and the exact settled result that followed it.
    """

    number: int
    action: ArcAction
    row: int | None
    col: int | None
    expected_outcome: str | None
    status: GameStatus
    levels_completed: int
    frame_hash: str
    previous_record_hash: str
    record_hash: str = ""

    def to_record(self) -> dict[str, Any]:
        """
        Return the canonical JSON record, including its chain digest.
        """

        record: dict[str, Any] = {
            "kind": "action",
            "number": self.number,
            "action": self.action.value,
            "row": self.row,
            "col": self.col,
            "expected_outcome": self.expected_outcome,
            "status": self.status.value,
            "levels_completed": self.levels_completed,
            "frame_hash": self.frame_hash,
            "previous_record_hash": self.previous_record_hash,
        }
        record["record_hash"] = _record_digest(record)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TraceStep:
        """
        Validate and parse one action record.
        """

        try:
            step = cls(
                number=int(record["number"]),
                action=ArcAction.coerce(record["action"]),
                row=None if record["row"] is None else int(record["row"]),
                col=None if record["col"] is None else int(record["col"]),
                expected_outcome=(
                    None if record["expected_outcome"] is None else str(record["expected_outcome"])
                ),
                status=GameStatus.coerce(record["status"]),
                levels_completed=int(record["levels_completed"]),
                frame_hash=str(record["frame_hash"]),
                previous_record_hash=str(record["previous_record_hash"]),
                record_hash=str(record["record_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceIntegrityError("trace action is missing or contains invalid fields") from exc
        if record.get("kind") != "action" or _record_digest(record) != step.record_hash:
            raise TraceIntegrityError("trace action hash does not match its content")
        if step.number <= 0 or step.levels_completed < 0 or not _is_sha256(step.frame_hash):
            raise TraceIntegrityError("trace action contains invalid progress metadata")
        if not _is_sha256(step.previous_record_hash) or not _is_sha256(step.record_hash):
            raise TraceIntegrityError("trace action contains invalid receipt digests")
        if step.action.uses_coordinates:
            if step.row is None or step.col is None or not (0 <= step.row < 64 and 0 <= step.col < 64):
                raise TraceIntegrityError("ACTION6 trace records require valid row and column values")
        elif step.row is not None or step.col is not None:
            raise TraceIntegrityError("only ACTION6 trace records may contain coordinates")
        return step


@dataclass(frozen=True, slots=True)
class TraceData:
    """
    One fully validated trace file.
    """

    header: TraceHeader
    steps: tuple[TraceStep, ...]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """
    Successful fresh-engine replay evidence.
    """

    verified: bool
    game_id: str
    status: GameStatus
    levels_completed: int
    win_levels: int
    actions: int
    per_level_actions: tuple[int, ...]
    rhae_percent: float
    final_frame_hash: str
    trace_sha256: str


class ActionTraceWriter:
    """
    Exclusive append-only JSONL writer with a per-record SHA-256 chain.
    """

    def __init__(self, path: str | Path, header: TraceHeader) -> None:
        header_record = header.to_record()
        validated_header = TraceHeader.from_record(header_record)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        self.header = validated_header
        self._last_hash = self.header.record_hash
        self._next_number = 1
        self._closed = False
        self._write_record(header_record)

    @classmethod
    def resume(cls, path: str | Path) -> ActionTraceWriter:
        """
        Reopen a fully validated existing trace without rewriting prior bytes.
        """

        trace_path = Path(path)
        data = load_trace(trace_path)
        writer = cls.__new__(cls)
        writer.path = trace_path
        writer.header = data.header
        writer._last_hash = data.steps[-1].record_hash if data.steps else data.header.record_hash
        writer._next_number = len(data.steps) + 1
        writer._closed = False
        writer._file = trace_path.open("a", encoding="utf-8")
        return writer

    def _write_record(self, record: Mapping[str, Any]) -> None:
        self._file.write(_canonical_json(record) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def append(
        self,
        *,
        action: ArcAction,
        row: int | None,
        col: int | None,
        expected_outcome: str | None,
        status: GameStatus,
        levels_completed: int,
        frame_hash: str,
    ) -> TraceStep:
        """
        Append one committed transition and return its signed record.
        """

        if self._closed:
            raise RuntimeError("cannot append to a closed action trace")
        unsigned = TraceStep(
            number=self._next_number,
            action=ArcAction.coerce(action),
            row=row,
            col=col,
            expected_outcome=expected_outcome,
            status=GameStatus.coerce(status),
            levels_completed=levels_completed,
            frame_hash=frame_hash,
            previous_record_hash=self._last_hash,
        )
        record = unsigned.to_record()
        step = replace(unsigned, record_hash=str(record["record_hash"]))
        TraceStep.from_record(record)
        self._write_record(record)
        self._last_hash = step.record_hash
        self._next_number += 1
        return step

    def close(self) -> None:
        """
        Flush and close the trace file.
        """

        if not self._closed:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._closed = True

    def __enter__(self) -> ActionTraceWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_trace(path: str | Path) -> TraceData:
    """
    Read and validate every JSONL record, refusing torn or blank lines.
    """

    trace_path = Path(path)
    try:
        raw = trace_path.read_bytes()
    except OSError as exc:
        raise TraceIntegrityError(f"cannot read trace {trace_path}") from exc
    if not raw or not raw.endswith(b"\n"):
        raise TraceIntegrityError("trace is empty or has a torn final record")
    lines = raw.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise TraceIntegrityError("trace contains an empty record")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TraceIntegrityError(f"trace record {index} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise TraceIntegrityError(f"trace record {index} is not an object")
        records.append(value)

    header = TraceHeader.from_record(records[0])
    previous_hash = header.record_hash
    steps: list[TraceStep] = []
    previous_levels = header.initial_levels_completed
    for expected_number, record in enumerate(records[1:], start=1):
        step = TraceStep.from_record(record)
        if step.number != expected_number:
            raise TraceIntegrityError("trace action sequence is not contiguous")
        if step.previous_record_hash != previous_hash:
            raise TraceIntegrityError("trace action chain is broken")
        if step.levels_completed < previous_levels or step.levels_completed > header.win_levels:
            raise TraceIntegrityError("trace level progress is invalid")
        if steps and steps[-1].status is GameStatus.WIN:
            raise TraceIntegrityError("trace contains actions after a WIN state")
        steps.append(step)
        previous_hash = step.record_hash
        previous_levels = step.levels_completed
    return TraceData(header=header, steps=tuple(steps))


def trace_sha256(path: str | Path) -> str:
    """
    Hash the exact immutable trace bytes for provenance and banking.
    """

    return sha256(Path(path).read_bytes()).hexdigest()


def replay_into_environment(data: TraceData, environment: ArcEnvironment) -> tuple[Any, ...]:
    """
    Replay validated records into a supplied fresh environment.

    The returned tuple contains the initial frame followed by every settled
    action frame, which also supports exact resume reconstruction.
    """

    if environment.info.game_id != data.header.game_id:
        raise ReplayDivergence(
            f"trace game {data.header.game_id!r} does not match environment {environment.info.game_id!r}"
        )
    if baseline_sha256(environment.info.baseline_actions) != data.header.baseline_digest:
        raise ReplayDivergence("human-baseline metadata differs from the recorded game version")
    if (
        data.header.engine_version != "unknown"
        and environment.info.engine_version != "unknown"
        and environment.info.engine_version != data.header.engine_version
    ):
        raise ReplayDivergence(
            f"engine version differs: recorded={data.header.engine_version}, "
            f"current={environment.info.engine_version}"
        )
    if (
        data.header.environment_version != "unknown"
        and environment.info.environment_version != "unknown"
        and environment.info.environment_version != data.header.environment_version
    ):
        raise ReplayDivergence(
            f"environment version differs: recorded={data.header.environment_version}, "
            f"current={environment.info.environment_version}"
        )
    initial = environment.frame
    initial_actual = (
        initial.status,
        initial.levels_completed,
        initial.win_levels,
        grid_sha256(initial.grid),
    )
    initial_expected = (
        data.header.initial_status,
        data.header.initial_levels_completed,
        data.header.win_levels,
        data.header.initial_frame_hash,
    )
    if initial_actual != initial_expected:
        raise ReplayDivergence(
            f"initial frame diverged: actual={initial_actual!r}, expected={initial_expected!r}"
        )
    frames = [initial]
    for step in data.steps:
        if step.action not in environment.frame.legal_actions:
            raise ReplayDivergence(
                f"trace action {step.number} is illegal in the replayed frame"
            )
        frame = environment.step(step.action, row=step.row, col=step.col)
        actual = (frame.status, frame.levels_completed, grid_sha256(frame.grid))
        expected = (step.status, step.levels_completed, step.frame_hash)
        if actual != expected:
            raise ReplayDivergence(
                f"replay diverged at action {step.number}: actual={actual!r}, expected={expected!r}"
            )
        frames.append(frame)
    return tuple(frames)


def validate_replay(path: str | Path, factory: ArcEnvironmentFactory) -> ReplayResult:
    """
    Validate a trace and every transition against a fresh environment.
    """

    data = load_trace(path)
    environment = factory(data.header.game_id)
    try:
        if len(data.steps) > action_budget(environment.info.baseline_actions):
            raise TraceIntegrityError("trace exceeds the official per-game action limit")
        frames = replay_into_environment(data, environment)
        final = frames[-1]
        per_level_actions = actions_per_completed_level(data.steps)
        rhae_percent = score_game(environment.info.baseline_actions, per_level_actions).percent
        return ReplayResult(
            verified=True,
            game_id=data.header.game_id,
            status=final.status,
            levels_completed=final.levels_completed,
            win_levels=final.win_levels,
            actions=len(data.steps),
            per_level_actions=per_level_actions,
            rhae_percent=rhae_percent,
            final_frame_hash=grid_sha256(final.grid),
            trace_sha256=trace_sha256(path),
        )
    finally:
        environment.close()
