"""
Small durable-I/O helpers shared by runtime services.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for developer use.
    fcntl = None  # type: ignore[assignment]


def utc_now() -> str:
    """
    Return a stable, timezone-qualified UTC timestamp.
    """

    return datetime.now(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    """
    Serialize a value deterministically for hashing and JSONL storage.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """
    Hash a JSON-compatible value using its canonical representation.
    """

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """
    Hold an advisory exclusive lock associated with ``path``.

    The locked file contains no application data, so locking never exposes or
    rewrites an append-only ledger.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """
    Append one canonical JSON record and durably flush it.

    Callers are responsible for taking the corresponding file lock.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(dict(value)) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file, rejecting blank or malformed records.
    """

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed JSONL record at {path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"non-object JSONL record at {path}:{line_number}")
            records.append(record)
    return records


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """
    Atomically replace a JSON file and fsync both file and parent directory.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(dict(value)))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
