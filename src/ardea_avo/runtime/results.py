"""
Immutable run manifests, hash-chained events, and atomic recovery state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from ardea_avo.runtime._io import (
    append_jsonl,
    atomic_write_json,
    canonical_json,
    file_lock,
    load_jsonl,
    sha256_json,
    utc_now,
)
from ardea_avo.runtime.budget import DEFAULT_MAX_COST_USD, BudgetLedger
from ardea_avo.runtime.lease import RunLease

_RUN_ID = re.compile(r"^[0-9]{6}-[0-9]{6}_[a-z0-9][a-z0-9_-]{0,47}$")
_SLUG_REPLACEMENT = re.compile(r"[^a-z0-9]+")
_GENESIS_HASH = "0" * 64
_SEALED_ARTIFACTS = "sealed-artifacts.json"
_SECRET_KEYS = {
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "credentials",
    "credential",
    "auth_file",
    "auth_path",
    "codex_home",
}


def _exclude_seal_artifact(relative: Path) -> bool:
    parts = relative.parts
    if relative.as_posix() in {"sealed.json", _SEALED_ARTIFACTS}:
        return True
    if len(parts) == 1 and parts[0] in {
        ".run.lock",
        ".events.lock",
        ".seal.lock",
        ".checkpoint.lock",
        ".budget.lock",
    }:
        return True
    if parts[-2:] == ("host", ".evolution.lock"):
        return True
    if parts[-3:] == ("host", "budget", ".budget.lock"):
        return True
    return (
        "provider-sessions" in parts
        and relative.name.startswith(".")
        and relative.suffix == ".lock"
    )


class EventChainError(ValueError):
    """
    Raised when durable provenance has been truncated, reordered, or changed.
    """


class RunMode(StrEnum):
    """
    Provenance lane of a run.
    """

    COLD = "cold"
    WARM = "warm"


@dataclass(frozen=True, slots=True)
class RunManifest:
    """
    Immutable original configuration and provenance for one run.
    """

    schema_version: int
    run_id: str
    mode: RunMode
    created_at: str
    parent_run_id: str | None
    backend: str
    auth_method: str
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "max"
    max_cost_usd: str = str(DEFAULT_MAX_COST_USD)
    observation_mode: str = "text"
    competition: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    parent_snapshot: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return a canonical JSON-compatible representation.
        """

        value = asdict(self)
        value["mode"] = self.mode.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        """
        Parse a manifest while rejecting missing and additional fields.
        """

        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(f"manifest fields mismatch; missing={missing}, extra={extra}")
        data = dict(value)
        data["mode"] = RunMode(data["mode"])
        manifest = cls(**data)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """
        Validate the stable run contract independently of filesystem state.
        """

        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ValueError("unsupported run manifest schema")
        if not isinstance(self.run_id, str) or not _RUN_ID.fullmatch(self.run_id):
            raise ValueError("invalid canonical run id")
        if self.mode is RunMode.COLD and self.parent_run_id is not None:
            raise ValueError("a cold run cannot have a parent")
        if self.mode is RunMode.WARM and not self.parent_run_id:
            raise ValueError("a warm run requires a parent")
        if self.mode is RunMode.WARM and not self.parent_snapshot:
            raise ValueError("a warm run requires a bound parent snapshot")
        if self.parent_run_id is not None and (
            not isinstance(self.parent_run_id, str)
            or not _RUN_ID.fullmatch(self.parent_run_id)
        ):
            raise ValueError("manifest parent run id is invalid")
        if not isinstance(self.config, Mapping) or not isinstance(
            self.provenance, Mapping
        ):
            raise ValueError("manifest config and provenance must be mappings")
        if self.parent_snapshot is not None and not isinstance(
            self.parent_snapshot, Mapping
        ):
            raise ValueError("manifest parent snapshot must be a mapping")
        if not isinstance(self.competition, bool):
            raise ValueError("manifest competition flag must be a boolean")
        identity_values = (
            self.created_at,
            self.backend,
            self.auth_method,
            self.model,
            self.reasoning_effort,
        )
        if not all(
            isinstance(value, str) and value.strip() for value in identity_values
        ):
            raise ValueError("manifest identity fields cannot be blank")
        if self.observation_mode not in {"text", "png"}:
            raise ValueError("observation mode must be text or png")
        try:
            maximum = Decimal(self.max_cost_usd)
        except Exception as error:
            raise ValueError("invalid manifest max_cost_usd") from error
        if not maximum.is_finite() or maximum <= 0:
            raise ValueError("manifest max_cost_usd must be positive and finite")
        _reject_secrets(self.to_dict())


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Validated access to one results directory.
    """

    directory: Path
    manifest: RunManifest

    @property
    def manifest_path(self) -> Path:
        """
        Return the immutable manifest path.
        """

        return self.directory / "manifest.json"

    @property
    def events_path(self) -> Path:
        """
        Return the append-only event path.
        """

        return self.directory / "events.jsonl"

    @property
    def checkpoint_path(self) -> Path:
        """
        Return the atomically replaced checkpoint path.
        """

        return self.directory / "checkpoint.json"

    @property
    def sealed_path(self) -> Path:
        """
        Return the marker that prevents further parent mutation.
        """

        return self.directory / "sealed.json"

    @property
    def is_sealed(self) -> bool:
        """
        Report whether this run has been frozen as a warm parent.
        """

        return self.sealed_path.exists()

    def append_event(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """
        Append one secret-free event to the validated hash chain.
        """

        if self.is_sealed:
            raise RuntimeError("sealed parent runs are immutable")
        if not kind.strip():
            raise ValueError("event kind cannot be blank")
        _reject_secrets(payload)
        lock_path = self.directory / ".events.lock"
        with file_lock(lock_path):
            if self.is_sealed:
                raise RuntimeError("sealed parent runs are immutable")
            return self._append_event_unlocked(kind, payload)

    def events(self) -> tuple[dict[str, Any], ...]:
        """
        Load and validate all provenance events.
        """

        with file_lock(self.directory / ".events.lock"):
            return tuple(self._load_and_validate_events())

    def head(self) -> tuple[int, str]:
        """
        Return the current event sequence and hash.
        """

        events = self.events()
        if not events:
            return -1, _GENESIS_HASH
        return int(events[-1]["sequence"]), str(events[-1]["hash"])

    def write_checkpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """
        Atomically write recovery state bound to a manifest and event prefix.
        """

        _reject_secrets(payload)
        with file_lock(self.directory / ".seal.lock"):
            if self.is_sealed:
                raise RuntimeError("sealed parent runs are immutable")
            sequence, event_hash = self.head()
            body = {
                "schema_version": 1,
                "run_id": self.manifest.run_id,
                "manifest_hash": sha256_json(self.manifest.to_dict()),
                "event_sequence": sequence,
                "event_hash": event_hash,
                "updated_at": utc_now(),
                "payload": dict(payload),
            }
            checkpoint = {**body, "checkpoint_hash": sha256_json(body)}
            with file_lock(self.directory / ".checkpoint.lock"):
                atomic_write_json(self.checkpoint_path, checkpoint)
            return checkpoint

    def read_checkpoint(self) -> dict[str, Any] | None:
        """
        Load and validate the latest atomic checkpoint and its event prefix.
        """

        if not self.checkpoint_path.exists():
            return None
        with file_lock(self.directory / ".checkpoint.lock"):
            try:
                value = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise EventChainError("checkpoint is unreadable") from error
        if not isinstance(value, dict):
            raise EventChainError("checkpoint is not a JSON object")
        checkpoint_hash = value.pop("checkpoint_hash", None)
        if checkpoint_hash != sha256_json(value):
            raise EventChainError("checkpoint hash mismatch")
        if value.get("run_id") != self.manifest.run_id:
            raise EventChainError("checkpoint belongs to a different run")
        if value.get("manifest_hash") != sha256_json(self.manifest.to_dict()):
            raise EventChainError("checkpoint manifest hash mismatch")
        sequence = value.get("event_sequence")
        expected_hash = value.get("event_hash")
        events = self.events()
        if sequence == -1:
            actual_hash = _GENESIS_HASH
        elif isinstance(sequence, int) and 0 <= sequence < len(events):
            actual_hash = events[sequence]["hash"]
        else:
            raise EventChainError("checkpoint references an unavailable event")
        if expected_hash != actual_hash:
            raise EventChainError("checkpoint event hash mismatch")
        return {**value, "checkpoint_hash": checkpoint_hash}

    def seal(self, *, reason: str = "warm child created") -> dict[str, Any]:
        """
        Freeze a completed parent and return its immutable snapshot identity.
        """

        lock_path = self.directory / ".seal.lock"
        with file_lock(lock_path):
            if self.sealed_path.exists():
                return self._read_seal()
            self._checkpoint_memory_database()
            with file_lock(self.directory / ".events.lock"):
                self._append_event_unlocked("run.sealed", {"reason": reason})
                events = self._load_and_validate_events()
                artifacts = self._artifact_manifest()
                atomic_write_json(self.directory / _SEALED_ARTIFACTS, artifacts)
                body = {
                    "schema_version": 2,
                    "run_id": self.manifest.run_id,
                    "manifest_hash": sha256_json(self.manifest.to_dict()),
                    "event_sequence": int(events[-1]["sequence"]),
                    "event_hash": str(events[-1]["hash"]),
                    "artifacts_sha256": sha256_json(artifacts),
                    "artifact_count": len(artifacts["files"]),
                    "sealed_at": utc_now(),
                }
                seal = {**body, "seal_hash": sha256_json(body)}
                atomic_write_json(self.sealed_path, seal)
                return seal

    def validate(self) -> None:
        """
        Validate manifest identity, event chain, checkpoint, and seal.
        """

        self.manifest.validate()
        expected_manifest_hash = (
            self.directory / "manifest.sha256"
        ).read_text(encoding="ascii").strip()
        if expected_manifest_hash != sha256_json(self.manifest.to_dict()):
            raise EventChainError("manifest hash mismatch")
        events = self.events()
        if not events:
            raise EventChainError("run event chain is missing its creation event")
        creation = events[0]
        expected_creation = {
            "mode": self.manifest.mode.value,
            "parent_run_id": self.manifest.parent_run_id,
            "manifest_hash": sha256_json(self.manifest.to_dict()),
        }
        if creation["kind"] != "run.created" or creation["payload"] != expected_creation:
            raise EventChainError("creation event does not match the immutable manifest")
        self.read_checkpoint()
        if self.sealed_path.exists():
            self._read_seal()

    def assert_parent_unchanged(self, parent: RunContext) -> None:
        """
        Confirm that a warm run's bound parent still matches its seal.
        """

        snapshot = self.manifest.parent_snapshot
        if snapshot is None or self.manifest.parent_run_id != parent.manifest.run_id:
            raise EventChainError("warm run does not reference this parent")
        actual = parent._read_seal()
        if dict(snapshot) != actual:
            raise EventChainError("warm parent changed after child creation")
        parent.validate()

    def _load_and_validate_events(self) -> list[dict[str, Any]]:
        records = load_jsonl(self.events_path)
        previous_hash = _GENESIS_HASH
        required = {"sequence", "timestamp", "kind", "payload", "previous_hash", "hash"}
        for sequence, record in enumerate(records):
            if set(record) != required:
                raise EventChainError(f"event {sequence} fields do not match schema")
            if record["sequence"] != sequence:
                raise EventChainError(f"event {sequence} has an invalid sequence")
            if record["previous_hash"] != previous_hash:
                raise EventChainError(f"event {sequence} has an invalid predecessor")
            body = {key: record[key] for key in required if key != "hash"}
            if record["hash"] != sha256_json(body):
                raise EventChainError(f"event {sequence} hash mismatch")
            _reject_secrets(record["payload"])
            previous_hash = record["hash"]
        return records

    def _append_event_unlocked(
        self, kind: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        events = self._load_and_validate_events()
        previous_hash = events[-1]["hash"] if events else _GENESIS_HASH
        body = {
            "sequence": len(events),
            "timestamp": utc_now(),
            "kind": kind,
            "payload": dict(payload),
            "previous_hash": previous_hash,
        }
        record = {**body, "hash": sha256_json(body)}
        append_jsonl(self.events_path, record)
        return record

    def _read_seal(self) -> dict[str, Any]:
        try:
            seal = json.loads(self.sealed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EventChainError("run seal is unreadable") from error
        if not isinstance(seal, dict):
            raise EventChainError("run seal is not a JSON object")
        body = {key: value for key, value in seal.items() if key != "seal_hash"}
        if set(body) != {
            "schema_version",
            "run_id",
            "manifest_hash",
            "event_sequence",
            "event_hash",
            "artifacts_sha256",
            "artifact_count",
            "sealed_at",
        }:
            raise EventChainError("run seal fields do not match schema")
        if seal.get("seal_hash") != sha256_json(body):
            raise EventChainError("run seal hash mismatch")
        if body["schema_version"] != 2:
            raise EventChainError("unsupported run seal schema")
        artifacts_path = self.directory / _SEALED_ARTIFACTS
        try:
            artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EventChainError("sealed artifact manifest is unreadable") from error
        if (
            not isinstance(artifacts, dict)
            or sha256_json(artifacts) != body["artifacts_sha256"]
            or body["artifact_count"] != len(artifacts.get("files", ()))
            or artifacts != self._artifact_manifest()
        ):
            raise EventChainError("sealed run artifacts changed after snapshot")
        sequence, event_hash = self.head()
        if (
            body["run_id"] != self.manifest.run_id
            or body["manifest_hash"] != sha256_json(self.manifest.to_dict())
            or body["event_sequence"] != sequence
            or body["event_hash"] != event_hash
        ):
            raise EventChainError("run seal does not match current state")
        events = self.events()
        if not events or events[-1]["kind"] != "run.sealed":
            raise EventChainError("run seal is not anchored by a final seal event")
        return seal

    def _checkpoint_memory_database(self) -> None:
        memory_path = self.directory / "memory.sqlite"
        if not memory_path.exists():
            return
        try:
            connection = sqlite3.connect(memory_path, timeout=30)
            try:
                result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise EventChainError("parent memory could not be checkpointed before sealing") from error
        if result is not None and result[0] != 0:
            raise EventChainError("parent memory is busy and cannot be sealed")

    def _artifact_manifest(self) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(
            self.directory.rglob("*"),
            key=lambda item: item.relative_to(self.directory).as_posix(),
        ):
            relative = path.relative_to(self.directory)
            if _exclude_seal_artifact(relative):
                continue
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8", errors="surrogateescape")
                files.append(
                    {
                        "path": relative.as_posix(),
                        "kind": "symlink",
                        "mode": stat.S_IMODE(path.lstat().st_mode),
                        "size": len(target),
                        "sha256": hashlib.sha256(target).hexdigest(),
                    }
                )
            elif path.is_file():
                content = path.read_bytes()
                files.append(
                    {
                        "path": relative.as_posix(),
                        "kind": "file",
                        "mode": stat.S_IMODE(path.stat().st_mode),
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
        return {
            "schema": "ardea.run-artifacts.v1",
            "run_id": self.manifest.run_id,
            "files": files,
        }


class ResultsManager:
    """
    Create and reopen canonical cold, warm, and resumed result directories.
    """

    def __init__(
        self,
        root: Path | str = "results",
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """
        Configure a results root without creating a run.
        """

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(UTC))

    def new_run_id(self, slug: str) -> str:
        """
        Generate ``YYMMDD-HHMMSS_slug`` in UTC with a sanitized slug.
        """

        normalized = _SLUG_REPLACEMENT.sub("_", slug.strip().lower()).strip("_")
        normalized = normalized[:48].rstrip("_") or "run"
        current = self.clock()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        current = current.astimezone(UTC)
        return f"{current:%y%m%d-%H%M%S}_{normalized}"

    def create_cold(
        self,
        slug: str,
        *,
        backend: str = "codex-oauth",
        auth_method: str = "chatgpt-oauth",
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "max",
        max_cost_usd: Decimal | str | float = DEFAULT_MAX_COST_USD,
        observation_mode: str = "text",
        competition: bool = False,
        config: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> RunContext:
        """
        Create a new cold run with no inherited knowledge.
        """

        return self._create(
            slug=slug,
            mode=RunMode.COLD,
            parent_run_id=None,
            parent_snapshot=None,
            backend=backend,
            auth_method=auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            max_cost_usd=max_cost_usd,
            observation_mode=observation_mode,
            competition=competition,
            config=config or {},
            provenance=provenance or {},
        )

    def create_warm(
        self,
        parent_run_id: str,
        slug: str,
        *,
        backend: str | None = None,
        auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_cost_usd: Decimal | str | float = DEFAULT_MAX_COST_USD,
        observation_mode: str | None = None,
        competition: bool = False,
        config: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> RunContext:
        """
        Seal a parent and create a new warm child bound to its exact state.
        """

        parent = self.open(parent_run_id)
        with RunLease(parent.directory):
            if (parent.directory / "config-revisions.jsonl").exists():
                active = BudgetLedger(parent.directory).active_reservations()
                if active:
                    raise RuntimeError(
                        "parent has active model reservations; resume and recover it before creating a child"
                    )
            snapshot = parent.seal()
            return self._create(
                slug=slug,
                mode=RunMode.WARM,
                parent_run_id=parent_run_id,
                parent_snapshot=snapshot,
                backend=backend or parent.manifest.backend,
                auth_method=auth_method or parent.manifest.auth_method,
                model=model or parent.manifest.model,
                reasoning_effort=reasoning_effort or parent.manifest.reasoning_effort,
                max_cost_usd=max_cost_usd,
                observation_mode=observation_mode or parent.manifest.observation_mode,
                competition=competition,
                config=config or parent.manifest.config,
                provenance=provenance or parent.manifest.provenance,
            )

    def open(self, run_id: str) -> RunContext:
        """
        Open and fully validate an existing run, sealed or active.
        """

        directory = self._directory(run_id)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run does not exist: {run_id}")
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise EventChainError("manifest is malformed") from error
        if not isinstance(value, dict):
            raise EventChainError("manifest is not a JSON object")
        context = RunContext(directory=directory, manifest=RunManifest.from_dict(value))
        if context.manifest.run_id != run_id:
            raise EventChainError("manifest run id does not match its directory")
        context.validate()
        return context

    def resume(
        self,
        run_id: str,
        *,
        expected: Mapping[str, Any] | None = None,
    ) -> RunContext:
        """
        Reopen the same active directory and validate expected provenance.
        """

        context = self.open(run_id)
        if context.is_sealed:
            raise RuntimeError("a sealed warm parent cannot be resumed")
        if expected is not None:
            actual = context.manifest.to_dict()
            _assert_expected(actual, expected, path="manifest")
        if context.manifest.mode is RunMode.WARM:
            assert context.manifest.parent_run_id is not None
            parent = self.open(context.manifest.parent_run_id)
            context.assert_parent_unchanged(parent)
        return context

    def _create(
        self,
        *,
        slug: str,
        mode: RunMode,
        parent_run_id: str | None,
        parent_snapshot: Mapping[str, Any] | None,
        backend: str,
        auth_method: str,
        model: str,
        reasoning_effort: str,
        max_cost_usd: Decimal | str | float,
        observation_mode: str,
        competition: bool,
        config: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> RunContext:
        run_id = self.new_run_id(slug)
        directory = self._directory(run_id)
        manifest = RunManifest(
            schema_version=1,
            run_id=run_id,
            mode=mode,
            created_at=utc_now(),
            parent_run_id=parent_run_id,
            backend=backend,
            auth_method=auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            max_cost_usd=str(Decimal(str(max_cost_usd))),
            observation_mode=observation_mode,
            competition=competition,
            config=dict(config),
            provenance=dict(provenance),
            parent_snapshot=(dict(parent_snapshot) if parent_snapshot else None),
        )
        manifest.validate()
        canonical_json(manifest.to_dict())
        try:
            directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"run id collision within one UTC second: {run_id}"
            ) from error
        _write_exclusive_json(directory / "manifest.json", manifest.to_dict())
        hash_path = directory / "manifest.sha256"
        hash_path.write_text(sha256_json(manifest.to_dict()) + "\n", encoding="ascii")
        os.chmod(directory / "manifest.json", 0o444)
        os.chmod(hash_path, 0o444)
        context = RunContext(directory=directory, manifest=manifest)
        context.append_event(
            "run.created",
            {
                "mode": mode.value,
                "parent_run_id": parent_run_id,
                "manifest_hash": sha256_json(manifest.to_dict()),
            },
        )
        return context

    def _directory(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run id must use YYMMDD-HHMMSS_slug format")
        directory = self.root / run_id
        if directory.parent.resolve() != self.root.resolve():
            raise ValueError("run id escapes the results root")
        if directory.is_symlink():
            raise ValueError("run directories cannot be symbolic links")
        return directory


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (canonical_json(dict(value)) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_secrets(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if (
                normalized in _SECRET_KEYS
                or normalized.endswith("_api_key")
                or normalized.endswith("_token")
                or normalized.endswith("_access_token")
                or normalized.endswith("_refresh_token")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
            ):
                raise ValueError(f"refusing to persist secret-bearing field: {path}.{key}")
            _reject_secrets(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_secrets(nested, path=f"{path}[{index}]")


def _assert_expected(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise EventChainError(f"expected mapping at {path}")
        for key, nested in expected.items():
            if key not in actual:
                raise EventChainError(f"missing expected provenance field: {path}.{key}")
            _assert_expected(actual[key], nested, path=f"{path}.{key}")
    elif actual != expected:
        raise EventChainError(f"provenance mismatch at {path}: {actual!r} != {expected!r}")
