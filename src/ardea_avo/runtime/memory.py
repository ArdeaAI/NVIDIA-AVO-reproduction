"""
Evidence-linked persistent memory for cold and warm runs.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from ardea_avo.runtime._io import utc_now

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class MemoryStatus(StrEnum):
    """
    Epistemic state of a stored claim.
    """

    HYPOTHESIS = "hypothesis"
    VERIFIED = "verified"
    FALSIFIED = "falsified"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """
    One scoped claim with immutable provenance and mutable review status.
    """

    id: str
    scope: str
    scope_id: str | None
    claim: str
    status: MemoryStatus
    confidence: float
    evidence: tuple[str, ...]
    contradictions: tuple[str, ...]
    origin_run_id: str
    origin_model: str
    created_at: str
    supersedes: str | None = None
    approved_for_warm: bool = False
    imported_from_run: str | None = None
    origin_record_id: str | None = None
    cost_usd: str = "0"


EvidenceValidator = Callable[[str], bool]


class MemoryStore:
    """
    Store claims in SQLite and copy only approved, evidence-valid warm memory.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str,
        read_only: bool = False,
    ) -> None:
        """
        Open or create a run-bound memory database.
        """

        self.path = Path(path)
        self.run_id = run_id
        self.read_only = read_only
        if read_only:
            if not self.path.is_file() or self.path.is_symlink():
                raise ValueError("read-only memory requires an existing regular database")
            uri = self.path.resolve().as_uri() + "?mode=ro&immutable=1"
            self.connection = sqlite3.connect(uri, timeout=30, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        if not read_only:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        try:
            if read_only:
                self._validate_metadata()
            else:
                self._initialize()
        except Exception:
            self.connection.close()
            raise

    def __enter__(self) -> MemoryStore:
        """
        Return this open store for context-manager use.
        """

        return self

    def __exit__(self, *_: object) -> None:
        """
        Close the database at context-manager exit.
        """

        self.close()

    def close(self) -> None:
        """
        Close the SQLite connection.
        """

        self.connection.close()

    def add(
        self,
        *,
        scope: str,
        claim: str,
        status: MemoryStatus | str = MemoryStatus.HYPOTHESIS,
        confidence: float = 0.5,
        evidence: Iterable[str] = (),
        contradictions: Iterable[str] = (),
        origin_model: str = "gpt-5.6-sol",
        scope_id: str | None = None,
        supersedes: str | None = None,
        cost_usd: str = "0",
    ) -> MemoryRecord:
        """
        Add a new claim after validating its evidence and scope.
        """

        self._ensure_writable()
        normalized_status = MemoryStatus(status)
        evidence_tuple = tuple(evidence)
        contradictions_tuple = tuple(contradictions)
        self._validate_claim(
            scope=scope,
            claim=claim,
            status=normalized_status,
            confidence=confidence,
            evidence=evidence_tuple,
            contradictions=contradictions_tuple,
        )
        if supersedes is not None and self.get(supersedes) is None:
            raise KeyError(f"superseded memory does not exist: {supersedes}")
        record = MemoryRecord(
            id=str(uuid4()),
            scope=scope,
            scope_id=scope_id,
            claim=claim.strip(),
            status=normalized_status,
            confidence=confidence,
            evidence=evidence_tuple,
            contradictions=contradictions_tuple,
            origin_run_id=self.run_id,
            origin_model=origin_model,
            created_at=utc_now(),
            supersedes=supersedes,
            cost_usd=str(_cost(cost_usd)),
        )
        self._insert(record)
        return record

    def get(self, record_id: str) -> MemoryRecord | None:
        """
        Fetch one record by id.
        """

        row = self.connection.execute(
            "SELECT * FROM memory_records WHERE id = ?", (record_id,)
        ).fetchone()
        return self._from_row(row) if row is not None else None

    def list(
        self,
        *,
        status: MemoryStatus | str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """
        List records deterministically with optional exact filters.
        """

        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(MemoryStatus(status).value)
        if scope is not None:
            clauses.append("scope = ?")
            values.append(scope)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            values.append(scope_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM memory_records{where} ORDER BY created_at, id", values
        ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def search(self, query: str, *, limit: int = 20) -> tuple[MemoryRecord, ...]:
        """
        Find claims containing every case-insensitive query term.
        """

        terms = tuple(term for term in query.casefold().split() if term)
        if not terms:
            raise ValueError("memory search query cannot be blank")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("memory search limit must be a positive integer")
        clauses = " AND ".join("LOWER(claim) LIKE ?" for _ in terms)
        values = [f"%{term}%" for term in terms]
        rows = self.connection.execute(
            f"""
            SELECT * FROM memory_records
            WHERE {clauses}
            ORDER BY approved_for_warm DESC, confidence DESC, created_at, id
            LIMIT ?
            """,
            (*values, limit),
        ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def resolve(
        self,
        record_id: str,
        *,
        status: MemoryStatus | str,
        evidence: Iterable[str] = (),
        contradictions: Iterable[str] = (),
        confidence: float,
    ) -> MemoryRecord:
        """
        Resolve a hypothesis to verified or falsified with evidence.
        """

        self._ensure_writable()
        normalized_status = MemoryStatus(status)
        if normalized_status is MemoryStatus.HYPOTHESIS:
            raise ValueError("resolve requires verified or falsified status")
        existing = self.get(record_id)
        if existing is None:
            raise KeyError(record_id)
        if existing.status is not MemoryStatus.HYPOTHESIS:
            raise ValueError("only a hypothesis can be resolved")
        evidence_tuple = tuple(evidence)
        contradictions_tuple = tuple(contradictions)
        self._validate_claim(
            scope=existing.scope,
            claim=existing.claim,
            status=normalized_status,
            confidence=confidence,
            evidence=evidence_tuple,
            contradictions=contradictions_tuple,
        )
        with self.connection:
            self.connection.execute(
                """
                UPDATE memory_records
                SET status = ?, confidence = ?, evidence_json = ?,
                    contradictions_json = ?, approved_for_warm = 0
                WHERE id = ?
                """,
                (
                    normalized_status.value,
                    confidence,
                    json.dumps(evidence_tuple),
                    json.dumps(contradictions_tuple),
                    record_id,
                ),
            )
        result = self.get(record_id)
        assert result is not None
        return result

    def approve_for_warm(self, record_id: str) -> MemoryRecord:
        """
        Mark an evidence-backed resolved record as eligible for child runs.
        """

        self._ensure_writable()
        record = self.get(record_id)
        if record is None:
            raise KeyError(record_id)
        if record.status is MemoryStatus.HYPOTHESIS:
            raise ValueError("hypotheses cannot be approved for warm import")
        if not record.evidence and not record.contradictions:
            raise ValueError("warm memory must have linked evidence")
        with self.connection:
            self.connection.execute(
                "UPDATE memory_records SET approved_for_warm = 1 WHERE id = ?",
                (record_id,),
            )
        result = self.get(record_id)
        assert result is not None
        return result

    def import_from_parent(
        self,
        parent: MemoryStore,
        *,
        evidence_validator: EvidenceValidator,
    ) -> tuple[MemoryRecord, ...]:
        """
        Deep-copy eligible parent knowledge into this warm child.

        Action traces, sessions, hypotheses, workspaces, and live state are not
        represented in this database and therefore cannot cross this boundary.
        """

        self._ensure_writable()
        if parent.run_id == self.run_id:
            raise ValueError("a run cannot import memory from itself")
        imported: list[MemoryRecord] = []
        for source in parent.list():
            if not source.approved_for_warm:
                continue
            if source.status not in {MemoryStatus.VERIFIED, MemoryStatus.FALSIFIED}:
                continue
            references = source.evidence + source.contradictions
            if not references or not all(evidence_validator(item) for item in references):
                continue
            existing = self.connection.execute(
                """
                SELECT id FROM memory_records
                WHERE imported_from_run = ? AND origin_record_id = ?
                """,
                (parent.run_id, source.id),
            ).fetchone()
            if existing is not None:
                continue
            copied = MemoryRecord(
                id=str(uuid4()),
                scope=source.scope,
                scope_id=source.scope_id,
                claim=source.claim,
                status=source.status,
                confidence=source.confidence,
                evidence=tuple(source.evidence),
                contradictions=tuple(source.contradictions),
                origin_run_id=source.origin_run_id,
                origin_model=source.origin_model,
                created_at=utc_now(),
                supersedes=None,
                approved_for_warm=False,
                imported_from_run=parent.run_id,
                origin_record_id=source.id,
                cost_usd=source.cost_usd,
            )
            self._insert(copied)
            imported.append(copied)
        return tuple(imported)

    def _initialize(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    scope_id TEXT,
                    claim TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('hypothesis', 'verified', 'falsified')
                    ),
                    confidence REAL NOT NULL CHECK (
                        confidence >= 0.0 AND confidence <= 1.0
                    ),
                    evidence_json TEXT NOT NULL,
                    contradictions_json TEXT NOT NULL,
                    origin_run_id TEXT NOT NULL,
                    origin_model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    supersedes TEXT,
                    approved_for_warm INTEGER NOT NULL DEFAULT 0 CHECK (
                        approved_for_warm IN (0, 1)
                    ),
                    imported_from_run TEXT,
                    origin_record_id TEXT,
                    cost_usd TEXT NOT NULL,
                    FOREIGN KEY (supersedes) REFERENCES memory_records(id),
                    UNIQUE (imported_from_run, origin_record_id)
                )
                """
            )
            rows = dict(self.connection.execute("SELECT key, value FROM metadata"))
            if not rows:
                self.connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (("schema_version", "1"), ("run_id", self.run_id)),
                )
            elif rows.get("schema_version") != "1" or rows.get("run_id") != self.run_id:
                raise ValueError("memory database metadata does not match this run")

    def _validate_metadata(self) -> None:
        try:
            rows = dict(self.connection.execute("SELECT key, value FROM metadata"))
        except sqlite3.Error as error:
            raise ValueError("memory database metadata is unreadable") from error
        if rows.get("schema_version") != "1" or rows.get("run_id") != self.run_id:
            raise ValueError("memory database metadata does not match this run")

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("read-only memory stores are immutable")
        if (self.path.parent / "sealed.json").exists():
            raise RuntimeError("sealed parent memory is immutable")

    def _insert(self, record: MemoryRecord) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO memory_records (
                    id, scope, scope_id, claim, status, confidence,
                    evidence_json, contradictions_json, origin_run_id,
                    origin_model, created_at, supersedes, approved_for_warm,
                    imported_from_run, origin_record_id, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.scope,
                    record.scope_id,
                    record.claim,
                    record.status.value,
                    record.confidence,
                    json.dumps(record.evidence),
                    json.dumps(record.contradictions),
                    record.origin_run_id,
                    record.origin_model,
                    record.created_at,
                    record.supersedes,
                    int(record.approved_for_warm),
                    record.imported_from_run,
                    record.origin_record_id,
                    record.cost_usd,
                ),
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=str(row["id"]),
            scope=str(row["scope"]),
            scope_id=row["scope_id"],
            claim=str(row["claim"]),
            status=MemoryStatus(row["status"]),
            confidence=float(row["confidence"]),
            evidence=tuple(json.loads(row["evidence_json"])),
            contradictions=tuple(json.loads(row["contradictions_json"])),
            origin_run_id=str(row["origin_run_id"]),
            origin_model=str(row["origin_model"]),
            created_at=str(row["created_at"]),
            supersedes=row["supersedes"],
            approved_for_warm=bool(row["approved_for_warm"]),
            imported_from_run=row["imported_from_run"],
            origin_record_id=row["origin_record_id"],
            cost_usd=str(row["cost_usd"]),
        )

    @staticmethod
    def _validate_claim(
        *,
        scope: str,
        claim: str,
        status: MemoryStatus,
        confidence: float,
        evidence: tuple[str, ...],
        contradictions: tuple[str, ...],
    ) -> None:
        if scope not in {"run", "game", "level"}:
            raise ValueError("memory scope must be run, game, or level")
        if not claim.strip():
            raise ValueError("memory claim cannot be blank")
        if isinstance(confidence, bool) or not 0.0 <= confidence <= 1.0:
            raise ValueError("memory confidence must be between zero and one")
        for digest in evidence + contradictions:
            if not _DIGEST.fullmatch(digest):
                raise ValueError("memory evidence references must be SHA-256 digests")
        if status is MemoryStatus.VERIFIED and not evidence:
            raise ValueError("verified memory requires supporting evidence")
        if status is MemoryStatus.FALSIFIED and not contradictions:
            raise ValueError("falsified memory requires contradictory evidence")


def _cost(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("memory cost must be a decimal amount") from error
    if not result.is_finite() or result < 0:
        raise ValueError("memory cost must be finite and non-negative")
    return result
