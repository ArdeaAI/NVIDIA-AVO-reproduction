"""
Verified winning-trace selection for ARC campaigns.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .trace import ReplayResult, load_trace, trace_sha256
from .types import GameStatus


class BankValidationError(RuntimeError):
    """
    Raised when a proposed or selected bank entry lacks valid evidence.
    """


@dataclass(frozen=True, slots=True)
class BankedWin:
    """
    Immutable evidence pointer for one selected winning trace.
    """

    game_id: str
    trace_path: str
    trace_sha256: str
    actions: int
    levels_completed: int
    win_levels: int
    rhae_percent: float


class CampaignBank:
    """
    Persist and select replay-verified wins without copying mutable state.
    """

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path).resolve() if manifest_path is not None else None
        self._entries: dict[str, BankedWin] = {}
        if self.manifest_path is not None and self.manifest_path.exists():
            self._load()

    def _load(self) -> None:
        assert self.manifest_path is not None
        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if document.get("schema") != "ardea.arc.campaign-bank.v1":
                raise BankValidationError("unsupported campaign bank schema")
            entries = document.get("games")
            if not isinstance(entries, dict):
                raise BankValidationError("campaign bank games must be an object")
            self._entries = {
                str(game_id): BankedWin(**value)
                for game_id, value in entries.items()
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, BankValidationError):
                raise
            raise BankValidationError("campaign bank manifest is malformed") from exc

    def _write(self) -> None:
        if self.manifest_path is None:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        document: dict[str, Any] = {
            "schema": "ardea.arc.campaign-bank.v1",
            "games": {
                game_id: asdict(entry)
                for game_id, entry in sorted(self._entries.items())
            },
        }
        temporary = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        temporary.unlink(missing_ok=True)
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(document, file, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.manifest_path)

    def selected(self, game_id: str) -> BankedWin | None:
        """
        Return the selected entry for a game, if any.
        """

        return self._entries.get(game_id)

    def entries(self) -> tuple[BankedWin, ...]:
        """
        Return all selected wins in deterministic game order.
        """

        return tuple(self._entries[key] for key in sorted(self._entries))

    def consider(
        self,
        trace_path: str | Path,
        replay: ReplayResult,
        *,
        optimize_actions: bool = False,
    ) -> bool:
        """
        Bank a first verified win or replace it with an objectively better win.

        By default a first verified win is immutable. With optimization enabled,
        higher RHAE wins first and fewer submitted actions break score ties.
        """

        path = Path(trace_path).resolve()
        data = load_trace(path)
        digest = trace_sha256(path)
        if not replay.verified or replay.trace_sha256 != digest:
            raise BankValidationError("bank candidates require replay evidence for the exact trace bytes")
        if replay.game_id != data.header.game_id:
            raise BankValidationError("replay and trace game identifiers differ")
        if replay.status is not GameStatus.WIN:
            raise BankValidationError("only WIN traces can be banked")
        if replay.levels_completed != data.header.win_levels or replay.win_levels != data.header.win_levels:
            raise BankValidationError("WIN trace does not prove completion of every game level")
        if replay.actions != len(data.steps):
            raise BankValidationError("replay action count does not match the trace")
        candidate = BankedWin(
            game_id=data.header.game_id,
            trace_path=str(path),
            trace_sha256=digest,
            actions=len(data.steps),
            levels_completed=replay.levels_completed,
            win_levels=replay.win_levels,
            rhae_percent=replay.rhae_percent,
        )
        current = self._entries.get(candidate.game_id)
        if current is not None:
            if not optimize_actions:
                return False
            current_rank = (current.rhae_percent, -current.actions)
            candidate_rank = (candidate.rhae_percent, -candidate.actions)
            if candidate_rank <= current_rank:
                return False
        self._entries[candidate.game_id] = candidate
        self._write()
        return True

    def validate_selected(self, game_id: str) -> BankedWin:
        """
        Recheck that a selected trace still has its recorded byte digest.
        """

        entry = self._entries.get(game_id)
        if entry is None:
            raise BankValidationError(f"game {game_id!r} has no selected win")
        if trace_sha256(entry.trace_path) != entry.trace_sha256:
            raise BankValidationError(f"banked trace for {game_id!r} changed after selection")
        data = load_trace(entry.trace_path)
        if data.header.game_id != game_id or not data.steps or data.steps[-1].status is not GameStatus.WIN:
            raise BankValidationError(f"banked trace for {game_id!r} is no longer a complete WIN")
        return entry
