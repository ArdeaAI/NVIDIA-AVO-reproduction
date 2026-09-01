"""
Stable types and protocols for ARC-AGI-3 environments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from operator import index
from typing import Any, Protocol, runtime_checkable

GRID_SIZE = 64
MIN_COLOR = 0
MAX_COLOR = 15

Grid = tuple[tuple[int, ...], ...]


class ArcContractError(ValueError):
    """
    Raised when an ARC engine value violates the adapter contract.
    """


class ArcAction(StrEnum):
    """
    Actions supported by the ARC-AGI-3 public environments.
    """

    RESET = "RESET"
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
    ACTION6 = "ACTION6"
    ACTION7 = "ACTION7"

    @classmethod
    def coerce(cls, value: ArcAction | str) -> ArcAction:
        """
        Normalize an action name while rejecting unknown actions.
        """

        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            choices = ", ".join(member.name for member in cls)
            raise ArcContractError(f"unknown ARC action {value!r}; expected one of {choices}") from exc

    @property
    def uses_coordinates(self) -> bool:
        """
        Return whether this action takes a row and column.
        """

        return self is ArcAction.ACTION6


class GameStatus(StrEnum):
    """
    Canonical environment states used independently of arcengine enums.
    """

    NOT_FINISHED = "NOT_FINISHED"
    GAME_OVER = "GAME_OVER"
    WIN = "WIN"

    @classmethod
    def coerce(cls, value: GameStatus | str | Any) -> GameStatus:
        """
        Normalize common engine state representations.
        """

        if isinstance(value, cls):
            return value
        raw = getattr(value, "value", value)
        raw = getattr(raw, "name", raw)
        normalized = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "PLAYING": cls.NOT_FINISHED,
            "RUNNING": cls.NOT_FINISHED,
            "NOTFINISHED": cls.NOT_FINISHED,
            "LOSE": cls.GAME_OVER,
            "LOSS": cls.GAME_OVER,
            "WON": cls.WIN,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls[normalized]
        except KeyError as exc:
            raise ArcContractError(f"unknown ARC game state {value!r}") from exc


def normalize_grid(value: Any) -> Grid:
    """
    Convert an array-like value to an immutable exact 64 by 64 color grid.
    """

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArcContractError("grid must be a two-dimensional sequence")
    if len(value) != GRID_SIZE:
        raise ArcContractError(f"grid must have exactly {GRID_SIZE} rows; got {len(value)}")

    rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(value):
        if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
            raise ArcContractError(f"grid row {row_index} is not a sequence")
        if len(raw_row) != GRID_SIZE:
            raise ArcContractError(
                f"grid row {row_index} must have exactly {GRID_SIZE} columns; got {len(raw_row)}"
            )
        row: list[int] = []
        for column_index, raw_cell in enumerate(raw_row):
            if isinstance(raw_cell, bool):
                raise ArcContractError(f"grid cell r{row_index}c{column_index} cannot be boolean")
            try:
                cell = index(raw_cell)
            except TypeError as exc:
                raise ArcContractError(
                    f"grid cell r{row_index}c{column_index} is not an integer"
                ) from exc
            if not MIN_COLOR <= cell <= MAX_COLOR:
                raise ArcContractError(
                    f"grid cell r{row_index}c{column_index} must be an integer from "
                    f"{MIN_COLOR} through {MAX_COLOR}; got {raw_cell!r}"
                )
            row.append(cell)
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class ArcFrame:
    """
    One settled environment frame and its progress metadata.
    """

    grid: Grid
    status: GameStatus
    levels_completed: int
    win_levels: int
    available_actions: tuple[ArcAction, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = normalize_grid(self.grid)
        object.__setattr__(self, "grid", normalized)
        object.__setattr__(self, "status", GameStatus.coerce(self.status))
        if (
            isinstance(self.levels_completed, bool)
            or not isinstance(self.levels_completed, int)
            or self.levels_completed < 0
        ):
            raise ArcContractError("levels_completed must be a non-negative integer")
        if isinstance(self.win_levels, bool) or not isinstance(self.win_levels, int) or self.win_levels <= 0:
            raise ArcContractError("win_levels must be a positive integer")
        if self.levels_completed > self.win_levels:
            raise ArcContractError("levels_completed cannot exceed win_levels")
        normalized_actions: list[ArcAction] = []
        for action in self.available_actions:
            converted = ArcAction.coerce(action)
            if converted is ArcAction.RESET:
                continue
            if converted not in normalized_actions:
                normalized_actions.append(converted)
        object.__setattr__(self, "available_actions", tuple(normalized_actions))

    @property
    def legal_actions(self) -> tuple[ArcAction, ...]:
        """
        Return legal actions under the benchmark reset semantics.
        """

        if self.status is GameStatus.WIN:
            return ()
        if self.status is GameStatus.GAME_OVER:
            return (ArcAction.RESET,)
        return (ArcAction.RESET, *self.available_actions)


@dataclass(frozen=True, slots=True)
class ArcGameInfo:
    """
    Host-only metadata for one versioned ARC game.

    Baselines are intentionally excluded from the representation so accidental
    logging does not reveal them to the playing agent.
    """

    game_id: str
    baseline_actions: tuple[int, ...] = field(repr=False)
    engine_version: str = "unknown"
    environment_version: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.game_id, str) or not self.game_id.strip():
            raise ArcContractError("game_id cannot be empty")
        if not self.baseline_actions:
            raise ArcContractError("a game must have at least one human action baseline")
        for baseline in self.baseline_actions:
            if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline <= 0:
                raise ArcContractError("human action baselines must be positive integers")


@runtime_checkable
class ArcEnvironment(Protocol):
    """
    Small host-side interface required by the ARC tool runtime.
    """

    @property
    def info(self) -> ArcGameInfo:
        """
        Return host-only game metadata.
        """

        ...

    @property
    def frame(self) -> ArcFrame:
        """
        Return the latest settled frame.
        """

        ...

    def step(
        self,
        action: ArcAction,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> ArcFrame:
        """
        Commit one environment action and return its settled frame.
        """

        ...

    def close(self) -> None:
        """
        Release resources held by the environment.
        """

        ...


@runtime_checkable
class ArcEnvironmentFactory(Protocol):
    """
    Factory used for fresh-engine replay validation.
    """

    def __call__(self, game_id: str) -> ArcEnvironment:
        """
        Create a fresh environment for the exact game identifier.
        """

        ...


@dataclass(frozen=True, slots=True)
class ToolReply:
    """
    Provider-neutral result from an ARC tool call.
    """

    content: str
    counted: bool = False
    terminal: bool = False
    is_error: bool = False
