"""
Exact text observations and deterministic grid analysis helpers.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256

from .types import GRID_SIZE, ArcFrame, Grid, normalize_grid


@dataclass(frozen=True, slots=True)
class CellChange:
    """
    One exact cell transition between settled frames.
    """

    row: int
    col: int
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class Segment:
    """
    One four-connected, single-color component.
    """

    color: int
    size: int
    top: int
    left: int
    bottom: int
    right: int


def grid_sha256(grid: Grid | Sequence[Sequence[int]]) -> str:
    """
    Compute a full SHA-256 digest with a versioned canonical encoding.
    """

    normalized = normalize_grid(grid)
    payload = bytearray(b"ARDEA-ARC-GRID-v1\0")
    payload.extend((GRID_SIZE, GRID_SIZE))
    for row in normalized:
        payload.extend(row)
    return sha256(payload).hexdigest()


def _validate_bounds(top: int, left: int, bottom: int, right: int) -> None:
    if not (0 <= top < bottom <= GRID_SIZE and 0 <= left < right <= GRID_SIZE):
        raise ValueError(
            f"crop must satisfy 0 <= top < bottom <= {GRID_SIZE} and "
            f"0 <= left < right <= {GRID_SIZE}"
        )


def serialize_grid(
    grid: Grid | Sequence[Sequence[int]],
    *,
    top: int = 0,
    left: int = 0,
    bottom: int = GRID_SIZE,
    right: int = GRID_SIZE,
) -> str:
    """
    Serialize an exact grid or crop using zero-indexed row and column labels.
    """

    normalized = normalize_grid(grid)
    _validate_bounds(top, left, bottom, right)
    lines = [
        f"grid={bottom - top}x{right - left}; rows={top:02d}-{bottom - 1:02d}; "
        f"cols={left:02d}-{right - 1:02d}; encoding=hex"
    ]
    for row_index in range(top, bottom):
        cells = normalized[row_index][left:right]
        groups = [
            "".join(format(cell, "x") for cell in cells[offset : offset + 8])
            for offset in range(0, len(cells), 8)
        ]
        lines.append(f"r{row_index:02d} " + " ".join(groups))
    return "\n".join(lines)


def serialize_observation(
    frame: ArcFrame,
    *,
    action_count: int,
    exhausted: bool = False,
    diff: str | None = None,
) -> str:
    """
    Serialize progress and the authoritative full 64 by 64 text grid.

    The host action limit and human baselines are deliberately omitted.
    """

    legal = () if exhausted else frame.legal_actions
    legal_text = ", ".join(action.value for action in legal) or "none; stop"
    status = (
        f"state={frame.status.value}; levels={frame.levels_completed}/{frame.win_levels}; "
        f"actions={action_count}; legal={legal_text}"
    )
    notes = "Rows and columns are zero-indexed. Cells are hexadecimal colors 0-f."
    sections = [status]
    if diff:
        sections.append(diff)
    sections.extend((notes, serialize_grid(frame.grid)))
    return "\n".join(sections)


def diff_cells(
    before: Grid | Sequence[Sequence[int]],
    after: Grid | Sequence[Sequence[int]],
) -> tuple[CellChange, ...]:
    """
    Return every changed cell in deterministic row-major order.
    """

    left = normalize_grid(before)
    right = normalize_grid(after)
    return tuple(
        CellChange(row, col, left[row][col], right[row][col])
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
        if left[row][col] != right[row][col]
    )


def format_diff(changes: Sequence[CellChange], *, limit: int = 256) -> str:
    """
    Format exact cell changes without silently misstating omitted cells.
    """

    if limit <= 0:
        raise ValueError("diff limit must be positive")
    if not changes:
        return "diff: no cells changed"
    shown = " ".join(
        f"r{change.row:02d}c{change.col:02d}:{change.before:x}>{change.after:x}"
        for change in changes[:limit]
    )
    suffix = f" +{len(changes) - limit} more" if len(changes) > limit else ""
    return f"diff: {len(changes)} cells changed; {shown}{suffix}"


def read_pixels(
    grid: Grid | Sequence[Sequence[int]],
    coordinates: Iterable[tuple[int, int]],
    *,
    maximum: int = 256,
) -> tuple[tuple[int, int, int], ...]:
    """
    Read selected coordinates in caller order.
    """

    normalized = normalize_grid(grid)
    requested = tuple(coordinates)
    if len(requested) > maximum:
        raise ValueError(f"at most {maximum} pixels may be read per call")
    values: list[tuple[int, int, int]] = []
    for row, col in requested:
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            raise ValueError(f"pixel coordinate ({row}, {col}) is outside the 64x64 grid")
        values.append((row, col, normalized[row][col]))
    return tuple(values)


def connected_segments(
    grid: Grid | Sequence[Sequence[int]],
    *,
    include_zero: bool = False,
) -> tuple[Segment, ...]:
    """
    Find four-connected components, ordered by top-left cell then color.
    """

    normalized = normalize_grid(grid)
    visited: set[tuple[int, int]] = set()
    segments: list[Segment] = []
    for start_row in range(GRID_SIZE):
        for start_col in range(GRID_SIZE):
            if (start_row, start_col) in visited:
                continue
            color = normalized[start_row][start_col]
            if color == 0 and not include_zero:
                visited.add((start_row, start_col))
                continue
            queue = deque([(start_row, start_col)])
            visited.add((start_row, start_col))
            cells: list[tuple[int, int]] = []
            while queue:
                row, col = queue.popleft()
                cells.append((row, col))
                for neighbor_row, neighbor_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if not (0 <= neighbor_row < GRID_SIZE and 0 <= neighbor_col < GRID_SIZE):
                        continue
                    coordinate = (neighbor_row, neighbor_col)
                    if coordinate in visited or normalized[neighbor_row][neighbor_col] != color:
                        continue
                    visited.add(coordinate)
                    queue.append(coordinate)
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            segments.append(
                Segment(
                    color=color,
                    size=len(cells),
                    top=min(rows),
                    left=min(cols),
                    bottom=max(rows),
                    right=max(cols),
                )
            )
    return tuple(segments)
