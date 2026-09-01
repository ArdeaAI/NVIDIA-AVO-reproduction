"""
Stateful ARC tool runtime with one counted environment operation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .observation import (
    connected_segments,
    diff_cells,
    format_diff,
    grid_sha256,
    serialize_grid,
    serialize_observation,
)
from .observation import read_pixels as read_grid_pixels
from .scoring import action_budget, baseline_sha256
from .trace import (
    ActionTraceWriter,
    TraceHeader,
    TraceStep,
    load_trace,
    replay_into_environment,
)
from .types import ArcAction, ArcEnvironment, ArcFrame, GameStatus, ToolReply


class ActionCommitError(RuntimeError):
    """
    Raised after an engine step when durable trace status is uncertain.
    """


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """
    Host-side record of one committed action and settled frame.
    """

    turn: int
    action: ArcAction | None
    row: int | None
    col: int | None
    frame: ArcFrame
    frame_hash: str


class ArcToolRuntime:
    """
    Own an ARC environment, its private budget, history, and action trace.
    """

    def __init__(
        self,
        environment: ArcEnvironment,
        *,
        trace_path: str | Path | None = None,
        hard_action_cap: int = 2_500,
        episode_per_level: bool = True,
    ) -> None:
        self._environment = environment
        self._action_limit = action_budget(environment.info.baseline_actions, hard_cap=hard_action_cap)
        self._episode_per_level = episode_per_level
        self._episode_boundary = False
        self._action_count = 0
        self._fatal_error: str | None = None
        initial = environment.frame
        if len(environment.info.baseline_actions) != initial.win_levels:
            raise ValueError("private baseline metadata does not match the environment level count")
        self._history: list[HistoryEntry] = [
            HistoryEntry(
                turn=0,
                action=None,
                row=None,
                col=None,
                frame=initial,
                frame_hash=grid_sha256(initial.grid),
            )
        ]
        self._steps: list[TraceStep] = []
        self._writer: ActionTraceWriter | None = None
        if trace_path is not None:
            info = environment.info
            header = TraceHeader(
                game_id=info.game_id,
                engine_version=info.engine_version,
                environment_version=info.environment_version,
                win_levels=initial.win_levels,
                baseline_digest=baseline_sha256(info.baseline_actions),
                initial_status=initial.status,
                initial_levels_completed=initial.levels_completed,
                initial_frame_hash=grid_sha256(initial.grid),
            )
            self._writer = ActionTraceWriter(trace_path, header)

    @classmethod
    def resume(
        cls,
        environment: ArcEnvironment,
        trace_path: str | Path,
        *,
        hard_action_cap: int = 2_500,
        episode_per_level: bool = True,
    ) -> ArcToolRuntime:
        """
        Rebuild exact state by validating and replaying an existing trace.
        """

        data = load_trace(trace_path)
        frames = replay_into_environment(data, environment)
        runtime = cls.__new__(cls)
        runtime._environment = environment
        if len(environment.info.baseline_actions) != environment.frame.win_levels:
            raise ValueError("private baseline metadata does not match the environment level count")
        runtime._action_limit = action_budget(
            environment.info.baseline_actions,
            hard_cap=hard_action_cap,
        )
        runtime._episode_per_level = episode_per_level
        runtime._episode_boundary = False
        runtime._action_count = len(data.steps)
        if runtime._action_count > runtime._action_limit:
            raise ActionCommitError("recorded actions exceed the configured resume limit")
        runtime._fatal_error = None
        runtime._steps = list(data.steps)
        runtime._history = [
            HistoryEntry(
                turn=0,
                action=None,
                row=None,
                col=None,
                frame=frames[0],
                frame_hash=grid_sha256(frames[0].grid),
            )
        ]
        for step, frame in zip(data.steps, frames[1:], strict=True):
            runtime._history.append(
                HistoryEntry(
                    turn=step.number,
                    action=step.action,
                    row=step.row,
                    col=step.col,
                    frame=frame,
                    frame_hash=step.frame_hash,
                )
            )
        runtime._writer = ActionTraceWriter.resume(trace_path)
        return runtime

    @property
    def game_id(self) -> str:
        """
        Return the full versioned game identifier.
        """

        return self._environment.info.game_id

    @property
    def frame(self) -> ArcFrame:
        """
        Return the latest settled frame.
        """

        return self._history[-1].frame

    @property
    def action_count(self) -> int:
        """
        Return committed environment actions.
        """

        return self._action_count

    @property
    def action_limit(self) -> int:
        """
        Return the private host action limit.

        This property is for the orchestrator and must not be placed in agent
        context because it is derived from the hidden human baselines.
        """

        return self._action_limit

    @property
    def exhausted(self) -> bool:
        """
        Return whether no more environment actions may be committed.
        """

        return self._action_count >= self._action_limit

    @property
    def evidence_hashes(self) -> frozenset[str]:
        """
        Return frame and receipt digests admissible as memory evidence.
        """

        frame_hashes = {entry.frame_hash for entry in self._history}
        receipt_hashes = {step.record_hash for step in self._steps}
        return frozenset(frame_hashes | receipt_hashes)

    def _select_turn(self, turn: int | None) -> HistoryEntry:
        selected = self._action_count if turn is None else turn
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise ValueError("turn must be an integer")
        if not 0 <= selected <= self._action_count:
            raise ValueError(f"turn must be between 0 and {self._action_count}")
        return self._history[selected]

    def observe(self) -> str:
        """
        Return the current exact observation without consuming an action.
        """

        return serialize_observation(
            self.frame,
            action_count=self._action_count,
            exhausted=self.exhausted,
        )

    def inspect(
        self,
        *,
        turn: int | None = None,
        top: int = 0,
        left: int = 0,
        bottom: int = 64,
        right: int = 64,
    ) -> str:
        """
        Return an exact historical grid crop without consuming an action.
        """

        entry = self._select_turn(turn)
        return (
            f"turn={entry.turn}; state={entry.frame.status.value}; "
            f"levels={entry.frame.levels_completed}/{entry.frame.win_levels}; hash={entry.frame_hash}\n"
            + serialize_grid(entry.frame.grid, top=top, left=left, bottom=bottom, right=right)
        )

    def read_pixels(
        self,
        coordinates: Iterable[tuple[int, int]],
        *,
        turn: int | None = None,
    ) -> str:
        """
        Return exact values for selected historical coordinates.
        """

        entry = self._select_turn(turn)
        values = read_grid_pixels(entry.frame.grid, coordinates)
        rendered = " ".join(f"r{row:02d}c{col:02d}={value:x}" for row, col, value in values)
        return f"turn={entry.turn}; pixels={rendered or 'none requested'}"

    def history(self, *, last: int = 30) -> str:
        """
        Return compact action and hash history without frame payloads.
        """

        if isinstance(last, bool) or not isinstance(last, int) or not 1 <= last <= 1_000:
            raise ValueError("last must be an integer from 1 through 1000")
        entries = self._history[1:][-last:]
        if not entries:
            return "history: no committed actions"
        lines = []
        for entry in entries:
            coordinate = (
                f"(r{entry.row:02d},c{entry.col:02d})"
                if entry.action is ArcAction.ACTION6 and entry.row is not None and entry.col is not None
                else ""
            )
            lines.append(
                f"turn={entry.turn}; action={entry.action.value if entry.action else 'none'}{coordinate}; "
                f"state={entry.frame.status.value}; levels={entry.frame.levels_completed}; hash={entry.frame_hash}"
            )
        return "\n".join(lines)

    def diff(
        self,
        *,
        before_turn: int | None = None,
        after_turn: int | None = None,
        limit: int = 256,
    ) -> str:
        """
        Return exact changes between two historical settled frames.
        """

        after = self._select_turn(after_turn)
        if before_turn is None:
            before_turn = max(0, after.turn - 1)
        before = self._select_turn(before_turn)
        if before.turn > after.turn:
            raise ValueError("before_turn cannot be later than after_turn")
        return f"turns={before.turn}->{after.turn}; " + format_diff(
            diff_cells(before.frame.grid, after.frame.grid),
            limit=limit,
        )

    def segments(
        self,
        *,
        turn: int | None = None,
        include_zero: bool = False,
        limit: int = 128,
    ) -> str:
        """
        Return four-connected component summaries for a settled frame.
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("segment limit must be a positive integer")
        entry = self._select_turn(turn)
        found = connected_segments(entry.frame.grid, include_zero=include_zero)
        payload = [
            {
                "color": segment.color,
                "size": segment.size,
                "top": segment.top,
                "left": segment.left,
                "bottom": segment.bottom,
                "right": segment.right,
            }
            for segment in found[:limit]
        ]
        return json.dumps(
            {
                "turn": entry.turn,
                "count": len(found),
                "returned": len(payload),
                "segments": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _invalid(self, message: str) -> ToolReply:
        return ToolReply(content=message, counted=False, terminal=False, is_error=True)

    def play(
        self,
        action: ArcAction | str,
        *,
        row: int | None = None,
        col: int | None = None,
        expected_outcome: str | None = None,
    ) -> ToolReply:
        """
        Commit the sole counted tool operation after strict validation.
        """

        if self._fatal_error is not None:
            return self._invalid(f"play is disabled after a trace commit failure: {self._fatal_error}")
        if self._episode_boundary:
            return ToolReply(
                "level complete; stop this agent episode so the host can start a fresh level context",
                terminal=True,
                is_error=True,
            )
        if self.frame.status is GameStatus.WIN:
            return ToolReply("state=WIN; the game is complete; stop", terminal=True, is_error=True)
        if self.exhausted:
            return ToolReply("action budget exhausted; stop", terminal=True, is_error=True)
        try:
            normalized = ArcAction.coerce(action)
        except ValueError as exc:
            return self._invalid(str(exc))
        if normalized not in self.frame.legal_actions:
            legal = ", ".join(action.value for action in self.frame.legal_actions)
            return self._invalid(f"illegal action {normalized.value}; legal actions: {legal or 'none'}")
        if normalized.uses_coordinates:
            if (
                isinstance(row, bool)
                or isinstance(col, bool)
                or not isinstance(row, int)
                or not isinstance(col, int)
                or not (0 <= row < 64 and 0 <= col < 64)
            ):
                return self._invalid("ACTION6 requires integer row and col between 0 and 63")
        elif row is not None or col is not None:
            return self._invalid("row and col are valid only for ACTION6")
        if expected_outcome is not None and (
            not isinstance(expected_outcome, str) or len(expected_outcome) > 1_000
        ):
            return self._invalid("expected_outcome must be a string of at most 1000 characters")

        prior = self.frame
        levels_before = prior.levels_completed
        try:
            settled = self._environment.step(normalized, row=row, col=col)
        except Exception as exc:
            self._fatal_error = f"{type(exc).__name__}: {exc}"
            raise ActionCommitError(
                "the engine step failed and commit status is unknown; abort and replay the durable prefix"
            ) from exc

        self._action_count += 1
        settled_hash = grid_sha256(settled.grid)
        self._history.append(
            HistoryEntry(
                turn=self._action_count,
                action=normalized,
                row=row,
                col=col,
                frame=settled,
                frame_hash=settled_hash,
            )
        )
        try:
            if self._writer is not None:
                step = self._writer.append(
                    action=normalized,
                    row=row,
                    col=col,
                    expected_outcome=expected_outcome,
                    status=settled.status,
                    levels_completed=settled.levels_completed,
                    frame_hash=settled_hash,
                )
                self._steps.append(step)
        except Exception as exc:
            self._fatal_error = f"{type(exc).__name__}: {exc}"
            raise ActionCommitError(
                "the engine advanced but its trace could not be committed; abort and replay the durable prefix"
            ) from exc

        changes = format_diff(diff_cells(prior.grid, settled.grid))
        terminal = settled.status is GameStatus.WIN or self.exhausted
        if self._episode_per_level and settled.levels_completed > levels_before:
            self._episode_boundary = True
            terminal = True
            changes += "; episode_boundary=level_complete; stop this agent episode"
        return ToolReply(
            content=serialize_observation(
                settled,
                action_count=self._action_count,
                exhausted=self.exhausted,
                diff=changes,
            ),
            counted=True,
            terminal=terminal,
        )

    def close(self) -> None:
        """
        Flush the trace and close the environment.
        """

        if self._writer is not None:
            self._writer.close()
        self._environment.close()

    def __enter__(self) -> ArcToolRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def coordinates_from_json(value: Any) -> tuple[tuple[int, int], ...]:
    """
    Parse MCP coordinate objects or two-element arrays.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("coordinates must be an array")
    parsed: list[tuple[int, int]] = []
    for coordinate in value:
        if isinstance(coordinate, Mapping):
            row, col = coordinate.get("row"), coordinate.get("col")
        elif isinstance(coordinate, Sequence) and not isinstance(coordinate, (str, bytes)) and len(coordinate) == 2:
            row, col = coordinate
        else:
            raise ValueError("each coordinate must contain row and col")
        if isinstance(row, bool) or isinstance(col, bool) or not isinstance(row, int) or not isinstance(col, int):
            raise ValueError("coordinate row and col must be integers")
        parsed.append((row, col))
    return tuple(parsed)
