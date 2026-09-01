"""
Dependency-free local MCP-compatible surface for ARC tools.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .tools import ArcToolRuntime, coordinates_from_json
from .types import ToolReply


def _integer(description: str, *, minimum: int = 0, maximum: int = 63) -> dict[str, Any]:
    return {"type": "integer", "description": description, "minimum": minimum, "maximum": maximum}


TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "play",
        "description": "Commit one counted ARC environment action.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"],
                },
                "row": _integer("Zero-indexed row used only by ACTION6."),
                "col": _integer("Zero-indexed column used only by ACTION6."),
                "expected_outcome": {"type": "string", "maxLength": 1000},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "observe",
        "description": "Read the current exact 64x64 grid and status without acting.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "inspect",
        "description": "Read an exact grid or crop from a settled historical turn.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "turn": _integer("Historical turn.", maximum=2_147_483_647),
                "top": _integer("Inclusive crop row."),
                "left": _integer("Inclusive crop column."),
                "bottom": _integer("Exclusive crop row.", minimum=1, maximum=64),
                "right": _integer("Exclusive crop column.", minimum=1, maximum=64),
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "read_pixels",
        "description": "Read exact pixel values at selected row/column coordinates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "turn": _integer("Historical turn.", maximum=2_147_483_647),
                "coordinates": {
                    "type": "array",
                    "maxItems": 256,
                    "items": {
                        "type": "object",
                        "properties": {"row": _integer("Row."), "col": _integer("Column.")},
                        "required": ["row", "col"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["coordinates"],
            "additionalProperties": False,
        },
    },
    {
        "name": "history",
        "description": "Read compact committed-action history without acting.",
        "inputSchema": {
            "type": "object",
            "properties": {"last": _integer("Number of recent turns.", minimum=1, maximum=1000)},
            "additionalProperties": False,
        },
    },
    {
        "name": "diff",
        "description": "Read exact cell changes between two settled turns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "before_turn": _integer("Earlier historical turn.", maximum=2_147_483_647),
                "after_turn": _integer("Later historical turn.", maximum=2_147_483_647),
                "limit": _integer("Maximum changes rendered.", minimum=1, maximum=4096),
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "segments",
        "description": "Summarize four-connected single-color components without acting.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "turn": _integer("Historical turn.", maximum=2_147_483_647),
                "include_zero": {"type": "boolean"},
                "limit": _integer("Maximum segment summaries.", minimum=1, maximum=4096),
            },
            "additionalProperties": False,
        },
    },
)


class LocalArcMcpSurface:
    """
    Expose MCP list-tools and call-tool payloads without an SDK dependency.

    A transport can directly forward these dictionaries over stdio. Only
    ``play`` can mutate the environment or increment its action counter.
    """

    counted_tool_names = frozenset({"play"})

    def __init__(self, runtime: ArcToolRuntime) -> None:
        self.runtime = runtime

    def list_tools(self) -> list[dict[str, Any]]:
        """
        Return fresh MCP-compatible tool descriptors.
        """

        return [dict(definition) for definition in TOOL_DEFINITIONS]

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """
        Dispatch one validated local tool call and return MCP content blocks.
        """

        args = dict(arguments or {})
        try:
            allowed = {
                "play": {"action", "row", "col", "expected_outcome"},
                "observe": set(),
                "inspect": {"turn", "top", "left", "bottom", "right"},
                "read_pixels": {"turn", "coordinates"},
                "history": {"last"},
                "diff": {"before_turn", "after_turn", "limit"},
                "segments": {"turn", "include_zero", "limit"},
            }
            if name not in allowed:
                raise ValueError(f"unknown ARC tool {name!r}")
            unexpected = set(args) - allowed[name]
            if unexpected:
                raise ValueError(f"unexpected arguments for {name}: {', '.join(sorted(unexpected))}")
            if name == "play":
                reply = self.runtime.play(
                    args.pop("action"),
                    row=args.pop("row", None),
                    col=args.pop("col", None),
                    expected_outcome=args.pop("expected_outcome", None),
                )
            elif name == "observe":
                reply = ToolReply(self.runtime.observe())
            elif name == "inspect":
                reply = ToolReply(self.runtime.inspect(**args))
                args.clear()
            elif name == "read_pixels":
                coordinates = coordinates_from_json(args.pop("coordinates"))
                reply = ToolReply(self.runtime.read_pixels(coordinates, turn=args.pop("turn", None)))
            elif name == "history":
                reply = ToolReply(self.runtime.history(last=args.pop("last", 30)))
            elif name == "diff":
                reply = ToolReply(self.runtime.diff(**args))
                args.clear()
            elif name == "segments":
                reply = ToolReply(self.runtime.segments(**args))
                args.clear()
            else:
                raise AssertionError("tool routing table is incomplete")
            if args:
                raise ValueError(f"unexpected arguments for {name}: {', '.join(sorted(args))}")
        except (KeyError, TypeError, ValueError) as exc:
            reply = ToolReply(str(exc), is_error=True)
        return {
            "content": [{"type": "text", "text": reply.content}],
            "isError": reply.is_error,
            "_meta": {
                "counted": reply.counted,
                "terminal": reply.terminal,
            },
        }
