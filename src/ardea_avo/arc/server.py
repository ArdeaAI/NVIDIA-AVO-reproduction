"""
Offline stdio MCP server for one replayable ARC game session.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter import OfficialArcadeFactory
from .tools import ArcToolRuntime, coordinates_from_json


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """
    Validated command-line configuration for one offline game server.
    """

    game_id: str
    cache_dir: Path
    trace: Path
    memory_db: Path | None
    run_id: str | None


def build_parser() -> argparse.ArgumentParser:
    """
    Build the stdio server argument parser without loading ARC or MCP packages.
    """

    parser = argparse.ArgumentParser(description="Serve one offline ARC-AGI-3 game over stdio MCP.")
    parser.add_argument("--game-id", required=True, help="Full versioned ARC game identifier.")
    parser.add_argument("--cache-dir", required=True, type=Path, help="Downloaded official environment cache.")
    parser.add_argument("--trace", required=True, type=Path, help="Append-only action trace path.")
    parser.add_argument("--memory-db", type=Path, help="Optional run-scoped SQLite memory database.")
    parser.add_argument("--run-id", help="Owning run id, required with --memory-db.")
    return parser


def parse_server_args(argv: Sequence[str] | None = None) -> ServerConfig:
    """
    Parse and normalize server arguments without changing external state.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    game_id = args.game_id.strip()
    if not game_id:
        parser.error("--game-id cannot be blank")
    if (args.memory_db is None) != (args.run_id is None):
        parser.error("--memory-db and --run-id must be supplied together")
    if args.run_id is not None and not args.run_id.strip():
        parser.error("--run-id cannot be blank")
    return ServerConfig(
        game_id=game_id,
        cache_dir=args.cache_dir.expanduser().resolve(),
        trace=args.trace.expanduser().resolve(),
        memory_db=None if args.memory_db is None else args.memory_db.expanduser().resolve(),
        run_id=None if args.run_id is None else args.run_id.strip(),
    )


def create_mcp_server(runtime: ArcToolRuntime, memory_store: Any | None = None) -> Any:
    """
    Create a real FastMCP server bound to one host-owned ARC runtime.

    Importing the MCP SDK is deferred so parser and diagnostic commands remain
    usable before optional runtime dependencies are installed.
    """

    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "ardea-arc",
        instructions=(
            "Interact with one ARC-AGI-3 game. The text grid is authoritative. "
            "Only play consumes an environment action. Human baselines are private."
        ),
        log_level="ERROR",
    )

    @server.tool(description="Commit one counted ARC environment action.")
    def play(
        action: str,
        row: int | None = None,
        col: int | None = None,
        expected_outcome: str | None = None,
    ) -> str:
        """
        Commit one validated action and return its exact settled observation.
        """

        reply = runtime.play(
            action,
            row=row,
            col=col,
            expected_outcome=expected_outcome,
        )
        if reply.is_error:
            raise ValueError(reply.content)
        return reply.content

    @server.tool(description="Read the current exact 64x64 grid and status without acting.")
    def observe() -> str:
        """
        Return the current text observation.
        """

        return runtime.observe()

    @server.tool(description="Read an exact grid or crop from a settled historical turn.")
    def inspect(
        turn: int | None = None,
        top: int = 0,
        left: int = 0,
        bottom: int = 64,
        right: int = 64,
    ) -> str:
        """
        Return an exact historical frame or crop.
        """

        return runtime.inspect(turn=turn, top=top, left=left, bottom=bottom, right=right)

    @server.tool(description="Read exact pixel values at selected row and column coordinates.")
    def read_pixels(coordinates: list[dict[str, int]], turn: int | None = None) -> str:
        """
        Return exact historical pixel values.
        """

        return runtime.read_pixels(coordinates_from_json(coordinates), turn=turn)

    @server.tool(description="Read compact committed-action history without acting.")
    def history(last: int = 30) -> str:
        """
        Return recent committed action receipts.
        """

        return runtime.history(last=last)

    @server.tool(description="Read exact cell changes between two settled turns.")
    def diff(
        before_turn: int | None = None,
        after_turn: int | None = None,
        limit: int = 256,
    ) -> str:
        """
        Return exact changes between two settled frames.
        """

        return runtime.diff(before_turn=before_turn, after_turn=after_turn, limit=limit)

    @server.tool(description="Summarize four-connected single-color components without acting.")
    def segments(
        turn: int | None = None,
        include_zero: bool = False,
        limit: int = 128,
    ) -> str:
        """
        Return deterministic connected-component summaries.
        """

        return runtime.segments(turn=turn, include_zero=include_zero, limit=limit)

    @server.tool(description="Propose an evidence-linked game-memory claim without taking an action.")
    def propose_memory(
        claim: str,
        status: str = "hypothesis",
        confidence: float = 0.5,
        scope: str = "game",
        evidence: list[str] | None = None,
        contradictions: list[str] | None = None,
        origin_model: str = "gpt-5.6-sol",
    ) -> str:
        """
        Store a scoped claim after checking every cited receipt or frame hash.
        """

        if memory_store is None:
            raise ValueError("memory storage was not configured for this server")
        evidence_values = tuple(evidence or ())
        contradiction_values = tuple(contradictions or ())
        unknown = (set(evidence_values) | set(contradiction_values)) - runtime.evidence_hashes
        if unknown:
            raise ValueError("memory cites evidence that is not present in this game trace")
        if scope == "game":
            scope_id = runtime.game_id
        elif scope == "level":
            scope_id = f"{runtime.game_id}:level:{runtime.frame.levels_completed + 1}"
        elif scope == "run":
            scope_id = None
        else:
            raise ValueError("memory scope must be run, game, or level")
        record = memory_store.add(
            scope=scope,
            scope_id=scope_id,
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence_values,
            contradictions=contradiction_values,
            origin_model=origin_model,
        )
        normalized_status = getattr(record.status, "value", str(record.status))
        if normalized_status in {"verified", "falsified"}:
            record = memory_store.approve_for_warm(record.id)
        return json.dumps(
            {
                "id": record.id,
                "status": normalized_status,
                "claim": record.claim,
                "confidence": record.confidence,
                "evidence": list(record.evidence),
                "contradictions": list(record.contradictions),
                "approved_for_warm": record.approved_for_warm,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    return server


def run_server(config: ServerConfig) -> None:
    """
    Replay or create one trace and serve it through offline stdio MCP.
    """

    factory = OfficialArcadeFactory(config.cache_dir)
    environment = factory(config.game_id)
    runtime: ArcToolRuntime | None = None
    memory_store: Any | None = None
    try:
        if config.trace.exists():
            runtime = ArcToolRuntime.resume(environment, config.trace)
        else:
            runtime = ArcToolRuntime(environment, trace_path=config.trace)
        if config.memory_db is not None:
            from ardea_avo.runtime.memory import MemoryStore

            assert config.run_id is not None
            memory_store = MemoryStore(config.memory_db, run_id=config.run_id)
        create_mcp_server(runtime, memory_store).run(transport="stdio")
    finally:
        if runtime is not None:
            runtime.close()
        else:
            environment.close()
        if memory_store is not None:
            memory_store.close()


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the offline ARC stdio MCP server.
    """

    run_server(parse_server_args(argv))


if __name__ == "__main__":
    main()
