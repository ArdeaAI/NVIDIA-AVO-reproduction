"""
Backend adapter that turns a coding agent into an AVO variation operator.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from ardea_avo.core import VariationRequest, VariationResult
from ardea_avo.runtime import (
    AgentRequest,
    AnthropicMessagesBackend,
    OpenAIResponsesBackend,
    ToolDefinition,
)


class Backend(Protocol):
    """
    Minimal model backend used by the variation adapter.
    """

    def run(self, request: AgentRequest) -> Any:
        """
        Execute one autonomous turn.
        """
        ...


class ConfinedWorkspace:
    """
    Small function-tool filesystem rooted in a dedicated candidate directory.
    """

    def __init__(
        self,
        root: Path,
        *,
        check_argv: tuple[str, ...] = (),
        check_timeout_seconds: float = 120.0,
    ) -> None:
        """
        Configure path confinement and an optional fixed validation command.
        """
        self.root = root.resolve()
        self.check_argv = check_argv
        self.check_timeout_seconds = check_timeout_seconds

    def _path(self, relative: str, *, may_not_exist: bool = False) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("path must be a non-empty relative string")
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts or ".git" in raw.parts:
            raise ValueError("path must stay inside the candidate tree and cannot address .git")
        current = self.root
        for part in raw.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("candidate tools do not follow symlinks")
        resolved_parent = current.parent.resolve()
        if resolved_parent != self.root and not resolved_parent.is_relative_to(self.root):
            raise ValueError("path escapes the candidate tree")
        if not may_not_exist and not current.exists():
            raise FileNotFoundError(relative)
        return current

    def list_files(self) -> dict[str, Any]:
        """
        List regular candidate files without following symlinks.
        """
        files: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if ".git" in relative.parts or path.is_symlink() or not path.is_file():
                continue
            files.append({"path": relative.as_posix(), "bytes": path.stat().st_size})
            if len(files) >= 2_000:
                break
        return {"files": files, "truncated": len(files) == 2_000}

    def read_file(self, path: str) -> dict[str, Any]:
        """
        Read one UTF-8 candidate file with a bounded payload.
        """
        selected = self._path(path)
        if not selected.is_file() or selected.stat().st_size > 1_000_000:
            raise ValueError("read_file requires a regular file no larger than 1 MB")
        return {"path": path, "content": selected.read_text(encoding="utf-8")}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """
        Replace one UTF-8 candidate file without traversing links.
        """
        if not isinstance(content, str) or len(content.encode("utf-8")) > 1_000_000:
            raise ValueError("write_file content must be UTF-8 text no larger than 1 MB")
        selected = self._path(path, may_not_exist=True)
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(content, encoding="utf-8")
        return {"path": path, "bytes": len(content.encode("utf-8"))}

    def delete_file(self, path: str) -> dict[str, Any]:
        """
        Delete one regular candidate file.
        """
        selected = self._path(path)
        if not selected.is_file():
            raise ValueError("delete_file supports regular files only")
        selected.unlink()
        return {"path": path, "deleted": True}

    def run_check(self) -> dict[str, Any]:
        """
        Run the target's fixed validation argv without a shell.
        """
        if not self.check_argv:
            raise ValueError("this target does not define a candidate-side check command")
        allowed_environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"HOME", "LANG", "LC_ALL", "LC_CTYPE", "PATH", "TEMP", "TMP", "TMPDIR"}
        }
        completed = subprocess.run(
            list(self.check_argv),
            cwd=self.root,
            env=allowed_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.check_timeout_seconds,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20_000:],
            "stderr": completed.stderr[-20_000:],
        }

    def tools(self) -> tuple[ToolDefinition, ...]:
        """
        Return strict Responses API definitions for the confined operations.
        """
        no_arguments = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        path_arguments = {
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
            "additionalProperties": False,
        }
        return (
            ToolDefinition(
                "list_files", "List candidate files.", no_arguments, self.list_files
            ),
            ToolDefinition("read_file", "Read one candidate text file.", path_arguments, self.read_file),
            ToolDefinition(
                "write_file",
                "Create or replace one candidate text file.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                self.write_file,
                sequential=True,
            ),
            ToolDefinition(
                "delete_file",
                "Delete one candidate file.",
                path_arguments,
                self.delete_file,
                sequential=True,
            ),
            ToolDefinition(
                "run_check",
                "Run the fixed candidate validation command.",
                no_arguments,
                self.run_check,
                sequential=True,
            ),
        )


class BackendVariationAgent:
    """
    Use a supported model backend as the paper's autonomous variation operator.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        check_argv: tuple[str, ...] = (),
        session_id: str | None = None,
    ) -> None:
        """
        Configure a persistent lineage session and optional candidate check.
        """
        self.backend = backend
        self.check_argv = check_argv
        self.session_id = session_id

    def vary(self, request: VariationRequest, workspace: Path) -> VariationResult:
        """
        Ask the coding agent to inspect and mutate the complete candidate tree.
        """
        system_prompt = (
            "You are the Agentic Variation Operator for one accepted candidate lineage. "
            "Inspect the candidate, choose one coherent improvement, edit only the candidate tree, "
            "run useful checks, and leave the workspace in the exact state you want the host evaluator "
            "to score. Do not access or modify evaluator, ledger, result, credential, or Git metadata. "
            "Exact-score ties may be accepted when they enable later work. End with a concise summary "
            "of the hypothesis, edits, checks, and unresolved risks."
        )
        prompt = (
            f"Variation attempt {request.attempt}.\n"
            "The immutable request follows as JSON:\n"
            f"{request.model_dump_json(indent=2)}"
        )
        confined = ConfinedWorkspace(workspace, check_argv=self.check_argv)
        tools = (
            confined.tools()
            if isinstance(
                self.backend,
                (OpenAIResponsesBackend, AnthropicMessagesBackend),
            )
            else ()
        )
        result = self.backend.run(
            AgentRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                cwd=workspace,
                session_id=self.session_id,
                role="variation",
                reasoning_effort="max",
                tools=tools,
                metadata={"attempt": request.attempt, "candidate_id": request.candidate.candidate_id},
            )
        )
        self.session_id = result.session_id
        summary = result.text.strip() or "Variation backend completed without a textual summary."
        return VariationResult(
            summary=summary[:10_000],
            metadata={
                "backend_session_id": result.session_id,
                "usage": asdict(result.usage),
                "cost_usd": str(result.cost_usd),
                "warnings": list(result.warnings),
            },
        )
