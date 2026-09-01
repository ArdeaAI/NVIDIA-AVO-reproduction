"""
Read-only diagnostics for local ARC and model prerequisites.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Check:
    """
    One non-secret diagnostic result.
    """

    name: str
    ok: bool
    detail: str
    required: bool = True


def _package_check(distribution: str, expected: str) -> Check:
    """
    Check an installed distribution without importing it.
    """
    try:
        installed = version(distribution)
    except PackageNotFoundError:
        return Check(distribution, False, "not installed")
    return Check(distribution, installed == expected, f"{installed}; expected {expected}")


def _codex_checks() -> list[Check]:
    """
    Verify the CLI and saved ChatGPT authentication without exposing credentials.
    """
    executable = shutil.which("codex")
    if executable is None:
        return [Check("codex", False, "not found on PATH"), Check("codex_auth", False, "not checked")]
    try:
        release = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [executable, "login", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [Check("codex", False, str(error)), Check("codex_auth", False, "not checked")]
    auth_text = " ".join(part.strip() for part in (status.stdout, status.stderr) if part.strip())
    chatgpt = status.returncode == 0 and "chatgpt" in auth_text.lower()
    return [
        Check("codex", True, release),
        Check("codex_auth", chatgpt, "saved ChatGPT sign-in" if chatgpt else "run `codex login`"),
    ]


def run_checks(cache_dir: Path, backend: str = "codex-oauth") -> list[Check]:
    """
    Run all safe diagnostics and return structured results.
    """
    checks = [
        Check(
            "python",
            sys.version_info[:2] == (3, 12),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ),
        _package_check("arc-agi", "0.9.9"),
        _package_check("arcengine", "0.9.3"),
        Check("results", os.access(Path.cwd(), os.W_OK), str(Path.cwd() / "results")),
    ]
    metadata = list(cache_dir.rglob("metadata.json")) if cache_dir.is_dir() else []
    checks.append(
        Check(
            "arc_cache",
            len(metadata) == 25,
            f"{len(metadata)}/25 metadata files at {cache_dir}",
            required=False,
        )
    )
    checks.append(
        Check(
            "arc_api_key",
            bool(os.environ.get("ARC_API_KEY")),
            "set" if os.environ.get("ARC_API_KEY") else "not set; needed only for setup/scorecards",
            required=False,
        )
    )
    if backend == "codex-oauth":
        checks.extend(_codex_checks())
    elif backend == "openai-api":
        checks.append(
            Check(
                "openai_api_key",
                bool(os.environ.get("OPENAI_API_KEY")),
                "set" if os.environ.get("OPENAI_API_KEY") else "not set",
            )
        )
    elif backend == "anthropic-api":
        checks.append(
            Check(
                "anthropic_api_key",
                bool(os.environ.get("ANTHROPIC_API_KEY")),
                "set" if os.environ.get("ANTHROPIC_API_KEY") else "not set",
            )
        )
    else:
        checks.append(Check("backend", False, f"unsupported backend: {backend}"))
    model = "claude-opus-5" if backend == "anthropic-api" else "gpt-5.6-sol"
    checks.append(
        Check(
            "model",
            True,
            f"{model} configured; account availability is checked on first model call",
            required=False,
        )
    )
    return checks


def checks_pass(checks: list[Check]) -> bool:
    """
    Return whether every required check passed.
    """
    return all(check.ok or not check.required for check in checks)
