"""
Repository-level enforcement for the Ardea Python docstring convention.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _docstring_nodes(tree: ast.AST) -> list[ast.Expr]:
    """
    Return every module, class, and function docstring expression.
    """
    found: list[ast.Expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            found.append(first)
    return found


def test_docstring_quotes_have_dedicated_lines() -> None:
    """
    Require opening and closing triple quotes to occupy their own lines.
    """
    failures: list[str] = []
    source_root = Path(__file__).parents[1] / "src"
    for path in sorted(source_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        for expression in _docstring_nodes(tree):
            assert expression.end_lineno is not None
            first = lines[expression.lineno - 1].strip()
            last = lines[expression.end_lineno - 1].strip()
            if first != '"""' or last != '"""' or expression.lineno == expression.end_lineno:
                failures.append(f"{path.relative_to(source_root)}:{expression.lineno}")
    assert not failures, "docstring quotes need dedicated lines: " + ", ".join(failures)

