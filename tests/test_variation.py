"""
Tests for the backend variation adapter's file boundary.
"""

from pathlib import Path

import pytest

from ardea_avo.variation import ConfinedWorkspace


def test_confined_workspace_reads_and_writes_candidate(tmp_path: Path) -> None:
    """
    Allow ordinary files while keeping paths relative.
    """
    workspace = ConfinedWorkspace(tmp_path)
    assert workspace.write_file("pkg/solution.py", "answer = 1\n")["bytes"] > 0
    assert workspace.read_file("pkg/solution.py")["content"] == "answer = 1\n"
    assert workspace.list_files()["files"] == [{"path": "pkg/solution.py", "bytes": 11}]
    assert workspace.delete_file("pkg/solution.py")["deleted"] is True


@pytest.mark.parametrize("path", ["../secret", "/tmp/secret", ".git/config", "pkg/../../secret"])
def test_confined_workspace_rejects_escape(tmp_path: Path, path: str) -> None:
    """
    Reject parent, absolute, and Git-metadata paths.
    """
    with pytest.raises(ValueError):
        ConfinedWorkspace(tmp_path).write_file(path, "bad")


def test_confined_workspace_rejects_symlink(tmp_path: Path) -> None:
    """
    Never follow a candidate-controlled link outside the worktree.
    """
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        ConfinedWorkspace(tmp_path).read_file("link")


def test_confined_workspace_rejects_dangling_write_symlink(tmp_path: Path) -> None:
    """
    A dangling final link cannot turn a model write into an out-of-tree write.
    """

    outside = tmp_path.parent / "not-created.txt"
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        ConfinedWorkspace(tmp_path).write_file("link", "escape")
    assert not outside.exists()
