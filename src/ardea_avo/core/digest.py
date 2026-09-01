"""
Deterministic candidate-tree hashing.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from ardea_avo.core.exceptions import LineageStateError


def tree_digest(root: Path) -> str:
    """
    Hash every file, symlink target, relative path, and executable bit.

    Empty directories are intentionally omitted because Git cannot represent them.
    Symlinks are hashed without following them. The resulting digest is stable across
    platforms for trees with the same Git-representable contents. Any entry named
    ``.git`` is rejected because Git reserves that name and it cannot be an unmeasured
    or agent-controlled metadata channel.
    """
    return _tree_digest(root, reject_git=True)


def archive_tree_digest(root: Path) -> str:
    """
    Hash an invalid rejected tree, including forbidden ``.git`` evidence.

    This digest is only for a rejected-attempt record. It must never authorize
    evaluation or promotion of the tree.
    """
    return _tree_digest(root, reject_git=False)


def _tree_digest(root: Path, *, reject_git: bool) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise LineageStateError(f"candidate root is not a directory: {root}")

    entries: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        forbidden_directories = [name for name in directory_names if name.casefold() == ".git"]
        forbidden_files = [name for name in file_names if name.casefold() == ".git"]
        forbidden = forbidden_directories or forbidden_files
        if reject_git and forbidden:
            raise LineageStateError(
                f"candidate tree contains forbidden .git entry: {current / forbidden[0]}"
            )
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            entries.append(current / name)
        for name in directory_names:
            path = current / name
            if path.is_symlink():
                entries.append(path)

    digest = hashlib.sha256()
    for path in sorted(entries, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            kind = b"L"
            mode = b"120000"
            content = os.readlink(path).encode("utf-8", errors="surrogateescape")
        elif stat.S_ISREG(file_stat.st_mode):
            kind = b"F"
            mode = b"100755" if file_stat.st_mode & stat.S_IXUSR else b"100644"
            content = path.read_bytes()
        else:
            raise LineageStateError(f"unsupported candidate-tree entry: {path}")

        digest.update(kind)
        digest.update(b"\0")
        digest.update(mode)
        digest.update(b"\0")
        digest.update(str(len(relative)).encode("ascii"))
        digest.update(b":")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b":")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()
