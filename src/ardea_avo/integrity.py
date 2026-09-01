"""
Offline provenance, deterministic manifests, and contamination checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any


class IntegrityError(RuntimeError):
    """
    Base error for provenance or filesystem integrity failures.
    """


class GitProvenanceError(IntegrityError):
    """
    Raised when repository identity cannot be captured deterministically.
    """


class UnsafeFilesystemEntry(IntegrityError):
    """
    Raised when a manifest tree contains a symlink or non-regular entry.
    """


class ContaminationKind(StrEnum):
    """
    Fail-closed reasons an agent-controlled bundle is not clean.
    """

    SYMLINK = "symlink"
    NON_REGULAR = "non_regular"
    RESERVED_GIT = "reserved_git"
    FORBIDDEN_GAME_ID = "forbidden_game_id"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class ContaminationFinding:
    """
    One path-scoped bundle-integrity failure.
    """

    kind: ContaminationKind
    path: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """
        Return a JSON-compatible finding.
        """

        return {"kind": self.kind.value, "path": self.path, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    """
    Complete deterministic scan result for one agent-controlled bundle.
    """

    root: str
    scanned_files: int
    findings: tuple[ContaminationFinding, ...]

    @property
    def clean(self) -> bool:
        """
        Return whether no contamination or unsafe entry was found.
        """

        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible report suitable for a run manifest.
        """

        return {
            "root": self.root,
            "scanned_files": self.scanned_files,
            "clean": self.clean,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def require_clean(self) -> ContaminationReport:
        """
        Return this report or raise with all findings attached.
        """

        if not self.clean:
            raise ContaminationError(self)
        return self


class ContaminationError(IntegrityError):
    """
    Raised when an agent-controlled bundle fails its contamination scan.
    """

    def __init__(self, report: ContaminationReport) -> None:
        self.report = report
        summary = "; ".join(
            f"{finding.kind.value}:{finding.path}" for finding in report.findings[:8]
        )
        if len(report.findings) > 8:
            summary += f"; +{len(report.findings) - 8} more"
        super().__init__(f"agent bundle contamination detected: {summary}")


@dataclass(frozen=True, slots=True)
class FileManifestEntry:
    """
    Digest and Git-relevant metadata for one regular file.
    """

    path: str
    size: int
    sha256: str
    executable: bool

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible file record.
        """

        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentBundleManifest:
    """
    Frozen clean bundle identity independent of its absolute location.
    """

    root: str
    sha256: str
    files: tuple[FileManifestEntry, ...]
    contamination: ContaminationReport

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible bundle manifest.
        """

        return {
            "root": self.root,
            "sha256": self.sha256,
            "files": [entry.to_dict() for entry in self.files],
            "contamination": self.contamination.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CacheManifest:
    """
    Deterministic identity of every regular artifact in a local cache.
    """

    root: str
    sha256: str
    file_count: int
    total_bytes: int
    files: tuple[FileManifestEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible cache manifest.
        """

        return {
            "root": self.root,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "files": [entry.to_dict() for entry in self.files],
        }


@dataclass(frozen=True, slots=True)
class GitProvenance:
    """
    Repository commit and exact working-tree delta identity.
    """

    repository_root: str
    commit: str
    dirty: bool
    diff_sha256: str
    status: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible repository provenance record.
        """

        return {
            "repository_root": self.repository_root,
            "commit": self.commit,
            "dirty": self.dirty,
            "diff_sha256": self.diff_sha256,
            "status": list(self.status),
        }


def _absolute_without_following(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _tree_paths(root: Path) -> tuple[Path, ...]:
    if root.is_symlink():
        raise UnsafeFilesystemEntry(f"manifest root cannot be a symlink: {root}")
    if not root.is_dir():
        raise IntegrityError(f"manifest root is not a directory: {root}")
    paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        directory_names.sort()
        file_names.sort()
        retained_directories: list[str] = []
        for name in directory_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise UnsafeFilesystemEntry(f"manifest trees cannot contain symlinks: {_relative(path, root)}")
            if not stat.S_ISDIR(mode):
                raise UnsafeFilesystemEntry(
                    f"manifest tree contains a non-directory entry: {_relative(path, root)}"
                )
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in file_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise UnsafeFilesystemEntry(f"manifest trees cannot contain symlinks: {_relative(path, root)}")
            if not stat.S_ISREG(mode):
                raise UnsafeFilesystemEntry(
                    f"manifest trees require regular files: {_relative(path, root)}"
                )
            paths.append(path)
    return tuple(sorted(paths, key=lambda item: _relative(item, root)))


def _read_stable_file(path: Path) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise UnsafeFilesystemEntry(f"file changed to an unsafe entry during capture: {path}")
    content = path.read_bytes()
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode)
    if identity_before != identity_after or len(content) != after.st_size:
        raise IntegrityError(f"file changed while its manifest was being captured: {path}")
    return content, after


def _capture_files(root: Path, *, domain: bytes) -> tuple[tuple[FileManifestEntry, ...], str]:
    entries: list[FileManifestEntry] = []
    digest = hashlib.sha256(domain + b"\0")
    for path in _tree_paths(root):
        content, metadata = _read_stable_file(path)
        entry = FileManifestEntry(
            path=_relative(path, root),
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            executable=bool(metadata.st_mode & stat.S_IXUSR),
        )
        entries.append(entry)
        encoded = json.dumps(
            entry.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return tuple(entries), digest.hexdigest()


def scan_agent_bundle(
    root: str | Path,
    forbidden_game_ids: Iterable[str] = (),
) -> ContaminationReport:
    """
    Scan paths and bytes without following symlinks or executing bundle code.
    """

    bundle_root = _absolute_without_following(root)
    identifiers = tuple(sorted({value.strip() for value in forbidden_game_ids if value.strip()}))
    identifier_bytes = tuple((value, value.casefold().encode("utf-8")) for value in identifiers)
    findings: list[ContaminationFinding] = []
    scanned_files = 0
    if bundle_root.is_symlink():
        findings.append(
            ContaminationFinding(
                kind=ContaminationKind.SYMLINK,
                path=".",
                detail="the bundle root is a symlink",
            )
        )
        return ContaminationReport(str(bundle_root), scanned_files, tuple(findings))
    if not bundle_root.is_dir():
        raise IntegrityError(f"agent bundle root is not a directory: {bundle_root}")

    for directory, directory_names, file_names in os.walk(bundle_root, followlinks=False):
        current = Path(directory)
        directory_names.sort()
        file_names.sort()
        retained: list[str] = []
        for name in directory_names:
            path = current / name
            relative = _relative(path, bundle_root)
            if name.casefold() == ".git":
                findings.append(
                    ContaminationFinding(
                        ContaminationKind.RESERVED_GIT,
                        relative,
                        "agent bundles cannot contain Git metadata",
                    )
                )
                continue
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                findings.append(
                    ContaminationFinding(ContaminationKind.UNREADABLE, relative, type(exc).__name__)
                )
                continue
            if stat.S_ISLNK(mode):
                findings.append(
                    ContaminationFinding(
                        ContaminationKind.SYMLINK,
                        relative,
                        "symlink entries are forbidden",
                    )
                )
                continue
            if not stat.S_ISDIR(mode):
                findings.append(
                    ContaminationFinding(
                        ContaminationKind.NON_REGULAR,
                        relative,
                        "bundle directory entry is not a directory",
                    )
                )
                continue
            retained.append(name)
            folded_path = relative.casefold().encode("utf-8")
            for identifier, needle in identifier_bytes:
                if needle in folded_path:
                    findings.append(
                        ContaminationFinding(
                            ContaminationKind.FORBIDDEN_GAME_ID,
                            relative,
                            f"path contains forbidden full game id {identifier!r}",
                        )
                    )
        directory_names[:] = retained

        for name in file_names:
            path = current / name
            relative = _relative(path, bundle_root)
            if name.casefold() == ".git":
                findings.append(
                    ContaminationFinding(
                        ContaminationKind.RESERVED_GIT,
                        relative,
                        "agent bundles cannot contain Git metadata",
                    )
                )
                continue
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                findings.append(
                    ContaminationFinding(ContaminationKind.UNREADABLE, relative, type(exc).__name__)
                )
                continue
            if stat.S_ISLNK(mode):
                findings.append(
                    ContaminationFinding(
                        ContaminationKind.SYMLINK,
                        relative,
                        "symlink entries are forbidden",
                    )
                )
                continue
            if not stat.S_ISREG(mode):
                findings.append(
                    ContaminationFinding(
                        ContaminationKind.NON_REGULAR,
                        relative,
                        "agent bundle entries must be regular files",
                    )
                )
                continue
            scanned_files += 1
            try:
                content, _ = _read_stable_file(path)
            except (OSError, IntegrityError) as exc:
                findings.append(
                    ContaminationFinding(ContaminationKind.UNREADABLE, relative, type(exc).__name__)
                )
                continue
            folded_path = relative.casefold().encode("utf-8")
            folded_content = content.lower()
            for identifier, needle in identifier_bytes:
                if needle in folded_path or needle in folded_content:
                    location = "path" if needle in folded_path else "content"
                    findings.append(
                        ContaminationFinding(
                            ContaminationKind.FORBIDDEN_GAME_ID,
                            relative,
                            f"{location} contains forbidden full game id {identifier!r}",
                        )
                    )

    ordered = tuple(sorted(findings, key=lambda item: (item.path, item.kind.value, item.detail)))
    return ContaminationReport(str(bundle_root), scanned_files, ordered)


def capture_agent_bundle(
    root: str | Path,
    forbidden_game_ids: Iterable[str] = (),
) -> AgentBundleManifest:
    """
    Capture a stable deterministic bundle only when its scan is clean.
    """

    bundle_root = _absolute_without_following(root)
    identifiers = tuple(forbidden_game_ids)
    first_report = scan_agent_bundle(bundle_root, identifiers).require_clean()
    before_files, before_digest = _capture_files(bundle_root, domain=b"ARDEA-AGENT-BUNDLE-v1")
    report = scan_agent_bundle(bundle_root, identifiers).require_clean()
    after_files, after_digest = _capture_files(bundle_root, domain=b"ARDEA-AGENT-BUNDLE-v1")
    if first_report != report or before_files != after_files or before_digest != after_digest:
        raise IntegrityError("agent bundle changed while it was being scanned")
    return AgentBundleManifest(
        root=str(bundle_root),
        sha256=after_digest,
        files=after_files,
        contamination=report,
    )


def agent_bundle_digest(root: str | Path, forbidden_game_ids: Iterable[str] = ()) -> str:
    """
    Return the deterministic digest of a clean agent-controlled bundle.
    """

    return capture_agent_bundle(root, forbidden_game_ids).sha256


def capture_cache_manifest(root: str | Path) -> CacheManifest:
    """
    Capture every cache file while rejecting symlinks and special entries.
    """

    cache_root = _absolute_without_following(root)
    first_files, first_digest = _capture_files(cache_root, domain=b"ARDEA-CACHE-MANIFEST-v1")
    files, digest = _capture_files(cache_root, domain=b"ARDEA-CACHE-MANIFEST-v1")
    if first_files != files or first_digest != digest:
        raise IntegrityError("cache changed while its manifest was being captured")
    return CacheManifest(
        root=str(cache_root),
        sha256=digest,
        file_count=len(files),
        total_bytes=sum(entry.size for entry in files),
        files=files,
    )


def _run_git(path: Path, arguments: Sequence[str], *, allow_failure: bool = False) -> bytes:
    command = [
        "git",
        "-c",
        "color.ui=false",
        "-c",
        "core.quotepath=false",
        "-C",
        str(path),
        *arguments,
    ]
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C", "LANG": "C"})
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
        )
    except OSError as exc:
        raise GitProvenanceError("could not execute Git") from exc
    if result.returncode != 0 and not allow_failure:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitProvenanceError(message or f"Git command failed: {' '.join(arguments)}")
    return result.stdout if result.returncode == 0 else b""


def _safe_git_relative(raw: bytes) -> PurePosixPath:
    value = raw.decode("utf-8", errors="surrogateescape")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise GitProvenanceError("Git returned an unsafe untracked path")
    return path


def _hash_untracked(root: Path, raw_paths: bytes, digest: Any) -> None:
    for raw in sorted(part for part in raw_paths.split(b"\0") if part):
        relative = _safe_git_relative(raw)
        path = root.joinpath(*relative.parts)
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise GitProvenanceError(f"untracked path changed during capture: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            content = os.readlink(path).encode("utf-8", errors="surrogateescape")
            executable = False
            if path.lstat() != metadata:
                raise GitProvenanceError(f"untracked symlink changed during capture: {relative}")
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            try:
                content, stable = _read_stable_file(path)
            except IntegrityError as exc:
                raise GitProvenanceError(f"untracked file changed during capture: {relative}") from exc
            executable = bool(stable.st_mode & stat.S_IXUSR)
        else:
            raise GitProvenanceError(f"untracked path has an unsupported type: {relative}")
        digest.update(kind)
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
        digest.update(b"1" if executable else b"0")
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)


def _capture_git_delta(root: Path) -> tuple[bytes, bytes, bytes, str]:
    status_raw = _run_git(root, ("status", "--porcelain=v1", "-z", "--untracked-files=all"))
    diff_raw = _run_git(
        root,
        ("diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", "HEAD", "--"),
    )
    untracked_raw = _run_git(root, ("ls-files", "--others", "--exclude-standard", "-z"))
    digest = hashlib.sha256(b"ARDEA-GIT-DIFF-v1\0")
    digest.update(len(status_raw).to_bytes(8, "big"))
    digest.update(status_raw)
    digest.update(len(diff_raw).to_bytes(8, "big"))
    digest.update(diff_raw)
    _hash_untracked(root, untracked_raw, digest)
    return status_raw, diff_raw, untracked_raw, digest.hexdigest()


def capture_git_provenance(path: str | Path) -> GitProvenance:
    """
    Capture the full commit and deterministic tracked/untracked delta digest.
    """

    requested = _absolute_without_following(path)
    if requested.is_file():
        requested = requested.parent
    root_raw = _run_git(requested, ("rev-parse", "--show-toplevel"))
    root = Path(root_raw.decode("utf-8", errors="surrogateescape").strip()).resolve()
    commit = _run_git(root, ("rev-parse", "--verify", "HEAD")).decode("ascii").strip()
    if len(commit) < 40 or any(character not in "0123456789abcdef" for character in commit):
        raise GitProvenanceError("Git returned an invalid full commit identifier")
    status_raw, diff_raw, untracked_raw, diff_digest = _capture_git_delta(root)
    repeated = _capture_git_delta(root)
    repeated_commit = _run_git(root, ("rev-parse", "--verify", "HEAD")).decode("ascii").strip()
    if repeated_commit != commit or repeated[:3] != (status_raw, diff_raw, untracked_raw) or repeated[3] != diff_digest:
        raise GitProvenanceError("repository changed while provenance was being captured")
    status = tuple(
        value.decode("utf-8", errors="backslashreplace")
        for value in status_raw.split(b"\0")
        if value
    )
    return GitProvenance(
        repository_root=str(root),
        commit=commit,
        dirty=bool(status_raw),
        diff_sha256=diff_digest,
        status=status,
    )


def provenance_payload(
    *,
    repository: GitProvenance,
    agent_bundle: AgentBundleManifest,
    cache: CacheManifest,
) -> Mapping[str, Any]:
    """
    Compose the integrity records into one JSON-compatible manifest payload.
    """

    return {
        "repository": repository.to_dict(),
        "agent_bundle": agent_bundle.to_dict(),
        "cache": cache.to_dict(),
    }
