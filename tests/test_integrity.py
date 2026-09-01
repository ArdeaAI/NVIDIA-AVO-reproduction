"""
Offline integrity and contamination tests.
"""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

from ardea_avo.integrity import (
    ContaminationError,
    ContaminationKind,
    UnsafeFilesystemEntry,
    agent_bundle_digest,
    capture_agent_bundle,
    capture_cache_manifest,
    capture_git_provenance,
    provenance_payload,
    scan_agent_bundle,
)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def make_repository(root: Path) -> str:
    root.mkdir()
    git(root, "init", "--quiet")
    git(root, "config", "user.name", "Integrity Test")
    git(root, "config", "user.email", "integrity@example.invalid")
    write(root / "tracked.txt", "original\n")
    git(root, "add", "tracked.txt")
    git(root, "commit", "--quiet", "-m", "seed")
    return git(root, "rev-parse", "HEAD")


def test_git_provenance_captures_clean_commit_and_all_dirty_bytes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit = make_repository(repository)
    clean = capture_git_provenance(repository / "tracked.txt")
    assert clean.commit == commit
    assert clean.dirty is False
    assert clean.status == ()
    assert len(clean.diff_sha256) == 64
    assert json.loads(json.dumps(clean.to_dict()))["commit"] == commit

    write(repository / "tracked.txt", "changed\n")
    write(repository / "new.txt", "first\n")
    dirty = capture_git_provenance(repository)
    assert dirty.dirty is True
    assert any("tracked.txt" in item for item in dirty.status)
    assert any("new.txt" in item for item in dirty.status)
    assert dirty.diff_sha256 != clean.diff_sha256
    assert capture_git_provenance(repository).diff_sha256 == dirty.diff_sha256

    write(repository / "new.txt", "second\n")
    assert capture_git_provenance(repository).diff_sha256 != dirty.diff_sha256


def test_untracked_executable_bit_and_symlink_target_affect_git_diff_digest(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    make_repository(repository)
    script = repository / "script.sh"
    write(script, "#!/bin/sh\n")
    plain = capture_git_provenance(repository).diff_sha256
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    executable = capture_git_provenance(repository).diff_sha256
    assert executable != plain

    link = repository / "pointer"
    link.symlink_to("tracked.txt")
    first_link = capture_git_provenance(repository).diff_sha256
    link.unlink()
    link.symlink_to("script.sh")
    assert capture_git_provenance(repository).diff_sha256 != first_link


def test_bundle_digest_is_location_and_creation_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write(first / "b" / "two.md", "two\n")
    write(first / "one.md", "one\n")
    write(second / "one.md", "one\n")
    write(second / "b" / "two.md", "two\n")
    first_manifest = capture_agent_bundle(first)
    second_manifest = capture_agent_bundle(second)
    assert first_manifest.sha256 == second_manifest.sha256
    assert [entry.path for entry in first_manifest.files] == ["b/two.md", "one.md"]
    assert first_manifest.contamination.clean is True
    assert json.dumps(first_manifest.to_dict(), sort_keys=True)

    write(second / "one.md", "changed\n")
    assert agent_bundle_digest(second) != first_manifest.sha256


def test_bundle_scan_rejects_symlinks_git_metadata_and_full_game_ids(tmp_path: Path) -> None:
    game_id = "ls20-9607627b"
    bundle = tmp_path / "bundle"
    write(bundle / "system.md", f"Prior solution for {game_id.upper()}\n")
    write(bundle / game_id / "notes.md", "hidden\n")
    write(bundle / ".git" / "config", "not really git\n")
    (bundle / "outside-link").symlink_to(tmp_path / "outside")
    report = scan_agent_bundle(bundle, [game_id])
    kinds = {finding.kind for finding in report.findings}
    assert report.clean is False
    assert ContaminationKind.FORBIDDEN_GAME_ID in kinds
    assert ContaminationKind.RESERVED_GIT in kinds
    assert ContaminationKind.SYMLINK in kinds
    assert report.scanned_files == 2
    with pytest.raises(ContaminationError) as caught:
        capture_agent_bundle(bundle, [game_id])
    assert caught.value.report.clean is False


def test_bundle_root_symlink_is_reported_without_following(tmp_path: Path) -> None:
    real = tmp_path / "real"
    write(real / "system.md", "generic\n")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    report = scan_agent_bundle(link)
    assert [(finding.kind, finding.path) for finding in report.findings] == [
        (ContaminationKind.SYMLINK, ".")
    ]


def test_cache_manifest_is_deterministic_json_and_rejects_symlinks(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    write(cache / "z.dat", "zzz")
    write(cache / "nested" / "a.dat", "a")
    manifest = capture_cache_manifest(cache)
    assert manifest.file_count == 2
    assert manifest.total_bytes == 4
    assert [entry.path for entry in manifest.files] == ["nested/a.dat", "z.dat"]
    assert json.loads(json.dumps(manifest.to_dict()))["sha256"] == manifest.sha256
    assert capture_cache_manifest(cache).sha256 == manifest.sha256

    (cache / "link").symlink_to(cache / "z.dat")
    with pytest.raises(UnsafeFilesystemEntry, match="symlinks"):
        capture_cache_manifest(cache)


def test_composed_provenance_payload_is_json_compatible(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    make_repository(repository)
    bundle = tmp_path / "bundle"
    cache = tmp_path / "cache"
    write(bundle / "system.md", "generic\n")
    write(cache / "game.dat", "cache\n")
    payload = provenance_payload(
        repository=capture_git_provenance(repository),
        agent_bundle=capture_agent_bundle(bundle),
        cache=capture_cache_manifest(cache),
    )
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert json.loads(encoded)["agent_bundle"]["contamination"]["clean"] is True
