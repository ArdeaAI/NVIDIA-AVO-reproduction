from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ardea_avo.core import (
    Candidate,
    Evaluation,
    GitLineage,
    LineageStateError,
    Score,
    StaleArtifactError,
    tree_digest,
)


def _evaluation(candidate: Candidate, value: float = 1) -> Evaluation:
    return Evaluation(
        candidate_id=candidate.candidate_id,
        artifact_digest=candidate.artifact_digest,
        score=Score(correct=True, metrics={"score": value}),
        evaluator="fixture",
    )


def _seed(lineage: GitLineage) -> Candidate:
    candidate = Candidate(
        candidate_id="v0",
        parent_id=None,
        generation=0,
        artifact_digest=tree_digest(lineage.workspace),
    )
    return lineage.commit_seed(candidate, _evaluation(candidate))


def _git(lineage: GitLineage, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            f"--git-dir={lineage.git_dir}",
            f"--work-tree={lineage.workspace}",
            *args,
        ],
        cwd=lineage.workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_git_lineage_commits_evaluated_seed_and_complete_candidate_tree(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "remove.txt").write_text("old", encoding="utf-8")
    (workspace / ".gitignore").write_text("normally-ignored.bin\n", encoding="utf-8")
    lineage = GitLineage(workspace, tmp_path / "rejected")

    seed = _seed(lineage)
    assert not (workspace / ".git").exists()
    assert seed.commit_hash == _git(lineage, "rev-parse", "HEAD").strip()
    assert "accept evaluated seed v0" in _git(lineage, "log", "-1", "--format=%B")

    (workspace / "remove.txt").unlink()
    (workspace / "nested").mkdir()
    (workspace / "nested" / "new.txt").write_text("new", encoding="utf-8")
    (workspace / "normally-ignored.bin").write_bytes(b"\x00\xff")
    proposal = Candidate(
        candidate_id="candidate-1",
        parent_id=seed.candidate_id,
        generation=1,
        artifact_digest=tree_digest(workspace),
    )
    accepted = lineage.accept(proposal, _evaluation(proposal, 2))

    tracked = set(_git(lineage, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
    assert tracked == {".gitignore", "nested/new.txt", "normally-ignored.bin"}
    lineage.ensure_matches(accepted)


def test_git_lineage_archives_binary_untracked_and_ignored_changes_then_rolls_back(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "kept.txt").write_text("accepted", encoding="utf-8")
    (workspace / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
    lineage = GitLineage(workspace, tmp_path / "rejected")
    seed = _seed(lineage)

    (workspace / "kept.txt").write_text("rejected", encoding="utf-8")
    (workspace / "new.bin").write_bytes(b"\x00\x01\xff")
    (workspace / "ignored.bin").write_bytes(b"ignored but material")
    proposal = Candidate(
        candidate_id="candidate-1",
        parent_id=seed.candidate_id,
        generation=1,
        artifact_digest=tree_digest(workspace),
    )
    evaluation = _evaluation(proposal, 0)

    archive = lineage.reject(proposal, evaluation, reason="regressed")

    assert Path(archive.patch_path).is_file()
    assert Path(archive.record_path).is_file()
    patch = Path(archive.patch_path).read_bytes()
    assert b"kept.txt" in patch
    assert b"new.bin" in patch
    assert b"ignored.bin" in patch
    record = json.loads(Path(archive.record_path).read_text(encoding="utf-8"))
    assert record["reason"] == "regressed"
    assert record["patch_sha256"] == archive.patch_sha256
    assert (workspace / "kept.txt").read_text(encoding="utf-8") == "accepted"
    assert not (workspace / "new.bin").exists()
    assert not (workspace / "ignored.bin").exists()
    lineage.ensure_matches(seed)


def test_git_lineage_refuses_stale_candidate_or_evaluation_digest(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "file").write_text("seed", encoding="utf-8")
    lineage = GitLineage(workspace, tmp_path / "rejected")
    seed = _seed(lineage)
    (workspace / "file").write_text("proposal", encoding="utf-8")
    proposal = Candidate(
        candidate_id="candidate-1",
        parent_id=seed.candidate_id,
        generation=1,
        artifact_digest=tree_digest(workspace),
    )
    (workspace / "file").write_text("changed-after-digest", encoding="utf-8")

    with pytest.raises(StaleArtifactError, match="tree digest mismatch"):
        lineage.accept(proposal, _evaluation(proposal))

    current = proposal.model_copy(update={"artifact_digest": tree_digest(workspace)})
    with pytest.raises(StaleArtifactError, match="evaluation digest mismatch"):
        lineage.accept(current, _evaluation(proposal))


def test_git_metadata_is_external_and_candidate_git_entry_cannot_redirect_host(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "file").write_text("seed", encoding="utf-8")
    lineage = GitLineage(workspace, tmp_path / "host-owned")
    seed = _seed(lineage)

    assert lineage.git_dir.is_relative_to(tmp_path / "host-owned")
    assert not (workspace / ".git").exists()
    (workspace / ".git").write_text("gitdir: /tmp/attacker-controlled", encoding="utf-8")

    assert lineage.current_commit() == seed.commit_hash
    with pytest.raises(LineageStateError, match=r"forbidden \.git entry"):
        tree_digest(workspace)
    with pytest.raises(LineageStateError, match=r"forbidden \.git entry"):
        lineage.ensure_matches(seed)
