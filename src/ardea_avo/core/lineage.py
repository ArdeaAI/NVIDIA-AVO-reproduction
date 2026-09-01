"""
Git-backed accepted lineage and external rejected-patch archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ardea_avo.core.digest import archive_tree_digest, tree_digest
from ardea_avo.core.exceptions import LineageError, LineageStateError, StaleArtifactError
from ardea_avo.core.models import Candidate, Evaluation, RejectionArchive


class GitLineage:
    """
    Store accepted full candidate trees as commits in a dedicated Git worktree.

    Rejected patches are written outside the worktree before destructive rollback.
    Callers must dedicate ``workspace`` to candidate contents because rejection uses
    ``git reset --hard`` and ``git clean -fdx`` to restore the accepted commit exactly.
    """

    def __init__(
        self,
        workspace: Path,
        archive_dir: Path,
        *,
        git_dir: Path | None = None,
        git_executable: str = "git",
        author_name: str = "Ardea AVO",
        author_email: str = "avo@localhost",
    ) -> None:
        """
        Initialize or open a dedicated lineage repository.
        """
        self._workspace = workspace.resolve()
        self._archive_dir = archive_dir.resolve()
        self._git_dir = (
            git_dir.resolve() if git_dir is not None else (self._archive_dir / "lineage.git").resolve()
        )
        self._git = git_executable
        if self._workspace == Path(self._workspace.anchor):
            raise LineageError("filesystem root cannot be used as a candidate workspace")
        if self._archive_dir == self._workspace or self._archive_dir.is_relative_to(self._workspace):
            raise LineageError("rejection archive must be outside the candidate workspace")
        if self._git_dir == self._workspace or self._git_dir.is_relative_to(self._workspace):
            raise LineageError("Git metadata must be outside the candidate workspace")
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        tree_digest(self._workspace)
        if not self._git_dir.exists():
            self._git_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run_initialization()
        elif not self._git_dir.is_dir():
            raise LineageError(f"Git metadata path is not a directory: {self._git_dir}")
        is_bare = self._run_git("config", "--bool", "core.bare").strip()
        if is_bare != "true":
            raise LineageError(f"external Git metadata repository must be bare: {self._git_dir}")
        top_level = Path(self._run_git("rev-parse", "--show-toplevel").strip()).resolve()
        if top_level != self._workspace:
            raise LineageError(
                f"candidate workspace must be a dedicated Git root: expected={self._workspace}, "
                f"actual={top_level}"
            )
        self._run_git("config", "user.name", author_name)
        self._run_git("config", "user.email", author_email)

    @property
    def workspace(self) -> Path:
        """
        Return the dedicated mutable candidate worktree.
        """
        return self._workspace

    @property
    def archive_dir(self) -> Path:
        """
        Return the external rejected-attempt archive directory.
        """
        return self._archive_dir

    @property
    def git_dir(self) -> Path:
        """
        Return the host-owned Git metadata directory outside the candidate tree.
        """
        return self._git_dir

    def commit_seed(self, candidate: Candidate, evaluation: Evaluation) -> Candidate:
        """
        Commit a digest-bound, evaluated generation-zero seed as ``v0``.
        """
        if self._has_head():
            raise LineageStateError("cannot commit seed: lineage already has a HEAD commit")
        if candidate.generation != 0 or candidate.parent_id is not None:
            raise LineageStateError("seed candidate must have generation 0 and no parent")
        if candidate.commit_hash is not None:
            raise LineageStateError("seed candidate must not already have a commit hash")
        if not evaluation.score.correct:
            raise LineageStateError("cannot commit a seed that failed the correctness gate")
        self._validate_binding(candidate, evaluation)
        self._ensure_current_digest(candidate)
        return self._commit(candidate, evaluation, subject="avo: accept evaluated seed v0")

    def accept(self, candidate: Candidate, evaluation: Evaluation) -> Candidate:
        """
        Commit an already-authorized proposal as the next full-tree lineage version.
        """
        if not self._has_head():
            raise LineageStateError("cannot accept a proposal before committing a seed")
        if candidate.parent_id is None or candidate.generation < 1:
            raise LineageStateError("accepted proposal must identify a parent and positive generation")
        if candidate.commit_hash is not None:
            raise LineageStateError("proposal must not already have a commit hash")
        if not evaluation.score.correct:
            raise LineageStateError("cannot accept a candidate that failed the correctness gate")
        head_candidate = self._head_candidate()
        if candidate.parent_id != head_candidate.candidate_id:
            raise LineageStateError(
                f"proposal parent does not match lineage HEAD: expected={head_candidate.candidate_id}, "
                f"actual={candidate.parent_id}"
            )
        if candidate.generation != head_candidate.generation + 1:
            raise LineageStateError(
                f"proposal generation must follow lineage HEAD: expected={head_candidate.generation + 1}, "
                f"actual={candidate.generation}"
            )
        self._validate_binding(candidate, evaluation)
        self._ensure_current_digest(candidate)
        return self._commit(
            candidate,
            evaluation,
            subject=f"avo: accept generation {candidate.generation} ({candidate.candidate_id})",
        )

    def reject(
        self,
        candidate: Candidate,
        evaluation: Evaluation | None,
        *,
        reason: str,
    ) -> RejectionArchive:
        """
        Archive all tracked, untracked, ignored, and binary changes, then roll back.
        """
        if not self._has_head():
            raise LineageStateError("cannot reject a proposal before committing a seed")
        if not reason.strip():
            raise ValueError("rejection reason must not be empty")
        forbidden_entries = self._forbidden_git_entries()
        if forbidden_entries:
            actual = archive_tree_digest(self._workspace)
            if actual != candidate.artifact_digest:
                raise StaleArtifactError(
                    f"invalid candidate tree digest mismatch: expected={candidate.artifact_digest}, "
                    f"actual={actual}"
                )
        else:
            self._ensure_current_digest(candidate)
        if evaluation is not None:
            self._validate_binding(candidate, evaluation)

        archive: RejectionArchive | None = None
        primary_error: Exception | None = None
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            stem = f"{timestamp}_{candidate.candidate_id}"
            quarantined = self._quarantine_forbidden_entries(forbidden_entries, stem)
            self._run_git("add", "-A", "-f", "--", ".")
            patch = self._run_git_bytes("diff", "--cached", "--binary", "--full-index", "HEAD", "--")
            patch_digest = hashlib.sha256(patch).hexdigest()
            patch_path = self._archive_dir / f"{stem}.patch"
            record_path = self._archive_dir / f"{stem}.json"
            patch_path.write_bytes(patch)
            record = {
                "schema_version": 1,
                "candidate": candidate.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json") if evaluation is not None else None,
                "reason": reason,
                "patch_sha256": patch_digest,
                "quarantined_entries": quarantined,
                "archived_at": datetime.now(UTC).isoformat(),
            }
            record_path.write_text(
                json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            archive = RejectionArchive(
                record_path=str(record_path),
                patch_path=str(patch_path),
                patch_sha256=patch_digest,
            )
        except Exception as exc:
            primary_error = exc
        finally:
            try:
                self._rollback()
            except Exception as rollback_error:
                if primary_error is not None:
                    raise LineageError(
                        f"rejection archive failed ({primary_error}); rollback also failed: {rollback_error}"
                    ) from rollback_error
                raise

        if primary_error is not None:
            raise LineageError(f"could not archive rejected candidate: {primary_error}") from primary_error
        if archive is None:
            raise LineageError("rejection archive was not created")
        return archive

    def ensure_matches(self, candidate: Candidate) -> None:
        """
        Verify that HEAD, worktree status, and tree digest match a checkpoint candidate.
        """
        if candidate.commit_hash is None:
            raise LineageStateError("accepted candidate has no commit hash")
        head = self.current_commit()
        if head != candidate.commit_hash:
            raise LineageStateError(
                f"lineage HEAD does not match candidate: expected={candidate.commit_hash}, actual={head}"
            )
        self._ensure_current_digest(candidate)
        status = self._run_git("status", "--porcelain=v1", "--untracked-files=all")
        if status:
            raise LineageStateError("candidate worktree contains uncommitted changes")

    def current_commit(self) -> str:
        """
        Return the full object identifier of the accepted HEAD commit.
        """
        if not self._has_head():
            raise LineageStateError("lineage has no HEAD commit")
        return self._run_git("rev-parse", "HEAD").strip()

    def _head_candidate(self) -> Candidate:
        message = self._run_git("log", "-1", "--format=%B")
        prefix = "AVO-Record: "
        payload_line = next((line for line in message.splitlines() if line.startswith(prefix)), None)
        if payload_line is None:
            raise LineageStateError("lineage HEAD is missing its AVO candidate record")
        try:
            payload = json.loads(payload_line.removeprefix(prefix))
            return Candidate.model_validate(payload["candidate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageStateError("lineage HEAD contains an invalid AVO candidate record") from exc

    def _commit(self, candidate: Candidate, evaluation: Evaluation, *, subject: str) -> Candidate:
        self._run_git("add", "-A", "-f", "--", ".")
        payload = json.dumps(
            {
                "candidate": candidate.model_dump(mode="json"),
                "evaluation": evaluation.model_dump(mode="json"),
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._run_git(
            "commit",
            "--allow-empty",
            "--no-gpg-sign",
            "-m",
            subject,
            "-m",
            f"AVO-Record: {payload}",
        )
        commit_hash = self.current_commit()
        committed = candidate.model_copy(update={"commit_hash": commit_hash})
        self.ensure_matches(committed)
        return committed

    def _validate_binding(self, candidate: Candidate, evaluation: Evaluation) -> None:
        if evaluation.candidate_id != candidate.candidate_id:
            raise StaleArtifactError(
                f"evaluation candidate mismatch: expected={candidate.candidate_id}, "
                f"actual={evaluation.candidate_id}"
            )
        if evaluation.artifact_digest != candidate.artifact_digest:
            raise StaleArtifactError(
                f"evaluation digest mismatch: expected={candidate.artifact_digest}, "
                f"actual={evaluation.artifact_digest}"
            )

    def _ensure_current_digest(self, candidate: Candidate) -> None:
        actual = tree_digest(self._workspace)
        if actual != candidate.artifact_digest:
            raise StaleArtifactError(
                f"candidate tree digest mismatch: expected={candidate.artifact_digest}, actual={actual}"
            )

    def _forbidden_git_entries(self) -> list[Path]:
        entries: list[Path] = []
        for directory, directory_names, file_names in os.walk(self._workspace, followlinks=False):
            current = Path(directory)
            forbidden_directories = [
                name for name in directory_names if name.casefold() == ".git"
            ]
            forbidden_files = [name for name in file_names if name.casefold() == ".git"]
            entries.extend(current / name for name in forbidden_directories)
            entries.extend(current / name for name in forbidden_files)
            directory_names[:] = [
                name for name in directory_names if name.casefold() != ".git"
            ]
        return sorted(entries, key=lambda path: path.relative_to(self._workspace).as_posix())

    def _quarantine_forbidden_entries(self, entries: list[Path], stem: str) -> list[dict[str, str]]:
        quarantined: list[dict[str, str]] = []
        quarantine_root = self._archive_dir / "invalid" / stem
        for source in entries:
            relative = source.relative_to(self._workspace)
            destination = quarantine_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            quarantined.append(
                {
                    "candidate_path": relative.as_posix(),
                    "archive_path": str(destination),
                }
            )
        return quarantined

    def _rollback(self) -> None:
        self._run_git("reset", "--hard", "HEAD")
        self._run_git("clean", "-fdx", "--")

    def _has_head(self) -> bool:
        try:
            completed = subprocess.run(
                [*self._git_prefix(), "rev-parse", "--verify", "HEAD"],
                cwd=self._workspace,
                check=False,
                capture_output=True,
                text=True,
                env=self._git_environment(),
            )
        except OSError as exc:
            raise LineageError(f"could not run Git: {exc}") from exc
        return completed.returncode == 0

    def _run_git(self, *arguments: str) -> str:
        try:
            completed = subprocess.run(
                [*self._git_prefix(), *arguments],
                cwd=self._workspace,
                check=False,
                capture_output=True,
                text=True,
                env=self._git_environment(),
            )
        except OSError as exc:
            raise LineageError(f"could not run Git: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise LineageError(
                f"git {' '.join(arguments[:2])} failed with status {completed.returncode}: {detail}"
            )
        return completed.stdout

    def _run_git_bytes(self, *arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                [*self._git_prefix(), *arguments],
                cwd=self._workspace,
                check=False,
                capture_output=True,
                env=self._git_environment(),
            )
        except OSError as exc:
            raise LineageError(f"could not run Git: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip() or "no stderr"
            raise LineageError(
                f"git {' '.join(arguments[:2])} failed with status {completed.returncode}: {detail}"
            )
        return completed.stdout

    def _run_initialization(self) -> None:
        try:
            completed = subprocess.run(
                [
                    self._git,
                    "init",
                    "--bare",
                    "--initial-branch=avo-lineage",
                    str(self._git_dir),
                ],
                cwd=self._archive_dir,
                check=False,
                capture_output=True,
                text=True,
                env=self._git_environment(),
            )
        except OSError as exc:
            raise LineageError(f"could not initialize external Git metadata: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise LineageError(f"could not initialize external Git metadata: {detail}")

    def _git_prefix(self) -> list[str]:
        return [
            self._git,
            "-c",
            f"core.hooksPath={os.devnull}",
            f"--git-dir={self._git_dir}",
            f"--work-tree={self._workspace}",
        ]

    @staticmethod
    def _git_environment() -> dict[str, str]:
        environment = {
            name: value for name, value in os.environ.items() if not name.startswith("GIT_")
        }
        environment.update(
            {
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "LC_ALL": "C",
            }
        )
        return environment
