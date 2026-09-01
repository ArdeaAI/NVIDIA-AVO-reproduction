"""
Tests for exclusive, non-blocking run leases.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from ardea_avo.runtime import RunLease, RunLeaseError, RunLeaseUnavailableError
from ardea_avo.runtime import lease as lease_module


def _try_thread_lease(run_directory: str) -> str:
    """
    Attempt a second lease from a worker thread and return its failure text.
    """

    try:
        with RunLease(run_directory):
            return "unexpectedly acquired"
    except RunLeaseError as error:
        return str(error)


def test_context_manager_owns_exact_lock_path_and_releases(tmp_path) -> None:
    """
    Normal exit and exceptional exit both make the persistent lock file reusable.
    """

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    lease = RunLease(run_directory)
    assert lease.path == run_directory / ".run.lock"
    assert not lease.acquired

    with lease as acquired:
        assert acquired is lease
        assert lease.acquired
        assert lease.path.is_file()
    assert not lease.acquired

    with pytest.raises(LookupError), RunLease(run_directory):
        raise LookupError("body failed")
    with RunLease(run_directory):
        pass


def test_second_thread_fails_immediately(tmp_path) -> None:
    """
    Process-local tracking rejects another thread instead of relying on flock semantics.
    """

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    with RunLease(run_directory), ThreadPoolExecutor(max_workers=1) as pool:
        message = pool.submit(_try_thread_lease, str(run_directory)).result(timeout=2)
    assert "already active in this process" in message


def test_second_process_fails_without_waiting(tmp_path) -> None:
    """
    The advisory OS lock excludes an independent interpreter immediately.
    """

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    script = (
        "import sys\n"
        "from ardea_avo.runtime import RunLease, RunLeaseError\n"
        "try:\n"
        "    with RunLease(sys.argv[1]):\n"
        "        raise SystemExit(9)\n"
        "except RunLeaseError as error:\n"
        "    print(error)\n"
    )
    with RunLease(run_directory):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(run_directory)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    assert completed.returncode == 0
    assert "already active in another process" in completed.stdout


def test_missing_fcntl_fails_closed(tmp_path, monkeypatch) -> None:
    """
    Unsupported platforms never fall back to process-local-only protection.
    """

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    monkeypatch.setattr(lease_module, "fcntl", None)
    with pytest.raises(RunLeaseUnavailableError, match="refusing to continue"):
        RunLease(run_directory).acquire()


def test_reentrant_acquire_is_rejected_and_release_is_idempotent(tmp_path) -> None:
    """
    One lease object cannot silently weaken ownership through nested acquisition.
    """

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    lease = RunLease(run_directory).acquire()
    try:
        with pytest.raises(RunLeaseError, match="already acquired"):
            lease.acquire()
    finally:
        lease.release()
    lease.release()
