"""
Exclusive, non-blocking ownership for mutable run operations.

Unix uses an advisory ``flock`` on ``<run>/.run.lock`` plus a process-local
registry so a second thread fails immediately as well. Platforms without
``fcntl`` fail closed instead of pretending that a process-local guard provides
cross-process exclusivity.
"""

from __future__ import annotations

import errno
import os
import stat
import threading
from contextlib import suppress
from pathlib import Path
from types import TracebackType

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised through a fail-closed mock.
    fcntl = None  # type: ignore[assignment]


class RunLeaseError(RuntimeError):
    """
    Report that exclusive ownership of a run could not be established.
    """


class RunLeaseUnavailableError(RunLeaseError):
    """
    Report that this platform cannot provide the required OS-level lease.
    """


_PROCESS_GUARD = threading.Lock()
_PROCESS_LEASES: set[Path] = set()


class RunLease:
    """
    Hold exclusive non-blocking ownership of one run directory.

    Use this as a context manager around the complete mutable operation, not
    merely around individual event or ledger writes. Acquisition never waits:
    another owner produces ``RunLeaseError`` immediately.
    """

    def __init__(self, run_directory: str | Path) -> None:
        """
        Bind a lease to an existing run directory without acquiring it yet.
        """

        directory = Path(run_directory).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {directory}")
        self.run_directory = directory.resolve()
        self.path = self.run_directory / ".run.lock"
        self._descriptor: int | None = None

    @property
    def acquired(self) -> bool:
        """
        Return whether this object currently owns the run lease.
        """

        return self._descriptor is not None

    def acquire(self) -> RunLease:
        """
        Acquire this lease immediately or fail without waiting.
        """

        if self._descriptor is not None:
            raise RunLeaseError(f"run lease is already acquired: {self.path}")
        if fcntl is None:
            raise RunLeaseUnavailableError(
                "exclusive run leases require fcntl; refusing to continue without an OS lock"
            )

        with _PROCESS_GUARD:
            if self.path in _PROCESS_LEASES:
                raise RunLeaseError(
                    f"run is already active in this process: {self.run_directory}"
                )
            _PROCESS_LEASES.add(self.path)

        descriptor: int | None = None
        try:
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            os.set_inheritable(descriptor, False)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RunLeaseError(f"run lease path is not a regular file: {self.path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise RunLeaseError(
                        f"run is already active in another process: {self.run_directory}"
                    ) from error
                raise
            self._descriptor = descriptor
            return self
        except BaseException:
            self._descriptor = None
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with _PROCESS_GUARD:
                _PROCESS_LEASES.discard(self.path)
            raise

    def release(self) -> None:
        """
        Release ownership; calling this on an idle lease is harmless.
        """

        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            try:
                os.close(descriptor)
            finally:
                with _PROCESS_GUARD:
                    _PROCESS_LEASES.discard(self.path)

    def __enter__(self) -> RunLease:
        """
        Acquire and return this lease for a ``with`` statement.
        """

        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the lease without suppressing an exception from its body.
        """

        self.release()
