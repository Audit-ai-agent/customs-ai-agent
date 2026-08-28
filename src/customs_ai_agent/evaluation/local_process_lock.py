"""Small OS-owned locks for restartable local build transactions.

The lock *file* is intentionally durable, while ownership belongs to the
open operating-system handle.  A host crash releases the handle automatically,
so a later non-human retry can continue without treating a stale marker as a
live owner.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
from typing import Any, BinaryIO, Iterator, cast


class LocalProcessLockBusy(RuntimeError):
    """Another living process owns the requested transaction lock."""


class LocalProcessLockInvalid(RuntimeError):
    """The lock target cannot safely be used as a local lock file."""


def _ensure_one_lock_byte(stream: BinaryIO) -> None:
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
    except OSError:
        raise LocalProcessLockInvalid from None


@contextmanager
def exclusive_local_process_lock(lock_path: Path) -> Iterator[None]:
    """Hold a non-blocking, process-owned lock without deleting its marker.

    The project supports Windows as the OCR execution platform.  A POSIX
    branch is kept for deterministic local test/inspection environments; a
    platform without either primitive fails closed rather than resurrecting
    the old create-only lock behaviour.
    """

    stream: BinaryIO | None = None
    unlock = None
    try:
        if lock_path.is_symlink() or not lock_path.parent.is_dir():
            raise LocalProcessLockInvalid
        stream = lock_path.open("a+b")
        if lock_path.is_symlink():
            raise LocalProcessLockInvalid
        _ensure_one_lock_byte(stream)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise LocalProcessLockBusy from None

            def unlock() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            try:
                import fcntl
            except ImportError:
                raise LocalProcessLockInvalid from None
            fcntl_module = cast(Any, fcntl)
            try:
                fcntl_module.flock(stream.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise LocalProcessLockBusy from None
                raise LocalProcessLockInvalid from None

            def unlock() -> None:
                fcntl_module.flock(stream.fileno(), fcntl_module.LOCK_UN)
    except (LocalProcessLockBusy, LocalProcessLockInvalid):
        if stream is not None:
            stream.close()
        raise
    except OSError:
        if stream is not None:
            stream.close()
        raise LocalProcessLockInvalid from None
    try:
        yield
    finally:
        try:
            if unlock is not None:
                unlock()
        except OSError:
            # Closing the owning descriptor still releases the OS lock.
            pass
        finally:
            if stream is not None:
                stream.close()


__all__ = [
    "LocalProcessLockBusy",
    "LocalProcessLockInvalid",
    "exclusive_local_process_lock",
]
