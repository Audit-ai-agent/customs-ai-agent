"""Detached, content-free supervision for the long Raster text activation.

The actual build/evaluation transaction remains owned by
``tools/auto_activate_synthetic_raster_text_v1.py``.  This module only starts
that fixed command in a detached worker, holds a process-owned single-run
lock, and publishes small canonical status/exit receipts.  Child stderr is
discarded; stdout is read only through a bounded strict parser that retains a
fixed status and safe error code, never OCR text or native provider details.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Literal, Protocol, cast
import uuid

from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)


_DATASET_ID = "synthetic-raster-text-v1"
_ROOT_SUFFIX = ("_local", "data", "document-preview")
_STATE_DIRECTORY_NAME = "raster-text-auto-activation-v1"
_STATUS_NAME = "status.json"
_LOCK_NAME = "activation.lock"
_RUNS_DIRECTORY = "runs"
_TARGET_TOOL_RELATIVE_PATH = Path("tools") / "auto_activate_synthetic_raster_text_v1.py"
_HEARTBEAT_SECONDS = 30.0
_POLL_SECONDS = 1.0
_WORKER_LOCK_RETRY_SECONDS = 20.0
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_CHILD_STDOUT_BYTES = 4 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_CHILD_CODE = re.compile(r"^[a-z0-9_]{1,128}$")
_PROFILE_REVISION = re.compile(r"^synthetic-raster-text-v1:[0-9a-f]{24}$")

_ActiveStatus = Literal["starting", "running"]
_TerminalStatus = Literal["succeeded", "failed", "interrupted"]


class RasterTextBackgroundActivationError(RuntimeError):
    """A fixed, content-free launcher failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        raise RasterTextBackgroundActivationError(
            "raster_text_background_command_invalid"
        ) from None
    return digest.hexdigest()


def _binding_digest(path: Path) -> str:
    normalized = os.path.normcase(str(path.resolve(strict=True)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _document_preview_root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise OSError
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:]) != _ROOT_SUFFIX
        ):
            raise OSError
        return root
    except OSError:
        raise RasterTextBackgroundActivationError("raster_text_background_root_invalid") from None


def _repository_root() -> Path:
    try:
        root = Path(__file__).resolve(strict=True).parents[3]
        target = root / _TARGET_TOOL_RELATIVE_PATH
        if root.is_symlink() or not root.is_dir() or target.is_symlink() or not target.is_file():
            raise OSError
        return root
    except (IndexError, OSError):
        raise RasterTextBackgroundActivationError(
            "raster_text_background_command_invalid"
        ) from None


def _command_binding(document_preview_root: Path) -> dict[str, str]:
    workdir = _repository_root()
    interpreter = Path(sys.executable)
    try:
        # The production bundle is Windows-only and keeps rejecting a
        # redirectable interpreter path.  POSIX virtual environments normally
        # expose ``sys.executable`` as a symlink, so CI may resolve and hash its
        # exact target instead of making the security tests Windows-host-only.
        if os.name == "nt" and interpreter.is_symlink():
            raise OSError
        interpreter = interpreter.resolve(strict=True)
        if not interpreter.is_file():
            raise OSError
    except OSError:
        raise RasterTextBackgroundActivationError(
            "raster_text_background_command_invalid"
        ) from None
    target = workdir / _TARGET_TOOL_RELATIVE_PATH
    document_preview_root_binding_sha256 = _binding_digest(document_preview_root)
    interpreter_sha256 = _sha256_file(interpreter)
    target_script_sha256 = _sha256_file(target)
    workdir_binding_sha256 = _binding_digest(workdir)
    descriptor = {
        "argument_contract": [
            "python-interpreter",
            _TARGET_TOOL_RELATIVE_PATH.as_posix(),
            "--document-preview-root",
            "bound-document-preview-root",
        ],
        "document_preview_root_binding_sha256": document_preview_root_binding_sha256,
        "interpreter_sha256": interpreter_sha256,
        "target_script_sha256": target_script_sha256,
        "workdir_binding_sha256": workdir_binding_sha256,
    }
    return {
        "command_fingerprint_sha256": hashlib.sha256(_canonical_json(descriptor)).hexdigest(),
        "document_preview_root_binding_sha256": document_preview_root_binding_sha256,
        "interpreter_sha256": interpreter_sha256,
        "target_script_sha256": target_script_sha256,
        "workdir_binding_sha256": workdir_binding_sha256,
    }


def _state_directory(document_preview_root: Path, *, create: bool) -> Path:
    try:
        local_root = document_preview_root.parents[1]
    except IndexError:
        raise RasterTextBackgroundActivationError("raster_text_background_state_invalid") from None
    directory = local_root / "state" / _STATE_DIRECTORY_NAME
    try:
        if local_root.name.casefold() != "_local" or local_root.is_symlink():
            raise OSError
        if create:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / _RUNS_DIRECTORY).mkdir(exist_ok=True)
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise OSError
        runs = directory / _RUNS_DIRECTORY
        if runs.is_symlink() or (runs.exists() and not runs.is_dir()):
            raise OSError
        return directory
    except OSError:
        raise RasterTextBackgroundActivationError("raster_text_background_state_invalid") from None


def _atomic_write(path: Path, value: object) -> bytes:
    payload = _canonical_json(value)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
            raise OSError
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass
        raise RasterTextBackgroundActivationError(
            "raster_text_background_receipt_write_failed"
        ) from None
    return payload


def _write_immutable(path: Path, value: object) -> tuple[str, bytes]:
    expected = _canonical_json(value)
    if path.exists():
        raw = _read_bytes(path)
        if raw != expected:
            raise RasterTextBackgroundActivationError("raster_text_background_receipt_conflict")
        return hashlib.sha256(raw).hexdigest(), raw
    raw = _atomic_write(path, value)
    return hashlib.sha256(raw).hexdigest(), raw


def _read_bytes(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_RECEIPT_BYTES:
            raise OSError
        return path.read_bytes()
    except OSError:
        raise RasterTextBackgroundActivationError(
            "raster_text_background_receipt_invalid"
        ) from None


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


_STATUS_KEYS = {
    "activation_pid",
    "artifact_kind",
    "child_code",
    "child_status",
    "command_fingerprint_sha256",
    "dataset_id",
    "document_preview_root_binding_sha256",
    "exit_code",
    "exit_receipt_relative_path",
    "exit_receipt_sha256",
    "finished_at_utc",
    "heartbeat_at_utc",
    "interpreter_sha256",
    "phase",
    "run_id",
    "schema_version",
    "started_at_utc",
    "status",
    "target_script_sha256",
    "termination_code",
    "workdir_binding_sha256",
    "worker_pid",
}


def _validate_status(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _STATUS_KEYS:
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    status = value.get("status")
    terminal = status in {"succeeded", "failed", "interrupted"}
    active = status in {"starting", "running"}
    hashes = (
        "command_fingerprint_sha256",
        "document_preview_root_binding_sha256",
        "interpreter_sha256",
        "target_script_sha256",
        "workdir_binding_sha256",
    )
    if (
        not (active or terminal)
        or value.get("artifact_kind") != "synthetic_raster_text_background_activation_status"
        or value.get("schema_version") != "1.0"
        or value.get("dataset_id") != _DATASET_ID
        or not isinstance(value.get("run_id"), str)
        or _RUN_ID.fullmatch(value["run_id"]) is None
        or any(
            not isinstance(value.get(key), str) or _SHA256.fullmatch(value[key]) is None
            for key in hashes
        )
        or type(value.get("worker_pid")) is not int
        or value["worker_pid"] <= 0
        or not _valid_timestamp(value.get("started_at_utc"))
        or not _valid_timestamp(value.get("heartbeat_at_utc"))
    ):
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    activation_pid = value.get("activation_pid")
    if activation_pid is not None and (type(activation_pid) is not int or activation_pid <= 0):
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    if active and any(
        value.get(key) is not None
        for key in (
            "child_code",
            "child_status",
            "exit_code",
            "exit_receipt_relative_path",
            "exit_receipt_sha256",
            "finished_at_utc",
            "termination_code",
        )
    ):
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    if status == "starting" and activation_pid is not None:
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    if status == "running" and activation_pid is None:
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    if terminal:
        relative = value.get("exit_receipt_relative_path")
        digest = value.get("exit_receipt_sha256")
        if (
            not _valid_timestamp(value.get("finished_at_utc"))
            or not isinstance(value.get("termination_code"), str)
            or _CHILD_CODE.fullmatch(value["termination_code"]) is None
            or not isinstance(relative, str)
            or re.fullmatch(r"runs/[0-9a-f]{32}\.exit\.json", relative) is None
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or (value.get("exit_code") is not None and type(value["exit_code"]) is not int)
        ):
            raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
        child_status = value.get("child_status")
        child_code = value.get("child_code")
        if child_status not in {None, "activated", "failed", "invalid"} or (
            child_code is not None
            and (not isinstance(child_code, str) or _CHILD_CODE.fullmatch(child_code) is None)
        ):
            raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    expected_phase = {
        "starting": "worker_starting",
        "running": "activation_running",
        "succeeded": "completed",
        "failed": "completed",
        "interrupted": "completed",
    }[cast(str, status)]
    if value.get("phase") != expected_phase:
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    return cast(dict[str, Any], value)


def _read_status(path: Path) -> dict[str, Any]:
    raw = _read_bytes(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RasterTextBackgroundActivationError(
            "raster_text_background_receipt_invalid"
        ) from None
    if raw != _canonical_json(value):
        raise RasterTextBackgroundActivationError("raster_text_background_receipt_invalid")
    return _validate_status(value)


def _status_payload(
    *,
    binding: dict[str, str],
    run_id: str,
    status: _ActiveStatus,
    worker_pid: int,
    activation_pid: int | None,
    started_at_utc: str,
    heartbeat_at_utc: str,
) -> dict[str, object]:
    return {
        "activation_pid": activation_pid,
        "artifact_kind": "synthetic_raster_text_background_activation_status",
        "child_code": None,
        "child_status": None,
        "command_fingerprint_sha256": binding["command_fingerprint_sha256"],
        "dataset_id": _DATASET_ID,
        "document_preview_root_binding_sha256": binding["document_preview_root_binding_sha256"],
        "exit_code": None,
        "exit_receipt_relative_path": None,
        "exit_receipt_sha256": None,
        "finished_at_utc": None,
        "heartbeat_at_utc": heartbeat_at_utc,
        "interpreter_sha256": binding["interpreter_sha256"],
        "phase": "worker_starting" if status == "starting" else "activation_running",
        "run_id": run_id,
        "schema_version": "1.0",
        "started_at_utc": started_at_utc,
        "status": status,
        "target_script_sha256": binding["target_script_sha256"],
        "termination_code": None,
        "workdir_binding_sha256": binding["workdir_binding_sha256"],
        "worker_pid": worker_pid,
    }


def _exit_payload(
    status: dict[str, Any],
    *,
    terminal_status: _TerminalStatus,
    exit_code: int | None,
    finished_at_utc: str,
    termination_code: str,
    child_status: str | None,
    child_code: str | None,
) -> dict[str, object]:
    return {
        "activation_pid": status["activation_pid"],
        "artifact_kind": "synthetic_raster_text_background_activation_exit",
        "child_code": child_code,
        "child_status": child_status,
        "command_fingerprint_sha256": status["command_fingerprint_sha256"],
        "dataset_id": _DATASET_ID,
        "document_preview_root_binding_sha256": status["document_preview_root_binding_sha256"],
        "exit_code": exit_code,
        "finished_at_utc": finished_at_utc,
        "interpreter_sha256": status["interpreter_sha256"],
        "phase": "completed",
        "run_id": status["run_id"],
        "schema_version": "1.0",
        "started_at_utc": status["started_at_utc"],
        "status": terminal_status,
        "target_script_sha256": status["target_script_sha256"],
        "termination_code": termination_code,
        "workdir_binding_sha256": status["workdir_binding_sha256"],
        "worker_pid": status["worker_pid"],
    }


def _finish(
    *,
    document_preview_root: Path,
    state_directory: Path,
    status: dict[str, Any],
    terminal_status: _TerminalStatus,
    exit_code: int | None,
    termination_code: str,
    child_status: str | None = None,
    child_code: str | None = None,
) -> dict[str, Any]:
    finished = _utc_now()
    exit_value = _exit_payload(
        status,
        terminal_status=terminal_status,
        exit_code=exit_code,
        finished_at_utc=finished,
        termination_code=termination_code,
        child_status=child_status,
        child_code=child_code,
    )
    exit_relative_path = f"{_RUNS_DIRECTORY}/{status['run_id']}.exit.json"
    exit_path = state_directory / _RUNS_DIRECTORY / f"{status['run_id']}.exit.json"
    exit_sha256, _raw = _write_immutable(exit_path, exit_value)
    terminal = {
        **status,
        "child_code": child_code,
        "child_status": child_status,
        "exit_code": exit_code,
        "exit_receipt_relative_path": exit_relative_path,
        "exit_receipt_sha256": exit_sha256,
        "finished_at_utc": finished,
        "heartbeat_at_utc": finished,
        "phase": "completed",
        "status": terminal_status,
        "termination_code": termination_code,
    }
    _validate_status(terminal)
    _atomic_write(state_directory / _STATUS_NAME, terminal)
    return terminal


class _WindowsProcessApi(Protocol):
    def open_process(self, pid: int) -> int: ...

    def get_exit_code(self, handle: int) -> int | None: ...

    def close_handle(self, handle: int) -> None: ...

    def last_error(self) -> int: ...


class _CtypesWindowsProcessApi:
    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        # ``ctypes`` exposes the dynamic DLL handle only on Windows.  Keep the
        # untyped boundary on that handle, while the public adapter remains a
        # typed protocol on every platform.
        self._kernel32: Any = kernel32

    def open_process(self, pid: int) -> int:
        # Read-only query plus wait permission.  No terminate right is opened.
        handle = self._kernel32.OpenProcess(0x00100000 | 0x00001000, False, pid)
        return int(handle) if handle else 0

    def get_exit_code(self, handle: int) -> int | None:
        code = ctypes.c_ulong()
        if not self._kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return int(code.value)

    def close_handle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)

    def last_error(self) -> int:
        return int(getattr(ctypes, "get_last_error")())


def _windows_pid_alive(pid: int, *, api: _WindowsProcessApi | None = None) -> bool:
    process_api = _CtypesWindowsProcessApi() if api is None else api
    handle = process_api.open_process(pid)
    if handle == 0:
        # Access denied means the PID exists but is not queryable.  Treat it as
        # alive so a duplicate activation remains fail-closed.
        return process_api.last_error() == 5
    try:
        exit_code = process_api.get_exit_code(handle)
        return True if exit_code is None else exit_code == 259  # STILL_ACTIVE
    finally:
        process_api.close_handle(handle)


def _pid_alive(pid: int | None, *, platform: str | None = None) -> bool:
    if pid is None or pid <= 0:
        return False
    current_platform = os.name if platform is None else platform
    if current_platform == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _worker_creation_flags() -> int:
    # ``DETACHED_PROCESS`` is not reliable for the venv redirector executable
    # used by the real Windows OCR environment: Popen succeeds, but the
    # redirector can exit before the Python worker reaches its receipt
    # handshake.  ``CREATE_NO_WINDOW`` keeps the worker independent and hidden
    # while preserving normal redirected standard handles.
    #
    # Numeric constants keep the branch testable on POSIX, where subprocess
    # intentionally does not expose the Windows-only names.
    return 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP


def _activation_creation_flags() -> int:
    return 0x08000000  # CREATE_NO_WINDOW


def _hidden_startupinfo() -> object:
    startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return startupinfo


def _spawn_worker(command: list[str], *, workdir: Path) -> subprocess.Popen[bytes]:
    try:
        if os.name == "nt":
            return subprocess.Popen(
                command,
                close_fds=True,
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_worker_creation_flags(),
                startupinfo=_hidden_startupinfo(),
            )
        return subprocess.Popen(
            command,
            close_fds=True,
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        raise RasterTextBackgroundActivationError(
            "raster_text_background_worker_unavailable"
        ) from None


def _spawn_activation(command: list[str], *, workdir: Path) -> subprocess.Popen[bytes]:
    try:
        if os.name == "nt":
            return subprocess.Popen(
                command,
                close_fds=True,
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=_activation_creation_flags(),
                startupinfo=_hidden_startupinfo(),
            )
        return subprocess.Popen(
            command,
            close_fds=True,
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        raise RasterTextBackgroundActivationError(
            "raster_text_background_activation_unavailable"
        ) from None


@dataclass
class _BoundedChildStdout:
    data: bytearray = field(default_factory=bytearray)
    overflow: bool = False

    def consume(self, stream: Any) -> None:
        try:
            while True:
                block = stream.read(1024)
                if not block:
                    return
                remaining = _MAX_CHILD_STDOUT_BYTES + 1 - len(self.data)
                if remaining > 0:
                    self.data.extend(block[:remaining])
                if len(block) > remaining or len(self.data) > _MAX_CHILD_STDOUT_BYTES:
                    self.overflow = True
        except (OSError, ValueError):
            self.overflow = True


def _stdout_reader(
    process: subprocess.Popen[bytes],
) -> tuple[_BoundedChildStdout, threading.Thread]:
    if process.stdout is None:
        raise RasterTextBackgroundActivationError(
            "raster_text_background_activation_output_invalid"
        )
    bounded = _BoundedChildStdout()
    thread = threading.Thread(target=bounded.consume, args=(process.stdout,), daemon=True)
    thread.start()
    return bounded, thread


def _parse_child_stdout(bounded: _BoundedChildStdout) -> tuple[str, str | None] | None:
    raw = bytes(bounded.data)
    if (
        bounded.overflow
        or not raw
        or len(raw) > _MAX_CHILD_STDOUT_BYTES
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
    ):
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if set(value) == {"status", "code"} and value.get("status") == "failed":
        code = value.get("code")
        if isinstance(code, str) and _CHILD_CODE.fullmatch(code) is not None:
            return "failed", code
        return None
    if (
        set(value)
        == {
            "activation_pointer_sha256",
            "dataset_id",
            "raster_text_profile_revision",
            "runtime_network",
            "status",
        }
        and value.get("status") == "activated"
        and value.get("dataset_id") == _DATASET_ID
        and value.get("runtime_network") == "disabled"
        and isinstance(value.get("activation_pointer_sha256"), str)
        and _SHA256.fullmatch(value["activation_pointer_sha256"]) is not None
        and isinstance(value.get("raster_text_profile_revision"), str)
        and _PROFILE_REVISION.fullmatch(value["raster_text_profile_revision"]) is not None
    ):
        return "activated", None
    return None


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


def start_background_activation(document_preview_root: Path) -> dict[str, object]:
    """Start one detached worker and return a content-free launch summary."""

    root = _document_preview_root(document_preview_root)
    state_directory = _state_directory(root, create=True)
    lock_path = state_directory / _LOCK_NAME
    binding = _command_binding(root)
    try:
        with exclusive_local_process_lock(lock_path):
            status_path = state_directory / _STATUS_NAME
            if status_path.exists():
                previous = _read_status(status_path)
                if previous["status"] in {"starting", "running"}:
                    if _pid_alive(previous["worker_pid"]) or _pid_alive(previous["activation_pid"]):
                        raise RasterTextBackgroundActivationError("raster_text_background_busy")
                    _finish(
                        document_preview_root=root,
                        state_directory=state_directory,
                        status=previous,
                        terminal_status="interrupted",
                        exit_code=None,
                        termination_code="background_process_missing",
                    )
            run_id = uuid.uuid4().hex
            started = _utc_now()
            worker_command = [
                sys.executable,
                str((_repository_root() / "tools" / "run_synthetic_raster_text_activation.py")),
                "_worker",
                "--document-preview-root",
                str(root),
                "--run-id",
                run_id,
                "--command-fingerprint",
                binding["command_fingerprint_sha256"],
            ]
            worker = _spawn_worker(worker_command, workdir=_repository_root())
            starting = _status_payload(
                binding=binding,
                run_id=run_id,
                status="starting",
                worker_pid=worker.pid,
                activation_pid=None,
                started_at_utc=started,
                heartbeat_at_utc=started,
            )
            _validate_status(starting)
            _atomic_write(status_path, starting)
    except LocalProcessLockBusy:
        raise RasterTextBackgroundActivationError("raster_text_background_busy") from None
    except LocalProcessLockInvalid:
        raise RasterTextBackgroundActivationError("raster_text_background_lock_invalid") from None
    return {
        "command_fingerprint_sha256": binding["command_fingerprint_sha256"],
        "run_id": run_id,
        "status": "starting",
        "worker_pid": worker.pid,
    }


def background_activation_status(document_preview_root: Path) -> dict[str, object]:
    """Read the last canonical receipt and add live-process observations."""

    root = _document_preview_root(document_preview_root)
    directory = _state_directory(root, create=False)
    path = directory / _STATUS_NAME
    if not path.exists():
        return {"dataset_id": _DATASET_ID, "status": "not_started"}
    status = _read_status(path)
    active = status["status"] in {"starting", "running"}
    return {
        **status,
        "observed_activation_alive": active and _pid_alive(status["activation_pid"]),
        "observed_worker_alive": active and _pid_alive(status["worker_pid"]),
    }


def _refresh_running_status(
    *,
    document_preview_root: Path,
    state_directory: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    updated = {
        **status,
        "heartbeat_at_utc": _utc_now(),
    }
    _validate_status(updated)
    _atomic_write(state_directory / _STATUS_NAME, updated)
    return updated


def _run_worker_locked(
    *,
    document_preview_root: Path,
    run_id: str,
    command_fingerprint: str,
) -> int:
    state_directory = _state_directory(document_preview_root, create=False)
    status_path = state_directory / _STATUS_NAME
    status = _read_status(status_path)
    binding = _command_binding(document_preview_root)
    if (
        status["status"] != "starting"
        or status["run_id"] != run_id
        or status["command_fingerprint_sha256"] != command_fingerprint
        or binding["command_fingerprint_sha256"] != command_fingerprint
        or any(status[key] != binding[key] for key in binding)
    ):
        raise RasterTextBackgroundActivationError("raster_text_background_worker_invalid")

    # On Windows, ``sys.executable`` inside a venv is a redirector.  The PID
    # returned by Popen belongs to that short-lived launcher, while
    # ``os.getpid()`` belongs to the real base-interpreter process.  Requiring
    # those values to be equal made every real detached launch exit before it
    # could publish an exit receipt.  The exact run ID and command binding are
    # already checked above, and this worker owns the process lock, so it can
    # safely replace the launcher PID with the supervising interpreter PID.
    claimed = {
        **status,
        "worker_pid": os.getpid(),
        "heartbeat_at_utc": _utc_now(),
    }
    _validate_status(claimed)
    _atomic_write(status_path, claimed)
    status = claimed

    workdir = _repository_root()
    command = [
        sys.executable,
        str(workdir / _TARGET_TOOL_RELATIVE_PATH),
        "--document-preview-root",
        str(document_preview_root),
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        process = _spawn_activation(command, workdir=workdir)
        bounded_stdout, stdout_thread = _stdout_reader(process)
        running = {
            **status,
            "activation_pid": process.pid,
            "heartbeat_at_utc": _utc_now(),
            "phase": "activation_running",
            "status": "running",
        }
        _validate_status(running)
        _atomic_write(status_path, running)
        status = running
        last_heartbeat = time.monotonic()
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                break
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_SECONDS:
                status = _refresh_running_status(
                    document_preview_root=document_preview_root,
                    state_directory=state_directory,
                    status=status,
                )
                last_heartbeat = now
            time.sleep(_POLL_SECONDS)
        stdout_thread.join(timeout=5)
        parsed_output = None if stdout_thread.is_alive() else _parse_child_stdout(bounded_stdout)
        if exit_code == 0 and parsed_output == ("activated", None):
            final_status: _TerminalStatus = "succeeded"
            termination_code = "activation_command_succeeded"
            child_status, child_code = parsed_output
        elif exit_code != 0 and parsed_output is not None and parsed_output[0] == "failed":
            final_status = "failed"
            termination_code = "activation_command_failed"
            child_status, child_code = parsed_output
        else:
            final_status = "failed"
            termination_code = "activation_command_output_invalid"
            child_status, child_code = "invalid", None
        _finish(
            document_preview_root=document_preview_root,
            state_directory=state_directory,
            status=status,
            terminal_status=final_status,
            exit_code=exit_code,
            termination_code=termination_code,
            child_status=child_status,
            child_code=child_code,
        )
        return 0 if final_status == "succeeded" else 2
    except Exception as error:
        if process is not None:
            _terminate(process)
        # A valid starting/running status is safe to finalize.  Never persist
        # the exception text; only the bounded code owned by this module.
        termination_code = (
            error.code
            if isinstance(error, RasterTextBackgroundActivationError)
            else "raster_text_background_worker_failed"
        )
        try:
            current = _read_status(status_path)
            if current["run_id"] == run_id and current["status"] in {"starting", "running"}:
                _finish(
                    document_preview_root=document_preview_root,
                    state_directory=state_directory,
                    status=current,
                    terminal_status="failed",
                    exit_code=None,
                    termination_code=termination_code,
                )
        except RasterTextBackgroundActivationError:
            pass
        return 3


def run_background_worker(
    document_preview_root: Path,
    *,
    run_id: str,
    command_fingerprint: str,
) -> int:
    """Acquire the long-lived lock and supervise the fixed activation child."""

    root = _document_preview_root(document_preview_root)
    if _RUN_ID.fullmatch(run_id) is None or _SHA256.fullmatch(command_fingerprint) is None:
        return 3
    state_directory = _state_directory(root, create=False)
    lock_path = state_directory / _LOCK_NAME
    deadline = time.monotonic() + _WORKER_LOCK_RETRY_SECONDS
    while True:
        try:
            with cast(AbstractContextManager[None], exclusive_local_process_lock(lock_path)):
                return _run_worker_locked(
                    document_preview_root=root,
                    run_id=run_id,
                    command_fingerprint=command_fingerprint,
                )
        except LocalProcessLockBusy:
            if time.monotonic() >= deadline:
                return 3
            time.sleep(0.1)
        except LocalProcessLockInvalid:
            return 3
        except RasterTextBackgroundActivationError:
            return 3


__all__ = [
    "RasterTextBackgroundActivationError",
    "background_activation_status",
    "run_background_worker",
    "start_background_activation",
]
