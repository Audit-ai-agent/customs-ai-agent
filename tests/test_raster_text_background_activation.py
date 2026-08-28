"""Content-free process supervision tests; no OCR child is started."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Iterator

import pytest

from customs_ai_agent.evaluation import raster_text_background_activation as background
from customs_ai_agent.evaluation.local_process_lock import LocalProcessLockBusy
from tools import run_synthetic_raster_text_activation as cli


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _activated_stdout() -> bytes:
    return (
        json.dumps(
            {
                "activation_pointer_sha256": "b" * 64,
                "dataset_id": "synthetic-raster-text-v1",
                "raster_text_profile_revision": "synthetic-raster-text-v1:" + "a" * 24,
                "runtime_network": "disabled",
                "status": "activated",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


class _Process:
    def __init__(self, *, pid: int, exit_code: int = 0, stdout: bytes = b"") -> None:
        self.pid = pid
        self._exit_code = exit_code
        self.stdout = io.BytesIO(stdout)

    def poll(self) -> int:
        return self._exit_code

    def terminate(self) -> None:
        raise AssertionError("a completed fake process must not be terminated")


def _starting_status(root: Path, *, worker_pid: int | None = None) -> dict[str, object]:
    state = background._state_directory(root, create=True)  # noqa: SLF001
    binding = background._command_binding(root)  # noqa: SLF001
    started = "2026-08-27T00:00:00Z"
    value = background._status_payload(  # noqa: SLF001
        binding=binding,
        run_id="a" * 32,
        status="starting",
        worker_pid=os.getpid() if worker_pid is None else worker_pid,
        activation_pid=None,
        started_at_utc=started,
        heartbeat_at_utc=started,
    )
    background._atomic_write(state / "status.json", value)  # noqa: SLF001
    return value


def _state(root: Path) -> Path:
    return root.parents[1] / "state" / "raster-text-auto-activation-v1"


def test_start_writes_state_outside_data_and_uses_fixed_worker_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    calls: list[tuple[list[str], Path]] = []

    def spawn(command: list[str], *, workdir: Path) -> _Process:
        calls.append((command, workdir))
        return _Process(pid=4321)

    monkeypatch.setattr(background, "_spawn_worker", spawn)
    result = background.start_background_activation(root)

    assert result["status"] == "starting"
    assert result["worker_pid"] == 4321
    command, workdir = calls[0]
    assert workdir == background._repository_root()  # noqa: SLF001
    assert command[0] == os.sys.executable
    assert Path(command[1]).name == "run_synthetic_raster_text_activation.py"
    assert command[2:4] == ["_worker", "--document-preview-root"]
    assert command[4] == str(root.resolve())
    assert command[5] == "--run-id"
    assert command[7] == "--command-fingerprint"

    status_path = _state(root) / "status.json"
    assert status_path.is_file()
    assert not (root / "synthetic-raster-text-v1" / "background-activation-v1").exists()
    raw = status_path.read_bytes()
    parsed = json.loads(raw)
    assert raw == background._canonical_json(parsed)  # noqa: SLF001
    assert parsed["phase"] == "worker_starting"
    serialized = raw.decode("ascii")
    for forbidden in (
        "stdout",
        "stderr",
        "ocr_text",
        "model_error",
        "checkpoint_files",
        str(root.resolve()),
    ):
        assert forbidden not in serialized


def test_duplicate_start_fails_closed_when_process_lock_is_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)

    @contextmanager
    def busy(_path: Path) -> Iterator[None]:
        raise LocalProcessLockBusy
        yield

    monkeypatch.setattr(background, "exclusive_local_process_lock", busy)
    with pytest.raises(background.RasterTextBackgroundActivationError) as captured:
        background.start_background_activation(root)
    assert captured.value.code == "raster_text_background_busy"


def test_live_receipt_blocks_a_second_worker_even_after_lock_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    _starting_status(root, worker_pid=9876)
    monkeypatch.setattr(background, "_pid_alive", lambda pid: pid == 9876)
    monkeypatch.setattr(
        background,
        "_spawn_worker",
        lambda *_args, **_kwargs: pytest.fail("duplicate worker must not be spawned"),
    )
    with pytest.raises(background.RasterTextBackgroundActivationError) as captured:
        background.start_background_activation(root)
    assert captured.value.code == "raster_text_background_busy"


def test_dead_active_receipt_is_closed_as_interrupted_before_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    previous = _starting_status(root, worker_pid=9876)
    monkeypatch.setattr(background, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        background,
        "_spawn_worker",
        lambda *_args, **_kwargs: _Process(pid=4321),
    )
    current = background.start_background_activation(root)

    assert current["run_id"] != previous["run_id"]
    old_exit = _state(root) / "runs" / f"{previous['run_id']}.exit.json"
    receipt = json.loads(old_exit.read_bytes())
    assert receipt["status"] == "interrupted"
    assert receipt["exit_code"] is None
    assert receipt["termination_code"] == "background_process_missing"
    assert receipt["child_status"] is None


@pytest.mark.parametrize(
    (
        "exit_code",
        "stdout",
        "expected_status",
        "expected_child_status",
        "expected_code",
        "expected_termination",
        "worker_result",
    ),
    [
        (0, _activated_stdout(), "succeeded", "activated", None, "activation_command_succeeded", 0),
        (
            2,
            b'{"status":"failed","code":"raster_text_sealed_gate_failed"}\n',
            "failed",
            "failed",
            "raster_text_sealed_gate_failed",
            "activation_command_failed",
            2,
        ),
        (
            0,
            b"x" * 5000,
            "failed",
            "invalid",
            None,
            "activation_command_output_invalid",
            2,
        ),
    ],
)
def test_worker_preserves_only_bounded_child_status_and_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    stdout: bytes,
    expected_status: str,
    expected_child_status: str,
    expected_code: str | None,
    expected_termination: str,
    worker_result: int,
) -> None:
    root = _preview_root(tmp_path).resolve()
    starting = _starting_status(root)
    calls: list[tuple[list[str], Path]] = []

    def spawn(command: list[str], *, workdir: Path) -> _Process:
        calls.append((command, workdir))
        return _Process(pid=2468, exit_code=exit_code, stdout=stdout)

    monkeypatch.setattr(background, "_spawn_activation", spawn)
    result = background._run_worker_locked(  # noqa: SLF001
        document_preview_root=root,
        run_id=str(starting["run_id"]),
        command_fingerprint=str(starting["command_fingerprint_sha256"]),
    )

    assert result == worker_result
    command, workdir = calls[0]
    assert command == [
        os.sys.executable,
        str(workdir / "tools" / "auto_activate_synthetic_raster_text_v1.py"),
        "--document-preview-root",
        str(root),
    ]
    status = background.background_activation_status(root)
    assert status["status"] == expected_status
    assert status["exit_code"] == exit_code
    assert status["child_status"] == expected_child_status
    assert status["child_code"] == expected_code
    assert status["termination_code"] == expected_termination
    exit_path = _state(root) / Path(str(status["exit_receipt_relative_path"]))
    raw = exit_path.read_bytes()
    assert raw == background._canonical_json(json.loads(raw))  # noqa: SLF001
    assert status["exit_receipt_sha256"] == hashlib.sha256(raw).hexdigest()
    serialized = raw.decode("ascii")
    assert str(root) not in serialized
    for forbidden in ("stdout", "stderr", "traceback", "ocr_text", "model_error"):
        assert forbidden not in serialized


def test_worker_claims_actual_interpreter_pid_from_windows_venv_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path).resolve()
    starting = _starting_status(root, worker_pid=4321)
    monkeypatch.setattr(
        background,
        "_spawn_activation",
        lambda *_args, **_kwargs: _Process(
            pid=2468, exit_code=0, stdout=_activated_stdout()
        ),
    )

    result = background._run_worker_locked(  # noqa: SLF001
        document_preview_root=root,
        run_id=str(starting["run_id"]),
        command_fingerprint=str(starting["command_fingerprint_sha256"]),
    )

    assert result == 0
    status = background.background_activation_status(root)
    assert status["status"] == "succeeded"
    assert status["worker_pid"] == os.getpid()


class _WindowsApi:
    def __init__(self, *, handle: int, exit_code: int | None, last_error: int = 0) -> None:
        self.handle = handle
        self.exit_code = exit_code
        self.error = last_error
        self.opened: list[int] = []
        self.closed: list[int] = []

    def open_process(self, pid: int) -> int:
        self.opened.append(pid)
        return self.handle

    def get_exit_code(self, handle: int) -> int | None:
        assert handle == self.handle
        return self.exit_code

    def close_handle(self, handle: int) -> None:
        self.closed.append(handle)

    def last_error(self) -> int:
        return self.error


@pytest.mark.parametrize(
    ("api", "expected", "closed"),
    [
        (_WindowsApi(handle=10, exit_code=259), True, [10]),
        (_WindowsApi(handle=11, exit_code=0), False, [11]),
        (_WindowsApi(handle=0, exit_code=None, last_error=5), True, []),
        (_WindowsApi(handle=0, exit_code=None, last_error=87), False, []),
    ],
)
def test_windows_pid_probe_is_read_only_and_closes_query_handle(
    api: _WindowsApi, expected: bool, closed: list[int]
) -> None:
    assert background._windows_pid_alive(1234, api=api) is expected  # noqa: SLF001
    assert api.opened == [1234]
    assert api.closed == closed


def test_windows_pid_branch_never_calls_os_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(background, "_windows_pid_alive", lambda pid: pid == 1234)
    monkeypatch.setattr(
        background.os,
        "kill",
        lambda *_args: pytest.fail("Windows status must not call os.kill"),
    )
    assert background._pid_alive(1234, platform="nt") is True  # noqa: SLF001


def test_noncanonical_status_receipt_fails_closed(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)
    _starting_status(root)
    path = _state(root) / "status.json"
    value = json.loads(path.read_bytes())
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(background.RasterTextBackgroundActivationError) as captured:
        background.background_activation_status(root)
    assert captured.value.code == "raster_text_background_receipt_invalid"


def test_spawn_contract_hides_windows_children_and_bounds_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def popen(command: list[str], **kwargs: object) -> _Process:
        calls.append((command, kwargs))
        return _Process(pid=1111)

    monkeypatch.setattr(background.subprocess, "Popen", popen)
    workdir = tmp_path.resolve()
    background._spawn_worker(["fixed-worker"], workdir=workdir)  # noqa: SLF001
    background._spawn_activation(["fixed-activation"], workdir=workdir)  # noqa: SLF001

    worker_kwargs = calls[0][1]
    activation_kwargs = calls[1][1]
    assert worker_kwargs["stdout"] == subprocess.DEVNULL
    assert activation_kwargs["stdout"] == subprocess.PIPE
    for kwargs in (worker_kwargs, activation_kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["close_fds"] is True
        assert "shell" not in kwargs
    if os.name == "nt":
        assert worker_kwargs["creationflags"] == background._worker_creation_flags()  # noqa: SLF001
        assert activation_kwargs["creationflags"] == background._activation_creation_flags()  # noqa: SLF001
        assert "startupinfo" in worker_kwargs
        assert "startupinfo" in activation_kwargs
    else:
        assert worker_kwargs["start_new_session"] is True
        assert "start_new_session" not in activation_kwargs


def test_cli_start_status_and_failure_are_content_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _preview_root(tmp_path)
    monkeypatch.setattr(
        cli,
        "start_background_activation",
        lambda _root: {"run_id": "a" * 32, "status": "starting", "worker_pid": 123},
    )
    assert cli.main(["start", "--document-preview-root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "a" * 32,
        "status": "starting",
        "worker_pid": 123,
    }

    def fail(_root: Path) -> dict[str, object]:
        raise background.RasterTextBackgroundActivationError("raster_text_background_busy")

    monkeypatch.setattr(cli, "background_activation_status", fail)
    assert cli.main(["status", "--document-preview-root", str(root)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "code": "raster_text_background_busy",
        "status": "failed",
    }
