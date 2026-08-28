"""Focused preview-97 command tests; no OCR or pointer writer runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    SyntheticRasterTextActivationError,
)
from tools import auto_activate_synthetic_raster_text_v1 as auto_cli
from tools import auto_activate_synthetic_raster_v1 as compatibility_cli


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        raster_text_profile_revision="synthetic-raster-text-v1:" + "b" * 24,
        activation_pointer_sha256="a" * 64,
    )


def test_clean_state_command_enters_only_preview97_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    expected = _result()
    calls: list[Path] = []

    def activate(*, document_preview_root: Path) -> SimpleNamespace:
        calls.append(document_preview_root)
        return expected

    monkeypatch.setattr(auto_cli, "auto_activate_synthetic_raster_text_preview97_v1", activate)

    result = auto_cli._run(root)

    assert result is expected
    assert calls == [root.resolve()]
    for removed_training_entrypoint in (
        "_ensure_training_dataset",
        "_run_bounded_training_process",
        "_ensure_source_attested_recognizer",
        "_training_command",
    ):
        assert not hasattr(auto_cli, removed_training_entrypoint)


def test_activation_failure_is_content_free_and_preserves_existing_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _preview_root(tmp_path)
    pointer = root / "approvals" / "raster-text-active.json"
    pointer.parent.mkdir()
    previous = b'{"previous":"binding"}\n'
    pointer.write_bytes(previous)

    def fail_activation(**_kwargs: object) -> None:
        raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")

    monkeypatch.setattr(
        auto_cli, "auto_activate_synthetic_raster_text_preview97_v1", fail_activation
    )

    exit_code = auto_cli.main(["--document-preview-root", str(root)])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "code": "raster_text_smoke_gate_failed",
    }
    assert pointer.read_bytes() == previous


def test_one_command_accepts_only_the_document_preview_root() -> None:
    for module in (auto_cli, compatibility_cli):
        parser = module._parser()
        option_destinations = {
            action.dest
            for action in parser._actions
            if action.option_strings  # noqa: SLF001
        }
        assert option_destinations == {"help", "document_preview_root"}


def test_invalid_root_stops_before_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = tmp_path / "not-document-preview"
    invalid.mkdir()
    monkeypatch.setattr(
        auto_cli,
        "auto_activate_synthetic_raster_text_preview97_v1",
        lambda **_kwargs: pytest.fail("invalid root must stop before activation"),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        auto_cli._run(invalid)

    assert captured.value.code == "raster_text_activation_root_invalid"


def test_text_command_reports_the_activated_preview97_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _preview_root(tmp_path)
    expected = _result()
    monkeypatch.setattr(
        auto_cli,
        "auto_activate_synthetic_raster_text_preview97_v1",
        lambda **_kwargs: expected,
    )

    assert auto_cli.main(["--document-preview-root", str(root)]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "status": "activated",
        "dataset_id": "synthetic-raster-text-v1",
        "raster_text_profile_revision": expected.raster_text_profile_revision,
        "activation_pointer_sha256": expected.activation_pointer_sha256,
        "runtime_network": "disabled",
    }


def test_compatibility_command_uses_the_same_v15_text_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _preview_root(tmp_path)
    expected = _result()
    calls: list[Path] = []

    def activate(*, document_preview_root: Path) -> SimpleNamespace:
        calls.append(document_preview_root)
        return expected

    monkeypatch.setattr(
        compatibility_cli,
        "auto_activate_synthetic_raster_text_v1",
        activate,
    )

    assert compatibility_cli.main(["--document-preview-root", str(root)]) == 0

    assert calls == [root]
    assert json.loads(capsys.readouterr().out) == {
        "status": "activated",
        "dataset_id": "synthetic-raster-text-v1",
        "raster_text_profile_revision": expected.raster_text_profile_revision,
        "activation_pointer_sha256": expected.activation_pointer_sha256,
        "runtime_network": "disabled",
    }
