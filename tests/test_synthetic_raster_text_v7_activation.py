"""Focused V19 auto-activation tests; no native training or OCR runs."""

from __future__ import annotations

from contextlib import contextmanager
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import cast, Iterator

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_finetuning import (
    PublicFineTuneReceiptV1,
    PublicFineTuneReceiptV2,
    VerifiedPublicFineTunedRecognizerReleaseV1,
)
from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    RasterTextFreshScopeSelectionReceiptV2,
    SyntheticRasterTextActivationError,
    SyntheticRasterTextAutoActivationResultV1,
)
from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextActivationPointerV8,
    RasterTextOfflineProfileV19,
)


_SELECTION_SHA = "1" * 64
_PARENT_SELECTION_SHA = "2" * 64
_RECEIPT_SHA = "3" * 64
_INTENT_SHA = "4" * 64
_RELEASE_SHA = "5" * 64
_SELECTION_PATH = f"synthetic-raster-text-v1/fresh-scope-selections/{_SELECTION_SHA}.json"


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root.resolve()


def _receipt_v2() -> PublicFineTuneReceiptV2:
    return PublicFineTuneReceiptV2.model_construct(
        schema_version="2.0",
        trainer_revision="public-paddle-jit-ctc-leakage-safe-head-v2",
        model_revision="en-ppocrv5-mobile-rec-source-attested-head-v2",
        stage="leakage-safe-v2",
        tuning_scope="head",
        maximum_epochs=4,
        overlap_verifier_revision="selection-v2-all-holdouts-zero-overlap-v1",
        train_holdout_overlap_count=0,
        receipt_sha256=_RECEIPT_SHA,
        training_intent_sha256=_INTENT_SHA,
        selection_receipt_relative_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA,
        parent_selection_receipt_sha256=_PARENT_SELECTION_SHA,
        train_crop_manifest=SimpleNamespace(
            family_count=500,
            usable_crop_count=1_000,
        ),
        development_crop_manifest=SimpleNamespace(
            family_count=100,
            usable_crop_count=200,
        ),
    )


def _release(receipt: object) -> VerifiedPublicFineTunedRecognizerReleaseV1:
    return cast(
        VerifiedPublicFineTunedRecognizerReleaseV1,
        SimpleNamespace(
            release_sha256=_RELEASE_SHA,
            receipt=receipt,
            manifest=SimpleNamespace(receipt_sha256=_RECEIPT_SHA),
        ),
    )


def _selection_context() -> SimpleNamespace:
    return SimpleNamespace(
        selection=RasterTextFreshScopeSelectionReceiptV2.model_construct(
            parent_selection_receipt_sha256=_PARENT_SELECTION_SHA,
        ),
        selection_relative_path=_SELECTION_PATH,
        selection_sha256=_SELECTION_SHA,
    )


def test_v19_training_contract_accepts_only_exact_v2_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    release = _release(_receipt_v2())
    monkeypatch.setattr(
        activation,
        "load_verified_public_finetuned_recognizer_release_v1",
        lambda *_args: release,
    )
    monkeypatch.setattr(
        activation,
        "_fresh_scope_context",
        lambda **_kwargs: _selection_context(),
    )

    observed = activation._verify_v19_public_finetune_contract(  # noqa: SLF001
        root=root,
        recognizer_release=release,
    )

    assert observed is release


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("trainer_revision", "public-paddle-jit-ctc-domain-finetune-v1"),
        ("stage", "overfit"),
        ("maximum_epochs", 8),
        ("train_holdout_overlap_count", 1),
        ("selection_receipt_sha256", "9" * 64),
    ],
)
def test_v19_training_contract_rejects_drift_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
) -> None:
    root = _preview_root(tmp_path)
    receipt = _receipt_v2().model_copy(update={field: invalid_value})
    release = _release(receipt)
    monkeypatch.setattr(
        activation,
        "load_verified_public_finetuned_recognizer_release_v1",
        lambda *_args: release,
    )
    monkeypatch.setattr(
        activation,
        "_fresh_scope_context",
        lambda **_kwargs: _selection_context(),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation._verify_v19_public_finetune_contract(  # noqa: SLF001
            root=root,
            recognizer_release=release,
        )

    assert captured.value.code == "public_finetune_v2_contract_invalid"


def test_v19_training_contract_rejects_legacy_overfit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    release = _release(PublicFineTuneReceiptV1.model_construct())
    monkeypatch.setattr(
        activation,
        "load_verified_public_finetuned_recognizer_release_v1",
        lambda *_args: release,
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation._verify_v19_public_finetune_contract(  # noqa: SLF001
            root=root,
            recognizer_release=release,
        )

    assert captured.value.code == "public_finetune_v2_contract_invalid"


@contextmanager
def _no_context(*_args: object, **_kwargs: object) -> Iterator[None]:
    yield


class _FakeFineTuneProcess:
    def __init__(
        self,
        payload: bytes,
        *,
        return_code: int = 0,
        first_wait_times_out: bool = False,
    ) -> None:
        self.stdout = io.BytesIO(payload)
        self.return_code = return_code
        self.first_wait_times_out = first_wait_times_out
        self.alive = True
        self.terminated = False
        self.killed = False

    def wait(self, *, timeout: int) -> int:
        if self.first_wait_times_out:
            self.first_wait_times_out = False
            raise subprocess.TimeoutExpired("fixed-fine-tune", timeout)
        self.alive = False
        return self.return_code

    def poll(self) -> int | None:
        return None if self.alive else self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False


def _fine_tune_success_stdout(relative_path: str) -> bytes:
    return (
        json.dumps(
            {
                "release_relative_path": relative_path,
                "release_sha256": _RELEASE_SHA,
                "stage": "leakage-safe-v2",
                "status": "published",
                "tuning_scope": "head",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def test_fixed_v19_training_child_isolated_from_parent_native_imports_and_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    relative_path = "raster-text-public-finetuned-recognizers-v1/releases/" + _RELEASE_SHA
    release_path = root / Path(*relative_path.split("/"))
    release_path.mkdir(parents=True)
    release = cast(
        VerifiedPublicFineTunedRecognizerReleaseV1,
        SimpleNamespace(release_path=release_path, release_sha256=_RELEASE_SHA),
    )
    process = _FakeFineTuneProcess(_fine_tune_success_stdout(relative_path))
    observed: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "PIL", ModuleType("PIL"))
    monkeypatch.setenv("PYTHONPATH", "untrusted-parent-path")
    monkeypatch.setenv("HTTPS_PROXY", "http://network.invalid")

    def popen(command: list[str], **kwargs: object) -> _FakeFineTuneProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return process

    monkeypatch.setattr(activation.subprocess, "Popen", popen)
    monkeypatch.setattr(
        activation,
        "load_verified_public_finetuned_recognizer_release_v1",
        lambda observed_root, digest: (
            release
            if observed_root == root and digest == _RELEASE_SHA
            else pytest.fail("child release must be reopened by its digest")
        ),
    )

    result = activation._run_fixed_v19_fine_tune_subprocess(root)  # noqa: SLF001

    assert result is release
    command = cast(list[str], observed["command"])
    assert command[0] == str(Path(sys.executable).resolve(strict=True))
    assert command[1:3] == ["-I", "-B"]
    assert command[-4:] == [
        "--document-preview-root",
        str(root),
        "--stage",
        "leakage-safe-v2",
    ]
    kwargs = cast(dict[str, object], observed["kwargs"])
    assert kwargs["shell"] is False
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    environment = cast(dict[str, str], kwargs["env"])
    assert "PYTHONPATH" not in environment
    assert "HTTPS_PROXY" not in environment
    assert environment["OMP_NUM_THREADS"] == "6"
    assert environment["FLAGS_json_format_model"] == "1"


@pytest.mark.parametrize(
    ("payload", "return_code", "expected_code"),
    [
        (b"native output\n{}\n", 0, "public_finetune_subprocess_output_invalid"),
        (
            b'{"code":"public_finetune_execution_failed","status":"failed"}\n',
            2,
            "public_finetune_execution_failed",
        ),
        (
            b'{"status":"published"}\n',
            0,
            "public_finetune_subprocess_output_invalid",
        ),
    ],
)
def test_fixed_v19_training_child_output_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    return_code: int,
    expected_code: str,
) -> None:
    root = _preview_root(tmp_path)
    process = _FakeFineTuneProcess(payload, return_code=return_code)
    monkeypatch.setattr(activation.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        activation,
        "load_verified_public_finetuned_recognizer_release_v1",
        lambda *_args: pytest.fail("invalid child output must not be reopened"),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation._run_fixed_v19_fine_tune_subprocess(root)  # noqa: SLF001

    assert captured.value.code == expected_code


def test_fixed_v19_training_child_timeout_is_bounded_and_terminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    process = _FakeFineTuneProcess(b"", first_wait_times_out=True)
    monkeypatch.setattr(activation.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation._run_fixed_v19_fine_tune_subprocess(root)  # noqa: SLF001

    assert captured.value.code == "public_finetune_subprocess_timeout"
    assert process.terminated
    assert not process.killed
    assert activation._V19_FINE_TUNE_SUBPROCESS_TIMEOUT_SECONDS == 60 * 60  # noqa: SLF001


def test_auto_v7_orders_holdout_before_fixed_training_and_enters_only_v19(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    sidecar = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="6" * 64,
        bundle_root_relative_path="sidecar/release",
    )
    mobile = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="7" * 64,
        bundle_root_relative_path="mobile/release",
    )
    release = _release(_receipt_v2())
    model_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="8" * 64,
        bundle_root_relative_path="public-finetuned/release",
    )
    build_inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-v1/build-inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="test-pdfium"),
    )
    profile_info = SimpleNamespace(
        runtime_configuration_relative_path="profiles/v19.json",
        runtime_configuration_sha256="9" * 64,
    )
    profile = RasterTextOfflineProfileV19.model_construct()
    result = SyntheticRasterTextAutoActivationResultV1(
        raster_text_profile_revision="synthetic-raster-text-v1:" + "a" * 24,
        activation_pointer_sha256="b" * 64,
        sealed_evaluation_relative_path="sealed/v7.json",
        sealed_evaluation_sha256="c" * 64,
    )
    events: list[str] = []
    transaction: dict[str, object] = {}

    monkeypatch.setattr(activation, "_deny_network", _no_context)
    monkeypatch.setattr(activation, "_lock", _no_context)
    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: sidecar,
    )
    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile,
    )
    monkeypatch.setattr(
        activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: build_inputs,
    )
    monkeypatch.setattr(
        activation,
        "_existing_synthetic_raster_dataset",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        activation,
        "compile_synthetic_raster_dataset",
        lambda **_kwargs: events.append("dataset"),
    )

    def write_selection(**_kwargs: object) -> tuple[str, str]:
        events.append("selection")
        return _SELECTION_PATH, _SELECTION_SHA

    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_scope_selection_receipt",
        write_selection,
    )

    # This mirrors the real compiler's parent-process side effect.  V7 must
    # still call the clean child boundary, never the in-process trainer.
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "PIL", ModuleType("PIL"))

    def ensure_training(observed_root: Path) -> object:
        assert observed_root == root
        assert "numpy" in sys.modules
        assert "PIL" in sys.modules
        events.append("fine-tune-child")
        return release

    monkeypatch.setattr(
        activation,
        "_run_fixed_v19_fine_tune_subprocess",
        ensure_training,
    )

    def verify_contract(**kwargs: object) -> object:
        events.append("verify-training")
        return kwargs["recognizer_release"]

    monkeypatch.setattr(
        activation,
        "_verify_v19_public_finetune_contract",
        verify_contract,
    )

    def materialize_model(*args: object) -> object:
        assert args == (mobile, release)
        events.append("combined-model")
        return model_bundle

    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_public_finetuned_paddle_bundle_v1",
        materialize_model,
    )

    def prepare(*args: object, **kwargs: object) -> object:
        assert args == (sidecar, model_bundle)
        assert kwargs["minimum_page_confidence"] == 0.0
        assert kwargs["cpu_threads"] == 6
        events.append("profile-v19")
        return profile_info

    monkeypatch.setattr(
        activation,
        "prepare_raster_text_public_finetuned_runtime_profile_v19",
        prepare,
    )
    monkeypatch.setattr(activation, "_read", lambda *_args, **_kwargs: b"profile")
    monkeypatch.setattr(
        activation,
        "parse_raster_text_offline_profile",
        lambda _raw: profile,
    )

    def run_transaction(**kwargs: object) -> SyntheticRasterTextAutoActivationResultV1:
        transaction.update(kwargs)
        events.append("evaluation")
        return result

    monkeypatch.setattr(
        activation,
        "_run_v2_staged_candidate_activation",
        run_transaction,
    )

    observed = activation.auto_activate_synthetic_raster_text_v7(document_preview_root=root)

    assert observed is result
    assert events.index("selection") < events.index("fine-tune-child")
    assert events.index("fine-tune-child") < events.index("combined-model")
    assert transaction["expected_profile_type"] is RasterTextOfflineProfileV19
    assert transaction["expected_pointer_type"] is RasterTextActivationPointerV8
    assert callable(transaction["pre_sealed_activation_check"])
    assert transaction["dataset_materials_preverified"] is True
    assert transaction["write_calibration"] is (
        activation.write_raster_semantic_text_fresh_calibration_receipt_v7
    )
    assert transaction["write_sealed"] is (
        activation.write_raster_semantic_text_fresh_sealed_evaluation_v7
    )


@pytest.mark.parametrize(
    "failure_code",
    [
        "public_finetune_execution_failed",
        "public_finetune_subprocess_timeout",
        "public_finetune_subprocess_output_invalid",
    ],
)
def test_auto_v7_training_child_failure_never_enters_evaluation_or_changes_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
) -> None:
    root = _preview_root(tmp_path)
    pointer = root / "approvals" / "raster-text-active.json"
    pointer.parent.mkdir()
    previous = b'{"previous":"binding"}\n'
    pointer.write_bytes(previous)
    sidecar = SimpleNamespace(document_preview_root=root)
    build_inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-v1/build-inputs.json"
    )
    monkeypatch.setattr(activation, "_deny_network", _no_context)
    monkeypatch.setattr(activation, "_lock", _no_context)
    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: sidecar,
    )
    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: build_inputs,
    )
    monkeypatch.setattr(
        activation,
        "_existing_synthetic_raster_dataset",
        lambda *_args: True,
    )
    events: list[str] = []
    monkeypatch.setattr(
        activation,
        "compile_synthetic_raster_dataset",
        lambda **_kwargs: events.append("dataset-verified"),
    )
    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: (_SELECTION_PATH, _SELECTION_SHA),
    )
    def fail_fine_tune(*_args: object, **_kwargs: object) -> None:
        events.append("fine-tune-child")
        raise SyntheticRasterTextActivationError(failure_code)

    monkeypatch.setattr(
        activation,
        "_run_fixed_v19_fine_tune_subprocess",
        fail_fine_tune,
    )
    monkeypatch.setattr(
        activation,
        "_run_v2_staged_candidate_activation",
        lambda **_kwargs: pytest.fail("evaluation must not run after training failure"),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation.auto_activate_synthetic_raster_text_v7(document_preview_root=root)

    assert captured.value.code == failure_code
    assert events == ["dataset-verified", "fine-tune-child"]
    assert pointer.read_bytes() == previous
