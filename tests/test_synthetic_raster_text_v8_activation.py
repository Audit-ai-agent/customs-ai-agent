"""Focused V20/V9 auto-activation tests; no native training or OCR runs."""

from __future__ import annotations

from contextlib import contextmanager
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast, Iterator

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_voyage_replay import (
    VerifiedVoyageReplayReleaseV1,
    VoyageReplayReceiptV3,
)
from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    RasterTextFreshScopeSelectionReceiptV2,
    SyntheticRasterTextActivationError,
    SyntheticRasterTextAutoActivationResultV1,
)
from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextActivationPointerV9,
    RasterTextOfflineProfileV19,
    RasterTextOfflineProfileV20,
)


_SELECTION_SHA = "1" * 64
_PARENT_SELECTION_SHA = "2" * 64
_RECEIPT_SHA = "3" * 64
_INTENT_SHA = "4" * 64
_RELEASE_SHA = "5" * 64
_TRAIN_MANIFEST_SHA = "6" * 64
_DEVELOPMENT_MANIFEST_SHA = "7" * 64
_PARENT_RELEASE_SHA = "51f4324be43c1840a574cdb4b9febb58ec175df80de79cd95a0ef9ff33e0bea5"
_PARENT_RECEIPT_SHA = "b09e8a208407af78e6e1b3535a25abacce3a765f952d73dbfbc68193cdaa42a6"
_SELECTION_PATH = f"synthetic-raster-text-v1/fresh-scope-selections/{_SELECTION_SHA}.json"


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root.resolve()


@contextmanager
def _no_context(*_args: object, **_kwargs: object) -> Iterator[None]:
    yield


def _manifest(*, split: str) -> SimpleNamespace:
    train = split == "train"
    return SimpleNamespace(
        split=split,
        purpose="optimizer" if train else "validation-only",
        selected_family_count=500 if train else 100,
        voyage_crop_count=500 if train else 200,
        general_crop_count=250 if train else 200,
        total_crop_count=750 if train else 400,
        holdout_overlap_count=0,
        independent_audit_family_overlap_count=0,
        independent_audit_source_overlap_count=0,
        selection_receipt_relative_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA,
        parent_selection_receipt_sha256=_PARENT_SELECTION_SHA,
        voyage_recipe_set_sha256="8" * 64,
    )


def _receipt_v3(**updates: object) -> VoyageReplayReceiptV3:
    values: dict[str, object] = {
        "schema_version": "3.0",
        "trainer_revision": "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3",
        "model_revision": "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3",
        "output_model_name": ("source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"),
        "stage": "voyage-replay-v3",
        "tuning_scope": "ctc_head_only",
        "training_intent_sha256": _INTENT_SHA,
        "parent_release_sha256": _PARENT_RELEASE_SHA,
        "parent_receipt_sha256": _PARENT_RECEIPT_SHA,
        "maximum_epochs": 2,
        "batch_size": 30,
        "voyage_per_batch": 20,
        "general_per_batch": 10,
        "batches_per_epoch": 25,
        "trainable_tensor_count": 2,
        "trainable_parameter_count": 52_998,
        "static_roundtrip_verified": True,
        "train_manifest_sha256": _TRAIN_MANIFEST_SHA,
        "development_manifest_sha256": _DEVELOPMENT_MANIFEST_SHA,
        "train_corpus_manifest": _manifest(split="train"),
        "development_corpus_manifest": _manifest(split="development"),
        "receipt_sha256": _RECEIPT_SHA,
    }
    values.update(updates)
    return VoyageReplayReceiptV3.model_construct(**cast(Any, values))


def _release(receipt: VoyageReplayReceiptV3) -> VerifiedVoyageReplayReleaseV1:
    return cast(
        VerifiedVoyageReplayReleaseV1,
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


def test_v20_contract_accepts_only_exact_v3_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    release = _release(_receipt_v3())
    monkeypatch.setattr(
        activation,
        "load_verified_voyage_replay_release_v1",
        lambda *_args: release,
    )
    monkeypatch.setattr(
        activation,
        "_fresh_scope_context",
        lambda **_kwargs: _selection_context(),
    )

    observed = activation._verify_v20_voyage_replay_contract(  # noqa: SLF001
        root=root,
        recognizer_release=release,
    )

    assert observed is release


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("trainer_revision", "unbound-trainer"),
        ("stage", "overfit"),
        ("maximum_epochs", 3),
        ("batch_size", 16),
        ("static_roundtrip_verified", False),
    ],
)
def test_v20_contract_rejects_receipt_drift_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
) -> None:
    root = _preview_root(tmp_path)
    release = _release(_receipt_v3(**{field: invalid_value}))
    monkeypatch.setattr(
        activation,
        "load_verified_voyage_replay_release_v1",
        lambda *_args: release,
    )
    monkeypatch.setattr(
        activation,
        "_fresh_scope_context",
        lambda **_kwargs: _selection_context(),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation._verify_v20_voyage_replay_contract(  # noqa: SLF001
            root=root,
            recognizer_release=release,
        )

    assert captured.value.code == "voyage_replay_v3_contract_invalid"


class _FakeReplayProcess:
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
            raise subprocess.TimeoutExpired("fixed-voyage-replay", timeout)
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


def _replay_success_stdout(relative_path: str) -> bytes:
    return (
        json.dumps(
            {
                "release_relative_path": relative_path,
                "release_sha256": _RELEASE_SHA,
                "stage": "voyage-replay-v3",
                "status": "published",
                "tuning_scope": "ctc_head_only",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def test_fixed_v20_training_child_is_fresh_bounded_and_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    relative = "raster-text-public-voyage-replay-recognizers-v1/releases/" + _RELEASE_SHA
    release_path = root / Path(*relative.split("/"))
    release_path.mkdir(parents=True)
    release = cast(
        VerifiedVoyageReplayReleaseV1,
        SimpleNamespace(release_path=release_path, release_sha256=_RELEASE_SHA),
    )
    process = _FakeReplayProcess(_replay_success_stdout(relative))
    observed: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "PIL", ModuleType("PIL"))
    monkeypatch.setenv("PYTHONPATH", "untrusted-parent-path")
    monkeypatch.setenv("HTTPS_PROXY", "http://network.invalid")

    def popen(command: list[str], **kwargs: object) -> _FakeReplayProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return process

    monkeypatch.setattr(activation.subprocess, "Popen", popen)
    monkeypatch.setattr(
        activation,
        "load_verified_voyage_replay_release_v1",
        lambda observed_root, digest: (
            release
            if observed_root == root and digest == _RELEASE_SHA
            else pytest.fail("child release must be reopened by digest")
        ),
    )

    result = activation._run_fixed_voyage_replay_subprocess(root)  # noqa: SLF001

    assert result is release
    command = cast(list[str], observed["command"])
    assert command[0] == str(Path(sys.executable).resolve(strict=True))
    assert command[1:3] == ["-I", "-B"]
    assert command[-4:] == [
        "--document-preview-root",
        str(root),
        "--stage",
        "voyage-replay-v3",
    ]
    kwargs = cast(dict[str, object], observed["kwargs"])
    assert kwargs["shell"] is False
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    environment = cast(dict[str, str], kwargs["env"])
    assert "PYTHONPATH" not in environment
    assert "HTTPS_PROXY" not in environment
    assert environment["OMP_NUM_THREADS"] == "6"
    assert activation._MAX_VOYAGE_REPLAY_CHILD_STDOUT_BYTES == 4 * 1024  # noqa: SLF001
    assert activation._VOYAGE_REPLAY_SUBPROCESS_TIMEOUT_SECONDS == 30 * 60  # noqa: SLF001


def test_fixed_v20_training_child_timeout_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    process = _FakeReplayProcess(b"", first_wait_times_out=True)
    monkeypatch.setattr(activation.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation._run_fixed_voyage_replay_subprocess(root)  # noqa: SLF001

    assert captured.value.code == "voyage_replay_subprocess_timeout"
    assert process.terminated
    assert not process.killed


def test_independent_audit_accepts_only_exact_v19b_or_v20_tuple() -> None:
    old = RasterTextOfflineProfileV19.model_construct(
        schema_version="19.0",
        engine_kind=activation._V19B_ENGINE_KIND,  # noqa: SLF001
    )
    current = RasterTextOfflineProfileV20.model_construct(
        schema_version="20.0",
        engine_kind=activation._V20_ENGINE_KIND,  # noqa: SLF001
    )
    old_crossed = RasterTextOfflineProfileV19.model_construct(
        schema_version="19.0",
        engine_kind=activation._V20_ENGINE_KIND,  # noqa: SLF001
    )
    current_crossed = RasterTextOfflineProfileV20.model_construct(
        schema_version="20.0",
        engine_kind=activation._V19B_ENGINE_KIND,  # noqa: SLF001
    )

    assert activation._is_exact_bl_repair_audit_profile(old)  # noqa: SLF001
    assert activation._is_exact_bl_repair_audit_profile(current)  # noqa: SLF001
    assert not activation._is_exact_bl_repair_audit_profile(old_crossed)  # noqa: SLF001
    assert not activation._is_exact_bl_repair_audit_profile(current_crossed)  # noqa: SLF001


def test_auto_v8_orders_replay_before_v20_staged_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    sidecar = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="9" * 64,
        bundle_root_relative_path="sidecar/release",
    )
    mobile = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="a" * 64,
        bundle_root_relative_path="mobile/release",
    )
    release = _release(_receipt_v3())
    model_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="b" * 64,
        bundle_root_relative_path="voyage/release",
    )
    build_inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-v1/build-inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="test-pdfium"),
    )
    profile_info = SimpleNamespace(
        runtime_configuration_relative_path="profiles/v20.json",
        runtime_configuration_sha256="c" * 64,
    )
    profile = RasterTextOfflineProfileV20.model_construct(
        schema_version="20.0",
        engine_kind=activation._V20_ENGINE_KIND,  # noqa: SLF001
    )
    result = SyntheticRasterTextAutoActivationResultV1(
        raster_text_profile_revision="synthetic-raster-text-v1:" + "d" * 24,
        activation_pointer_sha256="e" * 64,
        sealed_evaluation_relative_path="sealed/v7.json",
        sealed_evaluation_sha256="f" * 64,
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
    monkeypatch.setattr(activation, "_existing_synthetic_raster_dataset", lambda *_: False)
    monkeypatch.setattr(
        activation,
        "compile_synthetic_raster_dataset",
        lambda **_kwargs: events.append("dataset"),
    )

    def selection(**_kwargs: object) -> tuple[str, str]:
        events.append("selection")
        return _SELECTION_PATH, _SELECTION_SHA

    monkeypatch.setattr(activation, "write_fresh_raster_text_scope_selection_receipt", selection)

    def replay(observed_root: Path) -> VerifiedVoyageReplayReleaseV1:
        assert observed_root == root
        events.append("replay-child")
        return release

    monkeypatch.setattr(activation, "_run_fixed_voyage_replay_subprocess", replay)
    monkeypatch.setattr(
        activation,
        "_verify_v20_voyage_replay_contract",
        lambda **kwargs: kwargs["recognizer_release"],
    )

    def combined(*args: object) -> object:
        assert args == (mobile, release)
        events.append("combined-model")
        return model_bundle

    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_public_voyage_replay_paddle_bundle_v1",
        combined,
    )

    def prepare(*args: object, **kwargs: object) -> object:
        assert args == (sidecar, model_bundle)
        assert kwargs["minimum_page_confidence"] == 0.0
        assert kwargs["cpu_threads"] == 6
        events.append("profile-v20")
        return profile_info

    monkeypatch.setattr(
        activation,
        "prepare_raster_text_public_voyage_replay_runtime_profile_v20",
        prepare,
    )
    monkeypatch.setattr(activation, "_read", lambda *_args, **_kwargs: b"profile")
    monkeypatch.setattr(activation, "parse_raster_text_offline_profile", lambda _raw: profile)

    def staged(**kwargs: object) -> SyntheticRasterTextAutoActivationResultV1:
        transaction.update(kwargs)
        events.append("evaluation")
        return result

    monkeypatch.setattr(activation, "_run_v2_staged_candidate_activation", staged)

    observed = activation.auto_activate_synthetic_raster_text_v8(document_preview_root=root)

    assert observed is result
    assert events.index("selection") < events.index("replay-child")
    assert events.index("replay-child") < events.index("combined-model")
    assert transaction["expected_profile_type"] is RasterTextOfflineProfileV20
    assert transaction["expected_pointer_type"] is RasterTextActivationPointerV9
    assert transaction["dataset_materials_preverified"] is True
    assert callable(transaction["pre_sealed_activation_check"])
    assert transaction["write_calibration"] is (
        activation.write_raster_semantic_text_fresh_calibration_receipt_v7
    )
    assert transaction["write_sealed"] is (
        activation.write_raster_semantic_text_fresh_sealed_evaluation_v7
    )


def test_auto_v8_child_failure_preserves_existing_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(activation, "_existing_synthetic_raster_dataset", lambda *_: True)
    monkeypatch.setattr(activation, "compile_synthetic_raster_dataset", lambda **_kwargs: None)
    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: (_SELECTION_PATH, _SELECTION_SHA),
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise SyntheticRasterTextActivationError("voyage_replay_development_gate_failed")

    monkeypatch.setattr(activation, "_run_fixed_voyage_replay_subprocess", fail)
    monkeypatch.setattr(
        activation,
        "_run_v2_staged_candidate_activation",
        lambda **_kwargs: pytest.fail("evaluation must not run after replay failure"),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation.auto_activate_synthetic_raster_text_v8(document_preview_root=root)

    assert captured.value.code == "voyage_replay_development_gate_failed"
    assert pointer.read_bytes() == previous
