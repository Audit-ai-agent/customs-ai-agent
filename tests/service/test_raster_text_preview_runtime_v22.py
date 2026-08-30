from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import ValidationError
import pytest

from customs_ai_agent.service import raster_text_preview_runtime as runtime


_PROFILE_REVISION = f"synthetic-raster-text-v1:{'a' * 24}"
_ENGINE: Literal[
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
] = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_METRIC: Literal["source-attested-semantic-text-v3"] = "source-attested-semantic-text-v3"
_EVIDENCE: Literal["source-attested-public-finetuned-staged-train-holdout-v2"] = (
    "source-attested-public-finetuned-staged-train-holdout-v2"
)
_QUALITY: Literal["preview-draft-critical97-cer2-v1"] = "preview-draft-critical97-cer2-v1"
_GENERATION: Literal["current-runner-native-full-gate-v1"] = "current-runner-native-full-gate-v1"


def _scores(value: float = 0.97) -> list[runtime.RasterTextSealedProfileScoreV8]:
    return [
        runtime.RasterTextSealedProfileScoreV8(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.02,
            critical_value_exact_match=value,
        )
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]


def _evaluation_v9() -> runtime.RasterTextSealedEvaluationV9:
    return runtime.RasterTextSealedEvaluationV9(
        metric_revision=_METRIC,
        evidence_revision=_EVIDENCE,
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="sealed-stage2-100-family-v1",
        stage1_decision="continue",
        family_count=100,
        asset_count=2_400,
        raster_text_profile_revision=_PROFILE_REVISION,
        runtime_configuration_sha256="1" * 64,
        extractor_revision="v19-linux-extractor",
        selection_receipt_path="selection/v2.json",
        selection_receipt_sha256="2" * 64,
        projection_receipt_path="projection/sealed-stage2.json",
        projection_receipt_sha256="3" * 64,
        stage1_decision_receipt_path="decisions/stage1-v2.json",
        stage1_decision_receipt_sha256="4" * 64,
        source_gold_receipt_path="gold/fresh-sealed.json",
        source_gold_receipt_sha256="5" * 64,
        calibration_receipt_path="evaluations/calibration-v7.json",
        calibration_receipt_sha256="6" * 64,
        candidate_execution_path="executions/linux-sealed.json",
        candidate_execution_sha256="7" * 64,
        warm_service_smoke_receipt_path="smoke/linux-current.json",
        warm_service_smoke_receipt_sha256="8" * 64,
        source_recognizer_release_sha256="9" * 64,
        source_recognizer_receipt_sha256="a" * 64,
        source_recognizer_training_intent_sha256="b" * 64,
        source_recognizer_parent_selection_receipt_sha256="c" * 64,
        latency_p95_seconds=7.8,
        service_warm_latency_p95_seconds=7.9,
        minimum_page_confidence=0.0,
        quality_gate_revision=_QUALITY,
        evidence_generation_revision=_GENERATION,
        current_runner_implementation_sha256="d" * 64,
        scores=_scores(),
    )


def _pointer_v12(
    evaluation: runtime.RasterTextSealedEvaluationV9,
) -> runtime.RasterTextActivationPointerV12:
    return runtime.RasterTextActivationPointerV12(
        engine_kind=_ENGINE,
        metric_revision=_METRIC,
        evidence_revision=_EVIDENCE,
        latency_gate_revision="warm-p95-10s-v1",
        quality_gate_revision=_QUALITY,
        evidence_generation_revision=_GENERATION,
        current_runner_implementation_sha256=(evaluation.current_runner_implementation_sha256),
        runtime_validation_revision="activation-full-runtime-compact-v1",
        raster_text_profile_revision=_PROFILE_REVISION,
        runtime_configuration_path="profiles/v19-linux.json",
        runtime_configuration_sha256=evaluation.runtime_configuration_sha256,
        sealed_evaluation_path="evaluations/sealed-v9.json",
        sealed_evaluation_sha256="e" * 64,
        source_recognizer_release_sha256=evaluation.source_recognizer_release_sha256,
        source_recognizer_receipt_sha256=evaluation.source_recognizer_receipt_sha256,
        source_recognizer_training_intent_sha256=(
            evaluation.source_recognizer_training_intent_sha256
        ),
        source_recognizer_selection_receipt_path=evaluation.selection_receipt_path,
        source_recognizer_selection_receipt_sha256=evaluation.selection_receipt_sha256,
        source_recognizer_parent_selection_receipt_sha256=(
            evaluation.source_recognizer_parent_selection_receipt_sha256
        ),
    )


def _profile(engine_kind: str = _ENGINE) -> runtime.RasterTextOfflineProfileV19:
    return runtime.RasterTextOfflineProfileV19.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19-linux-extractor",
        engine_kind=engine_kind,
        source_recognizer_release_sha256="9" * 64,
        source_recognizer_receipt_sha256="a" * 64,
        source_recognizer_training_intent_sha256="b" * 64,
        source_recognizer_selection_receipt_relative_path="selection/v2.json",
        source_recognizer_selection_receipt_sha256="2" * 64,
        source_recognizer_parent_selection_receipt_sha256="c" * 64,
    )


def test_v9_v12_parser_and_generation_binding_are_exact() -> None:
    evaluation = _evaluation_v9()
    pointer = _pointer_v12(evaluation)
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))

    assert type(runtime.parse_raster_text_activation_pointer(raw)) is (
        runtime.RasterTextActivationPointerV12
    )
    assert (
        type(
            runtime._parse_raster_text_sealed_evaluation(  # noqa: SLF001
                runtime._canonical_evaluation_json_bytes(
                    evaluation.model_dump(mode="json", warnings=False)
                )
            )
        )
        is runtime.RasterTextSealedEvaluationV9
    )
    calibration = runtime.RasterTextCalibrationReceiptV7.model_construct(
        metric_revision=_METRIC,
        evidence_revision=_EVIDENCE,
        latency_gate_revision="warm-p95-10s-v1",
    )
    assert (
        runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
        )
        == _METRIC
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=pointer.model_copy(update={"current_runner_implementation_sha256": "f" * 64}),
            evaluation=evaluation,
            calibration=calibration,
        )


def test_v9_keeps_cer_two_percent_and_accepts_critical_exact_97_only() -> None:
    assert min(score.critical_value_exact_match for score in _evaluation_v9().scores) == 0.97
    with pytest.raises(ValidationError):
        runtime.RasterTextSealedProfileScoreV8(
            document_type="bill_of_lading",
            profile_id="png-216",
            character_error_rate=0.02,
            critical_value_exact_match=0.9699,
        )


def test_v12_accepts_only_the_exact_one_pass_v19_engine() -> None:
    pointer = _pointer_v12(_evaluation_v9())
    runtime._verify_activation_pointer_engine_binding(pointer=pointer, profile=_profile())  # noqa: SLF001
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_profile_mismatch"):
        runtime._verify_activation_pointer_engine_binding(  # noqa: SLF001
            pointer=pointer,
            profile=_profile(f"{_ENGINE[:-7]}voyage_retry_onednn"),
        )


def test_v12_linux_platform_accepts_a_pinned_manifest_larger_than_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_raw = runtime._canonical_evaluation_json_bytes(  # noqa: SLF001
        {
            "bundle_id": "raster-text-direct-paddle-bundle-v1",
            "padding": "x" * (runtime._MAX_CONFIGURATION_BYTES + 1),  # noqa: SLF001
            "runtime_architecture": "amd64",
            "runtime_operating_system": "linux",
            "runtime_python_abi": "cp311",
            "schema_version": "2.0",
        }
    )
    assert runtime._MAX_CONFIGURATION_BYTES < len(manifest_raw)  # noqa: SLF001
    assert len(manifest_raw) <= runtime._MAX_DIRECT_BUNDLE_MANIFEST_BYTES  # noqa: SLF001
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_raw)
    profile = _profile().model_copy(
        update={
            "bundle_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "bundle_root_relative_path": "bundles/linux-v1",
        }
    )

    monkeypatch.setattr(runtime, "_v2_bundle_sidecar_path", lambda **_kwargs: manifest_path)
    monkeypatch.setattr(
        runtime,
        "current_raster_runtime_platform_binding",
        lambda: ("linux", "amd64", "cp311"),
    )

    runtime._verify_v12_linux_profile_runtime_platform(  # noqa: SLF001
        root=tmp_path,
        profile=profile,
    )


def test_direct_platform_accepts_a_schema2_manifest_larger_than_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_raw = runtime._canonical_evaluation_json_bytes(  # noqa: SLF001
        {
            "bundle_id": "raster-text-direct-paddle-bundle-v1",
            "padding": "x" * (runtime._MAX_CONFIGURATION_BYTES + 1),  # noqa: SLF001
            "runtime_architecture": "amd64",
            "runtime_operating_system": "linux",
            "runtime_python_abi": "cp311",
            "schema_version": "2.0",
            "sidecar_interpreter_relative_path": "sidecar/interpreter/bin/python3.11",
        }
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    root = tmp_path / "_local" / "data" / "document-preview"
    bundle_root = f"raster-text-direct-bundles-v1/releases/{manifest_sha256}"
    manifest_path = root / bundle_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(manifest_raw)
    profile = runtime.RasterTextOfflineProfileV2.model_construct(
        bundle_manifest_sha256=manifest_sha256,
        bundle_root_relative_path=bundle_root,
        sidecar_interpreter_path=f"{bundle_root}/sidecar/interpreter/bin/python3.11",
    )
    monkeypatch.setattr(
        runtime,
        "current_raster_runtime_platform_binding",
        lambda: ("linux", "amd64", "cp311"),
    )

    runtime._verify_direct_profile_runtime_platform(  # noqa: SLF001
        root=root,
        profile=profile,
    )


@pytest.mark.parametrize(
    ("verification_mode", "expected_complete"),
    [("runtime", False), ("activation", True)],
)
def test_v12_uses_compact_materials_only_during_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verification_mode: Literal["runtime", "activation"],
    expected_complete: bool,
) -> None:
    pointer = _pointer_v12(_evaluation_v9())
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration = root / "profiles" / "v19-linux.json"
    configuration.parent.mkdir(parents=True)
    configuration.write_bytes(b"profile")
    observed: list[bool] = []

    def stop_after_profile_resolution(**kwargs: object) -> object:
        observed.append(bool(kwargs["verify_complete_materials"]))
        raise runtime.RasterTextRuntimeError("test_stop")

    monkeypatch.setattr(runtime, "_load_profile_materials", stop_after_profile_resolution)
    with pytest.raises(runtime.RasterTextRuntimeError, match="test_stop"):
        runtime._load_binding_from_pointer(  # noqa: SLF001
            root=root,
            approval_path=root / "approvals" / "raster-text-active.json",
            approval_raw=raw,
            approval_sha256=hashlib.sha256(raw).hexdigest(),
            start_service_session=False,
            verification_mode=verification_mode,
        )
    assert observed == [expected_complete]


def test_v12_compact_evidence_uses_one_current_warm_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _evaluation_v9()
    pointer = _pointer_v12(evaluation)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime,
        "_current_raster_text_runner_implementation_sha256",
        lambda: evaluation.current_runner_implementation_sha256,
    )

    def verify_smoke(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(warm_latency_p95_seconds=7.9)

    monkeypatch.setattr(runtime, "_verify_runtime_attested_warm_service_smoke", verify_smoke)
    runtime._verify_runtime_attested_v12_evidence(  # noqa: SLF001
        root=Path("unused"),
        pointer=pointer,
        evaluation=evaluation,
        profile=_profile(),
        evidence_by_path={evaluation.warm_service_smoke_receipt_path: b"warm"},
    )
    assert len(calls) == 1
    assert calls[0]["expected_runner_implementation_sha256"] == (
        evaluation.current_runner_implementation_sha256
    )
    assert calls[0]["latency_p95_limit_seconds"] == 10.0

    monkeypatch.setattr(
        runtime,
        "_verify_runtime_attested_warm_service_smoke",
        lambda **_kwargs: SimpleNamespace(warm_latency_p95_seconds=8.0),
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._verify_runtime_attested_v12_evidence(  # noqa: SLF001
            root=Path("unused"),
            pointer=pointer,
            evaluation=evaluation,
            profile=_profile(),
            evidence_by_path={evaluation.warm_service_smoke_receipt_path: b"warm"},
        )


def test_pointer_writer_emits_v12_only_after_full_activation_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_path = root / "profiles" / "v19-linux.json"
    evaluation_path = root / "evaluations" / "sealed-v9.json"
    configuration_path.parent.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    configuration_raw = b"configuration"
    evaluation_raw = b"evaluation"
    configuration_path.write_bytes(configuration_raw)
    evaluation_path.write_bytes(evaluation_raw)
    evaluation = _evaluation_v9().model_copy(
        update={"runtime_configuration_sha256": hashlib.sha256(configuration_raw).hexdigest()}
    )
    profile = _profile()
    approval_path = root / "approvals" / "raster-text-active.json"
    approval_path.parent.mkdir(parents=True)
    approval_path.write_bytes(b"old-pointer\n")

    monkeypatch.setattr(runtime, "parse_raster_text_offline_profile", lambda _raw: profile)
    monkeypatch.setattr(runtime, "_parse_raster_text_sealed_evaluation", lambda _raw: evaluation)
    monkeypatch.setattr(
        runtime,
        "_verify_v12_linux_profile_runtime_platform",
        lambda **_kwargs: None,
    )
    observed_pointer_raw: list[bytes] = []

    def accept_pointer(**kwargs: object) -> None:
        raw = kwargs["approval_raw"]
        assert isinstance(raw, bytes)
        pointer = runtime.parse_raster_text_activation_pointer(raw)
        assert type(pointer) is runtime.RasterTextActivationPointerV12
        assert kwargs["verification_mode"] == "activation"
        assert kwargs["start_service_session"] is False
        observed_pointer_raw.append(raw)

    monkeypatch.setattr(runtime, "_load_binding_from_pointer", accept_pointer)
    digest, pointer = runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path="profiles/v19-linux.json",
        sealed_evaluation_relative_path="evaluations/sealed-v9.json",
    )

    assert type(pointer) is runtime.RasterTextActivationPointerV12
    assert pointer.engine_kind == _ENGINE
    assert pointer.evidence_generation_revision == _GENERATION
    assert approval_path.read_bytes() == observed_pointer_raw[0]
    assert hashlib.sha256(approval_path.read_bytes()).hexdigest() == digest

    approval_path.write_bytes(b"still-active\n")
    monkeypatch.setattr(
        runtime,
        "_load_binding_from_pointer",
        lambda **_kwargs: (_ for _ in ()).throw(
            runtime.RasterTextRuntimeError("raster_text_evaluation_invalid")
        ),
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="profiles/v19-linux.json",
            sealed_evaluation_relative_path="evaluations/sealed-v9.json",
        )
    assert approval_path.read_bytes() == b"still-active\n"
