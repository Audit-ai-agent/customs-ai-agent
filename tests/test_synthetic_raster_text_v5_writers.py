"""Focused V4/V5/V6/V7 fresh evidence writer generation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    RasterTextFreshEvaluationProjectionReceiptV1,
    RasterTextFreshProjectedCandidateBundleV1,
    RasterTextFreshProjectedCandidateExecutionV1,
    RasterTextFreshScopeSelectionReceiptV2,
    SyntheticRasterTextActivationError,
)
from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextActivationPointerV6,
    RasterTextCalibrationReceiptV5,
    RasterTextCalibrationReceiptV6,
    RasterTextCalibrationReceiptV7,
    RasterTextOfflineProfileV14,
    RasterTextOfflineProfileV15,
    RasterTextOfflineProfileV18,
    RasterTextOfflineProfileV19,
    RasterTextSealedEvaluationV5,
    RasterTextSealedEvaluationV6,
    RasterTextSealedEvaluationV7,
    RasterTextSealedProfileScoreV5,
    RasterTextSealedProfileScoreV6,
    RasterTextSealedProfileScoreV7,
    RasterTextWarmServiceSmokeReceiptV1,
)


_FINAL_REVISION = "synthetic-raster-text-v1:" + ("a" * 24)
_CALIBRATION_REVISION = "synthetic-raster-text-v1:" + ("b" * 24)
_EXTRACTOR_REVISION = "direct-text-test-v15-onednn"
_FINAL_CONFIGURATION_SHA256 = "c" * 64
_CALIBRATION_CONFIGURATION_SHA256 = "d" * 64
_SELECTION_PATH = "synthetic-raster-text-v1/fresh-selections/selection.json"
_SELECTION_SHA256 = "e" * 64
_RECOGNIZER_RELEASE_SHA256 = "3" * 64
_RECOGNIZER_RECEIPT_SHA256 = "4" * 64
_RECOGNIZER_INTENT_SHA256 = "5" * 64
_PARENT_SELECTION_SHA256 = "6" * 64
_PROJECTION_SEED_SHA256 = hashlib.sha256(
    b"synthetic-raster-text-v1:fresh-evaluation-projection:v2"
).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write(root: Path, relative_path: str, raw: bytes) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _v15_profile(*, revision: str, confidence: float) -> RasterTextOfflineProfileV15:
    return RasterTextOfflineProfileV15.model_construct(
        schema_version="15.0",
        engine_kind=(
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
            "semantic_repair_v2_onednn"
        ),
        extractor_revision=_EXTRACTOR_REVISION,
        minimum_page_confidence=confidence,
        profile_fingerprint_sha256="0" * 64,
        raster_text_profile_revision=revision,
    )


def _v18_profile(*, revision: str, confidence: float) -> RasterTextOfflineProfileV18:
    return RasterTextOfflineProfileV18.model_construct(
        schema_version="18.0",
        engine_kind=(
            "direct_paddle_mobile_det_server_rec_harmonic_recall_native960_lowbox_"
            "strip2_semantic_repair_v2_onednn"
        ),
        extractor_revision=_EXTRACTOR_REVISION,
        minimum_page_confidence=confidence,
        profile_fingerprint_sha256="0" * 64,
        raster_text_profile_revision=revision,
    )


def _v19_profile(*, revision: str, confidence: float) -> RasterTextOfflineProfileV19:
    return RasterTextOfflineProfileV19.model_construct(
        schema_version="19.0",
        engine_kind=(
            "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
            "lowbox_strip2_semantic_repair_v2_onednn"
        ),
        extractor_revision=_EXTRACTOR_REVISION,
        minimum_page_confidence=confidence,
        profile_fingerprint_sha256="0" * 64,
        raster_text_profile_revision=revision,
        source_recognizer_release_sha256=_RECOGNIZER_RELEASE_SHA256,
        source_recognizer_receipt_sha256=_RECOGNIZER_RECEIPT_SHA256,
        source_recognizer_training_intent_sha256=_RECOGNIZER_INTENT_SHA256,
        source_recognizer_selection_receipt_relative_path=_SELECTION_PATH,
        source_recognizer_selection_receipt_sha256=_SELECTION_SHA256,
        source_recognizer_parent_selection_receipt_sha256=(_PARENT_SELECTION_SHA256),
    )


def _projected_evidence(
    *,
    root: Path,
    scope: Literal["fresh-calibration", "fresh-sealed"],
    projection_kind: Literal[
        "calibration-10-family-v1",
        "sealed-stage1-50-family-v1",
        "sealed-stage2-100-family-v1",
    ],
    family_count: Literal[10, 50, 100],
    asset_count: Literal[240, 1_200, 2_400],
    per_stratum_asset_count: Literal[20, 100, 200],
    profile_revision: str,
    configuration_sha256: str,
    latency_p95_seconds: float = 9.0,
    parent_decision_path: str | None = None,
    parent_decision_sha256: str | None = None,
) -> tuple[
    SimpleNamespace,
    RasterTextFreshProjectedCandidateExecutionV1,
    RasterTextFreshProjectedCandidateBundleV1,
    list[object],
    str,
    str,
]:
    source_path = f"synthetic-raster-text-v1/test-gold/{scope}.json"
    execution_path = f"synthetic-raster-text-v1/test-executions/{scope}.json"
    projection_path = f"synthetic-raster-text-v1/test-projections/{projection_kind}.json"
    source_raw = f"source:{scope}".encode()
    execution_raw = f"execution:{projection_kind}".encode()
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    projection = RasterTextFreshEvaluationProjectionReceiptV1(
        projection_seed_sha256=_PROJECTION_SEED_SHA256,
        source_scope=scope,
        projection_kind=projection_kind,
        synthetic_raster_receipt_sha256="f" * 64,
        selection_receipt_relative_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA256,
        full_source_gold_receipt_relative_path=source_path,
        full_source_gold_receipt_sha256=source_sha256,
        family_ids=[f"family-{index:03d}" for index in range(family_count)],
        family_count=family_count,
        asset_count=asset_count,
        per_stratum_asset_count=per_stratum_asset_count,
        parent_stage1_decision_relative_path=parent_decision_path,
        parent_stage1_decision_sha256=parent_decision_sha256,
    )
    projection_raw = _canonical(projection.model_dump(mode="json", warnings=False))
    projection_sha256 = hashlib.sha256(projection_raw).hexdigest()
    _write(root, source_path, source_raw)
    _write(root, execution_path, execution_raw)
    _write(root, projection_path, projection_raw)
    source = SimpleNamespace(
        scope=scope,
        selection_receipt_relative_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA256,
    )
    execution = RasterTextFreshProjectedCandidateExecutionV1.model_construct(
        scope=scope,
        selection_receipt_relative_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA256,
        source_gold_receipt_relative_path=source_path,
        source_gold_receipt_sha256=source_sha256,
        projection_receipt_relative_path=projection_path,
        projection_receipt_sha256=projection_sha256,
        candidate_count=asset_count,
        raster_text_profile_revision=profile_revision,
        runtime_configuration_sha256=configuration_sha256,
        extractor_revision=_EXTRACTOR_REVISION,
        latency_p95_seconds=latency_p95_seconds,
    )
    bundle = RasterTextFreshProjectedCandidateBundleV1.model_construct(
        scope=scope,
        selection_receipt_relative_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA256,
        source_gold_receipt_sha256=source_sha256,
        projection_receipt_relative_path=projection_path,
        projection_receipt_sha256=projection_sha256,
        candidates=[None] * asset_count,
        raster_text_profile_revision=profile_revision,
        runtime_configuration_sha256=configuration_sha256,
        extractor_revision=_EXTRACTOR_REVISION,
    )
    return source, execution, bundle, [SimpleNamespace(scope=scope)], source_path, execution_path


def test_v5_calibration_writer_emits_projected_latency_bound_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    final_profile = _v15_profile(revision=_FINAL_REVISION, confidence=0.5)
    calibration_profile = _v15_profile(
        revision=_CALIBRATION_REVISION,
        confidence=0.0,
    )
    source, execution, bundle, cases, source_path, execution_path = _projected_evidence(
        root=root,
        scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
        profile_revision=_CALIBRATION_REVISION,
        configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: (source, execution, bundle, cases),
    )
    monkeypatch.setattr(
        activation,
        "_resolved_v15_calibration_profile",
        lambda **_kwargs: (calibration_profile, _CALIBRATION_CONFIGURATION_SHA256),
    )
    monkeypatch.setattr(activation, "_calibrated_floor", lambda *_args, **_kwargs: 0.5)

    relative_path, digest = activation.write_raster_semantic_text_fresh_calibration_receipt_v5(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_path,
        candidate_execution_relative_path=execution_path,
        calibration_runtime_configuration_relative_path="profiles/calibration-v15.json",
        activated_profile=final_profile,
        activated_runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
    )
    raw = (root / relative_path).read_bytes()
    receipt = RasterTextCalibrationReceiptV5.model_validate_json(raw)
    assert hashlib.sha256(raw).hexdigest() == digest
    assert receipt.projection_kind == "calibration-10-family-v1"
    assert receipt.asset_count == 240
    assert receipt.latency_p95_seconds == 9.0
    assert receipt.latency_gate_revision == "warm-p95-10s-v1"

    slow_execution = execution.model_copy(update={"latency_p95_seconds": 10.01})
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: (source, slow_execution, bundle, cases),
    )
    with pytest.raises(SyntheticRasterTextActivationError, match="calibration_invalid"):
        activation.write_raster_semantic_text_fresh_calibration_receipt_v5(
            document_preview_root=root,
            source_gold_receipt_relative_path=source_path,
            candidate_execution_relative_path=execution_path,
            calibration_runtime_configuration_relative_path="profiles/calibration-v15.json",
            activated_profile=final_profile,
            activated_runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        )


def test_v6_calibration_writer_binds_v18_and_selection_v2_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    final_profile = _v18_profile(revision=_FINAL_REVISION, confidence=0.5)
    calibration_profile = _v18_profile(
        revision=_CALIBRATION_REVISION,
        confidence=0.0,
    )
    source, execution, bundle, cases, source_path, execution_path = _projected_evidence(
        root=root,
        scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
        profile_revision=_CALIBRATION_REVISION,
        configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence_v6",
        lambda **_kwargs: (source, execution, bundle, cases),
    )
    monkeypatch.setattr(
        activation,
        "_resolved_v18_calibration_profile",
        lambda **_kwargs: (calibration_profile, _CALIBRATION_CONFIGURATION_SHA256),
    )
    monkeypatch.setattr(activation, "_calibrated_floor", lambda *_args, **_kwargs: 0.5)

    relative_path, digest = activation.write_raster_semantic_text_fresh_calibration_receipt_v6(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_path,
        candidate_execution_relative_path=execution_path,
        calibration_runtime_configuration_relative_path="profiles/calibration-v18.json",
        activated_profile=final_profile,
        activated_runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
    )
    raw = (root / relative_path).read_bytes()
    receipt = RasterTextCalibrationReceiptV6.model_validate_json(raw)
    assert hashlib.sha256(raw).hexdigest() == digest
    assert receipt.schema_version == "6.0"
    assert receipt.evidence_revision == ("source-attested-fresh-staged-train-holdout-v2")
    assert receipt.projection_kind == "calibration-10-family-v1"
    assert receipt.asset_count == 240
    assert receipt.hard_timeout_seconds == 30
    assert receipt.latency_p95_seconds == 9.0

    with pytest.raises(SyntheticRasterTextActivationError, match="calibration_invalid"):
        activation.write_raster_semantic_text_fresh_calibration_receipt_v6(
            document_preview_root=root,
            source_gold_receipt_relative_path=source_path,
            candidate_execution_relative_path=execution_path,
            calibration_runtime_configuration_relative_path="profiles/calibration-v18.json",
            activated_profile=_v15_profile(
                revision=_FINAL_REVISION,
                confidence=0.5,
            ),  # type: ignore[arg-type]
            activated_runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        )


def test_v7_calibration_writer_binds_v19_training_and_selection_v2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    final_profile = _v19_profile(revision=_FINAL_REVISION, confidence=0.5)
    calibration_profile = _v19_profile(
        revision=_CALIBRATION_REVISION,
        confidence=0.0,
    )
    source, execution, bundle, cases, source_path, execution_path = _projected_evidence(
        root=root,
        scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
        profile_revision=_CALIBRATION_REVISION,
        configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
    )
    selection = RasterTextFreshScopeSelectionReceiptV2.model_construct(
        parent_selection_receipt_sha256=_PARENT_SELECTION_SHA256,
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence_v6",
        lambda **_kwargs: (source, execution, bundle, cases),
    )
    monkeypatch.setattr(
        activation,
        "_resolved_v19_calibration_profile",
        lambda **_kwargs: (
            calibration_profile,
            _CALIBRATION_CONFIGURATION_SHA256,
        ),
    )
    monkeypatch.setattr(
        activation,
        "_fresh_scope_context",
        lambda **_kwargs: SimpleNamespace(selection=selection),
    )
    monkeypatch.setattr(
        activation,
        "_calibrated_floor",
        lambda *_args, **_kwargs: 0.5,
    )

    relative_path, digest = activation.write_raster_semantic_text_fresh_calibration_receipt_v7(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_path,
        candidate_execution_relative_path=execution_path,
        calibration_runtime_configuration_relative_path=("profiles/calibration-v19.json"),
        activated_profile=final_profile,
        activated_runtime_configuration_sha256=(_FINAL_CONFIGURATION_SHA256),
    )

    raw = (root / relative_path).read_bytes()
    receipt = RasterTextCalibrationReceiptV7.model_validate_json(raw)
    assert hashlib.sha256(raw).hexdigest() == digest
    assert receipt.schema_version == "7.0"
    assert receipt.evidence_revision == ("source-attested-public-finetuned-staged-train-holdout-v2")
    assert receipt.source_recognizer_release_sha256 == (_RECOGNIZER_RELEASE_SHA256)
    assert receipt.source_recognizer_receipt_sha256 == (_RECOGNIZER_RECEIPT_SHA256)
    assert receipt.source_recognizer_training_intent_sha256 == (_RECOGNIZER_INTENT_SHA256)
    assert receipt.source_recognizer_parent_selection_receipt_sha256 == (_PARENT_SELECTION_SHA256)

    drifted = final_profile.model_copy(
        update={"source_recognizer_selection_receipt_sha256": "9" * 64}
    )
    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        activation.write_raster_semantic_text_fresh_calibration_receipt_v7(
            document_preview_root=root,
            source_gold_receipt_relative_path=source_path,
            candidate_execution_relative_path=execution_path,
            calibration_runtime_configuration_relative_path=("profiles/calibration-v19.json"),
            activated_profile=drifted,
            activated_runtime_configuration_sha256=(_FINAL_CONFIGURATION_SHA256),
        )
    assert captured.value.code == "raster_text_v7_fresh_calibration_invalid"


@pytest.mark.parametrize(
    ("projection_kind", "decision", "family_count", "asset_count", "per_stratum"),
    [
        ("sealed-stage1-50-family-v1", "early_pass", 50, 1_200, 100),
        ("sealed-stage2-100-family-v1", "continue", 100, 2_400, 200),
    ],
)
def test_v5_sealed_writer_accepts_only_bound_stage_decisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"],
    decision: Literal["early_pass", "continue"],
    family_count: Literal[50, 100],
    asset_count: Literal[1_200, 2_400],
    per_stratum: Literal[100, 200],
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    final_profile = _v15_profile(revision=_FINAL_REVISION, confidence=0.5)
    calibration_profile = _v15_profile(
        revision=_CALIBRATION_REVISION,
        confidence=0.0,
    )
    decision_path = "synthetic-raster-text-v1/test-decisions/stage1.json"
    decision_raw = b"stage1-decision"
    decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
    _write(root, decision_path, decision_raw)
    sealed = _projected_evidence(
        root=root,
        scope="fresh-sealed",
        projection_kind=projection_kind,
        family_count=family_count,
        asset_count=asset_count,
        per_stratum_asset_count=per_stratum,
        profile_revision=_FINAL_REVISION,
        configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        parent_decision_path=(decision_path if decision == "continue" else None),
        parent_decision_sha256=(decision_sha256 if decision == "continue" else None),
    )
    calibration_evidence = _projected_evidence(
        root=root,
        scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
        profile_revision=_CALIBRATION_REVISION,
        configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
    )
    (
        sealed_source,
        sealed_execution,
        sealed_bundle,
        sealed_cases,
        sealed_source_path,
        sealed_execution_path,
    ) = sealed
    (
        calibration_source,
        calibration_execution,
        calibration_bundle,
        calibration_cases,
        calibration_source_path,
        calibration_execution_path,
    ) = calibration_evidence
    calibration_receipt = RasterTextCalibrationReceiptV5(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-staged-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        calibration_raster_text_profile_revision=_CALIBRATION_REVISION,
        calibration_runtime_configuration_path="profiles/calibration-v15.json",
        calibration_runtime_configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
        extractor_revision=_EXTRACTOR_REVISION,
        selection_receipt_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA256,
        projection_receipt_path=calibration_execution.projection_receipt_relative_path,
        projection_receipt_sha256=calibration_execution.projection_receipt_sha256,
        source_gold_receipt_path=calibration_source_path,
        source_gold_receipt_sha256=calibration_execution.source_gold_receipt_sha256,
        candidate_execution_path=calibration_execution_path,
        candidate_execution_sha256=hashlib.sha256(
            (root / calibration_execution_path).read_bytes()
        ).hexdigest(),
        minimum_page_confidence=0.5,
        latency_p95_seconds=9.0,
    )
    calibration_receipt_path = "synthetic-raster-text-v1/test-calibration-receipts/v5.json"
    _write(
        root,
        calibration_receipt_path,
        _canonical(calibration_receipt.model_dump(mode="json", warnings=False)),
    )

    def verified_evidence(**kwargs: object) -> tuple[object, object, object, list[object]]:
        if kwargs["source_gold_receipt_relative_path"] == sealed_source_path:
            return sealed_source, sealed_execution, sealed_bundle, sealed_cases
        return (
            calibration_source,
            calibration_execution,
            calibration_bundle,
            calibration_cases,
        )

    stage_decision = SimpleNamespace(
        decision=decision,
        source_gold_receipt_relative_path=sealed_source_path,
        source_gold_receipt_sha256=sealed_execution.source_gold_receipt_sha256,
        candidate_execution_relative_path=(
            sealed_execution_path if decision == "early_pass" else "stage1-execution.json"
        ),
        candidate_execution_sha256=(
            hashlib.sha256((root / sealed_execution_path).read_bytes()).hexdigest()
            if decision == "early_pass"
            else "1" * 64
        ),
        projection_receipt_relative_path=(
            sealed_execution.projection_receipt_relative_path
            if decision == "early_pass"
            else "stage1-projection.json"
        ),
        projection_receipt_sha256=(
            sealed_execution.projection_receipt_sha256 if decision == "early_pass" else "2" * 64
        ),
    )
    scores = [
        RasterTextSealedProfileScoreV5(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=0.99,
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]
    warm = RasterTextWarmServiceSmokeReceiptV1.model_construct(warm_latency_p95_seconds=9.0)
    warm_raw = _canonical(warm.model_dump(mode="json", warnings=False))
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence",
        verified_evidence,
    )
    monkeypatch.setattr(
        activation,
        "verify_fresh_raster_text_sealed_stage1_decision",
        lambda **_kwargs: stage_decision,
    )
    monkeypatch.setattr(
        activation,
        "_resolved_v15_calibration_profile",
        lambda **_kwargs: (calibration_profile, _CALIBRATION_CONFIGURATION_SHA256),
    )
    monkeypatch.setattr(
        activation,
        "_load_verified_final_warm_smoke",
        lambda **_kwargs: (warm, warm_raw),
    )
    monkeypatch.setattr(activation, "_calibrated_floor", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(
        activation,
        "evaluate_synthetic_raster_semantic_text_gates_v3",
        lambda _cases: scores,
    )
    monkeypatch.setattr(
        activation,
        "synthetic_raster_semantic_text_gate_passes_v3",
        lambda _scores: True,
    )
    relative_path, digest = activation.write_raster_semantic_text_fresh_sealed_evaluation_v5(
        document_preview_root=root,
        source_gold_receipt_relative_path=sealed_source_path,
        candidate_execution_relative_path=sealed_execution_path,
        calibration_receipt_relative_path=calibration_receipt_path,
        stage1_decision_receipt_relative_path=decision_path,
        warm_service_smoke_receipt_relative_path="warm/final.json",
        profile=final_profile,
        runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
    )
    raw = (root / relative_path).read_bytes()
    receipt = RasterTextSealedEvaluationV5.model_validate_json(raw)
    assert hashlib.sha256(raw).hexdigest() == digest
    assert receipt.projection_kind == projection_kind
    assert receipt.stage1_decision == decision
    assert receipt.asset_count == asset_count

    wrong_decision = SimpleNamespace(**vars(stage_decision))
    wrong_decision.decision = "continue" if decision == "early_pass" else "early_pass"
    monkeypatch.setattr(
        activation,
        "verify_fresh_raster_text_sealed_stage1_decision",
        lambda **_kwargs: wrong_decision,
    )
    with pytest.raises(SyntheticRasterTextActivationError, match="evaluation_invalid"):
        activation.write_raster_semantic_text_fresh_sealed_evaluation_v5(
            document_preview_root=root,
            source_gold_receipt_relative_path=sealed_source_path,
            candidate_execution_relative_path=sealed_execution_path,
            calibration_receipt_relative_path=calibration_receipt_path,
            stage1_decision_receipt_relative_path=decision_path,
            warm_service_smoke_receipt_relative_path="warm/final.json",
            profile=final_profile,
            runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        )


@pytest.mark.parametrize(
    ("projection_kind", "decision", "family_count", "asset_count", "per_stratum"),
    [
        ("sealed-stage1-50-family-v1", "early_pass", 50, 1_200, 100),
        ("sealed-stage2-100-family-v1", "continue", 100, 2_400, 200),
    ],
)
def test_v6_sealed_writer_emits_selection_v2_stage_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"],
    decision: Literal["early_pass", "continue"],
    family_count: Literal[50, 100],
    asset_count: Literal[1_200, 2_400],
    per_stratum: Literal[100, 200],
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    final_profile = _v18_profile(revision=_FINAL_REVISION, confidence=0.5)
    calibration_profile = _v18_profile(
        revision=_CALIBRATION_REVISION,
        confidence=0.0,
    )
    decision_path = "synthetic-raster-text-v1/test-decisions/v6-stage1.json"
    decision_raw = b"v6-stage1-decision"
    decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
    _write(root, decision_path, decision_raw)
    sealed = _projected_evidence(
        root=root,
        scope="fresh-sealed",
        projection_kind=projection_kind,
        family_count=family_count,
        asset_count=asset_count,
        per_stratum_asset_count=per_stratum,
        profile_revision=_FINAL_REVISION,
        configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        parent_decision_path=(decision_path if decision == "continue" else None),
        parent_decision_sha256=(decision_sha256 if decision == "continue" else None),
    )
    calibration_evidence = _projected_evidence(
        root=root,
        scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
        profile_revision=_CALIBRATION_REVISION,
        configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
    )
    (
        sealed_source,
        sealed_execution,
        sealed_bundle,
        sealed_cases,
        sealed_source_path,
        sealed_execution_path,
    ) = sealed
    (
        calibration_source,
        calibration_execution,
        calibration_bundle,
        calibration_cases,
        calibration_source_path,
        calibration_execution_path,
    ) = calibration_evidence
    calibration_receipt = RasterTextCalibrationReceiptV6(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-staged-train-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        calibration_raster_text_profile_revision=_CALIBRATION_REVISION,
        calibration_runtime_configuration_path="profiles/calibration-v18.json",
        calibration_runtime_configuration_sha256=(_CALIBRATION_CONFIGURATION_SHA256),
        extractor_revision=_EXTRACTOR_REVISION,
        selection_receipt_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA256,
        projection_receipt_path=(calibration_execution.projection_receipt_relative_path),
        projection_receipt_sha256=calibration_execution.projection_receipt_sha256,
        source_gold_receipt_path=calibration_source_path,
        source_gold_receipt_sha256=calibration_execution.source_gold_receipt_sha256,
        candidate_execution_path=calibration_execution_path,
        candidate_execution_sha256=hashlib.sha256(
            (root / calibration_execution_path).read_bytes()
        ).hexdigest(),
        minimum_page_confidence=0.5,
        latency_p95_seconds=9.0,
    )
    calibration_receipt_path = "synthetic-raster-text-v1/test-calibration-receipts/v6.json"
    _write(
        root,
        calibration_receipt_path,
        _canonical(calibration_receipt.model_dump(mode="json", warnings=False)),
    )

    def verified_evidence(
        **kwargs: object,
    ) -> tuple[object, object, object, list[object]]:
        if kwargs["source_gold_receipt_relative_path"] == sealed_source_path:
            return sealed_source, sealed_execution, sealed_bundle, sealed_cases
        return (
            calibration_source,
            calibration_execution,
            calibration_bundle,
            calibration_cases,
        )

    stage_decision = SimpleNamespace(
        decision=decision,
        source_gold_receipt_relative_path=sealed_source_path,
        source_gold_receipt_sha256=sealed_execution.source_gold_receipt_sha256,
        candidate_execution_relative_path=(
            sealed_execution_path if decision == "early_pass" else "stage1-execution.json"
        ),
        candidate_execution_sha256=(
            hashlib.sha256((root / sealed_execution_path).read_bytes()).hexdigest()
            if decision == "early_pass"
            else "1" * 64
        ),
        projection_receipt_relative_path=(
            sealed_execution.projection_receipt_relative_path
            if decision == "early_pass"
            else "stage1-projection.json"
        ),
        projection_receipt_sha256=(
            sealed_execution.projection_receipt_sha256 if decision == "early_pass" else "2" * 64
        ),
    )
    scores = [
        RasterTextSealedProfileScoreV6(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=0.99,
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]
    warm = RasterTextWarmServiceSmokeReceiptV1.model_construct(warm_latency_p95_seconds=9.0)
    warm_raw = _canonical(warm.model_dump(mode="json", warnings=False))
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence_v6",
        verified_evidence,
    )
    monkeypatch.setattr(
        activation,
        "verify_fresh_raster_text_sealed_stage1_decision_v6",
        lambda **_kwargs: stage_decision,
    )
    monkeypatch.setattr(
        activation,
        "_resolved_v18_calibration_profile",
        lambda **_kwargs: (calibration_profile, _CALIBRATION_CONFIGURATION_SHA256),
    )
    monkeypatch.setattr(
        activation,
        "_load_verified_final_warm_smoke",
        lambda **_kwargs: (warm, warm_raw),
    )
    monkeypatch.setattr(activation, "_calibrated_floor", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(
        activation,
        "evaluate_synthetic_raster_semantic_text_gates_v3",
        lambda _cases: scores,
    )
    monkeypatch.setattr(
        activation,
        "synthetic_raster_semantic_text_gate_passes_v3",
        lambda _scores: True,
    )

    relative_path, digest = activation.write_raster_semantic_text_fresh_sealed_evaluation_v6(
        document_preview_root=root,
        source_gold_receipt_relative_path=sealed_source_path,
        candidate_execution_relative_path=sealed_execution_path,
        calibration_receipt_relative_path=calibration_receipt_path,
        stage1_decision_receipt_relative_path=decision_path,
        warm_service_smoke_receipt_relative_path="warm/final-v18.json",
        profile=final_profile,
        runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
    )
    raw = (root / relative_path).read_bytes()
    receipt = RasterTextSealedEvaluationV6.model_validate_json(raw)
    assert hashlib.sha256(raw).hexdigest() == digest
    assert receipt.schema_version == "6.0"
    assert receipt.evidence_revision == ("source-attested-fresh-staged-train-holdout-v2")
    assert receipt.projection_kind == projection_kind
    assert receipt.stage1_decision == decision
    assert receipt.asset_count == asset_count
    assert receipt.hard_timeout_seconds == 30


@pytest.mark.parametrize(
    ("projection_kind", "decision", "family_count", "asset_count", "per_stratum"),
    [
        ("sealed-stage1-50-family-v1", "early_pass", 50, 1_200, 100),
        ("sealed-stage2-100-family-v1", "continue", 100, 2_400, 200),
    ],
)
def test_v7_sealed_writer_emits_finetune_bound_stage_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"],
    decision: Literal["early_pass", "continue"],
    family_count: Literal[50, 100],
    asset_count: Literal[1_200, 2_400],
    per_stratum: Literal[100, 200],
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    final_profile = _v19_profile(revision=_FINAL_REVISION, confidence=0.5)
    calibration_profile = _v19_profile(
        revision=_CALIBRATION_REVISION,
        confidence=0.0,
    )
    decision_path = "synthetic-raster-text-v1/test-decisions/v7-stage1.json"
    decision_raw = b"v7-stage1-decision"
    decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
    _write(root, decision_path, decision_raw)
    sealed = _projected_evidence(
        root=root,
        scope="fresh-sealed",
        projection_kind=projection_kind,
        family_count=family_count,
        asset_count=asset_count,
        per_stratum_asset_count=per_stratum,
        profile_revision=_FINAL_REVISION,
        configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        parent_decision_path=(decision_path if decision == "continue" else None),
        parent_decision_sha256=(decision_sha256 if decision == "continue" else None),
    )
    calibration_evidence = _projected_evidence(
        root=root,
        scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
        profile_revision=_CALIBRATION_REVISION,
        configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
    )
    (
        sealed_source,
        sealed_execution,
        sealed_bundle,
        sealed_cases,
        sealed_source_path,
        sealed_execution_path,
    ) = sealed
    (
        calibration_source,
        calibration_execution,
        calibration_bundle,
        calibration_cases,
        calibration_source_path,
        calibration_execution_path,
    ) = calibration_evidence
    calibration_receipt = RasterTextCalibrationReceiptV7(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision=("source-attested-public-finetuned-staged-train-holdout-v2"),
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        calibration_raster_text_profile_revision=_CALIBRATION_REVISION,
        calibration_runtime_configuration_path="profiles/calibration-v19.json",
        calibration_runtime_configuration_sha256=(_CALIBRATION_CONFIGURATION_SHA256),
        extractor_revision=_EXTRACTOR_REVISION,
        selection_receipt_path=_SELECTION_PATH,
        selection_receipt_sha256=_SELECTION_SHA256,
        projection_receipt_path=(calibration_execution.projection_receipt_relative_path),
        projection_receipt_sha256=(calibration_execution.projection_receipt_sha256),
        source_gold_receipt_path=calibration_source_path,
        source_gold_receipt_sha256=(calibration_execution.source_gold_receipt_sha256),
        candidate_execution_path=calibration_execution_path,
        candidate_execution_sha256=hashlib.sha256(
            (root / calibration_execution_path).read_bytes()
        ).hexdigest(),
        source_recognizer_release_sha256=_RECOGNIZER_RELEASE_SHA256,
        source_recognizer_receipt_sha256=_RECOGNIZER_RECEIPT_SHA256,
        source_recognizer_training_intent_sha256=_RECOGNIZER_INTENT_SHA256,
        source_recognizer_parent_selection_receipt_sha256=(_PARENT_SELECTION_SHA256),
        minimum_page_confidence=0.5,
        latency_p95_seconds=9.0,
    )
    calibration_receipt_path = "synthetic-raster-text-v1/test-calibration-receipts/v7.json"
    _write(
        root,
        calibration_receipt_path,
        _canonical(calibration_receipt.model_dump(mode="json", warnings=False)),
    )

    def verified_evidence(
        **kwargs: object,
    ) -> tuple[object, object, object, list[object]]:
        if kwargs["source_gold_receipt_relative_path"] == sealed_source_path:
            return sealed_source, sealed_execution, sealed_bundle, sealed_cases
        return (
            calibration_source,
            calibration_execution,
            calibration_bundle,
            calibration_cases,
        )

    stage_decision = SimpleNamespace(
        decision=decision,
        source_gold_receipt_relative_path=sealed_source_path,
        source_gold_receipt_sha256=sealed_execution.source_gold_receipt_sha256,
        candidate_execution_relative_path=(
            sealed_execution_path if decision == "early_pass" else "stage1-execution.json"
        ),
        candidate_execution_sha256=(
            hashlib.sha256((root / sealed_execution_path).read_bytes()).hexdigest()
            if decision == "early_pass"
            else "1" * 64
        ),
        projection_receipt_relative_path=(
            sealed_execution.projection_receipt_relative_path
            if decision == "early_pass"
            else "stage1-projection.json"
        ),
        projection_receipt_sha256=(
            sealed_execution.projection_receipt_sha256 if decision == "early_pass" else "2" * 64
        ),
    )
    scores = [
        RasterTextSealedProfileScoreV7(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=0.99,
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]
    warm = RasterTextWarmServiceSmokeReceiptV1.model_construct(warm_latency_p95_seconds=9.0)
    warm_raw = _canonical(warm.model_dump(mode="json", warnings=False))
    selection = RasterTextFreshScopeSelectionReceiptV2.model_construct(
        parent_selection_receipt_sha256=_PARENT_SELECTION_SHA256,
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence_v6",
        verified_evidence,
    )
    monkeypatch.setattr(
        activation,
        "verify_fresh_raster_text_sealed_stage1_decision_v6",
        lambda **_kwargs: stage_decision,
    )
    monkeypatch.setattr(
        activation,
        "_resolved_v19_calibration_profile",
        lambda **_kwargs: (
            calibration_profile,
            _CALIBRATION_CONFIGURATION_SHA256,
        ),
    )
    monkeypatch.setattr(
        activation,
        "_load_verified_final_warm_smoke",
        lambda **_kwargs: (warm, warm_raw),
    )
    monkeypatch.setattr(
        activation,
        "_fresh_scope_context",
        lambda **_kwargs: SimpleNamespace(selection=selection),
    )
    monkeypatch.setattr(
        activation,
        "_calibrated_floor",
        lambda *_args, **_kwargs: 0.5,
    )
    monkeypatch.setattr(
        activation,
        "evaluate_synthetic_raster_semantic_text_gates_v3",
        lambda _cases: scores,
    )
    monkeypatch.setattr(
        activation,
        "synthetic_raster_semantic_text_gate_passes_v3",
        lambda _scores: True,
    )

    relative_path, digest = activation.write_raster_semantic_text_fresh_sealed_evaluation_v7(
        document_preview_root=root,
        source_gold_receipt_relative_path=sealed_source_path,
        candidate_execution_relative_path=sealed_execution_path,
        calibration_receipt_relative_path=calibration_receipt_path,
        stage1_decision_receipt_relative_path=decision_path,
        warm_service_smoke_receipt_relative_path="warm/final-v19.json",
        profile=final_profile,
        runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
    )

    raw = (root / relative_path).read_bytes()
    receipt = RasterTextSealedEvaluationV7.model_validate_json(raw)
    assert hashlib.sha256(raw).hexdigest() == digest
    assert receipt.schema_version == "7.0"
    assert receipt.evidence_revision == ("source-attested-public-finetuned-staged-train-holdout-v2")
    assert receipt.source_recognizer_release_sha256 == (_RECOGNIZER_RELEASE_SHA256)
    assert receipt.source_recognizer_parent_selection_receipt_sha256 == (_PARENT_SELECTION_SHA256)
    assert receipt.projection_kind == projection_kind
    assert receipt.stage1_decision == decision
    assert receipt.asset_count == asset_count


def test_v4_writer_rejects_projection_and_v15_resolver_rejects_v14(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    projected = _projected_evidence(
        root=root,
        scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
        profile_revision=_CALIBRATION_REVISION,
        configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: projected[:4],
    )
    with pytest.raises(SyntheticRasterTextActivationError, match="calibration_invalid"):
        activation.write_raster_semantic_text_fresh_calibration_receipt_v4(
            document_preview_root=root,
            source_gold_receipt_relative_path=projected[4],
            candidate_execution_relative_path=projected[5],
            calibration_runtime_configuration_relative_path="profiles/calibration-v14.json",
            activated_profile=RasterTextOfflineProfileV14.model_construct(),
            activated_runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        )

    v14_profile = RasterTextOfflineProfileV14.model_construct()
    monkeypatch.setattr(
        activation,
        "resolve_paddleocr3_raster_text_candidate",
        lambda **_kwargs: SimpleNamespace(
            status="available",
            profile=v14_profile,
            runtime_configuration_sha256="f" * 64,
        ),
    )
    with pytest.raises(SyntheticRasterTextActivationError, match="calibration_invalid"):
        activation._resolved_v15_calibration_profile(  # noqa: SLF001
            root=root,
            runtime_configuration_relative_path="profiles/calibration-v15.json",
        )


def test_v6_evidence_wrapper_requires_selection_v2_and_reopens_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    source = SimpleNamespace(selection_receipt_relative_path=_SELECTION_PATH)
    execution = RasterTextFreshProjectedCandidateExecutionV1.model_construct(
        scope="fresh-calibration"
    )
    bundle = RasterTextFreshProjectedCandidateBundleV1.model_construct()
    cases: list[object] = [SimpleNamespace(scope="fresh-calibration")]
    observed: dict[str, object] = {}

    def verify(**kwargs: object) -> tuple[object, object, object, list[object]]:
        observed.update(kwargs)
        return source, execution, bundle, cases

    checkpoint_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence",
        verify,
    )
    monkeypatch.setattr(
        activation,
        "_verify_v6_candidate_checkpoints",
        lambda **kwargs: checkpoint_calls.append(kwargs),
    )
    result = activation.verify_source_attested_fresh_raster_text_evidence_v6(
        document_preview_root=root,
        source_gold_receipt_relative_path="gold.json",
        candidate_execution_relative_path="execution.json",
        scope="fresh-calibration",
    )

    assert result == (source, execution, bundle, cases)
    assert observed["required_selection_revision"] == ("source-receipt-train-holdout-v2")
    assert checkpoint_calls == [
        {
            "root": root.resolve(),
            "source": source,
            "execution": execution,
            "bundle": bundle,
            "expected_runner_implementation_sha256": None,
        }
    ]


def test_stage2_parent_verifier_forwards_selection_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    projection = RasterTextFreshEvaluationProjectionReceiptV1.model_construct(
        parent_stage1_decision_relative_path="decisions/stage1.json",
        parent_stage1_decision_sha256="1" * 64,
        full_source_gold_receipt_relative_path="gold/sealed.json",
        full_source_gold_receipt_sha256="2" * 64,
    )
    decision = SimpleNamespace(
        decision="continue",
        source_gold_receipt_relative_path="gold/sealed.json",
        source_gold_receipt_sha256="2" * 64,
    )

    def verify(**kwargs: object) -> object:
        observed.update(kwargs)
        return decision

    monkeypatch.setattr(
        activation,
        "verify_fresh_raster_text_sealed_stage1_decision",
        verify,
    )
    activation._verify_fresh_stage2_parent_decision(  # noqa: SLF001
        root=tmp_path.resolve(),
        projection=projection,
        required_selection_revision="source-receipt-train-holdout-v2",
    )
    assert observed["required_selection_revision"] == ("source-receipt-train-holdout-v2")


def test_auto_activation_v6_uses_only_v18_and_selection_v2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = (tmp_path / "_local" / "data" / "document-preview").resolve()
    root.mkdir(parents=True)
    sidecar = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="1" * 64,
        bundle_root_relative_path="sidecar/release",
    )
    mobile = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="2" * 64,
        bundle_root_relative_path="mobile/release",
    )
    offline = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="3" * 64,
        bundle_root_relative_path="offline/release",
    )
    server_manifest = SimpleNamespace(
        source_mobile_bundle_manifest_sha256=mobile.manifest_sha256,
        source_offline_bundle_manifest_sha256=offline.manifest_sha256,
    )
    server = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="4" * 64,
        bundle_root_relative_path="server/release",
        manifest=server_manifest,
    )
    build_inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-v1/build-inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="test-pdfium"),
    )
    final_profile = _v18_profile(revision=_FINAL_REVISION, confidence=0.5)
    calibration_profile_path = "profiles/calibration-v18.json"
    final_profile_path = "profiles/final-v18.json"
    _write(root, calibration_profile_path, b"calibration-v18")
    _write(root, final_profile_path, b"final-v18")
    calibration_profile_info = SimpleNamespace(
        raster_text_profile_revision=_CALIBRATION_REVISION,
        runtime_configuration_relative_path=calibration_profile_path,
        runtime_configuration_sha256=_CALIBRATION_CONFIGURATION_SHA256,
        minimum_page_confidence=0.0,
    )
    final_profile_info = SimpleNamespace(
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_relative_path=final_profile_path,
        runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        minimum_page_confidence=0.5,
    )
    prepare_calls: list[dict[str, object]] = []

    def prepare(*args: object, **kwargs: object) -> object:
        assert args == (sidecar, server)
        prepare_calls.append(kwargs)
        return (
            calibration_profile_info
            if kwargs["minimum_page_confidence"] == 0.0
            else final_profile_info
        )

    run_calls: list[dict[str, object]] = []

    def run_candidates(**kwargs: object) -> object:
        run_calls.append(kwargs)
        scope = str(kwargs["scope"])
        projection = kwargs.get("fresh_projection_kind")
        suffix = f"{scope}-{projection or 'full'}-{len(run_calls)}"
        return SimpleNamespace(
            source_gold_receipt_relative_path=f"gold/{suffix}.json",
            candidate_execution_relative_path=f"execution/{suffix}.json",
            latency_p95_seconds=8.0,
        )

    selection_calls: list[dict[str, object]] = []
    stage_calls: list[dict[str, object]] = []
    sealed_writer_calls: list[dict[str, object]] = []
    pointer_calls: list[dict[str, object]] = []
    events: list[str] = []
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
        "load_verified_raster_offline_bundle",
        lambda *_args, **_kwargs: offline,
    )
    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_direct_server_paddle_bundle_v1",
        lambda **_kwargs: server,
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
        lambda **_kwargs: events.append("compile"),
    )
    monkeypatch.setattr(
        activation,
        "prepare_raster_text_direct_server_runtime_profile_v18",
        prepare,
    )
    monkeypatch.setattr(
        activation,
        "run_paddleocr3_synthetic_raster_text_candidates",
        run_candidates,
    )
    monkeypatch.setattr(
        activation,
        "write_raster_semantic_text_smoke_quality_screen_v3",
        lambda **_kwargs: ("screen.json", "5" * 64, True),
    )

    def write_selection(**kwargs: object) -> tuple[str, str]:
        selection_calls.append(kwargs)
        return "selection.json", "6" * 64

    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_scope_selection_receipt",
        write_selection,
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence_v6",
        lambda **_kwargs: (object(), object(), object(), [object()]),
    )
    monkeypatch.setattr(
        activation,
        "_calibrated_floor",
        lambda *_args, **_kwargs: 0.5,
    )
    monkeypatch.setattr(
        activation,
        "parse_raster_text_offline_profile",
        lambda _raw: final_profile,
    )
    monkeypatch.setattr(
        activation,
        "run_warm_service_raster_text_smoke",
        lambda **_kwargs: ("warm/final.json", "7" * 64),
    )
    monkeypatch.setattr(
        activation,
        "_require_auto_activation_warm_latency",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        activation,
        "write_raster_semantic_text_fresh_calibration_receipt_v6",
        lambda **_kwargs: ("calibration/v6.json", "8" * 64),
    )

    def write_stage(**kwargs: object) -> tuple[str, str, str]:
        stage_calls.append(kwargs)
        return "decisions/stage1.json", "9" * 64, "continue"

    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_sealed_stage1_decision",
        write_stage,
    )

    def write_sealed(**kwargs: object) -> tuple[str, str]:
        sealed_writer_calls.append(kwargs)
        return "sealed/v6.json", "a" * 64

    monkeypatch.setattr(
        activation,
        "write_raster_semantic_text_fresh_sealed_evaluation_v6",
        write_sealed,
    )
    monkeypatch.setattr(
        activation,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: sidecar,
    )
    monkeypatch.setattr(
        activation,
        "load_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile,
    )
    monkeypatch.setattr(
        activation,
        "load_verified_raster_offline_bundle_release",
        lambda **_kwargs: offline,
    )
    monkeypatch.setattr(
        activation,
        "load_verified_raster_text_direct_server_paddle_bundle_v1",
        lambda **_kwargs: server,
    )
    monkeypatch.setattr(
        activation,
        "resolve_paddleocr3_raster_text_candidate",
        lambda **_kwargs: SimpleNamespace(
            status="available",
            profile=final_profile,
            runtime_configuration_sha256=_FINAL_CONFIGURATION_SHA256,
        ),
    )
    monkeypatch.setattr(
        activation,
        "_load_fresh_dataset_receipts",
        lambda *_args: (
            SimpleNamespace(build_inputs_relative_path=build_inputs.build_inputs_relative_path),
            "b" * 64,
            object(),
            "c" * 64,
        ),
    )
    pointer = RasterTextActivationPointerV6.model_construct(
        raster_text_profile_revision=_FINAL_REVISION
    )

    def write_pointer(**kwargs: object) -> tuple[str, object]:
        pointer_calls.append(kwargs)
        events.append("pointer")
        return "d" * 64, pointer

    monkeypatch.setattr(
        activation,
        "write_source_attested_synthetic_raster_text_pointer",
        write_pointer,
    )

    result = activation.auto_activate_synthetic_raster_text_v6(document_preview_root=root)

    assert result.raster_text_profile_revision == _FINAL_REVISION
    assert [item["minimum_page_confidence"] for item in prepare_calls] == [0.0, 0.5]
    fresh_runs = [
        item for item in run_calls if item["scope"] in {"fresh-calibration", "fresh-sealed"}
    ]
    assert len(fresh_runs) == 3
    assert {str(item["fresh_selection_revision"]) for item in fresh_runs} == {
        "source-receipt-train-holdout-v2"
    }
    assert fresh_runs[-1]["fresh_projection_kind"] == ("sealed-stage2-100-family-v1")
    assert fresh_runs[-1]["parent_stage1_decision_relative_path"] == ("decisions/stage1.json")
    assert selection_calls[0]["selection_revision"] == ("source-receipt-train-holdout-v2")
    assert stage_calls[0]["selection_revision"] == ("source-receipt-train-holdout-v2")
    assert sealed_writer_calls[0]["profile"] is final_profile
    assert "repair_audit_dataset_receipt_relative_path" not in pointer_calls[0]
    assert "repair_audit_evaluation_relative_path" not in pointer_calls[0]
    assert events == ["compile", "pointer"]
