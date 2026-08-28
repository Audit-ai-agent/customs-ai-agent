"""Focused contracts for projected/staged raster-text activation schema V5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast, Literal

from pydantic import ValidationError
import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation
from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    RasterTextFreshEvaluationProjectionReceiptV1,
    RasterTextFreshProjectedCandidateBundleV1,
    RasterTextFreshProjectedCandidateExecutionV1,
    SyntheticRasterTextActivationError,
)
import customs_ai_agent.service.raster_text_preview_runtime as runtime
from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextActivationPointerV4,
    RasterTextActivationPointerV5,
    RasterTextCalibrationReceiptV4,
    RasterTextCalibrationReceiptV5,
    RasterTextOfflineProfileV15,
    RasterTextRuntimeError,
    RasterTextSealedEvaluationV4,
    RasterTextSealedEvaluationV5,
    RasterTextSealedProfileScoreV4,
    RasterTextSealedProfileScoreV5,
    RasterTextWarmServiceSmokeReceiptV1,
)


_FINAL_REVISION = "synthetic-raster-text-v1:" + ("a" * 24)
_CALIBRATION_REVISION = "synthetic-raster-text-v1:" + ("b" * 24)
_EXTRACTOR_REVISION = "direct-text-test-v15-onednn"
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


def _canonical_pointer(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _passing_scores() -> list[RasterTextSealedProfileScoreV5]:
    return [
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


def _v5_models(
    *,
    runtime_configuration_sha256: str = "c" * 64,
    stage: Literal[
        "sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"
    ] = "sealed-stage1-50-family-v1",
) -> tuple[
    RasterTextActivationPointerV5,
    RasterTextCalibrationReceiptV5,
    RasterTextSealedEvaluationV5,
]:
    stage1_decision: Literal["early_pass", "continue"]
    family_count: Literal[50, 100]
    asset_count: Literal[1_200, 2_400]
    if stage == "sealed-stage1-50-family-v1":
        stage1_decision, family_count, asset_count = "early_pass", 50, 1_200
    else:
        stage1_decision, family_count, asset_count = "continue", 100, 2_400
    pointer = RasterTextActivationPointerV5(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-staged-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_path="profiles/final-v15.json",
        runtime_configuration_sha256=runtime_configuration_sha256,
        sealed_evaluation_path="fresh-staged-evaluations-v5/final.json",
        sealed_evaluation_sha256="d" * 64,
    )
    calibration = RasterTextCalibrationReceiptV5(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-staged-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256=runtime_configuration_sha256,
        calibration_raster_text_profile_revision=_CALIBRATION_REVISION,
        calibration_runtime_configuration_path="profiles/calibration-v15.json",
        calibration_runtime_configuration_sha256="e" * 64,
        extractor_revision=_EXTRACTOR_REVISION,
        selection_receipt_path="fresh-selections/selection.json",
        selection_receipt_sha256="f" * 64,
        projection_receipt_path="fresh-projections/calibration.json",
        projection_receipt_sha256="1" * 64,
        source_gold_receipt_path="fresh-gold/fresh-calibration/source.json",
        source_gold_receipt_sha256="2" * 64,
        candidate_execution_path="fresh-executions/fresh-calibration/execution.json",
        candidate_execution_sha256="3" * 64,
        minimum_page_confidence=0.5,
        latency_p95_seconds=9.0,
    )
    evaluation = RasterTextSealedEvaluationV5(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-staged-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind=stage,
        stage1_decision=stage1_decision,
        family_count=family_count,
        asset_count=asset_count,
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256=runtime_configuration_sha256,
        extractor_revision=_EXTRACTOR_REVISION,
        selection_receipt_path=calibration.selection_receipt_path,
        selection_receipt_sha256=calibration.selection_receipt_sha256,
        projection_receipt_path="fresh-projections/sealed.json",
        projection_receipt_sha256="4" * 64,
        stage1_decision_receipt_path="fresh-stage-decisions/stage1.json",
        stage1_decision_receipt_sha256="5" * 64,
        source_gold_receipt_path="fresh-gold/fresh-sealed/source.json",
        source_gold_receipt_sha256="6" * 64,
        calibration_receipt_path="fresh-calibration-receipts-v5/calibration.json",
        calibration_receipt_sha256="7" * 64,
        candidate_execution_path="fresh-executions/fresh-sealed/execution.json",
        candidate_execution_sha256="8" * 64,
        warm_service_smoke_receipt_path="warm-smoke/final.json",
        warm_service_smoke_receipt_sha256="9" * 64,
        latency_p95_seconds=9.0,
        service_warm_latency_p95_seconds=9.0,
        minimum_page_confidence=0.5,
        scores=_passing_scores(),
    )
    return pointer, calibration, evaluation


def _profile(*, revision: str, minimum_page_confidence: float) -> RasterTextOfflineProfileV15:
    return RasterTextOfflineProfileV15.model_construct(
        schema_version="15.0",
        engine_kind=(
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
            "semantic_repair_v2_onednn"
        ),
        extractor_revision=_EXTRACTOR_REVISION,
        minimum_page_confidence=minimum_page_confidence,
        profile_fingerprint_sha256="0" * 64,
        raster_text_profile_revision=revision,
    )


def test_v5_parser_is_strict_and_mixed_generations_fail_closed() -> None:
    pointer, calibration, evaluation = _v5_models()
    raw = _canonical_pointer(pointer.model_dump(mode="json", warnings=False))
    assert runtime.parse_raster_text_activation_pointer(raw) == pointer
    assert (
        runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
        )
        == "source-attested-semantic-text-v3"
    )

    values = pointer.model_dump(mode="json", warnings=False)
    values.pop("latency_gate_revision")
    with pytest.raises(RasterTextRuntimeError, match="activation_invalid"):
        runtime.parse_raster_text_activation_pointer(_canonical_pointer(values))

    v4_pointer = RasterTextActivationPointerV4(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_path="profiles/final-v14.json",
        runtime_configuration_sha256="c" * 64,
        sealed_evaluation_path="fresh-evaluations-v4/final.json",
        sealed_evaluation_sha256="d" * 64,
    )
    with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
        runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=v4_pointer,
            evaluation=evaluation,
            calibration=calibration,
        )


def test_v5_receipts_enforce_the_ten_second_latency_gate() -> None:
    _pointer, calibration, evaluation = _v5_models()
    calibration_values = calibration.model_dump(mode="json", warnings=False)
    calibration_values["latency_p95_seconds"] = 10.0001
    with pytest.raises(ValidationError):
        RasterTextCalibrationReceiptV5.model_validate(calibration_values)

    evaluation_values = evaluation.model_dump(mode="json", warnings=False)
    evaluation_values["service_warm_latency_p95_seconds"] = 10.0001
    with pytest.raises(ValidationError):
        RasterTextSealedEvaluationV5.model_validate(evaluation_values)


def _write_v5_verifier_materials(
    *,
    root: Path,
    calibration: RasterTextCalibrationReceiptV5,
    evaluation: RasterTextSealedEvaluationV5,
) -> tuple[RasterTextCalibrationReceiptV5, RasterTextSealedEvaluationV5]:
    selection_raw = b"selection"
    selection_path = root / evaluation.selection_receipt_path
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_bytes(selection_raw)

    calibration_configuration_raw = b"calibration-profile"
    calibration_configuration_path = root / calibration.calibration_runtime_configuration_path
    calibration_configuration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_configuration_path.write_bytes(calibration_configuration_raw)

    calibration_projection = RasterTextFreshEvaluationProjectionReceiptV1(
        projection_seed_sha256=_PROJECTION_SEED_SHA256,
        source_scope="fresh-calibration",
        projection_kind="calibration-10-family-v1",
        synthetic_raster_receipt_sha256="a" * 64,
        selection_receipt_relative_path=calibration.selection_receipt_path,
        selection_receipt_sha256=hashlib.sha256(selection_raw).hexdigest(),
        full_source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
        full_source_gold_receipt_sha256=calibration.source_gold_receipt_sha256,
        family_ids=[f"cal-{index:03d}" for index in range(10)],
        family_count=10,
        asset_count=240,
        per_stratum_asset_count=20,
    )
    is_stage2 = evaluation.projection_kind == "sealed-stage2-100-family-v1"
    sealed_projection = RasterTextFreshEvaluationProjectionReceiptV1(
        projection_seed_sha256=_PROJECTION_SEED_SHA256,
        source_scope="fresh-sealed",
        projection_kind=evaluation.projection_kind,
        synthetic_raster_receipt_sha256="a" * 64,
        selection_receipt_relative_path=evaluation.selection_receipt_path,
        selection_receipt_sha256=hashlib.sha256(selection_raw).hexdigest(),
        full_source_gold_receipt_relative_path=evaluation.source_gold_receipt_path,
        full_source_gold_receipt_sha256=evaluation.source_gold_receipt_sha256,
        family_ids=[f"sealed-{index:03d}" for index in range(evaluation.family_count)],
        family_count=evaluation.family_count,
        asset_count=evaluation.asset_count,
        per_stratum_asset_count=200 if is_stage2 else 100,
        parent_stage1_decision_relative_path=(
            evaluation.stage1_decision_receipt_path if is_stage2 else None
        ),
        parent_stage1_decision_sha256=(
            evaluation.stage1_decision_receipt_sha256 if is_stage2 else None
        ),
    )
    calibration_projection_raw = _canonical(
        calibration_projection.model_dump(mode="json", warnings=False)
    )
    sealed_projection_raw = _canonical(sealed_projection.model_dump(mode="json", warnings=False))
    calibration_projection_path = root / calibration.projection_receipt_path
    sealed_projection_path = root / evaluation.projection_receipt_path
    calibration_projection_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_projection_path.write_bytes(calibration_projection_raw)
    sealed_projection_path.write_bytes(sealed_projection_raw)
    return (
        calibration.model_copy(
            update={
                "selection_receipt_sha256": hashlib.sha256(selection_raw).hexdigest(),
                "calibration_runtime_configuration_sha256": hashlib.sha256(
                    calibration_configuration_raw
                ).hexdigest(),
                "projection_receipt_sha256": hashlib.sha256(calibration_projection_raw).hexdigest(),
            }
        ),
        evaluation.model_copy(
            update={
                "selection_receipt_sha256": hashlib.sha256(selection_raw).hexdigest(),
                "projection_receipt_sha256": hashlib.sha256(sealed_projection_raw).hexdigest(),
            }
        ),
    )


@pytest.mark.parametrize(
    "stage",
    ["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"],
)
def test_v5_verifier_recomputes_projection_quality_and_stage_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"],
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    pointer, calibration, evaluation = _v5_models(stage=stage)
    calibration, evaluation = _write_v5_verifier_materials(
        root=root,
        calibration=calibration,
        evaluation=evaluation,
    )
    final_profile = _profile(revision=_FINAL_REVISION, minimum_page_confidence=0.5)
    calibration_profile = _profile(
        revision=_CALIBRATION_REVISION,
        minimum_page_confidence=0.0,
    )
    warm = RasterTextWarmServiceSmokeReceiptV1(
        artifact_kind="synthetic_raster_text_warm_service_smoke",
        dataset_id="synthetic-raster-text-v1",
        scope="smoke",
        source_truth_kind="pdf_content_stream_text_v1",
        execution_mode="persistent_service_warm",
        service_protocol="customs-raster-text-sidecar-v1-service",
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256=pointer.runtime_configuration_sha256,
        extractor_revision=_EXTRACTOR_REVISION,
        source_gold_receipt_path="smoke/source.json",
        source_gold_receipt_sha256="1" * 64,
        candidate_execution_path="smoke/execution.json",
        candidate_execution_sha256="2" * 64,
        candidate_bundle_sha256="3" * 64,
        equality_mode="exact_canonical_result_v1",
        equality_set_sha256="4" * 64,
        asset_count=24,
        startup_latency_seconds=1.0,
        startup_timeout_seconds=60,
        warm_latency_p95_seconds=9.0,
        request_timeout_seconds=30,
    )
    calls: list[tuple[str, object]] = []

    def verified_fresh_evidence(**kwargs: object) -> tuple[object, object, object, list[object]]:
        scope = cast(str, kwargs["scope"])
        calls.append(("evidence", scope))
        is_sealed = scope == "fresh-sealed"
        receipt = evaluation if is_sealed else calibration
        profile = final_profile if is_sealed else calibration_profile
        configuration_sha256 = (
            pointer.runtime_configuration_sha256
            if is_sealed
            else calibration.calibration_runtime_configuration_sha256
        )
        count = evaluation.asset_count if is_sealed else calibration.asset_count
        return (
            SimpleNamespace(
                scope=scope,
                selection_receipt_relative_path=receipt.selection_receipt_path,
                selection_receipt_sha256=receipt.selection_receipt_sha256,
            ),
            RasterTextFreshProjectedCandidateExecutionV1.model_construct(
                scope=scope,
                selection_receipt_relative_path=receipt.selection_receipt_path,
                selection_receipt_sha256=receipt.selection_receipt_sha256,
                source_gold_receipt_relative_path=receipt.source_gold_receipt_path,
                source_gold_receipt_sha256=receipt.source_gold_receipt_sha256,
                projection_receipt_relative_path=receipt.projection_receipt_path,
                projection_receipt_sha256=receipt.projection_receipt_sha256,
                candidate_count=count,
                raster_text_profile_revision=profile.raster_text_profile_revision,
                runtime_configuration_sha256=configuration_sha256,
                extractor_revision=profile.extractor_revision,
                latency_p95_seconds=9.0,
            ),
            RasterTextFreshProjectedCandidateBundleV1.model_construct(
                scope=scope,
                selection_receipt_relative_path=receipt.selection_receipt_path,
                selection_receipt_sha256=receipt.selection_receipt_sha256,
                source_gold_receipt_sha256=receipt.source_gold_receipt_sha256,
                projection_receipt_relative_path=receipt.projection_receipt_path,
                projection_receipt_sha256=receipt.projection_receipt_sha256,
                candidates=[None] * count,
                raster_text_profile_revision=profile.raster_text_profile_revision,
                runtime_configuration_sha256=configuration_sha256,
                extractor_revision=profile.extractor_revision,
            ),
            [SimpleNamespace(scope=scope)],
        )

    def verified_stage_decision(**kwargs: object) -> object:
        calls.append(("stage-decision", kwargs["decision_relative_path"]))
        return SimpleNamespace(
            decision=evaluation.stage1_decision,
            source_gold_receipt_relative_path=evaluation.source_gold_receipt_path,
            source_gold_receipt_sha256=evaluation.source_gold_receipt_sha256,
            candidate_execution_relative_path=(
                evaluation.candidate_execution_path
                if stage == "sealed-stage1-50-family-v1"
                else "fresh-executions/fresh-sealed/stage1.json"
            ),
            candidate_execution_sha256=(
                evaluation.candidate_execution_sha256
                if stage == "sealed-stage1-50-family-v1"
                else "a" * 64
            ),
            projection_receipt_relative_path=(
                evaluation.projection_receipt_path
                if stage == "sealed-stage1-50-family-v1"
                else "fresh-projections/sealed-stage1.json"
            ),
            projection_receipt_sha256=(
                evaluation.projection_receipt_sha256
                if stage == "sealed-stage1-50-family-v1"
                else "b" * 64
            ),
        )

    monkeypatch.setattr(
        runtime, "parse_raster_text_offline_profile", lambda _raw: calibration_profile
    )
    monkeypatch.setattr(
        synthetic_raster_text_activation,
        "verify_source_attested_fresh_raster_text_evidence",
        verified_fresh_evidence,
    )
    monkeypatch.setattr(
        synthetic_raster_text_activation,
        "verify_fresh_raster_text_sealed_stage1_decision",
        verified_stage_decision,
    )
    monkeypatch.setattr(
        runtime,
        "_recalculate_source_attested_raster_text_quality",
        lambda **_kwargs: (evaluation.scores, 0.5, True),
    )

    runtime._verify_fresh_v5_evidence_chain(  # noqa: SLF001
        root=root,
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
        profile=final_profile,
        warm_service_smoke_raw=_canonical(warm.model_dump(mode="json", warnings=False)),
    )
    assert calls == [
        ("evidence", "fresh-sealed"),
        ("evidence", "fresh-calibration"),
        ("stage-decision", evaluation.stage1_decision_receipt_path),
    ]

    def rejected_stage_decision(**_kwargs: object) -> object:
        raise SyntheticRasterTextActivationError("raw_stage_counts_do_not_match")

    monkeypatch.setattr(
        synthetic_raster_text_activation,
        "verify_fresh_raster_text_sealed_stage1_decision",
        rejected_stage_decision,
    )
    with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
        runtime._verify_fresh_v5_evidence_chain(  # noqa: SLF001
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
            profile=final_profile,
            warm_service_smoke_raw=_canonical(warm.model_dump(mode="json", warnings=False)),
        )


def test_v5_pointer_writer_derives_staged_and_latency_revisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_relative = "profiles/final-v15.json"
    evaluation_relative = "fresh-staged-evaluations-v5/final.json"
    configuration_path = root / configuration_relative
    evaluation_path = root / evaluation_relative
    configuration_path.parent.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    configuration_raw = b"profile-v15"
    configuration_path.write_bytes(configuration_raw)
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    _pointer, _calibration, evaluation = _v5_models(
        runtime_configuration_sha256=configuration_sha256
    )
    evaluation_path.write_bytes(_canonical(evaluation.model_dump(mode="json", warnings=False)))
    profile = SimpleNamespace(
        raster_text_profile_revision=_FINAL_REVISION,
        extractor_revision=_EXTRACTOR_REVISION,
    )
    observed: list[RasterTextActivationPointerV5] = []
    monkeypatch.setattr(runtime, "parse_raster_text_offline_profile", lambda _raw: profile)

    def accept_pointer(**kwargs: object) -> SimpleNamespace:
        parsed = runtime.parse_raster_text_activation_pointer(cast(bytes, kwargs["approval_raw"]))
        assert isinstance(parsed, RasterTextActivationPointerV5)
        observed.append(parsed)
        return SimpleNamespace()

    monkeypatch.setattr(runtime, "_load_binding_from_pointer", accept_pointer)
    _digest, pointer = runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path=configuration_relative,
        sealed_evaluation_relative_path=evaluation_relative,
    )

    assert isinstance(pointer, RasterTextActivationPointerV5)
    assert pointer.evidence_revision == "source-attested-fresh-staged-holdout-v2"
    assert pointer.latency_gate_revision == "warm-p95-10s-v1"
    assert observed == [pointer]


def test_v4_models_remain_v14_full_holdout_only() -> None:
    calibration = RasterTextCalibrationReceiptV4(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256="c" * 64,
        calibration_raster_text_profile_revision=_CALIBRATION_REVISION,
        calibration_runtime_configuration_path="profiles/calibration-v14.json",
        calibration_runtime_configuration_sha256="d" * 64,
        extractor_revision="direct-text-test-v14",
        selection_receipt_path="fresh-selections/selection.json",
        selection_receipt_sha256="e" * 64,
        source_gold_receipt_path="fresh-gold/calibration.json",
        source_gold_receipt_sha256="f" * 64,
        candidate_execution_path="fresh-executions/calibration.json",
        candidate_execution_sha256="1" * 64,
        minimum_page_confidence=0.5,
        latency_p95_seconds=20.0,
    )
    evaluation = RasterTextSealedEvaluationV4(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=_FINAL_REVISION,
        runtime_configuration_sha256="c" * 64,
        extractor_revision="direct-text-test-v14",
        selection_receipt_path=calibration.selection_receipt_path,
        selection_receipt_sha256=calibration.selection_receipt_sha256,
        source_gold_receipt_path="fresh-gold/sealed.json",
        source_gold_receipt_sha256="2" * 64,
        calibration_receipt_path="fresh-calibration-v4/calibration.json",
        calibration_receipt_sha256="3" * 64,
        candidate_execution_path="fresh-executions/sealed.json",
        candidate_execution_sha256="4" * 64,
        warm_service_smoke_receipt_path="warm-smoke/final.json",
        warm_service_smoke_receipt_sha256="5" * 64,
        latency_p95_seconds=20.0,
        service_warm_latency_p95_seconds=20.0,
        minimum_page_confidence=0.5,
        scores=[
            RasterTextSealedProfileScoreV4.model_validate(
                score.model_dump(mode="json", warnings=False)
            )
            for score in _passing_scores()
        ],
    )
    assert calibration.schema_version == "4.0"
    assert evaluation.schema_version == "4.0"
