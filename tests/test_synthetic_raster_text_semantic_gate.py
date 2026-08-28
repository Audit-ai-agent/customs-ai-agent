from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.evaluation.synthetic_raster_text import (
    RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
    RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    RasterTextEvaluationCaseV1,
    RasterTextGateScoreV1,
    RasterTextGoldArtifactV1,
    canonicalize_synthetic_raster_semantic_text_v2,
    canonicalize_synthetic_raster_semantic_text_v3,
    evaluate_synthetic_raster_semantic_text_case_v2,
    evaluate_synthetic_raster_semantic_text_case_v3,
    evaluate_synthetic_raster_text_case,
    synthetic_raster_semantic_text_gate_passes_v3,
    synthetic_raster_semantic_text_preview_draft_gate_passes_v1,
)
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionResultV1,
    RasterTextPageV1,
    canonical_raster_text,
    canonical_raster_text_sha256,
)


_SHA = "a" * 64
_REVISION = "synthetic-raster-text-v1:0123456789abcdef01234567"


def test_preview_draft_gate_relaxes_only_critical_exact_to_97_percent() -> None:
    scores = [
        RasterTextGateScoreV1(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.02,
            critical_value_exact_match=0.97,
        )
        for document_type, profile_id in product(
            ("bill_of_lading", "commercial_invoice", "packing_list"),
            ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
        )
    ]

    assert synthetic_raster_semantic_text_preview_draft_gate_passes_v1(scores)
    assert not synthetic_raster_semantic_text_gate_passes_v3(scores)
    assert not synthetic_raster_semantic_text_preview_draft_gate_passes_v1(
        [scores[0].model_copy(update={"critical_value_exact_match": 0.9699}), *scores[1:]]
    )
    assert not synthetic_raster_semantic_text_preview_draft_gate_passes_v1(
        [scores[0].model_copy(update={"character_error_rate": 0.0201}), *scores[1:]]
    )


def _result(*, asset_id: str, text: str) -> RasterTextExtractionResultV1:
    pages = [RasterTextPageV1(page_number=1, text=text, confidence=0.9)]
    return RasterTextExtractionResultV1(
        document_asset_id=asset_id,
        extraction_request_id=f"request-{asset_id}",
        raster_text_profile_revision=_REVISION,
        status="preview_ready",
        media_type="image/png",
        byte_size=1,
        source_sha256=_SHA,
        canonical_text=canonical_raster_text(pages),
        canonical_text_sha256=canonical_raster_text_sha256(pages),
        extractor_revision="direct-text-test-v1",
        pages=pages,
        warnings=[],
        needs_review_reasons=[],
    )


def _case(
    *,
    asset_id: str = "asset-1",
    document_type: str = "commercial_invoice",
    profile_id: str = "png-216",
    reference: str,
    observed: str,
    critical_values: list[str] | None = None,
) -> RasterTextEvaluationCaseV1:
    return RasterTextEvaluationCaseV1(
        asset_id=asset_id,
        document_type=document_type,  # type: ignore[arg-type]
        profile_id=profile_id,  # type: ignore[arg-type]
        image_sha256=_SHA,
        gold=RasterTextGoldArtifactV1(
            asset_id=asset_id,
            family_id="family-1",
            document_type=document_type,  # type: ignore[arg-type]
            profile_id=profile_id,  # type: ignore[arg-type]
            source_pdf_sha256=_SHA,
            source_raster_gold_sha256=_SHA,
            canonical_text=reference,
            critical_values=critical_values or ["VALUE"],
        ),
        candidate=_result(asset_id=asset_id, text=observed),
    )


def _passing_matrix_cases() -> list[RasterTextEvaluationCaseV1]:
    return [
        _case(
            asset_id=f"asset-{index}",
            document_type=document_type,
            profile_id=profile_id,
            reference="VALUE",
            observed="VALUE",
        )
        for index, (document_type, profile_id) in enumerate(
            product(
                ("bill_of_lading", "commercial_invoice", "packing_list"),
                ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
            )
        )
    ]


def test_semantic_canonicalizer_is_strict_and_candidate_independent() -> None:
    value = "  Cafe\u0301\r\n■ \u25a1 \u25aa \u25ab \u2610 \u2611 \u2612\tA■B _ AB__CD x_y ! /  "
    assert canonicalize_synthetic_raster_semantic_text_v2(value) == ("Café A■B _ ABCD x_y ! /")


@pytest.mark.parametrize("joined", ["■X", "X■", "[■]"])
def test_semantic_canonicalizer_preserves_non_standalone_markers(joined: str) -> None:
    assert canonicalize_synthetic_raster_semantic_text_v2(joined) == joined


def test_semantic_canonicalizer_rejects_pathological_input() -> None:
    with pytest.raises(ValueError, match="too large"):
        canonicalize_synthetic_raster_semantic_text_v2("A" * 200_001)


def test_v3_removes_only_standalone_attested_circled_digits() -> None:
    circled_digits = " ".join(chr(codepoint) for codepoint in range(0x2460, 0x246A))
    value = f"START\n{circled_digits}\nEND"

    assert canonicalize_synthetic_raster_semantic_text_v2(value) == (f"START {circled_digits} END")
    assert canonicalize_synthetic_raster_semantic_text_v3(value) == "START END"


@pytest.mark.parametrize("joined", ["A①", "①A", "[①]", "①②", "⑟", "⑪"])
def test_v3_preserves_joined_embedded_and_out_of_range_circled_text(joined: str) -> None:
    assert canonicalize_synthetic_raster_semantic_text_v3(joined) == joined


def test_semantic_case_ignores_only_contract_excluded_decoration_and_whitespace() -> None:
    case = _case(
        reference="■\nVALUE____\nA_B",
        observed="VALUE A_B",
        critical_values=["VALUE", "A_B"],
    )
    legacy = evaluate_synthetic_raster_text_case(case)
    semantic = evaluate_synthetic_raster_semantic_text_case_v2(case)
    assert legacy.character_errors > 0
    assert semantic.character_errors == 0
    assert semantic.character_reference == len("VALUE A_B")
    assert (semantic.critical_matches, semantic.critical_total) == (2, 2)


def test_semantic_critical_match_keeps_slash_exact() -> None:
    counts = evaluate_synthetic_raster_semantic_text_case_v2(
        _case(reference="A/B", observed="A B", critical_values=["A/B"])
    )
    assert (counts.critical_matches, counts.critical_total) == (0, 1)


def test_v3_case_excludes_standalone_circled_digit_but_v2_does_not() -> None:
    case = _case(reference="①\nVALUE", observed="VALUE", critical_values=["VALUE"])
    v2 = evaluate_synthetic_raster_semantic_text_case_v2(case)
    v3 = evaluate_synthetic_raster_semantic_text_case_v3(case)

    assert v2.character_errors > 0
    assert v3.character_errors == 0
    assert v3.character_reference == len("VALUE")
    assert (v3.critical_matches, v3.critical_total) == (1, 1)


def test_v2_evidence_requires_explicit_metric_revision() -> None:
    scores = [
        RasterTextGateScoreV1.model_validate(
            {
                "document_type": document_type,
                "profile_id": profile_id,
                "character_error_rate": 0.0,
                "critical_value_exact_match": 1.0,
            }
        )
        for document_type, profile_id in product(
            ("bill_of_lading", "commercial_invoice", "packing_list"),
            ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
        )
    ]
    screen = activation.RasterTextSmokeQualityScreenV2(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
        source_gold_receipt_path="source.json",
        source_gold_receipt_sha256=_SHA,
        candidate_execution_path="candidate.json",
        candidate_execution_sha256=_SHA,
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        scores=scores,
        gate_passed=True,
    )
    dumped = screen.model_dump(mode="json")
    assert dumped["schema_version"] == "2.0"
    assert dumped["metric_revision"] == RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2
    dumped["metric_revision"] = "legacy-text-v1"
    with pytest.raises(ValidationError):
        activation.RasterTextSmokeQualityScreenV2.model_validate(dumped)
    with pytest.raises(ValidationError):
        activation.RasterTextSmokeQualityScreenV1.model_validate(screen.model_dump())


def test_calibration_and_sealed_v2_evidence_bind_the_same_metric_revision() -> None:
    calibration = activation.RasterTextCalibrationReceiptV2(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        calibration_raster_text_profile_revision=_REVISION,
        calibration_runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        source_gold_receipt_path="source.json",
        source_gold_receipt_sha256=_SHA,
        candidate_execution_path="candidate.json",
        candidate_execution_sha256=_SHA,
        minimum_page_confidence=0.5,
        latency_p95_seconds=1.0,
    )
    scores = [
        activation.RasterTextSealedProfileScoreV2.model_validate(
            {
                "document_type": document_type,
                "profile_id": profile_id,
                "character_error_rate": 0.02,
                "critical_value_exact_match": 0.98,
            }
        )
        for document_type, profile_id in product(
            ("bill_of_lading", "commercial_invoice", "packing_list"),
            ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
        )
    ]
    sealed = activation.RasterTextSealedEvaluationV2(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        source_gold_receipt_path="source.json",
        source_gold_receipt_sha256=_SHA,
        calibration_receipt_path="calibration.json",
        calibration_receipt_sha256=_SHA,
        candidate_execution_path="candidate.json",
        candidate_execution_sha256=_SHA,
        warm_service_smoke_receipt_path="warm.json",
        warm_service_smoke_receipt_sha256=_SHA,
        latency_p95_seconds=1.0,
        service_warm_latency_p95_seconds=1.0,
        minimum_page_confidence=0.5,
        scores=scores,
    )
    assert calibration.metric_revision == sealed.metric_revision
    assert calibration.schema_version == sealed.schema_version == "2.0"
    for model_type, value in (
        (activation.RasterTextCalibrationReceiptV2, calibration.model_dump()),
        (activation.RasterTextSealedEvaluationV2, sealed.model_dump()),
    ):
        value.pop("metric_revision")
        with pytest.raises(ValidationError):
            model_type.model_validate(value)


def test_v3_evidence_is_schema_and_metric_disjoint_from_v2() -> None:
    calibration = activation.RasterTextCalibrationReceiptV3(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        calibration_raster_text_profile_revision=_REVISION,
        calibration_runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        source_gold_receipt_path="source.json",
        source_gold_receipt_sha256=_SHA,
        candidate_execution_path="candidate.json",
        candidate_execution_sha256=_SHA,
        minimum_page_confidence=0.5,
        latency_p95_seconds=1.0,
    )
    scores = [
        activation.RasterTextSealedProfileScoreV3.model_validate(
            {
                "document_type": document_type,
                "profile_id": profile_id,
                "character_error_rate": 0.02,
                "critical_value_exact_match": 0.98,
            }
        )
        for document_type, profile_id in product(
            ("bill_of_lading", "commercial_invoice", "packing_list"),
            ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
        )
    ]
    sealed = activation.RasterTextSealedEvaluationV3(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        source_gold_receipt_path="source.json",
        source_gold_receipt_sha256=_SHA,
        calibration_receipt_path="calibration.json",
        calibration_receipt_sha256=_SHA,
        candidate_execution_path="candidate.json",
        candidate_execution_sha256=_SHA,
        warm_service_smoke_receipt_path="warm.json",
        warm_service_smoke_receipt_sha256=_SHA,
        latency_p95_seconds=1.0,
        service_warm_latency_p95_seconds=1.0,
        minimum_page_confidence=0.5,
        scores=scores,
    )

    assert calibration.metric_revision == sealed.metric_revision
    assert calibration.schema_version == sealed.schema_version == "3.0"
    for model_type, value in (
        (activation.RasterTextCalibrationReceiptV3, calibration.model_dump()),
        (activation.RasterTextSealedEvaluationV3, sealed.model_dump()),
    ):
        value.pop("metric_revision")
        with pytest.raises(ValidationError):
            model_type.model_validate(value)
    with pytest.raises(ValidationError):
        activation.RasterTextCalibrationReceiptV2.model_validate(calibration.model_dump())
    with pytest.raises(ValidationError):
        activation.RasterTextSealedEvaluationV2.model_validate(sealed.model_dump())


def test_smoke_writer_separates_v2_metric_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    cases = _passing_matrix_cases()
    source = SimpleNamespace(scope="smoke")
    execution = SimpleNamespace(
        scope="smoke",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        activation,
        "_load_source_gold_cases",
        lambda **_kwargs: (source, execution, SimpleNamespace(scope="smoke"), cases),
    )
    monkeypatch.setattr(activation, "_read", lambda *_args, **_kwargs: b"{}")

    def write(**kwargs: Any) -> tuple[str, str]:
        captured.update(kwargs["value"])
        digest = hashlib.sha256(activation._canonical_json(kwargs["value"])).hexdigest()
        return str(kwargs["relative_path"]), digest

    monkeypatch.setattr(activation, "write_or_verify_canonical_local_json", write)
    relative, _digest, passed = activation.write_raster_semantic_text_smoke_quality_screen_v2(
        document_preview_root=root,
        source_gold_receipt_relative_path="source.json",
        candidate_execution_relative_path="candidate.json",
    )

    assert passed
    assert "/semantic-smoke-quality-screens-v2/" in relative
    assert captured["schema_version"] == "2.0"
    assert captured["metric_revision"] == RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2


def test_smoke_writer_separates_v3_metric_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    cases = [
        _case(
            asset_id=case.asset_id,
            document_type=case.document_type,
            profile_id=case.profile_id,
            reference="① VALUE",
            observed="VALUE",
        )
        for case in _passing_matrix_cases()
    ]
    source = SimpleNamespace(scope="smoke")
    execution = SimpleNamespace(
        scope="smoke",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        activation,
        "_load_source_gold_cases",
        lambda **_kwargs: (source, execution, SimpleNamespace(scope="smoke"), cases),
    )
    monkeypatch.setattr(activation, "_read", lambda *_args, **_kwargs: b"{}")

    def write(**kwargs: Any) -> tuple[str, str]:
        captured.update(kwargs["value"])
        digest = hashlib.sha256(activation._canonical_json(kwargs["value"])).hexdigest()
        return str(kwargs["relative_path"]), digest

    monkeypatch.setattr(activation, "write_or_verify_canonical_local_json", write)
    relative, _digest, passed = activation.write_raster_semantic_text_smoke_quality_screen_v3(
        document_preview_root=root,
        source_gold_receipt_relative_path="source.json",
        candidate_execution_relative_path="candidate.json",
    )

    assert passed
    assert "/semantic-smoke-quality-screens-v3/" in relative
    assert captured["schema_version"] == "3.0"
    assert captured["metric_revision"] == RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3


def test_calibration_writer_separates_v3_metric_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    cases = [
        _case(
            asset_id=case.asset_id,
            document_type=case.document_type,
            profile_id=case.profile_id,
            reference="① VALUE",
            observed="VALUE",
        )
        for case in _passing_matrix_cases()
    ]
    source = SimpleNamespace(scope="calibration")
    execution = SimpleNamespace(
        scope="calibration",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        latency_p95_seconds=1.0,
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        activation,
        "verify_source_attested_raster_text_evidence",
        lambda **_kwargs: (source, execution, SimpleNamespace(scope="calibration"), cases),
    )
    monkeypatch.setattr(activation, "_read", lambda *_args, **_kwargs: b"{}")

    def write(**kwargs: Any) -> tuple[str, str]:
        captured.update(kwargs["value"])
        digest = hashlib.sha256(activation._canonical_json(kwargs["value"])).hexdigest()
        return str(kwargs["relative_path"]), digest

    monkeypatch.setattr(activation, "write_or_verify_canonical_local_json", write)
    relative, _digest = activation.write_raster_semantic_text_calibration_receipt_v3(
        document_preview_root=root,
        source_gold_receipt_relative_path="source.json",
        candidate_execution_relative_path="candidate.json",
        activated_profile=SimpleNamespace(
            raster_text_profile_revision=_REVISION,
            extractor_revision="direct-text-test-v1",
            minimum_page_confidence=0.9,
        ),
        activated_runtime_configuration_sha256=_SHA,
    )

    assert "/semantic-calibration-receipts-v3/" in relative
    assert captured["schema_version"] == "3.0"
    assert captured["metric_revision"] == RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3


def test_evidence_writer_rejects_an_unknown_metric_revision_before_io(tmp_path: Path) -> None:
    with pytest.raises(activation.SyntheticRasterTextActivationError) as captured:
        activation.write_raster_text_smoke_quality_screen(
            document_preview_root=tmp_path,
            source_gold_receipt_relative_path="source.json",
            candidate_execution_relative_path="candidate.json",
            metric_revision="semantic-text-v3",  # type: ignore[arg-type]
        )
    assert captured.value.code == "raster_text_metric_revision_invalid"
