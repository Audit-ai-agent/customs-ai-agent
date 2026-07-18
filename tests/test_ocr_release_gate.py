from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

import evals.combine_ocr_release_gate as release_gate
from customs_ai_agent.ocr_release import (
    OcrReleaseGateAttestationError,
    load_ocr_release_gate_attestation,
)
from customs_ai_agent.ocr_reading_order import OCR_READING_ORDER_POLICY_ID
from evals.combine_ocr_release_gate import (
    OcrReleaseGateError,
    _combine_invalidated_v2_diagnostic,
    combine,
    main,
)
from evals.release_holdout_protocol import RELEASE_INVALIDATION_CODE
from evals.run_ocr_benchmark import (
    BENCHMARK_ID,
    GOLD_READING_ORDER_POLICY_ID,
    LOCKED_THRESHOLDS,
    BenchmarkDocument,
    gold_document_metric_denominators,
)

WHEEL_SHA256 = "c" * 64
ENVIRONMENT_LOCK_SHA256 = {
    "paddle": "1" * 64,
    "docling": "2" * 64,
}
MODEL_ASSETS_LOCK_SHA256 = "3" * 64
CANONICAL_DESCRIPTOR_SHA256 = "d" * 64
CANONICAL_MANIFEST_SHA256 = "e" * 64
CRITICAL_RULES = [{"field_id_marker": "Invoice No.", "selector": "identity"}]
REDACTION = {
    "source_text": "OMITTED",
    "image_payloads": "OMITTED",
    "filesystem_paths": "OMITTED",
    "secrets": "OMITTED",
    "backend_exception_messages": "OMITTED",
}


def _match(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "f1": 2 * true_positive / denominator if denominator else 1.0,
    }


def _layout() -> dict[str, Any]:
    identifier = "COMMERCIAL-INVOICE-IDENTIFIER-INV-001"
    return {
        "fields": [{"field_id": "invoice:/Invoice No.", "raw_value": identifier}],
        "words": [
            {
                "word_id": "word-1",
                "text": identifier,
                "role": "field_value",
                "bbox_claude_input_px": [10, 10, 180, 20],
            },
            {
                "word_id": "word-2",
                "text": "ITEM",
                "role": "table_header",
                "bbox_claude_input_px": [10, 40, 80, 50],
            },
            {
                "word_id": "word-3",
                "text": "VALUE",
                "role": "table_value",
                "bbox_claude_input_px": [10, 50, 80, 70],
            },
        ],
        "tables": [
            {
                "rows": [
                    {
                        "bbox_claude_input_px": [10, 40, 80, 50],
                        "cells": [
                            {
                                "column_label": "ITEM",
                                "field_id": None,
                                "bbox_claude_input_px": [10, 40, 80, 50],
                            }
                        ],
                    },
                    {
                        "bbox_claude_input_px": [10, 50, 80, 70],
                        "cells": [
                            {
                                "column_label": "ITEM",
                                "field_id": "invoice:/Invoice No.",
                                "bbox_claude_input_px": [10, 50, 80, 70],
                            }
                        ],
                    },
                ]
            }
        ],
    }


def _expected_documents(sample: str) -> list[BenchmarkDocument]:
    count = 60 if sample == "locked" else 20
    return [
        BenchmarkDocument(
            benchmark_document_id=f"{sample}-document-{index:03d}",
            image_sha256=hashlib.sha256(f"{sample}-image-{index}".encode()).hexdigest(),
            layout_sha256=hashlib.sha256(f"{sample}-layout-{index}".encode()).hexdigest(),
            image_path=Path(f"{sample}-{index}.png"),
            layout=_layout(),
        )
        for index in range(count)
    ]


def _canonical_loader(
    *,
    descriptor_path: Path,
    rendered_root: Path,
    sample: str,
) -> tuple[str, str, list[BenchmarkDocument], list[dict[str, str]]]:
    del descriptor_path, rendered_root
    return (
        CANONICAL_DESCRIPTOR_SHA256,
        CANONICAL_MANIFEST_SHA256,
        _expected_documents(sample),
        CRITICAL_RULES,
    )


@pytest.fixture(autouse=True)
def _canonical_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release_gate, "load_benchmark", _canonical_loader)


def _document_metrics(
    document: BenchmarkDocument,
    *,
    character_errors: int,
) -> dict[str, Any]:
    gold = gold_document_metric_denominators(
        layout=document.layout,
        critical_rules=CRITICAL_RULES,
    )
    return {
        "critical_identifier_correct": gold["critical_identifier_total"],
        "critical_identifier_total": gold["critical_identifier_total"],
        "character_edit_distance": character_errors,
        "gold_character_count": gold["gold_character_count"],
        "table_rows": _match(gold["table_row_gold_count"], 0, 0),
        "table_cells": _match(gold["table_cell_gold_count"], 0, 0),
        "bboxes_iou_0_5": _match(gold["bbox_gold_count"], 0, 0),
    }


def _aggregate(document_reports: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [document["metrics"] for document in document_reports]
    correct = sum(item["critical_identifier_correct"] for item in metrics)
    total = sum(item["critical_identifier_total"] for item in metrics)
    edit_distance = sum(item["character_edit_distance"] for item in metrics)
    gold_characters = sum(item["gold_character_count"] for item in metrics)

    def summed_match(name: str) -> dict[str, Any]:
        return _match(
            sum(item[name]["true_positive"] for item in metrics),
            sum(item[name]["false_positive"] for item in metrics),
            sum(item[name]["false_negative"] for item in metrics),
        )

    return {
        "critical_identifier_exact_accuracy": correct / total if total else 0.0,
        "critical_identifier_correct": correct,
        "critical_identifier_total": total,
        "character_error_rate": edit_distance / gold_characters if gold_characters else 1.0,
        "character_edit_distance": edit_distance,
        "gold_character_count": gold_characters,
        "table_rows": summed_match("table_rows"),
        "table_cells": summed_match("table_cells"),
        "bboxes_iou_0_5": summed_match("bboxes_iou_0_5"),
        "crash_count": 0,
        "successful_document_count": len(document_reports),
        "document_count": len(document_reports),
    }


def _provider(
    name: str,
    *,
    sample: str,
    passing: bool,
    character_errors: int = 5,
    wheel_sha256: str = WHEEL_SHA256,
    latency_ns: int | None = None,
    peak_memory_bytes: int | None = None,
    disk_footprint_bytes: int | None = None,
) -> dict[str, Any]:
    if latency_ns is None:
        latency_ns = 10_000_000 if name == "paddle" else 20_000_000
    if peak_memory_bytes is None:
        peak_memory_bytes = 100_000_000 if name == "paddle" else 200_000_000
    if disk_footprint_bytes is None:
        disk_footprint_bytes = 1_000_000_000 if name == "paddle" else 2_000_000_000
    documents = _expected_documents(sample)
    total_gold = sum(
        gold_document_metric_denominators(
            layout=document.layout,
            critical_rules=CRITICAL_RULES,
        )["gold_character_count"]
        for document in documents
    )
    if not passing:
        character_errors = max(character_errors, math.floor(total_gold * 0.01) + 1)
    document_reports = []
    for index, document in enumerate(documents):
        document_reports.append(
            {
                "benchmark_document_id": document.benchmark_document_id,
                "input_sha256": document.image_sha256,
                "layout_sha256": document.layout_sha256,
                "status": "SCORED",
                "latency_ns": latency_ns,
                "metrics": _document_metrics(
                    document,
                    character_errors=character_errors if index == 0 else 0,
                ),
            }
        )
    metrics = _aggregate(document_reports)
    checks = {
        "critical_identifier_exact": (
            metrics["critical_identifier_exact_accuracy"]
            >= LOCKED_THRESHOLDS["critical_identifier_exact_accuracy_min"]
        ),
        "character_error_rate": (
            metrics["character_error_rate"] <= LOCKED_THRESHOLDS["character_error_rate_max"]
        ),
        "table_row_f1": (metrics["table_rows"]["f1"] >= LOCKED_THRESHOLDS["table_row_f1_min"]),
        "table_cell_f1": (metrics["table_cells"]["f1"] >= LOCKED_THRESHOLDS["table_cell_f1_min"]),
        "bbox_f1_iou_0_5": (
            metrics["bboxes_iou_0_5"]["f1"] >= LOCKED_THRESHOLDS["bbox_f1_iou_0_5_min"]
        ),
        "crash_count": metrics["crash_count"] <= LOCKED_THRESHOLDS["crash_count_max"],
        "all_documents_scored": metrics["successful_document_count"] == len(documents),
    }
    return {
        "provider": name,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "engine": {
            "provider_id": name,
            "engine_id": f"{name}-engine",
            "engine_version": "1.0",
        },
        "metrics": metrics,
        "runtime_metrics": {
            "latency_sample_count": len(documents),
            "latency_p95_ms": latency_ns / 1_000_000,
            "peak_memory_bytes": peak_memory_bytes,
            "disk_footprint_bytes": disk_footprint_bytes,
        },
        "checks": checks,
        "failure_codes": sorted(check for check, passed in checks.items() if not passed),
        "documents": document_reports,
        "installed_agent_wheel_sha256": wheel_sha256,
        "environment_lock_sha256": ENVIRONMENT_LOCK_SHA256[name],
        "model_assets_lock_sha256": MODEL_ASSETS_LOCK_SHA256,
        "cleanup_error_code": None,
        "runtime_error_code": None,
    }


def _quality(provider: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = provider["metrics"]
    return (
        metrics["critical_identifier_exact_accuracy"],
        -metrics["character_error_rate"],
        min(metrics["table_rows"]["f1"], metrics["table_cells"]["f1"]),
        metrics["bboxes_iou_0_5"]["f1"],
    )


def _report(
    sample: str,
    *,
    paddle_pass: bool,
    docling_pass: bool,
    paddle_character_errors: int = 5,
    docling_character_errors: int = 6,
    descriptor_sha256: str = CANONICAL_DESCRIPTOR_SHA256,
    manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    wheel_sha256: str = WHEEL_SHA256,
    paddle_runtime: tuple[int, int, int] | None = None,
    docling_runtime: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    paddle_runtime = paddle_runtime or (10_000_000, 100_000_000, 1_000_000_000)
    docling_runtime = docling_runtime or (20_000_000, 200_000_000, 2_000_000_000)
    providers = [
        _provider(
            "paddle",
            sample=sample,
            passing=paddle_pass,
            character_errors=paddle_character_errors,
            wheel_sha256=wheel_sha256,
            latency_ns=paddle_runtime[0],
            peak_memory_bytes=paddle_runtime[1],
            disk_footprint_bytes=paddle_runtime[2],
        ),
        _provider(
            "docling",
            sample=sample,
            passing=docling_pass,
            character_errors=docling_character_errors,
            wheel_sha256=wheel_sha256,
            latency_ns=docling_runtime[0],
            peak_memory_bytes=docling_runtime[1],
            disk_footprint_bytes=docling_runtime[2],
        ),
    ]
    passing = [provider for provider in providers if provider["status"] == "PASS"]
    if not passing:
        status, selected_provider, selection_status = "FAIL", None, "NO_PROVIDER_PASSED"
    elif len(passing) == 1:
        status = "PASS"
        selected_provider = passing[0]["provider"]
        selection_status = "SELECTED_ONLY_PASSING_PROVIDER"
    elif _quality(providers[0]) == _quality(providers[1]):
        status, selected_provider = "FAIL", None
        selection_status = "QUALITY_TIE_NO_AUTOMATIC_SELECTION"
    else:
        status = "PASS"
        selected_provider = max(passing, key=_quality)["provider"]
        selection_status = "SELECTED_BY_LOCKED_QUALITY_METRICS"
    document_count = 60 if sample == "locked" else 20
    return {
        "schema_version": "ocr-benchmark-report/v2",
        "benchmark_id": BENCHMARK_ID,
        "sample": sample,
        "status": status,
        "selection_status": selection_status,
        "selected_provider": selected_provider,
        "descriptor_sha256": descriptor_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "thresholds": LOCKED_THRESHOLDS,
        "document_count": document_count,
        "coverage": {
            "selected_sample": sample,
            "selected_document_count": document_count,
            "locked_case_count": 20,
            "locked_triplet_document_count": 60,
            "co_document_count": 20,
            "release_gate_document_count": 80,
        },
        "reading_order_contract": {
            "gold_policy_id": GOLD_READING_ORDER_POLICY_ID,
            "prediction_policy_id": OCR_READING_ORDER_POLICY_ID,
        },
        "providers": providers,
        "redaction": REDACTION,
    }


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _combine(locked: Path, co_regression: Path) -> tuple[int, dict[str, Any]]:
    return _combine_invalidated_v2_diagnostic(
        locked_report_path=locked,
        co_regression_report_path=co_regression,
        descriptor_path=Path("canonical-descriptor.json"),
        rendered_root=Path("canonical-rendered"),
    )


def test_selects_only_provider_that_passes_locked_and_co_regression(tmp_path: Path) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report("locked", paddle_pass=True, docling_pass=True),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report("co_regression", paddle_pass=True, docling_pass=False),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 0
    assert report["status"] == "NONRELEASE_DIAGNOSTIC_PASS"
    assert report["release_eligible"] is False
    assert report["release_holdout_status"] == RELEASE_INVALIDATION_CODE
    assert report["selected_provider"] == "paddle"
    assert report["selection_status"] == "SELECTED_SINGLE_COMMON_PROVIDER"
    assert report["smoke_included"] is False
    assert report["release_document_count"] == 80
    assert report["descriptor_sha256"] == CANONICAL_DESCRIPTOR_SHA256
    assert report["dataset_manifest_sha256"] == CANONICAL_MANIFEST_SHA256
    assert report["environment_lock_sha256"] == ENVIRONMENT_LOCK_SHA256["paddle"]
    assert report["model_assets_lock_sha256"] == MODEL_ASSETS_LOCK_SHA256


def test_different_sample_winners_do_not_form_a_release_provider(tmp_path: Path) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report("locked", paddle_pass=True, docling_pass=False),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report("co_regression", paddle_pass=False, docling_pass=True),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 1
    assert report["selected_provider"] is None
    assert report["failure_codes"] == ["no_provider_passed_both_release_samples"]


def test_common_providers_are_ordered_by_locked_quality_only(tmp_path: Path) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report(
            "locked",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=9,
            docling_character_errors=2,
        ),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=1,
            docling_character_errors=8,
        ),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 0
    assert report["selected_provider"] == "docling"
    assert report["selection_status"] == "SELECTED_BY_LOCKED_QUALITY_METRICS"


def test_exact_locked_quality_tie_uses_attested_runtime_order(tmp_path: Path) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report(
            "locked",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
        ),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
        ),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 0
    assert report["selected_provider"] == "paddle"
    assert report["selection_status"] == "SELECTED_BY_ATTESTED_RUNTIME_METRICS"
    assert report["failure_codes"] == []
    assert report["runtime_tiebreak"]["selection_order"] == [
        "latency_p95_ms",
        "peak_memory_bytes",
        "disk_footprint_bytes",
    ]

    release_parser_fixture = dict(report)
    release_parser_fixture["schema_version"] = release_gate.REPORT_SCHEMA
    release_parser_fixture["status"] = "PASS"
    del release_parser_fixture["release_eligible"]
    del release_parser_fixture["release_holdout_status"]
    combined_path = _write(tmp_path / "combined.json", release_parser_fixture)
    combined_sha256 = hashlib.sha256(combined_path.read_bytes()).hexdigest()
    attestation = load_ocr_release_gate_attestation(
        path=combined_path,
        expected_report_sha256=combined_sha256,
        expected_provider="paddle",
        expected_descriptor_sha256=CANONICAL_DESCRIPTOR_SHA256,
        expected_manifest_sha256=CANONICAL_MANIFEST_SHA256,
        expected_agent_wheel_sha256=WHEEL_SHA256,
        expected_environment_lock_sha256=ENVIRONMENT_LOCK_SHA256["paddle"],
        expected_model_assets_lock_sha256=MODEL_ASSETS_LOCK_SHA256,
    )
    assert attestation.provider == "paddle"

    release_parser_fixture["runtime_tiebreak"]["providers"][0]["latency_p95_ms"] = 30.0
    combined_path = _write(
        tmp_path / "combined-tampered.json",
        release_parser_fixture,
    )
    combined_sha256 = hashlib.sha256(combined_path.read_bytes()).hexdigest()
    with pytest.raises(OcrReleaseGateAttestationError) as captured:
        load_ocr_release_gate_attestation(
            path=combined_path,
            expected_report_sha256=combined_sha256,
            expected_provider="paddle",
            expected_descriptor_sha256=CANONICAL_DESCRIPTOR_SHA256,
            expected_manifest_sha256=CANONICAL_MANIFEST_SHA256,
            expected_agent_wheel_sha256=WHEEL_SHA256,
            expected_environment_lock_sha256=ENVIRONMENT_LOCK_SHA256["paddle"],
            expected_model_assets_lock_sha256=MODEL_ASSETS_LOCK_SHA256,
        )
    assert captured.value.code == "ocr_release_gate_runtime_tiebreak_inconsistent"


def test_exact_quality_and_runtime_tie_remains_fail_closed(tmp_path: Path) -> None:
    shared_runtime = (10_000_000, 100_000_000, 1_000_000_000)
    locked = _write(
        tmp_path / "locked.json",
        _report(
            "locked",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
            paddle_runtime=shared_runtime,
            docling_runtime=shared_runtime,
        ),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
            paddle_runtime=shared_runtime,
            docling_runtime=shared_runtime,
        ),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 1
    assert report["selected_provider"] is None
    assert report["selection_status"] == "RUNTIME_METRICS_TIE_NO_AUTOMATIC_SELECTION"
    assert report["failure_codes"] == ["runtime_metrics_tie_no_automatic_selection"]


def test_quality_tie_with_missing_runtime_metrics_fails_closed(tmp_path: Path) -> None:
    locked_value = _report(
        "locked",
        paddle_pass=True,
        docling_pass=True,
        paddle_character_errors=5,
        docling_character_errors=5,
    )
    locked_value["providers"][0]["runtime_metrics"] = None
    locked_value["providers"][0]["runtime_error_code"] = "benchmark_runtime_memory_unavailable"
    locked = _write(tmp_path / "locked.json", locked_value)
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
        ),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 1
    assert report["selected_provider"] is None
    assert report["selection_status"] == "QUALITY_TIE_RUNTIME_METRICS_UNAVAILABLE"
    assert report["failure_codes"] == ["quality_tie_runtime_metrics_unavailable"]


@pytest.mark.parametrize(
    ("paddle_runtime", "docling_runtime", "expected_provider"),
    [
        (
            (10_000_000, 100_000_000, 9_000_000_000),
            (10_000_000, 200_000_000, 1_000_000_000),
            "paddle",
        ),
        (
            (10_000_000, 100_000_000, 2_000_000_000),
            (10_000_000, 100_000_000, 1_000_000_000),
            "docling",
        ),
    ],
)
def test_runtime_tiebreak_is_lexicographic(
    tmp_path: Path,
    paddle_runtime: tuple[int, int, int],
    docling_runtime: tuple[int, int, int],
    expected_provider: str,
) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report(
            "locked",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
            paddle_runtime=paddle_runtime,
            docling_runtime=docling_runtime,
        ),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
            paddle_runtime=paddle_runtime,
            docling_runtime=docling_runtime,
        ),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 0
    assert report["selected_provider"] == expected_provider
    assert report["selection_status"] == "SELECTED_BY_ATTESTED_RUNTIME_METRICS"


def test_runtime_metric_tamper_and_identity_drift_fail_closed(tmp_path: Path) -> None:
    locked_value = _report(
        "locked",
        paddle_pass=True,
        docling_pass=True,
        paddle_character_errors=5,
        docling_character_errors=5,
    )
    locked_value["providers"][0]["runtime_metrics"]["latency_p95_ms"] = -1
    locked = _write(tmp_path / "locked-invalid.json", locked_value)
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
        ),
    )
    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)
    assert captured.value.code == "ocr_release_runtime_metrics_invalid"

    locked = _write(
        tmp_path / "locked.json",
        _report(
            "locked",
            paddle_pass=True,
            docling_pass=True,
            paddle_character_errors=5,
            docling_character_errors=5,
        ),
    )
    co_value = _report(
        "co_regression",
        paddle_pass=True,
        docling_pass=True,
        paddle_character_errors=5,
        docling_character_errors=5,
    )
    co_value["providers"][0]["runtime_metrics"]["disk_footprint_bytes"] += 1
    co_regression = _write(tmp_path / "co-drift.json", co_value)
    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)
    assert captured.value.code == "ocr_release_runtime_identity_mismatch"


def test_arbitrary_matching_descriptor_and_manifest_hashes_cannot_pass(
    tmp_path: Path,
) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report(
            "locked",
            paddle_pass=True,
            docling_pass=False,
            descriptor_sha256="a" * 64,
            manifest_sha256="b" * 64,
        ),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=False,
            descriptor_sha256="a" * 64,
            manifest_sha256="b" * 64,
        ),
    )

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_report_identity_invalid"


def test_per_document_identity_is_bound_to_canonical_descriptor(tmp_path: Path) -> None:
    locked_value = _report("locked", paddle_pass=True, docling_pass=False)
    locked_value["providers"][0]["documents"][0]["input_sha256"] = "f" * 64
    locked = _write(tmp_path / "locked.json", locked_value)
    co_regression = _write(
        tmp_path / "co.json",
        _report("co_regression", paddle_pass=True, docling_pass=False),
    )

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_provider_document_identity_invalid"


def test_per_document_gold_counts_and_aggregate_are_recomputed(tmp_path: Path) -> None:
    locked_value = _report("locked", paddle_pass=True, docling_pass=False)
    locked_value["providers"][0]["documents"][0]["metrics"]["gold_character_count"] += 1
    locked = _write(tmp_path / "locked.json", locked_value)
    co_regression = _write(
        tmp_path / "co.json",
        _report("co_regression", paddle_pass=True, docling_pass=False),
    )

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_document_gold_counts_invalid"

    locked_value = _report("locked", paddle_pass=True, docling_pass=False)
    locked_value["providers"][0]["metrics"]["character_edit_distance"] += 1
    locked = _write(tmp_path / "locked-aggregate.json", locked_value)
    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)
    assert captured.value.code in {
        "ocr_release_provider_metrics_invalid",
        "ocr_release_provider_aggregate_mismatch",
    }


def test_all_provider_records_must_attest_one_agent_wheel(tmp_path: Path) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report("locked", paddle_pass=True, docling_pass=False),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=False,
            wheel_sha256="f" * 64,
        ),
    )

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_wheel_identity_mismatch"


def test_provider_environment_lock_must_match_across_release_samples(
    tmp_path: Path,
) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report("locked", paddle_pass=True, docling_pass=False),
    )
    co_value = _report("co_regression", paddle_pass=True, docling_pass=False)
    co_value["providers"][0]["environment_lock_sha256"] = "f" * 64
    co_regression = _write(tmp_path / "co.json", co_value)

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_environment_lock_identity_mismatch"


def test_model_assets_lock_must_match_every_provider_and_release_sample(
    tmp_path: Path,
) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report("locked", paddle_pass=True, docling_pass=False),
    )
    co_value = _report("co_regression", paddle_pass=True, docling_pass=False)
    co_value["providers"][1]["model_assets_lock_sha256"] = "f" * 64
    co_regression = _write(tmp_path / "co.json", co_value)

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_model_assets_lock_identity_mismatch"


def test_common_provider_must_keep_one_engine_identity(tmp_path: Path) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report("locked", paddle_pass=True, docling_pass=False),
    )
    co_value = _report("co_regression", paddle_pass=True, docling_pass=False)
    co_value["providers"][0]["engine"]["engine_version"] = "2.0"
    co_regression = _write(tmp_path / "co.json", co_value)

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_engine_identity_mismatch"


def test_provider_pass_cannot_disagree_with_threshold_checks(tmp_path: Path) -> None:
    locked_value = _report("locked", paddle_pass=True, docling_pass=False)
    locked_value["providers"][0]["checks"]["character_error_rate"] = False
    locked_value["providers"][0]["failure_codes"] = ["character_error_rate"]
    locked = _write(tmp_path / "locked.json", locked_value)
    co_regression = _write(
        tmp_path / "co.json",
        _report("co_regression", paddle_pass=True, docling_pass=False),
    )

    with pytest.raises(OcrReleaseGateError) as captured:
        _combine(locked, co_regression)

    assert captured.value.code == "ocr_release_provider_invalid"


def test_failed_provider_may_have_character_error_rate_above_one(tmp_path: Path) -> None:
    locked = _write(
        tmp_path / "locked.json",
        _report(
            "locked",
            paddle_pass=True,
            docling_pass=False,
            docling_character_errors=100_000,
        ),
    )
    co_regression = _write(
        tmp_path / "co.json",
        _report(
            "co_regression",
            paddle_pass=True,
            docling_pass=False,
            docling_character_errors=100_000,
        ),
    )

    exit_code, report = _combine(locked, co_regression)

    assert exit_code == 0
    assert report["selected_provider"] == "paddle"


def test_public_combiner_blocks_invalidated_v2_before_any_input_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-path-token"
    loader_called = False

    def fail_loader(**_kwargs: Any) -> Any:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("invalidated v2 input must not be loaded")

    monkeypatch.setattr(release_gate, "load_benchmark", fail_loader)
    with pytest.raises(OcrReleaseGateError) as captured:
        combine(
            locked_report_path=tmp_path / secret / "locked.json",
            co_regression_report_path=tmp_path / "co.json",
            descriptor_path=tmp_path / secret / "descriptor.json",
            rendered_root=tmp_path / secret / "rendered",
        )

    assert captured.value.code == RELEASE_INVALIDATION_CODE
    assert loader_called is False
    assert secret not in str(captured.value)
    assert str(tmp_path) not in str(captured.value)


def test_cli_preflight_failure_emits_only_safe_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "sensitive-path-token"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "combine_ocr_release_gate.py",
            "--locked-report",
            str(tmp_path / secret / "missing-locked.json"),
            "--co-regression-report",
            str(tmp_path / "missing-co.json"),
            "--descriptor",
            str(tmp_path / "descriptor.json"),
            "--rendered-root",
            str(tmp_path / "rendered"),
        ],
    )

    assert main() == 2
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["selection_status"] == "PREFLIGHT_FAILED"
    assert report["error_code"] == RELEASE_INVALIDATION_CODE
    assert secret not in output
    assert str(tmp_path) not in output
