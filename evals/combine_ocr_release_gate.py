"""Validate legacy OCR reports without creating release evidence.

Evaluation-v2 is permanently release-ineligible because its membership is
publicly derivable.  The public ``combine`` entry point therefore stops before
reading any descriptor, rendered document, or provider report.  The legacy
validation algorithm remains private and can emit only an explicitly
non-release diagnostic for regression tests and historical audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from customs_ai_agent.ocr_release import (
    OCR_RELEASE_DOCUMENT_COUNT,
    OCR_RELEASE_REDACTION,
    OCR_RELEASE_REPORT_SCHEMA,
    OCR_RELEASE_SAMPLE_DOCUMENT_COUNTS,
)
from customs_ai_agent.ocr_reading_order import OCR_READING_ORDER_POLICY_ID
from evals.run_ocr_benchmark import (
    BENCHMARK_ID,
    BenchmarkDocument,
    GOLD_READING_ORDER_POLICY_ID,
    LOCKED_THRESHOLDS,
    OcrBenchmarkError,
    PROVIDERS,
    REPORT_SCHEMA as BENCHMARK_REPORT_SCHEMA,
    gold_document_metric_denominators,
    load_benchmark,
)
from evals.release_holdout_protocol import (
    RELEASE_INVALIDATION_CODE,
)

REPORT_SCHEMA = OCR_RELEASE_REPORT_SCHEMA
DIAGNOSTIC_REPORT_SCHEMA = "ocr-invalidated-v2-combination-diagnostic/v1"
SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
MAX_REPORT_BYTES = 4 * 1024 * 1024
RELEASE_DOCUMENT_COUNT = OCR_RELEASE_DOCUMENT_COUNT
SAMPLE_DOCUMENT_COUNTS = OCR_RELEASE_SAMPLE_DOCUMENT_COUNTS
CHECK_NAMES = frozenset(
    {
        "critical_identifier_exact",
        "character_error_rate",
        "table_row_f1",
        "table_cell_f1",
        "bbox_f1_iou_0_5",
        "crash_count",
        "all_documents_scored",
    }
)
REPORT_KEYS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "sample",
        "status",
        "selection_status",
        "selected_provider",
        "descriptor_sha256",
        "dataset_manifest_sha256",
        "thresholds",
        "document_count",
        "coverage",
        "reading_order_contract",
        "providers",
        "redaction",
    }
)
PROVIDER_KEYS = frozenset(
    {
        "provider",
        "status",
        "engine",
        "metrics",
        "runtime_metrics",
        "checks",
        "failure_codes",
        "documents",
        "installed_agent_wheel_sha256",
        "environment_lock_sha256",
        "model_assets_lock_sha256",
        "cleanup_error_code",
        "runtime_error_code",
    }
)
METRIC_KEYS = frozenset(
    {
        "critical_identifier_exact_accuracy",
        "critical_identifier_correct",
        "critical_identifier_total",
        "character_error_rate",
        "character_edit_distance",
        "gold_character_count",
        "table_rows",
        "table_cells",
        "bboxes_iou_0_5",
        "crash_count",
        "successful_document_count",
        "document_count",
    }
)
MATCH_KEYS = frozenset({"true_positive", "false_positive", "false_negative", "f1"})
DOCUMENT_METRIC_KEYS = frozenset(
    {
        "critical_identifier_correct",
        "critical_identifier_total",
        "character_edit_distance",
        "gold_character_count",
        "table_rows",
        "table_cells",
        "bboxes_iou_0_5",
    }
)
DOCUMENT_BASE_KEYS = frozenset(
    {
        "benchmark_document_id",
        "input_sha256",
        "layout_sha256",
        "status",
        "latency_ns",
        "metrics",
    }
)
RUNTIME_METRIC_KEYS = frozenset(
    {
        "latency_sample_count",
        "latency_p95_ms",
        "peak_memory_bytes",
        "disk_footprint_bytes",
    }
)
REDACTION = OCR_RELEASE_REDACTION

ProviderName = Literal["paddle", "docling"]
SampleName = Literal["locked", "co_regression"]


class OcrReleaseGateError(RuntimeError):
    """Safe preflight failure represented only by a stable code."""

    def __init__(self, code: str) -> None:
        if SAFE_CODE.fullmatch(code) is None:
            code = "ocr_release_gate_internal_error"
        super().__init__(code)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OcrReleaseGateError("ocr_release_report_unreadable") from exc
    return digest.hexdigest()


def _load_report(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
    except OSError as exc:
        raise OcrReleaseGateError("ocr_release_report_unreadable") from exc
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or path.is_symlink()
        or size < 2
        or size > MAX_REPORT_BYTES
    ):
        raise OcrReleaseGateError("ocr_release_report_invalid")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrReleaseGateError("ocr_release_report_invalid") from exc
    if not isinstance(value, dict):
        raise OcrReleaseGateError("ocr_release_report_invalid")
    return _sha256(resolved), cast(dict[str, Any], value)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_ratio(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _is_non_negative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validated_match_metrics(value: object) -> dict[str, int | float]:
    if not isinstance(value, dict) or set(value) != MATCH_KEYS:
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    if not all(_is_count(value.get(key)) for key in MATCH_KEYS - {"f1"}):
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    if not _is_ratio(value.get("f1")):
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    true_positive = cast(int, value["true_positive"])
    false_positive = cast(int, value["false_positive"])
    false_negative = cast(int, value["false_negative"])
    denominator = 2 * true_positive + false_positive + false_negative
    expected_f1 = 2 * true_positive / denominator if denominator else 1.0
    if not math.isclose(float(value["f1"]), expected_f1, rel_tol=0.0, abs_tol=1e-12):
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    return cast(dict[str, int | float], value)


def _validated_document_metrics(
    value: object,
    *,
    expected_document: BenchmarkDocument,
    critical_rules: Sequence[Mapping[str, str]],
    failed: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != DOCUMENT_METRIC_KEYS:
        raise OcrReleaseGateError("ocr_release_document_metrics_invalid")
    if not all(
        _is_count(value.get(key))
        for key in (
            "critical_identifier_correct",
            "critical_identifier_total",
            "character_edit_distance",
            "gold_character_count",
        )
    ):
        raise OcrReleaseGateError("ocr_release_document_metrics_invalid")
    correct = cast(int, value["critical_identifier_correct"])
    total = cast(int, value["critical_identifier_total"])
    if correct > total:
        raise OcrReleaseGateError("ocr_release_document_metrics_invalid")
    rows = _validated_match_metrics(value.get("table_rows"))
    cells = _validated_match_metrics(value.get("table_cells"))
    bboxes = _validated_match_metrics(value.get("bboxes_iou_0_5"))
    gold = gold_document_metric_denominators(
        layout=expected_document.layout,
        critical_rules=critical_rules,
    )
    if (
        total != gold["critical_identifier_total"]
        or value["gold_character_count"] != gold["gold_character_count"]
        or rows["true_positive"] + rows["false_negative"] != gold["table_row_gold_count"]
        or cells["true_positive"] + cells["false_negative"] != gold["table_cell_gold_count"]
        or bboxes["true_positive"] + bboxes["false_negative"] != gold["bbox_gold_count"]
    ):
        raise OcrReleaseGateError("ocr_release_document_gold_counts_invalid")
    if failed and (
        correct != 0
        or value["character_edit_distance"] != gold["gold_character_count"]
        or any(
            metrics["true_positive"] != 0
            or metrics["false_positive"] != 0
            or metrics["false_negative"] != expected_gold
            for metrics, expected_gold in (
                (rows, gold["table_row_gold_count"]),
                (cells, gold["table_cell_gold_count"]),
                (bboxes, gold["bbox_gold_count"]),
            )
        )
    ):
        raise OcrReleaseGateError("ocr_release_failed_document_metrics_invalid")
    return {
        **{
            key: cast(int, value[key])
            for key in (
                "critical_identifier_correct",
                "critical_identifier_total",
                "character_edit_distance",
                "gold_character_count",
            )
        },
        "table_rows": rows,
        "table_cells": cells,
        "bboxes_iou_0_5": bboxes,
    }


def _validated_documents(
    value: object,
    *,
    expected_documents: Sequence[BenchmarkDocument],
    critical_rules: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], int, int]:
    if not isinstance(value, list) or len(value) != len(expected_documents):
        raise OcrReleaseGateError("ocr_release_provider_documents_invalid")
    validated: list[dict[str, Any]] = []
    successful_count = 0
    failed_count = 0
    for raw, expected in zip(value, expected_documents, strict=True):
        if not isinstance(raw, dict) or raw.get("status") not in {"SCORED", "FAIL"}:
            raise OcrReleaseGateError("ocr_release_provider_document_invalid")
        failed = raw["status"] == "FAIL"
        expected_keys = DOCUMENT_BASE_KEYS | ({"error_code"} if failed else set())
        if (
            set(raw) != expected_keys
            or raw.get("benchmark_document_id") != expected.benchmark_document_id
            or raw.get("input_sha256") != expected.image_sha256
            or raw.get("layout_sha256") != expected.layout_sha256
            or not _is_count(raw.get("latency_ns"))
            or (not failed and raw.get("latency_ns") == 0)
        ):
            raise OcrReleaseGateError("ocr_release_provider_document_identity_invalid")
        if failed:
            error_code = raw.get("error_code")
            if not isinstance(error_code, str) or SAFE_CODE.fullmatch(error_code) is None:
                raise OcrReleaseGateError("ocr_release_provider_document_invalid")
            failed_count += 1
        else:
            successful_count += 1
        validated.append(
            {
                "status": raw["status"],
                "latency_ns": raw["latency_ns"],
                "metrics": _validated_document_metrics(
                    raw.get("metrics"),
                    expected_document=expected,
                    critical_rules=critical_rules,
                    failed=failed,
                ),
            }
        )
    return validated, successful_count, failed_count


def _recomputed_aggregate_metrics(
    documents: Sequence[Mapping[str, Any]],
    *,
    successful_document_count: int,
    crash_count: int,
) -> dict[str, Any]:
    critical_correct = sum(
        cast(int, cast(Mapping[str, Any], document["metrics"])["critical_identifier_correct"])
        for document in documents
    )
    critical_total = sum(
        cast(int, cast(Mapping[str, Any], document["metrics"])["critical_identifier_total"])
        for document in documents
    )
    edit_distance = sum(
        cast(int, cast(Mapping[str, Any], document["metrics"])["character_edit_distance"])
        for document in documents
    )
    gold_characters = sum(
        cast(int, cast(Mapping[str, Any], document["metrics"])["gold_character_count"])
        for document in documents
    )

    def summed_match(name: str) -> dict[str, int | float]:
        true_positive = sum(
            cast(
                int,
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], document["metrics"])[name],
                )["true_positive"],
            )
            for document in documents
        )
        false_positive = sum(
            cast(
                int,
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], document["metrics"])[name],
                )["false_positive"],
            )
            for document in documents
        )
        false_negative = sum(
            cast(
                int,
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], document["metrics"])[name],
                )["false_negative"],
            )
            for document in documents
        )
        denominator = 2 * true_positive + false_positive + false_negative
        return {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "f1": 2 * true_positive / denominator if denominator else 1.0,
        }

    return {
        "critical_identifier_exact_accuracy": (
            critical_correct / critical_total if critical_total else 0.0
        ),
        "critical_identifier_correct": critical_correct,
        "critical_identifier_total": critical_total,
        "character_error_rate": edit_distance / gold_characters if gold_characters else 1.0,
        "character_edit_distance": edit_distance,
        "gold_character_count": gold_characters,
        "table_rows": summed_match("table_rows"),
        "table_cells": summed_match("table_cells"),
        "bboxes_iou_0_5": summed_match("bboxes_iou_0_5"),
        "crash_count": crash_count,
        "successful_document_count": successful_document_count,
        "document_count": len(documents),
    }


def _assert_aggregate_metrics_match(
    reported: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    for key in (
        "critical_identifier_correct",
        "critical_identifier_total",
        "character_edit_distance",
        "gold_character_count",
        "crash_count",
        "successful_document_count",
        "document_count",
    ):
        if reported[key] != recomputed[key]:
            raise OcrReleaseGateError("ocr_release_provider_aggregate_mismatch")
    for key in ("critical_identifier_exact_accuracy", "character_error_rate"):
        if not math.isclose(
            float(reported[key]),
            float(recomputed[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise OcrReleaseGateError("ocr_release_provider_aggregate_mismatch")
    for key in ("table_rows", "table_cells", "bboxes_iou_0_5"):
        reported_match = cast(Mapping[str, Any], reported[key])
        recomputed_match = cast(Mapping[str, Any], recomputed[key])
        if any(
            reported_match[count_key] != recomputed_match[count_key]
            for count_key in ("true_positive", "false_positive", "false_negative")
        ) or not math.isclose(
            float(reported_match["f1"]),
            float(recomputed_match["f1"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise OcrReleaseGateError("ocr_release_provider_aggregate_mismatch")


def _validated_quality(
    provider: Mapping[str, Any],
    *,
    expected_document_count: int,
) -> tuple[float, float, float, float]:
    metrics = provider.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    if (
        not _is_ratio(metrics.get("critical_identifier_exact_accuracy"))
        or not _is_non_negative_number(metrics.get("character_error_rate"))
        or not all(
            _is_count(metrics.get(key))
            for key in (
                "critical_identifier_correct",
                "critical_identifier_total",
                "character_edit_distance",
                "gold_character_count",
                "crash_count",
                "successful_document_count",
                "document_count",
            )
        )
        or metrics["document_count"] != expected_document_count
        or metrics["successful_document_count"] > expected_document_count
        or metrics["critical_identifier_correct"] > metrics["critical_identifier_total"]
    ):
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    total = cast(int, metrics["critical_identifier_total"])
    expected_accuracy = cast(int, metrics["critical_identifier_correct"]) / total if total else 0.0
    if not math.isclose(
        float(metrics["critical_identifier_exact_accuracy"]),
        expected_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    gold_characters = cast(int, metrics["gold_character_count"])
    expected_cer = (
        cast(int, metrics["character_edit_distance"]) / gold_characters if gold_characters else 1.0
    )
    if not math.isclose(
        float(metrics["character_error_rate"]),
        expected_cer,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise OcrReleaseGateError("ocr_release_provider_metrics_invalid")
    rows = _validated_match_metrics(metrics.get("table_rows"))
    cells = _validated_match_metrics(metrics.get("table_cells"))
    bboxes = _validated_match_metrics(metrics.get("bboxes_iou_0_5"))
    return (
        float(metrics["critical_identifier_exact_accuracy"]),
        -float(metrics["character_error_rate"]),
        min(float(rows["f1"]), float(cells["f1"])),
        float(bboxes["f1"]),
    )


def _validated_runtime_metrics(
    value: object,
    *,
    documents: Sequence[Mapping[str, Any]],
) -> tuple[float, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != RUNTIME_METRIC_KEYS:
        raise OcrReleaseGateError("ocr_release_runtime_metrics_invalid")
    latency_sample_count = value.get("latency_sample_count")
    latency_p95_ms = value.get("latency_p95_ms")
    peak_memory_bytes = value.get("peak_memory_bytes")
    disk_footprint_bytes = value.get("disk_footprint_bytes")
    if (
        latency_sample_count != len(documents)
        or not _is_non_negative_number(latency_p95_ms)
        or float(cast(float, latency_p95_ms)) <= 0.0
        or not _is_count(peak_memory_bytes)
        or cast(int, peak_memory_bytes) < 1
        or not _is_count(disk_footprint_bytes)
        or cast(int, disk_footprint_bytes) < 1
    ):
        raise OcrReleaseGateError("ocr_release_runtime_metrics_invalid")
    raw_latencies = [document.get("latency_ns") for document in documents]
    if any(
        not isinstance(latency, int) or isinstance(latency, bool) or latency <= 0
        for latency in raw_latencies
    ):
        raise OcrReleaseGateError("ocr_release_runtime_metrics_invalid")
    latencies = sorted(cast(list[int], raw_latencies))
    expected_p95_ms = latencies[math.ceil(0.95 * len(latencies)) - 1] / 1_000_000
    if not math.isclose(
        float(cast(float, latency_p95_ms)),
        expected_p95_ms,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise OcrReleaseGateError("ocr_release_runtime_metrics_invalid")
    return (
        expected_p95_ms,
        cast(int, peak_memory_bytes),
        cast(int, disk_footprint_bytes),
    )


def _validate_provider(
    value: object,
    *,
    provider_name: ProviderName,
    expected_documents: Sequence[BenchmarkDocument],
    critical_rules: Sequence[Mapping[str, str]],
) -> tuple[
    dict[str, Any],
    tuple[float, float, float, float],
    tuple[float, int, int] | None,
]:
    expected_document_count = len(expected_documents)
    if not isinstance(value, dict) or set(value) != PROVIDER_KEYS:
        raise OcrReleaseGateError("ocr_release_provider_invalid")
    if value.get("provider") != provider_name or value.get("status") not in {"PASS", "FAIL"}:
        raise OcrReleaseGateError("ocr_release_provider_invalid")
    wheel_sha256 = value.get("installed_agent_wheel_sha256")
    environment_lock_sha256 = value.get("environment_lock_sha256")
    model_assets_lock_sha256 = value.get("model_assets_lock_sha256")
    checks = value.get("checks")
    failure_codes = value.get("failure_codes")
    documents = value.get("documents")
    cleanup_error_code = value.get("cleanup_error_code")
    runtime_error_code = value.get("runtime_error_code")
    if (
        not isinstance(wheel_sha256, str)
        or SHA256.fullmatch(wheel_sha256) is None
        or not isinstance(environment_lock_sha256, str)
        or SHA256.fullmatch(environment_lock_sha256) is None
        or not isinstance(model_assets_lock_sha256, str)
        or SHA256.fullmatch(model_assets_lock_sha256) is None
        or not isinstance(checks, dict)
        or set(checks) != CHECK_NAMES
        or not all(isinstance(item, bool) for item in checks.values())
        or not isinstance(failure_codes, list)
        or failure_codes != sorted(name for name, passed in checks.items() if not passed)
        or (
            cleanup_error_code is not None
            and (
                not isinstance(cleanup_error_code, str)
                or SAFE_CODE.fullmatch(cleanup_error_code) is None
            )
        )
        or (
            runtime_error_code is not None
            and (
                not isinstance(runtime_error_code, str)
                or SAFE_CODE.fullmatch(runtime_error_code) is None
            )
        )
    ):
        raise OcrReleaseGateError("ocr_release_provider_invalid")
    passed = all(checks.values())
    if value["status"] != ("PASS" if passed else "FAIL"):
        raise OcrReleaseGateError("ocr_release_provider_invalid")
    engine = value.get("engine")
    if passed:
        if (
            not isinstance(engine, dict)
            or set(engine) != {"provider_id", "engine_id", "engine_version"}
            or engine.get("provider_id") != provider_name
            or not all(
                isinstance(engine.get(key), str) and engine[key]
                for key in ("engine_id", "engine_version")
            )
        ):
            raise OcrReleaseGateError("ocr_release_provider_invalid")
    elif engine is not None and (
        not isinstance(engine, dict)
        or set(engine) != {"provider_id", "engine_id", "engine_version"}
        or engine.get("provider_id") != provider_name
    ):
        raise OcrReleaseGateError("ocr_release_provider_invalid")
    validated_documents, successful_document_count, failed_document_count = _validated_documents(
        documents,
        expected_documents=expected_documents,
        critical_rules=critical_rules,
    )
    runtime_metrics = _validated_runtime_metrics(
        value.get("runtime_metrics"),
        documents=validated_documents,
    )
    if runtime_metrics is not None and runtime_error_code is not None:
        raise OcrReleaseGateError("ocr_release_runtime_metrics_invalid")
    quality = _validated_quality(value, expected_document_count=expected_document_count)
    metrics = cast(Mapping[str, Any], value["metrics"])
    recomputed_metrics = _recomputed_aggregate_metrics(
        validated_documents,
        successful_document_count=successful_document_count,
        crash_count=failed_document_count + int(cleanup_error_code is not None),
    )
    _assert_aggregate_metrics_match(metrics, recomputed_metrics)
    if (
        checks["critical_identifier_exact"]
        != (
            float(metrics["critical_identifier_exact_accuracy"])
            >= LOCKED_THRESHOLDS["critical_identifier_exact_accuracy_min"]
        )
        or checks["character_error_rate"]
        != (float(metrics["character_error_rate"]) <= LOCKED_THRESHOLDS["character_error_rate_max"])
        or checks["table_row_f1"]
        != (
            float(cast(Mapping[str, Any], metrics["table_rows"])["f1"])
            >= LOCKED_THRESHOLDS["table_row_f1_min"]
        )
        or checks["table_cell_f1"]
        != (
            float(cast(Mapping[str, Any], metrics["table_cells"])["f1"])
            >= LOCKED_THRESHOLDS["table_cell_f1_min"]
        )
        or checks["bbox_f1_iou_0_5"]
        != (
            float(cast(Mapping[str, Any], metrics["bboxes_iou_0_5"])["f1"])
            >= LOCKED_THRESHOLDS["bbox_f1_iou_0_5_min"]
        )
        or checks["crash_count"] != (metrics["crash_count"] <= LOCKED_THRESHOLDS["crash_count_max"])
        or checks["all_documents_scored"]
        != (metrics["successful_document_count"] == expected_document_count)
    ):
        raise OcrReleaseGateError("ocr_release_provider_checks_invalid")
    return cast(dict[str, Any], value), quality, runtime_metrics


def _validate_top_selection(
    report: Mapping[str, Any],
    *,
    providers: Mapping[ProviderName, Mapping[str, Any]],
    qualities: Mapping[ProviderName, tuple[float, float, float, float]],
) -> None:
    passing = [provider for provider in PROVIDERS if providers[provider]["status"] == "PASS"]
    status = report.get("status")
    selected = report.get("selected_provider")
    selection_status = report.get("selection_status")
    expected: tuple[str, ProviderName | None, str]
    if not passing:
        expected = ("FAIL", None, "NO_PROVIDER_PASSED")
    elif len(passing) == 1:
        expected = ("PASS", passing[0], "SELECTED_ONLY_PASSING_PROVIDER")
    elif qualities["paddle"] == qualities["docling"]:
        expected = ("FAIL", None, "QUALITY_TIE_NO_AUTOMATIC_SELECTION")
    else:
        expected_provider = max(passing, key=lambda provider: qualities[provider])
        expected = ("PASS", expected_provider, "SELECTED_BY_LOCKED_QUALITY_METRICS")
    if (status, selected, selection_status) != expected:
        raise OcrReleaseGateError("ocr_release_report_selection_invalid")


def _validate_report(
    report: dict[str, Any],
    *,
    sample: SampleName,
    expected_documents: Sequence[BenchmarkDocument],
    critical_rules: Sequence[Mapping[str, str]],
    descriptor_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[
    dict[ProviderName, dict[str, Any]],
    dict[ProviderName, tuple[float, float, float, float]],
    dict[ProviderName, tuple[float, int, int] | None],
]:
    expected_document_count = SAMPLE_DOCUMENT_COUNTS[sample]
    if (
        set(report) != REPORT_KEYS
        or report.get("schema_version") != BENCHMARK_REPORT_SCHEMA
        or report.get("benchmark_id") != BENCHMARK_ID
        or report.get("sample") != sample
        or report.get("thresholds") != LOCKED_THRESHOLDS
        or report.get("document_count") != expected_document_count
        or report.get("reading_order_contract")
        != {
            "gold_policy_id": GOLD_READING_ORDER_POLICY_ID,
            "prediction_policy_id": OCR_READING_ORDER_POLICY_ID,
        }
        or report.get("redaction") != REDACTION
        or len(expected_documents) != expected_document_count
    ):
        raise OcrReleaseGateError("ocr_release_report_invalid")
    report_descriptor_sha256 = report.get("descriptor_sha256")
    report_manifest_sha256 = report.get("dataset_manifest_sha256")
    if (
        not isinstance(report_descriptor_sha256, str)
        or SHA256.fullmatch(report_descriptor_sha256) is None
        or report_descriptor_sha256 != descriptor_sha256
        or not isinstance(report_manifest_sha256, str)
        or SHA256.fullmatch(report_manifest_sha256) is None
        or report_manifest_sha256 != dataset_manifest_sha256
    ):
        raise OcrReleaseGateError("ocr_release_report_identity_invalid")
    coverage = report.get("coverage")
    if not isinstance(coverage, dict) or coverage != {
        "selected_sample": sample,
        "selected_document_count": expected_document_count,
        "locked_case_count": 20,
        "locked_triplet_document_count": 60,
        "co_document_count": 20,
        "release_gate_document_count": RELEASE_DOCUMENT_COUNT,
    }:
        raise OcrReleaseGateError("ocr_release_report_coverage_invalid")
    raw_providers = report.get("providers")
    if (
        not isinstance(raw_providers, list)
        or len(raw_providers) != len(PROVIDERS)
        or [item.get("provider") if isinstance(item, dict) else None for item in raw_providers]
        != list(PROVIDERS)
    ):
        raise OcrReleaseGateError("ocr_release_provider_set_invalid")
    providers: dict[ProviderName, dict[str, Any]] = {}
    qualities: dict[ProviderName, tuple[float, float, float, float]] = {}
    runtime_metrics: dict[ProviderName, tuple[float, int, int] | None] = {}
    for provider_name, raw_provider in zip(PROVIDERS, raw_providers, strict=True):
        provider, quality, provider_runtime_metrics = _validate_provider(
            raw_provider,
            provider_name=provider_name,
            expected_documents=expected_documents,
            critical_rules=critical_rules,
        )
        providers[provider_name] = provider
        qualities[provider_name] = quality
        runtime_metrics[provider_name] = provider_runtime_metrics
    _validate_top_selection(report, providers=providers, qualities=qualities)
    return providers, qualities, runtime_metrics


def _combine_invalidated_v2_diagnostic(
    *,
    locked_report_path: Path,
    co_regression_report_path: Path,
    descriptor_path: Path,
    rendered_root: Path,
) -> tuple[int, dict[str, Any]]:
    """Exercise legacy validation without producing a release PASS artifact."""

    try:
        (
            locked_descriptor_sha256,
            locked_manifest_sha256,
            locked_documents,
            locked_critical_rules,
        ) = load_benchmark(
            descriptor_path=descriptor_path,
            rendered_root=rendered_root,
            sample="locked",
        )
        (
            co_descriptor_sha256,
            co_manifest_sha256,
            co_documents,
            co_critical_rules,
        ) = load_benchmark(
            descriptor_path=descriptor_path,
            rendered_root=rendered_root,
            sample="co_regression",
        )
    except OcrBenchmarkError as exc:
        raise OcrReleaseGateError("ocr_release_canonical_input_invalid") from exc
    if (
        locked_descriptor_sha256 != co_descriptor_sha256
        or locked_manifest_sha256 != co_manifest_sha256
        or locked_critical_rules != co_critical_rules
    ):
        raise OcrReleaseGateError("ocr_release_canonical_identity_mismatch")

    locked_report_sha256, locked_report = _load_report(locked_report_path)
    co_report_sha256, co_report = _load_report(co_regression_report_path)
    locked_providers, locked_qualities, locked_runtime_metrics = _validate_report(
        locked_report,
        sample="locked",
        expected_documents=locked_documents,
        critical_rules=locked_critical_rules,
        descriptor_sha256=locked_descriptor_sha256,
        dataset_manifest_sha256=locked_manifest_sha256,
    )
    co_providers, _, co_runtime_metrics = _validate_report(
        co_report,
        sample="co_regression",
        expected_documents=co_documents,
        critical_rules=co_critical_rules,
        descriptor_sha256=co_descriptor_sha256,
        dataset_manifest_sha256=co_manifest_sha256,
    )
    for key in ("descriptor_sha256", "dataset_manifest_sha256", "thresholds"):
        if locked_report[key] != co_report[key]:
            raise OcrReleaseGateError("ocr_release_report_identity_mismatch")
    wheel_hashes = {
        cast(str, provider["installed_agent_wheel_sha256"])
        for provider in (*locked_providers.values(), *co_providers.values())
    }
    if len(wheel_hashes) != 1:
        raise OcrReleaseGateError("ocr_release_wheel_identity_mismatch")
    agent_wheel_sha256 = wheel_hashes.pop()
    environment_lock_hashes: dict[ProviderName, str] = {}
    for provider in PROVIDERS:
        provider_hashes = {
            cast(str, locked_providers[provider]["environment_lock_sha256"]),
            cast(str, co_providers[provider]["environment_lock_sha256"]),
        }
        if len(provider_hashes) != 1:
            raise OcrReleaseGateError("ocr_release_environment_lock_identity_mismatch")
        environment_lock_hashes[provider] = provider_hashes.pop()
    model_assets_lock_hashes = {
        cast(str, provider["model_assets_lock_sha256"])
        for provider in (*locked_providers.values(), *co_providers.values())
    }
    if len(model_assets_lock_hashes) != 1:
        raise OcrReleaseGateError("ocr_release_model_assets_lock_identity_mismatch")
    model_assets_lock_sha256 = model_assets_lock_hashes.pop()
    eligible: list[ProviderName] = [
        provider
        for provider in PROVIDERS
        if locked_providers[provider]["status"] == "PASS"
        and co_providers[provider]["status"] == "PASS"
    ]
    for provider in eligible:
        if locked_providers[provider]["engine"] != co_providers[provider]["engine"]:
            raise OcrReleaseGateError("ocr_release_engine_identity_mismatch")
    selected_provider: ProviderName | None = None
    selection_status = "NO_PROVIDER_PASSED_BOTH_RELEASE_SAMPLES"
    runtime_tiebreak: dict[str, Any] | None = None
    if eligible:
        if len(eligible) == 1:
            selected_provider = eligible[0]
            selection_status = "SELECTED_SINGLE_COMMON_PROVIDER"
        elif len({locked_qualities[provider] for provider in eligible}) == 1:
            runtime_values = {provider: locked_runtime_metrics[provider] for provider in eligible}
            co_runtime_values = {provider: co_runtime_metrics[provider] for provider in eligible}
            if any(
                value is None for value in (*runtime_values.values(), *co_runtime_values.values())
            ):
                selection_status = "QUALITY_TIE_RUNTIME_METRICS_UNAVAILABLE"
            else:
                comparable = {
                    provider: cast(tuple[float, int, int], value)
                    for provider, value in runtime_values.items()
                }
                co_comparable = {
                    provider: cast(tuple[float, int, int], value)
                    for provider, value in co_runtime_values.items()
                }
                if any(
                    comparable[provider][2] != co_comparable[provider][2] for provider in eligible
                ):
                    raise OcrReleaseGateError("ocr_release_runtime_identity_mismatch")
                runtime_tiebreak = {
                    "source_sample": "locked",
                    "selection_order": [
                        "latency_p95_ms",
                        "peak_memory_bytes",
                        "disk_footprint_bytes",
                    ],
                    "providers": [
                        {
                            "provider": provider,
                            "latency_p95_ms": comparable[provider][0],
                            "peak_memory_bytes": comparable[provider][1],
                            "disk_footprint_bytes": comparable[provider][2],
                        }
                        for provider in PROVIDERS
                    ],
                }
                if len(set(comparable.values())) == 1:
                    selection_status = "RUNTIME_METRICS_TIE_NO_AUTOMATIC_SELECTION"
                else:
                    selected_provider = (
                        "paddle" if comparable["paddle"] < comparable["docling"] else "docling"
                    )
                    selection_status = "SELECTED_BY_ATTESTED_RUNTIME_METRICS"
        else:
            best_provider: ProviderName = max(
                eligible,
                key=lambda provider: locked_qualities[provider],
            )
            selected_provider = best_provider
            selection_status = "SELECTED_BY_LOCKED_QUALITY_METRICS"
    report = {
        "schema_version": DIAGNOSTIC_REPORT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": (
            "NONRELEASE_DIAGNOSTIC_PASS"
            if selected_provider is not None
            else "NONRELEASE_DIAGNOSTIC_FAIL"
        ),
        "release_eligible": False,
        "release_holdout_status": RELEASE_INVALIDATION_CODE,
        "selection_status": selection_status,
        "selected_provider": selected_provider,
        "smoke_included": False,
        "release_document_count": RELEASE_DOCUMENT_COUNT,
        "descriptor_sha256": locked_descriptor_sha256,
        "dataset_manifest_sha256": locked_manifest_sha256,
        "agent_wheel_sha256": agent_wheel_sha256,
        "environment_lock_sha256": (
            environment_lock_hashes[selected_provider] if selected_provider is not None else None
        ),
        "model_assets_lock_sha256": (
            model_assets_lock_sha256 if selected_provider is not None else None
        ),
        "selected_engine": (
            locked_providers[selected_provider]["engine"] if selected_provider is not None else None
        ),
        "input_reports": {
            "locked": {
                "sha256": locked_report_sha256,
                "document_count": SAMPLE_DOCUMENT_COUNTS["locked"],
            },
            "co_regression": {
                "sha256": co_report_sha256,
                "document_count": SAMPLE_DOCUMENT_COUNTS["co_regression"],
            },
        },
        "provider_eligibility": [
            {
                "provider": provider,
                "locked_status": locked_providers[provider]["status"],
                "co_regression_status": co_providers[provider]["status"],
                "eligible": provider in eligible,
            }
            for provider in PROVIDERS
        ],
        "runtime_tiebreak": runtime_tiebreak,
        "failure_codes": (
            []
            if selected_provider is not None
            else (
                (
                    ["quality_tie_runtime_metrics_unavailable"]
                    if selection_status == "QUALITY_TIE_RUNTIME_METRICS_UNAVAILABLE"
                    else ["runtime_metrics_tie_no_automatic_selection"]
                )
                if eligible
                else ["no_provider_passed_both_release_samples"]
            )
        ),
        "redaction": REDACTION,
    }
    return (0 if selected_provider is not None else 1), report


def combine(
    *,
    locked_report_path: Path,
    co_regression_report_path: Path,
    descriptor_path: Path,
    rendered_root: Path,
) -> tuple[int, dict[str, Any]]:
    """Block the invalidated v2 release combiner before any input access.

    A future release combiner must be implemented against an evaluation-v3
    sealed runner and a verified ``OcrFreezeAttestation``.  Keeping the
    arguments preserves CLI compatibility while making their non-access an
    explicit, testable security boundary.
    """

    del (
        locked_report_path,
        co_regression_report_path,
        descriptor_path,
        rendered_root,
    )
    raise OcrReleaseGateError(RELEASE_INVALIDATION_CODE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine locked and C/O OCR reports into one offline release Gate."
    )
    parser.add_argument("--locked-report", required=True, type=Path)
    parser.add_argument("--co-regression-report", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--rendered-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        exit_code, report = combine(
            locked_report_path=args.locked_report,
            co_regression_report_path=args.co_regression_report,
            descriptor_path=args.descriptor,
            rendered_root=args.rendered_root,
        )
    except OcrReleaseGateError as exc:
        exit_code = 2
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "FAIL",
            "selection_status": "PREFLIGHT_FAILED",
            "selected_provider": None,
            "error_code": exc.code,
            "redaction": REDACTION,
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
        except OSError:
            return 2
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
