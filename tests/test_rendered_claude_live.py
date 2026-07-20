from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
import pytest

from customs_ai_agent.adapters import OcrLayoutProviderError
from customs_ai_agent.adapters.llm import (
    DEFAULT_CLAUDE_MODEL,
    DocumentAnalysis,
    LLMError,
    ModelCallMetadata,
)
from customs_ai_agent.schemas.llm import (
    ClaudePixelBBox,
    ClaudeVisionDocumentCandidate,
    ClaudeVisionEvidence,
    ClaudeVisionFieldCandidate,
)
from customs_ai_agent.schemas.models import (
    BBox,
    OcrLayoutLedger,
    OcrLayoutPage,
    OcrLayoutSpan,
    stable_ocr_ledger_id,
)
from customs_ai_agent.ocr_identity import (
    OCR_MODEL_ASSETS_LOCK_SHA256,
    OCR_MODEL_SET_IDS,
)
from evals.combine_ocr_release_gate import REDACTION as OCR_RELEASE_REDACTION
from evals import run_rendered_claude_live as live_eval
from evals.run_ocr_benchmark import (
    CANONICAL_SPLITS_SHA256,
    LOCKED_THRESHOLDS,
    OcrBenchmarkError,
)

VISIBLE_VALUE = "INV-LOCKED-001 / 2026-07-18"
TEST_LOCKED_CASE_IDS = tuple(f"trade-triplet-{index:04d}" for index in range(22, 42))
FIRST_LOCKED_BENCHMARK_ID = "trade-triplet-0022-commercial-invoice"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    image_path = rendered / "document.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    image_sha = _sha256(image_path)

    locked_case_ids = list(TEST_LOCKED_CASE_IDS)
    type_records = (
        ("commercial_invoice", "commercial-invoice"),
        ("packing_list", "packing-list"),
        ("bill_of_lading", "bill-of-lading"),
    )
    locked_records = []
    generated_files = [
        {
            "relative_path": image_path.name,
            "sha256": image_sha,
            "byte_size": image_path.stat().st_size,
        }
    ]
    layouts_root = rendered / "layouts"
    layouts_root.mkdir()
    for case_id in locked_case_ids:
        for document_type, suffix in type_records:
            benchmark_document_id = f"{case_id}-{suffix}"
            layout_relative_path = f"layouts/{benchmark_document_id}.layout-gold.json"
            layout_path = rendered / Path(layout_relative_path)
            if document_type == "bill_of_lading":
                field_id = "bill_of_lading:/B~1L No. (Bill of Lading Number)"
                raw_value = "INV-LOCKED-001"
            else:
                field_id = f"{document_type}:/Invoice No. and Date"
                raw_value = VISIBLE_VALUE
            layout = {
                "schema_version": "trade-document-layout-gold/v2",
                "case_id": case_id,
                "document_id": f"{case_id}:{document_type}",
                "document_type": document_type,
                "page": {
                    "claude_input_width_px": 20,
                    "claude_input_height_px": 20,
                    "claude_input_visual_tokens": 10,
                },
                "files": {
                    "claude_input_png_relative_path": image_path.name,
                    "claude_input_png_sha256": image_sha,
                },
                "fields": [
                    {
                        "field_id": field_id,
                        "raw_value": raw_value,
                        "rendered": True,
                    }
                ],
                "words": [
                    {
                        "word_id": "word-1",
                        "text": raw_value,
                        "role": "value",
                        "bbox_claude_input_px": [0, 0, 20, 10],
                    }
                ],
                "tables": [],
            }
            _write_json(layout_path, layout)
            layout_sha = _sha256(layout_path)
            generated_files.append(
                {
                    "relative_path": layout_relative_path,
                    "sha256": layout_sha,
                    "byte_size": layout_path.stat().st_size,
                }
            )
            locked_records.append(
                {
                    "benchmark_document_id": benchmark_document_id,
                    "image_relative_path": image_path.name,
                    "image_sha256": image_sha,
                    "layout_relative_path": layout_relative_path,
                    "layout_sha256": layout_sha,
                }
            )

    first_layout_record = locked_records[0]
    smoke_record = {
        **first_layout_record,
        "benchmark_document_id": "trade-triplet-0001-commercial-invoice",
    }
    co_records = [
        {
            **first_layout_record,
            "benchmark_document_id": (f"co-synthetic-{index:04d}-certificate-of-origin"),
        }
        for index in range(1, 21)
    ]
    for index in range(1280 - len(generated_files)):
        generated_files.append(
            {
                "relative_path": f"unused/{index:04d}.bin",
                "sha256": "0" * 64,
                "byte_size": 0,
            }
        )
    manifest_path = rendered / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "trade-document-render-manifest/v2",
            "dataset_id": "trade_document_triplets",
            "dataset_version": "source-v1",
            "counts": {
                "source_case_count": 100,
                "triplet_document_count": 300,
                "certificate_of_origin_document_count": 20,
                "document_count": 320,
                "generated_document_file_count": 1280,
                "claude_input_resized_document_count": 320,
            },
            "generated_files": generated_files,
        },
    )
    manifest_sha = _sha256(manifest_path)

    descriptor_path = tmp_path / "descriptor.json"
    descriptor = {
        "schema_version": "ocr-benchmark-descriptor/v2",
        "benchmark_id": "rendered-trade-documents-ocr-v2",
        "evaluation_version": "evaluation-v2",
        "dataset_manifest_sha256": manifest_sha,
        "dataset_splits_sha256": CANONICAL_SPLITS_SHA256,
        "release_context": {
            "authorization_commit_sha": "d" * 40,
            "authorization_receipt_sha256": "e" * 64,
            "authorization_receipt_registry_key": "f" * 64,
        },
        "input_variant": "claude_input_png",
        "thresholds": LOCKED_THRESHOLDS,
        "critical_identifier_rules": [
            {
                "field_id_marker": "Invoice No. and Date",
                "selector": "before_slash",
            }
        ],
        "samples": {
            "smoke": ["trade-triplet-0001-commercial-invoice"],
            "locked": [record["benchmark_document_id"] for record in locked_records],
            "co_regression": [record["benchmark_document_id"] for record in co_records],
        },
        "coverage": {
            "smoke_split": "development",
            "smoke_case_id": "trade-triplet-0001",
            "smoke_document_count": 1,
            "locked_split": "locked_test",
            "locked_case_ids": locked_case_ids,
            "locked_case_count": 20,
            "locked_document_types": [
                "commercial_invoice",
                "packing_list",
                "bill_of_lading",
            ],
            "locked_triplet_document_count": 60,
            "co_document_count": 20,
            "release_gate_document_count": 80,
            "descriptor_document_count": 81,
        },
        "documents": [smoke_record, *locked_records, *co_records],
    }
    _write_json(descriptor_path, descriptor)
    descriptor_sha = _sha256(descriptor_path)
    environment_lock_path = tmp_path / "paddle.requirements.lock.txt"
    environment_lock_path.write_text("fixture==1.0 --hash=sha256:" + "d" * 64, encoding="utf-8")
    environment_lock_sha256 = _sha256(environment_lock_path)
    model_assets_lock_path = tmp_path / "model-assets.lock.json"
    model_assets_lock_path.write_text('{"fixture":"model-assets"}', encoding="utf-8")
    model_assets_lock_sha256 = _sha256(model_assets_lock_path)

    benchmark_path = tmp_path / "ocr-release-gate.json"
    _write_json(
        benchmark_path,
        {
            "schema_version": "ocr-release-gate-report/v2",
            "benchmark_id": "rendered-trade-documents-ocr-v2",
            "status": "PASS",
            "selection_status": "SELECTED_SINGLE_COMMON_PROVIDER",
            "selected_provider": "paddle",
            "smoke_included": False,
            "release_document_count": 80,
            "descriptor_sha256": descriptor_sha,
            "dataset_manifest_sha256": manifest_sha,
            "agent_wheel_sha256": "a" * 64,
            "environment_lock_sha256": environment_lock_sha256,
            "model_assets_lock_sha256": model_assets_lock_sha256,
            "selected_engine": {
                "provider_id": "paddle",
                "engine_id": "fixture-engine",
                "engine_version": "1.0",
            },
            "input_reports": {
                "locked": {
                    "sha256": "b" * 64,
                    "document_count": 60,
                },
                "co_regression": {
                    "sha256": "c" * 64,
                    "document_count": 20,
                },
            },
            "provider_eligibility": [
                {
                    "provider": "paddle",
                    "locked_status": "PASS",
                    "co_regression_status": "PASS",
                    "eligible": True,
                },
                {
                    "provider": "docling",
                    "locked_status": "FAIL",
                    "co_regression_status": "FAIL",
                    "eligible": False,
                },
            ],
            "runtime_tiebreak": None,
            "failure_codes": [],
            "redaction": OCR_RELEASE_REDACTION,
        },
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ANTHROPIC_API_KEY=unit-test-placeholder\nCUSTOMS_AGENT_CLAUDE_MAX_TOKENS=1024\n",
        encoding="utf-8",
    )
    return {
        "env_file": env_path,
        "descriptor_path": descriptor_path,
        "rendered_root": rendered,
        "ocr_release_gate_report": benchmark_path,
        "ocr_release_gate_report_sha256": _sha256(benchmark_path),
        "expected_agent_wheel_sha256": "a" * 64,
        "ocr_environment_lock": environment_lock_path,
        "ocr_environment_lock_sha256": environment_lock_sha256,
        "ocr_model_assets_lock": model_assets_lock_path,
        "ocr_model_assets_lock_sha256": model_assets_lock_sha256,
        "ocr_provider": "paddle",
        "ocr_python": Path(sys.executable).resolve(),
        "max_documents": 1,
        "max_calls": 1,
        "max_total_input_tokens": 100_000,
        "max_total_output_tokens": 10_000,
        "wheel_attestor": lambda _executable: "a" * 64,
        "environment_attestor": (lambda _executable, _lock: environment_lock_sha256),
        "model_assets_attestor": (lambda _provider, _lock, _engine: model_assets_lock_sha256),
    }


class FakeProvider:
    def __init__(
        self,
        *,
        grounded: bool = True,
        provider_id: str = "paddle",
        engine_id: str = "fixture-engine",
        engine_version: str = "1.0",
    ) -> None:
        self.grounded = grounded
        self.provider_id = provider_id
        self.engine_id = engine_id
        self.engine_version = engine_version
        self.calls = 0
        self.closed = False

    def analyze(self, *, document_id: str, pages: list[Any]) -> OcrLayoutLedger:
        self.calls += 1
        text = VISIBLE_VALUE if self.grounded else "DIFFERENT OCR TEXT"
        span = OcrLayoutSpan(
            span_id="span-1",
            page=1,
            reading_order=0,
            role="value",
            text=text,
            bbox=BBox(page=1, x1=0, y1=0, x2=20, y2=10),
            confidence=1.0,
        )
        page = OcrLayoutPage(
            page_number=1,
            image_sha256=pages[0].sha256,
            width_px=pages[0].width_px,
            height_px=pages[0].height_px,
            spans=[span],
        )
        ledger_id = stable_ocr_ledger_id(
            document_id=document_id,
            provider_id=self.provider_id,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            model_id=OCR_MODEL_SET_IDS[self.provider_id],
            model_assets_lock_sha256=OCR_MODEL_ASSETS_LOCK_SHA256,
            pages=[page],
        )
        return OcrLayoutLedger(
            ledger_id=ledger_id,
            document_id=document_id,
            provider_id=self.provider_id,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            model_id=OCR_MODEL_SET_IDS[self.provider_id],
            model_assets_lock_sha256=OCR_MODEL_ASSETS_LOCK_SHA256,
            pages=[page],
        )

    def close(self) -> None:
        self.closed = True


class FakeAnalyzer:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    def analyze(self, document: Mapping[str, Any]) -> DocumentAnalysis:
        pages = list(document["image_pages"])
        ledger = self.provider.analyze(
            document_id=str(document["document_id"]),
            pages=pages,
        )
        evidence = ClaudeVisionEvidence(
            source_text=VISIBLE_VALUE,
            model_observed_bbox=ClaudePixelBBox(
                page=1,
                x1=0,
                y1=0,
                x2=20,
                y2=10,
            ),
            ocr_span_ids=["span-1"],
        )
        candidate = ClaudeVisionDocumentCandidate(
            schema_version="1.0",
            document_type="invoice",
            classification_evidence=evidence,
            fields=[
                ClaudeVisionFieldCandidate(
                    name="invoice_no",
                    value="INV-LOCKED-001",
                    evidence=evidence,
                ),
                ClaudeVisionFieldCandidate(
                    name="invoice_date",
                    value="2026-07-18",
                    evidence=evidence,
                ),
            ],
            line_items=[],
        )
        return DocumentAnalysis(
            candidate=candidate,
            metadata=ModelCallMetadata(
                provider="anthropic",
                model_id=DEFAULT_CLAUDE_MODEL,
                prompt_id=live_eval.DOCUMENT_VISION_ANALYSIS_PROMPT_ID,
                prompt_version=live_eval.DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION,
                prompt_sha256=live_eval.VISION_PROMPT_SHA256,
                output_schema_id="customs_claude_vision_document_candidate",
                output_schema_version="1.0",
                output_schema_sha256=live_eval.VISION_SCHEMA_SHA256,
                request_id="redacted-by-report",
                message_id="redacted-by-report",
                stop_reason="end_turn",
                input_tokens=100,
                output_tokens=20,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                latency_ms=12,
                input_modality="image",
                ocr_ledger_id=ledger.ledger_id,
                ocr_provider_id=ledger.provider_id,
                ocr_engine_id=ledger.engine_id,
                ocr_engine_version=ledger.engine_version,
                ocr_model_id=ledger.model_id,
                ocr_model_assets_lock_sha256=ledger.model_assets_lock_sha256,
            ),
            ocr_layout_ledger=ledger,
        )


def test_live_flag_failure_never_creates_provider_or_analyzer(tmp_path: Path):
    arguments = _fixture(tmp_path)
    counters = {"provider": 0, "analyzer": 0}

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        counters["provider"] += 1
        return FakeProvider()

    def analyzer_factory(_env: Path, provider: Any) -> FakeAnalyzer:
        counters["analyzer"] += 1
        return FakeAnalyzer(provider)

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=False,
        provider_factory=provider_factory,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["failures"] == [{"code": "live_flag_required"}]
    assert report["ocr_provider_created"] is False
    assert report["anthropic_provider_called"] is False
    assert counters == {"provider": 0, "analyzer": 0}


def test_provider_lock_hash_failure_never_creates_provider(tmp_path: Path):
    arguments = _fixture(tmp_path)
    arguments["ocr_release_gate_report_sha256"] = "0" * 64
    provider_calls = 0

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        nonlocal provider_calls
        provider_calls += 1
        return FakeProvider()

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=provider_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"code": "ocr_release_gate_report_hash_mismatch"}]
    assert provider_calls == 0


def test_ocr_python_wheel_mismatch_fails_before_provider_or_claude(
    tmp_path: Path,
):
    arguments = _fixture(tmp_path)
    attested_paths: list[Path] = []
    counters = {"provider": 0, "analyzer": 0}

    def wheel_attestor(executable: Path) -> str:
        attested_paths.append(executable)
        return "b" * 64

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        counters["provider"] += 1
        return FakeProvider()

    def analyzer_factory(_env: Path, provider: Any) -> FakeAnalyzer:
        counters["analyzer"] += 1
        return FakeAnalyzer(provider)

    arguments["wheel_attestor"] = wheel_attestor
    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=provider_factory,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"code": "ocr_release_gate_installed_wheel_mismatch"}]
    assert attested_paths == [Path(sys.executable).resolve()]
    assert report["ocr_provider_created"] is False
    assert report["anthropic_provider_called"] is False
    assert counters == {"provider": 0, "analyzer": 0}


@pytest.mark.parametrize(
    ("attestor_name", "expected_code"),
    [
        (
            "environment_attestor",
            "ocr_release_gate_environment_lock_hash_mismatch",
        ),
        (
            "model_assets_attestor",
            "ocr_release_gate_model_assets_lock_hash_mismatch",
        ),
    ],
)
def test_runtime_dependency_or_model_inventory_drift_fails_before_provider(
    tmp_path: Path,
    attestor_name: str,
    expected_code: str,
) -> None:
    arguments = _fixture(tmp_path)
    counters = {"provider": 0, "analyzer": 0}
    if attestor_name == "environment_attestor":
        arguments[attestor_name] = lambda _executable, _lock: "f" * 64
    else:
        arguments[attestor_name] = lambda _provider, _lock, _engine: "f" * 64

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        counters["provider"] += 1
        return FakeProvider()

    def analyzer_factory(_env: Path, provider: Any) -> FakeAnalyzer:
        counters["analyzer"] += 1
        return FakeAnalyzer(provider)

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=provider_factory,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"code": expected_code}]
    assert report["ocr_provider_created"] is False
    assert report["anthropic_provider_called"] is False
    assert counters == {"provider": 0, "analyzer": 0}


def test_selected_provider_requires_its_own_installed_wheel_hash(
    tmp_path: Path,
):
    arguments = _fixture(tmp_path)
    benchmark_path = arguments["ocr_release_gate_report"]
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    del benchmark["agent_wheel_sha256"]
    _write_json(benchmark_path, benchmark)
    arguments["ocr_release_gate_report_sha256"] = _sha256(benchmark_path)
    counters = {"attestor": 0, "provider": 0, "analyzer": 0}

    def wheel_attestor(_executable: Path) -> str:
        counters["attestor"] += 1
        return "a" * 64

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        counters["provider"] += 1
        return FakeProvider()

    def analyzer_factory(_env: Path, provider: Any) -> FakeAnalyzer:
        counters["analyzer"] += 1
        return FakeAnalyzer(provider)

    arguments["wheel_attestor"] = wheel_attestor
    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=provider_factory,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"code": "ocr_release_gate_report_invalid"}]
    assert report["ocr_provider_created"] is False
    assert report["anthropic_provider_called"] is False
    assert counters == {"attestor": 0, "provider": 0, "analyzer": 0}


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("combined_fail", "ocr_release_gate_status_not_passing"),
        ("quality_tie", "ocr_release_gate_status_not_passing"),
        ("no_provider", "ocr_release_gate_status_not_passing"),
        ("provider", "ocr_release_gate_provider_mismatch"),
        ("engine", "ocr_release_gate_engine_mismatch"),
        ("wheel", "ocr_release_gate_expected_wheel_mismatch"),
        (
            "environment_lock",
            "ocr_release_gate_environment_lock_hash_mismatch",
        ),
        (
            "model_assets_lock",
            "ocr_release_gate_model_assets_lock_hash_mismatch",
        ),
        (
            "selected_sample_failure",
            "ocr_release_gate_selected_provider_not_passing",
        ),
    ],
)
def test_combined_release_gate_defects_block_provider_and_claude(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
):
    arguments = _fixture(tmp_path)
    report_path = arguments["ocr_release_gate_report"]
    release_report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "combined_fail":
        release_report["status"] = "FAIL"
    elif mutation == "quality_tie":
        release_report["status"] = "FAIL"
        release_report["selection_status"] = "QUALITY_TIE_REQUIRES_RUNTIME_BENCHMARK"
        release_report["selected_provider"] = None
        release_report["selected_engine"] = None
        release_report["failure_codes"] = ["quality_tie_requires_runtime_benchmark"]
    elif mutation == "no_provider":
        release_report["status"] = "FAIL"
        release_report["selection_status"] = "NO_PROVIDER_PASSED_BOTH_RELEASE_SAMPLES"
        release_report["selected_provider"] = None
        release_report["selected_engine"] = None
        release_report["failure_codes"] = ["no_provider_passed_both_release_samples"]
    elif mutation == "provider":
        release_report["selected_provider"] = "docling"
        release_report["selected_engine"]["provider_id"] = "docling"
        release_report["provider_eligibility"] = [
            {
                "provider": "paddle",
                "locked_status": "FAIL",
                "co_regression_status": "FAIL",
                "eligible": False,
            },
            {
                "provider": "docling",
                "locked_status": "PASS",
                "co_regression_status": "PASS",
                "eligible": True,
            },
        ]
    elif mutation == "engine":
        release_report["selected_engine"]["provider_id"] = "docling"
    elif mutation == "wheel":
        release_report["agent_wheel_sha256"] = "d" * 64
    elif mutation == "environment_lock":
        release_report["environment_lock_sha256"] = "d" * 64
    elif mutation == "model_assets_lock":
        release_report["model_assets_lock_sha256"] = "d" * 64
    elif mutation == "selected_sample_failure":
        release_report["provider_eligibility"][0]["co_regression_status"] = "FAIL"
        release_report["provider_eligibility"][0]["eligible"] = False
    _write_json(report_path, release_report)
    arguments["ocr_release_gate_report_sha256"] = _sha256(report_path)
    counters = {"attestor": 0, "provider": 0, "analyzer": 0}

    def wheel_attestor(_executable: Path) -> str:
        counters["attestor"] += 1
        return "a" * 64

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        counters["provider"] += 1
        return FakeProvider()

    def analyzer_factory(_settings: Any, provider: Any) -> FakeAnalyzer:
        counters["analyzer"] += 1
        return FakeAnalyzer(provider)

    arguments["wheel_attestor"] = wheel_attestor
    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=provider_factory,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"code": expected_code}]
    assert counters == {"attestor": 0, "provider": 0, "analyzer": 0}
    assert report["anthropic_provider_called"] is False


def test_old_single_sample_benchmark_report_is_rejected_before_any_provider(
    tmp_path: Path,
):
    arguments = _fixture(tmp_path)
    report_path = arguments["ocr_release_gate_report"]
    _write_json(
        report_path,
        {
            "schema_version": "ocr-benchmark-report/v1",
            "benchmark_id": "rendered-trade-documents-ocr-v1",
            "sample": "locked",
            "status": "PASS",
            "selected_provider": "paddle",
        },
    )
    arguments["ocr_release_gate_report_sha256"] = _sha256(report_path)
    provider_calls = 0

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        nonlocal provider_calls
        provider_calls += 1
        return FakeProvider()

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=provider_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"code": "ocr_release_gate_report_invalid"}]
    assert provider_calls == 0
    assert report["anthropic_provider_called"] is False


def test_tampered_rendered_image_fails_before_provider_creation(tmp_path: Path):
    arguments = _fixture(tmp_path)
    image_path = arguments["rendered_root"] / "document.png"
    image_path.write_bytes(image_path.read_bytes() + b"tamper")
    provider_calls = 0

    def provider_factory(_provider: str, _executable: Path) -> FakeProvider:
        nonlocal provider_calls
        provider_calls += 1
        return FakeProvider()

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=provider_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [{"code": "benchmark_document_hash_mismatch"}]
    assert provider_calls == 0


def test_runtime_ocr_engine_drift_is_blocked_before_claude_transport(
    tmp_path: Path,
):
    arguments = _fixture(tmp_path)
    provider = FakeProvider(engine_version="2.0")
    external_calls = 0

    class OcrThenClaudeAnalyzer:
        def __init__(self, active_provider):
            self.provider = active_provider

        def analyze(self, document):
            nonlocal external_calls
            try:
                self.provider.analyze(
                    document_id=str(document["document_id"]),
                    pages=document["image_pages"],
                )
            except OcrLayoutProviderError as exc:
                raise LLMError(exc.code, "runtime OCR identity rejected") from None
            external_calls += 1
            raise AssertionError("Claude transport must not be reached")

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=lambda _name, _python: provider,
        analyzer_factory=lambda _settings, active: OcrThenClaudeAnalyzer(active),
    )

    assert exit_code == 1
    assert report["failures"] == [
        {
            "document_reference": hashlib.sha256(
                FIRST_LOCKED_BENCHMARK_ID.encode("utf-8")
            ).hexdigest(),
            "code": "ocr_release_gate_runtime_engine_mismatch",
        }
    ]
    assert provider.calls == 1
    assert external_calls == 0


def test_passing_extraction_is_locked_grounded_and_redacted(tmp_path: Path):
    arguments = _fixture(tmp_path)
    provider = FakeProvider()

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=lambda _name, _python: provider,
        analyzer_factory=lambda _env, active: FakeAnalyzer(active),
    )

    assert exit_code == 0
    assert report["status"] == "DIAGNOSTIC_PASS"
    assert report["release_score_eligible"] is False
    assert report["evaluation_mode"] == "diagnostic"
    assert report["scope"] == ("RENDERED_GOLD_EXTRACTION_ONLY_NOT_WORKFLOW_OR_SYSTEM_RELEASE")
    assert report["metrics"]["document_type"]["exact_accuracy"] == 1.0
    assert report["metrics"]["critical_identifiers"]["exact_accuracy"] == 1.0
    assert report["metrics"]["required_fields"]["exact_accuracy"] == 1.0
    assert report["metrics"]["hallucinated_required_field_count"] == 0
    assert report["metrics"]["ocr_grounding_failure_count"] == 0
    assert report["call_count"] == 1
    assert report["ocr_provider_created"] is True
    assert report["anthropic_provider_called"] is True
    assert provider.calls == 1
    assert provider.closed is True
    serialized = json.dumps(report, ensure_ascii=False)
    assert VISIBLE_VALUE not in serialized
    assert "unit-test-placeholder" not in serialized
    assert str(tmp_path) not in serialized
    assert "redacted-by-report" not in serialized
    assert "data_base64" not in serialized


def test_release_mode_rejects_missing_co_regression_before_provider_or_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    original_loader = live_eval.load_benchmark
    calls = {"provider": 0, "analyzer": 0}

    def missing_co_loader(**kwargs):
        if kwargs["sample"] == "co_regression":
            raise OcrBenchmarkError("benchmark_coverage_invalid")
        return original_loader(**kwargs)

    def provider_factory(_name, _python):
        calls["provider"] += 1
        raise AssertionError("missing C/O must fail before provider creation")

    def analyzer_factory(_settings, _provider):
        calls["analyzer"] += 1
        raise AssertionError("missing C/O must fail before Claude creation")

    monkeypatch.setattr(live_eval, "load_benchmark", missing_co_loader)
    arguments["max_documents"] = live_eval.RELEASE_DOCUMENT_COUNT
    arguments["max_calls"] = live_eval.RELEASE_DOCUMENT_COUNT

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        release_mode=True,
        provider_factory=provider_factory,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["release_score_eligible"] is False
    assert report["failures"] == [{"code": "co_regression_release_sample_invalid"}]
    assert calls == {"provider": 0, "analyzer": 0}


def test_rendered_release_pass_status_requires_release_eligible_denominator() -> None:
    assert (
        live_eval._score_status(
            passed=True,
            release_score_eligible=False,
        )
        == "DIAGNOSTIC_PASS"
    )
    assert (
        live_eval._score_status(
            passed=True,
            release_score_eligible=True,
        )
        == "PASS"
    )


def test_ungrounded_candidate_fails_without_leaking_values(tmp_path: Path):
    arguments = _fixture(tmp_path)
    provider = FakeProvider(grounded=False)

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=lambda _name, _python: provider,
        analyzer_factory=lambda _env, active: FakeAnalyzer(active),
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["metrics"]["ocr_grounding_failure_count"] == 3
    assert report["checks"]["ocr_grounding"] is False
    serialized = json.dumps(report, ensure_ascii=False)
    assert VISIBLE_VALUE not in serialized
    assert "DIFFERENT OCR TEXT" not in serialized
    assert str(tmp_path) not in serialized


def test_failed_primary_call_is_attempted_once_and_disables_sdk_retries(
    tmp_path: Path,
):
    arguments = _fixture(tmp_path)
    provider = FakeProvider()
    captured_settings = []

    class FailingAnalyzer:
        def analyze(self, _document):
            raise LLMError("provider_unavailable", "simulated provider failure")

    def analyzer_factory(settings, _active):
        captured_settings.append(settings)
        return FailingAnalyzer()

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=lambda _name, _python: provider,
        analyzer_factory=analyzer_factory,
    )

    assert exit_code == 1
    assert report["failures"] == [
        {
            "document_reference": hashlib.sha256(
                FIRST_LOCKED_BENCHMARK_ID.encode("utf-8")
            ).hexdigest(),
            "code": "provider_unavailable",
        }
    ]
    assert report["call_accounting"] == {
        "attempted": 1,
        "succeeded": 0,
        "retried": 0,
        "main_attempted": 1,
        "crop_attempted": 0,
    }
    assert report["call_count"] == 1
    assert report["cost_budget"]["configured_max_retries"] == 0
    assert len(captured_settings) == 1
    assert captured_settings[0].max_retries == 0


def test_failed_crop_attempt_is_included_in_total_attempted_calls(
    tmp_path: Path,
):
    arguments = _fixture(tmp_path)
    arguments["max_calls"] = 2
    provider = FakeProvider()

    class CropFailureAnalyzer(FakeAnalyzer):
        def analyze(self, document):
            crop_budget = document["_runtime_crop_budget"]
            assert crop_budget.claim() is True
            return super().analyze(document)

    exit_code, report = live_eval.evaluate(
        **arguments,
        live=True,
        provider_factory=lambda _name, _python: provider,
        analyzer_factory=lambda _settings, active: CropFailureAnalyzer(active),
    )

    assert exit_code == 0
    assert report["call_accounting"] == {
        "attempted": 2,
        "succeeded": 1,
        "retried": 0,
        "main_attempted": 1,
        "crop_attempted": 1,
    }
    assert report["call_count"] == 2
