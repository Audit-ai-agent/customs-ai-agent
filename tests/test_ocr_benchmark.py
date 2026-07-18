from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from customs_ai_agent.adapters.ocr import OcrLayoutProviderError
from customs_ai_agent.ocr_release import _WHEEL_ATTESTATION_SCRIPT
from customs_ai_agent.ocr_freeze import OcrFreezeVerificationError
from customs_ai_agent.schemas.models import (
    BBox,
    ImagePageInput,
    OcrLayoutLedger,
    OcrLayoutPage,
    OcrLayoutSpan,
    stable_ocr_ledger_id,
)
from customs_ai_agent.ocr_identity import (
    OCR_MODEL_ASSETS_LOCK_SHA256,
    OCR_MODEL_SET_IDS,
)
from evals.run_ocr_benchmark import (
    CANONICAL_ENVIRONMENT_LOCK_SHA256,
    CANONICAL_MODEL_ASSETS_LOCK_SHA256,
    CANONICAL_SPLITS_SHA256,
    LOCKED_THRESHOLDS,
    OcrBenchmarkError,
    PROVIDER_ENGINE_IDS,
    _installed_agent_wheel_sha256,
    evaluate,
    load_benchmark,
    score_document,
)
from tools.build_ocr_benchmark_descriptor import (
    DescriptorBuildError,
    build_descriptor,
)

TEST_LOCKED_CASE_IDS = tuple(f"trade-triplet-{index:04d}" for index in range(22, 42))
TEST_RELEASE_CONTEXT = {
    "authorization_commit_sha": "a" * 40,
    "authorization_receipt_sha256": "b" * 64,
    "authorization_receipt_registry_key": "c" * 64,
}


def _layout(image_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "trade-document-layout-gold/v2",
        "document_id": "fixture:invoice",
        "page": {
            "claude_input_width_px": 200,
            "claude_input_height_px": 100,
        },
        "files": {"claude_input_png_sha256": image_sha256},
        "fields": [
            {
                "field_id": "invoice:/Invoice No.",
                "raw_value": "INV-001",
            },
            {"field_id": "invoice:/item", "raw_value": "A"},
            {"field_id": "invoice:/qty", "raw_value": "1"},
        ],
        "words": [
            {
                "word_id": "word-1",
                "text": "INV-001",
                "role": "field_value",
                "bbox_claude_input_px": [10, 10, 50, 20],
            },
            {
                "word_id": "word-2",
                "text": "ITEM",
                "role": "table_header",
                "bbox_claude_input_px": [10, 40, 60, 50],
            },
            {
                "word_id": "word-3",
                "text": "QTY",
                "role": "table_header",
                "bbox_claude_input_px": [60, 40, 100, 50],
            },
            {
                "word_id": "word-4",
                "text": "A",
                "role": "table_value",
                "bbox_claude_input_px": [10, 50, 60, 70],
            },
            {
                "word_id": "word-5",
                "text": "1",
                "role": "table_value",
                "bbox_claude_input_px": [60, 50, 100, 70],
            },
        ],
        "tables": [
            {
                "table_id": "line-items",
                "bbox_claude_input_px": [10, 40, 100, 70],
                "rows": [
                    {
                        "row_index": 0,
                        "role": "header",
                        "bbox_claude_input_px": [10, 40, 100, 50],
                        "cells": [
                            {
                                "column_label": "ITEM",
                                "field_id": None,
                                "bbox_claude_input_px": [10, 40, 60, 50],
                            },
                            {
                                "column_label": "QTY",
                                "field_id": None,
                                "bbox_claude_input_px": [60, 40, 100, 50],
                            },
                        ],
                    },
                    {
                        "row_index": 1,
                        "role": "data",
                        "bbox_claude_input_px": [10, 50, 100, 70],
                        "cells": [
                            {
                                "column_label": "ITEM",
                                "field_id": "invoice:/item",
                                "bbox_claude_input_px": [10, 50, 60, 70],
                            },
                            {
                                "column_label": "QTY",
                                "field_id": "invoice:/qty",
                                "bbox_claude_input_px": [60, 50, 100, 70],
                            },
                        ],
                    },
                ],
            }
        ],
    }


def _span(
    *,
    span_id: str,
    reading_order: int,
    role: str,
    text: str,
    bbox: tuple[float, float, float, float],
    table_id: str | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
) -> OcrLayoutSpan:
    return OcrLayoutSpan(
        span_id=span_id,
        page=1,
        reading_order=reading_order,
        role=role,
        text=text,
        bbox=BBox(
            page=1,
            x1=bbox[0],
            y1=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
        ),
        confidence=1.0,
        table_id=table_id,
        row_index=row_index,
        column_index=column_index,
    )


def _perfect_ledger(
    *,
    document_id: str,
    image_sha256: str,
    provider: str = "paddle",
) -> OcrLayoutLedger:
    spans = [
        _span(
            span_id="span-1",
            reading_order=0,
            role="text",
            text="INV-001",
            bbox=(10, 10, 50, 20),
        ),
        _span(
            span_id="span-2",
            reading_order=1,
            role="text",
            text="ITEM QTY",
            bbox=(10, 40, 100, 50),
        ),
        _span(
            span_id="span-3",
            reading_order=2,
            role="table_row",
            text="ITEM | QTY",
            bbox=(10, 40, 100, 50),
            table_id="table-1",
            row_index=0,
        ),
        _span(
            span_id="span-4",
            reading_order=3,
            role="table_cell",
            text="ITEM",
            bbox=(10, 40, 60, 50),
            table_id="table-1",
            row_index=0,
            column_index=0,
        ),
        _span(
            span_id="span-5",
            reading_order=4,
            role="table_cell",
            text="QTY",
            bbox=(60, 40, 100, 50),
            table_id="table-1",
            row_index=0,
            column_index=1,
        ),
        _span(
            span_id="span-6",
            reading_order=5,
            role="text",
            text="A 1",
            bbox=(10, 50, 100, 70),
        ),
        _span(
            span_id="span-7",
            reading_order=6,
            role="table_row",
            text="A | 1",
            bbox=(10, 50, 100, 70),
            table_id="table-1",
            row_index=1,
        ),
        _span(
            span_id="span-8",
            reading_order=7,
            role="table_cell",
            text="A",
            bbox=(10, 50, 60, 70),
            table_id="table-1",
            row_index=1,
            column_index=0,
        ),
        _span(
            span_id="span-9",
            reading_order=8,
            role="table_cell",
            text="1",
            bbox=(60, 50, 100, 70),
            table_id="table-1",
            row_index=1,
            column_index=1,
        ),
    ]
    page = OcrLayoutPage(
        page_number=1,
        image_sha256=image_sha256,
        width_px=200,
        height_px=100,
        spans=spans,
    )
    engine_id = PROVIDER_ENGINE_IDS[provider]  # type: ignore[index]
    ledger_id = stable_ocr_ledger_id(
        document_id=document_id,
        provider_id=provider,
        engine_id=engine_id,
        engine_version="1.0",
        model_id=OCR_MODEL_SET_IDS[provider],
        model_assets_lock_sha256=OCR_MODEL_ASSETS_LOCK_SHA256,
        pages=[page],
    )
    return OcrLayoutLedger(
        ledger_id=ledger_id,
        document_id=document_id,
        provider_id=provider,
        engine_id=engine_id,
        engine_version="1.0",
        model_id=OCR_MODEL_SET_IDS[provider],
        model_assets_lock_sha256=OCR_MODEL_ASSETS_LOCK_SHA256,
        pages=[page],
    )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    source_image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(source_image, format="PNG")
    image_bytes = source_image.read_bytes()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    generated_files: list[dict[str, Any]] = []
    documents: list[dict[str, str]] = []

    def add_document(
        *,
        benchmark_document_id: str,
        case_id: str,
        document_type: str,
        stem: str,
    ) -> None:
        image_relative_path = f"{stem}.claude.png"
        layout_relative_path = f"{stem}.layout-gold.json"
        image_path = rendered / Path(image_relative_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        layout = _layout(image_sha256)
        layout["document_id"] = f"{case_id}:{document_type}"
        layout["case_id"] = case_id
        layout["document_type"] = document_type
        layout["files"]["claude_input_png_relative_path"] = image_relative_path
        layout_path = rendered / Path(layout_relative_path)
        layout_bytes = json.dumps(layout, sort_keys=True).encode("utf-8")
        layout_path.write_bytes(layout_bytes)
        layout_sha256 = hashlib.sha256(layout_bytes).hexdigest()
        generated_files.extend(
            [
                {
                    "relative_path": image_relative_path,
                    "sha256": image_sha256,
                    "byte_size": len(image_bytes),
                },
                {
                    "relative_path": layout_relative_path,
                    "sha256": layout_sha256,
                    "byte_size": len(layout_bytes),
                },
            ]
        )
        documents.append(
            {
                "benchmark_document_id": benchmark_document_id,
                "image_relative_path": image_relative_path,
                "image_sha256": image_sha256,
                "layout_relative_path": layout_relative_path,
                "layout_sha256": layout_sha256,
            }
        )

    add_document(
        benchmark_document_id="trade-triplet-0001-commercial-invoice",
        case_id="trade-triplet-0001",
        document_type="commercial_invoice",
        stem="documents/trade-triplet-0001/commercial_invoice",
    )
    locked_case_ids = list(TEST_LOCKED_CASE_IDS)
    document_types = (
        ("commercial_invoice", "commercial-invoice"),
        ("packing_list", "packing-list"),
        ("bill_of_lading", "bill-of-lading"),
    )
    for case_id in locked_case_ids:
        for document_type, suffix in document_types:
            add_document(
                benchmark_document_id=f"{case_id}-{suffix}",
                case_id=case_id,
                document_type=document_type,
                stem=f"documents/{case_id}/{document_type}",
            )
    for index in range(1, 21):
        case_id = f"co-synthetic-{index:04d}"
        add_document(
            benchmark_document_id=f"{case_id}-certificate-of-origin",
            case_id=case_id,
            document_type="certificate_of_origin",
            stem=f"certificate-of-origin/{case_id}/certificate_of_origin",
        )
    while len(generated_files) < 1280:
        index = len(generated_files)
        generated_files.append(
            {
                "relative_path": f"unselected/filler-{index:04d}.bin",
                "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "byte_size": 0,
            }
        )
    manifest_path = rendered / "manifest.json"
    manifest_path.write_text(
        json.dumps(
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
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    descriptor = {
        "schema_version": "ocr-benchmark-descriptor/v2",
        "benchmark_id": "rendered-trade-documents-ocr-v2",
        "evaluation_version": "evaluation-v2",
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_splits_sha256": CANONICAL_SPLITS_SHA256,
        "release_context": TEST_RELEASE_CONTEXT,
        "input_variant": "claude_input_png",
        "thresholds": LOCKED_THRESHOLDS,
        "critical_identifier_rules": [
            {
                "field_id_marker": "Invoice No.",
                "selector": "identity",
            }
        ],
        "coverage": {
            "smoke_split": "development",
            "smoke_case_id": "trade-triplet-0001",
            "smoke_document_count": 1,
            "locked_split": "locked_test",
            "locked_case_ids": locked_case_ids,
            "locked_case_count": 20,
            "locked_document_types": [name for name, _ in document_types],
            "locked_triplet_document_count": 60,
            "co_document_count": 20,
            "release_gate_document_count": 80,
            "descriptor_document_count": 81,
        },
        "samples": {
            "smoke": ["trade-triplet-0001-commercial-invoice"],
            "locked": [
                f"{case_id}-{suffix}" for case_id in locked_case_ids for _, suffix in document_types
            ],
            "co_regression": [
                f"co-synthetic-{index:04d}-certificate-of-origin" for index in range(1, 21)
            ],
        },
        "documents": documents,
    }
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(
        json.dumps(descriptor, sort_keys=True),
        encoding="utf-8",
    )
    return descriptor_path, rendered, image_sha256


class _PerfectProvider:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.closed = False

    def analyze(
        self,
        *,
        document_id: str,
        pages: list[ImagePageInput],
    ) -> OcrLayoutLedger:
        return _perfect_ledger(
            document_id=document_id,
            image_sha256=pages[0].sha256,
            provider=self.provider,
        )

    def close(self) -> None:
        self.closed = True


class _FailingProvider:
    def analyze(
        self,
        *,
        document_id: str,
        pages: list[ImagePageInput],
    ) -> OcrLayoutLedger:
        del document_id, pages
        raise OcrLayoutProviderError(
            "ocr_backend_inference_failed",
            "safe failure",
        )

    def close(self) -> None:
        return None


def _attestation_arguments(wheel_sha256: str) -> dict[str, Any]:
    environment_locks = {
        "paddle": Path("paddle.requirements.lock.txt"),
        "docling": Path("docling.requirements.lock.txt"),
    }

    def environment_attestor(_executable: Path, lock_path: Path) -> str:
        provider = "paddle" if lock_path == environment_locks["paddle"] else "docling"
        return CANONICAL_ENVIRONMENT_LOCK_SHA256[provider]

    return {
        "repo_root": Path("repo"),
        "authorization_receipt_path": Path("authorization-receipt.json"),
        "source_freeze_verifier": lambda **_kwargs: None,
        "provider_environment_locks": environment_locks,
        "provider_environment_lock_sha256": CANONICAL_ENVIRONMENT_LOCK_SHA256,
        "model_assets_lock_path": Path("model-assets.lock.json"),
        "model_assets_lock_sha256": CANONICAL_MODEL_ASSETS_LOCK_SHA256,
        "wheel_attestor": lambda _executable: wheel_sha256,
        "environment_attestor": environment_attestor,
        "model_assets_attestor": (
            lambda _provider, _lock_path, _engine_id: CANONICAL_MODEL_ASSETS_LOCK_SHA256
        ),
        "runtime_attestor": (
            lambda provider, _executable, _runtime_provider, _model_lock: (
                (100_000_000, 2_000_000_000)
                if provider == "paddle"
                else (200_000_000, 3_000_000_000)
            )
        ),
    }


def _record_digest(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()


def _write_synthetic_wheel_install(tmp_path: Path) -> tuple[Path, Path]:
    site_packages = tmp_path / "site-packages"
    package = site_packages / "customs_ai_agent"
    dist_info = site_packages / "customs_ai_agent-0.3.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    payloads = {
        "customs_ai_agent/__init__.py": b"",
        "customs_ai_agent/module.py": b"VALUE = 1\n",
        "customs_ai_agent-0.3.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: customs-ai-agent\nVersion: 0.3.0\n"
        ),
        "customs_ai_agent-0.3.0.dist-info/direct_url.json": json.dumps(
            {
                "archive_info": {"hashes": {"sha256": "a" * 64}},
                "url": "file:///release/customs_ai_agent-0.3.0-py3-none-any.whl",
            },
            sort_keys=True,
        ).encode("utf-8"),
    }
    for relative_name, payload in payloads.items():
        path = site_packages / Path(relative_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    record_relative = "customs_ai_agent-0.3.0.dist-info/RECORD"
    record_payload = "".join(
        [
            *[
                f"{name},sha256={_record_digest(payload)},{len(payload)}\n"
                for name, payload in sorted(payloads.items())
            ],
            f"{record_relative},,\n",
        ]
    ).encode("utf-8")
    record_path = site_packages / record_relative
    record_path.write_bytes(record_payload)
    return site_packages, record_path


def _run_wheel_attestation_script(site_packages: Path) -> dict[str, Any]:
    bootstrap = (
        f"import sys\nsys.path.insert(0, {str(site_packages)!r})\n{_WHEEL_ATTESTATION_SCRIPT}"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", bootstrap],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return json.loads(completed.stdout.decode("utf-8"))


def test_perfect_ledger_meets_every_locked_metric() -> None:
    image_sha256 = "a" * 64
    score = score_document(
        layout=_layout(image_sha256),
        ledger=_perfect_ledger(
            document_id="fixture:invoice",
            image_sha256=image_sha256,
        ),
        critical_rules=[
            {
                "field_id_marker": "Invoice No.",
                "selector": "identity",
            }
        ],
    )

    assert score["critical_identifier_correct"] == 1
    assert score["critical_identifier_total"] == 1
    assert score["character_edit_distance"] == 0
    assert score["table_rows"].f1 == 1.0
    assert score["table_cells"].f1 == 1.0
    assert score["bboxes"].f1 == 1.0


def test_changed_threshold_is_rejected_before_provider_setup(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["thresholds"]["character_error_rate_max"] = 0.02
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(OcrBenchmarkError) as captured:
        load_benchmark(
            descriptor_path=descriptor_path,
            rendered_root=rendered,
            sample="locked",
        )
    assert captured.value.code == "benchmark_thresholds_changed"


def test_descriptor_builder_rejects_compromised_v2_before_any_access(
    tmp_path: Path,
) -> None:
    access_called = False

    def release_access(*_args: object) -> dict[str, str]:
        nonlocal access_called
        access_called = True
        raise AssertionError("compromised v2 must stop before release access")

    with pytest.raises(
        DescriptorBuildError,
        match="release_holdout_v2_publicly_derivable",
    ):
        build_descriptor(
            splits_path=tmp_path / "must-not-be-read-splits.json",
            rendered_root=tmp_path / "must-not-be-read-rendered",
            manifest_sha256="f" * 64,
            repo_root=tmp_path,
            release_receipt_path=tmp_path / "receipt.json",
            release_access_validator=release_access,
        )
    assert access_called is False


def test_under_scoped_locked_coverage_is_rejected(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["coverage"]["locked_triplet_document_count"] = 3
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(OcrBenchmarkError) as captured:
        load_benchmark(
            descriptor_path=descriptor_path,
            rendered_root=rendered,
            sample="locked",
        )
    assert captured.value.code == "benchmark_coverage_invalid"


def test_only_passing_provider_is_selected(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    providers: list[_PerfectProvider] = []

    def factory(provider: str, _executable: Path) -> Any:
        if provider == "paddle":
            instance = _PerfectProvider(provider)
            providers.append(instance)
            return instance
        return _FailingProvider()

    exit_code, report = evaluate(
        descriptor_path=descriptor_path,
        rendered_root=rendered,
        sample="locked",
        provider_executables={
            "paddle": Path("paddle-python"),
            "docling": Path("docling-python"),
        },
        wheel_sha256="b" * 64,
        provider_factory=factory,
        **_attestation_arguments("b" * 64),
    )

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["selected_provider"] == "paddle"
    assert report["selection_status"] == "SELECTED_ONLY_PASSING_PROVIDER"
    assert report["providers"][0]["status"] == "PASS"
    assert report["providers"][1]["status"] == "FAIL"
    assert report["providers"][0]["installed_agent_wheel_sha256"] == "b" * 64
    assert report["providers"][1]["installed_agent_wheel_sha256"] == "b" * 64
    assert (
        report["providers"][0]["environment_lock_sha256"]
        == CANONICAL_ENVIRONMENT_LOCK_SHA256["paddle"]
    )
    assert report["providers"][0]["model_assets_lock_sha256"] == CANONICAL_MODEL_ASSETS_LOCK_SHA256
    paddle_latencies = sorted(
        document["latency_ns"] for document in report["providers"][0]["documents"]
    )
    assert report["providers"][0]["runtime_metrics"] == {
        "latency_sample_count": 60,
        "latency_p95_ms": paddle_latencies[56] / 1_000_000,
        "peak_memory_bytes": 100_000_000,
        "disk_footprint_bytes": 2_000_000_000,
    }
    assert report["providers"][0]["runtime_error_code"] is None
    scored_metrics = report["providers"][0]["documents"][0]["metrics"]
    assert set(scored_metrics) == {
        "critical_identifier_correct",
        "critical_identifier_total",
        "character_edit_distance",
        "gold_character_count",
        "table_rows",
        "table_cells",
        "bboxes_iou_0_5",
    }
    assert set(scored_metrics["table_rows"]) == {
        "true_positive",
        "false_positive",
        "false_negative",
        "f1",
    }
    assert report["providers"][0]["cleanup_error_code"] is None
    assert "agent_wheel_sha256" not in report
    assert providers[0].closed is True


def test_both_fail_records_metrics_and_selects_no_provider(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)

    exit_code, report = evaluate(
        descriptor_path=descriptor_path,
        rendered_root=rendered,
        sample="locked",
        provider_executables={
            "paddle": Path("paddle-python"),
            "docling": Path("docling-python"),
        },
        wheel_sha256="c" * 64,
        provider_factory=lambda _provider, _executable: _FailingProvider(),
        **_attestation_arguments("c" * 64),
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["selected_provider"] is None
    assert report["selection_status"] == "NO_PROVIDER_PASSED"
    for provider in report["providers"]:
        assert provider["status"] == "FAIL"
        assert provider["metrics"]["crash_count"] == 60
        assert provider["metrics"]["successful_document_count"] == 0
        assert provider["metrics"]["critical_identifier_exact_accuracy"] == 0.0
        assert provider["metrics"]["character_error_rate"] == 1.0
        assert provider["metrics"]["table_rows"]["f1"] == 0.0
        assert provider["metrics"]["table_cells"]["f1"] == 0.0
        assert provider["metrics"]["bboxes_iou_0_5"]["f1"] == 0.0
        failed_document = provider["documents"][0]
        assert failed_document["status"] == "FAIL"
        assert failed_document["metrics"]["critical_identifier_correct"] == 0
        assert (
            failed_document["metrics"]["character_edit_distance"]
            == failed_document["metrics"]["gold_character_count"]
        )
        assert failed_document["metrics"]["table_rows"]["true_positive"] == 0
        assert failed_document["metrics"]["table_rows"]["false_positive"] == 0


def test_report_never_contains_source_text_paths_secrets_or_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    secret = "secret-token-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    def exploding_factory(_provider: str, _executable: Path) -> Any:
        raise RuntimeError(f"{secret} {tmp_path} INV-001")

    _, report = evaluate(
        descriptor_path=descriptor_path,
        rendered_root=rendered,
        sample="locked",
        provider_executables={
            "paddle": tmp_path / "paddle-python",
            "docling": tmp_path / "docling-python",
        },
        wheel_sha256="d" * 64,
        provider_factory=exploding_factory,
        **_attestation_arguments("d" * 64),
    )
    serialized = json.dumps(report, ensure_ascii=False)

    assert secret not in serialized
    assert str(tmp_path) not in serialized
    assert "INV-001" not in serialized
    assert "document.png" not in serialized
    assert "layout-gold" not in serialized
    assert report["selected_provider"] is None


def test_wheel_hash_mismatch_fails_before_provider_setup(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    provider_setup_called = False

    def factory(_provider: str, _executable: Path) -> Any:
        nonlocal provider_setup_called
        provider_setup_called = True
        return _FailingProvider()

    with pytest.raises(OcrBenchmarkError) as captured:
        evaluate(
            descriptor_path=descriptor_path,
            rendered_root=rendered,
            sample="locked",
            provider_executables={
                "paddle": Path("paddle-python"),
                "docling": Path("docling-python"),
            },
            wheel_sha256="e" * 64,
            provider_factory=factory,
            **_attestation_arguments("f" * 64),
        )

    assert captured.value.code == "benchmark_wheel_hash_mismatch"
    assert provider_setup_called is False


def test_source_freeze_failure_stops_before_wheel_or_provider_setup(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    wheel_attestor_called = False
    provider_setup_called = False
    arguments = _attestation_arguments("e" * 64)

    def wheel_attestor(_executable: Path) -> str:
        nonlocal wheel_attestor_called
        wheel_attestor_called = True
        return "e" * 64

    def provider_factory(_provider: str, _executable: Path) -> Any:
        nonlocal provider_setup_called
        provider_setup_called = True
        return _FailingProvider()

    def reject_freeze(**_kwargs: object) -> None:
        raise OcrFreezeVerificationError("ocr_release_freeze_commit_topology_invalid")

    arguments["wheel_attestor"] = wheel_attestor
    arguments["source_freeze_verifier"] = reject_freeze
    with pytest.raises(
        OcrBenchmarkError,
        match="benchmark_release_freeze_commit_topology_invalid",
    ):
        evaluate(
            descriptor_path=descriptor_path,
            rendered_root=rendered,
            sample="locked",
            provider_executables={
                "paddle": Path("paddle-python"),
                "docling": Path("docling-python"),
            },
            wheel_sha256="e" * 64,
            provider_factory=provider_factory,
            **arguments,
        )

    assert wheel_attestor_called is False
    assert provider_setup_called is False


def test_environment_lock_mismatch_fails_before_provider_setup(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    provider_setup_called = False

    def factory(_provider: str, _executable: Path) -> Any:
        nonlocal provider_setup_called
        provider_setup_called = True
        return _FailingProvider()

    arguments = _attestation_arguments("e" * 64)
    arguments["environment_attestor"] = lambda _executable, _lock: "0" * 64

    with pytest.raises(OcrBenchmarkError) as captured:
        evaluate(
            descriptor_path=descriptor_path,
            rendered_root=rendered,
            sample="locked",
            provider_executables={
                "paddle": Path("paddle-python"),
                "docling": Path("docling-python"),
            },
            wheel_sha256="e" * 64,
            provider_factory=factory,
            **arguments,
        )

    assert captured.value.code == "benchmark_environment_lock_hash_mismatch"
    assert provider_setup_called is False


def test_model_assets_lock_mismatch_fails_before_provider_setup(
    tmp_path: Path,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    provider_setup_called = False

    def factory(_provider: str, _executable: Path) -> Any:
        nonlocal provider_setup_called
        provider_setup_called = True
        return _FailingProvider()

    arguments = _attestation_arguments("e" * 64)
    arguments["model_assets_attestor"] = lambda _provider, _lock_path, _engine_id: "0" * 64

    with pytest.raises(OcrBenchmarkError) as captured:
        evaluate(
            descriptor_path=descriptor_path,
            rendered_root=rendered,
            sample="locked",
            provider_executables={
                "paddle": Path("paddle-python"),
                "docling": Path("docling-python"),
            },
            wheel_sha256="e" * 64,
            provider_factory=factory,
            **arguments,
        )

    assert captured.value.code == "benchmark_model_assets_lock_hash_mismatch"
    assert provider_setup_called is False


@pytest.mark.parametrize(
    ("attestor_name", "expected_code"),
    [
        ("wheel_attestor", "benchmark_wheel_attestation_failed"),
        ("environment_attestor", "benchmark_environment_attestation_failed"),
        ("model_assets_attestor", "benchmark_model_assets_attestation_failed"),
    ],
)
def test_unexpected_attestor_exception_is_redacted_before_provider_setup(
    tmp_path: Path,
    attestor_name: str,
    expected_code: str,
) -> None:
    descriptor_path, rendered, _ = _write_fixture(tmp_path)
    provider_setup_called = False
    secret = f"secret-{tmp_path}"

    def factory(_provider: str, _executable: Path) -> Any:
        nonlocal provider_setup_called
        provider_setup_called = True
        return _FailingProvider()

    def exploding_attestor(*_args: object) -> str:
        raise RuntimeError(secret)

    arguments = _attestation_arguments("e" * 64)
    arguments[attestor_name] = exploding_attestor

    with pytest.raises(OcrBenchmarkError) as captured:
        evaluate(
            descriptor_path=descriptor_path,
            rendered_root=rendered,
            sample="locked",
            provider_executables={
                "paddle": Path("paddle-python"),
                "docling": Path("docling-python"),
            },
            wheel_sha256="e" * 64,
            provider_factory=factory,
            **arguments,
        )

    assert captured.value.code == expected_code
    assert secret not in str(captured.value)
    assert provider_setup_called is False


def test_wheel_attestor_accepts_only_safe_exact_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema_version": "customs-ai-agent-wheel-attestation/v1",
        "status": "ok",
        "installed_agent_wheel_sha256": "a" * 64,
        "installed_agent_record_sha256": "b" * 64,
        "verified_executable_file_count": 44,
    }

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[sys.executable],
            returncode=0,
            stdout=json.dumps(payload).encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _installed_agent_wheel_sha256(Path(sys.executable)) == "a" * 64

    payload["unexpected"] = str(Path.home())
    with pytest.raises(OcrBenchmarkError) as captured:
        _installed_agent_wheel_sha256(Path(sys.executable))
    assert captured.value.code == "benchmark_wheel_attestation_failed"


def test_wheel_attestor_preserves_safe_source_install_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema_version": "customs-ai-agent-wheel-attestation/v1",
        "status": "error",
        "error_code": "agent_source_install_rejected",
    }

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[sys.executable],
            returncode=0,
            stdout=json.dumps(payload).encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(OcrBenchmarkError) as captured:
        _installed_agent_wheel_sha256(Path(sys.executable))
    assert captured.value.code == "benchmark_agent_source_install_rejected"


def test_wheel_attestation_script_verifies_exact_record_inventory(
    tmp_path: Path,
) -> None:
    site_packages, record_path = _write_synthetic_wheel_install(tmp_path)

    payload = _run_wheel_attestation_script(site_packages)

    assert payload == {
        "schema_version": "customs-ai-agent-wheel-attestation/v1",
        "status": "ok",
        "installed_agent_wheel_sha256": "a" * 64,
        "installed_agent_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "verified_executable_file_count": 2,
    }


@pytest.mark.parametrize(
    "mutated_payload",
    [
        b"VALUE = 2\n",
        b"VALUE = 'longer than the recorded size'\n",
    ],
)
def test_wheel_attestation_script_rejects_modified_recorded_source(
    tmp_path: Path,
    mutated_payload: bytes,
) -> None:
    site_packages, _ = _write_synthetic_wheel_install(tmp_path)
    (site_packages / "customs_ai_agent" / "module.py").write_bytes(mutated_payload)

    payload = _run_wheel_attestation_script(site_packages)

    assert payload == {
        "schema_version": "customs-ai-agent-wheel-attestation/v1",
        "status": "error",
        "error_code": "agent_record_mismatch",
    }


def test_wheel_attestation_script_rejects_untracked_executable(
    tmp_path: Path,
) -> None:
    site_packages, _ = _write_synthetic_wheel_install(tmp_path)
    (site_packages / "customs_ai_agent" / "rogue.py").write_text(
        "VALUE = 'untracked'\n",
        encoding="utf-8",
    )

    payload = _run_wheel_attestation_script(site_packages)

    assert payload == {
        "schema_version": "customs-ai-agent-wheel-attestation/v1",
        "status": "error",
        "error_code": "agent_untracked_executable",
    }


def test_wheel_attestation_script_rejects_unhashed_record_entry(
    tmp_path: Path,
) -> None:
    site_packages, record_path = _write_synthetic_wheel_install(tmp_path)
    lines = record_path.read_text(encoding="utf-8").splitlines()
    lines = [
        "customs_ai_agent/module.py,," if line.startswith("customs_ai_agent/module.py,") else line
        for line in lines
    ]
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = _run_wheel_attestation_script(site_packages)

    assert payload == {
        "schema_version": "customs-ai-agent-wheel-attestation/v1",
        "status": "error",
        "error_code": "agent_record_invalid",
    }
