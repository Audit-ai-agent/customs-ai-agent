from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    write_runtime_sidecars,
)
from tools.activate_source_attested_synthetic_pdf_preview import (
    activate_source_attested_synthetic_runtime,
)


def _canonical(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _canonical_source_artifact(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _package_version(_name: str) -> str:
    return "test-v1"


def _configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": "pdf_text_layer",
        "extractor_revision": "pdf-text-test-v1",
        "options": {},
    }


def _inventory() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": "pdf_text_layer",
        "packages": [
            {
                "name": "pypdf",
                "version": "test-v1",
                "license_id": "BSD-3-Clause",
                "source_uri": "https://pypi.org/project/pypdf/",
            }
        ],
        "artifacts": [],
    }


def _result(configuration_sha256: str, inventory_sha256: str) -> bytes:
    unsigned: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_basis": "source_attested_synthetic",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "packet_kind": "sealed",
        "packet_receipt_sha256": "a" * 64,
        "evaluation_scope": "synthetic_content_stream_compatibility",
        "layout_truth_kind": "pdf_content_stream_layout_v1",
        "layout_truth_receipt_sha256": "b" * 64,
        "layout_truth_parser_spec_sha256": "c" * 64,
        "development_selection_receipt_sha256": "d" * 64,
        "sealed_run_receipt_sha256": "e" * 64,
        "candidate_bundle_sha256": "f" * 64,
        "engine_kind": "pdf_text_layer",
        "extractor_revision": "pdf-text-test-v1",
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": inventory_sha256,
        "candidate_parser_revision": "synthetic-pdf-visible-layout-normalizer-v3",
        "candidate_execution_receipt_sha256": "1" * 64,
        "passed": True,
        "metrics": {
            "field_exact_match": 0.99,
            "field_exact_match_count": 99,
            "field_exact_match_total": 100,
            "field_exact_match_denominators": [
                {
                    "document_type": "invoice",
                    "document_count": 1,
                    "fields_per_document": 100,
                    "total_fields": 100,
                }
            ],
            "ocr_cer": 0.01,
            "table_cell_f1": 0.98,
        },
    }
    value = {
        **unsigned,
        "evaluation_result_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    return _canonical(value, newline=True)


def test_activation_tool_rejects_missing_local_evidence_chain(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    raw = _result(configuration_sha256, inventory_sha256)
    result_sha256 = json.loads(raw)["evaluation_result_sha256"]
    result_path = root / "source-attested-evaluation-results" / f"{result_sha256}.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(_canonical_source_artifact(json.loads(raw)))

    with pytest.raises(DocumentPreviewRuntimeError) as error:
        activate_source_attested_synthetic_runtime(
            document_preview_root=root,
            evaluation_result_path=result_path,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=inventory_sha256,
            package_version_reader=_package_version,
        )

    assert error.value.code != "document_preview_source_attested_activation_input_invalid"
    assert not (root / "approvals" / "pdf-preview-active.json").exists()


def test_activation_tool_rejects_generic_human_evaluation_directory(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    generic_result = root / "evaluation-results" / ("a" * 64 + ".json")
    generic_result.parent.mkdir()
    generic_result.write_bytes(b"{}\n")

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_activation_input_invalid",
    ):
        activate_source_attested_synthetic_runtime(
            document_preview_root=root,
            evaluation_result_path=generic_result,
            configuration_sha256="b" * 64,
            model_inventory_sha256="c" * 64,
            package_version_reader=_package_version,
        )


def test_activation_tool_rejects_non_hash_named_source_result(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    source_result = root / "source-attested-evaluation-results" / "not-a-result.json"
    source_result.parent.mkdir()
    source_result.write_bytes(_canonical_source_artifact(json.loads(_result("b" * 64, "c" * 64))))

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_activation_input_invalid",
    ):
        activate_source_attested_synthetic_runtime(
            document_preview_root=root,
            evaluation_result_path=source_result,
            configuration_sha256="b" * 64,
            model_inventory_sha256="c" * 64,
            package_version_reader=_package_version,
        )
