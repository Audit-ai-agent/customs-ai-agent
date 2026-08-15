from __future__ import annotations

import hashlib
import json
from pathlib import Path

from customs_ai_agent.service.document_preview_runtime import write_runtime_sidecars
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
        "engine_kind": "pdf_text_layer",
        "extractor_revision": "pdf-text-test-v1",
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": inventory_sha256,
        "passed": True,
        "metrics": {
            "field_exact_match": 0.99,
            "ocr_cer": 0.01,
            "table_cell_f1": 0.98,
        },
    }
    value = {
        **unsigned,
        "evaluation_result_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    return _canonical(value, newline=True)


def test_activation_tool_uses_only_local_source_attested_result(tmp_path: Path) -> None:
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
    result_path = root / "evaluation-results" / f"{result_sha256}.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_bytes(raw)

    activation_sha256 = activate_source_attested_synthetic_runtime(
        document_preview_root=root,
        evaluation_result_path=result_path,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=inventory_sha256,
        package_version_reader=_package_version,
    )

    pointer = json.loads((root / "approvals" / "pdf-preview-active.json").read_bytes())
    assert activation_sha256 == hashlib.sha256(_canonical(pointer, newline=True)).hexdigest()
    assert pointer["activation_kind"] == "source_attested_synthetic"
    assert pointer["evaluation"]["evaluation_scope"] == "synthetic_content_stream_compatibility"
    assert "approval_id" not in pointer
    assert "gold_receipt_sha256" not in pointer["evaluation"]
