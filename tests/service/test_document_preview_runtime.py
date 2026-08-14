from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest
from pydantic import SecretStr

from customs_ai_agent.schemas.document_extraction import (
    DocumentPreviewPageV1,
    DocumentSpanV1,
    PreviewCandidateV1,
    PreviewInputV1,
)
from customs_ai_agent.schemas.document_extraction import DocumentExtractionRequestMetadataV1
from customs_ai_agent.service.document_extraction import DocumentPreviewUnavailable
from customs_ai_agent.service.document_preview_runtime import (
    ApprovedPdfOnlyPreviewEngine,
    DocumentPreviewRuntimeError,
    load_approved_pdf_preview_binding,
    local_files_only,
    parse_sealed_evaluation_result,
    resolve_approved_pdf_preview_engine,
    write_approved_pdf_preview_pointer,
    write_runtime_sidecars,
)
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings


def _canonical(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _root(tmp_path: Path) -> Path:
    return tmp_path / "_local" / "data" / "document-preview"


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


def _evaluation(configuration_sha256: str, inventory_sha256: str) -> bytes:
    unsigned: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "packet_kind": "sealed",
        "packet_receipt_sha256": "a" * 64,
        "gold_receipt_sha256": "b" * 64,
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


def _approved(tmp_path: Path) -> tuple[Path, str]:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_path = root / "staging" / "configuration.json"
    inventory_path = root / "staging" / "inventory.json"
    configuration_path.parent.mkdir()
    configuration_path.write_bytes(_canonical(_configuration(), newline=True))
    inventory_path.write_bytes(_canonical(_inventory(), newline=True))
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    evaluation = _evaluation(configuration_sha256, inventory_sha256)
    approval_sha256, _approval = write_approved_pdf_preview_pointer(
        document_preview_root=root,
        approval_id="human-approval-1",
        evaluation_raw=evaluation,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=inventory_sha256,
        package_version_reader=_package_version,
    )
    return root / "approvals" / "pdf-preview-active.json", approval_sha256


class _FakePdfEngine:
    def __init__(self, revision: str) -> None:
        self.revision = revision
        self.calls = 0

    def extract(self, _preview_input: PreviewInputV1) -> PreviewCandidateV1:
        self.calls += 1
        return PreviewCandidateV1(
            status="preview_ready",
            extractor_revision=self.revision,
            pages=[
                DocumentPreviewPageV1(
                    page_number=1,
                    width=1,
                    height=1,
                    coordinate_unit="pdf_points",
                    spans=[
                        DocumentSpanV1(
                            span_id="page-1-span-1",
                            ordinal=1,
                            reading_order=1,
                            text="LOCAL PDF PREVIEW",
                        )
                    ],
                )
            ],
            warnings=[],
        )


def _pdf_input() -> PreviewInputV1:
    content = b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n%%EOF\n"
    return PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="asset-1",
            extraction_request_id="extract-1",
            source_sha256=hashlib.sha256(content).hexdigest(),
            media_type="application/pdf",
        ),
        byte_size=len(content),
        content=content,
    )


def _jpeg_input() -> PreviewInputV1:
    content = b"jpeg"
    return PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="asset-2",
            extraction_request_id="extract-2",
            source_sha256=hashlib.sha256(content).hexdigest(),
            media_type="image/jpeg",
        ),
        byte_size=len(content),
        content=content,
    )


def test_approved_binding_requires_exact_sidecars_and_pdf_only_runtime(tmp_path: Path) -> None:
    approval_path, approval_sha256 = _approved(tmp_path)
    binding = load_approved_pdf_preview_binding(
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        package_version_reader=_package_version,
    )
    fake = _FakePdfEngine(binding.approval.extractor_revision)
    engine = ApprovedPdfOnlyPreviewEngine(binding=binding, delegate=fake)

    candidate = engine.extract(_pdf_input())

    assert candidate.extractor_revision == "pdf-text-test-v1"
    assert fake.calls == 1
    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_jpeg_input())
    assert fake.calls == 1


def test_runtime_pointer_drift_fails_closed_without_calling_adapter(tmp_path: Path) -> None:
    approval_path, approval_sha256 = _approved(tmp_path)
    binding = load_approved_pdf_preview_binding(
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        package_version_reader=_package_version,
    )
    fake = _FakePdfEngine(binding.approval.extractor_revision)
    engine = ApprovedPdfOnlyPreviewEngine(binding=binding, delegate=fake)
    approval_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_pdf_input())

    assert fake.calls == 0


def test_resolution_is_disabled_for_missing_or_mismatched_approval(tmp_path: Path) -> None:
    approval_path, approval_sha256 = _approved(tmp_path)
    missing = resolve_approved_pdf_preview_engine(
        approval_path=None,
        approval_sha256=None,
        package_version_reader=_package_version,
    )
    mismatch = resolve_approved_pdf_preview_engine(
        approval_path=approval_path,
        approval_sha256="0" * 64,
        package_version_reader=_package_version,
    )

    assert missing.status == "disabled"
    assert mismatch.status == "disabled"
    assert mismatch.reason_code == "document_preview_approval_hash_mismatch"


def test_sealed_evaluation_must_be_canonical_passing_and_bound() -> None:
    raw = _evaluation("c" * 64, "d" * 64)
    parsed = parse_sealed_evaluation_result(raw)

    assert parsed.packet_receipt_sha256 == "a" * 64
    noncanonical = b" " + raw
    with pytest.raises(DocumentPreviewRuntimeError, match="document_preview_evaluation_invalid"):
        parse_sealed_evaluation_result(noncanonical)

    value = json.loads(raw)
    value["metrics"]["ocr_cer"] = 0.02
    unsigned = dict(value)
    unsigned.pop("evaluation_result_sha256")
    value["evaluation_result_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_evaluation_gate_failed"
    ):
        parse_sealed_evaluation_result(_canonical(value, newline=True))


def test_settings_expose_only_paired_local_approval_opt_in(tmp_path: Path) -> None:
    pointer = _root(tmp_path) / "approvals" / "pdf-preview-active.json"
    token = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
    settings = ServiceSettings(
        bearer_token=SecretStr(token),
        evidence_base_url="http://127.0.0.1:18083",
        evidence_service_token=SecretStr(token),
        document_preview_approval_path=pointer,
        document_preview_approval_sha256="c" * 64,
    )

    assert settings.document_preview_approval_path == pointer
    with pytest.raises(
        ServiceConfigurationError, match="customs_agent_document_preview_approval_incomplete"
    ):
        ServiceSettings(
            bearer_token=SecretStr(token),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(token),
            document_preview_approval_path=pointer,
        )
    with pytest.raises(
        ServiceConfigurationError, match="customs_agent_document_preview_approval_path_invalid"
    ):
        ServiceSettings(
            bearer_token=SecretStr(token),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(token),
            document_preview_approval_path=tmp_path / "outside.json",
            document_preview_approval_sha256="c" * 64,
        )


def test_local_adapter_network_guard_blocks_socket_connection() -> None:
    with local_files_only(), pytest.raises(OSError, match="forbids network access"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
