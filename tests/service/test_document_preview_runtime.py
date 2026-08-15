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
    _docling_preview_pages,
    load_approved_pdf_preview_binding,
    local_files_only,
    parse_source_attested_synthetic_pdf_preview_activation,
    parse_source_attested_synthetic_sealed_evaluation_result,
    parse_sealed_evaluation_result,
    resolve_approved_pdf_preview_engine,
    write_approved_pdf_preview_pointer,
    write_runtime_sidecars,
    write_source_attested_synthetic_pdf_preview_pointer,
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


def _source_attested_evaluation(
    configuration_sha256: str,
    inventory_sha256: str,
    *,
    evaluation_scope: str = "synthetic_content_stream_compatibility",
    layout_truth_receipt_sha256: str = "c" * 64,
    layout_truth_parser_spec_sha256: str = "d" * 64,
    development_selection_receipt_sha256: str = "e" * 64,
    sealed_run_receipt_sha256: str = "f" * 64,
) -> bytes:
    unsigned: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_basis": "source_attested_synthetic",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "packet_kind": "sealed",
        "packet_receipt_sha256": "a" * 64,
        "evaluation_scope": evaluation_scope,
        "layout_truth_kind": "pdf_content_stream_layout_v1",
        "layout_truth_receipt_sha256": layout_truth_receipt_sha256,
        "layout_truth_parser_spec_sha256": layout_truth_parser_spec_sha256,
        "development_selection_receipt_sha256": development_selection_receipt_sha256,
        "sealed_run_receipt_sha256": sealed_run_receipt_sha256,
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


def _self_digest(value: dict[str, object], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _write_source_attested_receipt_chain(
    *, root: Path, configuration_sha256: str, inventory_sha256: str
) -> tuple[str, str, str]:
    parser_spec_sha256 = "c" * 64
    sealed_packet_receipt_sha256 = "a" * 64
    development_packet_receipt_sha256 = "b" * 64

    def write_layout_truth(*, split: str, packet_receipt_sha256: str, source_offset: int) -> str:
        records: list[dict[str, object]] = []
        strata = [
            (fixture_kind, document_type)
            for fixture_kind in ("clean", "error")
            for document_type in ("bill_of_lading", "invoice", "packing_list")
        ]
        # The real compiler owns deterministic selection.  The fixtures
        # merely reproduce the fixed shape: 60 development (10/stratum) and
        # 20 sealed (three/stratum plus two deterministic extras).
        per_stratum = 10 if split == "development" else 3
        selected_strata = [stratum for stratum in strata for _ in range(per_stratum)]
        if split == "sealed":
            selected_strata.extend([strata[0], strata[4]])
        for index, (fixture_kind, document_type) in enumerate(selected_strata):
            record: dict[str, object] = {
                "asset_id": f"asset-{split}-{index:02d}",
                "source_sha256": f"{source_offset + index:064x}",
                "fixture_kind": fixture_kind,
                "document_type": document_type,
                "set_id": f"set-{split}-{index:02d}",
                "content_stream_sha256": f"{source_offset + 1000 + index:064x}",
                "fields": [{"field": "value"}],
                "spans": [{"text": "value"}],
                "table_cells": [{"text": "value"}],
            }
            record["source_truth_record_sha256"] = _self_digest(
                record, "source_truth_record_sha256"
            )
            records.append(record)
        layout: dict[str, object] = {
            "schema_version": "1.0",
            "artifact_kind": "source_attested_synthetic_pdf_layout_truth",
            "provenance_kind": "source_attested_synthetic",
            "evaluation_scope": "synthetic_content_stream_compatibility",
            "layout_truth_kind": "pdf_content_stream_layout_v1",
            "compiler_revision": "test-layout-v1",
            "parser_spec_sha256": parser_spec_sha256,
            "dataset_id": "synthetic-pdf",
            "dataset_revision": "synthetic-pdf-r1",
            "dataset_receipt_sha256": "d" * 64,
            "packet_kind": "stratified-v2",
            "packet_split": split,
            "packet_receipt_sha256": packet_receipt_sha256,
            "clean_manifest_sha256": "1" * 64,
            "error_labels_sha256": "2" * 64,
            "source_manifest_mismatch_count": 0,
            "source_manifest_mismatch_receipt_sha256": "3" * 64,
            "records": records,
        }
        layout["layout_truth_receipt_sha256"] = _self_digest(layout, "layout_truth_receipt_sha256")
        layout_sha256 = str(layout["layout_truth_receipt_sha256"])
        layout_path = root / "source-layout-truths" / f"{layout_sha256}.json"
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        layout_path.write_bytes(_canonical(layout, newline=True))
        return layout_sha256

    development_layout_sha256 = write_layout_truth(
        split="development",
        packet_receipt_sha256=development_packet_receipt_sha256,
        source_offset=100,
    )
    layout_sha256 = write_layout_truth(
        split="sealed", packet_receipt_sha256=sealed_packet_receipt_sha256, source_offset=1000
    )

    candidates = [
        {
            "engine_kind": "pdf_text_layer",
            "extractor_revision": "pdf-text-test-v1",
            "configuration_sha256": configuration_sha256,
            "model_inventory_sha256": inventory_sha256,
            "evaluation_result_sha256": "4" * 64,
            "passed": True,
        },
        {
            "engine_kind": "paddleocr",
            "extractor_revision": "paddle-test-v1",
            "configuration_sha256": "5" * 64,
            "model_inventory_sha256": "6" * 64,
            "evaluation_result_sha256": "7" * 64,
            "passed": False,
        },
        {
            "engine_kind": "docling",
            "extractor_revision": "docling-test-v1",
            "configuration_sha256": "8" * 64,
            "model_inventory_sha256": "9" * 64,
            "evaluation_result_sha256": "a" * 64,
            "passed": False,
        },
    ]
    selection: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_development_selection",
        "evaluation_basis": "source_attested_synthetic",
        "evaluation_scope": "synthetic_content_stream_compatibility",
        "layout_truth_kind": "pdf_content_stream_layout_v1",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "development_packet_receipt_sha256": development_packet_receipt_sha256,
        "layout_truth_receipt_sha256": development_layout_sha256,
        "layout_truth_parser_spec_sha256": parser_spec_sha256,
        "candidate_results": candidates,
        "selected_engine_kind": "pdf_text_layer",
        "selected_extractor_revision": "pdf-text-test-v1",
        "selected_configuration_sha256": configuration_sha256,
        "selected_model_inventory_sha256": inventory_sha256,
        "selected_evaluation_result_sha256": "4" * 64,
    }
    selection["development_selection_receipt_sha256"] = _self_digest(
        selection, "development_selection_receipt_sha256"
    )
    selection_sha256 = str(selection["development_selection_receipt_sha256"])
    selection_path = root / "source-attested-development-selections" / f"{selection_sha256}.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_bytes(_canonical(selection, newline=True))

    reservation: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "sealed_synthetic_pdf_candidate_reservation",
        "profile_mode": "offline_synthetic_pdf_evaluation",
        "parser_revision": "synthetic-pdf-label-normalizer-v2",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "packet_kind": "sealed",
        "packet_receipt_sha256": sealed_packet_receipt_sha256,
        "engine_kind": "pdf_text_layer",
        "extractor_revision": "pdf-text-test-v1",
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": inventory_sha256,
    }
    reservation_raw = _canonical(reservation, newline=True)
    reservation_sha256 = hashlib.sha256(reservation_raw).hexdigest()
    reservation_path = (
        root / "candidate-run-reservations" / f"sealed-{sealed_packet_receipt_sha256}.json"
    )
    reservation_path.parent.mkdir(parents=True)
    reservation_path.write_bytes(reservation_raw)

    sealed_run: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_sealed_run",
        "evaluation_basis": "source_attested_synthetic",
        "evaluation_scope": "synthetic_content_stream_compatibility",
        "layout_truth_kind": "pdf_content_stream_layout_v1",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "sealed_packet_receipt_sha256": sealed_packet_receipt_sha256,
        "layout_truth_receipt_sha256": layout_sha256,
        "layout_truth_parser_spec_sha256": parser_spec_sha256,
        "development_selection_receipt_sha256": selection_sha256,
        "selected_engine_kind": "pdf_text_layer",
        "selected_extractor_revision": "pdf-text-test-v1",
        "selected_configuration_sha256": configuration_sha256,
        "selected_model_inventory_sha256": inventory_sha256,
        "selected_development_evaluation_result_sha256": "4" * 64,
        "candidate_bundle_sha256": "d" * 64,
        "candidate_parser_revision": "synthetic-pdf-label-normalizer-v2",
        "sealed_candidate_reservation_sha256": reservation_sha256,
    }
    sealed_run["sealed_run_receipt_sha256"] = _self_digest(sealed_run, "sealed_run_receipt_sha256")
    sealed_run_sha256 = str(sealed_run["sealed_run_receipt_sha256"])
    sealed_run_path = root / "source-attested-sealed-runs" / f"{sealed_run_sha256}.json"
    sealed_run_path.parent.mkdir(parents=True)
    sealed_run_path.write_bytes(_canonical(sealed_run, newline=True))
    return layout_sha256, selection_sha256, sealed_run_sha256


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


def _source_attested_activated(tmp_path: Path) -> tuple[Path, str]:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    layout_sha256, selection_sha256, sealed_run_sha256 = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    evaluation = _source_attested_evaluation(
        configuration_sha256,
        inventory_sha256,
        layout_truth_receipt_sha256=layout_sha256,
        layout_truth_parser_spec_sha256="c" * 64,
        development_selection_receipt_sha256=selection_sha256,
        sealed_run_receipt_sha256=sealed_run_sha256,
    )
    activation_sha256, _activation = write_source_attested_synthetic_pdf_preview_pointer(
        document_preview_root=root,
        evaluation_raw=evaluation,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=inventory_sha256,
        package_version_reader=_package_version,
    )
    return root / "approvals" / "pdf-preview-active.json", activation_sha256


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


def test_source_attested_synthetic_pointer_activates_pdf_only_without_human_gold(
    tmp_path: Path,
) -> None:
    activation_path, activation_sha256 = _source_attested_activated(tmp_path)
    binding = load_approved_pdf_preview_binding(
        approval_path=activation_path,
        approval_sha256=activation_sha256,
        package_version_reader=_package_version,
    )

    assert binding.approval.__class__.__name__ == "SourceAttestedSyntheticPdfPreviewActivationV1"
    assert (
        binding.evaluation.__class__.__name__ == "SourceAttestedSyntheticSealedEvaluationResultV1"
    )
    assert binding.source_attested_receipt_binding is not None
    assert (
        binding.source_attested_receipt_binding.development_layout_truth.packet_split
        == "development"
    )
    assert binding.source_attested_receipt_binding.layout_truth.packet_split == "sealed"
    assert (
        binding.source_attested_receipt_binding.development_layout_truth.receipt_sha256
        != binding.source_attested_receipt_binding.layout_truth.receipt_sha256
    )
    fake = _FakePdfEngine(binding.approval.extractor_revision)
    resolution = resolve_approved_pdf_preview_engine(
        approval_path=activation_path,
        approval_sha256=activation_sha256,
        adapter_factory=lambda _binding: fake,
        package_version_reader=_package_version,
    )

    assert resolution.status == "approved"
    assert resolution.engine.extract(_pdf_input()).extractor_revision == "pdf-text-test-v1"
    with pytest.raises(DocumentPreviewUnavailable):
        resolution.engine.extract(_jpeg_input())
    assert fake.calls == 1


def test_source_attested_parser_rejects_missing_scope_and_mixed_human_provenance() -> None:
    raw = _source_attested_evaluation("c" * 64, "d" * 64)
    assert (
        parse_source_attested_synthetic_sealed_evaluation_result(raw).layout_truth_kind
        == "pdf_content_stream_layout_v1"
    )

    missing_scope = json.loads(raw)
    missing_scope.pop("evaluation_scope")
    unsigned = dict(missing_scope)
    unsigned.pop("evaluation_result_sha256")
    missing_scope["evaluation_result_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_source_attested_evaluation_invalid"
    ):
        parse_source_attested_synthetic_sealed_evaluation_result(
            _canonical(missing_scope, newline=True)
        )

    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_source_attested_evaluation_invalid"
    ):
        parse_source_attested_synthetic_sealed_evaluation_result(
            _source_attested_evaluation(
                "c" * 64,
                "d" * 64,
                evaluation_scope="synthetic_optical_ocr",
            )
        )

    out_of_range = json.loads(raw)
    out_of_range["metrics"] = {
        "field_exact_match": 2.0,
        "ocr_cer": -0.01,
        "table_cell_f1": 2.0,
    }
    unsigned = dict(out_of_range)
    unsigned.pop("evaluation_result_sha256")
    out_of_range["evaluation_result_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_source_attested_evaluation_invalid"
    ):
        parse_source_attested_synthetic_sealed_evaluation_result(
            _canonical(out_of_range, newline=True)
        )

    mixed = json.loads(raw)
    mixed["gold_receipt_sha256"] = "e" * 64
    unsigned = dict(mixed)
    unsigned.pop("evaluation_result_sha256")
    mixed["evaluation_result_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_source_attested_evaluation_invalid"
    ):
        parse_source_attested_synthetic_sealed_evaluation_result(_canonical(mixed, newline=True))


def test_source_attested_pointer_parser_rejects_human_approval_fields(tmp_path: Path) -> None:
    activation_path, _activation_sha256 = _source_attested_activated(tmp_path)
    pointer = json.loads(activation_path.read_bytes())
    pointer["approval_id"] = "human-approval-1"
    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_source_attested_activation_invalid"
    ):
        parse_source_attested_synthetic_pdf_preview_activation(_canonical(pointer, newline=True))


def test_source_attested_writer_rejects_development_selection_without_three_engines(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    layout_sha256, selection_sha256, sealed_run_sha256 = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    selection_path = root / "source-attested-development-selections" / f"{selection_sha256}.json"
    invalid_selection = json.loads(selection_path.read_bytes())
    invalid_selection["candidate_results"][2]["engine_kind"] = "paddleocr"
    invalid_selection["development_selection_receipt_sha256"] = _self_digest(
        invalid_selection, "development_selection_receipt_sha256"
    )
    invalid_selection_sha256 = str(invalid_selection["development_selection_receipt_sha256"])
    (
        root / "source-attested-development-selections" / f"{invalid_selection_sha256}.json"
    ).write_bytes(_canonical(invalid_selection, newline=True))
    evaluation = _source_attested_evaluation(
        configuration_sha256,
        inventory_sha256,
        layout_truth_receipt_sha256=layout_sha256,
        layout_truth_parser_spec_sha256="c" * 64,
        development_selection_receipt_sha256=invalid_selection_sha256,
        sealed_run_receipt_sha256=sealed_run_sha256,
    )

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_development_selection_invalid",
    ):
        write_source_attested_synthetic_pdf_preview_pointer(
            document_preview_root=root,
            evaluation_raw=evaluation,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=inventory_sha256,
            package_version_reader=_package_version,
        )
    assert not (root / "approvals" / "pdf-preview-active.json").exists()


def test_source_attested_development_receipt_tamper_stops_an_already_loaded_preview(
    tmp_path: Path,
) -> None:
    activation_path, activation_sha256 = _source_attested_activated(tmp_path)
    binding = load_approved_pdf_preview_binding(
        approval_path=activation_path,
        approval_sha256=activation_sha256,
        package_version_reader=_package_version,
    )
    fake = _FakePdfEngine(binding.approval.extractor_revision)
    engine = ApprovedPdfOnlyPreviewEngine(binding=binding, delegate=fake)
    root = activation_path.parent.parent
    receipt_path = next(
        path
        for path in (root / "source-layout-truths").glob("*.json")
        if json.loads(path.read_bytes())["packet_split"] == "development"
    )
    receipt_path.write_bytes(b"{}\n")

    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_pdf_input())
    assert fake.calls == 0


def test_source_attested_reservation_tamper_stops_an_already_loaded_preview(tmp_path: Path) -> None:
    activation_path, activation_sha256 = _source_attested_activated(tmp_path)
    binding = load_approved_pdf_preview_binding(
        approval_path=activation_path,
        approval_sha256=activation_sha256,
        package_version_reader=_package_version,
    )
    fake = _FakePdfEngine(binding.approval.extractor_revision)
    engine = ApprovedPdfOnlyPreviewEngine(binding=binding, delegate=fake)
    root = activation_path.parent.parent
    reservation_path = next((root / "candidate-run-reservations").glob("*.json"))
    reservation_path.write_bytes(b"{}\n")

    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_pdf_input())
    assert fake.calls == 0


def test_source_attested_writer_rejects_a_valid_but_mismatched_sealed_run_receipt(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    layout_sha256, selection_sha256, sealed_run_sha256 = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    sealed_run_path = root / "source-attested-sealed-runs" / f"{sealed_run_sha256}.json"
    mismatched_run = json.loads(sealed_run_path.read_bytes())
    mismatched_run["selected_configuration_sha256"] = "f" * 64
    mismatched_run["sealed_run_receipt_sha256"] = _self_digest(
        mismatched_run, "sealed_run_receipt_sha256"
    )
    mismatched_run_sha256 = str(mismatched_run["sealed_run_receipt_sha256"])
    (root / "source-attested-sealed-runs" / f"{mismatched_run_sha256}.json").write_bytes(
        _canonical(mismatched_run, newline=True)
    )
    evaluation = _source_attested_evaluation(
        configuration_sha256,
        inventory_sha256,
        layout_truth_receipt_sha256=layout_sha256,
        layout_truth_parser_spec_sha256="c" * 64,
        development_selection_receipt_sha256=selection_sha256,
        sealed_run_receipt_sha256=mismatched_run_sha256,
    )

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_receipt_binding_mismatch",
    ):
        write_source_attested_synthetic_pdf_preview_pointer(
            document_preview_root=root,
            evaluation_raw=evaluation,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=inventory_sha256,
            package_version_reader=_package_version,
        )
    assert not (root / "approvals" / "pdf-preview-active.json").exists()


def test_source_attested_writer_rejects_manual_human_evaluation(tmp_path: Path) -> None:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )

    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_source_attested_evaluation_invalid"
    ):
        write_source_attested_synthetic_pdf_preview_pointer(
            document_preview_root=root,
            evaluation_raw=_evaluation(configuration_sha256, inventory_sha256),
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=inventory_sha256,
            package_version_reader=_package_version,
        )


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
    # Deliberately low-entropy dummy: it satisfies the service's distinct-character
    # guard without resembling a credential to secret scanners.
    token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBCDEFGHIJKL"
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


def test_docling_markdown_preview_keeps_every_bounded_chunk_without_guessing_pages() -> None:
    markdown = "A" * 8_000 + "B" * 19

    pages = _docling_preview_pages(page_numbers=[1, 2], markdown_text=markdown)

    assert [span.text for span in pages[0].spans] == ["A" * 8_000, "B" * 19]
    assert pages[1].spans == []
