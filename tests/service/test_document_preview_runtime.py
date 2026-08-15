from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import cast

import pytest
from pydantic import SecretStr

import customs_ai_agent.service.document_preview_runtime as runtime_module
from customs_ai_agent.schemas.document_extraction import (
    DocumentPreviewPageV1,
    DocumentSpanV1,
    PreviewCandidateV1,
    PreviewInputV1,
)
from customs_ai_agent.schemas.document_extraction import DocumentExtractionRequestMetadataV1
from customs_ai_agent.service.document_extraction import DocumentPreviewUnavailable
from customs_ai_agent.service.document_preview_runtime import (
    AutoActivatingSourceAttestedPdfPreviewEngine,
    ApprovedPdfOnlyPreviewEngine,
    DocumentPreviewRuntimeError,
    _docling_preview_pages,
    load_approved_pdf_preview_binding,
    load_source_attested_synthetic_receipt_binding,
    local_files_only,
    parse_source_attested_synthetic_pdf_preview_activation,
    parse_source_attested_synthetic_sealed_evaluation_result,
    parse_sealed_evaluation_result,
    resolve_document_preview_engine,
    resolve_approved_pdf_preview_engine,
    write_approved_pdf_preview_pointer,
    write_runtime_sidecars,
    write_source_attested_synthetic_pdf_preview_pointer,
)
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings


@pytest.fixture(autouse=True)
def _isolate_runtime_source_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep old compact runtime-chain fixtures separate from source truth data.

    The source verifier deliberately accepts only the immutable production
    6,000-PDF corpus and canonical 60/20 packets.  This service test module
    uses tiny content-free fixtures to exercise path/hash/pointer failures;
    the source-truth module owns full re-score tests.  Individual runtime
    wiring tests below replace this no-op and assert the loader invokes the
    verifier with every raw artifact.
    """

    monkeypatch.setattr(
        runtime_module,
        "_verify_source_attested_synthetic_replay",
        lambda **_kwargs: None,
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


def _canonical_local_artifact(value: object) -> bytes:
    """Match the source-attested evaluator's readable immutable files."""

    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


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


def _evaluation(
    configuration_sha256: str,
    inventory_sha256: str,
    *,
    engine_kind: str = "pdf_text_layer",
    extractor_revision: str = "pdf-text-test-v1",
) -> bytes:
    unsigned: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "packet_kind": "sealed",
        "packet_receipt_sha256": "a" * 64,
        "gold_receipt_sha256": "b" * 64,
        "engine_kind": engine_kind,
        "extractor_revision": extractor_revision,
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
    packet_receipt_sha256: str = "a" * 64,
    layout_truth_receipt_sha256: str = "c" * 64,
    layout_truth_parser_spec_sha256: str = "d" * 64,
    development_selection_receipt_sha256: str = "e" * 64,
    sealed_run_receipt_sha256: str = "f" * 64,
    candidate_parser_revision: str = "synthetic-pdf-label-normalizer-v2",
    candidate_bundle_sha256: str = "1" * 64,
    candidate_execution_receipt_sha256: str = "2" * 64,
    engine_kind: str = "pdf_text_layer",
    extractor_revision: str = "pdf-text-test-v1",
) -> bytes:
    unsigned: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_basis": "source_attested_synthetic",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "packet_kind": "sealed",
        "packet_receipt_sha256": packet_receipt_sha256,
        "evaluation_scope": evaluation_scope,
        "layout_truth_kind": "pdf_content_stream_layout_v1",
        "layout_truth_receipt_sha256": layout_truth_receipt_sha256,
        "layout_truth_parser_spec_sha256": layout_truth_parser_spec_sha256,
        "development_selection_receipt_sha256": development_selection_receipt_sha256,
        "sealed_run_receipt_sha256": sealed_run_receipt_sha256,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "engine_kind": engine_kind,
        "extractor_revision": extractor_revision,
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": inventory_sha256,
        "candidate_parser_revision": candidate_parser_revision,
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
        "candidate_execution_receipt_sha256": candidate_execution_receipt_sha256,
    }
    value = {
        **unsigned,
        "evaluation_result_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    return _canonical_local_artifact(value)


def _self_digest(value: dict[str, object], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _write_source_attested_evaluation_artifact(*, root: Path, raw: bytes) -> None:
    """Persist the evaluator's required pretty, hash-named source result."""

    value = json.loads(raw)
    result_sha256 = str(value["evaluation_result_sha256"])
    destination = root / "source-attested-evaluation-results" / f"{result_sha256}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_local_artifact(value))


def _rewrite_source_attested_evaluation(*, root: Path, value: dict[str, object]) -> bytes:
    """Re-hash one test-only source result after a content-free mutation."""

    value["evaluation_result_sha256"] = _self_digest(value, "evaluation_result_sha256")
    raw = _canonical_local_artifact(value)
    _write_source_attested_evaluation_artifact(root=root, raw=raw)
    return raw


def _write_source_attested_receipt_chain(
    *,
    root: Path,
    configuration_sha256: str,
    inventory_sha256: str,
    candidate_parser_revision: str = "synthetic-pdf-label-normalizer-v2",
) -> dict[str, str | bytes]:
    """Build a complete local source-attested chain, not hash-shaped placeholders."""

    parser_spec_sha256 = "c" * 64
    error_labels_sha256 = "2" * 64
    strata = [
        (fixture_kind, document_type)
        for fixture_kind in ("clean", "error")
        for document_type in ("bill_of_lading", "invoice", "packing_list")
    ]

    def make_records(*, split: str, source_offset: int) -> list[dict[str, object]]:
        per_stratum = 10 if split == "development" else 3
        selected_strata = [stratum for stratum in strata for _ in range(per_stratum)]
        if split == "sealed":
            selected_strata.extend([strata[0], strata[4]])
        records: list[dict[str, object]] = []
        for index, (fixture_kind, document_type) in enumerate(selected_strata):
            records.append(
                {
                    "asset_id": f"asset-{split}-{index:02d}",
                    "source_sha256": f"{source_offset + index:064x}",
                    "fixture_kind": fixture_kind,
                    "document_type": document_type,
                    "set_id": f"set-{split}-{index:02d}",
                }
            )
        return records

    development_records = make_records(split="development", source_offset=100)
    sealed_records = make_records(split="sealed", source_offset=1000)
    receipt: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "received_as_of": "2026-08-16",
        "assets": [
            {
                "asset_id": record["asset_id"],
                "relative_path": f"fixtures/{record['asset_id']}.pdf",
                "media_type": "application/pdf",
                "byte_size": 1,
                "sha256": record["source_sha256"],
                "triplet_id": None,
                "source_role": "preview_candidate",
            }
            for record in [*development_records, *sealed_records]
        ],
    }
    dataset_receipt_sha256 = hashlib.sha256(_canonical(receipt)).hexdigest()
    dataset_receipt_path = (
        root / "source-attested-dataset-receipts" / f"{dataset_receipt_sha256}.json"
    )
    dataset_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_receipt_path.write_bytes(_canonical_local_artifact(receipt))

    def write_packet(
        *, split: str, records: list[dict[str, object]]
    ) -> tuple[dict[str, object], str]:
        packet: dict[str, object] = {
            "schema_version": "2.0",
            "packet_kind": "stratified-v2",
            "split": split,
            "dataset_id": "synthetic-pdf",
            "dataset_revision": "synthetic-pdf-r1",
            "receipt_sha256": dataset_receipt_sha256,
            "error_labels_sha256": error_labels_sha256,
            "prepared_as_of": "2026-08-16",
            "items": [
                {
                    "asset_id": record["asset_id"],
                    "source_sha256": record["source_sha256"],
                    "media_type": "application/pdf",
                    "fixture_kind": record["fixture_kind"],
                    "document_type": record["document_type"],
                    "set_id": record["set_id"],
                }
                for record in records
            ],
            "required_gates": ["field_exact_match", "ocr_cer", "table_cell_f1"],
            "release_thresholds": {
                "field_exact_match_min": 0.99,
                "ocr_cer_max": 0.01,
                "table_cell_f1_min": 0.98,
            },
        }
        packet["packet_receipt_sha256"] = _self_digest(packet, "packet_receipt_sha256")
        packet_sha256 = str(packet["packet_receipt_sha256"])
        path = root / "source-attested-packets" / f"{packet_sha256}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_local_artifact(packet))
        return packet, packet_sha256

    development_packet, development_packet_sha256 = write_packet(
        split="development", records=development_records
    )
    sealed_packet, sealed_packet_sha256 = write_packet(split="sealed", records=sealed_records)

    def write_layout_truth(
        *, split: str, packet_sha256: str, records: list[dict[str, object]], source_offset: int
    ) -> str:
        layout_records: list[dict[str, object]] = []
        for index, record in enumerate(records):
            layout_record: dict[str, object] = {
                **record,
                "content_stream_sha256": f"{source_offset + index:064x}",
                "fields": [{"field": "value"}],
                "spans": [{"text": "value"}],
                "table_cells": [{"text": "value"}],
            }
            layout_record["source_truth_record_sha256"] = _self_digest(
                layout_record, "source_truth_record_sha256"
            )
            layout_records.append(layout_record)
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
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "packet_kind": "stratified-v2",
            "packet_split": split,
            "packet_receipt_sha256": packet_sha256,
            "clean_manifest_sha256": "1" * 64,
            "error_labels_sha256": error_labels_sha256,
            "source_manifest_mismatch_count": 0,
            "source_manifest_mismatch_receipt_sha256": "3" * 64,
            "records": layout_records,
        }
        layout["layout_truth_receipt_sha256"] = _self_digest(layout, "layout_truth_receipt_sha256")
        layout_sha256 = str(layout["layout_truth_receipt_sha256"])
        path = root / "source-layout-truths" / f"{layout_sha256}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_local_artifact(layout))
        return layout_sha256

    development_layout_sha256 = write_layout_truth(
        split="development",
        packet_sha256=development_packet_sha256,
        records=development_records,
        source_offset=10_000,
    )
    layout_sha256 = write_layout_truth(
        split="sealed",
        packet_sha256=sealed_packet_sha256,
        records=sealed_records,
        source_offset=20_000,
    )

    def write_candidate_bundle(
        *,
        packet: dict[str, object],
        records: list[dict[str, object]],
        engine_kind: str,
        extractor_revision: str,
        candidate_configuration_sha256: str,
        candidate_inventory_sha256: str,
    ) -> str:
        bundle: dict[str, object] = {
            "schema_version": "1.0",
            "artifact_kind": "synthetic_pdf_candidate_output",
            "profile_mode": "offline_synthetic_pdf_evaluation",
            "parser_revision": candidate_parser_revision,
            "dataset_id": "synthetic-pdf",
            "dataset_revision": "synthetic-pdf-r1",
            "packet_receipt_sha256": packet["packet_receipt_sha256"],
            "engine_kind": engine_kind,
            "extractor_revision": extractor_revision,
            "configuration_sha256": candidate_configuration_sha256,
            "model_inventory_sha256": candidate_inventory_sha256,
            "documents": [
                {
                    "asset_id": record["asset_id"],
                    "source_sha256": record["source_sha256"],
                    "fields": [],
                    "spans": [],
                    "table_cells": [],
                }
                for record in records
            ],
        }
        bundle["candidate_bundle_sha256"] = _self_digest(bundle, "candidate_bundle_sha256")
        bundle_sha256 = str(bundle["candidate_bundle_sha256"])
        path = root / "source-attested-candidate-bundles" / f"{bundle_sha256}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_local_artifact(bundle))
        return bundle_sha256

    def write_candidate_execution(
        *,
        packet_sha256: str,
        packet_kind: str,
        layout_truth_sha256: str,
        engine_kind: str,
        extractor_revision: str,
        candidate_configuration_sha256: str,
        candidate_inventory_sha256: str,
        candidate_bundle_sha256: str,
    ) -> str:
        """Create the immutable content-free execution sidecar expected by runtime."""

        candidate_raw = (
            root / "source-attested-candidate-bundles" / f"{candidate_bundle_sha256}.json"
        ).read_bytes()
        execution: dict[str, object] = {
            "schema_version": "1.0",
            "artifact_kind": "source_attested_synthetic_pdf_candidate_execution",
            "profile_mode": "offline_synthetic_pdf_evaluation",
            "dataset_id": "synthetic-pdf",
            "dataset_revision": "synthetic-pdf-r1",
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "packet_kind": packet_kind,
            "packet_receipt_sha256": packet_sha256,
            "layout_truth_receipt_sha256": layout_truth_sha256,
            "layout_truth_parser_spec_sha256": parser_spec_sha256,
            "engine_kind": engine_kind,
            "extractor_revision": extractor_revision,
            "configuration_sha256": candidate_configuration_sha256,
            "model_inventory_sha256": candidate_inventory_sha256,
            "candidate_parser_revision": candidate_parser_revision,
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "candidate_output_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        }
        execution["candidate_execution_receipt_sha256"] = _self_digest(
            execution, "candidate_execution_receipt_sha256"
        )
        execution_sha256 = str(execution["candidate_execution_receipt_sha256"])
        path = root / "source-attested-candidate-executions" / f"{execution_sha256}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_local_artifact(execution))
        return execution_sha256

    development_profiles = [
        (
            "pdf_text_layer",
            "pdf-text-test-v1",
            configuration_sha256,
            inventory_sha256,
            1.0,
            0.0,
            1.0,
        ),
        ("paddleocr", "paddle-test-v1", "5" * 64, "6" * 64, 0.995, 0.005, 0.985),
        ("docling", "docling-test-v1", "7" * 64, "8" * 64, 0.99, 0.01, 0.98),
    ]
    candidates: list[dict[str, object]] = []
    for (
        engine_kind,
        extractor_revision,
        candidate_configuration_sha256,
        candidate_inventory_sha256,
        field_exact_match,
        ocr_cer,
        table_cell_f1,
    ) in development_profiles:
        candidate_bundle_sha256 = write_candidate_bundle(
            packet=development_packet,
            records=development_records,
            engine_kind=engine_kind,
            extractor_revision=extractor_revision,
            candidate_configuration_sha256=candidate_configuration_sha256,
            candidate_inventory_sha256=candidate_inventory_sha256,
        )
        candidate_execution_sha256 = write_candidate_execution(
            packet_sha256=development_packet_sha256,
            packet_kind="development",
            layout_truth_sha256=development_layout_sha256,
            engine_kind=engine_kind,
            extractor_revision=extractor_revision,
            candidate_configuration_sha256=candidate_configuration_sha256,
            candidate_inventory_sha256=candidate_inventory_sha256,
            candidate_bundle_sha256=candidate_bundle_sha256,
        )
        field_exact_match_total = 200
        field_exact_match_count = round(field_exact_match * field_exact_match_total)
        result: dict[str, object] = {
            "schema_version": "1.0",
            "evaluation_basis": "source_attested_synthetic",
            "dataset_id": "synthetic-pdf",
            "dataset_revision": "synthetic-pdf-r1",
            "packet_kind": "development",
            "packet_receipt_sha256": development_packet_sha256,
            "evaluation_scope": "synthetic_content_stream_compatibility",
            "layout_truth_kind": "pdf_content_stream_layout_v1",
            "layout_truth_receipt_sha256": development_layout_sha256,
            "layout_truth_parser_spec_sha256": parser_spec_sha256,
            "development_selection_receipt_sha256": None,
            "sealed_run_receipt_sha256": None,
            "engine_kind": engine_kind,
            "extractor_revision": extractor_revision,
            "configuration_sha256": candidate_configuration_sha256,
            "model_inventory_sha256": candidate_inventory_sha256,
            "candidate_parser_revision": candidate_parser_revision,
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "candidate_execution_receipt_sha256": candidate_execution_sha256,
            "passed": True,
            "metrics": {
                "field_exact_match": field_exact_match,
                "field_exact_match_count": field_exact_match_count,
                "field_exact_match_total": field_exact_match_total,
                "field_exact_match_denominators": [
                    {
                        "document_type": "invoice",
                        "document_count": 10,
                        "fields_per_document": 20,
                        "total_fields": field_exact_match_total,
                    }
                ],
                "ocr_cer": ocr_cer,
                "table_cell_f1": table_cell_f1,
            },
        }
        result["evaluation_result_sha256"] = _self_digest(result, "evaluation_result_sha256")
        result_sha256 = str(result["evaluation_result_sha256"])
        result_path = root / "source-attested-evaluation-results" / f"{result_sha256}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(_canonical_local_artifact(result))
        candidates.append(
            {
                "engine_kind": engine_kind,
                "extractor_revision": extractor_revision,
                "configuration_sha256": candidate_configuration_sha256,
                "model_inventory_sha256": candidate_inventory_sha256,
                "candidate_parser_revision": candidate_parser_revision,
                "evaluation_result_sha256": result_sha256,
                "candidate_bundle_sha256": candidate_bundle_sha256,
                "candidate_execution_receipt_sha256": candidate_execution_sha256,
                "passed": True,
            }
        )

    selected = candidates[0]
    selection: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_development_selection",
        "evaluation_basis": "source_attested_synthetic",
        "evaluation_scope": "synthetic_content_stream_compatibility",
        "layout_truth_kind": "pdf_content_stream_layout_v1",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "development_packet_receipt_sha256": development_packet_sha256,
        "layout_truth_receipt_sha256": development_layout_sha256,
        "layout_truth_parser_spec_sha256": parser_spec_sha256,
        "candidate_parser_revision": candidate_parser_revision,
        "candidate_results": candidates,
        "selected_engine_kind": selected["engine_kind"],
        "selected_extractor_revision": selected["extractor_revision"],
        "selected_configuration_sha256": selected["configuration_sha256"],
        "selected_model_inventory_sha256": selected["model_inventory_sha256"],
        "selected_evaluation_result_sha256": selected["evaluation_result_sha256"],
    }
    selection["development_selection_receipt_sha256"] = _self_digest(
        selection, "development_selection_receipt_sha256"
    )
    selection_sha256 = str(selection["development_selection_receipt_sha256"])
    selection_path = root / "source-attested-development-selections" / f"{selection_sha256}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_bytes(_canonical_local_artifact(selection))

    sealed_candidate_bundle_sha256 = write_candidate_bundle(
        packet=sealed_packet,
        records=sealed_records,
        engine_kind="pdf_text_layer",
        extractor_revision="pdf-text-test-v1",
        candidate_configuration_sha256=configuration_sha256,
        candidate_inventory_sha256=inventory_sha256,
    )
    sealed_candidate_execution_sha256 = write_candidate_execution(
        packet_sha256=sealed_packet_sha256,
        packet_kind="sealed",
        layout_truth_sha256=layout_sha256,
        engine_kind="pdf_text_layer",
        extractor_revision="pdf-text-test-v1",
        candidate_configuration_sha256=configuration_sha256,
        candidate_inventory_sha256=inventory_sha256,
        candidate_bundle_sha256=sealed_candidate_bundle_sha256,
    )
    reservation: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "sealed_synthetic_pdf_candidate_reservation",
        "profile_mode": "offline_synthetic_pdf_evaluation",
        "parser_revision": candidate_parser_revision,
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "packet_kind": "sealed",
        "packet_receipt_sha256": sealed_packet_sha256,
        "engine_kind": "pdf_text_layer",
        "extractor_revision": "pdf-text-test-v1",
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": inventory_sha256,
    }
    reservation_raw = _canonical_local_artifact(reservation)
    reservation_sha256 = hashlib.sha256(reservation_raw).hexdigest()
    reservation_path = root / "candidate-run-reservations" / f"sealed-{sealed_packet_sha256}.json"
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    reservation_path.write_bytes(reservation_raw)
    sealed_run: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_sealed_run",
        "evaluation_basis": "source_attested_synthetic",
        "evaluation_scope": "synthetic_content_stream_compatibility",
        "layout_truth_kind": "pdf_content_stream_layout_v1",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "synthetic-pdf-r1",
        "sealed_packet_receipt_sha256": sealed_packet_sha256,
        "layout_truth_receipt_sha256": layout_sha256,
        "layout_truth_parser_spec_sha256": parser_spec_sha256,
        "development_selection_receipt_sha256": selection_sha256,
        "selected_engine_kind": selected["engine_kind"],
        "selected_extractor_revision": selected["extractor_revision"],
        "selected_configuration_sha256": selected["configuration_sha256"],
        "selected_model_inventory_sha256": selected["model_inventory_sha256"],
        "selected_development_evaluation_result_sha256": selected["evaluation_result_sha256"],
        "candidate_bundle_sha256": sealed_candidate_bundle_sha256,
        "candidate_parser_revision": candidate_parser_revision,
        "candidate_execution_receipt_sha256": sealed_candidate_execution_sha256,
        "sealed_candidate_reservation_sha256": reservation_sha256,
    }
    sealed_run["sealed_run_receipt_sha256"] = _self_digest(sealed_run, "sealed_run_receipt_sha256")
    sealed_run_sha256 = str(sealed_run["sealed_run_receipt_sha256"])
    sealed_run_path = root / "source-attested-sealed-runs" / f"{sealed_run_sha256}.json"
    sealed_run_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_run_path.write_bytes(_canonical_local_artifact(sealed_run))
    evaluation = _source_attested_evaluation(
        configuration_sha256,
        inventory_sha256,
        packet_receipt_sha256=sealed_packet_sha256,
        layout_truth_receipt_sha256=layout_sha256,
        layout_truth_parser_spec_sha256=parser_spec_sha256,
        development_selection_receipt_sha256=selection_sha256,
        sealed_run_receipt_sha256=sealed_run_sha256,
        candidate_bundle_sha256=sealed_candidate_bundle_sha256,
        candidate_parser_revision=candidate_parser_revision,
        candidate_execution_receipt_sha256=sealed_candidate_execution_sha256,
    )
    _write_source_attested_evaluation_artifact(root=root, raw=evaluation)
    return {
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "development_packet_receipt_sha256": development_packet_sha256,
        "sealed_packet_receipt_sha256": sealed_packet_sha256,
        "development_layout_sha256": development_layout_sha256,
        "layout_sha256": layout_sha256,
        "selection_sha256": selection_sha256,
        "sealed_run_sha256": sealed_run_sha256,
        "sealed_candidate_bundle_sha256": sealed_candidate_bundle_sha256,
        "sealed_candidate_execution_sha256": sealed_candidate_execution_sha256,
        "evaluation": evaluation,
    }


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
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    evaluation = chain["evaluation"]
    assert isinstance(evaluation, bytes)
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


def _source_attested_activation_pointer(*, extractor_revision: str) -> bytes:
    return _canonical(
        {
            "schema_version": "1.0",
            "activation_kind": "source_attested_synthetic",
            "status": "activated",
            "media_type": "application/pdf",
            "engine": {
                "kind": "pdf_text_layer",
                "extractor_revision": extractor_revision,
                "configuration_sha256": "a" * 64,
                "model_inventory_sha256": "b" * 64,
            },
            "evaluation": {
                "dataset_id": "synthetic-pdf",
                "dataset_revision": "synthetic-pdf-r1",
                "packet_kind": "sealed",
                "packet_receipt_sha256": "c" * 64,
                "evaluation_scope": "synthetic_content_stream_compatibility",
                "layout_truth_kind": "pdf_content_stream_layout_v1",
                "layout_truth_receipt_sha256": "d" * 64,
                "layout_truth_parser_spec_sha256": "e" * 64,
                "development_selection_receipt_sha256": "f" * 64,
                "sealed_run_receipt_sha256": "1" * 64,
                "evaluation_basis": "source_attested_synthetic",
                "evaluation_result_sha256": "2" * 64,
            },
        },
        newline=True,
    )


def _replace_pointer(path: Path, raw: bytes) -> None:
    temporary = path.parent / f".{path.name}.replacement"
    temporary.write_bytes(raw)
    temporary.replace(path)


class _DynamicFakeBinding:
    def __init__(self, approval: object) -> None:
        self.approval = approval
        self.verify_calls = 0

    def verify_runtime_state(self) -> None:
        self.verify_calls += 1


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


def test_runtime_rejects_an_added_file_in_the_active_docling_model_directory(
    tmp_path: Path,
) -> None:
    """A serving binding rechecks the exact inventory set before each preview."""

    root = _root(tmp_path)
    model_path = root / "models" / "docling" / "model.bin"
    model_path.parent.mkdir(parents=True)
    model_bytes = b"docling-test-model"
    model_path.write_bytes(model_bytes)
    configuration = {
        "schema_version": "1.0",
        "engine_kind": "docling",
        "extractor_revision": "docling-test-v1",
        "options": {"artifacts_directory": "models/docling", "ocr_enabled": True},
    }
    inventory = {
        "schema_version": "1.0",
        "engine_kind": "docling",
        "packages": [
            {
                "name": "pypdf",
                "version": "test-v1",
                "license_id": "BSD-3-Clause",
                "source_uri": "https://pypi.org/project/pypdf/",
            },
            {
                "name": "docling",
                "version": "test-v1",
                "license_id": "MIT",
                "source_uri": "https://pypi.org/project/docling/",
            },
        ],
        "artifacts": [
            {
                "artifact_id": "docling-model",
                "relative_path": "models/docling/model.bin",
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "byte_size": len(model_bytes),
                "license_id": "MIT",
                "source_uri": "https://example.test/docling/model",
            }
        ],
    }
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=configuration,
        inventory_value=inventory,
        package_version_reader=_package_version,
    )
    approval_sha256, _approval = write_approved_pdf_preview_pointer(
        document_preview_root=root,
        approval_id="human-approval-docling",
        evaluation_raw=_evaluation(
            configuration_sha256,
            inventory_sha256,
            engine_kind="docling",
            extractor_revision="docling-test-v1",
        ),
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=inventory_sha256,
        package_version_reader=_package_version,
    )
    binding = load_approved_pdf_preview_binding(
        approval_path=root / "approvals" / "pdf-preview-active.json",
        approval_sha256=approval_sha256,
        package_version_reader=_package_version,
    )

    (root / "models" / "docling" / "untracked-model.bin").write_bytes(b"untracked")

    with pytest.raises(DocumentPreviewRuntimeError, match="document_preview_model_runtime_changed"):
        binding.verify_runtime_state()


def test_source_attested_auto_engine_activates_then_clears_cache_on_bad_pointer(
    tmp_path: Path,
) -> None:
    """A running opt-in engine never needs an environment SHA update or restart."""

    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    evaluation = chain["evaluation"]
    assert isinstance(evaluation, bytes)
    activation_path = root / "approvals" / "pdf-preview-active.json"
    delegates: list[_FakePdfEngine] = []

    def adapter_factory(binding: object) -> _FakePdfEngine:
        approval = getattr(binding, "approval")
        delegate = _FakePdfEngine(getattr(approval, "extractor_revision"))
        delegates.append(delegate)
        return delegate

    engine = AutoActivatingSourceAttestedPdfPreviewEngine(
        approval_path=activation_path,
        adapter_factory=adapter_factory,
        package_version_reader=_package_version,
    )
    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_pdf_input())
    assert delegates == []

    write_source_attested_synthetic_pdf_preview_pointer(
        document_preview_root=root,
        evaluation_raw=evaluation,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=inventory_sha256,
        package_version_reader=_package_version,
    )
    assert engine.extract(_pdf_input()).extractor_revision == "pdf-text-test-v1"
    assert len(delegates) == 1
    assert delegates[0].calls == 1

    _replace_pointer(activation_path, b"{}\n")
    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_pdf_input())
    assert len(delegates) == 1
    assert delegates[0].calls == 1

    # A valid atomic replacement can recover, but must re-load rather than
    # retaining the instance that preceded the invalid pointer.
    write_source_attested_synthetic_pdf_preview_pointer(
        document_preview_root=root,
        evaluation_raw=evaluation,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=inventory_sha256,
        package_version_reader=_package_version,
    )
    assert engine.extract(_pdf_input()).extractor_revision == "pdf-text-test-v1"
    assert len(delegates) == 2
    assert delegates[1].calls == 1

    # A human-approved pointer is structurally valid for the preserved static
    # mode, but this dynamic source-attested mode must reject it before a
    # cached delegate can be called.
    write_approved_pdf_preview_pointer(
        document_preview_root=root,
        approval_id="human-approval-1",
        evaluation_raw=_evaluation(configuration_sha256, inventory_sha256),
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=inventory_sha256,
        package_version_reader=_package_version,
    )
    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_pdf_input())
    assert len(delegates) == 2
    assert delegates[1].calls == 1


def test_source_attested_auto_engine_validates_hot_swap_and_mid_load_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a fully stable replacement pointer reaches a new delegate."""

    activation_path = _root(tmp_path) / "approvals" / "pdf-preview-active.json"
    activation_path.parent.mkdir(parents=True)
    first = _source_attested_activation_pointer(extractor_revision="auto-first-v1")
    second = _source_attested_activation_pointer(extractor_revision="auto-second-v1")
    activation_path.write_bytes(first)
    loader_calls: list[str] = []
    delegates: list[_FakePdfEngine] = []
    mutate_during_load = False

    def loader(**_kwargs: object) -> _DynamicFakeBinding:
        nonlocal mutate_during_load
        raw = activation_path.read_bytes()
        approval = parse_source_attested_synthetic_pdf_preview_activation(raw)
        loader_calls.append(approval.extractor_revision)
        if mutate_during_load:
            _replace_pointer(activation_path, second)
        return _DynamicFakeBinding(approval)

    def adapter_factory(binding: object) -> _FakePdfEngine:
        approval = getattr(binding, "approval")
        delegate = _FakePdfEngine(getattr(approval, "extractor_revision"))
        delegates.append(delegate)
        return delegate

    monkeypatch.setattr(runtime_module, "load_approved_pdf_preview_binding", loader)
    engine = AutoActivatingSourceAttestedPdfPreviewEngine(
        approval_path=activation_path,
        adapter_factory=adapter_factory,
        package_version_reader=_package_version,
    )

    assert engine.extract(_pdf_input()).extractor_revision == "auto-first-v1"
    _replace_pointer(activation_path, second)
    assert engine.extract(_pdf_input()).extractor_revision == "auto-second-v1"
    assert loader_calls == ["auto-first-v1", "auto-second-v1"]
    assert [delegate.calls for delegate in delegates] == [1, 1]

    _replace_pointer(activation_path, first)
    mutate_during_load = True
    with pytest.raises(DocumentPreviewUnavailable):
        engine.extract(_pdf_input())
    # The confirmation read occurs before adapter construction, so a pointer
    # changed while loading cannot even create a third delegate.
    assert loader_calls == ["auto-first-v1", "auto-second-v1", "auto-first-v1"]
    assert [delegate.calls for delegate in delegates] == [1, 1]


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
        "field_exact_match_count": 200,
        "field_exact_match_total": 100,
        "field_exact_match_denominators": [
            {
                "document_type": "invoice",
                "document_count": 1,
                "fields_per_document": 100,
                "total_fields": 100,
            }
        ],
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
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    layout_sha256 = str(chain["layout_sha256"])
    selection_sha256 = str(chain["selection_sha256"])
    sealed_run_sha256 = str(chain["sealed_run_sha256"])
    selection_path = root / "source-attested-development-selections" / f"{selection_sha256}.json"
    invalid_selection = json.loads(selection_path.read_bytes())
    invalid_selection["candidate_results"][2]["engine_kind"] = "paddleocr"
    invalid_selection["development_selection_receipt_sha256"] = _self_digest(
        invalid_selection, "development_selection_receipt_sha256"
    )
    invalid_selection_sha256 = str(invalid_selection["development_selection_receipt_sha256"])
    (
        root / "source-attested-development-selections" / f"{invalid_selection_sha256}.json"
    ).write_bytes(_canonical_local_artifact(invalid_selection))
    evaluation = _source_attested_evaluation(
        configuration_sha256,
        inventory_sha256,
        layout_truth_receipt_sha256=layout_sha256,
        layout_truth_parser_spec_sha256="c" * 64,
        development_selection_receipt_sha256=invalid_selection_sha256,
        sealed_run_receipt_sha256=sealed_run_sha256,
        packet_receipt_sha256=str(chain["sealed_packet_receipt_sha256"]),
        candidate_bundle_sha256=str(chain["sealed_candidate_bundle_sha256"]),
        candidate_execution_receipt_sha256=str(chain["sealed_candidate_execution_sha256"]),
    )
    _write_source_attested_evaluation_artifact(root=root, raw=evaluation)

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
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    layout_sha256 = str(chain["layout_sha256"])
    selection_sha256 = str(chain["selection_sha256"])
    sealed_run_sha256 = str(chain["sealed_run_sha256"])
    sealed_run_path = root / "source-attested-sealed-runs" / f"{sealed_run_sha256}.json"
    mismatched_run = json.loads(sealed_run_path.read_bytes())
    mismatched_run["selected_configuration_sha256"] = "f" * 64
    mismatched_run["sealed_run_receipt_sha256"] = _self_digest(
        mismatched_run, "sealed_run_receipt_sha256"
    )
    mismatched_run_sha256 = str(mismatched_run["sealed_run_receipt_sha256"])
    (root / "source-attested-sealed-runs" / f"{mismatched_run_sha256}.json").write_bytes(
        _canonical_local_artifact(mismatched_run)
    )
    evaluation = _source_attested_evaluation(
        configuration_sha256,
        inventory_sha256,
        layout_truth_receipt_sha256=layout_sha256,
        layout_truth_parser_spec_sha256="c" * 64,
        development_selection_receipt_sha256=selection_sha256,
        sealed_run_receipt_sha256=mismatched_run_sha256,
        packet_receipt_sha256=str(chain["sealed_packet_receipt_sha256"]),
        candidate_bundle_sha256=str(chain["sealed_candidate_bundle_sha256"]),
        candidate_execution_receipt_sha256=str(chain["sealed_candidate_execution_sha256"]),
    )
    _write_source_attested_evaluation_artifact(root=root, raw=evaluation)

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


def test_source_attested_writer_rejects_missing_hash_named_dataset_receipt(tmp_path: Path) -> None:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    dataset_path = (
        root / "source-attested-dataset-receipts" / f"{chain['dataset_receipt_sha256']}.json"
    )
    dataset_path.unlink()
    evaluation = chain["evaluation"]
    assert isinstance(evaluation, bytes)

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_dataset_receipt_invalid",
    ):
        write_source_attested_synthetic_pdf_preview_pointer(
            document_preview_root=root,
            evaluation_raw=evaluation,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=inventory_sha256,
            package_version_reader=_package_version,
        )
    assert not (root / "approvals" / "pdf-preview-active.json").exists()


@pytest.mark.parametrize(
    ("artifact_kind", "expected_code"),
    [
        ("dataset", "document_preview_source_attested_dataset_receipt_invalid"),
        ("development_result", "document_preview_source_attested_development_evaluation_invalid"),
        ("sealed_candidate", "document_preview_source_attested_candidate_bundle_invalid"),
    ],
)
def test_source_attested_writer_rejects_tampered_actual_evidence_artifact(
    tmp_path: Path, artifact_kind: str, expected_code: str
) -> None:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    if artifact_kind == "dataset":
        target = (
            root / "source-attested-dataset-receipts" / f"{chain['dataset_receipt_sha256']}.json"
        )
    elif artifact_kind == "development_result":
        selection_path = (
            root / "source-attested-development-selections" / f"{chain['selection_sha256']}.json"
        )
        selection = json.loads(selection_path.read_bytes())
        development_result_sha256 = selection["candidate_results"][0]["evaluation_result_sha256"]
        target = root / "source-attested-evaluation-results" / f"{development_result_sha256}.json"
    else:
        target = (
            root
            / "source-attested-candidate-bundles"
            / f"{chain['sealed_candidate_bundle_sha256']}.json"
        )
    target.write_bytes(b"{}\n")
    evaluation = chain["evaluation"]
    assert isinstance(evaluation, bytes)

    with pytest.raises(DocumentPreviewRuntimeError, match=expected_code):
        write_source_attested_synthetic_pdf_preview_pointer(
            document_preview_root=root,
            evaluation_raw=evaluation,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=inventory_sha256,
            package_version_reader=_package_version,
        )
    assert not (root / "approvals" / "pdf-preview-active.json").exists()


def test_source_attested_loader_recomputes_development_selection_rank(tmp_path: Path) -> None:
    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    selection_path = (
        root / "source-attested-development-selections" / f"{chain['selection_sha256']}.json"
    )
    forged_selection = json.loads(selection_path.read_bytes())
    lower_ranked = forged_selection["candidate_results"][1]
    forged_selection["selected_engine_kind"] = lower_ranked["engine_kind"]
    forged_selection["selected_extractor_revision"] = lower_ranked["extractor_revision"]
    forged_selection["selected_configuration_sha256"] = lower_ranked["configuration_sha256"]
    forged_selection["selected_model_inventory_sha256"] = lower_ranked["model_inventory_sha256"]
    forged_selection["selected_evaluation_result_sha256"] = lower_ranked["evaluation_result_sha256"]
    forged_selection["development_selection_receipt_sha256"] = _self_digest(
        forged_selection, "development_selection_receipt_sha256"
    )
    forged_selection_sha256 = str(forged_selection["development_selection_receipt_sha256"])
    (
        root / "source-attested-development-selections" / f"{forged_selection_sha256}.json"
    ).write_bytes(_canonical_local_artifact(forged_selection))

    sealed_run_path = root / "source-attested-sealed-runs" / f"{chain['sealed_run_sha256']}.json"
    forged_run = json.loads(sealed_run_path.read_bytes())
    forged_run["development_selection_receipt_sha256"] = forged_selection_sha256
    forged_run["selected_engine_kind"] = lower_ranked["engine_kind"]
    forged_run["selected_extractor_revision"] = lower_ranked["extractor_revision"]
    forged_run["selected_configuration_sha256"] = lower_ranked["configuration_sha256"]
    forged_run["selected_model_inventory_sha256"] = lower_ranked["model_inventory_sha256"]
    forged_run["selected_development_evaluation_result_sha256"] = lower_ranked[
        "evaluation_result_sha256"
    ]
    forged_run["sealed_run_receipt_sha256"] = _self_digest(forged_run, "sealed_run_receipt_sha256")
    forged_run_sha256 = str(forged_run["sealed_run_receipt_sha256"])
    (root / "source-attested-sealed-runs" / f"{forged_run_sha256}.json").write_bytes(
        _canonical_local_artifact(forged_run)
    )
    forged_evaluation = _source_attested_evaluation(
        str(lower_ranked["configuration_sha256"]),
        str(lower_ranked["model_inventory_sha256"]),
        engine_kind=str(lower_ranked["engine_kind"]),
        extractor_revision=str(lower_ranked["extractor_revision"]),
        packet_receipt_sha256=str(chain["sealed_packet_receipt_sha256"]),
        layout_truth_receipt_sha256=str(chain["layout_sha256"]),
        layout_truth_parser_spec_sha256="c" * 64,
        development_selection_receipt_sha256=forged_selection_sha256,
        sealed_run_receipt_sha256=forged_run_sha256,
        candidate_bundle_sha256=str(chain["sealed_candidate_bundle_sha256"]),
        candidate_execution_receipt_sha256=str(chain["sealed_candidate_execution_sha256"]),
    )
    _write_source_attested_evaluation_artifact(root=root, raw=forged_evaluation)

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_development_selection_mismatch",
    ):
        load_source_attested_synthetic_receipt_binding(
            root=root,
            evaluation=parse_source_attested_synthetic_sealed_evaluation_result(forged_evaluation),
        )


def test_source_attested_loader_rejects_a_rehashed_candidate_with_incomplete_source_coverage(
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
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    candidate_path = (
        root
        / "source-attested-candidate-bundles"
        / f"{chain['sealed_candidate_bundle_sha256']}.json"
    )
    incomplete_candidate = json.loads(candidate_path.read_bytes())
    incomplete_candidate["documents"] = incomplete_candidate["documents"][:-1]
    incomplete_candidate["candidate_bundle_sha256"] = _self_digest(
        incomplete_candidate, "candidate_bundle_sha256"
    )
    incomplete_candidate_sha256 = str(incomplete_candidate["candidate_bundle_sha256"])
    (
        root / "source-attested-candidate-bundles" / f"{incomplete_candidate_sha256}.json"
    ).write_bytes(_canonical_local_artifact(incomplete_candidate))

    sealed_run_path = root / "source-attested-sealed-runs" / f"{chain['sealed_run_sha256']}.json"
    rewritten_run = json.loads(sealed_run_path.read_bytes())
    rewritten_run["candidate_bundle_sha256"] = incomplete_candidate_sha256
    rewritten_run["sealed_run_receipt_sha256"] = _self_digest(
        rewritten_run, "sealed_run_receipt_sha256"
    )
    rewritten_run_sha256 = str(rewritten_run["sealed_run_receipt_sha256"])
    (root / "source-attested-sealed-runs" / f"{rewritten_run_sha256}.json").write_bytes(
        _canonical_local_artifact(rewritten_run)
    )
    rewritten_evaluation = _source_attested_evaluation(
        configuration_sha256,
        inventory_sha256,
        packet_receipt_sha256=str(chain["sealed_packet_receipt_sha256"]),
        layout_truth_receipt_sha256=str(chain["layout_sha256"]),
        layout_truth_parser_spec_sha256="c" * 64,
        development_selection_receipt_sha256=str(chain["selection_sha256"]),
        sealed_run_receipt_sha256=rewritten_run_sha256,
        candidate_bundle_sha256=incomplete_candidate_sha256,
        candidate_execution_receipt_sha256=str(chain["sealed_candidate_execution_sha256"]),
    )
    _write_source_attested_evaluation_artifact(root=root, raw=rewritten_evaluation)

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_candidate_bundle_binding_mismatch",
    ):
        load_source_attested_synthetic_receipt_binding(
            root=root,
            evaluation=parse_source_attested_synthetic_sealed_evaluation_result(
                rewritten_evaluation
            ),
        )


def test_source_attested_loader_rejects_an_altered_raw_candidate_bundle(tmp_path: Path) -> None:
    """A changed candidate byte cannot pass the hash-named canonical boundary."""

    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    candidate_path = (
        root
        / "source-attested-candidate-bundles"
        / f"{chain['sealed_candidate_bundle_sha256']}.json"
    )
    candidate_path.write_bytes(b"{}\n")
    evaluation = chain["evaluation"]
    assert isinstance(evaluation, bytes)

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_candidate_bundle_invalid",
    ):
        load_source_attested_synthetic_receipt_binding(
            root=root,
            evaluation=parse_source_attested_synthetic_sealed_evaluation_result(evaluation),
        )


def test_source_attested_replay_rejects_a_rehashed_result_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared pointer loader invokes replay after all receipt readers bind."""

    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    result = json.loads(chain["evaluation"])
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    metrics["field_exact_match"] = 1.0
    metrics["field_exact_match_count"] = 100
    metrics["field_exact_match_total"] = 100
    metrics["field_exact_match_denominators"] = [
        {
            "document_type": "invoice",
            "document_count": 1,
            "fields_per_document": 100,
            "total_fields": 100,
        }
    ]
    rewritten = _rewrite_source_attested_evaluation(root=root, value=result)
    replay_calls: list[dict[str, object]] = []

    def reject_replayed_mismatch(**kwargs: object) -> None:
        replay_calls.append(kwargs)
        raise DocumentPreviewRuntimeError("document_preview_source_attested_replay_mismatch")

    monkeypatch.setattr(
        runtime_module, "_verify_source_attested_synthetic_replay", reject_replayed_mismatch
    )
    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_replay_mismatch",
    ):
        write_source_attested_synthetic_pdf_preview_pointer(
            document_preview_root=root,
            evaluation_raw=rewritten,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=inventory_sha256,
            package_version_reader=_package_version,
        )
    assert len(replay_calls) == 1
    assert len(cast(tuple[bytes, ...], replay_calls[0]["development_candidate_raws"])) == 3
    assert isinstance(replay_calls[0]["sealed_candidate_raw"], bytes)
    assert not (root / "approvals" / "pdf-preview-active.json").exists()


def test_source_attested_loader_rejects_execution_receipt_mismatch(tmp_path: Path) -> None:
    """A sealed result may not substitute a development execution sidecar."""

    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    selection = json.loads(
        (
            root / "source-attested-development-selections" / f"{chain['selection_sha256']}.json"
        ).read_bytes()
    )
    development_execution = selection["candidate_results"][0]["candidate_execution_receipt_sha256"]
    result = json.loads(chain["evaluation"])
    result["candidate_execution_receipt_sha256"] = development_execution
    rewritten = _rewrite_source_attested_evaluation(root=root, value=result)

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_receipt_binding_mismatch",
    ):
        load_source_attested_synthetic_receipt_binding(
            root=root,
            evaluation=parse_source_attested_synthetic_sealed_evaluation_result(rewritten),
        )


def test_source_attested_loader_accepts_one_consistent_visible_layout_v3_chain(
    tmp_path: Path,
) -> None:
    """The new v3 parser is accepted only when every immutable binding agrees."""

    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
        candidate_parser_revision="synthetic-pdf-visible-layout-normalizer-v3",
    )
    evaluation = chain["evaluation"]
    assert isinstance(evaluation, bytes)

    binding = load_source_attested_synthetic_receipt_binding(
        root=root,
        evaluation=parse_source_attested_synthetic_sealed_evaluation_result(evaluation),
    )

    assert binding.development_selection.candidate_parser_revision == (
        "synthetic-pdf-visible-layout-normalizer-v3"
    )
    assert binding.sealed_run.candidate_parser_revision == (
        "synthetic-pdf-visible-layout-normalizer-v3"
    )


def test_source_attested_loader_rejects_candidate_parser_revision_mismatch(tmp_path: Path) -> None:
    """A pointer is pinned to the exact candidate parser revision, including v3."""

    root = _root(tmp_path)
    root.mkdir(parents=True)
    configuration_sha256, inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=_configuration(),
        inventory_value=_inventory(),
        package_version_reader=_package_version,
    )
    chain = _write_source_attested_receipt_chain(
        root=root,
        configuration_sha256=configuration_sha256,
        inventory_sha256=inventory_sha256,
    )
    result = json.loads(chain["evaluation"])
    result["candidate_parser_revision"] = "synthetic-pdf-visible-layout-normalizer-v3"
    rewritten = _rewrite_source_attested_evaluation(root=root, value=result)

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_source_attested_receipt_binding_mismatch",
    ):
        load_source_attested_synthetic_receipt_binding(
            root=root,
            evaluation=parse_source_attested_synthetic_sealed_evaluation_result(rewritten),
        )


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


def test_settings_keep_static_pin_and_allow_only_explicit_source_auto_mode(tmp_path: Path) -> None:
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
    assert settings.document_preview_auto_activation_mode is None
    automatic = ServiceSettings(
        bearer_token=SecretStr(token),
        evidence_base_url="http://127.0.0.1:18083",
        evidence_service_token=SecretStr(token),
        document_preview_approval_path=pointer,
        document_preview_auto_activation_mode="source-attested-synthetic",
    )
    assert automatic.document_preview_approval_sha256 is None
    assert automatic.document_preview_auto_activation_mode == "source-attested-synthetic"
    resolution = resolve_document_preview_engine(
        approval_path=pointer,
        approval_sha256=None,
        source_attested_auto_activation=True,
        package_version_reader=_package_version,
    )
    assert resolution.status == "auto-pending"
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
    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_document_preview_auto_activation_incomplete",
    ):
        ServiceSettings(
            bearer_token=SecretStr(token),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(token),
            document_preview_auto_activation_mode="source-attested-synthetic",
        )
    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_document_preview_auto_activation_incomplete",
    ):
        ServiceSettings(
            bearer_token=SecretStr(token),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(token),
            document_preview_approval_path=pointer,
            document_preview_approval_sha256="c" * 64,
            document_preview_auto_activation_mode="source-attested-synthetic",
        )


def test_settings_environment_requires_xor_between_static_hash_and_auto_mode(
    tmp_path: Path,
) -> None:
    pointer = _root(tmp_path) / "approvals" / "pdf-preview-active.json"
    token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaBCDEFGHIJKL"
    environment = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": token,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
        "EVIDENCE_SERVICE_TOKEN": token,
        "CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_PATH": str(pointer),
        "CUSTOMS_AGENT_DOCUMENT_PREVIEW_AUTO_ACTIVATION": "source-attested-synthetic",
    }
    automatic = ServiceSettings.from_environment(environment)
    assert automatic.document_preview_auto_activation_mode == "source-attested-synthetic"
    assert automatic.document_preview_approval_sha256 is None

    with_hash = {
        **environment,
        "CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_SHA256": "c" * 64,
    }
    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_document_preview_auto_activation_incomplete",
    ):
        ServiceSettings.from_environment(with_hash)


def test_local_adapter_network_guard_blocks_socket_connection() -> None:
    with local_files_only(), pytest.raises(OSError, match="forbids network access"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)


def test_docling_markdown_preview_keeps_every_bounded_chunk_without_guessing_pages() -> None:
    markdown = "A" * 8_000 + "B" * 19

    pages = _docling_preview_pages(page_numbers=[1, 2], markdown_text=markdown)

    assert [span.text for span in pages[0].spans] == ["A" * 8_000, "B" * 19]
    assert pages[1].spans == []
