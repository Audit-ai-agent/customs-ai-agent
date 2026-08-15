from __future__ import annotations

import base64
import csv
import hashlib
import zlib
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateDocumentOutputV1,
    CandidateFieldV1,
    CandidateSpanV1,
    CandidateTableCellV1,
    EvaluationPacketV2,
    StratifiedPacketItemV2,
    SyntheticPdfCandidateOutputBundleV1,
    candidate_bundle_sha256,
    canonical_json_file_bytes,
    packet_receipt_sha256,
    sha256_file,
)
from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    SourceAttestedSyntheticCandidateExecutionV1,
    SourceAttestedSyntheticDatasetScopeV1,
    SourceAttestedSyntheticOcrEvaluationResultV1,
    SourceAttestedSyntheticPdfTruthError,
    SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION,
    SOURCE_ATTESTED_REQUIRED_FIELDS,
    build_source_attested_synthetic_candidate_execution,
    build_source_attested_synthetic_development_selection,
    build_source_attested_synthetic_sealed_run,
    compile_source_attested_synthetic_pdf_layout_truth,
    recompute_source_attested_synthetic_pdf_candidate_metrics,
    read_canonical_source_attested_synthetic_candidate_bundle,
    read_canonical_source_attested_synthetic_candidate_execution,
    read_canonical_source_attested_synthetic_dataset_receipt,
    read_canonical_source_attested_synthetic_evaluation_result,
    read_canonical_source_attested_synthetic_layout_truth,
    read_canonical_source_attested_synthetic_packet,
    score_source_attested_synthetic_pdf_candidate_output,
    source_attested_evaluation_result_sha256,
    verify_source_attested_synthetic_pdf_evaluation_replay,
    write_source_attested_synthetic_candidate_bundle,
    write_source_attested_synthetic_candidate_execution,
    write_source_attested_synthetic_dataset_receipt,
    write_source_attested_synthetic_development_selection,
    write_source_attested_synthetic_evaluation_result,
    write_source_attested_synthetic_layout_truth,
    write_source_attested_synthetic_packet,
    write_source_attested_synthetic_sealed_run,
)
from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import (
    SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    SyntheticPdfCandidateRunnerError,
)
from tools.evaluate_source_attested_synthetic_pdf import evaluate_source_attested_synthetic_pdf


_CLEAN_HEADERS = [
    "set_id",
    "exporter_name",
    "importer_name",
    "consignee",
    "bank",
    "load_port",
    "disch_port",
    "final_dest",
    "vessel",
    "voyage",
    "bl_no",
    "inv_no",
    "inv_date",
    "lc_no",
    "container_no",
    "seal_no",
    "good",
    "unit",
    "qty",
    "unit_price",
    "amount",
    "total",
    "cartons",
    "containers",
    "net_weight",
    "gross_weight",
    "cbm",
]
_ERROR_HEADERS = [
    "set_id",
    "num_errors_in_set",
    "error_type",
    "affected_doc",
    "field",
    "correct_value",
    "wrong_value",
    "note",
]
_HEADER_RECTS = (
    (45.0, 527.8898, 105.0, 16.0),
    (150.0, 527.8898, 170.0, 16.0),
    (320.0, 527.8898, 80.0, 16.0),
    (400.0, 527.8898, 80.0, 16.0),
    (480.0, 527.8898, 70.27559, 16.0),
)
_BODY_RECTS = (
    (45.0, 377.8898, 105.0, 150.0),
    (150.0, 377.8898, 170.0, 150.0),
    (320.0, 377.8898, 80.0, 150.0),
    (400.0, 377.8898, 80.0, 150.0),
    (480.0, 377.8898, 70.27559, 150.0),
)

_TEST_INVOICE_REQUIRED_FIELDS = (
    ("exporter_name", "text-v1"),
    ("importer_name", "text-v1"),
    ("consignee", "text-v1"),
    ("load_port", "text-v1"),
    ("disch_port", "text-v1"),
    ("vessel", "text-v1"),
    ("inv_no", "identifier-v1"),
    ("inv_date", "identifier-v1"),
    ("lc_no", "identifier-v1"),
    ("good", "text-v1"),
    ("unit", "text-v1"),
    ("qty", "numeric-v1"),
    ("unit_price", "numeric-v1"),
    ("amount", "numeric-v1"),
    ("total", "numeric-v1"),
)
_TEST_REQUIRED_FIELD_SCHEMA = {"invoice": _TEST_INVOICE_REQUIRED_FIELDS}


def _sha(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _invoice_pdf(*, text_prefix: str, total: str, shift_header: bool = False) -> bytes:
    """Build the pinned invoice profile without pulling a PDF package in tests."""

    rectangles = [(10.0, 100.0, 20.0, 10.0)] * 11
    header = list(_HEADER_RECTS)
    if shift_header:
        x, y, width, height = header[0]
        header[0] = (x + 1.0, y, width, height)
    rectangles.extend(header)
    rectangles.extend(_BODY_RECTS)
    rectangles.extend(((10.0, 120.0, 20.0, 10.0), (10.0, 140.0, 20.0, 10.0)))
    assert len(rectangles) == 23

    positions = [(20.0, 760.0)] * 41
    field_positions = {
        3: (49.0, 743.8898),
        6: (301.6378, 743.8898),
        7: (301.6378, 734.3898),
        9: (49.0, 673.8898),
        12: (301.6378, 673.8898),
        15: (49.0, 609.8898),
        17: (49.0, 572.8898),
        19: (175.3189, 572.8898),
        24: (49.0, 550.8898),
        35: (153.0, 513.8898),
        40: (487.1316, 362.8898),
    }
    for reading_order, position in field_positions.items():
        positions[reading_order - 1] = position
    for offset, rectangle in enumerate(header, start=27):
        positions[offset - 1] = (rectangle[0] + 1.0, rectangle[1] + 1.0)
    for reading_order, position in {
        32: (48.0, 513.8898),
        33: (48.0, 504.3898),
        34: (48.0, 494.8898),
        35: (153.0, 513.8898),
        36: (358.256, 513.8898),
        37: (453.327, 513.8898),
        38: (494.5246, 513.8898),
    }.items():
        positions[reading_order - 1] = position

    lines = ["0.5 w"]
    lines.extend(f"{x} {y} {width} {height} re S" for x, y, width, height in rectangles)
    lines.extend(("BT", "/F1 10 Tf", "12 TL"))
    for reading_order, (x, y) in enumerate(positions, start=1):
        lines.append(f"1 0 0 1 {x} {y} Tm")
        rendered = {
            36: "2 PCS",
            37: "5.00",
            38: "10.00",
            40: total,
        }.get(reading_order, f"{text_prefix}-{reading_order}")
        lines.append(f"({rendered}) Tj")
    lines.append("ET")
    compressed = zlib.compress("\n".join(lines).encode("ascii"))
    encoded = base64.a85encode(compressed, adobe=True)
    return b"".join(
        (
            b"%PDF-1.3\n",
            b"1 0 obj\n<< /Type /Page /MediaBox [0 0 595.2756 841.8898] /Contents 2 0 R >>\nendobj\n",
            b"2 0 obj\n<< /Length ",
            str(len(encoded)).encode("ascii"),
            b" /Filter [/ASCII85Decode /FlateDecode] >>\nstream\n",
            encoded,
            b"\nendstream\nendobj\n%%EOF\n",
        )
    )


def _write_csv_sources(dataset_root: Path, *, wrong_total: str) -> tuple[Path, Path]:
    clean_manifest = dataset_root / "clean_manifest.csv"
    error_labels = dataset_root / "error_labels.csv"
    with clean_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CLEAN_HEADERS)
        writer.writeheader()
        for index in (1, 2):
            writer.writerow(
                {field: f"manifest-{field}-{index}" for field in _CLEAN_HEADERS}
                | {"set_id": f"C{index:04}", "qty": "2", "amount": "10", "total": "10"}
            )
    with error_labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_ERROR_HEADERS)
        writer.writeheader()
        for index in (1, 2):
            writer.writerow(
                {
                    "set_id": f"E{index:04}",
                    "num_errors_in_set": "1",
                    "error_type": "synthetic",
                    "affected_doc": "COMMERCIAL_INVOICE",
                    "field": "TOTAL",
                    "correct_value": "10",
                    "wrong_value": wrong_total,
                    "note": "test",
                }
            )
    return clean_manifest, error_labels


def _packet(*, receipt: DatasetReceiptV1, labels_sha256: str, split: str) -> EvaluationPacketV2:
    number = 1 if split == "development" else 2
    items = [
        StratifiedPacketItemV2(
            asset_id=f"asset-clean-{number}",
            source_sha256=next(
                asset.sha256
                for asset in receipt.assets
                if asset.asset_id == f"asset-clean-{number}"
            ),
            fixture_kind="clean",
            document_type="invoice",
            set_id=f"C{number:04}",
        ),
        StratifiedPacketItemV2(
            asset_id=f"asset-error-{number}",
            source_sha256=next(
                asset.sha256
                for asset in receipt.assets
                if asset.asset_id == f"asset-error-{number}"
            ),
            fixture_kind="error",
            document_type="invoice",
            set_id=f"E{number:04}",
        ),
    ]
    payload: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "split": split,
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "receipt_sha256": receipt_sha256(receipt),
        "error_labels_sha256": labels_sha256,
        "prepared_as_of": "2026-08-16",
        "items": [item.model_dump(mode="json") for item in items],
        "required_gates": ["field_exact_match", "ocr_cer", "table_cell_f1"],
        "release_thresholds": {
            "field_exact_match_min": 0.99,
            "ocr_cer_max": 0.01,
            "table_cell_f1_min": 0.98,
        },
    }
    payload["packet_receipt_sha256"] = packet_receipt_sha256(payload)
    return EvaluationPacketV2.model_validate(payload)


def _fixture(
    tmp_path: Path,
    *,
    unchanged_error: bool = False,
    shifted_template: bool = False,
    wrong_total: str = "20",
) -> dict[str, object]:
    local_data = tmp_path / "workspace" / "_local" / "data"
    dataset_root = local_data / "synthetic-pdf"
    assets: list[DatasetAssetReceiptV1] = []
    for number in (1, 2):
        clean_bytes = _invoice_pdf(text_prefix=f"clean{number}", total="10")
        error_bytes = (
            clean_bytes
            if unchanged_error and number == 1
            else _invoice_pdf(
                text_prefix=f"error{number}", total="20", shift_header=shifted_template
            )
        )
        for fixture_kind, prefix, content in (
            ("clean", "C", clean_bytes),
            ("error", "E", error_bytes),
        ):
            relative = f"{fixture_kind}_sets/set_{number:04}/{prefix}{number:04}_INV.pdf"
            path = dataset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            assets.append(
                DatasetAssetReceiptV1(
                    asset_id=f"asset-{fixture_kind}-{number}",
                    relative_path=relative,
                    media_type="application/pdf",
                    byte_size=len(content),
                    sha256=_sha(content),
                    source_role="preview_candidate",
                )
            )
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-pdf",
        dataset_revision="source-v1",
        received_as_of=date(2026, 8, 16),
        assets=assets,
    )
    clean_manifest, error_labels = _write_csv_sources(dataset_root, wrong_total=wrong_total)
    return {
        "dataset_root": dataset_root,
        "receipt": receipt,
        "clean_manifest": clean_manifest,
        "error_labels": error_labels,
        "development_packet": _packet(
            receipt=receipt, labels_sha256=sha256_file(error_labels), split="development"
        ),
        "sealed_packet": _packet(
            receipt=receipt, labels_sha256=sha256_file(error_labels), split="sealed"
        ),
        "preview_root": local_data / "document-preview",
    }


def _test_scope(fixture: dict[str, object]) -> SourceAttestedSyntheticDatasetScopeV1:
    receipt = fixture["receipt"]
    assert isinstance(receipt, DatasetReceiptV1)
    return SourceAttestedSyntheticDatasetScopeV1(
        dataset_id=receipt.dataset_id,
        dataset_revision=receipt.dataset_revision,
        dataset_receipt_sha256=receipt_sha256(receipt),
        total_pdf_assets=4,
        clean_pdf_assets=2,
        error_pdf_assets=2,
        sets_per_fixture_kind=2,
        require_complete_canonical_inventory=False,
        require_canonical_packets=False,
    )


def _truth(fixture: dict[str, object], split: str):
    return compile_source_attested_synthetic_pdf_layout_truth(
        receipt=fixture["receipt"],  # type: ignore[arg-type]
        packet=fixture[f"{split}_packet"],  # type: ignore[arg-type]
        source_directory=fixture["dataset_root"],  # type: ignore[arg-type]
        clean_manifest_path=fixture["clean_manifest"],  # type: ignore[arg-type]
        error_labels_path=fixture["error_labels"],  # type: ignore[arg-type]
        expected_dataset_scope=_test_scope(fixture),
        required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
    )


def _profile(engine_kind: str) -> tuple[str, str, str]:
    return (
        f"{engine_kind}-test-v1",
        _sha(f"configuration:{engine_kind}"),
        _sha(f"inventory:{engine_kind}"),
    )


def _matching_candidate(*, packet: EvaluationPacketV2, truth, engine_kind: str):
    extractor_revision, configuration_sha256, model_inventory_sha256 = _profile(engine_kind)
    documents: list[CandidateDocumentOutputV1] = []
    for record in truth.records:
        documents.append(
            CandidateDocumentOutputV1(
                asset_id=record.asset_id,
                source_sha256=record.source_sha256,
                fields=[
                    CandidateFieldV1(field_name=field.field_name, value=field.value)
                    for field in record.fields
                ],
                spans=[
                    CandidateSpanV1(
                        page_number=span.page_number,
                        reading_order=span.reading_order,
                        text=span.text,
                    )
                    for span in record.spans
                ],
                table_cells=[
                    CandidateTableCellV1(
                        page_number=cell.page_number,
                        table_index=cell.table_index,
                        row_index=cell.row_index,
                        column_index=cell.column_index,
                        text=cell.text,
                    )
                    for cell in record.table_cells
                ],
            )
        )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "synthetic_pdf_candidate_output",
        "profile_mode": "offline_synthetic_pdf_evaluation",
        "parser_revision": SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "engine_kind": engine_kind,
        "extractor_revision": extractor_revision,
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": model_inventory_sha256,
        "documents": [document.model_dump(mode="json") for document in documents],
    }
    payload["candidate_bundle_sha256"] = candidate_bundle_sha256(payload)
    return SyntheticPdfCandidateOutputBundleV1.model_validate(payload)


def _candidate_raw(candidate: SyntheticPdfCandidateOutputBundleV1) -> bytes:
    return canonical_json_file_bytes(candidate.model_dump(mode="json"))


def _execution(
    *, fixture: dict[str, object], packet: EvaluationPacketV2, truth, candidate, engine_kind: str
) -> SourceAttestedSyntheticCandidateExecutionV1:
    extractor_revision, configuration_sha256, model_inventory_sha256 = _profile(engine_kind)
    return build_source_attested_synthetic_candidate_execution(
        receipt=fixture["receipt"],  # type: ignore[arg-type]
        packet=packet,
        layout_truth=truth,
        candidate_output=candidate,
        candidate_output_raw=_candidate_raw(candidate),
        engine_kind=engine_kind,  # type: ignore[arg-type]
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        expected_dataset_scope=_test_scope(fixture),
        required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
    )


def _score(*, fixture: dict[str, object], split: str, truth, engine_kind: str):
    packet = fixture[f"{split}_packet"]
    candidate = _matching_candidate(packet=packet, truth=truth, engine_kind=engine_kind)
    extractor_revision, configuration_sha256, model_inventory_sha256 = _profile(engine_kind)
    execution = _execution(
        fixture=fixture,
        packet=packet,
        truth=truth,
        candidate=candidate,
        engine_kind=engine_kind,
    )
    result = score_source_attested_synthetic_pdf_candidate_output(
        receipt=fixture["receipt"],
        packet=packet,
        layout_truth=truth,
        candidate_output_raw=_candidate_raw(candidate),
        candidate_execution=execution,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        expected_dataset_scope=_test_scope(fixture),
        required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
    )
    return candidate, result


def test_compiler_uses_content_stream_not_manifest_and_writes_immutable_local_truth(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    truth = _truth(fixture, "development")

    assert len(truth.records) == 2
    assert truth.source_manifest_mismatch_count > 0
    assert {record.fixture_kind for record in truth.records} == {"clean", "error"}
    assert truth.records[0].fields[0].value.startswith("clean1-")
    assert len(truth.records[0].table_cells) == 10

    destination = write_source_attested_synthetic_layout_truth(
        truth=truth,
        output_directory=fixture["preview_root"] / "source-layout-truths",  # type: ignore[operator]
    )
    assert destination.name == f"{truth.layout_truth_receipt_sha256}.json"
    assert read_canonical_source_attested_synthetic_layout_truth(destination) == truth
    receipt_path = write_source_attested_synthetic_dataset_receipt(
        receipt=fixture["receipt"],
        output_directory=fixture["preview_root"] / "source-attested-dataset-receipts",  # type: ignore[operator]
    )
    assert (
        read_canonical_source_attested_synthetic_dataset_receipt(receipt_path) == fixture["receipt"]
    )
    packet_path = write_source_attested_synthetic_packet(
        packet=fixture["development_packet"],
        output_directory=fixture["preview_root"] / "source-attested-packets",  # type: ignore[operator]
    )
    assert (
        read_canonical_source_attested_synthetic_packet(packet_path)
        == fixture["development_packet"]
    )
    candidate = _matching_candidate(
        packet=fixture["development_packet"], truth=truth, engine_kind="pdf_text_layer"
    )
    execution = _execution(
        fixture=fixture,
        packet=fixture["development_packet"],
        truth=truth,
        candidate=candidate,
        engine_kind="pdf_text_layer",
    )
    execution_path = write_source_attested_synthetic_candidate_execution(
        execution=execution,
        output_directory=fixture["preview_root"] / "source-attested-candidate-executions",  # type: ignore[operator]
    )
    assert read_canonical_source_attested_synthetic_candidate_execution(execution_path) == execution


def test_compiler_fails_closed_for_an_unchanged_error_or_changed_template(tmp_path: Path) -> None:
    unchanged = _fixture(tmp_path / "unchanged", unchanged_error=True)
    with pytest.raises(SourceAttestedSyntheticPdfTruthError, match="error_content_unchanged"):
        _truth(unchanged, "development")

    shifted = _fixture(tmp_path / "shifted", shifted_template=True)
    with pytest.raises(SourceAttestedSyntheticPdfTruthError, match="template_profile_mismatch"):
        _truth(shifted, "development")


def test_automatic_scope_rejects_a_small_or_wrong_packet_fixture(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(SourceAttestedSyntheticPdfTruthError, match="dataset_scope_invalid"):
        compile_source_attested_synthetic_pdf_layout_truth(
            receipt=fixture["receipt"],  # type: ignore[arg-type]
            packet=fixture["development_packet"],  # type: ignore[arg-type]
            source_directory=fixture["dataset_root"],  # type: ignore[arg-type]
            clean_manifest_path=fixture["clean_manifest"],  # type: ignore[arg-type]
            error_labels_path=fixture["error_labels"],  # type: ignore[arg-type]
        )


def test_default_full_field_universe_matches_the_rendered_invoice_profile(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    truth = compile_source_attested_synthetic_pdf_layout_truth(
        receipt=fixture["receipt"],  # type: ignore[arg-type]
        packet=fixture["development_packet"],  # type: ignore[arg-type]
        source_directory=fixture["dataset_root"],  # type: ignore[arg-type]
        clean_manifest_path=fixture["clean_manifest"],  # type: ignore[arg-type]
        error_labels_path=fixture["error_labels"],  # type: ignore[arg-type]
        expected_dataset_scope=_test_scope(fixture),
    )
    assert {field.field_name for field in truth.records[0].fields} == {
        name for name, _normalizer in SOURCE_ATTESTED_REQUIRED_FIELDS["invoice"]
    }
    assert len(truth.records[0].fields) == 15


def test_rendered_field_contract_exposes_bl13_invoice15_and_pl12() -> None:
    counts = {
        document_type: len(fields)
        for document_type, fields in SOURCE_ATTESTED_REQUIRED_FIELDS.items()
    }

    assert counts == {"bill_of_lading": 13, "invoice": 15, "packing_list": 12}
    assert "containers" not in {
        field_name for field_name, _normalizer in SOURCE_ATTESTED_REQUIRED_FIELDS["packing_list"]
    }
    assert SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION == SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION


def test_source_truth_rejects_pre_v3_candidate_parser_even_for_test_scope(tmp_path: Path) -> None:
    """A coherent older parser cannot create new source-attested evidence."""

    fixture = _fixture(tmp_path)
    truth = _truth(fixture, "development")
    packet = fixture["development_packet"]
    candidate = _matching_candidate(packet=packet, truth=truth, engine_kind="pdf_text_layer")
    extractor_revision, configuration_sha256, model_inventory_sha256 = _profile("pdf_text_layer")

    with pytest.raises(
        SourceAttestedSyntheticPdfTruthError,
        match="candidate_parser_revision_invalid",
    ):
        recompute_source_attested_synthetic_pdf_candidate_metrics(
            receipt=fixture["receipt"],  # type: ignore[arg-type]
            packet=packet,
            layout_truth=truth,
            candidate_output_raw=_candidate_raw(candidate),
            engine_kind="pdf_text_layer",
            extractor_revision=extractor_revision,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=model_inventory_sha256,
            parser_revision="synthetic-pdf-label-normalizer-v2",
            expected_dataset_scope=_test_scope(fixture),
            required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
        )


def test_error_label_must_match_a_rendered_source_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, wrong_total="999")
    with pytest.raises(
        SourceAttestedSyntheticPdfTruthError, match="error_label_rendering_unverified"
    ):
        _truth(fixture, "development")


def test_raw_candidate_rescore_and_execution_receipt_are_replay_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    truth = _truth(fixture, "development")
    packet = fixture["development_packet"]
    candidate, result = _score(
        fixture=fixture,
        split="development",
        truth=truth,
        engine_kind="pdf_text_layer",
    )
    raw = _candidate_raw(candidate)
    execution = _execution(
        fixture=fixture,
        packet=packet,
        truth=truth,
        candidate=candidate,
        engine_kind="pdf_text_layer",
    )
    extractor_revision, configuration_sha256, model_inventory_sha256 = _profile("pdf_text_layer")
    rescore = recompute_source_attested_synthetic_pdf_candidate_metrics(
        receipt=fixture["receipt"],  # type: ignore[arg-type]
        packet=packet,
        layout_truth=truth,
        candidate_output_raw=raw,
        engine_kind="pdf_text_layer",
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        expected_dataset_scope=_test_scope(fixture),
        required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
    )
    assert rescore.candidate_bundle_sha256 == candidate.candidate_bundle_sha256
    assert rescore.candidate_output_sha256 == _sha(raw)
    assert rescore.metrics == result.metrics
    assert rescore.metrics.field_exact_match_count == 30
    assert rescore.metrics.field_exact_match_total == 30
    assert [
        item.model_dump(mode="json") for item in rescore.metrics.field_exact_match_denominators
    ] == [
        {
            "document_type": "invoice",
            "document_count": 2,
            "fields_per_document": 15,
            "total_fields": 30,
        }
    ]
    verify_source_attested_synthetic_pdf_evaluation_replay(
        result=result,
        receipt=fixture["receipt"],  # type: ignore[arg-type]
        packet=packet,
        layout_truth=truth,
        candidate_output_raw=raw,
        candidate_execution=execution,
        engine_kind="pdf_text_layer",
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        expected_dataset_scope=_test_scope(fixture),
        required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
    )
    with pytest.raises(
        SourceAttestedSyntheticPdfTruthError, match="candidate_output_not_canonical"
    ):
        recompute_source_attested_synthetic_pdf_candidate_metrics(
            receipt=fixture["receipt"],  # type: ignore[arg-type]
            packet=packet,
            layout_truth=truth,
            candidate_output_raw=raw + b"\n",
            engine_kind="pdf_text_layer",
            extractor_revision=extractor_revision,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=model_inventory_sha256,
            parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
            expected_dataset_scope=_test_scope(fixture),
            required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
        )
    forged_payload = result.model_dump(mode="json")
    forged_payload["metrics"] = dict(forged_payload["metrics"]) | {
        "field_exact_match": 0.0,
        "field_exact_match_count": 0,
    }
    forged_payload["passed"] = False
    forged_payload["evaluation_result_sha256"] = source_attested_evaluation_result_sha256(
        forged_payload
    )
    forged = SourceAttestedSyntheticOcrEvaluationResultV1.model_validate(forged_payload)
    with pytest.raises(SourceAttestedSyntheticPdfTruthError, match="evaluation_replay_mismatch"):
        verify_source_attested_synthetic_pdf_evaluation_replay(
            result=forged,
            receipt=fixture["receipt"],  # type: ignore[arg-type]
            packet=packet,
            layout_truth=truth,
            candidate_output_raw=raw,
            candidate_execution=execution,
            engine_kind="pdf_text_layer",
            extractor_revision=extractor_revision,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=model_inventory_sha256,
            parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
            expected_dataset_scope=_test_scope(fixture),
            required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
        )


def test_three_engine_selection_and_distinct_sealed_layout_receipts_are_bound(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    development_truth = _truth(fixture, "development")
    development_results = [
        _score(
            fixture=fixture,
            split="development",
            truth=development_truth,
            engine_kind=engine_kind,
        )[1]
        for engine_kind in ("pdf_text_layer", "paddleocr", "docling")
    ]
    selection = build_source_attested_synthetic_development_selection(results=development_results)
    assert selection.selected_engine_kind == "docling"
    write_source_attested_synthetic_development_selection(
        selection=selection,
        output_directory=fixture["preview_root"] / "source-attested-development-selections",  # type: ignore[operator]
    )

    sealed_truth = _truth(fixture, "sealed")
    sealed_packet = fixture["sealed_packet"]
    sealed_candidate = _matching_candidate(
        packet=sealed_packet, truth=sealed_truth, engine_kind=selection.selected_engine_kind
    )
    candidate_path = write_source_attested_synthetic_candidate_bundle(
        candidate_bundle=sealed_candidate,
        output_directory=fixture["preview_root"] / "source-attested-candidate-bundles",  # type: ignore[operator]
    )
    assert (
        read_canonical_source_attested_synthetic_candidate_bundle(candidate_path)
        == sealed_candidate
    )
    extractor_revision, configuration_sha256, model_inventory_sha256 = _profile(
        selection.selected_engine_kind
    )
    sealed_execution = _execution(
        fixture=fixture,
        packet=sealed_packet,
        truth=sealed_truth,
        candidate=sealed_candidate,
        engine_kind=selection.selected_engine_kind,
    )
    reservation = (
        fixture["preview_root"]
        / "candidate-run-reservations"
        / (  # type: ignore[operator]
            f"sealed-{sealed_packet.packet_receipt_sha256}.json"
        )
    )
    reservation.parent.mkdir(parents=True)
    reservation.write_bytes(
        canonical_json_file_bytes(
            {
                "schema_version": "1.0",
                "artifact_kind": "sealed_synthetic_pdf_candidate_reservation",
                "profile_mode": "offline_synthetic_pdf_evaluation",
                "parser_revision": SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
                "dataset_id": sealed_packet.dataset_id,
                "dataset_revision": sealed_packet.dataset_revision,
                "packet_kind": "sealed",
                "packet_receipt_sha256": sealed_packet.packet_receipt_sha256,
                "engine_kind": selection.selected_engine_kind,
                "extractor_revision": extractor_revision,
                "configuration_sha256": configuration_sha256,
                "model_inventory_sha256": model_inventory_sha256,
            }
        )
    )
    sealed_run = build_source_attested_synthetic_sealed_run(
        packet=sealed_packet,
        layout_truth=sealed_truth,
        development_selection=selection,
        candidate_output=sealed_candidate,
        candidate_execution=sealed_execution,
        document_preview_root=fixture["preview_root"],
        engine_kind=selection.selected_engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    )
    assert sealed_run.layout_truth_receipt_sha256 == sealed_truth.layout_truth_receipt_sha256
    assert sealed_run.layout_truth_receipt_sha256 != selection.layout_truth_receipt_sha256
    write_source_attested_synthetic_sealed_run(
        sealed_run=sealed_run,
        output_directory=fixture["preview_root"] / "source-attested-sealed-runs",  # type: ignore[operator]
    )
    sealed_result = score_source_attested_synthetic_pdf_candidate_output(
        receipt=fixture["receipt"],
        packet=sealed_packet,
        layout_truth=sealed_truth,
        candidate_output_raw=_candidate_raw(sealed_candidate),
        candidate_execution=sealed_execution,
        engine_kind=selection.selected_engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        development_selection=selection,
        sealed_run=sealed_run,
        expected_dataset_scope=_test_scope(fixture),
        required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
    )
    assert sealed_result.passed is True
    result_path = write_source_attested_synthetic_evaluation_result(
        result=sealed_result,
        output_directory=fixture["preview_root"] / "source-attested-evaluation-results",  # type: ignore[operator]
    )
    assert read_canonical_source_attested_synthetic_evaluation_result(result_path) == sealed_result
    with pytest.raises(SourceAttestedSyntheticPdfTruthError, match="sealed_evaluation_consumed"):
        write_source_attested_synthetic_evaluation_result(
            result=sealed_result,
            output_directory=fixture["preview_root"] / "source-attested-evaluation-results",  # type: ignore[operator]
        )


def test_selection_requires_all_three_engine_results(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    truth = _truth(fixture, "development")
    two_results = [
        _score(fixture=fixture, split="development", truth=truth, engine_kind=engine_kind)[1]
        for engine_kind in ("pdf_text_layer", "paddleocr")
    ]
    with pytest.raises(
        SourceAttestedSyntheticPdfTruthError, match="development_result_set_invalid"
    ):
        build_source_attested_synthetic_development_selection(results=two_results)


def test_source_evaluator_refuses_runner_parser_revision_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI refuses before it can write local evidence with an unapproved parser."""

    monkeypatch.setattr(
        "tools.evaluate_source_attested_synthetic_pdf.SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION",
        "synthetic-pdf-label-normalizer-v2",
    )
    with pytest.raises(
        SyntheticPdfCandidateRunnerError,
        match="source_attested_candidate_parser_revision_invalid",
    ):
        evaluate_source_attested_synthetic_pdf(
            receipt_path=tmp_path / "receipt.json",
            packet_path=tmp_path / "packet.json",
            layout_truth_path=tmp_path / "truth.json",
            candidate_output_path=tmp_path / "candidate.json",
            document_preview_root=tmp_path / "document-preview",
            configuration_path=tmp_path / "configuration.json",
            inventory_path=tmp_path / "inventory.json",
            engine_kind="pdf_text_layer",
        )


def test_explicit_auto_activation_runs_only_after_a_passing_sealed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    preview_root = fixture["preview_root"]
    preview_root.mkdir(parents=True)
    development_truth = _truth(fixture, "development")
    write_source_attested_synthetic_layout_truth(
        truth=development_truth,
        output_directory=preview_root / "source-layout-truths",
    )
    development_results = [
        _score(
            fixture=fixture,
            split="development",
            truth=development_truth,
            engine_kind=engine_kind,
        )[1]
        for engine_kind in ("pdf_text_layer", "paddleocr", "docling")
    ]
    selection = build_source_attested_synthetic_development_selection(results=development_results)
    selection_path = write_source_attested_synthetic_development_selection(
        selection=selection,
        output_directory=preview_root / "source-attested-development-selections",
    )
    sealed_truth = _truth(fixture, "sealed")
    layout_path = write_source_attested_synthetic_layout_truth(
        truth=sealed_truth,
        output_directory=preview_root / "source-layout-truths",
    )
    sealed_packet = fixture["sealed_packet"]
    candidate = _matching_candidate(
        packet=sealed_packet,
        truth=sealed_truth,
        engine_kind=selection.selected_engine_kind,
    )
    candidate_path = (
        preview_root
        / "candidate-outputs"
        / "docling"
        / (f"{candidate.candidate_bundle_sha256}.json")
    )
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_bytes(canonical_json_file_bytes(candidate.model_dump(mode="json")))
    extractor_revision, configuration_sha256, model_inventory_sha256 = _profile("docling")
    reservation_path = (
        preview_root
        / "candidate-run-reservations"
        / (f"sealed-{sealed_packet.packet_receipt_sha256}.json")
    )
    reservation_path.parent.mkdir(parents=True)
    reservation_path.write_bytes(
        canonical_json_file_bytes(
            {
                "schema_version": "1.0",
                "artifact_kind": "sealed_synthetic_pdf_candidate_reservation",
                "profile_mode": "offline_synthetic_pdf_evaluation",
                "parser_revision": SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
                "dataset_id": sealed_packet.dataset_id,
                "dataset_revision": sealed_packet.dataset_revision,
                "packet_kind": "sealed",
                "packet_receipt_sha256": sealed_packet.packet_receipt_sha256,
                "engine_kind": "docling",
                "extractor_revision": extractor_revision,
                "configuration_sha256": configuration_sha256,
                "model_inventory_sha256": model_inventory_sha256,
            }
        )
    )
    receipt_path = write_source_attested_synthetic_dataset_receipt(
        receipt=fixture["receipt"],
        output_directory=preview_root / "source-attested-dataset-receipts",
    )
    packet_path = write_source_attested_synthetic_packet(
        packet=sealed_packet,
        output_directory=preview_root / "source-attested-packets",
    )
    profile = SimpleNamespace(
        document_preview_root=preview_root.resolve(),
        engine_kind="docling",
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
    )
    monkeypatch.setattr(
        "tools.evaluate_source_attested_synthetic_pdf.load_unapproved_local_candidate_profile",
        lambda **_kwargs: profile,
    )
    monkeypatch.setattr(
        "tools.evaluate_source_attested_synthetic_pdf.build_source_attested_synthetic_candidate_execution",
        lambda **kwargs: build_source_attested_synthetic_candidate_execution(
            **kwargs,
            expected_dataset_scope=_test_scope(fixture),
            required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
        ),
    )
    monkeypatch.setattr(
        "tools.evaluate_source_attested_synthetic_pdf.score_source_attested_synthetic_pdf_candidate_output",
        lambda **kwargs: score_source_attested_synthetic_pdf_candidate_output(
            **kwargs,
            expected_dataset_scope=_test_scope(fixture),
            required_field_schema=_TEST_REQUIRED_FIELD_SCHEMA,
        ),
    )
    import customs_ai_agent.service.document_preview_runtime as runtime

    calls: list[dict[str, object]] = []

    def write_pointer(**kwargs: object) -> tuple[str, dict[str, object]]:
        calls.append(kwargs)
        assert list((preview_root / "source-attested-evaluation-results").glob("*.json"))
        return _sha("activation"), {}

    monkeypatch.setattr(
        runtime,
        "write_source_attested_synthetic_pdf_preview_pointer",
        write_pointer,
        raising=False,
    )
    summary = evaluate_source_attested_synthetic_pdf(
        receipt_path=receipt_path,
        packet_path=packet_path,
        layout_truth_path=layout_path,
        candidate_output_path=candidate_path,
        document_preview_root=preview_root,
        configuration_path=preview_root / "runtime-configurations" / "unused.json",
        inventory_path=preview_root / "model-inventories" / "unused.json",
        engine_kind="docling",
        development_selection_path=selection_path,
        auto_activate_source_attested_synthetic=True,
    )
    assert summary["passed"] is True
    assert summary["candidate_execution_receipt_sha256"]
    assert summary["activation_sha256"] == _sha("activation")
    assert len(calls) == 1
    assert calls[0]["configuration_sha256"] == configuration_sha256
    assert calls[0]["model_inventory_sha256"] == model_inventory_sha256
