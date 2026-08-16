from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
from pathlib import Path

import pytest

import customs_ai_agent.evaluation.native_pdf_semantic_adapter as semantic_adapter
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateSpanV1,
    CandidateTableCellV1,
    canonical_json_file_bytes,
)
from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    NativePdfSemanticModelArtifactV1,
    bind_native_pdf_semantic_model_artifact,
)
from customs_ai_agent.evaluation.native_pdf_vector_grid import (
    NativePdfVectorFormRegion,
    NativePdfVectorGridExtraction,
    NativePdfVectorGridPage,
)


def _span_selector(*, reading_order: int) -> dict[str, object]:
    return {
        "source_kind": "span_reading_order",
        "page_number": 1,
        "reading_order": reading_order,
    }


def _anchor_selector(*, anchor_x: float, anchor_y: float) -> dict[str, object]:
    return {
        "source_kind": "span_anchor",
        "page_number": 1,
        "anchor_x": anchor_x,
        "anchor_y": anchor_y,
        "anchor_tolerance": 0.001,
    }


def _table_selector(*, row_index: int, column_index: int) -> dict[str, object]:
    return {
        "source_kind": "table_cell_coordinate",
        "page_number": 1,
        "table_index": 1,
        "row_index": row_index,
        "column_index": column_index,
    }


def _model_binding():  # type: ignore[no-untyped-def]
    payload = {
        "schema_version": "1.0",
        "artifact_kind": "native_pdf_semantic_model",
        "model_id": "demo-native-semantic",
        "model_revision": "v1",
        "inference_revision": "native-template-selector-v1",
        "documents": [
            {
                "document_type": "demo_document",
                "structural_signature": {
                    "page_count": 1,
                    "minimum_span_count": 3,
                    "minimum_table_cell_count": 2,
                    "required_selectors": [_span_selector(reading_order=1)],
                },
                "fields": [
                    {
                        "field_name": "reference",
                        "normalizer": "identifier-v1",
                        "selectors": [_span_selector(reading_order=2)],
                        "value_transform": "raw-v1",
                    },
                    {
                        "field_name": "alpha_code",
                        "normalizer": "identifier-v1",
                        "selectors": [_anchor_selector(anchor_x=0.60, anchor_y=0.20)],
                        "value_transform": "single-alpha-token-v1",
                    },
                    {
                        "field_name": "quantity",
                        "normalizer": "numeric-v1",
                        "selectors": [_table_selector(row_index=1, column_index=0)],
                        "value_transform": "single-numeric-token-v1",
                    },
                    {
                        "field_name": "reference_quantity",
                        "normalizer": "identifier-v1",
                        "selectors": [
                            _span_selector(reading_order=2),
                            _table_selector(row_index=1, column_index=0),
                        ],
                        "value_transform": "join-slash-v1",
                    },
                ],
            }
        ],
    }
    artifact = NativePdfSemanticModelArtifactV1.model_validate(payload)
    artifact_bytes = canonical_json_file_bytes(artifact.model_dump(mode="json", exclude_none=True))
    return bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )


def _native_extraction(*, duplicate_table_cell: bool = False) -> NativePdfVectorGridExtraction:
    spans = (
        CandidateSpanV1(
            page_number=1,
            reading_order=1,
            text="profile marker",
            anchor_x=0.05,
            anchor_y=0.05,
        ),
        CandidateSpanV1(
            page_number=1,
            reading_order=2,
            text="REF-001",
            anchor_x=0.20,
            anchor_y=0.20,
        ),
        CandidateSpanV1(
            page_number=1,
            reading_order=3,
            text="ABC",
            anchor_x=0.60,
            anchor_y=0.20,
        ),
    )
    table_cells = [
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=0,
            column_index=0,
            text="header",
        ),
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=1,
            column_index=0,
            text="15",
        ),
    ]
    if duplicate_table_cell:
        table_cells.append(
            CandidateTableCellV1(
                page_number=1,
                table_index=1,
                row_index=1,
                column_index=0,
                text="16",
            )
        )
    return NativePdfVectorGridExtraction(
        pages=(NativePdfVectorGridPage(page_number=1, width=200, height=120),),
        spans=spans,
        table_cells=tuple(table_cells),
        form_regions=(
            NativePdfVectorFormRegion(
                page_number=1,
                x=0.05,
                y=0.10,
                width=0.60,
                height=0.20,
            ),
        ),
    )


def test_adapter_preserves_native_evidence_and_applies_learned_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _native_extraction()
    observed_pdf_bytes: list[bytes] = []

    def fake_native_extract(*, pdf_bytes: bytes) -> NativePdfVectorGridExtraction:
        observed_pdf_bytes.append(pdf_bytes)
        return extraction

    monkeypatch.setattr(
        semantic_adapter,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        fake_native_extract,
    )
    adapter = semantic_adapter.NativePdfSemanticAdapter(model=_model_binding())

    output = adapter.extract_pdf_bytes(b"%PDF-selected-only")

    assert observed_pdf_bytes == [b"%PDF-selected-only"]
    assert output.document_type == "demo_document"
    assert [(field.field_name, field.value) for field in output.fields] == [
        ("alpha_code", "ABC"),
        ("quantity", "15"),
        ("reference", "REF-001"),
        ("reference_quantity", "REF-001 / 15"),
    ]
    assert output.spans == extraction.spans
    assert output.table_cells == extraction.table_cells
    assert output.form_regions == extraction.form_regions


def test_duplicate_selector_source_is_shape_mismatch_and_field_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _native_extraction(duplicate_table_cell=True)
    monkeypatch.setattr(
        semantic_adapter,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        lambda *, pdf_bytes: extraction,
    )
    adapter = semantic_adapter.NativePdfSemanticAdapter(model=_model_binding())

    output = adapter.extract_pdf_bytes(b"%PDF-selected-only")

    # The structural signature does not depend on the duplicated cell, so the
    # profile remains known.  Every field selecting that coordinate is omitted
    # rather than selecting one value by position or content.
    assert output.document_type == "demo_document"
    assert [(field.field_name, field.value) for field in output.fields] == [
        ("alpha_code", "ABC"),
        ("reference", "REF-001"),
    ]


def test_same_native_source_cannot_satisfy_two_join_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _native_extraction()
    binding = _model_binding()
    artifact_payload = binding.artifact.model_dump(mode="json", exclude_none=True)
    documents = artifact_payload["documents"]
    assert isinstance(documents, list)
    document = documents[0]
    assert isinstance(document, dict)
    fields = document["fields"]
    assert isinstance(fields, list)
    joined = fields[3]
    assert isinstance(joined, dict)
    joined["selectors"] = [
        _span_selector(reading_order=2),
        _anchor_selector(anchor_x=0.20, anchor_y=0.20),
    ]
    artifact = NativePdfSemanticModelArtifactV1.model_validate(artifact_payload)
    artifact_bytes = canonical_json_file_bytes(artifact.model_dump(mode="json", exclude_none=True))
    duplicate_binding = bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        semantic_adapter,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        lambda *, pdf_bytes: extraction,
    )

    output = semantic_adapter.NativePdfSemanticAdapter(model=duplicate_binding).extract_pdf_bytes(
        b"%PDF-selected-only"
    )

    assert {field.field_name for field in output.fields} == {
        "alpha_code",
        "quantity",
        "reference",
    }


def test_token_transforms_extract_one_bounded_token_from_one_native_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _native_extraction()
    binding = _model_binding()
    artifact_payload = binding.artifact.model_dump(mode="json", exclude_none=True)
    documents = artifact_payload["documents"]
    assert isinstance(documents, list)
    document = documents[0]
    assert isinstance(document, dict)
    fields = document["fields"]
    assert isinstance(fields, list)
    fields[:] = [
        {
            "field_name": "quantity_from_composite_run",
            "normalizer": "numeric-v1",
            "selectors": [_span_selector(reading_order=2)],
            "value_transform": "single-numeric-token-v1",
        },
        {
            "field_name": "unit_from_composite_run",
            "normalizer": "text-v1",
            "selectors": [_span_selector(reading_order=2)],
            "value_transform": "single-alpha-token-v1",
        },
    ]
    artifact = NativePdfSemanticModelArtifactV1.model_validate(artifact_payload)
    artifact_bytes = canonical_json_file_bytes(artifact.model_dump(mode="json", exclude_none=True))
    composite_binding = bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )
    composite_spans = list(extraction.spans)
    composite_spans[1] = CandidateSpanV1(
        page_number=1,
        reading_order=2,
        text="15 PCS",
        anchor_x=0.20,
        anchor_y=0.20,
    )
    composite = NativePdfVectorGridExtraction(
        pages=extraction.pages,
        spans=tuple(composite_spans),
        table_cells=extraction.table_cells,
        form_regions=extraction.form_regions,
    )
    monkeypatch.setattr(
        semantic_adapter,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        lambda *, pdf_bytes: composite,
    )

    output = semantic_adapter.NativePdfSemanticAdapter(model=composite_binding).extract_pdf_bytes(
        b"%PDF-selected-only"
    )

    assert [(field.field_name, field.value) for field in output.fields] == [
        ("quantity_from_composite_run", "15"),
        ("unit_from_composite_run", "PCS"),
    ]


def test_token_transform_rejects_multiple_tokens_in_one_native_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _native_extraction()
    binding = _model_binding()
    artifact_payload = binding.artifact.model_dump(mode="json", exclude_none=True)
    documents = artifact_payload["documents"]
    assert isinstance(documents, list)
    document = documents[0]
    assert isinstance(document, dict)
    fields = document["fields"]
    assert isinstance(fields, list)
    fields[:] = [
        {
            "field_name": "ambiguous_quantity",
            "normalizer": "numeric-v1",
            "selectors": [_span_selector(reading_order=2)],
            "value_transform": "single-numeric-token-v1",
        }
    ]
    artifact = NativePdfSemanticModelArtifactV1.model_validate(artifact_payload)
    artifact_bytes = canonical_json_file_bytes(artifact.model_dump(mode="json", exclude_none=True))
    ambiguous_binding = bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )
    ambiguous_spans = list(extraction.spans)
    ambiguous_spans[1] = CandidateSpanV1(
        page_number=1,
        reading_order=2,
        text="15 OR 16",
        anchor_x=0.20,
        anchor_y=0.20,
    )
    ambiguous = NativePdfVectorGridExtraction(
        pages=extraction.pages,
        spans=tuple(ambiguous_spans),
        table_cells=extraction.table_cells,
        form_regions=extraction.form_regions,
    )
    monkeypatch.setattr(
        semantic_adapter,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        lambda *, pdf_bytes: ambiguous,
    )

    output = semantic_adapter.NativePdfSemanticAdapter(model=ambiguous_binding).extract_pdf_bytes(
        b"%PDF-selected-only"
    )

    assert output.document_type == "demo_document"
    assert output.fields == ()


def test_semantic_runtime_does_not_import_source_truth_or_training_only_modules() -> None:
    tree = ast.parse(Path(semantic_adapter.__file__).read_text(encoding="utf-8"))
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]

    assert all("source_attested_synthetic_pdf_truth" not in name for name in imports)
    assert all("supervised_template_training" not in name for name in imports)


def test_ambiguous_structural_profile_fails_closed_without_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _native_extraction()
    binding = _model_binding()
    artifact_payload = binding.artifact.model_dump(mode="json", exclude_none=True)
    documents = artifact_payload["documents"]
    assert isinstance(documents, list)
    documents.append(documents[0] | {"document_type": "other_document"})
    artifact = NativePdfSemanticModelArtifactV1.model_validate(artifact_payload)
    artifact_bytes = canonical_json_file_bytes(artifact.model_dump(mode="json", exclude_none=True))
    ambiguous_binding = bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        semantic_adapter,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        lambda *, pdf_bytes: extraction,
    )

    output = semantic_adapter.NativePdfSemanticAdapter(model=ambiguous_binding).extract_pdf_bytes(
        b"%PDF-selected-only"
    )

    assert output.document_type is None
    assert output.fields == ()


def test_adapter_denies_any_filesystem_read_after_model_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _native_extraction()

    def fake_native_extract(*, pdf_bytes: bytes) -> NativePdfVectorGridExtraction:
        assert pdf_bytes == b"%PDF-selected-only"
        return extraction

    def deny_open(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        raise AssertionError("semantic inference attempted a filesystem read")

    monkeypatch.setattr(
        semantic_adapter,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        fake_native_extract,
    )
    monkeypatch.setattr(Path, "open", deny_open)
    monkeypatch.setattr(builtins, "open", deny_open)
    adapter = semantic_adapter.NativePdfSemanticAdapter(model=_model_binding())

    output = adapter.extract_pdf_bytes(b"%PDF-selected-only")

    assert output.fields[0].field_name == "alpha_code"


def test_adapter_boundary_has_no_asset_id_or_truth_inputs() -> None:
    source = inspect.getsource(semantic_adapter)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    signature = inspect.signature(semantic_adapter.NativePdfSemanticAdapter.extract_pdf_bytes)

    assert "customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth" not in imported_modules
    assert "customs_ai_agent.evaluation.document_gold" not in imported_modules
    assert "csv" not in imported_modules
    assert "asset_id" not in source
    assert tuple(signature.parameters) == ("self", "pdf_bytes")
