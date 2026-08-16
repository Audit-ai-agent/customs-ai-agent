from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from customs_ai_agent.evaluation.document_ocr_quality import canonical_json_file_bytes
from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    NativePdfSemanticModelArtifactV1,
    NativePdfSemanticModelError,
    bind_native_pdf_semantic_model_artifact,
    load_hash_bound_native_pdf_semantic_model,
)


def _span_selector(*, reading_order: int) -> dict[str, object]:
    return {
        "source_kind": "span_reading_order",
        "page_number": 1,
        "reading_order": reading_order,
    }


def _table_selector(*, row_index: int, column_index: int) -> dict[str, object]:
    return {
        "source_kind": "table_cell_coordinate",
        "page_number": 1,
        "table_index": 1,
        "row_index": row_index,
        "column_index": column_index,
    }


def _artifact_payload() -> dict[str, object]:
    """A generic selector model, not a corpus-derived artifact."""

    return {
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
                    "minimum_span_count": 2,
                    "minimum_table_cell_count": 1,
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
                        "field_name": "quantity",
                        "normalizer": "numeric-v1",
                        "selectors": [_table_selector(row_index=1, column_index=0)],
                        "value_transform": "single-numeric-token-v1",
                    },
                ],
            }
        ],
    }


def _artifact_bytes() -> bytes:
    artifact = NativePdfSemanticModelArtifactV1.model_validate(_artifact_payload())
    return canonical_json_file_bytes(artifact.model_dump(mode="json", exclude_none=True))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_binds_only_canonical_artifact_bytes_to_expected_sha256() -> None:
    artifact_bytes = _artifact_bytes()

    binding = bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=_sha256(artifact_bytes),
    )

    assert binding.artifact.model_id == "demo-native-semantic"
    assert binding.verified_artifact() == binding.artifact


def test_rejects_noncanonical_or_forbidden_artifact_content() -> None:
    noncanonical = json.dumps(_artifact_payload(), ensure_ascii=False, indent=4).encode("utf-8")
    forbidden = _artifact_payload()
    documents = forbidden["documents"]
    assert isinstance(documents, list)
    document = documents[0]
    assert isinstance(document, dict)
    document["source_sha256"] = "a" * 64
    forbidden_bytes = canonical_json_file_bytes(forbidden)

    with pytest.raises(NativePdfSemanticModelError, match="semantic_model_artifact_schema_invalid"):
        bind_native_pdf_semantic_model_artifact(
            artifact_bytes=forbidden_bytes,
            expected_sha256=_sha256(forbidden_bytes),
        )
    with pytest.raises(NativePdfSemanticModelError, match="semantic_model_artifact_not_canonical"):
        bind_native_pdf_semantic_model_artifact(
            artifact_bytes=noncanonical,
            expected_sha256=_sha256(noncanonical),
        )


def test_rejects_raw_value_and_invalid_selector_shape() -> None:
    raw_value_payload = _artifact_payload()
    documents = raw_value_payload["documents"]
    assert isinstance(documents, list)
    document = documents[0]
    assert isinstance(document, dict)
    fields = document["fields"]
    assert isinstance(fields, list)
    raw_value_field = fields[0]
    assert isinstance(raw_value_field, dict)
    raw_value_field["raw_value"] = "not-allowed"
    raw_value_bytes = canonical_json_file_bytes(raw_value_payload)

    invalid_shape_payload = _artifact_payload()
    documents = invalid_shape_payload["documents"]
    assert isinstance(documents, list)
    document = documents[0]
    assert isinstance(document, dict)
    signature = document["structural_signature"]
    assert isinstance(signature, dict)
    selectors = signature["required_selectors"]
    assert isinstance(selectors, list)
    selector = selectors[0]
    assert isinstance(selector, dict)
    selector["table_index"] = 1
    invalid_shape_bytes = canonical_json_file_bytes(invalid_shape_payload)

    with pytest.raises(NativePdfSemanticModelError, match="semantic_model_artifact_schema_invalid"):
        bind_native_pdf_semantic_model_artifact(
            artifact_bytes=raw_value_bytes,
            expected_sha256=_sha256(raw_value_bytes),
        )
    with pytest.raises(NativePdfSemanticModelError, match="semantic_model_artifact_schema_invalid"):
        bind_native_pdf_semantic_model_artifact(
            artifact_bytes=invalid_shape_bytes,
            expected_sha256=_sha256(invalid_shape_bytes),
        )


def test_rejects_hash_mismatch_and_memory_binding_mutation() -> None:
    artifact_bytes = _artifact_bytes()

    with pytest.raises(NativePdfSemanticModelError, match="semantic_model_artifact_sha256_mismatch"):
        bind_native_pdf_semantic_model_artifact(
            artifact_bytes=artifact_bytes,
            expected_sha256="0" * 64,
        )

    binding = bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=_sha256(artifact_bytes),
    )
    mutated = replace(binding, artifact_bytes=artifact_bytes + b" ")
    with pytest.raises(NativePdfSemanticModelError, match="semantic_model_binding_invalid"):
        mutated.verified_artifact()


def test_loader_reads_only_explicit_regular_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_path = tmp_path / "semantic-model.json"
    artifact_bytes = _artifact_bytes()
    artifact_path.write_bytes(artifact_bytes)
    opened: list[Path] = []
    original_open = Path.open

    def guarded_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        opened.append(self)
        if self != artifact_path:
            raise AssertionError("loader attempted to read a non-artifact file")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    binding = load_hash_bound_native_pdf_semantic_model(
        artifact_path=artifact_path,
        expected_sha256=_sha256(artifact_bytes),
    )

    assert binding.artifact_sha256 == _sha256(artifact_bytes)
    assert opened == [artifact_path]


def test_loader_rejects_symlink_before_opening_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "semantic-model.json"
    artifact_bytes = _artifact_bytes()
    artifact_path.write_bytes(artifact_bytes)
    original_lstat = Path.lstat

    def symlink_lstat(self: Path) -> SimpleNamespace:
        if self == artifact_path:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return original_lstat(self)  # type: ignore[return-value]

    monkeypatch.setattr(Path, "lstat", symlink_lstat)

    with pytest.raises(NativePdfSemanticModelError, match="semantic_model_artifact_symlink_invalid"):
        load_hash_bound_native_pdf_semantic_model(
            artifact_path=artifact_path,
            expected_sha256=_sha256(artifact_bytes),
        )
