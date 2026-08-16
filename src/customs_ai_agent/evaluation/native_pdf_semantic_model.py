"""Strict hash-bound artifacts for offline native-PDF semantic selectors.

The artifact contains learned *structural* parameters only: native span
reading-order/anchor selectors, native table-cell coordinates, and a small
allowlisted transform.  It deliberately cannot carry candidate values, asset
identifiers, source hashes, CSV/Gold/source-truth data, source-parser profiles,
or text-run payloads.  Loading reads one explicitly named model artifact and
binds its exact canonical bytes to a caller-supplied SHA-256; inference then
uses that in-memory binding and selected PDF bytes only.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.document_ocr_quality import (
    FieldNormalizer,
    canonical_json_file_bytes,
)


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_MAX_ARTIFACT_BYTES: Final = 512 * 1024


class NativePdfSemanticModelError(ValueError):
    """Content-free error safe to include in local operator summaries."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _ArtifactWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NativePdfSemanticSelectorV1(_ArtifactWireModel):
    """One learned selector over direct candidate-native extraction evidence."""

    source_kind: Literal["span_reading_order", "span_anchor", "table_cell_coordinate"]
    page_number: int = Field(ge=1, le=1_000)
    reading_order: int | None = Field(default=None, ge=1, le=100_000)
    anchor_x: float | None = Field(default=None, ge=0.0, le=1.0)
    anchor_y: float | None = Field(default=None, ge=0.0, le=1.0)
    anchor_tolerance: float | None = Field(default=None, gt=0.0, le=0.05)
    table_index: int | None = Field(default=None, ge=1, le=10_000)
    row_index: int | None = Field(default=None, ge=0, le=100_000)
    column_index: int | None = Field(default=None, ge=0, le=100_000)

    @model_validator(mode="after")
    def _has_exactly_one_selector_shape(self) -> "NativePdfSemanticSelectorV1":
        span_anchor_values = (self.anchor_x, self.anchor_y, self.anchor_tolerance)
        table_values = (self.table_index, self.row_index, self.column_index)
        if self.source_kind == "span_reading_order":
            if self.reading_order is None or any(value is not None for value in span_anchor_values):
                raise ValueError("span reading-order selector shape is invalid")
            if any(value is not None for value in table_values):
                raise ValueError("span reading-order selector must not contain table coordinates")
            return self
        if self.source_kind == "span_anchor":
            if self.reading_order is not None or any(value is None for value in span_anchor_values):
                raise ValueError("span anchor selector shape is invalid")
            if any(value is not None for value in table_values):
                raise ValueError("span anchor selector must not contain table coordinates")
            return self
        if self.reading_order is not None or any(value is not None for value in span_anchor_values):
            raise ValueError("table selector must not contain span coordinates")
        if any(value is None for value in table_values):
            raise ValueError("table coordinate selector shape is invalid")
        return self


def _selector_key(selector: NativePdfSemanticSelectorV1) -> str:
    return json.dumps(
        selector.model_dump(mode="json", warnings=False, exclude_none=True),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class NativePdfSemanticStructuralSignatureV1(_ArtifactWireModel):
    """Document-profile shape used before any field value is emitted."""

    page_count: int = Field(ge=1, le=1_000)
    minimum_span_count: int = Field(ge=0, le=100_000)
    minimum_table_cell_count: int = Field(ge=0, le=100_000)
    required_selectors: list[NativePdfSemanticSelectorV1] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _required_selectors_are_unique(self) -> "NativePdfSemanticStructuralSignatureV1":
        keys = [_selector_key(selector) for selector in self.required_selectors]
        if len(keys) != len(set(keys)):
            raise ValueError("structural signature selectors must be unique")
        return self


class NativePdfSemanticFieldModelV1(_ArtifactWireModel):
    """One field's learned direct selectors and conservative value transform."""

    field_name: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]{0,127}$")
    normalizer: FieldNormalizer
    selectors: list[NativePdfSemanticSelectorV1] = Field(min_length=1, max_length=4)
    value_transform: Literal[
        "raw-v1",
        "single-numeric-token-v1",
        "single-alpha-token-v1",
        "join-slash-v1",
    ]

    @model_validator(mode="after")
    def _selector_count_matches_transform(self) -> "NativePdfSemanticFieldModelV1":
        selector_count = len(self.selectors)
        if self.value_transform == "join-slash-v1":
            if not 2 <= selector_count <= 4:
                raise ValueError("join-slash transform requires two through four selectors")
        elif selector_count != 1:
            raise ValueError("single-value transform requires exactly one selector")
        keys = [_selector_key(selector) for selector in self.selectors]
        if len(keys) != len(set(keys)):
            raise ValueError("field selectors must be unique")
        return self


class NativePdfSemanticDocumentModelV1(_ArtifactWireModel):
    """One learned structural document profile and its direct field selectors."""

    document_type: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    structural_signature: NativePdfSemanticStructuralSignatureV1
    fields: list[NativePdfSemanticFieldModelV1] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _field_names_are_unique(self) -> "NativePdfSemanticDocumentModelV1":
        field_names = [field.field_name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("document model field names must be unique")
        return self


class NativePdfSemanticModelArtifactV1(_ArtifactWireModel):
    """Canonical standalone learned selector parameters for byte-only inference."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["native_pdf_semantic_model"] = "native_pdf_semantic_model"
    model_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    model_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    inference_revision: Literal["native-template-selector-v1"] = "native-template-selector-v1"
    documents: list[NativePdfSemanticDocumentModelV1] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _document_types_are_unique(self) -> "NativePdfSemanticModelArtifactV1":
        document_types = [document.document_type for document in self.documents]
        if len(document_types) != len(set(document_types)):
            raise ValueError("semantic model document types must be unique")
        return self


@dataclass(frozen=True, slots=True)
class HashBoundNativePdfSemanticModelV1:
    """Validated artifact bytes and parsed model, bound to one SHA-256 value."""

    artifact: NativePdfSemanticModelArtifactV1
    artifact_bytes: bytes
    artifact_sha256: str

    def verified_artifact(self) -> NativePdfSemanticModelArtifactV1:
        """Revalidate the in-memory binding before each inference invocation."""

        if hashlib.sha256(self.artifact_bytes).hexdigest() != self.artifact_sha256:
            raise NativePdfSemanticModelError("semantic_model_binding_invalid")
        parsed = _parse_model_artifact_bytes(self.artifact_bytes)
        if parsed != self.artifact:
            raise NativePdfSemanticModelError("semantic_model_binding_invalid")
        return self.artifact


def _validate_expected_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativePdfSemanticModelError("semantic_model_expected_sha256_invalid")
    return value


def _decode_json_object(artifact_bytes: bytes) -> dict[str, object]:
    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite JSON value")

    try:
        value = json.loads(artifact_bytes.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise NativePdfSemanticModelError("semantic_model_artifact_json_invalid") from error
    if not isinstance(value, dict):
        raise NativePdfSemanticModelError("semantic_model_artifact_json_invalid")
    return value


def _parse_model_artifact_bytes(artifact_bytes: bytes) -> NativePdfSemanticModelArtifactV1:
    if not isinstance(artifact_bytes, bytes) or not 1 <= len(artifact_bytes) <= _MAX_ARTIFACT_BYTES:
        raise NativePdfSemanticModelError("semantic_model_artifact_size_invalid")
    payload = _decode_json_object(artifact_bytes)
    try:
        artifact = NativePdfSemanticModelArtifactV1.model_validate(payload)
    except ValidationError as error:
        raise NativePdfSemanticModelError("semantic_model_artifact_schema_invalid") from error
    canonical = canonical_json_file_bytes(
        artifact.model_dump(mode="json", warnings=False, exclude_none=True)
    )
    if artifact_bytes != canonical:
        raise NativePdfSemanticModelError("semantic_model_artifact_not_canonical")
    return artifact


def bind_native_pdf_semantic_model_artifact(
    *, artifact_bytes: bytes, expected_sha256: str
) -> HashBoundNativePdfSemanticModelV1:
    """Parse exact artifact bytes and bind them to an expected SHA-256 value."""

    expected = _validate_expected_sha256(expected_sha256)
    if not isinstance(artifact_bytes, bytes):
        raise NativePdfSemanticModelError("semantic_model_artifact_size_invalid")
    actual = hashlib.sha256(artifact_bytes).hexdigest()
    if actual != expected:
        raise NativePdfSemanticModelError("semantic_model_artifact_sha256_mismatch")
    artifact = _parse_model_artifact_bytes(artifact_bytes)
    return HashBoundNativePdfSemanticModelV1(
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        artifact_sha256=actual,
    )


def _reject_symlink_path_components(path: Path) -> None:
    """Reject a model artifact reached through any symlink component."""

    current = path
    while True:
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise NativePdfSemanticModelError("semantic_model_artifact_path_invalid") from error
        if stat.S_ISLNK(mode):
            raise NativePdfSemanticModelError("semantic_model_artifact_symlink_invalid")
        parent = current.parent
        if parent == current:
            return
        current = parent


def load_hash_bound_native_pdf_semantic_model(
    *, artifact_path: Path, expected_sha256: str
) -> HashBoundNativePdfSemanticModelV1:
    """Read exactly one regular, non-symlink artifact then verify its binding.

    The filesystem boundary ends here.  Callers pass the returned in-memory
    binding to the adapter, whose inference method accepts PDF bytes only.
    """

    if not isinstance(artifact_path, Path):
        raise NativePdfSemanticModelError("semantic_model_artifact_path_invalid")
    _reject_symlink_path_components(artifact_path)
    try:
        mode = artifact_path.lstat().st_mode
    except OSError as error:
        raise NativePdfSemanticModelError("semantic_model_artifact_path_invalid") from error
    if not stat.S_ISREG(mode):
        raise NativePdfSemanticModelError("semantic_model_artifact_path_invalid")
    try:
        with artifact_path.open("rb") as stream:
            artifact_bytes = stream.read(_MAX_ARTIFACT_BYTES + 1)
    except OSError as error:
        raise NativePdfSemanticModelError("semantic_model_artifact_read_failed") from error
    return bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=expected_sha256,
    )


__all__ = [
    "HashBoundNativePdfSemanticModelV1",
    "NativePdfSemanticDocumentModelV1",
    "NativePdfSemanticFieldModelV1",
    "NativePdfSemanticModelArtifactV1",
    "NativePdfSemanticModelError",
    "NativePdfSemanticSelectorV1",
    "NativePdfSemanticStructuralSignatureV1",
    "bind_native_pdf_semantic_model_artifact",
    "load_hash_bound_native_pdf_semantic_model",
]
