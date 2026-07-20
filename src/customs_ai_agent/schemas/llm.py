from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from customs_ai_agent.schemas.models import DocumentType


class StrictCandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaudeFieldCandidate(StrictCandidateModel):
    """One verbatim field candidate produced by Claude.

    Missing fields are omitted from the enclosing list. Empty strings are never
    a substitute for a missing value.
    """

    name: str
    value: str
    source_quote: str

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _non_empty_and_bounded(self) -> "ClaudeFieldCandidate":
        if not self.name.strip():
            raise ValueError("candidate field name cannot be blank")
        if not self.value.strip():
            raise ValueError("candidate field value cannot be blank")
        if not self.source_quote.strip():
            raise ValueError("candidate source_quote cannot be blank")
        if len(self.name) > 64 or len(self.value) > 4000 or len(self.source_quote) > 8000:
            raise ValueError("candidate field exceeds the local safety limit")
        return self


class ClaudeLineItemCandidate(StrictCandidateModel):
    """A line item in source order.

    The model does not create an item ID. The Agent assigns a stable ID from the
    owning document and source order after grounding succeeds.
    """

    source_quote: str
    fields: list[ClaudeFieldCandidate]

    @model_validator(mode="after")
    def _valid_line_item(self) -> "ClaudeLineItemCandidate":
        if not self.source_quote.strip():
            raise ValueError("line item source_quote cannot be blank")
        if len(self.source_quote) > 12000:
            raise ValueError("line item source_quote exceeds the local safety limit")
        if not self.fields or len(self.fields) > 16:
            raise ValueError("line item must contain 1..16 fields")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("line item candidate field names must be unique")
        return self


class ClaudeDocumentCandidate(StrictCandidateModel):
    """The only structured shape accepted from the model."""

    schema_version: Literal["1.0"]
    document_type: DocumentType
    classification_source_quote: str
    fields: list[ClaudeFieldCandidate]
    line_items: list[ClaudeLineItemCandidate]

    @field_validator("document_type", mode="before")
    @classmethod
    def _normalize_document_type(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _valid_document_candidate(self) -> "ClaudeDocumentCandidate":
        if len(self.classification_source_quote) > 8000:
            raise ValueError("classification_source_quote exceeds the local safety limit")
        if len(self.fields) > 32:
            raise ValueError("document candidate contains too many fields")
        if len(self.line_items) > 200:
            raise ValueError("document candidate contains too many line items")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("document candidate field names must be unique")
        if self.document_type == "unknown":
            if self.classification_source_quote.strip():
                raise ValueError("unknown classification must not claim a classification quote")
        elif not self.classification_source_quote.strip():
            raise ValueError("known classification requires a source quote")
        return self


class ClaudePixelBBox(StrictCandidateModel):
    """Absolute half-open pixel coordinates in the original, unresized page."""

    page: int
    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="after")
    def _valid_bbox(self) -> "ClaudePixelBBox":
        if self.page < 1:
            raise ValueError("bbox page must be positive")
        if min(self.x1, self.y1, self.x2, self.y2) < 0:
            raise ValueError("bbox coordinates cannot be negative")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox x2/y2 must be greater than x1/y1")
        return self


class ClaudeVisionEvidence(StrictCandidateModel):
    """Claude's mapping of a field to independent OCR/layout observations.

    `source_text` and `model_observed_bbox` remain unverified model
    observations. Only `ocr_span_ids` can become field provenance after local
    reconciliation with the independently produced OCR ledger.
    """

    source_text: str
    model_observed_bbox: ClaudePixelBBox
    ocr_span_ids: list[str]

    @model_validator(mode="after")
    def _valid_evidence(self) -> "ClaudeVisionEvidence":
        if not self.source_text.strip():
            raise ValueError("vision source_text cannot be blank")
        if len(self.source_text) > 12_000:
            raise ValueError("vision source_text exceeds the local safety limit")
        if not self.ocr_span_ids or len(self.ocr_span_ids) > 128:
            raise ValueError("vision evidence must reference 1..128 OCR spans")
        if len(self.ocr_span_ids) != len(set(self.ocr_span_ids)):
            raise ValueError("vision OCR span references must be unique")
        if any(
            not span_id
            or len(span_id) > 128
            or not span_id[0].isalnum()
            or any(not (character.isalnum() or character in "._:-") for character in span_id)
            for span_id in self.ocr_span_ids
        ):
            raise ValueError("vision OCR span reference has an invalid identifier")
        return self


class ClaudeVisionFieldCandidate(StrictCandidateModel):
    name: str
    value: str
    evidence: ClaudeVisionEvidence

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _valid_field(self) -> "ClaudeVisionFieldCandidate":
        if not self.name.strip() or not self.value.strip():
            raise ValueError("vision field name and value cannot be blank")
        if len(self.name) > 64 or len(self.value) > 4000:
            raise ValueError("vision field exceeds the local safety limit")
        return self


class ClaudeVisionLineItemCandidate(StrictCandidateModel):
    row_evidence: ClaudeVisionEvidence
    fields: list[ClaudeVisionFieldCandidate]

    @model_validator(mode="after")
    def _valid_line_item(self) -> "ClaudeVisionLineItemCandidate":
        if not self.fields or len(self.fields) > 16:
            raise ValueError("vision line item must contain 1..16 fields")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("vision line item field names must be unique")
        return self


class ClaudeVisionDocumentCandidate(StrictCandidateModel):
    """The only structured shape accepted from Claude for rasterized pages."""

    schema_version: Literal["1.0"]
    document_type: DocumentType
    classification_evidence: ClaudeVisionEvidence | None
    fields: list[ClaudeVisionFieldCandidate]
    line_items: list[ClaudeVisionLineItemCandidate]

    @field_validator("document_type", mode="before")
    @classmethod
    def _normalize_document_type(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _valid_document(self) -> "ClaudeVisionDocumentCandidate":
        if len(self.fields) > 48:
            raise ValueError("vision document candidate contains too many fields")
        if len(self.line_items) > 200:
            raise ValueError("vision document candidate contains too many line items")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("vision document field names must be unique")
        if self.document_type == "unknown":
            if self.classification_evidence is not None:
                raise ValueError("unknown classification cannot claim evidence")
            if self.fields or self.line_items:
                raise ValueError("unknown vision document cannot contain extracted values")
        elif self.classification_evidence is None:
            raise ValueError("known vision classification requires evidence")
        return self


class ClaudeCropReadCandidate(StrictCandidateModel):
    """A bounded reread of one OCR-derived crop.

    Local policy, never Claude, decides whether this observation resolves the
    OCR/model conflict.
    """

    schema_version: Literal["1.0"]
    value: str | None
    source_text: str
    ocr_span_ids: list[str]

    @model_validator(mode="after")
    def _valid_crop_read(self) -> "ClaudeCropReadCandidate":
        if not self.source_text.strip():
            raise ValueError("crop source_text cannot be blank")
        if len(self.source_text) > 12_000:
            raise ValueError("crop source_text exceeds the local safety limit")
        if not self.ocr_span_ids or len(self.ocr_span_ids) > 128:
            raise ValueError("crop read must reference 1..128 OCR spans")
        if len(self.ocr_span_ids) != len(set(self.ocr_span_ids)):
            raise ValueError("crop OCR span references must be unique")
        if any(
            not span_id
            or len(span_id) > 128
            or not span_id[0].isalnum()
            or any(not (character.isalnum() or character in "._:-") for character in span_id)
            for span_id in self.ocr_span_ids
        ):
            raise ValueError("crop OCR span reference has an invalid identifier")
        if self.value is not None:
            if not self.value.strip() or len(self.value) > 4000:
                raise ValueError("crop value must be non-empty and bounded")
        return self
