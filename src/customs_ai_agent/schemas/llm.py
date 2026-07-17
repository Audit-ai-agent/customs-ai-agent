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
