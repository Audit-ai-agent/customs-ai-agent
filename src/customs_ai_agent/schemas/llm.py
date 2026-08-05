"""Small source-grounded text candidate shape used by the Agent profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaudeFieldCandidate(_Candidate):
    name: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=2000)
    source_quote: str = Field(min_length=1, max_length=4000)

    @field_validator("name", mode="before")
    @classmethod
    def _normalise_name(cls, value: object) -> object:
        return value.strip().casefold() if isinstance(value, str) else value


class ClaudeLineItemCandidate(_Candidate):
    source_quote: str = Field(min_length=1, max_length=4000)
    fields: list[ClaudeFieldCandidate] = Field(default_factory=list, max_length=32)


class ClaudeDocumentCandidate(_Candidate):
    schema_version: Literal["1.0"] = "1.0"
    document_type: Literal[
        "invoice",
        "packing_list",
        "bill_of_lading",
        "certificate_of_origin",
        "unknown",
    ]
    classification_source_quote: str = Field(default="", max_length=4000)
    fields: list[ClaudeFieldCandidate] = Field(default_factory=list, max_length=64)
    line_items: list[ClaudeLineItemCandidate] = Field(default_factory=list, max_length=100)


__all__ = ["ClaudeDocumentCandidate", "ClaudeFieldCandidate", "ClaudeLineItemCandidate"]
