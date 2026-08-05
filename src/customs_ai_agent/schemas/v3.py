"""Small text-only wire models for the active Backend-to-Agent endpoint."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class V3WireModel(BaseModel):
    """Reject undeclared wire properties at the Agent boundary."""

    model_config = ConfigDict(extra="forbid")


class TextDocumentInputV3(V3WireModel):
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    filename: str = Field(min_length=1, max_length=255)
    content_text: str = Field(min_length=1, max_length=200_000)
    declared_type: str | None = Field(default=None, max_length=64)


class ServiceReviewRunRequestV3(V3WireModel):
    case_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_run_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_as_of: date
    trade_type: Literal["import", "export"]
    jurisdiction: Literal["KR"]
    documents: list[TextDocumentInputV3] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _document_ids_are_unique(self) -> "ServiceReviewRunRequestV3":
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique within a review run")
        return self


class ReviewReportV3(V3WireModel):
    overall_status: Literal["needs_review", "blocked"]
    summary: str = Field(min_length=1, max_length=4000)
    disclaimer: str = Field(min_length=1, max_length=2000)


class EvidenceV3(V3WireModel):
    evidence_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1, max_length=20_000)
    source_uri: str = Field(min_length=1, max_length=4000)
    source_type: str = Field(min_length=1, max_length=128)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    law_id: str | None = Field(default=None, max_length=512)
    law_name: str | None = Field(default=None, max_length=2000)


class FindingV3(V3WireModel):
    kind: Literal["validation_issue", "regulatory_finding"]
    code: str = Field(min_length=1, max_length=128)
    severity: Literal["low", "medium", "high", "critical"]
    message: str = Field(min_length=1, max_length=4000)
    item_id: str | None = Field(default=None, max_length=128)
    evidence_ids: list[str] = Field(default_factory=list, max_length=256)


class RequiredActionV1(V3WireModel):
    action_code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class EngineV3(V3WireModel):
    kind: Literal["deterministic", "claude"]
    model_id: str = Field(min_length=1, max_length=256)


class ReviewRunResultV3(V3WireModel):
    case_id: str = Field(min_length=1, max_length=128)
    review_run_id: str = Field(min_length=1, max_length=128)
    review_as_of: date
    trade_type: Literal["import", "export"]
    jurisdiction: Literal["KR"]
    status: Literal["needs_review", "blocked"]
    requires_human_review: Literal[True] = True
    execution_status: Literal["succeeded", "partial", "failed"]
    assessment: Literal["no_issue_detected", "issues_detected", "indeterminate"]
    input_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    report: ReviewReportV3
    evidence: list[EvidenceV3] = Field(default_factory=list, max_length=10_000)
    findings: list[FindingV3] = Field(default_factory=list, max_length=2000)
    required_actions: list[RequiredActionV1] = Field(min_length=1, max_length=100)
    result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    schema_version: Literal["3.0"] = "3.0"
    engine: EngineV3 | None = None

    @model_validator(mode="after")
    def _human_only_and_hashed(self) -> "ReviewRunResultV3":
        if self.status == "blocked" and self.execution_status != "failed":
            raise ValueError("blocked result must have failed execution_status")
        if self.status == "needs_review" and self.report.overall_status != "needs_review":
            raise ValueError("needs_review result must have needs_review report status")
        if self.result_sha256 != review_result_v3_sha256(self):
            raise ValueError("result_sha256 does not match the public result")
        return self


def _canonical_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported canonical value: {type(value)!r}")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        default=_canonical_default,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def review_result_v3_sha256(result: ReviewRunResultV3 | Mapping[str, Any]) -> str:
    if isinstance(result, ReviewRunResultV3):
        payload = result.model_dump(mode="json", warnings=False)
    else:
        payload = dict(result)
    payload.pop("result_sha256", None)
    return canonical_json_sha256(payload)


__all__ = [
    "EngineV3",
    "EvidenceV3",
    "FindingV3",
    "RequiredActionV1",
    "ReviewReportV3",
    "ReviewRunResultV3",
    "ServiceReviewRunRequestV3",
    "TextDocumentInputV3",
    "canonical_json_sha256",
    "review_result_v3_sha256",
]
