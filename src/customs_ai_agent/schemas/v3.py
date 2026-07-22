from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from customs_ai_agent.schemas.models import (
    DocumentInputV2,
    ImagePageInput,
    ReviewRunResultV2,
    UserCorrection,
)
from customs_ai_agent.transport_limits import (
    TEXT_FIRST_MAX_DOCUMENTS,
    TEXT_FIRST_MAX_TEXT_CHARACTERS,
)
from customs_ai_agent.v3_semantics import (
    ACTIVE_FAILURE_REASON_CODES,
    allowed_framed_claim_texts,
    citable_regulatory_evidence_ids,
    expected_narrative_claims,
    expected_required_actions,
    has_citable_regulatory_evidence,
    reconstruct_v3_current_semantics,
)


Identifier = str
Sha256 = str
Hsk10 = Annotated[str, Field(pattern=r"^[0-9]{10}$")]
FactKind = Literal["field_source", "issue", "finding", "evidence", "outcome"]
AllowedToolName = Literal[
    "document.reread_span/v1",
    "evidence.retrieve/v3",
    "human.request/v1",
    "explanation.compose/v1",
    "case.answer/v1",
]


class V3WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")


class VerificationReceiptV1(V3WireModel):
    """Detached, case-bound proof produced by Backend or a trusted connector."""

    trust_domain: Literal["customs-ai-agent/supporting-document-verification/v1"]
    case_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    method: Literal["trusted_connector", "human_approver"]
    authority_id: str = Field(min_length=1, max_length=128)
    authority_release: str = Field(min_length=1, max_length=128)
    key_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    supporting_document_claims_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_release: str = Field(min_length=1, max_length=128)
    decision_record_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    connector_result_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    signed_at: datetime
    expires_at: datetime
    signature_base64url: str = Field(pattern=r"^[A-Za-z0-9_-]{86}$")

    @model_validator(mode="after")
    def _method_bound_record_and_time(self) -> "VerificationReceiptV1":
        if self.signed_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("verification receipt times must include a timezone")
        if self.expires_at <= self.signed_at:
            raise ValueError("verification receipt expiry must follow its signature time")
        if self.method == "human_approver":
            if self.decision_record_sha256 is None or self.connector_result_sha256 is not None:
                raise ValueError("human verification receipt requires one decision record")
        elif self.connector_result_sha256 is None or self.decision_record_sha256 is not None:
            raise ValueError("connector verification receipt requires one connector result")
        return self


class VerificationAssertionV1(V3WireModel):
    assertion_id: str = Field(pattern=r"^assert_[a-f0-9]{64}$")
    method: Literal["trusted_connector", "human_approver", "test_issuer"]
    subject_document_id: str = Field(min_length=1, max_length=128)
    asserted_by: str = Field(min_length=1, max_length=256)
    asserted_at: datetime
    authority_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    authority_release: str = Field(min_length=1, max_length=128)
    source_reference_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    receipt: VerificationReceiptV1 | None = None

    @model_validator(mode="after")
    def _content_bound_assertion_id(self) -> "VerificationAssertionV1":
        if self.method == "human_approver":
            match = re.fullmatch(
                r"backend:principal-sha256:([a-f0-9]{64})",
                self.authority_id,
            )
            if match is None:
                raise ValueError("human approval requires the Backend principal authority")
            if match.group(1) != hashlib.sha256(self.asserted_by.encode("utf-8")).hexdigest():
                raise ValueError("human authority must bind the asserted Backend principal")
            if self.authority_release != "backend-human-decision-policy/v1":
                raise ValueError("human approval authority release is not allowed")
        elif self.method == "trusted_connector":
            match = re.fullmatch(r"connector:([A-Za-z0-9._-]+)", self.authority_id)
            if match is None or self.authority_release != (f"connector-policy:{match.group(1)}/v1"):
                raise ValueError("connector assertion authority release is not allowed")
        elif not self.authority_id.startswith("test:") or not self.authority_release.startswith(
            "synthetic-"
        ):
            raise ValueError("test issuer assertion must remain in the synthetic namespace")
        if self.receipt is not None:
            if (
                self.method == "test_issuer"
                or self.receipt.method != self.method
                or self.receipt.authority_id != self.authority_id
                or self.receipt.authority_release != self.authority_release
                or self.receipt.signed_at != self.asserted_at
            ):
                raise ValueError("verification receipt does not match its assertion")
        payload = verification_assertion_unsigned_payload(self)
        if self.assertion_id != stable_object_id("assert", payload):
            raise ValueError("verification assertion ID must bind its canonical contents")
        return self


class SupportingDocumentV1(V3WireModel):
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    document_code: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    issuer: str = Field(min_length=1, max_length=512)
    document_number: str = Field(min_length=1, max_length=256)
    issued_on: date
    valid_until: date | None = None
    subject_item_ids: list[str] = Field(min_length=1, max_length=100)
    hs_codes: list[Hsk10] = Field(default_factory=list, max_length=100)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verification_assertions: list[VerificationAssertionV1] = Field(
        default_factory=list,
        max_length=20,
    )

    @field_validator("subject_item_ids", "hs_codes")
    @classmethod
    def _unique_sorted_values(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 128 for value in values):
            raise ValueError("supporting document references must be 1..128 characters")
        if len(values) != len(set(values)):
            raise ValueError("supporting document references must be unique")
        return sorted(values)

    @model_validator(mode="after")
    def _valid_supporting_document(self) -> "SupportingDocumentV1":
        if self.valid_until is not None and self.valid_until < self.issued_on:
            raise ValueError("supporting document validity cannot precede its issue date")
        assertion_ids = [item.assertion_id for item in self.verification_assertions]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("verification assertion IDs must be unique")
        if any(
            item.subject_document_id != self.document_id for item in self.verification_assertions
        ):
            raise ValueError("verification assertion subject must match its supporting document")
        return self


def supporting_document_claims_sha256(document: SupportingDocumentV1) -> str:
    """Hash all material document claims while excluding mutable proof envelopes."""

    return canonical_json_sha256(
        document.model_dump(mode="json", exclude={"verification_assertions"}, warnings=False)
    )


def verification_receipt_unsigned_payload(receipt: VerificationReceiptV1) -> dict[str, Any]:
    return receipt.model_dump(mode="json", exclude={"signature_base64url"}, warnings=False)


def verification_assertion_unsigned_payload(
    assertion: VerificationAssertionV1,
) -> dict[str, Any]:
    payload = assertion.model_dump(
        mode="json",
        exclude={"assertion_id"},
        exclude_none=True,
        warnings=False,
    )
    receipt = payload.get("receipt")
    if isinstance(receipt, dict):
        receipt.pop("signature_base64url", None)
    return payload


class TextDocumentInputV3(DocumentInputV2):
    """V3 text wire; empty Backend visual defaults are allowed, payloads are not."""

    content_text: str = Field(min_length=1, max_length=200_000)
    image_pages: list[ImagePageInput] = Field(default_factory=list, max_length=0)
    pdf_pages: None = None


class ServiceReviewRunRequestV3(V3WireModel):
    """Additive text-first V3 request; V2 remains byte-for-byte unchanged."""

    case_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    documents: list[TextDocumentInputV3] = Field(
        min_length=1,
        max_length=TEXT_FIRST_MAX_DOCUMENTS,
    )
    llm_mode: Literal["llm"]
    review_run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_as_of: date = Field(default_factory=date.today)
    trade_type: Literal["import", "export"] = "import"
    jurisdiction: Literal["KR"] = "KR"
    supporting_documents: list[SupportingDocumentV1] = Field(
        default_factory=list,
        max_length=100,
    )
    document_language: Literal["unknown", "ko", "en"] = "unknown"
    assertion_cutoff_at: datetime | None = None

    # V2 accepted self-declared code metadata.  It remains visible only for wire
    # compatibility and must be empty in V3; verified SupportingDocument records
    # and the released Data V3 policy are the sole compliance inputs.
    uploaded_requirement_document_codes: list[str] = Field(
        default_factory=list,
        max_length=0,
    )

    @model_validator(mode="before")
    @classmethod
    def _v3_text_transport_limits(cls, value: Any) -> Any:
        """Fail before nested parsing when the V3 contract limit is exceeded."""

        if not isinstance(value, dict):
            return value
        documents = value.get("documents")
        if not isinstance(documents, list):
            return value
        if len(documents) > TEXT_FIRST_MAX_DOCUMENTS:
            raise ValueError("document count exceeds the V3 text-first contract limit")
        text_characters = 0
        for raw_document in documents:
            if isinstance(raw_document, BaseModel):
                raw_document = raw_document.model_dump(mode="python", warnings=False)
            if not isinstance(raw_document, dict):
                continue
            content_text = raw_document.get("content_text")
            if isinstance(content_text, str):
                text_characters += len(content_text)
                if text_characters > TEXT_FIRST_MAX_TEXT_CHARACTERS:
                    raise ValueError("combined document text exceeds the V3 contract limit")
        return value

    @model_validator(mode="after")
    def _unique_supporting_documents(self) -> "ServiceReviewRunRequestV3":
        if self.assertion_cutoff_at is not None and self.assertion_cutoff_at.tzinfo is None:
            raise ValueError("assertion cutoff must include a timezone")
        document_ids = [item.document_id for item in self.supporting_documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("supporting document IDs must be unique")
        source_hashes = [item.source_sha256 for item in self.supporting_documents]
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError("supporting document source hashes must be unique")
        if self.uploaded_requirement_document_codes:
            raise ValueError("V3 forbids self-declared uploaded requirement document codes")
        primary_document_ids = [document.document_id for document in self.documents]
        if len(primary_document_ids) != len(set(primary_document_ids)):
            raise ValueError("document_id values must be unique within a review run")
        if any(document.requirement_document_codes for document in self.documents):
            raise ValueError("V3 forbids document-level requirement code metadata")
        return self


class AgenticCallAuditV1(V3WireModel):
    task_id: Literal[
        "customs_review_coordinator",
        "customs_narrative_report",
        "customs_case_answer",
    ]
    provider: Literal["anthropic"] = "anthropic"
    model_id: str = Field(min_length=1, max_length=128)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_id_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    message_id_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0, le=300_000)


class FactLedgerEntryV1(V3WireModel):
    fact_id: str = Field(pattern=r"^fact_[a-f0-9]{64}$")
    kind: FactKind
    reference_id: str = Field(min_length=1, max_length=256)
    label: str = Field(min_length=1, max_length=256)
    value: str = Field(min_length=1, max_length=4000)
    document_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class FactLedgerV1(V3WireModel):
    ledger_id: str = Field(pattern=r"^ledger_[a-f0-9]{64}$")
    entries: list[FactLedgerEntryV1] = Field(default_factory=list, max_length=20_000)

    @model_validator(mode="after")
    def _unique_facts(self) -> "FactLedgerV1":
        fact_ids = [item.fact_id for item in self.entries]
        reference_ids = [item.reference_id for item in self.entries]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact IDs must be unique")
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("fact reference IDs must be globally unique")
        for item in self.entries:
            canonical = {
                "kind": item.kind,
                "reference_id": item.reference_id,
                "label": item.label,
                "value": item.value,
                "document_id": item.document_id,
            }
            if item.fact_id != stable_object_id("fact", canonical):
                raise ValueError("fact ID does not match its canonical contents")
            if item.source_sha256 != canonical_json_sha256(canonical):
                raise ValueError("fact source hash does not match its canonical contents")
        expected = stable_object_id(
            "ledger",
            [item.model_dump(mode="json") for item in self.entries],
        )
        if self.ledger_id != expected:
            raise ValueError("fact ledger ID does not match its canonical entries")
        return self


class CitationReferenceV1(V3WireModel):
    kind: FactKind
    reference_id: str = Field(min_length=1, max_length=256)


class NarrativeClaimV1(V3WireModel):
    claim_id: str = Field(pattern=r"^claim_[a-f0-9]{64}$")
    text: str = Field(min_length=1, max_length=4000)
    material: bool = True
    citations: list[CitationReferenceV1] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def _material_claim_is_cited(self) -> "NarrativeClaimV1":
        if self.material and not self.citations:
            raise ValueError("material narrative claims require at least one citation")
        references = [(item.kind, item.reference_id) for item in self.citations]
        if len(references) != len(set(references)):
            raise ValueError("narrative citations must be unique")
        expected = stable_object_id(
            "claim",
            {
                "text": self.text,
                "material": self.material,
                "citations": [item.model_dump(mode="json") for item in self.citations],
            },
        )
        if self.claim_id != expected:
            raise ValueError("claim ID does not match its canonical contents")
        return self


class NarrativeReportV1(V3WireModel):
    language: Literal["ko-KR"] = "ko-KR"
    summary: list[NarrativeClaimV1] = Field(min_length=1, max_length=20)
    document_consistency: list[NarrativeClaimV1] = Field(default_factory=list, max_length=100)
    regulatory_review: list[NarrativeClaimV1] = Field(default_factory=list, max_length=100)
    required_actions: list[NarrativeClaimV1] = Field(default_factory=list, max_length=100)
    generation_mode: Literal["claude", "deterministic_fallback"]
    citation_verification: Literal["passed"] = "passed"
    model_call: AgenticCallAuditV1 | None = None

    @model_validator(mode="after")
    def _call_matches_generation_mode(self) -> "NarrativeReportV1":
        if (self.generation_mode == "claude") != (self.model_call is not None):
            raise ValueError("Claude narrative generation requires complete model-call audit")
        return self


class ReviewGapV1(V3WireModel):
    gap_id: str = Field(pattern=r"^gap_[a-f0-9]{64}$")
    gap_type: Literal[
        "field_uncertainty",
        "validation_issue",
        "regulatory_finding",
        "evidence_incomplete",
        "human_approval_required",
        "release_gate",
    ]
    reason_code: str = Field(min_length=1, max_length=128)
    source_reference_ids: list[str] = Field(default_factory=list, max_length=100)
    status: Literal["open", "resolved", "deferred"] = "open"


class PlannedToolActionV1(V3WireModel):
    action_id: str = Field(pattern=r"^action_[a-f0-9]{64}$")
    tool_name: AllowedToolName
    gap_id: str = Field(pattern=r"^gap_[a-f0-9]{64}$")
    reason_code: str = Field(min_length=1, max_length=128)
    input_reference_ids: list[str] = Field(default_factory=list, max_length=100)


class ReviewPlanProposalV1(V3WireModel):
    plan_id: str = Field(pattern=r"^plan_[a-f0-9]{64}$")
    mode: Literal["shadow", "active"] = "shadow"
    cycle_limit: int = Field(default=2, ge=0, le=2)
    tool_call_limit: int = Field(default=6, ge=0, le=6)
    actions: list[PlannedToolActionV1] = Field(default_factory=list, max_length=6)
    provider: Literal["anthropic", "deterministic_policy"] = "deterministic_policy"
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    output_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0, le=300_000)

    @model_validator(mode="after")
    def _bounded_plan(self) -> "ReviewPlanProposalV1":
        if len(self.actions) > self.tool_call_limit:
            raise ValueError("plan exceeds its declared Tool-call limit")
        action_ids = [item.action_id for item in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("planned action IDs must be unique")
        expected = stable_object_id(
            "plan",
            {
                "mode": self.mode,
                "cycle_limit": self.cycle_limit,
                "tool_call_limit": self.tool_call_limit,
                "actions": [item.model_dump(mode="json") for item in self.actions],
                "provider": self.provider,
                "model_id": self.model_id,
                "prompt_sha256": self.prompt_sha256,
                "output_schema_sha256": self.output_schema_sha256,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "latency_ms": self.latency_ms,
            },
        )
        if self.plan_id != expected:
            raise ValueError("plan ID does not match its canonical contents")
        return self


class ActionTraceV1(V3WireModel):
    action_id: str = Field(pattern=r"^action_[a-f0-9]{64}$")
    cycle: int = Field(ge=0, le=2)
    tool_name: AllowedToolName
    reason_code: str = Field(min_length=1, max_length=128)
    input_reference_ids: list[str] = Field(default_factory=list, max_length=100)
    output_reference_ids: list[str] = Field(default_factory=list, max_length=100)
    status: Literal["planned", "succeeded", "failed", "deferred"]
    duration_ms: int = Field(default=0, ge=0, le=300_000)
    provider: Literal["anthropic", "deterministic_policy"]
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ToolObservationV1(V3WireModel):
    observation_id: str = Field(pattern=r"^observation_[a-f0-9]{64}$")
    action_id: str = Field(pattern=r"^action_[a-f0-9]{64}$")
    cycle: int = Field(ge=0, le=2)
    tool_name: AllowedToolName
    status: Literal["succeeded", "failed", "deferred"]
    reason_code: str = Field(min_length=1, max_length=128)
    input_reference_ids: list[str] = Field(default_factory=list, max_length=100)
    output_reference_ids: list[str] = Field(default_factory=list, max_length=100)


class CompletionEligibilityV1(V3WireModel):
    eligible: bool
    profile_id: str | None = Field(default=None, min_length=1, max_length=256)
    scope: list[str] = Field(default_factory=list, max_length=100)
    unmet_reason_codes: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def _consistent_eligibility(self) -> "CompletionEligibilityV1":
        if self.eligible and (not self.profile_id or self.unmet_reason_codes):
            raise ValueError("eligible completion requires a profile and no unmet reasons")
        if not self.eligible and not self.unmet_reason_codes:
            raise ValueError("ineligible completion requires at least one unmet reason")
        return self


class RuntimeReleaseIdentityV1(V3WireModel):
    agent_package_version: str = Field(min_length=1, max_length=64)
    agent_git_sha: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    agent_source_snapshot_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    agent_code_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_id: str = Field(min_length=1, max_length=128)
    model_revision: str | None = Field(default=None, min_length=1, max_length=128)
    extraction_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    coordinator_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    narrative_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chat_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extraction_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    coordinator_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    narrative_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chat_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_id: str = Field(min_length=1, max_length=128)
    tool_registry_id: str = Field(min_length=1, max_length=128)
    evidence_snapshot_id: str | None = Field(default=None, min_length=1, max_length=256)
    evidence_runtime_v3_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    release_profile_id: str = Field(min_length=1, max_length=256)
    release_profile_state: Literal["unreleased", "approved"]
    release_profile_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    release_profile_subject_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    release_profile_evaluation_report_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    reviewer_trust_store_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    assertion_trust_store_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assertion_trust_store_state: Literal["unreleased", "active"]
    kill_switch_active: bool


class RequiredActionV1(V3WireModel):
    action_code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    source_reference_ids: list[str] = Field(default_factory=list, max_length=100)


class ReviewRunResultV3(ReviewRunResultV2):
    """V3 adds explicit assurance and Agentic audit without weakening V2 invariants."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    schema_version: Literal["3.0"] = "3.0"  # type: ignore[assignment]
    execution_status: Literal["succeeded", "partial", "failed"]
    assessment: Literal["no_issue_detected", "issues_detected", "indeterminate"]
    review_disposition: Literal[
        "auto_review_complete",
        "human_review_required",
        "blocked",
    ]
    assurance_level: Literal["diagnostic", "production_candidate", "production"]
    completion_eligibility: CompletionEligibilityV1
    required_actions: list[RequiredActionV1] = Field(default_factory=list, max_length=100)
    fact_ledger: FactLedgerV1
    review_gaps: list[ReviewGapV1] = Field(default_factory=list, max_length=1000)
    review_plan: ReviewPlanProposalV1
    narrative_report: NarrativeReportV1
    action_trace: list[ActionTraceV1] = Field(default_factory=list, max_length=100)
    observations: list[ToolObservationV1] = Field(default_factory=list, max_length=100)
    runtime_release: RuntimeReleaseIdentityV1
    input_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _v3_policy_invariants(self) -> "ReviewRunResultV3":
        semantics = reconstruct_v3_current_semantics(self)
        expected_current_reference_ids = {
            str(item["reference_id"]) for item in semantics.current_fact_entries
        }
        investigation_entries = [
            item
            for item in self.fact_ledger.entries
            if item.reference_id not in expected_current_reference_ids
            and item.label.startswith("investigation_input:")
        ]
        current_entries = [
            item
            for item in self.fact_ledger.entries
            if item.reference_id in expected_current_reference_ids
            or not item.label.startswith("investigation_input:")
        ]
        if [item.model_dump(mode="json") for item in current_entries] != list(
            semantics.current_fact_entries
        ):
            raise ValueError(
                "current FactLedger must be exactly reconstructed from the V2 review result"
            )
        current_references = {
            (str(item["kind"]), str(item["reference_id"]))
            for item in semantics.current_fact_entries
        }
        claims = [
            *self.narrative_report.summary,
            *self.narrative_report.document_consistency,
            *self.narrative_report.regulatory_review,
            *self.narrative_report.required_actions,
        ]
        for claim in claims:
            if any(
                (citation.kind, citation.reference_id) not in current_references
                for citation in claim.citations
            ):
                raise ValueError("narrative claim must cite a canonical current fact")
        narrative_call = self.narrative_report.model_call
        if self.narrative_report.generation_mode == "claude" and (
            narrative_call is None
            or narrative_call.task_id != "customs_narrative_report"
            or narrative_call.provider != "anthropic"
            or narrative_call.model_id != self.runtime_release.model_id
            or narrative_call.prompt_sha256 != self.runtime_release.narrative_prompt_sha256
            or narrative_call.output_schema_sha256 != self.runtime_release.narrative_schema_sha256
            or narrative_call.input_tokens <= 0
            or narrative_call.output_tokens <= 0
        ):
            raise ValueError("Claude narrative requires exact runtime release identity")
        if self.review_plan.provider == "anthropic":
            if (
                self.review_plan.model_id != self.runtime_release.model_id
                or self.review_plan.prompt_sha256 != self.runtime_release.coordinator_prompt_sha256
                or self.review_plan.output_schema_sha256
                != self.runtime_release.coordinator_schema_sha256
                or self.review_plan.input_tokens <= 0
                or self.review_plan.output_tokens <= 0
            ):
                raise ValueError("Anthropic review plan requires exact runtime release identity")
        elif (
            self.review_plan.model_id is not None
            or self.review_plan.prompt_sha256 is not None
            or self.review_plan.output_schema_sha256 is not None
            or self.review_plan.input_tokens != 0
            or self.review_plan.output_tokens != 0
            or self.review_plan.latency_ms != 0
        ):
            raise ValueError("deterministic review plan cannot claim model execution")
        expected_claims = expected_narrative_claims(
            self,
            semantics,
            release_approved=(
                self.completion_eligibility.eligible
                or set(self.completion_eligibility.unmet_reason_codes)
                == {"narrative_release_identity_incomplete"}
            ),
        )
        for section, expected_section in expected_claims.items():
            actual_section = getattr(self.narrative_report, section)
            if len(actual_section) != len(expected_section):
                raise ValueError("narrative sections must exactly cover code-owned claims")
            for actual_claim, expected_claim in zip(
                actual_section,
                expected_section,
                strict=True,
            ):
                if (
                    not actual_claim.material
                    or [item.model_dump(mode="json") for item in actual_claim.citations]
                    != list(expected_claim["citations"])
                    or (
                        actual_claim.text != str(expected_claim["text"])
                        if self.narrative_report.generation_mode == "deterministic_fallback"
                        else actual_claim.text
                        not in allowed_framed_claim_texts(
                            section,
                            str(expected_claim["text"]),
                        )
                    )
                ):
                    raise ValueError(
                        "narrative claims must use code-owned text frames and exact citations"
                    )
        active_failure_gaps = [
            gap
            for gap in self.review_gaps
            if gap.gap_type == "release_gate" and gap.reason_code in ACTIVE_FAILURE_REASON_CODES
        ]
        if any(
            gap.source_reference_ids != [semantics.outcome_reference_id] or gap.status != "open"
            for gap in active_failure_gaps
        ):
            raise ValueError("active failure gaps must bind the current outcome and remain open")
        active_failure_codes = sorted({gap.reason_code for gap in active_failure_gaps})
        expected_execution_status = (
            "failed"
            if self.status == "blocked"
            else (
                "partial"
                if self.model_errors
                or active_failure_codes
                or any(item.status != "succeeded" for item in self.observations)
                else "succeeded"
            )
        )
        if self.execution_status != expected_execution_status:
            raise ValueError("execution status must be derived from deterministic run failures")
        if (self.status == "blocked") != (self.review_disposition == "blocked"):
            raise ValueError("blocked review disposition must exactly match the V2 blocked status")
        if (self.assurance_level == "production") != self.completion_eligibility.eligible:
            raise ValueError("production assurance must exactly match completion eligibility")
        expected_actions = expected_required_actions(
            self,
            semantics,
            completion_eligible=self.completion_eligibility.eligible,
            supporting_verification_incomplete=(
                "supporting_document_verification_incomplete"
                in self.completion_eligibility.unmet_reason_codes
            ),
            active_failure_codes=active_failure_codes,
        )
        if [item.model_dump(mode="json") for item in self.required_actions] != list(
            expected_actions
        ):
            raise ValueError("required actions must exactly match the code-owned policy projection")
        review_gap_ids = [item.gap_id for item in self.review_gaps]
        if len(review_gap_ids) != len(set(review_gap_ids)):
            raise ValueError("review gap IDs must be unique")
        gap_ids = set(review_gap_ids)
        gaps_by_id = {item.gap_id: item for item in self.review_gaps}
        if any(action.gap_id not in gap_ids for action in self.review_plan.actions):
            raise ValueError("planned actions must reference declared review gaps")
        fact_kind_by_reference_id = {
            item.reference_id: item.kind for item in self.fact_ledger.entries
        }
        fact_reference_ids = set(fact_kind_by_reference_id)
        for gap in self.review_gaps:
            if len(gap.source_reference_ids) != len(set(gap.source_reference_ids)):
                raise ValueError("review gap source references must be unique")
            if gap.source_reference_ids != sorted(gap.source_reference_ids):
                raise ValueError("review gap source references must use canonical sorted order")
            if not set(gap.source_reference_ids).issubset(fact_reference_ids):
                raise ValueError("review gap sources must exist in the canonical fact ledger")
        primary_tool_by_gap_type: dict[str, AllowedToolName] = {
            "field_uncertainty": "document.reread_span/v1",
            "evidence_incomplete": "evidence.retrieve/v3",
            "validation_issue": "human.request/v1",
            "regulatory_finding": "human.request/v1",
            "human_approval_required": "human.request/v1",
            "release_gate": "human.request/v1",
        }
        reread_reference_ids: set[str] = set()
        for action in self.review_plan.actions:
            gap = gaps_by_id[action.gap_id]
            if len(action.input_reference_ids) != len(set(action.input_reference_ids)):
                raise ValueError("planned action input references must be unique")
            if action.input_reference_ids != sorted(action.input_reference_ids):
                raise ValueError("planned action inputs must use canonical sorted order")
            if not set(action.input_reference_ids).issubset(gap.source_reference_ids):
                raise ValueError("planned action inputs must be bound to its review gap")
            if not set(action.input_reference_ids).issubset(fact_reference_ids):
                raise ValueError("planned action inputs must exist in the canonical fact ledger")
            expected_tool = primary_tool_by_gap_type[gap.gap_type]
            if action.tool_name not in {expected_tool, "explanation.compose/v1"}:
                raise ValueError("planned Tool does not match the review gap policy")
            if action.tool_name == "document.reread_span/v1":
                if any(
                    fact_kind_by_reference_id[reference_id] != "field_source"
                    for reference_id in action.input_reference_ids
                ):
                    raise ValueError("field reread inputs must reference field-source facts")
                if reread_reference_ids.intersection(action.input_reference_ids):
                    raise ValueError("a field reference may be reread at most once per plan")
                reread_reference_ids.update(action.input_reference_ids)
            if action.tool_name == "evidence.retrieve/v3" and any(
                fact_kind_by_reference_id[reference_id] not in {"evidence", "outcome"}
                for reference_id in action.input_reference_ids
            ):
                raise ValueError(
                    "Evidence retrieval inputs must reference Evidence or outcome facts"
                )
        planned_input_reference_ids = {
            reference_id
            for action in self.review_plan.actions
            for reference_id in action.input_reference_ids
        }
        current_reference_ids = {
            str(item["reference_id"]) for item in semantics.current_fact_entries
        }
        if {
            item.reference_id for item in investigation_entries
        } != planned_input_reference_ids - current_reference_ids or any(
            item.label == "investigation_input:" for item in investigation_entries
        ):
            raise ValueError(
                "investigation input facts must exactly preserve missing planned inputs"
            )
        planned_ids = {item.action_id for item in self.review_plan.actions}
        if any(item.action_id not in planned_ids for item in self.action_trace):
            raise ValueError("action trace must reference the compiled review plan")
        if any(item.action_id not in planned_ids for item in self.observations):
            raise ValueError("Tool observations must reference the compiled review plan")
        trace_action_ids = [item.action_id for item in self.action_trace]
        if len(trace_action_ids) != len(set(trace_action_ids)):
            raise ValueError("action trace action IDs must be unique")
        planned_action_order = [item.action_id for item in self.review_plan.actions]
        if trace_action_ids != planned_action_order[: len(trace_action_ids)]:
            raise ValueError("action trace must preserve the planned execution prefix order")
        if self.action_trace and (
            self.review_plan.cycle_limit == 0
            or any(item.cycle >= self.review_plan.cycle_limit for item in self.action_trace)
        ):
            raise ValueError("action trace cycle must be within the compiled cycle limit")
        observation_ids = [item.observation_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Tool observation IDs must be unique")
        observation_action_order = [item.action_id for item in self.observations]
        if observation_action_order != trace_action_ids:
            raise ValueError("Tool observations must preserve action trace order")
        if len(self.action_trace) > self.review_plan.tool_call_limit:
            raise ValueError("action trace exceeds the compiled Tool-call limit")
        if len(self.observations) != len(self.action_trace):
            raise ValueError("every Tool action requires exactly one observation")
        planned_by_action_id = {item.action_id: item for item in self.review_plan.actions}
        for trace in self.action_trace:
            planned = planned_by_action_id[trace.action_id]
            if (
                trace.tool_name != planned.tool_name
                or trace.reason_code != planned.reason_code
                or trace.input_reference_ids != planned.input_reference_ids
            ):
                raise ValueError("action trace must exactly match its compiled planned action")
            if trace.tool_name != "explanation.compose/v1" and (
                trace.provider != "deterministic_policy"
                or trace.model_id is not None
                or trace.prompt_sha256 is not None
                or trace.input_tokens != 0
                or trace.output_tokens != 0
            ):
                raise ValueError("non-explanation Tool traces cannot claim model execution")
            if trace.tool_name == "explanation.compose/v1":
                narrative_call = self.narrative_report.model_call
                if trace.provider == "anthropic" and (
                    narrative_call is None
                    or narrative_call.task_id != "customs_narrative_report"
                    or trace.model_id != narrative_call.model_id
                    or trace.model_id != self.runtime_release.model_id
                    or trace.prompt_sha256 != narrative_call.prompt_sha256
                    or trace.prompt_sha256 != self.runtime_release.narrative_prompt_sha256
                    or narrative_call.output_schema_sha256
                    != self.runtime_release.narrative_schema_sha256
                    or trace.input_tokens != narrative_call.input_tokens
                    or trace.output_tokens != narrative_call.output_tokens
                    or trace.duration_ms != narrative_call.latency_ms
                ):
                    raise ValueError(
                        "Anthropic explanation traces require exact narrative call identity"
                    )
                if trace.provider == "deterministic_policy" and (
                    trace.model_id is not None
                    or trace.prompt_sha256 is not None
                    or trace.input_tokens != 0
                    or trace.output_tokens != 0
                ):
                    raise ValueError(
                        "deterministic explanation traces cannot claim model execution"
                    )
        observations_by_action_id = {item.action_id: item for item in self.observations}
        if len(observations_by_action_id) != len(self.observations) or set(
            observations_by_action_id
        ) != set(trace_action_ids):
            raise ValueError("every traced Tool action requires exactly one observation")
        for trace in self.action_trace:
            observation = observations_by_action_id[trace.action_id]
            if len(trace.output_reference_ids) != len(set(trace.output_reference_ids)):
                raise ValueError("Tool output references must be unique")
            narrative_claim_ids = {claim.claim_id for claim in claims}
            allowed_output_reference_ids = {
                *fact_reference_ids,
                *gap_ids,
                *narrative_claim_ids,
            }
            if not set(trace.output_reference_ids).issubset(allowed_output_reference_ids):
                raise ValueError(
                    "Tool outputs must reference canonical facts, gaps, or narrative claims"
                )
            if (
                observation.tool_name != trace.tool_name
                or observation.cycle != trace.cycle
                or observation.status != trace.status
                # The observation reason is the execution outcome code; the
                # trace reason remains the distinct compiled-plan rationale.
                or observation.input_reference_ids != trace.input_reference_ids
                or observation.output_reference_ids != trace.output_reference_ids
            ):
                raise ValueError("Tool observation must exactly match its correlated action trace")
        expected_assessment = (
            "issues_detected"
            if self.validation_issues or self.regulatory_findings
            else (
                "no_issue_detected"
                if self.retrieval_status == "ok"
                and self.retrieval_coverage == "complete"
                and has_citable_regulatory_evidence(self)
                else "indeterminate"
            )
        )
        if self.assessment != expected_assessment:
            raise ValueError("assessment must be derived from deterministic issues and Evidence")
        resolved_gaps = [item for item in self.review_gaps if item.status == "resolved"]
        if (
            self.review_disposition == "auto_review_complete"
            and (resolved_gaps or self.review_plan.actions)
            and self.review_plan.mode != "active"
        ):
            raise ValueError(
                "automatic review completion with investigation requires an active plan"
            )
        if self.review_disposition == "auto_review_complete" and (
            planned_ids != set(trace_action_ids) or planned_ids != set(observations_by_action_id)
        ):
            raise ValueError("automatic review completion requires exact planned Tool coverage")
        if self.review_disposition == "auto_review_complete":
            citable_evidence_ids = set(citable_regulatory_evidence_ids(self))
            if not citable_evidence_ids or any(
                not any(
                    citation.kind == "evidence" and citation.reference_id in citable_evidence_ids
                    for citation in claim.citations
                )
                for claim in self.narrative_report.regulatory_review
            ):
                raise ValueError(
                    "automatic review completion requires current Evidence-cited regulatory claims"
                )
            if (
                self.review_plan.provider != "anthropic"
                or self.review_plan.model_id != self.runtime_release.model_id
                or self.review_plan.prompt_sha256 != self.runtime_release.coordinator_prompt_sha256
                or self.review_plan.output_schema_sha256
                != self.runtime_release.coordinator_schema_sha256
                or self.review_plan.input_tokens <= 0
                or self.review_plan.output_tokens <= 0
            ):
                raise ValueError(
                    "automatic review completion requires exact Claude coordinator audit"
                )
            narrative_call = self.narrative_report.model_call
            if (
                self.narrative_report.generation_mode != "claude"
                or narrative_call is None
                or narrative_call.task_id != "customs_narrative_report"
                or narrative_call.model_id != self.runtime_release.model_id
                or narrative_call.prompt_sha256 != self.runtime_release.narrative_prompt_sha256
                or narrative_call.output_schema_sha256
                != self.runtime_release.narrative_schema_sha256
                or narrative_call.input_tokens <= 0
                or narrative_call.output_tokens <= 0
            ):
                raise ValueError(
                    "automatic review completion requires exact Claude narrative audit"
                )
            successful_action_ids = {
                item.action_id for item in self.action_trace if item.status == "succeeded"
            }
            resolver_tool_by_gap_type: dict[str, AllowedToolName] = {
                "field_uncertainty": "document.reread_span/v1",
                "evidence_incomplete": "evidence.retrieve/v3",
            }
            for gap in resolved_gaps:
                resolver_tool = resolver_tool_by_gap_type.get(gap.gap_type)
                if resolver_tool is None:
                    raise ValueError(
                        "automatic review completion cannot resolve this gap type in-run"
                    )
                if not any(
                    action.gap_id == gap.gap_id
                    and action.tool_name == resolver_tool
                    and bool(action.input_reference_ids)
                    and action.action_id in successful_action_ids
                    for action in self.review_plan.actions
                ):
                    raise ValueError("every resolved gap requires its successful policy resolver")
        if self.review_disposition == "auto_review_complete" and (
            not self.completion_eligibility.eligible
            or self.assessment != "no_issue_detected"
            or self.execution_status != "succeeded"
            or self.required_actions
            or any(gap.status != "resolved" for gap in self.review_gaps)
            or self.validation_issues
            or self.regulatory_findings
            or any(item.status != "succeeded" for item in self.action_trace)
            or any(item.status != "succeeded" for item in self.observations)
            or self.runtime_release.release_profile_state != "approved"
            or self.runtime_release.release_profile_sha256 is None
            or self.runtime_release.release_profile_subject_sha256 is None
            or self.runtime_release.kill_switch_active
            or self.runtime_release.evidence_runtime_v3_sha256 is None
            or self.status != "completed"
        ):
            raise ValueError("automatic review completion requires every deterministic Gate")
        if (
            self.assurance_level != "production"
            and self.review_disposition == "auto_review_complete"
        ):
            raise ValueError("non-production assurance cannot auto-complete a review")
        for gap in self.review_gaps:
            expected_gap_id = stable_object_id(
                "gap",
                {
                    "gap_type": gap.gap_type,
                    "reason_code": gap.reason_code,
                    "source_reference_ids": gap.source_reference_ids,
                },
            )
            if gap.gap_id != expected_gap_id:
                raise ValueError("review gap ID does not match its canonical contents")
        for action in self.review_plan.actions:
            expected_action_id = stable_object_id(
                "action",
                action.model_dump(mode="json", exclude={"action_id"}),
            )
            if action.action_id != expected_action_id:
                raise ValueError("planned action ID does not match its canonical contents")
        for observation in self.observations:
            expected_observation_id = stable_object_id(
                "observation",
                observation.model_dump(mode="json", exclude={"observation_id"}),
            )
            if observation.observation_id != expected_observation_id:
                raise ValueError("Tool observation ID does not match its canonical contents")
        expected_hash = review_result_v3_sha256(self)
        if self.result_sha256 != expected_hash:
            raise ValueError("result_sha256 does not match the canonical V3 result")
        return self


class GroundedQuestionV1(V3WireModel):
    question_id: str = Field(pattern=r"^question_[a-f0-9]{64}$")
    question: str = Field(min_length=1, max_length=2000)
    locale: Literal["ko-KR"] = "ko-KR"
    review_result: ReviewRunResultV3


class GroundedAnswerV1(V3WireModel):
    question_id: str = Field(pattern=r"^question_[a-f0-9]{64}$")
    answer: str = Field(min_length=1, max_length=8000)
    status: Literal["answered", "insufficient_evidence", "out_of_scope"]
    citations: list[CitationReferenceV1] = Field(default_factory=list, max_length=100)
    generation_mode: Literal["claude", "deterministic_fallback"]
    citation_verification: Literal["passed"] = "passed"
    model_call: AgenticCallAuditV1 | None = None

    @model_validator(mode="after")
    def _answered_requires_citations(self) -> "GroundedAnswerV1":
        if self.status == "answered" and not self.citations:
            raise ValueError("answered Case questions require grounded citations")
        if self.status != "answered" and self.citations:
            raise ValueError("refused Case questions cannot claim citations")
        if (self.generation_mode == "claude") != (self.model_call is not None):
            raise ValueError("Claude Case answers require complete model-call audit")
        return self


class HumanDecisionV1(V3WireModel):
    decision_id: str = Field(pattern=r"^decision_[a-f0-9]{64}$")
    decision: Literal[
        "confirm_field",
        "correct_field",
        "add_supporting_document",
        "approve_review",
        "reject_review",
    ]
    actor_id: str = Field(min_length=1, max_length=256)
    actor_role: Literal["reviewer", "approver"]
    decided_at: datetime
    reason: str = Field(min_length=1, max_length=2000)
    confirmed_reference_id: str | None = Field(default=None, min_length=1, max_length=256)
    correction: UserCorrection | None = None
    supporting_document: SupportingDocumentV1 | None = None

    @model_validator(mode="after")
    def _correction_only_for_correct_field(self) -> "HumanDecisionV1":
        if self.decided_at.tzinfo is None:
            raise ValueError("human decision time must include a timezone")
        if (self.decision == "correct_field") != (self.correction is not None):
            raise ValueError("only correct_field decisions contain a correction")
        if (self.decision == "add_supporting_document") != (self.supporting_document is not None):
            raise ValueError("only add_supporting_document decisions contain a document")
        if (self.decision == "confirm_field") != (self.confirmed_reference_id is not None):
            raise ValueError("only confirm_field decisions contain a confirmed reference")
        if self.decision == "approve_review" and self.actor_role != "approver":
            raise ValueError("review approval requires the approver role")
        payload = self.model_dump(mode="json", exclude={"decision_id"})
        if self.decision_id != stable_object_id("decision", payload):
            raise ValueError("human decision ID must bind its canonical contents")
        return self


class ReviewRevisionRequestV1(V3WireModel):
    parent_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_input_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_result: ReviewRunResultV3
    parent_request: ServiceReviewRunRequestV3
    revised_request: ServiceReviewRunRequestV3
    decisions: list[HumanDecisionV1] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _correlated_revision(self) -> "ReviewRevisionRequestV1":
        if self.parent_result.result_sha256 != self.parent_result_sha256:
            raise ValueError("parent result hash is stale or mismatched")
        if self.parent_result.input_manifest_sha256 != self.parent_input_manifest_sha256:
            raise ValueError("parent input manifest hash is stale or mismatched")
        if canonical_json_sha256(self.parent_request) != self.parent_input_manifest_sha256:
            raise ValueError("parent request does not match the immutable input manifest")
        if self.revised_request.case_id != self.parent_result.case_id:
            raise ValueError("revision cannot change its parent Case identity")
        if self.revised_request.review_run_id == self.parent_result.review_run_id:
            raise ValueError("revision requires a new child review_run_id")
        decision_ids = [item.decision_id for item in self.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("revision decision IDs must be unique")
        terminal = [
            item for item in self.decisions if item.decision in {"approve_review", "reject_review"}
        ]
        if len(terminal) > 1:
            raise ValueError("revision permits at most one terminal human decision")
        _assert_revision_is_decision_compiled(self)
        return self


def _assert_revision_is_decision_compiled(revision: ReviewRevisionRequestV1) -> None:
    expected = revision.parent_request.model_copy(deep=True)
    expected.review_run_id = revision.revised_request.review_run_id
    additions = [
        decision.supporting_document
        for decision in revision.decisions
        if decision.decision == "add_supporting_document"
        and decision.supporting_document is not None
    ]
    expected.supporting_documents = [*expected.supporting_documents, *additions]
    expected = ServiceReviewRunRequestV3.model_validate(
        expected.model_dump(mode="python", warnings=False)
    )
    if expected != revision.revised_request:
        raise ValueError("revised request contains a change not compiled from approved decisions")

    parent_fields = {
        (document.document_id, f"fields.{field.name}")
        for document in revision.parent_result.document_results
        for field in document.fields
    }
    parent_line_fields = {
        (document.document_id, f"line_items.{index}.{field.name}")
        for document in revision.parent_result.document_results
        for index, item in enumerate(document.line_items)
        for field in item.fields
    }
    allowed_paths = parent_fields | parent_line_fields
    allowed_confirmations = {
        item.reference_id
        for item in revision.parent_result.fact_ledger.entries
        if item.kind == "field_source"
    }
    for decision in revision.decisions:
        if (
            decision.confirmed_reference_id is not None
            and decision.confirmed_reference_id not in allowed_confirmations
        ):
            raise ValueError("confirmation must reference an existing public result field")
        if decision.correction is None:
            continue
        key = (decision.correction.document_id, decision.correction.field_path)
        if key not in allowed_paths:
            raise ValueError("correction must reference an existing public result field")


class ReviewRevisionResultV1(V3WireModel):
    parent_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    child_result: ReviewRunResultV3
    applied_decision_ids: list[str] = Field(min_length=1, max_length=100)


def canonical_json_sha256(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_canonical_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def stable_object_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{canonical_json_sha256(value)}"


def review_result_v3_sha256(result: ReviewRunResultV3 | dict[str, Any]) -> str:
    payload = (
        result.model_dump(mode="json", warnings=False)
        if isinstance(result, BaseModel)
        else dict(result)
    )
    payload["result_sha256"] = "0" * 64
    return canonical_json_sha256(payload)


__all__ = [
    "ActionTraceV1",
    "AgenticCallAuditV1",
    "AllowedToolName",
    "CitationReferenceV1",
    "CompletionEligibilityV1",
    "FactLedgerEntryV1",
    "FactLedgerV1",
    "GroundedAnswerV1",
    "GroundedQuestionV1",
    "HumanDecisionV1",
    "NarrativeClaimV1",
    "NarrativeReportV1",
    "PlannedToolActionV1",
    "RequiredActionV1",
    "ReviewGapV1",
    "ReviewPlanProposalV1",
    "ReviewRevisionRequestV1",
    "ReviewRevisionResultV1",
    "ReviewRunResultV3",
    "RuntimeReleaseIdentityV1",
    "ServiceReviewRunRequestV3",
    "SupportingDocumentV1",
    "TextDocumentInputV3",
    "ToolObservationV1",
    "VerificationAssertionV1",
    "VerificationReceiptV1",
    "canonical_json_sha256",
    "review_result_v3_sha256",
    "stable_object_id",
    "supporting_document_claims_sha256",
    "verification_assertion_unsigned_payload",
    "verification_receipt_unsigned_payload",
]
