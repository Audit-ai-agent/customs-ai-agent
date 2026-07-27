from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from customs_ai_agent.schemas.v3 import (
    CitationReferenceV1,
    ReviewRunResultV3,
    V3WireModel,
    canonical_json_sha256,
    review_result_v3_sha256,
    stable_object_id,
)


PresentationSectionCode = Literal[
    "conclusion",
    "process",
    "document_comparison",
    "issues_and_causes",
    "regulatory_meaning",
    "business_impact",
    "next_actions",
    "scope_and_limits",
]
PresentationStatementRole = Literal[
    "conclusion",
    "observation",
    "reason",
    "evidence",
    "impact",
    "action",
    "limitation",
]
AnswerScopeV2 = Literal[
    "case_grounded",
    "evidence_grounded",
    "general_customs_guidance",
    "clarification_needed",
    "insufficient_evidence",
    "out_of_scope",
]
ReviewStageCode = Literal[
    "request_validation",
    "document_analysis",
    "normalization",
    "document_pairing",
    "cross_validation",
    "adaptive_investigation",
    "evidence_retrieval",
    "compliance_review",
    "report_composition",
    "grounding_verification",
    "result_validation",
    "conversation_binding",
    "answer_planning",
    "answer_composition",
    "completed",
    "failed",
]


class AnswerScopeClassificationAuditV1(V3WireModel):
    mode: Literal["claude", "deterministic_guard", "fallback"]
    task_id: Literal["customs_case_scope_classifier"]
    classified_scope: AnswerScopeV2
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    semantic_output_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0, le=300_000)
    reason_code: str = Field(min_length=1, max_length=128)
    guidance_topics: list[
        Literal[
            "invoice",
            "packing_list",
            "bill_of_lading",
            "certificate_of_origin",
            "hsk",
            "hs_hsk_comparison",
            "customs_review",
            "import_declaration",
            "customs_valuation",
            "duties_and_taxes",
            "free_trade_agreement",
            "import_requirements",
            "inspection_and_supplement",
            "document_correction",
            "customs_broker",
        ]
    ] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _classification_audit_is_complete(self) -> "AnswerScopeClassificationAuditV1":
        if self.mode == "claude":
            if self.model_id is None or self.input_tokens <= 0 or self.output_tokens <= 0:
                raise ValueError("Claude scope classification requires complete model audit")
        elif self.model_id is not None or self.input_tokens or self.output_tokens:
            raise ValueError("non-model scope classification cannot claim a model call")
        if self.mode == "deterministic_guard" and self.classified_scope != "out_of_scope":
            raise ValueError("the deterministic scope guard is reserved for blocked input")
        if self.mode == "fallback" and self.classified_scope != "clarification_needed":
            raise ValueError("scope-classification fallback must request clarification")
        if self.classified_scope == "general_customs_guidance":
            if not self.guidance_topics:
                raise ValueError("general guidance classification requires a topic")
        elif self.guidance_topics:
            raise ValueError("guidance topics are allowed only for general guidance")
        return self


class ExperienceGenerationAuditV1(V3WireModel):
    mode: Literal["claude", "deterministic_fallback"]
    task_id: str = Field(min_length=1, max_length=128)
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0, le=300_000)
    attempt_count: int = Field(default=1, ge=1, le=2)
    scope_classification: AnswerScopeClassificationAuditV1 | None = None

    @model_validator(mode="after")
    def _mode_matches_model_audit(self) -> "ExperienceGenerationAuditV1":
        if self.mode == "claude":
            if self.model_id is None or self.input_tokens <= 0 or self.output_tokens <= 0:
                raise ValueError("Claude experience generation requires complete model audit")
        elif self.model_id is not None or self.input_tokens or self.output_tokens:
            raise ValueError("deterministic experience generation cannot claim a model call")
        return self


class ExperienceVerificationAuditV1(V3WireModel):
    status: Literal["passed", "fallback"]
    deterministic_checks_passed: bool
    entailment_verifier: Literal["claude", "not_run"]
    verifier_model_id: str | None = Field(default=None, min_length=1, max_length=128)
    verifier_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verifier_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verifier_context_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    verified_content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    checked_statement_count: int = Field(ge=0, le=1000)
    reason_codes: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def _verifier_identity_is_consistent(self) -> "ExperienceVerificationAuditV1":
        claude = self.entailment_verifier == "claude"
        if claude != (
            self.verifier_model_id is not None and self.verifier_context_sha256 is not None
        ):
            raise ValueError("entailment verifier identity is incomplete")
        if not claude and self.verifier_context_sha256 is not None:
            raise ValueError("a non-model verifier cannot claim a context identity")
        if self.status == "passed" and (not self.deterministic_checks_passed or self.reason_codes):
            raise ValueError("passed verification cannot contain blockers")
        if self.status == "fallback" and not self.reason_codes:
            raise ValueError("fallback verification requires a reason")
        return self


class CanonicalClaimV1(V3WireModel):
    claim_id: str = Field(pattern=r"^experience_claim_[a-f0-9]{64}$")
    section_code: PresentationSectionCode
    role: PresentationStatementRole
    canonical_text: str = Field(min_length=1, max_length=4000)
    citation_references: list[CitationReferenceV1] = Field(
        default_factory=list,
        max_length=100,
    )
    material: bool = True

    @model_validator(mode="after")
    def _claim_identity_and_grounding(self) -> "CanonicalClaimV1":
        if self.material and not self.citation_references:
            raise ValueError("material presentation claims require citations")
        payload = self.model_dump(mode="json", exclude={"claim_id"}, warnings=False)
        if self.claim_id != stable_object_id("experience_claim", payload):
            raise ValueError("experience claim ID does not match its canonical contents")
        return self


class GroundedPresentationStatementV1(V3WireModel):
    statement_id: str = Field(pattern=r"^statement_[a-f0-9]{64}$")
    role: PresentationStatementRole
    text: str = Field(min_length=1, max_length=4000)
    source_claim_ids: list[str] = Field(
        min_length=1,
        max_length=20,
    )
    citation_references: list[CitationReferenceV1] = Field(
        default_factory=list,
        max_length=100,
    )
    material: bool = True

    @model_validator(mode="after")
    def _statement_identity_and_grounding(self) -> "GroundedPresentationStatementV1":
        if len(self.source_claim_ids) != len(set(self.source_claim_ids)):
            raise ValueError("presentation source claim IDs must be unique")
        if self.material and not self.citation_references:
            raise ValueError("material presentation statements require citations")
        payload = self.model_dump(mode="json", exclude={"statement_id"}, warnings=False)
        if self.statement_id != stable_object_id("statement", payload):
            raise ValueError("presentation statement ID does not match its canonical contents")
        return self


class GroundedPresentationSectionV1(V3WireModel):
    section_code: PresentationSectionCode
    title: str = Field(min_length=1, max_length=120)
    statements: list[GroundedPresentationStatementV1] = Field(
        min_length=1,
        max_length=100,
    )


def presentation_verified_content_sha256(
    headline: str,
    sections: Sequence[GroundedPresentationSectionV1],
) -> str:
    """Hash the exact report content that the verifier approved."""

    return canonical_json_sha256(
        {
            "headline": headline,
            "sections": [section.model_dump(mode="json", warnings=False) for section in sections],
        }
    )


class GroundedPresentationV1(V3WireModel):
    schema_version: Literal["1.0"] = "1.0"
    headline: str = Field(min_length=1, max_length=300)
    sections: list[GroundedPresentationSectionV1] = Field(min_length=8, max_length=8)
    generation_audit: ExperienceGenerationAuditV1
    verification_audit: ExperienceVerificationAuditV1
    presentation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _presentation_is_complete_and_hashed(self) -> "GroundedPresentationV1":
        expected_order: list[PresentationSectionCode] = [
            "conclusion",
            "process",
            "document_comparison",
            "issues_and_causes",
            "regulatory_meaning",
            "business_impact",
            "next_actions",
            "scope_and_limits",
        ]
        if [item.section_code for item in self.sections] != expected_order:
            raise ValueError("presentation sections must be complete and ordered")
        statement_ids = [
            statement.statement_id for section in self.sections for statement in section.statements
        ]
        if len(statement_ids) != len(set(statement_ids)):
            raise ValueError("presentation statement IDs must be unique")
        if self.verification_audit.verified_content_sha256 != (
            presentation_verified_content_sha256(self.headline, self.sections)
        ):
            raise ValueError("presentation verifier receipt is not bound to its visible content")
        expected = canonical_json_sha256(
            self.model_dump(
                mode="json",
                exclude={"presentation_sha256"},
                warnings=False,
            )
        )
        if self.presentation_sha256 != expected:
            raise ValueError("presentation hash does not match its canonical contents")
        return self


class AgentUatCapabilityAttestationV1(V3WireModel):
    schema_version: Literal["1.0"] = "1.0"
    profile_id: str = Field(min_length=1, max_length=256)
    environment: Literal["local_uat"]
    evidence_profile: Literal["preserved-3law-v3-scenario-eval"]
    loopback_only: Literal[True] = True
    issued_at: datetime
    expires_at: datetime
    agent_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    agent_code_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_id: str = Field(min_length=1, max_length=128)
    prompt_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    schema_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tool_registry_id: str = Field(min_length=1, max_length=128)
    tool_registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_runtime_v3_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_cycles: Literal[2] = 2
    max_tool_calls: Literal[6] = 6
    max_rereads_per_field: Literal[1] = 1
    adaptive_active: bool
    auto_completion_enabled: Literal[False] = False
    capability_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _valid_local_uat_capability(self) -> "AgentUatCapabilityAttestationV1":
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("UAT capability times must be timezone-aware")
        lifetime = (self.expires_at - self.issued_at).total_seconds()
        if lifetime <= 0 or lifetime > 86_400:
            raise ValueError("UAT capability lifetime must be within 24 hours")
        expected = canonical_json_sha256(
            self.model_dump(
                mode="json",
                exclude={"capability_sha256"},
                warnings=False,
            )
        )
        if self.capability_sha256 != expected:
            raise ValueError("UAT capability hash does not match its canonical contents")
        return self


class UserActionEventV1(V3WireModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(pattern=r"^user_action_[a-f0-9]{64}$")
    action_code: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    status: Literal["completed", "needs_attention", "deferred"]
    source_reference_ids: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def _event_identity(self) -> "UserActionEventV1":
        payload = self.model_dump(mode="json", exclude={"event_id"}, warnings=False)
        if self.event_id != stable_object_id("user_action", payload):
            raise ValueError("user action event ID does not match its canonical contents")
        return self


class ReviewExperienceEnvelopeV1(V3WireModel):
    schema_version: Literal["1.0"] = "1.0"
    canonical_result: ReviewRunResultV3
    user_presentation: GroundedPresentationV1
    capability_attestation: AgentUatCapabilityAttestationV1
    user_action_events: list[UserActionEventV1] = Field(default_factory=list, max_length=100)
    envelope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _envelope_is_bound_to_canonical_result(self) -> "ReviewExperienceEnvelopeV1":
        if review_result_v3_sha256(self.canonical_result) != self.canonical_result.result_sha256:
            raise ValueError("experience envelope contains a stale canonical result")
        valid_claim_ids = {
            claim_id
            for section in self.user_presentation.sections
            for statement in section.statements
            for claim_id in statement.source_claim_ids
        }
        if not valid_claim_ids:
            raise ValueError("experience presentation contains no claim references")
        fact_refs = {
            (entry.kind, entry.reference_id) for entry in self.canonical_result.fact_ledger.entries
        }
        for section in self.user_presentation.sections:
            for statement in section.statements:
                if any(
                    (citation.kind, citation.reference_id) not in fact_refs
                    for citation in statement.citation_references
                ):
                    raise ValueError("experience statement cites an unknown canonical fact")
        expected = canonical_json_sha256(
            self.model_dump(
                mode="json",
                exclude={"envelope_sha256"},
                warnings=False,
            )
        )
        if self.envelope_sha256 != expected:
            raise ValueError("experience envelope hash does not match its canonical contents")
        return self


class ConversationTurnV2(V3WireModel):
    message_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    parent_message_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)
    created_at: datetime
    answer_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _assistant_turn_is_hashed(self) -> "ConversationTurnV2":
        if self.created_at.tzinfo is None:
            raise ValueError("conversation turn time must be timezone-aware")
        if (self.role == "assistant") != (self.answer_sha256 is not None):
            raise ValueError("only assistant turns carry an answer hash")
        return self


def conversation_context_sha256(turns: list[ConversationTurnV2]) -> str:
    return canonical_json_sha256([turn.model_dump(mode="json", warnings=False) for turn in turns])


class CaseConversationRequestV2(V3WireModel):
    schema_version: Literal["2.0"] = "2.0"
    conversation_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    parent_message_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    question: str = Field(min_length=1, max_length=2000)
    bound_result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    conversation_context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    recent_turns: list[ConversationTurnV2] = Field(default_factory=list, max_length=8)
    review_result: ReviewRunResultV3
    locale: Literal["ko-KR"] = "ko-KR"

    @model_validator(mode="after")
    def _conversation_is_result_and_parent_bound(self) -> "CaseConversationRequestV2":
        if (
            self.bound_result_sha256 != self.review_result.result_sha256
            or review_result_v3_sha256(self.review_result) != self.bound_result_sha256
        ):
            raise ValueError("conversation request is bound to a stale review result")
        if self.conversation_context_sha256 != conversation_context_sha256(self.recent_turns):
            raise ValueError("conversation context hash does not match recent turns")
        message_ids = [turn.message_id for turn in self.recent_turns]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("conversation message IDs must be unique")
        expected_parent = message_ids[-1] if message_ids else None
        if self.parent_message_id != expected_parent:
            raise ValueError("conversation parent message does not match the latest turn")
        return self


def case_conversation_request_sha256(request: CaseConversationRequestV2) -> str:
    """Canonical binding for one question, result, and Backend-owned context."""

    return canonical_json_sha256(request.model_dump(mode="json", warnings=False))


_ANSWER_VERIFIED_CONTENT_FIELDS = (
    "request_sha256",
    "answer_scope",
    "status",
    "direct_answer",
    "explanation",
    "evidence_meaning",
    "impact",
    "next_steps",
    "limitations",
    "claim_selections",
    "citations",
    "suggested_questions",
    "used_context_message_ids",
)


def assistant_answer_verified_content_sha256(values: Mapping[str, Any]) -> str:
    """Hash every user-visible or lineage-bearing value approved by the verifier."""

    return canonical_json_sha256(
        {field: values[field] for field in _ANSWER_VERIFIED_CONTENT_FIELDS}
    )


class AnswerClaimSelectionsV1(V3WireModel):
    """Canonical claim IDs selected for each visible answer area."""

    direct_answer: list[str] = Field(default_factory=list, max_length=20)
    explanation: list[str] = Field(default_factory=list, max_length=40)
    evidence_meaning: list[str] = Field(default_factory=list, max_length=40)
    impact: list[str] = Field(default_factory=list, max_length=40)
    next_steps: list[str] = Field(default_factory=list, max_length=40)

    @field_validator(
        "direct_answer",
        "explanation",
        "evidence_meaning",
        "impact",
        "next_steps",
    )
    @classmethod
    def _claim_ids_are_unique_and_typed(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(
            re.fullmatch(r"experience_claim_[a-f0-9]{64}", item) is None for item in value
        ):
            raise ValueError("answer claim selections contain an invalid claim ID")
        return value


class AssistantAnswerV2(V3WireModel):
    schema_version: Literal["2.0"] = "2.0"
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_scope: AnswerScopeV2
    status: Literal[
        "answered",
        "clarification_needed",
        "insufficient_evidence",
        "out_of_scope",
    ]
    direct_answer: str = Field(min_length=1, max_length=4000)
    explanation: list[str] = Field(default_factory=list, max_length=20)
    evidence_meaning: list[str] = Field(default_factory=list, max_length=20)
    impact: list[str] = Field(default_factory=list, max_length=20)
    next_steps: list[str] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=20)
    claim_selections: AnswerClaimSelectionsV1 = Field(default_factory=AnswerClaimSelectionsV1)
    citations: list[CitationReferenceV1] = Field(default_factory=list, max_length=100)
    suggested_questions: list[str] = Field(default_factory=list, max_length=3)
    used_context_message_ids: list[str] = Field(default_factory=list, max_length=8)
    generation_audit: ExperienceGenerationAuditV1
    verification_audit: ExperienceVerificationAuditV1
    answer_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _answer_is_grounded_and_hashed(self) -> "AssistantAnswerV2":
        if (
            self.generation_audit.scope_classification is None
            or self.generation_audit.scope_classification.classified_scope != self.answer_scope
        ):
            raise ValueError("assistant answer is missing its exact scope-classification audit")
        if self.answer_scope in {"case_grounded", "evidence_grounded"} and not self.citations:
            raise ValueError("case and Evidence answers require citations")
        if self.status in {"clarification_needed", "out_of_scope"} and self.citations:
            raise ValueError("clarification and out-of-scope answers cannot claim citations")
        if self.status == "answered" and self.answer_scope in {
            "clarification_needed",
            "insufficient_evidence",
            "out_of_scope",
        }:
            raise ValueError("answer status is inconsistent with its scope")
        values = self.model_dump(mode="json", warnings=False)
        if self.verification_audit.verified_content_sha256 != (
            assistant_answer_verified_content_sha256(values)
        ):
            raise ValueError("assistant verifier receipt is not bound to its visible content")
        expected = canonical_json_sha256(
            self.model_dump(mode="json", exclude={"answer_sha256"}, warnings=False)
        )
        if self.answer_sha256 != expected:
            raise ValueError("assistant answer hash does not match its canonical contents")
        return self


class ReviewStageEventV1(V3WireModel):
    schema_version: Literal["1.0"] = "1.0"
    operation_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    sequence: int = Field(ge=0, le=10_000)
    stage_code: ReviewStageCode
    state: Literal["started", "completed", "skipped", "failed"]
    occurred_at: datetime
    duration_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    safe_counts: dict[str, int] = Field(default_factory=dict)
    message_key: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_.-]*$",
    )
    event_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _safe_and_hashed_event(self) -> "ReviewStageEventV1":
        if self.occurred_at.tzinfo is None:
            raise ValueError("stage event time must be timezone-aware")
        if len(self.safe_counts) > 20 or any(
            not key.isascii() or value < 0 for key, value in self.safe_counts.items()
        ):
            raise ValueError("stage event safe counts are invalid")
        if self.state == "started" and self.duration_ms is not None:
            raise ValueError("started stage events cannot claim a duration")
        expected = canonical_json_sha256(
            self.model_dump(mode="json", exclude={"event_sha256"}, warnings=False)
        )
        if self.event_sha256 != expected:
            raise ValueError("stage event hash does not match its canonical contents")
        return self


def hashed_model_payload(
    model: type[V3WireModel],
    payload: dict[str, Any],
    *,
    hash_field: str,
) -> V3WireModel:
    complete = dict(payload)
    complete[hash_field] = canonical_json_sha256(payload)
    return model.model_validate(complete)


__all__ = [
    "AgentUatCapabilityAttestationV1",
    "AnswerScopeClassificationAuditV1",
    "AnswerScopeV2",
    "AnswerClaimSelectionsV1",
    "AssistantAnswerV2",
    "CanonicalClaimV1",
    "CaseConversationRequestV2",
    "ConversationTurnV2",
    "ExperienceGenerationAuditV1",
    "ExperienceVerificationAuditV1",
    "GroundedPresentationSectionV1",
    "GroundedPresentationStatementV1",
    "GroundedPresentationV1",
    "PresentationSectionCode",
    "PresentationStatementRole",
    "ReviewExperienceEnvelopeV1",
    "ReviewStageCode",
    "ReviewStageEventV1",
    "UserActionEventV1",
    "conversation_context_sha256",
    "hashed_model_payload",
]
