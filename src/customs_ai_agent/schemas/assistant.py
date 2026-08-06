"""Small, internal wire models for one grounded assistant turn.

The Backend owns persisted conversations.  The Agent receives only a validated
review snapshot, one question, and at most eight recent persisted turns.  It
cannot mutate Case state or invoke an open-ended tool surface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.schemas.v3 import EngineV3, ReviewRunResultV3


GENERAL_GUIDANCE_DISCLAIMER = "이 안내는 현재 사건의 판단이나 결론이 아닙니다."


class AssistantWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssistantRecentTurnV1(AssistantWireModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AssistantCitationV1(AssistantWireModel):
    kind: Literal["fact", "comparison", "evidence"]
    reference_id: str = Field(min_length=1, max_length=256)


class AssistantCorrectionDraftV1(AssistantWireModel):
    """A non-mutating draft which Backend may present as a proposal form."""

    document_id: str = Field(min_length=1, max_length=128)
    item_id: str | None = Field(default=None, max_length=256)
    fact_id: str | None = Field(default=None, max_length=256)
    field_name: str = Field(min_length=1, max_length=64)
    new_value: str = Field(min_length=1, max_length=2000)
    unit: str | None = Field(default=None, max_length=64)
    reason: str = Field(min_length=1, max_length=2000)


class AssistantAnswerRequestV1(AssistantWireModel):
    schema_version: Literal["1.0"] = "1.0"
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
    conversation_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_result: ReviewRunResultV3
    question: str = Field(min_length=1, max_length=4000)
    recent_turns: list[AssistantRecentTurnV1] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _snapshot_matches_route_identity(self) -> "AssistantAnswerRequestV1":
        if self.review_result.case_id != self.case_id:
            raise ValueError("review_result case_id must match request case_id")
        if self.review_result.review_run_id != self.review_run_id:
            raise ValueError("review_result review_run_id must match request review_run_id")
        return self


class AssistantAnswerResponseV1(AssistantWireModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1, max_length=128)
    review_run_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    answer_scope: Literal[
        "case_grounded",
        "evidence_grounded",
        "general_customs_guidance",
        "clarification_needed",
        "insufficient_evidence",
        "out_of_scope",
    ]
    direct_answer: str = Field(min_length=1, max_length=4000)
    citations: list[AssistantCitationV1] = Field(default_factory=list, max_length=32)
    suggested_questions: list[str] = Field(default_factory=list, max_length=3)
    correction_draft: AssistantCorrectionDraftV1 | None = None
    engine: EngineV3

    @model_validator(mode="after")
    def _grounded_scopes_have_citations(self) -> "AssistantAnswerResponseV1":
        if self.answer_scope in {"case_grounded", "evidence_grounded"} and not self.citations:
            raise ValueError("grounded assistant answers require a citation")
        identities = [(citation.kind, citation.reference_id) for citation in self.citations]
        if len(identities) != len(set(identities)):
            raise ValueError("assistant citations must be unique")
        if self.answer_scope == "case_grounded" and not any(
            citation.kind in {"fact", "comparison"} for citation in self.citations
        ):
            raise ValueError("case-grounded answers require a fact or comparison citation")
        if self.answer_scope == "evidence_grounded" and not any(
            citation.kind == "evidence" for citation in self.citations
        ):
            raise ValueError("evidence-grounded answers require an evidence citation")
        if self.answer_scope not in {"case_grounded", "evidence_grounded"} and self.citations:
            raise ValueError("non-grounded assistant answers must not include citations")
        if self.answer_scope == "general_customs_guidance":
            if GENERAL_GUIDANCE_DISCLAIMER not in self.direct_answer:
                raise ValueError("general guidance must include the non-case disclaimer")
            if self.correction_draft is not None:
                raise ValueError("general guidance must not contain a correction draft")
        elif self.correction_draft is not None and self.answer_scope != "case_grounded":
            raise ValueError("correction drafts require a case-grounded answer")
        return self


__all__ = [
    "AssistantAnswerRequestV1",
    "AssistantAnswerResponseV1",
    "AssistantCitationV1",
    "AssistantCorrectionDraftV1",
    "AssistantRecentTurnV1",
    "GENERAL_GUIDANCE_DISCLAIMER",
]
