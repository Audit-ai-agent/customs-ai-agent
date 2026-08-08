"""Small, internal wire models for one grounded assistant turn.

The Backend owns persisted conversations.  The Agent receives only a validated
review snapshot, one question, and at most eight recent persisted turns.  It
cannot mutate Case state or invoke an open-ended tool surface.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.schemas.v3 import EngineV3, ReviewRunResultV3


GENERAL_GUIDANCE_DISCLAIMER = "이 안내는 현재 사건의 판단이나 결론이 아닙니다."
GuidanceSourceKind = Literal[
    "official_law",
    "official_law_catalog",
    "official_agency_guidance",
    "official_requirement_reference",
    "official_public_guidance",
    "development_fixture",
]


class AssistantWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssistantRecentTurnV1(AssistantWireModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AssistantCitationV1(AssistantWireModel):
    kind: Literal["fact", "comparison", "evidence"]
    reference_id: str = Field(min_length=1, max_length=256)


class GuidanceEvidenceV1(AssistantWireModel):
    """One versioned, official source returned by the Data guidance boundary.

    This is intentionally separate from ``AssistantCitationV1``.  A case
    citation points into an immutable review result, whereas a guidance
    citation identifies a source in a versioned official-knowledge snapshot.
    The Agent never turns a source identifier into a direct database or web
    access capability.
    """

    source_id: str = Field(min_length=1, max_length=128)
    source_kind: GuidanceSourceKind
    title: str = Field(min_length=1, max_length=1_000)
    uri: str = Field(min_length=1, max_length=2_000)
    topic: str = Field(min_length=1, max_length=128)
    direction: Literal["import", "export", "both"]
    excerpt: str = Field(min_length=1, max_length=2_000)
    as_of: date
    snapshot_id: str = Field(min_length=1, max_length=512)
    law_id: str | None = Field(default=None, max_length=512)
    law_name: str | None = Field(default=None, max_length=500)
    article_no: str | None = Field(default=None, max_length=128)
    version_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _official_law_provenance_is_precise(self) -> "GuidanceEvidenceV1":
        law_values = (self.law_id, self.law_name, self.article_no)
        if self.source_kind == "official_law" and any(value is None for value in law_values):
            raise ValueError("official law guidance requires law provenance")
        if self.source_kind != "official_law" and any(value is not None for value in law_values):
            raise ValueError("non-law guidance must not claim law provenance")
        return self


class GuidanceCitationV1(AssistantWireModel):
    """Compact public citation for one Data-approved official source."""

    kind: Literal["guidance_evidence"] = "guidance_evidence"
    source_id: str = Field(min_length=1, max_length=256)
    source_kind: GuidanceSourceKind
    source_title: str = Field(min_length=1, max_length=1_000)
    source_uri: str = Field(min_length=1, max_length=2_000)
    as_of: date
    snapshot_id: str = Field(min_length=1, max_length=512)
    coverage: Literal["covered", "no_coverage", "partial", "unavailable"]


AssistantAnswerCitationV1 = AssistantCitationV1 | GuidanceCitationV1


class AssistantAnswerSectionV1(AssistantWireModel):
    """A user-facing section for a case, official guidance, or both.

    Sections are additive to the legacy ``direct_answer`` field so old
    persisted conversations remain readable.  New replies use at most one
    case section and one general-guidance section.
    """

    kind: Literal["case_analysis", "general_guidance"]
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4_000)
    citations: list[AssistantAnswerCitationV1] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def _section_has_its_own_grounding(self) -> "AssistantAnswerSectionV1":
        if self.kind == "case_analysis":
            if not self.citations or any(
                isinstance(item, GuidanceCitationV1) for item in self.citations
            ):
                raise ValueError("case analysis sections require only case citations")
        elif not self.citations or any(
            not isinstance(item, GuidanceCitationV1) for item in self.citations
        ):
            raise ValueError("general guidance sections require only guidance citations")
        return self


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
    citations: list[AssistantAnswerCitationV1] = Field(default_factory=list, max_length=100)
    # ``answer_sections`` is additive and gives mixed case/general replies an
    # explicit visual boundary.  The legacy top-level fields still carry the
    # primary answer and its citations for unchanged consumers.
    answer_sections: list[AssistantAnswerSectionV1] = Field(default_factory=list, max_length=2)
    suggested_questions: list[str] = Field(default_factory=list, max_length=3)
    correction_draft: AssistantCorrectionDraftV1 | None = None
    engine: EngineV3

    @model_validator(mode="after")
    def _grounded_scopes_have_citations(self) -> "AssistantAnswerResponseV1":
        if self.answer_scope in {"case_grounded", "evidence_grounded"} and not self.citations:
            raise ValueError("grounded assistant answers require a citation")
        case_citations = [
            citation for citation in self.citations if isinstance(citation, AssistantCitationV1)
        ]
        guidance_citations = [
            citation for citation in self.citations if isinstance(citation, GuidanceCitationV1)
        ]
        identities = [
            (citation.kind, citation.reference_id)
            if isinstance(citation, AssistantCitationV1)
            else (citation.kind, citation.source_id)
            for citation in self.citations
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("assistant citations must be unique")
        if self.answer_scope == "case_grounded" and not any(
            citation.kind in {"fact", "comparison"} for citation in case_citations
        ):
            raise ValueError("case-grounded answers require a fact or comparison citation")
        if self.answer_scope == "evidence_grounded" and not any(
            citation.kind == "evidence" for citation in case_citations
        ):
            raise ValueError("evidence-grounded answers require an evidence citation")
        if (
            self.answer_scope in {"case_grounded", "evidence_grounded"}
            and guidance_citations
            and not self.answer_sections
        ):
            raise ValueError("mixed guidance citations require explicit answer sections")
        if self.answer_scope == "general_customs_guidance" and (
            not guidance_citations or case_citations
        ):
            raise ValueError("general guidance requires only official guidance citations")
        if (
            self.answer_scope
            not in {
                "case_grounded",
                "evidence_grounded",
                "general_customs_guidance",
            }
            and self.citations
        ):
            raise ValueError("non-grounded assistant answers must not include citations")
        if self.answer_scope == "general_customs_guidance":
            if GENERAL_GUIDANCE_DISCLAIMER not in self.direct_answer:
                raise ValueError("general guidance must include the non-case disclaimer")
            if self.correction_draft is not None:
                raise ValueError("general guidance must not contain a correction draft")
        elif self.correction_draft is not None and self.answer_scope != "case_grounded":
            raise ValueError("correction drafts require a case-grounded answer")
        section_kinds = [section.kind for section in self.answer_sections]
        if len(section_kinds) != len(set(section_kinds)):
            raise ValueError("assistant answer sections must be unique by kind")
        if self.answer_sections:
            case_sections = [item for item in self.answer_sections if item.kind == "case_analysis"]
            guidance_sections = [
                item for item in self.answer_sections if item.kind == "general_guidance"
            ]
            if case_sections and self.answer_scope not in {"case_grounded", "evidence_grounded"}:
                raise ValueError("case section requires a case-grounded top-level answer")
            if guidance_sections and not guidance_citations:
                raise ValueError("guidance section requires top-level guidance citations")
            section_guidance_ids = {
                item.source_id
                for section in guidance_sections
                for item in section.citations
                if isinstance(item, GuidanceCitationV1)
            }
            top_level_guidance_ids = {item.source_id for item in guidance_citations}
            if not section_guidance_ids.issubset(top_level_guidance_ids):
                raise ValueError("guidance section citations must be exposed at the top level")
        return self


class KnowledgeAnswerRequestV1(AssistantWireModel):
    """One conversation turn without a Case, Review Run, or document text."""

    schema_version: Literal["1.0"] = "1.0"
    conversation_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    question: str = Field(min_length=1, max_length=4000)
    trade_type: Literal["import", "export", "both"] = "both"
    as_of: date
    recent_turns: list[AssistantRecentTurnV1] = Field(default_factory=list, max_length=8)


class KnowledgeAnswerResponseV1(AssistantWireModel):
    """Grounded response for the standalone customs knowledge assistant."""

    schema_version: Literal["1.0"] = "1.0"
    conversation_id: str = Field(min_length=1, max_length=128)
    answer_scope: Literal[
        "general_customs_guidance",
        "clarification_needed",
        "insufficient_evidence",
        "out_of_scope",
    ]
    direct_answer: str = Field(min_length=1, max_length=4000)
    citations: list[GuidanceCitationV1] = Field(default_factory=list, max_length=100)
    answer_sections: list[AssistantAnswerSectionV1] = Field(default_factory=list, max_length=2)
    suggested_questions: list[str] = Field(default_factory=list, max_length=3)
    coverage: Literal["covered", "no_coverage", "partial", "unavailable"]
    snapshot_id: str | None = Field(default=None, max_length=512)
    as_of: date
    engine: EngineV3

    @model_validator(mode="after")
    def _official_claims_are_cited(self) -> "KnowledgeAnswerResponseV1":
        if self.answer_scope == "general_customs_guidance":
            if not self.citations or self.coverage not in {"covered", "partial"}:
                raise ValueError("general guidance requires covered official evidence")
        elif self.citations:
            raise ValueError("non-grounded knowledge answers must not include evidence")
        if self.answer_scope == "general_customs_guidance" and (
            GENERAL_GUIDANCE_DISCLAIMER not in self.direct_answer
        ):
            raise ValueError("standalone guidance must include the non-case disclaimer")
        return self


__all__ = [
    "AssistantAnswerRequestV1",
    "AssistantAnswerResponseV1",
    "AssistantAnswerSectionV1",
    "AssistantAnswerCitationV1",
    "AssistantCitationV1",
    "AssistantCorrectionDraftV1",
    "AssistantRecentTurnV1",
    "GuidanceEvidenceV1",
    "GuidanceCitationV1",
    "KnowledgeAnswerRequestV1",
    "KnowledgeAnswerResponseV1",
    "GENERAL_GUIDANCE_DISCLAIMER",
]
