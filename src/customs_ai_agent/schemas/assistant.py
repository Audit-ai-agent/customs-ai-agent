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
    # Data's page-level registry identity is necessary for Backend's strict
    # provenance check.  Backend removes it before any user-facing response.
    registry_source_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{2,63}$",
    )
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
        is_v2_page_source = self.registry_source_id.startswith("v2_")
        if self.source_kind == "official_law" and is_v2_page_source:
            # V2 registers exact law status/body pages rather than pretending
            # that every result is an article-level citation.  It may provide
            # no law fields, or a complete future article triplet, but never a
            # partial article claim.
            if any(value is not None for value in law_values) and not all(
                value is not None for value in law_values
            ):
                raise ValueError("v2 official law guidance must not claim partial law provenance")
        elif self.source_kind == "official_law" and any(value is None for value in law_values):
            raise ValueError("official law guidance requires law provenance")
        if self.source_kind != "official_law" and any(value is not None for value in law_values):
            raise ValueError("non-law guidance must not claim law provenance")
        return self


class GuidanceCitationV1(AssistantWireModel):
    """Compact public citation for one Data-approved official source."""

    kind: Literal["guidance_evidence"] = "guidance_evidence"
    source_id: str = Field(min_length=1, max_length=256)
    registry_source_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]{2,63}$",
    )
    source_kind: GuidanceSourceKind
    source_title: str = Field(min_length=1, max_length=1_000)
    source_uri: str = Field(min_length=1, max_length=2_000)
    as_of: date
    snapshot_id: str = Field(min_length=1, max_length=512)
    coverage: Literal["covered", "no_coverage", "partial", "unavailable"]
    # A two-scope answer labels each official citation with its independently
    # resolved scope.  ``None`` preserves the original single-scope contract.
    resolution_index: int | None = Field(default=None, ge=0, le=1)


AssistantAnswerCitationV1 = AssistantCitationV1 | GuidanceCitationV1


class AssistantAnswerSectionV1(AssistantWireModel):
    """A user-facing case, guidance, or evidence-linked application section.

    Sections are additive to the legacy ``direct_answer`` field so old
    persisted conversations remain readable.  Legacy replies use at most one
    case section, one general-guidance section, and one deterministic
    application section.  A verified V2 two-scope checklist instead has one
    case section, two independently indexed general-guidance sections, and
    one deterministic application section.  The application section never
    adds a new legal claim: it only identifies a next check from citations
    already present in the case and guidance sections.
    """

    kind: Literal["case_analysis", "general_guidance", "case_application"]
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4_000)
    citations: list[AssistantAnswerCitationV1] = Field(default_factory=list, max_length=100)
    # Case analysis and the final cross-scope checklist intentionally have no
    # single resolution owner.  Each general-guidance section has index 0 or 1.
    resolution_index: int | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _section_has_its_own_grounding(self) -> "AssistantAnswerSectionV1":
        if self.kind == "case_analysis":
            if not self.citations or any(
                isinstance(item, GuidanceCitationV1) for item in self.citations
            ):
                raise ValueError("case analysis sections require only case citations")
        elif self.kind == "general_guidance":
            if not self.citations or any(
                not isinstance(item, GuidanceCitationV1) for item in self.citations
            ):
                raise ValueError("general guidance sections require only guidance citations")
        elif (
            not self.citations
            or not any(isinstance(item, AssistantCitationV1) for item in self.citations)
            or not any(isinstance(item, GuidanceCitationV1) for item in self.citations)
        ):
            raise ValueError("case application sections require case and guidance citations")
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


class GuidanceResolutionV1(AssistantWireModel):
    """Safe public state of the one bounded general-guidance lookup.

    ``registry_sha256`` is intentionally absent.  The Agent verifies that
    digest at its Data boundary, while the Case UI only needs the official
    snapshot identity, date, selected scope, direction, and coverage state.
    """

    topic: str = Field(min_length=1, max_length=128)
    scope: str = Field(min_length=1, max_length=128)
    requested_direction: Literal["import", "export", "both"]
    applicable_directions: list[Literal["import", "export"]] = Field(min_length=1, max_length=2)
    coverage: Literal["covered", "no_coverage", "partial", "unavailable"]
    snapshot_id: str | None = Field(max_length=512)
    as_of: date
    registry_version: Literal["1.0", "2.0"]

    @model_validator(mode="after")
    def _directions_are_unique(self) -> "GuidanceResolutionV1":
        if len(self.applicable_directions) != len(set(self.applicable_directions)):
            raise ValueError("guidance resolution directions must be unique")
        if self.coverage in {"covered", "partial"} and self.snapshot_id is None:
            raise ValueError("covered guidance resolution requires a snapshot")
        return self


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
    # Backend owns durable conversation state.  It sends this only on a V2
    # follow-up after storing the immutable snapshot ID returned by Agent.
    expected_snapshot_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=81,
        pattern=r"^[a-z0-9][a-z0-9-]{2,80}$",
    )

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
    answer_sections: list[AssistantAnswerSectionV1] = Field(default_factory=list, max_length=4)
    suggested_questions: list[str] = Field(default_factory=list, max_length=3)
    correction_draft: AssistantCorrectionDraftV1 | None = None
    guidance_resolution: GuidanceResolutionV1 | None = None
    # Additive two-scope resolution surface.  It is XOR with the legacy
    # singular field, and always retains the classifier's deterministic order.
    guidance_resolutions: list[GuidanceResolutionV1] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )
    answer_format: Literal["multi_scope_checklist"] | None = None
    # Internal Agent→Backend provenance only.  Backend validates and removes
    # it before message persistence or user-facing serialization.
    guidance_registry_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    case_input_notice: Literal["unregistered_document_text"] | None = None
    engine: EngineV3

    @model_validator(mode="after")
    def _grounded_scopes_have_citations(self) -> "AssistantAnswerResponseV1":
        if self.guidance_resolutions is not None:
            _validate_assistant_multi_scope_answer(self)
            return self
        if self.answer_format is not None:
            raise ValueError("single-scope answers must not claim a multi-scope format")
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
        if any(citation.resolution_index is not None for citation in guidance_citations):
            raise ValueError("single-scope guidance citations must not carry a resolution index")
        if any(section.resolution_index is not None for section in self.answer_sections):
            raise ValueError("single-scope answer sections must not carry a resolution index")
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
            valid_orders = {
                (),
                ("case_analysis",),
                ("general_guidance",),
                ("case_analysis", "general_guidance"),
                ("case_analysis", "general_guidance", "case_application"),
            }
            if tuple(section_kinds) not in valid_orders:
                raise ValueError("assistant answer sections must follow the fixed order")
            case_sections = [item for item in self.answer_sections if item.kind == "case_analysis"]
            guidance_sections = [
                item for item in self.answer_sections if item.kind == "general_guidance"
            ]
            application_sections = [
                item for item in self.answer_sections if item.kind == "case_application"
            ]
            if case_sections and self.answer_scope not in {"case_grounded", "evidence_grounded"}:
                raise ValueError("case section requires a case-grounded top-level answer")
            if guidance_sections and not guidance_citations:
                raise ValueError("guidance section requires top-level guidance citations")
            if application_sections:
                if self.answer_scope not in {"case_grounded", "evidence_grounded"}:
                    raise ValueError("case application requires a case-grounded top-level answer")
                if not case_sections or not guidance_sections:
                    raise ValueError(
                        "case application requires preceding case and guidance sections"
                    )
            section_guidance_ids = {
                item.source_id
                for section in [*guidance_sections, *application_sections]
                for item in section.citations
                if isinstance(item, GuidanceCitationV1)
            }
            top_level_guidance_ids = {item.source_id for item in guidance_citations}
            if not section_guidance_ids.issubset(top_level_guidance_ids):
                raise ValueError("guidance section citations must be exposed at the top level")
            section_case_ids = {
                (item.kind, item.reference_id)
                for section in [*case_sections, *application_sections]
                for item in section.citations
                if isinstance(item, AssistantCitationV1)
            }
            top_level_case_ids = {(item.kind, item.reference_id) for item in case_citations}
            if not section_case_ids.issubset(top_level_case_ids):
                raise ValueError("case section citations must be exposed at the top level")
        if self.guidance_resolution is not None and self.case_input_notice is not None:
            raise ValueError("case input notice cannot claim a guidance lookup")
        if self.guidance_resolution is None and self.guidance_registry_sha256 is not None:
            raise ValueError("guidance registry hash requires a guidance resolution")
        if self.guidance_resolution is not None and self.guidance_registry_sha256 is None:
            raise ValueError("guidance resolution requires an internal registry hash")
        if (
            self.guidance_resolution is not None
            and self.guidance_resolution.registry_version == "2.0"
        ):
            if any(citation.registry_source_id is None for citation in guidance_citations):
                raise ValueError("v2 guidance citations require registry source identity")
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
    expected_snapshot_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=81,
        pattern=r"^[a-z0-9][a-z0-9-]{2,80}$",
    )


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
    guidance_resolution: GuidanceResolutionV1 | None = None
    guidance_resolutions: list[GuidanceResolutionV1] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )
    answer_format: Literal["multi_scope_checklist"] | None = None
    guidance_registry_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    engine: EngineV3

    @model_validator(mode="after")
    def _official_claims_are_cited(self) -> "KnowledgeAnswerResponseV1":
        if self.guidance_resolutions is not None:
            _validate_knowledge_multi_scope_answer(self)
            return self
        if self.answer_format is not None:
            raise ValueError("single-scope answers must not claim a multi-scope format")
        if self.answer_scope == "general_customs_guidance":
            if not self.citations or self.coverage not in {"covered", "partial"}:
                raise ValueError("general guidance requires covered official evidence")
        elif self.citations:
            raise ValueError("non-grounded knowledge answers must not include evidence")
        if self.answer_scope == "general_customs_guidance" and (
            GENERAL_GUIDANCE_DISCLAIMER not in self.direct_answer
        ):
            raise ValueError("standalone guidance must include the non-case disclaimer")
        if any(section.kind != "general_guidance" for section in self.answer_sections):
            raise ValueError("standalone guidance supports only general guidance sections")
        if self.answer_sections and self.answer_scope != "general_customs_guidance":
            raise ValueError("non-grounded knowledge answers must not have sections")
        if self.answer_sections:
            section_source_ids = {
                citation.source_id
                for section in self.answer_sections
                for citation in section.citations
                if isinstance(citation, GuidanceCitationV1)
            }
            citation_source_ids = {citation.source_id for citation in self.citations}
            if not section_source_ids or not section_source_ids.issubset(citation_source_ids):
                raise ValueError("knowledge section citations must be exposed at the top level")
        if any(citation.resolution_index is not None for citation in self.citations):
            raise ValueError("single-scope guidance citations must not carry a resolution index")
        if any(section.resolution_index is not None for section in self.answer_sections):
            raise ValueError("single-scope answer sections must not carry a resolution index")
        if self.guidance_resolution is None and self.guidance_registry_sha256 is not None:
            raise ValueError("guidance registry hash requires a guidance resolution")
        if self.guidance_resolution is not None and self.guidance_registry_sha256 is None:
            raise ValueError("guidance resolution requires an internal registry hash")
        if (
            self.guidance_resolution is not None
            and self.guidance_resolution.registry_version == "2.0"
        ):
            if any(citation.registry_source_id is None for citation in self.citations):
                raise ValueError("v2 guidance citations require registry source identity")
        return self


def _multi_scope_resolution_indexes(
    resolutions: list[GuidanceResolutionV1] | None,
) -> tuple[GuidanceResolutionV1, GuidanceResolutionV1]:
    """Validate the fixed two-target public contract once for both routes."""

    if resolutions is None or len(resolutions) != 2:
        raise ValueError("multi-scope answers require exactly two guidance resolutions")
    first, second = resolutions
    if first.registry_version != "2.0" or second.registry_version != "2.0":
        raise ValueError("multi-scope answers require the V2 guidance registry")
    if (first.topic, first.scope) == (second.topic, second.scope):
        raise ValueError("multi-scope answers require independent guidance scopes")
    return first, second


def _multi_guidance_citation_identity(citation: GuidanceCitationV1) -> tuple[str, int | None]:
    return citation.source_id, citation.resolution_index


def _validate_assistant_multi_scope_answer(answer: AssistantAnswerResponseV1) -> None:
    """Keep a multi-scope case checklist partitioned and non-conclusive.

    The Agent may report two unavailable/partial resolutions, but it may only
    emit the four-section checklist if both verified V2 scopes are covered and
    the supplied immutable Case review result contributes its own citation.
    """

    resolutions = _multi_scope_resolution_indexes(answer.guidance_resolutions)
    if answer.guidance_resolution is not None:
        raise ValueError("multi-scope answers are XOR with guidance_resolution")
    if answer.answer_format != "multi_scope_checklist":
        raise ValueError("multi-scope answers require the multi-scope format")
    if answer.guidance_registry_sha256 is None:
        raise ValueError("multi-scope answers require an internal registry hash")
    if answer.case_input_notice is not None:
        raise ValueError("case input notice cannot claim a multi-scope lookup")
    if answer.correction_draft is not None:
        raise ValueError("multi-scope checklists must not contain correction drafts")

    case_citations = [
        citation for citation in answer.citations if isinstance(citation, AssistantCitationV1)
    ]
    guidance_citations = [
        citation for citation in answer.citations if isinstance(citation, GuidanceCitationV1)
    ]
    identities = [
        (citation.kind, citation.reference_id)
        if isinstance(citation, AssistantCitationV1)
        else (citation.kind, citation.source_id, citation.resolution_index)
        for citation in answer.citations
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("assistant citations must be unique")

    # A missing scope never leaks the other scope's citations or a partial
    # synthetic application.  The two resolution records explain exactly what
    # was available without pretending it is a combined answer.
    if answer.answer_scope == "insufficient_evidence":
        if answer.citations or answer.answer_sections:
            raise ValueError("incomplete multi-scope answers must not expose citations or sections")
        return
    if answer.answer_scope not in {"case_grounded", "evidence_grounded"}:
        raise ValueError("multi-scope checklist requires a case-grounded answer")
    if any(resolution.coverage != "covered" for resolution in resolutions):
        raise ValueError("multi-scope checklist requires both scopes covered")
    snapshot_ids = {resolution.snapshot_id for resolution in resolutions}
    if None in snapshot_ids or len(snapshot_ids) != 1:
        raise ValueError("multi-scope checklist requires one immutable snapshot")
    if not case_citations or not guidance_citations:
        raise ValueError("multi-scope checklist requires case and guidance citations")
    if answer.answer_scope == "case_grounded" and not any(
        citation.kind in {"fact", "comparison"} for citation in case_citations
    ):
        raise ValueError("case-grounded answers require a fact or comparison citation")
    if answer.answer_scope == "evidence_grounded" and not any(
        citation.kind == "evidence" for citation in case_citations
    ):
        raise ValueError("evidence-grounded answers require an evidence citation")

    citation_indexes = {citation.resolution_index for citation in guidance_citations}
    if citation_indexes != {0, 1}:
        raise ValueError("multi-scope guidance citations require both resolution indexes")
    for index, resolution in enumerate(resolutions):
        scoped = [item for item in guidance_citations if item.resolution_index == index]
        if not scoped:
            raise ValueError("each multi-scope resolution requires a citation")
        if any(
            item.snapshot_id != resolution.snapshot_id
            or item.as_of != resolution.as_of
            or item.coverage != "covered"
            or item.registry_source_id is None
            for item in scoped
        ):
            raise ValueError("multi-scope guidance citation provenance mismatch")

    expected_sections = (
        ("case_analysis", None),
        ("general_guidance", 0),
        ("general_guidance", 1),
        ("case_application", None),
    )
    actual_sections = tuple(
        (section.kind, section.resolution_index) for section in answer.answer_sections
    )
    if actual_sections != expected_sections:
        raise ValueError("multi-scope checklist must use the fixed four-section order")
    case_section, first_guidance, second_guidance, application = answer.answer_sections
    if not case_section.citations or any(
        not isinstance(item, AssistantCitationV1) for item in case_section.citations
    ):
        raise ValueError("multi-scope case analysis requires only case citations")
    for index, section in enumerate((first_guidance, second_guidance)):
        if not section.citations or any(
            not isinstance(item, GuidanceCitationV1) or item.resolution_index != index
            for item in section.citations
        ):
            raise ValueError("each multi-scope guidance section requires its own citations")
    application_guidance_indexes = {
        item.resolution_index
        for item in application.citations
        if isinstance(item, GuidanceCitationV1)
    }
    if not any(isinstance(item, AssistantCitationV1) for item in application.citations) or (
        application_guidance_indexes != {0, 1}
    ):
        raise ValueError("multi-scope application requires case and both guidance citations")

    top_case_ids = {(item.kind, item.reference_id) for item in case_citations}
    top_guidance_ids = {_multi_guidance_citation_identity(item) for item in guidance_citations}
    section_case_ids = {
        (item.kind, item.reference_id)
        for section in (case_section, application)
        for item in section.citations
        if isinstance(item, AssistantCitationV1)
    }
    section_guidance_ids = {
        _multi_guidance_citation_identity(item)
        for section in (first_guidance, second_guidance, application)
        for item in section.citations
        if isinstance(item, GuidanceCitationV1)
    }
    if not section_case_ids.issubset(top_case_ids) or not section_guidance_ids.issubset(
        top_guidance_ids
    ):
        raise ValueError("multi-scope section citations must be exposed at the top level")


def _validate_knowledge_multi_scope_answer(answer: KnowledgeAnswerResponseV1) -> None:
    """Validate a standalone two-topic synthesis without Case application."""

    resolutions = _multi_scope_resolution_indexes(answer.guidance_resolutions)
    if answer.guidance_resolution is not None:
        raise ValueError("multi-scope answers are XOR with guidance_resolution")
    if answer.answer_format != "multi_scope_checklist":
        raise ValueError("multi-scope answers require the multi-scope format")
    if answer.guidance_registry_sha256 is None:
        raise ValueError("multi-scope answers require an internal registry hash")
    if answer.answer_scope == "insufficient_evidence":
        if answer.citations or answer.answer_sections:
            raise ValueError("incomplete multi-scope answers must not expose citations or sections")
        return
    if answer.answer_scope != "general_customs_guidance":
        raise ValueError("multi-scope knowledge answer requires official guidance scope")
    if any(resolution.coverage != "covered" for resolution in resolutions):
        raise ValueError("multi-scope knowledge answer requires both scopes covered")
    snapshot_ids = {resolution.snapshot_id for resolution in resolutions}
    if None in snapshot_ids or len(snapshot_ids) != 1 or answer.snapshot_id not in snapshot_ids:
        raise ValueError("multi-scope knowledge answer requires one immutable snapshot")
    if answer.coverage != "covered" or GENERAL_GUIDANCE_DISCLAIMER not in answer.direct_answer:
        raise ValueError("multi-scope knowledge answer requires covered official guidance")
    if len(answer.answer_sections) != 2 or tuple(
        (section.kind, section.resolution_index) for section in answer.answer_sections
    ) != (("general_guidance", 0), ("general_guidance", 1)):
        raise ValueError("multi-scope knowledge answer requires two indexed guidance sections")
    identities = [_multi_guidance_citation_identity(item) for item in answer.citations]
    if len(identities) != len(set(identities)):
        raise ValueError("knowledge guidance citations must be unique")
    if {item.resolution_index for item in answer.citations} != {0, 1}:
        raise ValueError("multi-scope guidance citations require both resolution indexes")
    top_ids = set(identities)
    for index, (resolution, section) in enumerate(zip(resolutions, answer.answer_sections)):
        scoped = [item for item in answer.citations if item.resolution_index == index]
        if (
            not scoped
            or not section.citations
            or any(
                not isinstance(item, GuidanceCitationV1) or item.resolution_index != index
                for item in section.citations
            )
        ):
            raise ValueError("each multi-scope knowledge section requires its own citations")
        if any(
            item.snapshot_id != resolution.snapshot_id
            or item.as_of != resolution.as_of
            or item.coverage != "covered"
            or item.registry_source_id is None
            for item in scoped
        ):
            raise ValueError("multi-scope guidance citation provenance mismatch")
        section_ids = {
            _multi_guidance_citation_identity(item)
            for item in section.citations
            if isinstance(item, GuidanceCitationV1)
        }
        if not section_ids or not section_ids.issubset(top_ids):
            raise ValueError("knowledge section citations must be exposed at the top level")


__all__ = [
    "AssistantAnswerRequestV1",
    "AssistantAnswerResponseV1",
    "AssistantAnswerSectionV1",
    "AssistantAnswerCitationV1",
    "AssistantCitationV1",
    "AssistantCorrectionDraftV1",
    "GuidanceResolutionV1",
    "AssistantRecentTurnV1",
    "GuidanceEvidenceV1",
    "GuidanceCitationV1",
    "KnowledgeAnswerRequestV1",
    "KnowledgeAnswerResponseV1",
    "GENERAL_GUIDANCE_DISCLAIMER",
]
