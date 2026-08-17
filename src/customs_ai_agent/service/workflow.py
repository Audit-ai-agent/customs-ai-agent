"""Bounded, source-grounded document-review workflow.

The workflow deliberately has no open-ended planner or tool surface.  A
profile-specific analyzer can only extract text facts.  Everything that turns
those facts into a comparison, finding, or retrieval request is deterministic
Python code in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field as dataclass_field, replace
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, cast

from customs_ai_agent.adapters.llm import DocumentAnalysis
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate
from customs_ai_agent.schemas.v3 import (
    ComparisonOperandV3,
    ComparisonV3,
    DocumentAnalysisV3,
    ExtractionWarningV3,
    FactV3,
    FindingV3,
    HumanProvidedSourceV3,
    LineItemV3,
    PlannerActionV3,
    PlannerTraceV3,
    ReextractionV3,
    ServiceReviewRunRequestV3,
    SourceV3,
    WorkflowStageV3,
    WorkflowV3,
)
from customs_ai_agent.scenario_transcript import ScenarioTranscriptRun


_KNOWN_TYPES = {
    "invoice",
    "packing_list",
    "bill_of_lading",
    "air_waybill",
    "certificate_of_origin",
    "export_declaration",
    "supporting_evidence",
}
_NUMERIC_FIELDS = {
    "quantity",
    "unit_price",
    "amount",
    "total_amount",
    "gross_weight_kg",
    "total_gross_weight_kg",
    "net_weight_kg",
    "total_net_weight_kg",
    "package_count",
    "packages",
    "measurement_cbm",
    "total_measurement_cbm",
}
_IDENTIFIER_FIELDS = {
    "invoice_no",
    "packing_no",
    "bl_no",
    "awb_no",
    "certificate_no",
    "export_declaration_no",
    "submission_receipt_no",
    "shipment_id",
    "container_numbers",
}
_ORIGIN_FIELDS = {"origin_country", "country_of_origin"}
_DESTINATION_FIELDS = {"destination_country", "country_of_destination"}
_DESCRIPTION_FIELDS = {"description", "goods_description"}
_HS_FIELDS = {"hs_code", "hs_codes"}
_DOCUMENT_LABELS = {
    "invoice": "상업송장",
    "packing_list": "포장명세서",
    "bill_of_lading": "선하증권",
    "air_waybill": "항공화물운송장",
    "certificate_of_origin": "원산지증명서",
    "export_declaration": "수출신고증",
    "supporting_evidence": "보완·제출 증빙",
    "unknown": "문서",
}
_FIELD_LABELS = {
    "invoice_no": "상업송장 번호",
    "packing_no": "포장명세서 번호",
    "bl_no": "B/L 번호",
    "awb_no": "AWB 번호",
    "certificate_no": "원산지증명서 번호",
    "export_declaration_no": "수출신고 번호",
    "submission_receipt_no": "접수·제출 번호",
    "filing_date": "신고·제출일",
    "submission_status": "기관 처리 상태",
    "origin_country": "원산지",
    "country_of_origin": "원산지",
    "hs_code": "HS 코드",
    "total_amount": "총액",
}
_CURRENCY_RE = re.compile(r"\b(KRW|USD|EUR|JPY|CNY|GBP)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CONTAINER_RE = re.compile(r"\b[A-Z]{4}\d{7}\b")
_SAFE_NAME = re.compile(r"[^a-z0-9_]+")


class _Analyzer(Protocol):
    model_id: str

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: ScenarioTranscriptRun | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis: ...


@dataclass
class _BoundedPlanner:
    """Small stateful planner for the only actions this workflow may execute.

    This is intentionally not a general tool loop.  The workflow chooses an
    action from a closed set when the preceding state makes it necessary, then
    records that action *after* it has run.  A trace therefore never contains a
    pre-filled skipped step that the Agent did not execute.
    """

    _actions: list[PlannerActionV3] = dataclass_field(default_factory=list)
    _cycle: int = 1

    def start_retry_cycle(self) -> None:
        """Move to the one permitted follow-up cycle for focused re-extraction."""

        if self._cycle != 1:
            raise RuntimeError("bounded planner cannot start more than two cycles")
        self._cycle = 2

    def record(
        self,
        action: Literal[
            "analyze_documents",
            "apply_human_corrections",
            "compare_documents",
            "reextract_documents",
            "retrieve_evidence",
            "generate_report",
        ],
        *,
        status: Literal["completed", "needs_review", "failed"],
        summary: str,
        correction_proposal_ids: list[str] | None = None,
    ) -> None:
        """Record one completed/attempted closed-surface action exactly once."""

        if len(self._actions) >= 6:
            raise RuntimeError("bounded planner action limit exceeded")
        if any(entry.action == action for entry in self._actions):
            raise RuntimeError(f"bounded planner action already recorded: {action}")
        self._actions.append(
            PlannerActionV3(
                cycle=self._cycle,
                action=action,
                status=status,
                summary=summary,
                correction_proposal_ids=correction_proposal_ids or [],
            )
        )

    def copy(self) -> "_BoundedPlanner":
        return _BoundedPlanner(_actions=list(self._actions), _cycle=self._cycle)

    def trace(self) -> PlannerTraceV3:
        return PlannerTraceV3(cycle_count=self._cycle, actions=list(self._actions))


@dataclass(frozen=True)
class BoundedWorkflowOutput:
    """The deterministic work completed before the single Evidence call."""

    document_analyses: list[DocumentAnalysisV3]
    comparisons: list[ComparisonV3]
    findings: list[FindingV3]
    workflow: WorkflowV3
    retrieval_items: list[dict[str, Any]]
    retrieval_document_ids: dict[str, list[str]]
    retrieval_excluded_document_ids: list[str]
    retrieval_skip_reason: (
        Literal["no_hs_qualified_commodity", "export_destination_country_missing"] | None
    )
    planner_state: _BoundedPlanner

    def with_evidence_status(self, data_status: str) -> "BoundedWorkflowOutput":
        """Record the one actual Evidence result without changing comparisons."""

        retrieval_status: Literal["completed", "needs_review", "failed"]
        if data_status == "ok":
            retrieval_status = "completed"
        elif data_status == "partial":
            retrieval_status = "needs_review"
        else:
            retrieval_status = "failed"
        stages = [
            stage.model_copy(
                update={
                    "status": retrieval_status,
                    "summary": _evidence_summary(
                        data_status,
                        excluded_document_count=len(self.retrieval_excluded_document_ids),
                    ),
                }
            )
            if stage.stage == "retrieve_evidence"
            else stage
            for stage in self.workflow.stages
        ]
        planner = self.planner_state.copy()
        planner.record(
            "retrieve_evidence",
            status=retrieval_status,
            summary=_evidence_summary(
                data_status,
                excluded_document_count=len(self.retrieval_excluded_document_ids),
            ),
        )
        workflow_status: Literal["completed", "partial", "failed"]
        if data_status == "failed":
            workflow_status = "failed"
        elif data_status == "partial" or self.workflow.status == "partial":
            workflow_status = "partial"
        else:
            workflow_status = "completed"
        return replace(
            self,
            workflow=WorkflowV3(
                status=workflow_status,
                stages=stages,
                reextractions=self.workflow.reextractions,
                planner=planner.trace(),
            ),
            planner_state=planner,
        )

    def with_evidence_skipped(self) -> "BoundedWorkflowOutput":
        """Record why the bounded Data request was not created.

        The shared Data contract requires at least one item.  Skipping an
        otherwise invalid request is an input limitation, not a Data partial
        failure, and does not pretend that a retrieval action ran.
        """

        summaries = {
            "no_hs_qualified_commodity": (
                "HS 코드가 확인된 품목이 없어 법령 근거 조회를 실행하지 않았습니다."
            ),
            "export_destination_country_missing": (
                "수출 목적국 코드가 문서에서 확인되지 않아 법령 근거 조회를 실행하지 않았습니다."
            ),
        }
        reason = self.retrieval_skip_reason or "no_hs_qualified_commodity"
        stages = [
            stage.model_copy(
                update={
                    "status": "skipped",
                    "summary": summaries[reason],
                }
            )
            if stage.stage == "retrieve_evidence"
            else stage
            for stage in self.workflow.stages
        ]
        return replace(
            self,
            workflow=WorkflowV3(
                status=self.workflow.status,
                stages=stages,
                reextractions=self.workflow.reextractions,
                planner=self.planner_state.trace(),
            ),
        )

    def with_report_generated(self) -> "BoundedWorkflowOutput":
        """Record report generation only after its grounded sections exist."""

        stages = [
            stage.model_copy(
                update={
                    "status": "completed",
                    "attempt_count": 1,
                    "summary": "근거가 연결된 8구역 검토 보고서를 생성했습니다.",
                }
            )
            if stage.stage == "generate_result"
            else stage
            for stage in self.workflow.stages
        ]
        planner = self.planner_state.copy()
        planner.record(
            "generate_report",
            status="completed",
            summary="근거가 연결된 8구역 검토 보고서를 생성했습니다.",
        )
        return replace(
            self,
            workflow=WorkflowV3(
                status=self.workflow.status,
                stages=stages,
                reextractions=self.workflow.reextractions,
                planner=planner.trace(),
            ),
            planner_state=planner,
        )


def run_bounded_workflow(
    request: ServiceReviewRunRequestV3,
    analyzer: _Analyzer,
    *,
    transcript: ScenarioTranscriptRun | None = None,
    on_stage: Callable[[str, str], None] | None = None,
) -> BoundedWorkflowOutput:
    """Extract once, make at most one focused retry per document, then compare."""

    def _notify(stage: str, status: str) -> None:
        # Display-only progress; a broken listener must never break the review.
        if on_stage is None:
            return
        try:
            on_stage(stage, status)
        except Exception:
            pass

    planner = _BoundedPlanner()
    _notify("classify_documents", "running")
    _notify("extract_facts", "running")
    initial: dict[str, DocumentAnalysis] = {}
    for document in request.documents:
        initial[document.document_id] = analyzer.analyze(
            document.model_dump(mode="json", warnings=False), transcript=transcript
        )
    planner.record(
        "analyze_documents",
        status="completed",
        summary="문서 유형과 원문 인용이 있는 사실만 추출·정규화했습니다.",
    )
    _notify("classify_documents", "completed")
    _notify("extract_facts", "completed")
    _notify("normalize_facts", "completed")

    retry_targets = _reextraction_targets(request, initial)
    analyses = dict(initial)
    reextractions: list[ReextractionV3] = []
    retry_unavailable = False
    if retry_targets:
        planner.start_retry_cycle()
        _notify("reextract_documents", "running")
    for document in request.documents:
        requested_fields = retry_targets.get(document.document_id)
        if not requested_fields:
            continue
        try:
            candidate = analyzer.analyze(
                document.model_dump(mode="json", warnings=False),
                transcript=transcript,
                focus_fields=tuple(requested_fields),
            )
        except Exception:
            retry_unavailable = True
            reextractions.append(
                ReextractionV3(
                    document_id=document.document_id,
                    requested_fields=requested_fields,
                    status="unavailable",
                )
            )
            continue
        analyses[document.document_id] = _prefer_more_complete(
            analyses[document.document_id], candidate
        )
        reextractions.append(
            ReextractionV3(
                document_id=document.document_id,
                requested_fields=requested_fields,
                status="completed",
            )
        )
    if reextractions:
        planner.record(
            "reextract_documents",
            status="needs_review" if retry_unavailable else "completed",
            summary=(
                "각 대상 문서를 최대 한 번 재추출했지만 일부 결과를 사용할 수 없어 사람 검토가 필요합니다."
                if retry_unavailable
                else "누락·모호한 비교 필드를 보완하기 위해 각 대상 문서를 최대 한 번 재추출했습니다."
            ),
        )

    if reextractions:
        _notify("reextract_documents", "completed")
    _notify("match_documents", "running")
    rendered = [
        _render_analysis(document, analyses[document.document_id]) for document in request.documents
    ]
    if request.human_corrections:
        rendered = _apply_human_corrections(rendered, request)
        planner.record(
            "apply_human_corrections",
            status="completed",
            summary="승인된 사람 수정은 원문 사실을 보존한 별도 human-provided 사실로 반영했습니다.",
            correction_proposal_ids=request.applied_correction_proposal_ids,
        )
    comparisons = _comparisons(rendered, trade_type=request.trade_type)
    planner.record(
        "compare_documents",
        status="completed",
        summary="코드 기반 수량·금액·원산지·B/L 비교 규칙을 실행했습니다.",
    )
    _notify("match_documents", "completed")
    _notify("validate_documents", "completed")
    findings = _findings_from_comparisons(comparisons)
    findings.extend(_warning_findings(rendered))
    (
        retrieval_items,
        retrieval_document_ids,
        retrieval_excluded_document_ids,
        retrieval_skip_reason,
    ) = retrieval_plan_from_analyses(request, rendered)
    stages = _workflow_stages(
        rendered,
        comparisons,
        reextractions,
        retry_unavailable=retry_unavailable,
    )
    workflow_status: Literal["completed", "partial", "failed"] = (
        "partial" if retry_unavailable else "completed"
    )
    return BoundedWorkflowOutput(
        document_analyses=rendered,
        comparisons=comparisons,
        findings=_dedupe_findings(findings),
        workflow=WorkflowV3(
            status=workflow_status,
            stages=stages,
            reextractions=reextractions,
            planner=planner.trace(),
        ),
        retrieval_items=retrieval_items,
        retrieval_document_ids=retrieval_document_ids,
        retrieval_excluded_document_ids=retrieval_excluded_document_ids,
        retrieval_skip_reason=retrieval_skip_reason,
        planner_state=planner,
    )


def retrieval_items_from_analyses(
    request: ServiceReviewRunRequestV3,
    analyses: list[DocumentAnalysisV3],
) -> list[dict[str, Any]]:
    """Return only the public Data request items from the internal plan."""

    return retrieval_plan_from_analyses(request, analyses)[0]


def retrieval_plan_from_analyses(
    request: ServiceReviewRunRequestV3,
    analyses: list[DocumentAnalysisV3],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[str]],
    list[str],
    Literal["no_hs_qualified_commodity", "export_destination_country_missing"] | None,
]:
    """Group source-grounded facts into one Data item per commodity.

    A law requirement applies to a commodity, not independently to every
    source document that repeats the same HS code.  The private mapping keeps
    all linked document IDs available for findings without extending the Data
    HTTP contract.  Documents without an HS code, including ordinary B/L
    documents, are not sent as malformed retrieval fallback items.
    """

    candidates: list[dict[str, Any]] = []
    excluded_document_ids: list[str] = []
    for document, analysis in zip(request.documents, analyses, strict=True):
        document_candidates: list[dict[str, Any]] = []
        for item in analysis.line_items:
            hs = _first_fact(item.facts, "hs_code")
            if hs is None or not (hs.normalized_value or hs.raw_value):
                continue
            description = _first_fact(item.facts, *_DESCRIPTION_FIELDS)
            origin = _first_fact(item.facts, *_ORIGIN_FIELDS)
            document_candidates.append(
                {
                    "document_id": document.document_id,
                    "description": (description.raw_value if description else item.source.quote)[
                        :2000
                    ],
                    "hs_code": hs.normalized_value or hs.raw_value,
                    "origin_country": origin.normalized_value if origin else None,
                }
            )
        if not document_candidates:
            document_hs = _first_fact(analysis.facts, "hs_code")
            if document_hs is not None and (document_hs.normalized_value or document_hs.raw_value):
                description = _first_fact(analysis.facts, *_DESCRIPTION_FIELDS)
                origin = _first_fact(analysis.facts, *_ORIGIN_FIELDS)
                document_candidates.append(
                    {
                        "document_id": document.document_id,
                        "description": (
                            description.raw_value if description else document.content_text[:2000]
                        )[:2000],
                        "hs_code": document_hs.normalized_value or document_hs.raw_value,
                        "origin_country": origin.normalized_value if origin else None,
                    }
                )
        if not document_candidates and analysis.document_type not in {
            "bill_of_lading",
            "air_waybill",
        }:
            excluded_document_ids.append(document.document_id)
        candidates.extend(document_candidates)

    origins_by_hs: dict[str, set[str]] = {}
    for candidate in candidates:
        hs_code = str(candidate["hs_code"])
        origin = candidate["origin_country"]
        if isinstance(origin, str) and origin:
            origins_by_hs.setdefault(hs_code, set()).add(origin)
    for candidate in candidates:
        if candidate["origin_country"] is None:
            origins = origins_by_hs.get(str(candidate["hs_code"]), set())
            if len(origins) == 1:
                candidate["origin_country"] = next(iter(origins))

    if not candidates:
        return [], {}, sorted(set(excluded_document_ids)), "no_hs_qualified_commodity"

    destination_country = "KR"
    if request.trade_type == "export":
        destination_country = _export_destination_country(analyses) or ""
        if not destination_country:
            # The Evidence API requires a jurisdictional destination.  Do not
            # silently reinterpret an export as a Korean import when that fact
            # is not grounded in the supplied export documentation.
            return (
                [],
                {},
                sorted(set(excluded_document_ids)),
                "export_destination_country_missing",
            )

    groups: dict[tuple[str, str | None], dict[str, Any]] = {}
    for candidate in candidates:
        key = (str(candidate["hs_code"]), cast(str | None, candidate["origin_country"]))
        group = groups.setdefault(
            key,
            {
                "description": candidate["description"],
                "document_ids": set(),
            },
        )
        group["document_ids"].add(candidate["document_id"])

    items: list[dict[str, Any]] = []
    document_ids_by_item: dict[str, list[str]] = {}
    for (hs_code, origin_country), group in sorted(groups.items()):
        identity = {
            "hs_code": hs_code,
            "origin_country": origin_country,
            "destination_country": destination_country,
        }
        item_id = (
            "commodity-"
            + hashlib.sha256(
                json.dumps(
                    identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()[:24]
        )
        items.append(
            {
                "item_id": item_id,
                "description": str(group["description"])[:2000],
                "hs_code": hs_code,
                "origin_country": origin_country,
                "destination_country": destination_country,
            }
        )
        document_ids_by_item[item_id] = sorted(group["document_ids"])
    return items[:100], document_ids_by_item, sorted(set(excluded_document_ids)), None


def finding_for_evidence_requirement(
    *,
    code: str,
    message: str,
    item_id: str,
    evidence_ids: list[str],
    document_ids: list[str],
) -> FindingV3:
    """Create a stable finding for a Data-required document check."""

    return _finding(
        code=code,
        severity="medium",
        message=message,
        item_id=item_id,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        document_ids=list(dict.fromkeys(document_ids)),
    )


def _render_analysis(
    document: Any,
    analysis: DocumentAnalysis,
) -> DocumentAnalysisV3:
    candidate = analysis.candidate
    warnings: list[ExtractionWarningV3] = []
    document_type = candidate.document_type
    classification_source: SourceV3 | None = None
    if document_type != "unknown":
        classification_source = _source(
            document,
            candidate.classification_source_quote,
            warnings,
            code="classification_source_invalid",
            label="문서 유형 근거",
        )
        if classification_source is None:
            document_type = "unknown"

    facts: list[FactV3] = []
    for index, field in enumerate(candidate.fields, start=1):
        fact = _render_fact(
            document,
            field.name,
            field.value,
            field.source_quote,
            fact_id=f"{document.document_id}:field:{_safe_name_value(field.name)}:{index}",
            warnings=warnings,
        )
        if fact is not None:
            facts.append(fact)

    line_items: list[LineItemV3] = []
    for item_index, item in enumerate(candidate.line_items, start=1):
        item_source = _source(
            document,
            item.source_quote,
            warnings,
            code="line_item_source_invalid",
            label="품목 근거",
        )
        if item_source is None:
            continue
        item_id = f"{document.document_id}:item:{item_index}"
        item_facts: list[FactV3] = []
        for field_index, field in enumerate(item.fields, start=1):
            fact = _render_fact(
                document,
                field.name,
                field.value,
                field.source_quote,
                fact_id=(f"{item_id}:field:{_safe_name_value(field.name)}:{field_index}"),
                warnings=warnings,
                containing_quote=item.source_quote,
            )
            if fact is not None:
                item_facts.append(fact)
        line_items.append(LineItemV3(item_id=item_id, source=item_source, facts=item_facts))

    return DocumentAnalysisV3(
        document_id=document.document_id,
        filename=document.filename,
        declared_type=document.declared_type,
        document_type=document_type,
        classification_source=classification_source,
        extraction_status="needs_review" if warnings or document_type == "unknown" else "complete",
        facts=facts,
        line_items=line_items,
        warnings=warnings,
    )


def _apply_human_corrections(
    analyses: list[DocumentAnalysisV3], request: ServiceReviewRunRequestV3
) -> list[DocumentAnalysisV3]:
    """Append approved human facts without replacing a document-derived fact.

    A correction is deliberately a second fact with an explicit source kind.
    This keeps the original quote auditable while allowing the deterministic
    comparison helpers to preferentially use the approved correction.
    """

    if not request.human_corrections:
        return analyses
    by_document = {analysis.document_id: analysis for analysis in analyses}
    for correction in request.human_corrections:
        analysis = by_document[correction.document_id]
        target_item = _correction_target_item(analysis, correction)
        normalized_value, normalized_unit = _normalize_value(
            correction.field_name, correction.new_value
        )
        fact = FactV3(
            fact_id=f"human:{correction.correction_proposal_id}",
            name=correction.field_name.strip().casefold(),
            raw_value=correction.new_value,
            normalized_value=normalized_value,
            unit=correction.unit or normalized_unit,
            source=HumanProvidedSourceV3(
                document_id=correction.document_id,
                correction_proposal_id=correction.correction_proposal_id,
                reason=correction.reason,
            ),
        )
        if target_item is None:
            by_document[correction.document_id] = analysis.model_copy(
                update={"facts": [*analysis.facts, fact]}
            )
            continue
        replacement_items = [
            item.model_copy(update={"facts": [*item.facts, fact]})
            if item.item_id == target_item.item_id
            else item
            for item in analysis.line_items
        ]
        by_document[correction.document_id] = analysis.model_copy(
            update={"line_items": replacement_items}
        )
    return [by_document[analysis.document_id] for analysis in analyses]


def _correction_target_item(
    analysis: DocumentAnalysisV3,
    correction: Any,
) -> LineItemV3 | None:
    """Resolve a correction target without ever widening it to another location.

    A supplied item/fact identifier is an ownership assertion from Backend.  If
    it does not describe this exact document and item, appending a document-level
    fact would make an invalid correction appear to have been applied.  Reject
    the child run instead.
    """

    normalized_name = correction.field_name.strip().casefold()
    item_by_id = {item.item_id: item for item in analysis.line_items}

    if correction.item_id is not None:
        target_item = item_by_id.get(correction.item_id)
        if target_item is None:
            raise ValueError("human_correction_item_target_not_found")
        if correction.fact_id is not None:
            target_fact = next(
                (fact for fact in target_item.facts if fact.fact_id == correction.fact_id), None
            )
            if target_fact is None:
                raise ValueError("human_correction_fact_not_owned_by_item")
            if not _correction_field_matches(target_fact.name, normalized_name):
                raise ValueError("human_correction_field_does_not_match_fact")
        return target_item

    if correction.fact_id is None:
        # A document-level correction may introduce a clearly labelled new fact.
        return None

    direct_fact = next(
        (fact for fact in analysis.facts if fact.fact_id == correction.fact_id), None
    )
    if direct_fact is not None:
        if not _correction_field_matches(direct_fact.name, normalized_name):
            raise ValueError("human_correction_field_does_not_match_fact")
        return None

    # Claude can return the same document-level fields in a different order on
    # a child run.  Fact IDs intentionally include that order, so an exact
    # parent ID is not a stable cross-call locator.  A correction remains safe
    # to apply only when the original document has exactly one document-level
    # fact with the declared field name.  Never broaden this fallback to a
    # line item or to multiple candidates: either case needs another human
    # review instead of silently moving a correction.
    same_name_facts = [
        fact for fact in analysis.facts if _correction_field_matches(fact.name, normalized_name)
    ]
    if len(same_name_facts) == 1:
        return None
    if len(same_name_facts) > 1:
        raise ValueError("human_correction_fact_target_ambiguous")

    for item in analysis.line_items:
        target_fact = next(
            (fact for fact in item.facts if fact.fact_id == correction.fact_id), None
        )
        if target_fact is None:
            continue
        if not _correction_field_matches(target_fact.name, normalized_name):
            raise ValueError("human_correction_field_does_not_match_fact")
        return item
    raise ValueError("human_correction_fact_target_not_found")


def _correction_field_matches(actual: str, requested: str) -> bool:
    """Match a correction field to one extraction field without broadening scope."""

    if _canonical_fact_name(actual) == _canonical_fact_name(requested):
        return True
    return False


def _source(
    document: Any,
    quote: str,
    warnings: list[ExtractionWarningV3],
    *,
    code: str,
    label: str,
) -> SourceV3 | None:
    text = document.content_text
    if not quote or quote not in text:
        warnings.append(
            ExtractionWarningV3(
                code=code,
                message=f"{label} 인용문이 입력 문서 본문에서 확인되지 않았습니다.",
            )
        )
        return None
    start = text.find(quote)
    occurrence = len(list(re.finditer(re.escape(quote), text[:start])))
    return SourceV3(document_id=document.document_id, quote=quote, occurrence=occurrence)


def _render_fact(
    document: Any,
    name: str,
    value: str,
    quote: str,
    *,
    fact_id: str,
    warnings: list[ExtractionWarningV3],
    containing_quote: str | None = None,
) -> FactV3 | None:
    if not value or value not in quote:
        warnings.append(
            ExtractionWarningV3(
                code="field_source_invalid",
                message=f"{name} 값의 원문 인용을 확인할 수 없습니다.",
            )
        )
        return None
    if containing_quote is not None and quote not in containing_quote:
        warnings.append(
            ExtractionWarningV3(
                code="line_item_field_source_invalid",
                message=f"{name} 값의 인용문이 품목 원문 범위 안에 있지 않습니다.",
            )
        )
        return None
    source = _source(
        document,
        quote,
        warnings,
        code="field_source_invalid",
        label=f"{name} 값",
    )
    if source is None:
        return None
    canonical_name = _canonical_fact_name(name)
    normalized_value, unit = _normalize_value(canonical_name, value, unit_context=quote)
    return FactV3(
        fact_id=fact_id,
        name=canonical_name,
        raw_value=value,
        normalized_value=normalized_value,
        unit=unit,
        source=source,
    )


def _normalize_value(
    name: str,
    raw: str,
    *,
    unit_context: str | None = None,
) -> tuple[str | None, str | None]:
    name = name.strip().casefold()
    text = unicodedata.normalize("NFKC", raw).strip()
    if not text:
        return None, None
    if name in {"hs_code", "hs_codes"}:
        digits = re.sub(r"\D", "", text)
        return (digits, None) if re.fullmatch(r"\d{6,10}", digits) else (None, None)
    if name in _ORIGIN_FIELDS | _DESTINATION_FIELDS:
        return re.sub(r"\s+", " ", text).upper(), None
    if name in _DESCRIPTION_FIELDS:
        return re.sub(r"\s+", " ", text).casefold(), None
    if name == "container_numbers":
        containers = sorted(set(_CONTAINER_RE.findall(text.upper())))
        if containers:
            return "|".join(containers), None
        return re.sub(r"\s+", "", text).upper(), None
    if name in _IDENTIFIER_FIELDS:
        return re.sub(r"\s+", "", text).upper(), None
    if name in _NUMERIC_FIELDS:
        match = _NUMBER_RE.search(text)
        if match is None:
            return None, _unit_for_numeric(name, text)
        try:
            number = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None, _unit_for_numeric(name, text)
        unit = _unit_for_numeric(name, text)
        if unit is None and unit_context is not None:
            unit = _unit_for_numeric(name, unit_context)
        return _decimal_text(number), unit
    return re.sub(r"\s+", " ", text).casefold(), None


def _canonical_fact_name(name: str) -> str:
    """Keep equivalent extracted field aliases on one comparison path."""

    normalized = name.strip().casefold()
    if normalized in _HS_FIELDS:
        return "hs_code"
    if normalized in _ORIGIN_FIELDS:
        return "origin_country"
    return normalized


def _document_label(analysis: DocumentAnalysisV3) -> str:
    return _DOCUMENT_LABELS.get(analysis.document_type, _DOCUMENT_LABELS["unknown"])


def _unit_for_numeric(name: str, text: str) -> str | None:
    currency = _CURRENCY_RE.search(text)
    if name in {"unit_price", "amount", "total_amount"}:
        return currency.group(1).upper() if currency else None
    unit = re.search(
        r"\b(KGS?|KG|GRAMS?|G|TONS?|TON|PCS?|PIECES?|EA|CTNS?|CARTONS?|CBM)\b", text, re.I
    )
    if unit is None:
        return None
    value = unit.group(1).upper()
    return {
        "KGS": "KG",
        "GRAM": "G",
        "GRAMS": "G",
        "TONS": "TON",
        "PC": "PCS",
        "PIECES": "PCS",
        "CTN": "CTN",
        "CARTONS": "CTN",
    }.get(value, value)


def _decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _comparisons(
    analyses: list[DocumentAnalysisV3], *, trade_type: Literal["import", "export"]
) -> list[ComparisonV3]:
    comparisons: list[ComparisonV3] = []
    comparisons.extend(_classification_and_required_field_checks(analyses))
    comparisons.extend(_invoice_packing_checks(analyses))
    comparisons.extend(_invoice_origin_checks(analyses))
    comparisons.extend(_transport_packing_checks(analyses, "bill_of_lading"))
    comparisons.extend(_transport_packing_checks(analyses, "air_waybill"))
    comparisons.extend(_invoice_export_declaration_checks(analyses, trade_type=trade_type))
    return comparisons


def _classification_and_required_field_checks(
    analyses: list[DocumentAnalysisV3],
) -> list[ComparisonV3]:
    comparisons: list[ComparisonV3] = []
    requirements: dict[str, tuple[str, ...]] = {
        "invoice": ("invoice_no", "total_amount", "hs_code"),
        "packing_list": ("packing_no",),
        "certificate_of_origin": ("certificate_no", "origin_country", "hs_code"),
        "bill_of_lading": ("bl_no",),
        "air_waybill": ("awb_no",),
        "export_declaration": ("export_declaration_no",),
        "supporting_evidence": ("submission_receipt_no",),
    }
    for analysis in analyses:
        if analysis.document_type == "unknown":
            comparisons.append(
                _comparison(
                    rule_code="document_type_classification",
                    status="needs_review",
                    left=_operand(document_id=analysis.document_id),
                    right=_operand(),
                    message="문서 유형을 신뢰할 수 있게 판별하지 못했습니다.",
                )
            )
            continue
        for name in requirements.get(analysis.document_type, ()):
            fact = _first_fact_from_analysis(analysis, name)
            code = "hs_code_presence" if name == "hs_code" else "required_field_presence"
            field_label = _FIELD_LABELS.get(name, name)
            comparisons.append(
                _comparison(
                    rule_code=code,
                    status="pass" if fact is not None else "needs_review",
                    left=_operand_from_fact(analysis, None, fact),
                    right=_operand(),
                    expected_value=name,
                    observed_value=fact.raw_value if fact else None,
                    unit=fact.unit if fact else None,
                    message=(
                        f"{_document_label(analysis)}에서 필수 {field_label}을 확인했습니다."
                        if fact is not None
                        else f"{_document_label(analysis)}에서 필수 {field_label}을 확인할 수 없습니다."
                    ),
                )
            )
    return comparisons


def _invoice_packing_checks(analyses: list[DocumentAnalysisV3]) -> list[ComparisonV3]:
    invoice, packing, ambiguity = _unique_pair(analyses, "invoice", "packing_list")
    if ambiguity is not None:
        return [ambiguity]
    if invoice is None or packing is None:
        return [
            _comparison(
                rule_code="invoice_packing_quantity_unit",
                status="skip",
                left=_operand(document_id=invoice.document_id if invoice else None),
                right=_operand(document_id=packing.document_id if packing else None),
                message="상업송장과 포장명세서가 모두 있을 때 수량과 단위를 대조합니다.",
            ),
            _invoice_total_comparison(invoice),
        ]
    comparisons = _match_invoice_packing_items(invoice, packing)
    comparisons.append(_invoice_total_comparison(invoice))
    return comparisons


def _match_invoice_packing_items(
    invoice: DocumentAnalysisV3, packing: DocumentAnalysisV3
) -> list[ComparisonV3]:
    if not invoice.line_items or not packing.line_items:
        return [
            _comparison(
                rule_code="invoice_packing_item_match",
                status="needs_review",
                left=_operand(document_id=invoice.document_id),
                right=_operand(document_id=packing.document_id),
                message="상업송장 또는 포장명세서에서 비교 가능한 품목을 추출하지 못했습니다.",
            )
        ]
    comparisons: list[ComparisonV3] = []
    unmatched = set(range(len(packing.line_items)))
    for invoice_item in invoice.line_items:
        matches = [
            index for index in unmatched if _item_match(invoice_item, packing.line_items[index])
        ]
        if len(matches) != 1:
            comparisons.append(
                _comparison(
                    rule_code="invoice_packing_item_match",
                    status="needs_review",
                    left=_operand_from_item(invoice, invoice_item),
                    right=_operand(document_id=packing.document_id),
                    message="상업송장 품목에 대응하는 포장명세서 품목을 하나로 정할 수 없습니다.",
                )
            )
            continue
        packing_item = packing.line_items[matches[0]]
        unmatched.remove(matches[0])
        comparisons.append(_quantity_unit_comparison(invoice, invoice_item, packing, packing_item))
        comparisons.append(_invoice_amount_comparison(invoice, invoice_item))
    for index in sorted(unmatched):
        comparisons.append(
            _comparison(
                rule_code="invoice_packing_item_match",
                status="needs_review",
                left=_operand(document_id=invoice.document_id),
                right=_operand_from_item(packing, packing.line_items[index]),
                message="포장명세서 품목에 대응하는 상업송장 품목을 하나로 정할 수 없습니다.",
            )
        )
    return comparisons


def _quantity_unit_comparison(
    invoice: DocumentAnalysisV3,
    invoice_item: LineItemV3,
    packing: DocumentAnalysisV3,
    packing_item: LineItemV3,
) -> ComparisonV3:
    invoice_qty = _first_fact(invoice_item.facts, "quantity")
    packing_qty = _first_fact(packing_item.facts, "quantity")
    if invoice_qty is None or packing_qty is None:
        return _comparison(
            rule_code="invoice_packing_quantity_unit",
            status="needs_review",
            left=_operand_from_fact(invoice, invoice_item, invoice_qty),
            right=_operand_from_fact(packing, packing_item, packing_qty),
            message="상업송장과 포장명세서의 수량 또는 단위를 모두 확인할 수 없습니다.",
        )
    if invoice_qty.normalized_value is None or packing_qty.normalized_value is None:
        return _comparison(
            rule_code="invoice_packing_quantity_unit",
            status="needs_review",
            left=_operand_from_fact(invoice, invoice_item, invoice_qty),
            right=_operand_from_fact(packing, packing_item, packing_qty),
            message="수량을 코드로 비교할 수 있는 형식으로 정규화하지 못했습니다.",
        )
    invoice_unit = _effective_quantity_unit(invoice_item, invoice_qty)
    packing_unit = _effective_quantity_unit(packing_item, packing_qty)
    if not invoice_unit or not packing_unit or invoice_unit != packing_unit:
        return _comparison(
            rule_code="invoice_packing_quantity_unit",
            status="needs_review",
            left=_operand_from_fact(invoice, invoice_item, invoice_qty),
            right=_operand_from_fact(packing, packing_item, packing_qty),
            expected_value=invoice_qty.normalized_value,
            observed_value=packing_qty.normalized_value,
            unit=invoice_unit or packing_unit,
            message="수량 단위가 없거나 서로 달라 자동으로 동등성을 판단하지 않았습니다.",
        )
    status = "pass" if invoice_qty.normalized_value == packing_qty.normalized_value else "mismatch"
    return _comparison(
        rule_code="invoice_packing_quantity_unit",
        status=status,
        left=_operand_from_fact(invoice, invoice_item, invoice_qty),
        right=_operand_from_fact(packing, packing_item, packing_qty),
        expected_value=invoice_qty.normalized_value,
        observed_value=packing_qty.normalized_value,
        unit=invoice_unit,
        message=(
            "상업송장과 포장명세서의 수량·단위가 일치합니다."
            if status == "pass"
            else "상업송장과 포장명세서의 수량이 일치하지 않습니다."
        ),
    )


def _cross_document_quantity_unit_comparison(
    left: DocumentAnalysisV3,
    left_item: LineItemV3,
    right: DocumentAnalysisV3,
    right_item: LineItemV3,
    *,
    rule_code: str,
    right_label: str,
) -> ComparisonV3:
    left_quantity = _first_fact(left_item.facts, "quantity")
    right_quantity = _first_fact(right_item.facts, "quantity")
    if left_quantity is None or right_quantity is None:
        return _comparison(
            rule_code=rule_code,
            status="needs_review",
            left=_operand_from_fact(left, left_item, left_quantity),
            right=_operand_from_fact(right, right_item, right_quantity),
            message=f"상업송장과 {right_label}의 수량 또는 단위를 모두 확인할 수 없습니다.",
        )
    if left_quantity.normalized_value is None or right_quantity.normalized_value is None:
        return _comparison(
            rule_code=rule_code,
            status="needs_review",
            left=_operand_from_fact(left, left_item, left_quantity),
            right=_operand_from_fact(right, right_item, right_quantity),
            message="수량을 코드로 비교할 수 있는 형식으로 정규화하지 못했습니다.",
        )
    left_unit = _effective_quantity_unit(left_item, left_quantity)
    right_unit = _effective_quantity_unit(right_item, right_quantity)
    if not left_unit or not right_unit or left_unit != right_unit:
        return _comparison(
            rule_code=rule_code,
            status="needs_review",
            left=_operand_from_fact(left, left_item, left_quantity),
            right=_operand_from_fact(right, right_item, right_quantity),
            expected_value=left_quantity.normalized_value,
            observed_value=right_quantity.normalized_value,
            unit=left_unit or right_unit,
            message="수량 단위가 없거나 서로 달라 자동으로 동등성을 판단하지 않았습니다.",
        )
    status = (
        "pass" if left_quantity.normalized_value == right_quantity.normalized_value else "mismatch"
    )
    return _comparison(
        rule_code=rule_code,
        status=status,
        left=_operand_from_fact(left, left_item, left_quantity),
        right=_operand_from_fact(right, right_item, right_quantity),
        expected_value=left_quantity.normalized_value,
        observed_value=right_quantity.normalized_value,
        unit=left_unit,
        message=(
            f"상업송장과 {right_label}의 수량·단위가 일치합니다."
            if status == "pass"
            else f"상업송장과 {right_label}의 수량이 일치하지 않습니다."
        ),
    )


def _invoice_amount_comparison(invoice: DocumentAnalysisV3, item: LineItemV3) -> ComparisonV3:
    quantity = _first_fact(item.facts, "quantity")
    unit_price = _first_fact(item.facts, "unit_price")
    amount = _first_fact(item.facts, "amount")
    facts = [fact for fact in (quantity, unit_price) if fact is not None]
    if quantity is None or unit_price is None or amount is None:
        return _comparison(
            rule_code="invoice_line_amount",
            status="needs_review",
            left=_operand_from_facts(invoice, item, facts),
            right=_operand_from_fact(invoice, item, amount),
            message="상업송장 품목의 수량·단가·금액을 모두 확인할 수 없습니다.",
        )
    if (
        quantity.normalized_value is None
        or unit_price.normalized_value is None
        or amount.normalized_value is None
        or not unit_price.unit
        or unit_price.unit != amount.unit
    ):
        return _comparison(
            rule_code="invoice_line_amount",
            status="needs_review",
            left=_operand_from_facts(invoice, item, facts),
            right=_operand_from_fact(invoice, item, amount),
            message="상업송장 품목 금액을 같은 통화와 숫자 형식으로 계산할 수 없습니다.",
        )
    expected = Decimal(quantity.normalized_value) * Decimal(unit_price.normalized_value)
    expected_text = _decimal_text(expected)
    status = "pass" if expected_text == amount.normalized_value else "mismatch"
    return _comparison(
        rule_code="invoice_line_amount",
        status=status,
        left=_operand_from_facts(
            invoice,
            item,
            facts,
            display_value=f"{quantity.normalized_value} × {unit_price.normalized_value}",
        ),
        right=_operand_from_fact(invoice, item, amount),
        expected_value=expected_text,
        observed_value=amount.normalized_value,
        unit=amount.unit,
        message=(
            "상업송장 품목의 수량×단가와 금액이 일치합니다."
            if status == "pass"
            else "상업송장 품목의 수량×단가와 기재 금액이 일치하지 않습니다."
        ),
    )


def _invoice_total_comparison(invoice: DocumentAnalysisV3 | None) -> ComparisonV3:
    if invoice is None:
        return _comparison(
            rule_code="invoice_total_amount",
            status="skip",
            left=_operand(),
            right=_operand(),
            message="상업송장이 있을 때 품목 금액 합계와 총액을 대조합니다.",
        )
    total = _first_fact(invoice.facts, "total_amount")
    amount_facts = [
        fact for item in invoice.line_items for fact in item.facts if fact.name == "amount"
    ]
    if total is None or not amount_facts:
        return _comparison(
            rule_code="invoice_total_amount",
            status="needs_review",
            left=_operand_from_facts(invoice, None, amount_facts),
            right=_operand_from_fact(invoice, None, total),
            message="상업송장 품목 금액과 총액을 모두 확인할 수 없습니다.",
        )
    if (
        total.normalized_value is None
        or not total.unit
        or any(fact.normalized_value is None or fact.unit != total.unit for fact in amount_facts)
    ):
        return _comparison(
            rule_code="invoice_total_amount",
            status="needs_review",
            left=_operand_from_facts(invoice, None, amount_facts),
            right=_operand_from_fact(invoice, None, total),
            message="상업송장 품목 금액과 총액의 통화 또는 숫자 형식을 통일할 수 없습니다.",
        )
    expected = sum((Decimal(fact.normalized_value or "0") for fact in amount_facts), Decimal("0"))
    expected_text = _decimal_text(expected)
    status = "pass" if expected_text == total.normalized_value else "mismatch"
    return _comparison(
        rule_code="invoice_total_amount",
        status=status,
        left=_operand_from_facts(invoice, None, amount_facts, display_value=expected_text),
        right=_operand_from_fact(invoice, None, total),
        expected_value=expected_text,
        observed_value=total.normalized_value,
        unit=total.unit,
        message=(
            "상업송장 품목 금액 합계와 총액이 일치합니다."
            if status == "pass"
            else "상업송장 품목 금액 합계와 총액이 일치하지 않습니다."
        ),
    )


def _invoice_origin_checks(analyses: list[DocumentAnalysisV3]) -> list[ComparisonV3]:
    invoice, certificate, ambiguity = _unique_pair(analyses, "invoice", "certificate_of_origin")
    if ambiguity is not None:
        return [ambiguity]
    if invoice is None or certificate is None:
        return [
            _comparison(
                rule_code="invoice_certificate_origin",
                status="skip",
                left=_operand(document_id=invoice.document_id if invoice else None),
                right=_operand(document_id=certificate.document_id if certificate else None),
                message="상업송장과 원산지증명서가 모두 있을 때 원산지를 대조합니다.",
            )
        ]
    invoice_origin = _first_fact_from_analysis(invoice, *_ORIGIN_FIELDS)
    certificate_origin = _first_fact_from_analysis(certificate, *_ORIGIN_FIELDS)
    return [
        _simple_fact_comparison(
            "invoice_certificate_origin",
            invoice,
            None,
            invoice_origin,
            certificate,
            None,
            certificate_origin,
            missing_message="상업송장과 원산지증명서의 원산지를 모두 확인할 수 없습니다.",
            pass_message="상업송장과 원산지증명서의 원산지가 일치합니다.",
            mismatch_message="상업송장과 원산지증명서의 원산지가 일치하지 않습니다.",
        )
    ]


def _invoice_export_declaration_checks(
    analyses: list[DocumentAnalysisV3], *, trade_type: Literal["import", "export"]
) -> list[ComparisonV3]:
    """Compare the source-grounded export declaration with commercial documents.

    The declaration is not treated as a second invoice.  It only supplies
    corroborating declaration facts, and ambiguous item correspondence remains
    a human-review result rather than a silent match.
    """

    invoice, declaration, ambiguity = _unique_pair(analyses, "invoice", "export_declaration")
    if ambiguity is not None:
        return [ambiguity]
    if declaration is None:
        if trade_type != "export":
            return []
        return [
            _comparison(
                rule_code="export_declaration_presence",
                status="needs_review",
                left=_operand(document_id=invoice.document_id if invoice else None),
                right=_operand(),
                message="수출 사건에 필요한 수출신고증을 확인할 수 없습니다.",
            )
        ]

    destination = _first_fact_from_analysis(declaration, "destination_country")
    comparisons: list[ComparisonV3] = [
        _comparison(
            rule_code="export_declaration_destination_country",
            status="pass"
            if destination is not None
            and destination.normalized_value is not None
            and _is_country_code(destination.normalized_value)
            else "needs_review",
            left=_operand_from_fact(declaration, None, destination),
            right=_operand(),
            expected_value="destination_country",
            observed_value=destination.raw_value if destination else None,
            message=(
                "수출신고증에서 목적국 코드를 확인했습니다."
                if destination is not None
                and destination.normalized_value is not None
                and _is_country_code(destination.normalized_value)
                else "수출신고증에서 두 자리 목적국 코드를 확인할 수 없습니다."
            ),
        )
    ]
    if invoice is None:
        comparisons.extend(
            [
                _comparison(
                    rule_code=rule_code,
                    status="skip",
                    left=_operand(),
                    right=_operand(document_id=declaration.document_id),
                    message="상업송장과 수출신고증이 모두 있을 때 대조합니다.",
                )
                for rule_code in (
                    "invoice_export_declaration_invoice_no",
                    "invoice_export_declaration_total_amount",
                    "invoice_export_declaration_origin",
                    "invoice_export_declaration_item_match",
                )
            ]
        )
        return comparisons

    comparisons.extend(
        [
            _simple_fact_comparison(
                "invoice_export_declaration_invoice_no",
                invoice,
                None,
                _first_fact_from_analysis(invoice, "invoice_no"),
                declaration,
                None,
                _first_fact_from_analysis(declaration, "invoice_no"),
                missing_message="상업송장과 수출신고증의 상업송장 번호를 모두 확인할 수 없습니다.",
                pass_message="상업송장과 수출신고증의 상업송장 번호가 일치합니다.",
                mismatch_message="상업송장과 수출신고증의 상업송장 번호가 일치하지 않습니다.",
            ),
            _simple_fact_comparison(
                "invoice_export_declaration_total_amount",
                invoice,
                None,
                _first_fact_from_analysis(invoice, "total_amount"),
                declaration,
                None,
                _first_fact_from_analysis(declaration, "total_amount"),
                missing_message="상업송장과 수출신고증의 총액을 모두 확인할 수 없습니다.",
                pass_message="상업송장과 수출신고증의 총액이 일치합니다.",
                mismatch_message="상업송장과 수출신고증의 총액이 일치하지 않습니다.",
            ),
            _simple_fact_comparison(
                "invoice_export_declaration_origin",
                invoice,
                None,
                _first_fact_from_analysis(invoice, *_ORIGIN_FIELDS),
                declaration,
                None,
                _first_fact_from_analysis(declaration, *_ORIGIN_FIELDS),
                missing_message="상업송장과 수출신고증의 원산지를 모두 확인할 수 없습니다.",
                pass_message="상업송장과 수출신고증의 원산지가 일치합니다.",
                mismatch_message="상업송장과 수출신고증의 원산지가 일치하지 않습니다.",
            ),
        ]
    )
    comparisons.extend(_match_invoice_export_declaration_items(invoice, declaration))
    return comparisons


def _match_invoice_export_declaration_items(
    invoice: DocumentAnalysisV3, declaration: DocumentAnalysisV3
) -> list[ComparisonV3]:
    if not invoice.line_items or not declaration.line_items:
        return [
            _comparison(
                rule_code="invoice_export_declaration_item_match",
                status="needs_review",
                left=_operand(document_id=invoice.document_id),
                right=_operand(document_id=declaration.document_id),
                message="상업송장 또는 수출신고증에서 비교 가능한 품목을 추출하지 못했습니다.",
            )
        ]
    comparisons: list[ComparisonV3] = []
    unmatched = set(range(len(declaration.line_items)))
    for invoice_item in invoice.line_items:
        matches = [
            index
            for index in unmatched
            if _cross_document_item_match(invoice_item, declaration.line_items[index])
        ]
        if len(matches) != 1:
            comparisons.append(
                _comparison(
                    rule_code="invoice_export_declaration_item_match",
                    status="needs_review",
                    left=_operand_from_item(invoice, invoice_item),
                    right=_operand(document_id=declaration.document_id),
                    message="상업송장 품목에 대응하는 수출신고증 품목을 하나로 정할 수 없습니다.",
                )
            )
            continue
        declaration_item = declaration.line_items[matches[0]]
        unmatched.remove(matches[0])
        comparisons.extend(
            [
                _simple_fact_comparison(
                    "invoice_export_declaration_hs_code",
                    invoice,
                    invoice_item,
                    _first_fact(invoice_item.facts, "hs_code"),
                    declaration,
                    declaration_item,
                    _first_fact(declaration_item.facts, "hs_code"),
                    missing_message="상업송장과 수출신고증 품목의 HS 코드를 모두 확인할 수 없습니다.",
                    pass_message="상업송장과 수출신고증 품목의 HS 코드가 일치합니다.",
                    mismatch_message="상업송장과 수출신고증 품목의 HS 코드가 일치하지 않습니다.",
                ),
                _cross_document_quantity_unit_comparison(
                    invoice,
                    invoice_item,
                    declaration,
                    declaration_item,
                    rule_code="invoice_export_declaration_quantity_unit",
                    right_label="수출신고증",
                ),
                _simple_fact_comparison(
                    "invoice_export_declaration_amount",
                    invoice,
                    invoice_item,
                    _first_fact(invoice_item.facts, "amount"),
                    declaration,
                    declaration_item,
                    _first_fact(declaration_item.facts, "amount"),
                    missing_message="상업송장과 수출신고증 품목의 금액을 모두 확인할 수 없습니다.",
                    pass_message="상업송장과 수출신고증 품목의 금액이 일치합니다.",
                    mismatch_message="상업송장과 수출신고증 품목의 금액이 일치하지 않습니다.",
                ),
            ]
        )
    for index in sorted(unmatched):
        comparisons.append(
            _comparison(
                rule_code="invoice_export_declaration_item_match",
                status="needs_review",
                left=_operand(document_id=invoice.document_id),
                right=_operand_from_item(declaration, declaration.line_items[index]),
                message="수출신고증 품목에 대응하는 상업송장 품목을 하나로 정할 수 없습니다.",
            )
        )
    return comparisons


def _transport_packing_checks(
    analyses: list[DocumentAnalysisV3], transport_type: Literal["bill_of_lading", "air_waybill"]
) -> list[ComparisonV3]:
    """Compare one transport document with its packing list without assuming sea freight.

    B/L and AWB have different transport identifiers.  Container numbers are
    a sea-freight field and therefore intentionally skipped for AWB instead of
    being treated as a missing AWB fact.
    """

    transport, packing, ambiguity = _unique_pair(analyses, transport_type, "packing_list")
    if ambiguity is not None:
        return [ambiguity]
    is_bill = transport_type == "bill_of_lading"
    identifier_name = "bl_no" if is_bill else "awb_no"
    identifier_label = "B/L" if is_bill else "AWB"
    prefix = "bill" if is_bill else "awb"
    # An AWB is an alternative transport document, not an extra mandatory
    # document beside a B/L.  Do not add meaningless skipped AWB rows to a
    # sea-freight review that contains no AWB at all.
    if not is_bill and transport is None:
        return []
    if transport is None or packing is None:
        rules = [f"{prefix}_packing_{identifier_name}", f"{prefix}_packing_gross_weight"]
        if is_bill:
            rules.insert(1, "bill_packing_container")
        return [
            _comparison(
                rule_code=rule,
                status="skip",
                left=_operand(document_id=transport.document_id if transport else None),
                right=_operand(document_id=packing.document_id if packing else None),
                message=f"{identifier_label}과 포장명세서가 모두 있을 때 대조합니다.",
            )
            for rule in rules
        ]
    checks = [
        _simple_fact_comparison(
            f"{prefix}_packing_{identifier_name}",
            transport,
            None,
            _first_fact_from_analysis(transport, identifier_name),
            packing,
            None,
            # Packing lists frequently carry a B/L reference but not an AWB.
            # We only compare an AWB when the packing list explicitly has it.
            _first_fact_from_analysis(packing, identifier_name),
            missing_message=f"{identifier_label} 번호를 두 문서에서 모두 확인할 수 없습니다.",
            pass_message=f"{identifier_label} 번호가 일치합니다.",
            mismatch_message=f"{identifier_label} 번호가 일치하지 않습니다.",
        ),
        _simple_fact_comparison(
            f"{prefix}_packing_gross_weight",
            transport,
            None,
            _first_fact_from_analysis(transport, "gross_weight_kg"),
            packing,
            None,
            _first_fact_from_analysis(packing, "total_gross_weight_kg"),
            missing_message="총중량을 두 문서에서 모두 확인할 수 없습니다.",
            pass_message=f"{identifier_label}과 포장명세서의 총중량이 일치합니다.",
            mismatch_message=f"{identifier_label}과 포장명세서의 총중량이 일치하지 않습니다.",
        ),
    ]
    if not is_bill:
        return checks
    checks.insert(
        1,
        _simple_fact_comparison(
            "bill_packing_container",
            transport,
            None,
            _first_fact_from_analysis(transport, "container_numbers"),
            packing,
            None,
            _first_fact_from_analysis(packing, "container_numbers"),
            missing_message="컨테이너 번호를 두 문서에서 모두 확인할 수 없습니다.",
            pass_message="컨테이너 번호가 일치합니다.",
            mismatch_message="컨테이너 번호가 일치하지 않습니다.",
        ),
    )
    return checks


def _simple_fact_comparison(
    rule_code: str,
    left_analysis: DocumentAnalysisV3,
    left_item: LineItemV3 | None,
    left_fact: FactV3 | None,
    right_analysis: DocumentAnalysisV3,
    right_item: LineItemV3 | None,
    right_fact: FactV3 | None,
    *,
    missing_message: str,
    pass_message: str,
    mismatch_message: str,
) -> ComparisonV3:
    if left_fact is None or right_fact is None:
        return _comparison(
            rule_code=rule_code,
            status="needs_review",
            left=_operand_from_fact(left_analysis, left_item, left_fact),
            right=_operand_from_fact(right_analysis, right_item, right_fact),
            message=missing_message,
        )
    if left_fact.normalized_value is None or right_fact.normalized_value is None:
        return _comparison(
            rule_code=rule_code,
            status="needs_review",
            left=_operand_from_fact(left_analysis, left_item, left_fact),
            right=_operand_from_fact(right_analysis, right_item, right_fact),
            message="값을 코드로 비교할 수 있는 형식으로 정규화하지 못했습니다.",
        )
    if left_fact.unit != right_fact.unit:
        return _comparison(
            rule_code=rule_code,
            status="needs_review",
            left=_operand_from_fact(left_analysis, left_item, left_fact),
            right=_operand_from_fact(right_analysis, right_item, right_fact),
            expected_value=left_fact.normalized_value,
            observed_value=right_fact.normalized_value,
            unit=left_fact.unit or right_fact.unit,
            message="단위가 없거나 서로 달라 자동으로 동등성을 판단하지 않았습니다.",
        )
    status = "pass" if left_fact.normalized_value == right_fact.normalized_value else "mismatch"
    return _comparison(
        rule_code=rule_code,
        status=status,
        left=_operand_from_fact(left_analysis, left_item, left_fact),
        right=_operand_from_fact(right_analysis, right_item, right_fact),
        expected_value=left_fact.normalized_value,
        observed_value=right_fact.normalized_value,
        unit=left_fact.unit,
        message=pass_message if status == "pass" else mismatch_message,
    )


def _unique_pair(
    analyses: list[DocumentAnalysisV3], left_type: str, right_type: str
) -> tuple[DocumentAnalysisV3 | None, DocumentAnalysisV3 | None, ComparisonV3 | None]:
    left = [analysis for analysis in analyses if analysis.document_type == left_type]
    right = [analysis for analysis in analyses if analysis.document_type == right_type]
    if len(left) <= 1 and len(right) <= 1:
        return (left[0] if left else None, right[0] if right else None, None)
    return (
        None,
        None,
        _comparison(
            rule_code="document_pair_selection",
            status="needs_review",
            left=_operand(document_id=left[0].document_id if left else None),
            right=_operand(document_id=right[0].document_id if right else None),
            message="같은 유형의 문서가 여러 건이라 자동 비교 대상을 하나로 정하지 않았습니다.",
        ),
    )


def _item_match(left: LineItemV3, right: LineItemV3) -> bool:
    left_hs = _first_fact(left.facts, "hs_code")
    right_hs = _first_fact(right.facts, "hs_code")
    if left_hs and right_hs and left_hs.normalized_value and right_hs.normalized_value:
        return left_hs.normalized_value == right_hs.normalized_value
    left_description = _first_fact(left.facts, *_DESCRIPTION_FIELDS)
    right_description = _first_fact(right.facts, *_DESCRIPTION_FIELDS)
    return bool(
        left_description
        and right_description
        and left_description.normalized_value
        and left_description.normalized_value == right_description.normalized_value
    )


def _cross_document_item_match(left: LineItemV3, right: LineItemV3) -> bool:
    """Match a declared export item conservatively before comparing its fields.

    A matching description is enough to reveal a HS-code mismatch.  If neither
    the description nor the HS code identifies one counterpart, the caller
    leaves the relation as ``needs_review``.
    """

    left_description = _first_fact(left.facts, *_DESCRIPTION_FIELDS)
    right_description = _first_fact(right.facts, *_DESCRIPTION_FIELDS)
    if (
        left_description
        and right_description
        and left_description.normalized_value
        and left_description.normalized_value == right_description.normalized_value
    ):
        return True
    left_hs = _first_fact(left.facts, "hs_code")
    right_hs = _first_fact(right.facts, "hs_code")
    return bool(
        left_hs
        and right_hs
        and left_hs.normalized_value
        and left_hs.normalized_value == right_hs.normalized_value
    )


def _findings_from_comparisons(comparisons: Iterable[ComparisonV3]) -> list[FindingV3]:
    findings: list[FindingV3] = []
    for comparison in comparisons:
        if comparison.status not in {"mismatch", "needs_review"}:
            continue
        if comparison.rule_code == "document_type_classification":
            code = "document_type_unknown"
            severity = "medium"
        elif comparison.rule_code == "hs_code_presence":
            code = "hs_code_not_found"
            severity = "medium"
        elif comparison.rule_code == "required_field_presence":
            code = "missing_required_field"
            severity = "medium"
        else:
            suffix = "mismatch" if comparison.status == "mismatch" else "needs_review"
            code = f"{comparison.rule_code}_{suffix}"[:128]
            severity = "high" if comparison.status == "mismatch" else "medium"
        document_ids = [
            document_id
            for document_id in (comparison.left.document_id, comparison.right.document_id)
            if document_id is not None
        ]
        fact_ids = [*comparison.left.fact_ids, *comparison.right.fact_ids]
        findings.append(
            _finding(
                code=code,
                severity=severity,
                message=comparison.message,
                comparison_id=comparison.comparison_id,
                document_ids=document_ids,
                fact_ids=fact_ids,
            )
        )
    return findings


def _warning_findings(analyses: Iterable[DocumentAnalysisV3]) -> list[FindingV3]:
    findings: list[FindingV3] = []
    for analysis in analyses:
        for warning in analysis.warnings:
            findings.append(
                _finding(
                    code=warning.code,
                    severity="medium",
                    message=warning.message,
                    document_ids=[analysis.document_id],
                )
            )
    return findings


def _finding(
    *,
    code: str,
    severity: str,
    message: str,
    item_id: str | None = None,
    evidence_ids: list[str] | None = None,
    comparison_id: str | None = None,
    document_ids: list[str] | None = None,
    fact_ids: list[str] | None = None,
) -> FindingV3:
    evidence_ids = evidence_ids or []
    document_ids = list(dict.fromkeys(document_ids or []))
    fact_ids = list(dict.fromkeys(fact_ids or []))
    identity = {
        "code": code,
        "comparison_id": comparison_id,
        "document_ids": document_ids,
        "fact_ids": fact_ids,
        "item_id": item_id,
        "evidence_ids": evidence_ids,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return FindingV3(
        finding_id=f"finding-{digest}",
        kind="regulatory_finding" if evidence_ids else "validation_issue",
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        item_id=item_id,
        evidence_ids=evidence_ids,
        comparison_id=comparison_id,
        document_ids=document_ids,
        fact_ids=fact_ids,
    )


def _dedupe_findings(findings: Iterable[FindingV3]) -> list[FindingV3]:
    unique: dict[str, FindingV3] = {}
    for finding in findings:
        unique.setdefault(finding.finding_id, finding)
    return list(unique.values())


def _workflow_stages(
    analyses: list[DocumentAnalysisV3],
    comparisons: list[ComparisonV3],
    reextractions: list[ReextractionV3],
    *,
    retry_unavailable: bool,
) -> list[WorkflowStageV3]:
    unknown_count = sum(analysis.document_type == "unknown" for analysis in analyses)
    warning_count = sum(len(analysis.warnings) for analysis in analyses)
    pair_exists = any(
        analysis.document_type
        in {
            "invoice",
            "packing_list",
            "bill_of_lading",
            "air_waybill",
            "certificate_of_origin",
            "export_declaration",
            "supporting_evidence",
        }
        for analysis in analyses
    )
    validation_needs_review = sum(
        comparison.status in {"mismatch", "needs_review"} for comparison in comparisons
    )
    return [
        WorkflowStageV3(
            stage="classify_documents",
            status="needs_review" if unknown_count else "completed",
            attempt_count=len(analyses),
            summary=(
                f"문서 {len(analyses)}건을 분류했고 {unknown_count}건은 담당자 확인이 필요합니다."
            ),
        ),
        WorkflowStageV3(
            stage="extract_facts",
            status="needs_review" if warning_count else "completed",
            attempt_count=len(analyses) + len(reextractions),
            summary=f"문서에서 구조화 사실을 추출했고 인용 경고는 {warning_count}건입니다.",
        ),
        WorkflowStageV3(
            stage="normalize_facts",
            status="completed",
            attempt_count=1,
            summary="수량·금액·식별자·원산지 값을 코드 비교용으로 정규화했습니다.",
        ),
        WorkflowStageV3(
            stage="match_documents",
            status="completed" if pair_exists else "skipped",
            attempt_count=1 if pair_exists else 0,
            summary=(
                "문서와 품목의 비교 대상을 결정했습니다."
                if pair_exists
                else "비교 가능한 문서 쌍이 없습니다."
            ),
        ),
        WorkflowStageV3(
            stage="validate_documents",
            status="needs_review" if validation_needs_review else "completed",
            attempt_count=2 if reextractions else 1,
            summary=f"코드 기반 정합성 비교 {len(comparisons)}건을 수행했습니다.",
        ),
        WorkflowStageV3(
            stage="reextract_documents",
            status=("needs_review" if retry_unavailable else "completed")
            if reextractions
            else "skipped",
            attempt_count=len(reextractions),
            summary=(
                f"누락·모호한 값 보완을 위해 {len(reextractions)}개 문서를 최대 한 번씩 재추출했습니다."
                if reextractions
                else "재추출이 필요한 문서가 없습니다."
            ),
        ),
        WorkflowStageV3(
            stage="retrieve_evidence",
            status="skipped",
            attempt_count=0,
            summary="법령 근거 조회를 아직 기록하지 않았습니다.",
        ),
        WorkflowStageV3(
            stage="generate_result",
            status="skipped",
            attempt_count=0,
            summary="최종 결과 생성을 아직 기록하지 않았습니다.",
        ),
    ]


def _evidence_summary(data_status: str, *, excluded_document_count: int = 0) -> str:
    summary = {
        "ok": "품목 기준으로 법령 근거를 한 번 조회했습니다.",
        "partial": "일부 법령 근거 조회 결과를 확인하지 못해 담당자 검토가 필요합니다.",
        "failed": "법령 근거 조회에 실패했습니다.",
    }.get(data_status, "법령 근거 조회 상태를 확인할 수 없습니다.")
    if excluded_document_count:
        return summary + " HS 코드가 없는 문서는 법령 근거 조회 대상에서 제외했습니다."
    return summary


def _export_destination_country(analyses: Iterable[DocumentAnalysisV3]) -> str | None:
    """Return one source-grounded alpha-2 destination for an export request.

    The Data item contract needs a country code.  Country names, conflicting
    values, and values absent from an export declaration are intentionally not
    guessed or mapped here; the document must be clarified first.
    """

    values = {
        fact.normalized_value
        for analysis in analyses
        if analysis.document_type == "export_declaration"
        for fact in analysis.facts
        if fact.name in _DESTINATION_FIELDS
        and fact.normalized_value is not None
        and _is_country_code(fact.normalized_value)
    }
    return next(iter(values)) if len(values) == 1 else None


def _is_country_code(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{2}", value))


def _reextraction_targets(
    request: ServiceReviewRunRequestV3,
    analyses: Mapping[str, DocumentAnalysis],
) -> dict[str, list[str]]:
    """Select only fields that unlock a defined comparison; each doc gets one retry."""

    targets: dict[str, set[str]] = {document.document_id: set() for document in request.documents}
    by_type: dict[str, list[Any]] = {}
    for document in request.documents:
        candidate = analyses[document.document_id].candidate
        by_type.setdefault(candidate.document_type, []).append(document)
        if candidate.document_type == "unknown":
            targets[document.document_id].add("document_type")
        available = _candidate_field_names(candidate)
        for name in _base_required_fields(candidate.document_type):
            if not _candidate_has(available, name):
                targets[document.document_id].add(name)
        targets[document.document_id].update(_ungrounded_candidate_fields(document, candidate))

    invoices = by_type.get("invoice", [])
    packings = by_type.get("packing_list", [])
    certificates = by_type.get("certificate_of_origin", [])
    bills = by_type.get("bill_of_lading", [])
    awbs = by_type.get("air_waybill", [])
    declarations = by_type.get("export_declaration", [])
    if len(invoices) == 1 and len(packings) == 1:
        _request_missing(
            analyses,
            invoices[0],
            targets,
            ("description", "quantity", "unit", "unit_price", "amount", "total_amount"),
        )
        _request_missing(
            analyses,
            packings[0],
            targets,
            (
                "description",
                "quantity",
                "unit",
                "bl_no",
                "container_numbers",
                "total_gross_weight_kg",
            ),
        )
    if len(invoices) == 1 and len(certificates) == 1:
        _request_missing(analyses, invoices[0], targets, ("country_of_origin", "origin_country"))
        _request_missing(analyses, certificates[0], targets, ("origin_country",))
    if len(bills) == 1 and len(packings) == 1:
        _request_missing(
            analyses, bills[0], targets, ("bl_no", "container_numbers", "gross_weight_kg")
        )
        _request_missing(
            analyses, packings[0], targets, ("bl_no", "container_numbers", "total_gross_weight_kg")
        )
    if len(awbs) == 1 and len(packings) == 1:
        _request_missing(analyses, awbs[0], targets, ("awb_no", "gross_weight_kg"))
        _request_missing(analyses, packings[0], targets, ("awb_no", "total_gross_weight_kg"))
    if request.trade_type == "export":
        if len(declarations) != 1:
            # A missing or ambiguous declaration is represented by the bounded
            # comparison path.  Retrying an unrelated document cannot create
            # a source-grounded declaration.
            return {
                document_id: sorted(values) for document_id, values in targets.items() if values
            }
        declaration = declarations[0]
        _request_missing(
            analyses,
            declaration,
            targets,
            (
                "export_declaration_no",
                "invoice_no",
                "total_amount",
                "origin_country",
                "destination_country",
                "description",
                "hs_code",
                "quantity",
                "unit",
                "amount",
            ),
        )
        if len(invoices) == 1:
            _request_missing(
                analyses,
                invoices[0],
                targets,
                (
                    "invoice_no",
                    "total_amount",
                    "origin_country",
                    "description",
                    "hs_code",
                    "quantity",
                    "unit",
                    "amount",
                ),
            )
    return {document_id: sorted(values) for document_id, values in targets.items() if values}


def _base_required_fields(document_type: str) -> tuple[str, ...]:
    return {
        "invoice": ("invoice_no", "total_amount", "hs_code"),
        "packing_list": ("packing_no",),
        "certificate_of_origin": ("certificate_no", "origin_country", "hs_code"),
        "bill_of_lading": ("bl_no",),
        "air_waybill": ("awb_no",),
        "export_declaration": ("export_declaration_no",),
        "supporting_evidence": ("submission_receipt_no",),
    }.get(document_type, ())


def _request_missing(
    analyses: Mapping[str, DocumentAnalysis],
    document: Any,
    targets: dict[str, set[str]],
    names: tuple[str, ...],
) -> None:
    available = _candidate_field_names(analyses[document.document_id].candidate)
    for name in names:
        if not _candidate_has(available, name):
            targets[document.document_id].add(name)


def _candidate_field_names(candidate: ClaudeDocumentCandidate) -> set[str]:
    return {field.name for field in candidate.fields} | {
        field.name for item in candidate.line_items for field in item.fields
    }


def _ungrounded_candidate_fields(document: Any, candidate: ClaudeDocumentCandidate) -> set[str]:
    """Retry only facts whose supplied source quote is unusable in this input."""

    text = document.content_text
    targets: set[str] = set()
    if candidate.document_type != "unknown" and candidate.classification_source_quote not in text:
        targets.add("document_type")
    for field in candidate.fields:
        if field.value not in field.source_quote or field.source_quote not in text:
            targets.add(field.name)
    for item in candidate.line_items:
        if item.source_quote not in text:
            targets.add("line_items")
        for field in item.fields:
            if (
                field.value not in field.source_quote
                or field.source_quote not in text
                or field.source_quote not in item.source_quote
            ):
                targets.add(field.name)
    return targets


def _candidate_has(available: set[str], name: str) -> bool:
    requested = _canonical_fact_name(name)
    return any(_canonical_fact_name(candidate) == requested for candidate in available)


def _prefer_more_complete(current: DocumentAnalysis, retry: DocumentAnalysis) -> DocumentAnalysis:
    def score(value: DocumentAnalysis) -> int:
        candidate = value.candidate
        return (
            (10 if candidate.document_type in _KNOWN_TYPES else 0)
            + len(candidate.fields)
            + sum(len(item.fields) + 1 for item in candidate.line_items)
        )

    return retry if score(retry) >= score(current) else current


def _first_fact_from_analysis(analysis: DocumentAnalysisV3, *names: str) -> FactV3 | None:
    direct = _first_fact(analysis.facts, *names)
    if direct is not None:
        return direct
    for item in analysis.line_items:
        value = _first_fact(item.facts, *names)
        if value is not None:
            return value
    return None


def _first_fact(facts: Iterable[FactV3], *names: str) -> FactV3 | None:
    expected = {_canonical_fact_name(name) for name in names}
    candidates = [fact for fact in facts if _canonical_fact_name(fact.name) in expected]
    return next(
        (fact for fact in reversed(candidates) if isinstance(fact.source, HumanProvidedSourceV3)),
        next(iter(candidates), None),
    )


def _effective_quantity_unit(item: LineItemV3, quantity: FactV3) -> str | None:
    """Use an explicit extracted unit when the quantity token itself has none."""

    if quantity.unit:
        return quantity.unit
    unit = _first_fact(item.facts, "unit")
    if unit is None:
        return None
    value = unit.normalized_value or unit.raw_value
    return re.sub(r"\s+", "", value).upper() or None


def _operand(
    *,
    document_id: str | None = None,
    item_id: str | None = None,
    fact_ids: Iterable[str] = (),
    display_value: str | None = None,
) -> ComparisonOperandV3:
    return ComparisonOperandV3(
        document_id=document_id,
        item_id=item_id,
        fact_ids=list(fact_ids),
        display_value=display_value,
    )


def _operand_from_fact(
    analysis: DocumentAnalysisV3, item: LineItemV3 | None, fact: FactV3 | None
) -> ComparisonOperandV3:
    return _operand(
        document_id=analysis.document_id,
        item_id=item.item_id if item else None,
        fact_ids=[fact.fact_id] if fact else [],
        display_value=fact.raw_value if fact else None,
    )


def _operand_from_facts(
    analysis: DocumentAnalysisV3,
    item: LineItemV3 | None,
    facts: Iterable[FactV3],
    *,
    display_value: str | None = None,
) -> ComparisonOperandV3:
    fact_list = list(facts)
    return _operand(
        document_id=analysis.document_id,
        item_id=item.item_id if item else None,
        fact_ids=[fact.fact_id for fact in fact_list],
        display_value=display_value or ", ".join(fact.raw_value for fact in fact_list) or None,
    )


def _operand_from_item(analysis: DocumentAnalysisV3, item: LineItemV3) -> ComparisonOperandV3:
    return _operand_from_facts(analysis, item, item.facts, display_value=item.source.quote)


def _comparison(
    *,
    rule_code: str,
    status: str,
    left: ComparisonOperandV3,
    right: ComparisonOperandV3,
    message: str,
    expected_value: str | None = None,
    observed_value: str | None = None,
    unit: str | None = None,
) -> ComparisonV3:
    identity = {
        "rule_code": rule_code,
        "left": left.model_dump(mode="json"),
        "right": right.model_dump(mode="json"),
        "expected_value": expected_value,
        "observed_value": observed_value,
        "unit": unit,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return ComparisonV3(
        comparison_id=f"comparison-{digest}",
        rule_code=rule_code,
        status=status,  # type: ignore[arg-type]
        left=left,
        right=right,
        expected_value=expected_value,
        observed_value=observed_value,
        unit=unit,
        message=message,
    )


def _safe_name_value(value: str) -> str:
    normalized = _SAFE_NAME.sub("_", value.strip().casefold()).strip("_")
    return normalized[:48] or "field"


__all__ = [
    "BoundedWorkflowOutput",
    "finding_for_evidence_requirement",
    "retrieval_items_from_analyses",
    "run_bounded_workflow",
]
