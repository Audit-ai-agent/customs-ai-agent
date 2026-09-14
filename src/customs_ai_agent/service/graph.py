"""LangGraph review graph: three agent nodes around the deterministic core.

Nodes
-----
``extraction_agent``        document classification and fact extraction (LLM or
                            deterministic analyzer); on the second iteration a
                            focused re-read of the fields the audit asked for.
``cross_validation_agent``  rendering, approved human corrections, the code
                            comparison rules, findings, and the audit decision:
                            does anything justify one more extraction pass?
``evidence_retrieval``      the single Data Evidence call (injected by the
                            profile runtime; skipped when no caller supplies it).
``expert_review_agent``     deterministic confidence, the review track, and the
                            advisory eligibility verdict (optionally tightened by
                            one LLM opinion that must cite existing IDs).
``report_generation``       the eight-section grounded report.

The audit loop is the only cycle: ``cross_validation_agent`` routes back to
``extraction_agent`` at most once, which is exactly the planner's second cycle.
Comparison verdicts always come from the rule engine; the LLM only labels and
explains them, and may ask for a re-read.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from customs_ai_agent.adapters.llm import DocumentAnalysis
from customs_ai_agent.schemas.v3 import (
    ComparisonV3,
    ConfidenceV3,
    DocumentAnalysisV3,
    EligibilityVerdictV3,
    EvidenceV3,
    FindingClassificationV3,
    FindingV3,
    GraphNodeRunV3,
    GraphTraceV3,
    ReextractionV3,
    ReportSectionV3,
    RequiredActionV1,
    ServiceReviewRunRequestV3,
    TemporalContextV3,
    TemporalCoverageV3,
    WorkflowV3,
)
from customs_ai_agent.scenario_transcript import ScenarioTranscriptRun
from customs_ai_agent.service.confidence import (
    attach_finding_confidence,
    comparison_key,
    review_confidence,
)
from customs_ai_agent.service.expert_review import merge_llm_verdict, rule_verdict
from customs_ai_agent.service.report import build_grounded_report_sections
from customs_ai_agent.service.runtime_errors import RuntimeExecutionError
from customs_ai_agent.service import workflow as wf


LoopReason = Literal["missing_fields", "ungrounded_fields", "mismatch_reread", "llm_reread_request"]
DataStatus = Literal["ok", "partial", "failed"]

_MAX_LLM_CONTEXT_ITEMS = 40
_ALLOWED_REREAD_FIELDS: frozenset[str] = frozenset(
    {
        *(name for kind in wf._KNOWN_TYPES for name in wf._base_required_fields(kind)),
        *wf._NUMERIC_FIELDS,
        *wf._IDENTIFIER_FIELDS,
        *wf._ORIGIN_FIELDS,
        *wf._DESTINATION_FIELDS,
        *wf._DESCRIPTION_FIELDS,
        *wf._HS_FIELDS,
        "document_type",
        "line_items",
    }
)
_ERROR_CATEGORIES = frozenset(
    {
        "value_mismatch",
        "missing_field",
        "unit_or_format",
        "document_type_uncertain",
        "regulatory_requirement",
        "other",
    }
)


def orchestrator_version() -> str:
    try:
        return importlib.metadata.version("langgraph")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - packaging guard
        return "unknown"


@dataclass(frozen=True)
class EvidenceStageResult:
    """What the profile runtime learned from its one Data Evidence request."""

    data_status: DataStatus
    skipped: bool
    evidence_records: list[dict[str, Any]]
    evidence_findings: list[FindingV3]
    public_evidence: list[EvidenceV3]
    temporal_context: TemporalContextV3 | None
    temporal_coverage: TemporalCoverageV3 | None


EvidenceStage = Callable[[ServiceReviewRunRequestV3, wf.BoundedWorkflowOutput], EvidenceStageResult]


@dataclass(frozen=True)
class ReviewGraphConfig:
    analyzer: Any
    mode: Literal["local-development", "functional-uat"] = "local-development"
    evidence_stage: EvidenceStage | None = None
    responder: Any | None = None
    confidence_threshold: float = 0.8
    transcript: ScenarioTranscriptRun | None = None
    on_stage: Callable[[str, str], None] | None = None


@dataclass(frozen=True)
class ReviewGraphResult:
    workflow_output: wf.BoundedWorkflowOutput
    graph_trace: GraphTraceV3
    findings: list[FindingV3]
    evidence: EvidenceStageResult | None
    status: Literal["needs_review", "blocked"] | None
    execution_status: Literal["succeeded", "partial", "failed"] | None
    assessment: Literal["no_issue_detected", "issues_detected", "indeterminate"] | None
    confidence: ConfidenceV3 | None
    eligibility_verdict: EligibilityVerdictV3 | None
    actions: list[RequiredActionV1] | None
    report_sections: list[ReportSectionV3] | None


class ReviewGraphState(TypedDict, total=False):
    iteration: int
    initial_analyses: dict[str, DocumentAnalysis]
    analyses: dict[str, DocumentAnalysis]
    reextraction_requests: dict[str, list[str]]
    reextractions: list[ReextractionV3]
    retry_unavailable: bool
    rendered: list[DocumentAnalysisV3]
    comparisons: list[ComparisonV3]
    preliminary_comparisons: list[ComparisonV3]
    findings: list[FindingV3]
    classifications: dict[str, FindingClassificationV3]
    loop_reason: LoopReason | None
    workflow_output: wf.BoundedWorkflowOutput | None
    evidence: EvidenceStageResult | None
    final_findings: list[FindingV3]
    status: Literal["needs_review", "blocked"] | None
    execution_status: Literal["succeeded", "partial", "failed"] | None
    assessment: Literal["no_issue_detected", "issues_detected", "indeterminate"] | None
    confidence: ConfidenceV3 | None
    eligibility_verdict: EligibilityVerdictV3 | None
    actions: list[RequiredActionV1] | None
    report_sections: list[ReportSectionV3] | None


def derive_statuses(
    *, data_status: str, workflow_status: str, has_findings: bool
) -> tuple[
    Literal["needs_review", "blocked"],
    Literal["succeeded", "partial", "failed"],
    Literal["no_issue_detected", "issues_detected", "indeterminate"],
]:
    status: Literal["needs_review", "blocked"] = (
        "blocked" if data_status == "failed" else "needs_review"
    )
    execution_status: Literal["succeeded", "partial", "failed"] = (
        "failed"
        if status == "blocked"
        else (
            "partial" if data_status == "partial" or workflow_status == "partial" else "succeeded"
        )
    )
    assessment: Literal["no_issue_detected", "issues_detected", "indeterminate"] = (
        "issues_detected"
        if has_findings
        else (
            "no_issue_detected"
            if data_status == "ok" and workflow_status == "completed"
            else "indeterminate"
        )
    )
    return status, execution_status, assessment


@dataclass
class _RunContext:
    request: ServiceReviewRunRequestV3
    config: ReviewGraphConfig
    planner: wf._BoundedPlanner = dataclass_field(default_factory=wf._BoundedPlanner)
    node_runs: list[GraphNodeRunV3] = dataclass_field(default_factory=list)

    def notify(self, stage: str, status: str) -> None:
        # Display-only progress; a broken listener must never break the review.
        if self.config.on_stage is None:
            return
        try:
            self.config.on_stage(stage, status)
        except Exception:
            pass

    def record(self, node: str, iteration: int, status: str, summary: str) -> None:
        self.node_runs.append(
            GraphNodeRunV3(
                node=node,  # type: ignore[arg-type]
                iteration=iteration,
                status=status,  # type: ignore[arg-type]
                summary=summary,
            )
        )


def _rule_classification(
    finding: FindingV3, comparison: ComparisonV3 | None
) -> FindingClassificationV3:
    if finding.kind == "regulatory_finding":
        category = "regulatory_requirement"
        explanation = "법령 근거가 요구하는 확인 항목입니다."
    elif finding.code == "document_type_unknown":
        category = "document_type_uncertain"
        explanation = "문서 유형을 원문 인용으로 확정하지 못했습니다."
    elif finding.code in {"missing_required_field", "hs_code_not_found"}:
        category = "missing_field"
        explanation = "비교에 필요한 필드가 문서에서 확인되지 않았습니다."
    elif comparison is None:
        category = "other"
        explanation = "추출 과정에서 기록된 경고입니다."
    elif comparison.status == "mismatch":
        category = "value_mismatch"
        explanation = "정규화한 두 값이 서로 다릅니다."
    elif comparison.unit is None and any(
        not operand.fact_ids for operand in (comparison.left, comparison.right)
    ):
        category = "missing_field"
        explanation = "한쪽 문서의 비교 값이 없어 대조하지 못했습니다."
    else:
        category = "unit_or_format"
        explanation = "단위·통화·표기가 달라 같다고도 다르다고도 단정하지 않았습니다."
    return FindingClassificationV3(
        error_category=category,  # type: ignore[arg-type]
        explanation=explanation,
        source="rule",
    )


def _llm_classifications(
    ctx: _RunContext,
    *,
    rendered: list[DocumentAnalysisV3],
    comparisons: list[ComparisonV3],
    findings: list[FindingV3],
) -> tuple[dict[str, FindingClassificationV3], dict[str, list[str]]]:
    """One optional model call: labels for findings and re-read requests.

    Any malformed part of the answer is dropped; the rule labels remain.
    """

    responder = ctx.config.responder
    if ctx.config.mode != "functional-uat" or responder is None or not findings:
        return {}, {}
    document_ids = {document.document_id for document in ctx.request.documents}
    payload = {
        "trade_type": ctx.request.trade_type,
        "documents": [
            {
                "document_id": analysis.document_id,
                "document_type": analysis.document_type,
                "extraction_status": analysis.extraction_status,
                "warning_codes": [warning.code for warning in analysis.warnings][:16],
            }
            for analysis in rendered
        ],
        "comparisons": [
            {
                "comparison_id": comparison.comparison_id,
                "rule_code": comparison.rule_code,
                "status": comparison.status,
                "expected_value": comparison.expected_value,
                "observed_value": comparison.observed_value,
                "unit": comparison.unit,
                "message": comparison.message[:300],
                "left": {
                    "document_id": comparison.left.document_id,
                    "item_id": comparison.left.item_id,
                    "has_value": bool(comparison.left.fact_ids),
                },
                "right": {
                    "document_id": comparison.right.document_id,
                    "item_id": comparison.right.item_id,
                    "has_value": bool(comparison.right.fact_ids),
                },
            }
            for comparison in comparisons
            if comparison.status in {"mismatch", "needs_review"}
        ][:_MAX_LLM_CONTEXT_ITEMS],
        "findings": [
            {
                "finding_id": finding.finding_id,
                "code": finding.code,
                "severity": finding.severity,
                "message": finding.message[:300],
                "comparison_id": finding.comparison_id,
                "document_ids": finding.document_ids,
            }
            for finding in findings
        ][:_MAX_LLM_CONTEXT_ITEMS],
        "allowed_reread_fields": sorted(_ALLOWED_REREAD_FIELDS),
    }
    try:
        answer = responder.respond(
            payload, transcript=ctx.config.transcript, purpose="finding_classification"
        )
    except Exception:
        return {}, {}
    known = {finding.finding_id for finding in findings}
    labels: dict[str, FindingClassificationV3] = {}
    raw_labels = answer.get("classifications") if isinstance(answer, dict) else None
    if isinstance(raw_labels, list):
        for raw in raw_labels[:_MAX_LLM_CONTEXT_ITEMS]:
            if not isinstance(raw, dict):
                continue
            finding_id = raw.get("finding_id")
            category = raw.get("error_category")
            explanation = raw.get("explanation")
            if (
                finding_id in known
                and category in _ERROR_CATEGORIES
                and isinstance(explanation, str)
                and 1 <= len(explanation.strip()) <= 2000
            ):
                labels[str(finding_id)] = FindingClassificationV3(
                    error_category=category,  # type: ignore[arg-type]
                    explanation=explanation.strip(),
                    source="llm",
                )
    rereads: dict[str, list[str]] = {}
    raw_rereads = answer.get("reread_requests") if isinstance(answer, dict) else None
    if isinstance(raw_rereads, list):
        for raw in raw_rereads[:20]:
            if not isinstance(raw, dict):
                continue
            document_id = raw.get("document_id")
            fields = raw.get("fields")
            if document_id not in document_ids or not isinstance(fields, list):
                continue
            accepted = sorted(
                {
                    str(name)
                    for name in fields
                    if isinstance(name, str) and name in _ALLOWED_REREAD_FIELDS
                }
            )
            if accepted:
                rereads[str(document_id)] = accepted
    return labels, rereads


def _mismatch_reread_targets(
    rendered: list[DocumentAnalysisV3], comparisons: list[ComparisonV3]
) -> dict[str, set[str]]:
    """Fields to re-read because a rule found the documents disagreeing."""

    fact_names: dict[str, str] = {}
    for analysis in rendered:
        for fact in analysis.facts:
            fact_names[fact.fact_id] = fact.name
        for item in analysis.line_items:
            for fact in item.facts:
                fact_names[fact.fact_id] = fact.name
    targets: dict[str, set[str]] = {}
    for comparison in comparisons:
        if comparison.status != "mismatch":
            continue
        for operand in (comparison.left, comparison.right):
            if operand.document_id is None:
                continue
            for fact_id in operand.fact_ids:
                name = fact_names.get(fact_id)
                if name is not None and name in _ALLOWED_REREAD_FIELDS:
                    targets.setdefault(operand.document_id, set()).add(name)
    return targets


def _ungrounded_by_document(
    request: ServiceReviewRunRequestV3, analyses: Mapping[str, DocumentAnalysis]
) -> dict[str, set[str]]:
    return {
        document.document_id: wf._ungrounded_candidate_fields(
            document, analyses[document.document_id].candidate
        )
        for document in request.documents
    }


def _audit(
    ctx: _RunContext,
    *,
    initial: Mapping[str, DocumentAnalysis],
    rendered: list[DocumentAnalysisV3],
    comparisons: list[ComparisonV3],
    llm_rereads: Mapping[str, list[str]],
) -> tuple[dict[str, list[str]], LoopReason | None]:
    """Decide the single allowed re-extraction pass and name its reason."""

    requests: dict[str, set[str]] = {
        document_id: set(fields)
        for document_id, fields in wf._reextraction_targets(ctx.request, initial).items()
    }
    reason: LoopReason | None = None
    if requests:
        ungrounded = _ungrounded_by_document(ctx.request, initial)
        reason = (
            "ungrounded_fields"
            if all(
                fields <= ungrounded.get(document_id, set())
                for document_id, fields in requests.items()
            )
            else "missing_fields"
        )
    if ctx.config.mode != "local-development":
        # The deterministic analyzer cannot answer a re-read differently, so a
        # mismatch-driven audit pass exists only for a model extractor.
        for document_id, mismatch_fields in _mismatch_reread_targets(rendered, comparisons).items():
            requests.setdefault(document_id, set()).update(mismatch_fields)
            reason = reason or "mismatch_reread"
        for document_id, llm_fields in llm_rereads.items():
            requests.setdefault(document_id, set()).update(llm_fields)
            reason = reason or "llm_reread_request"
    return {document_id: sorted(fields) for document_id, fields in requests.items() if fields}, (
        reason if requests else None
    )


def _build_graph(ctx: _RunContext) -> Any:
    request = ctx.request
    config = ctx.config
    analyzer = config.analyzer

    def extraction_agent(state: ReviewGraphState) -> ReviewGraphState:
        iteration = state.get("iteration", 0) + 1
        if iteration == 1:
            ctx.notify("classify_documents", "running")
            ctx.notify("extract_facts", "running")
            initial: dict[str, DocumentAnalysis] = {}
            for document in request.documents:
                initial[document.document_id] = analyzer.analyze(
                    document.model_dump(mode="json", warnings=False), transcript=config.transcript
                )
            ctx.planner.record(
                "analyze_documents",
                status="completed",
                summary="문서 유형과 원문 인용이 있는 사실만 추출·정규화했습니다.",
            )
            ctx.notify("classify_documents", "completed")
            ctx.notify("extract_facts", "completed")
            ctx.notify("normalize_facts", "completed")
            unknown = sum(
                1 for analysis in initial.values() if analysis.candidate.document_type == "unknown"
            )
            ctx.record(
                "extraction_agent",
                1,
                "needs_review" if unknown else "completed",
                f"문서 {len(initial)}건을 분류·추출했습니다"
                + (f" (유형 미확정 {unknown}건)." if unknown else "."),
            )
            return {
                "iteration": 1,
                "initial_analyses": initial,
                "analyses": dict(initial),
                "reextractions": [],
                "retry_unavailable": False,
            }

        requests = state.get("reextraction_requests", {})
        analyses = dict(state["analyses"])
        reextractions: list[ReextractionV3] = []
        retry_unavailable = False
        ctx.planner.start_retry_cycle()
        ctx.notify("reextract_documents", "running")
        for document in request.documents:
            requested_fields = requests.get(document.document_id)
            if not requested_fields:
                continue
            try:
                candidate = analyzer.analyze(
                    document.model_dump(mode="json", warnings=False),
                    transcript=config.transcript,
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
            analyses[document.document_id] = wf._prefer_more_complete(
                analyses[document.document_id], candidate
            )
            reextractions.append(
                ReextractionV3(
                    document_id=document.document_id,
                    requested_fields=requested_fields,
                    status="completed",
                )
            )
        ctx.planner.record(
            "reextract_documents",
            status="needs_review" if retry_unavailable else "completed",
            summary=(
                "각 대상 문서를 최대 한 번 재추출했지만 일부 결과를 사용할 수 없어 사람 검토가 필요합니다."
                if retry_unavailable
                else "누락·모호한 비교 필드를 보완하기 위해 각 대상 문서를 최대 한 번 재추출했습니다."
            ),
        )
        ctx.notify("reextract_documents", "completed")
        ctx.record(
            "extraction_agent",
            2,
            "needs_review" if retry_unavailable else "completed",
            f"감사 요청에 따라 문서 {len(reextractions)}건의 지정 필드를 한 번 더 판독했습니다.",
        )
        return {
            "iteration": 2,
            "analyses": analyses,
            "reextractions": reextractions,
            "retry_unavailable": retry_unavailable,
        }

    def cross_validation_agent(state: ReviewGraphState) -> ReviewGraphState:
        iteration = state["iteration"]
        analyses = state["analyses"]
        if iteration == 2 or not state.get("preliminary_comparisons"):
            ctx.notify("match_documents", "running")
        rendered = [
            wf._render_analysis(document, analyses[document.document_id])
            for document in request.documents
        ]
        if request.human_corrections:
            rendered = wf._apply_human_corrections(rendered, request)
        comparisons = wf._comparisons(rendered, trade_type=request.trade_type)
        findings = wf._findings_from_comparisons(comparisons)
        findings.extend(wf._warning_findings(rendered))
        findings = wf._dedupe_findings(findings)
        comparisons_by_id = {comparison.comparison_id: comparison for comparison in comparisons}
        labels, llm_rereads = _llm_classifications(
            ctx, rendered=rendered, comparisons=comparisons, findings=findings
        )
        classifications = {
            finding.finding_id: labels.get(
                finding.finding_id,
                _rule_classification(finding, comparisons_by_id.get(finding.comparison_id or "")),
            )
            for finding in findings
        }
        update: ReviewGraphState = {
            "rendered": rendered,
            "comparisons": comparisons,
            "findings": findings,
            "classifications": classifications,
        }
        mismatch_count = sum(1 for comparison in comparisons if comparison.status == "mismatch")
        review_count = sum(1 for comparison in comparisons if comparison.status == "needs_review")
        if iteration == 1:
            requests, reason = _audit(
                ctx,
                initial=state["initial_analyses"],
                rendered=rendered,
                comparisons=comparisons,
                llm_rereads=llm_rereads,
            )
            update["reextraction_requests"] = requests
            update["loop_reason"] = reason
            update["preliminary_comparisons"] = comparisons
            if requests:
                ctx.record(
                    "cross_validation_agent",
                    1,
                    "needs_review",
                    f"예비 비교: 불일치 {mismatch_count}건·확인 필요 {review_count}건, "
                    f"문서 {len(requests)}건에 재판독을 요청했습니다.",
                )
                return update
        if request.human_corrections:
            ctx.planner.record(
                "apply_human_corrections",
                status="completed",
                summary="승인된 사람 수정은 원문 사실을 보존한 별도 human-provided 사실로 반영했습니다.",
                correction_proposal_ids=request.applied_correction_proposal_ids,
            )
        ctx.planner.record(
            "compare_documents",
            status="completed",
            summary="코드 기반 수량·금액·원산지·B/L 비교 규칙을 실행했습니다.",
        )
        ctx.notify("match_documents", "completed")
        ctx.notify("validate_documents", "completed")
        ctx.record(
            "cross_validation_agent",
            iteration,
            "needs_review" if mismatch_count or review_count else "completed",
            f"코드 규칙 비교 {len(comparisons)}건: 불일치 {mismatch_count}건, "
            f"확인 필요 {review_count}건, 확인 사항 {len(findings)}건.",
        )
        reextractions = state.get("reextractions", [])
        retry_unavailable = state.get("retry_unavailable", False)
        (
            retrieval_items,
            retrieval_document_ids,
            retrieval_excluded_document_ids,
            retrieval_skip_reason,
            retrieval_reviewer_hs_code,
        ) = wf.retrieval_plan_from_analyses(request, rendered)
        stages = wf._workflow_stages(
            rendered, comparisons, reextractions, retry_unavailable=retry_unavailable
        )
        workflow_status: Literal["completed", "partial", "failed"] = (
            "partial" if retry_unavailable else "completed"
        )
        update["workflow_output"] = wf.BoundedWorkflowOutput(
            document_analyses=rendered,
            comparisons=comparisons,
            findings=findings,
            workflow=WorkflowV3(
                status=workflow_status,
                stages=stages,
                reextractions=reextractions,
                planner=ctx.planner.trace(),
            ),
            retrieval_items=retrieval_items,
            retrieval_document_ids=retrieval_document_ids,
            retrieval_excluded_document_ids=retrieval_excluded_document_ids,
            retrieval_skip_reason=retrieval_skip_reason,
            retrieval_reviewer_hs_code=retrieval_reviewer_hs_code,
            planner_state=ctx.planner,
        )
        return update

    def route_after_cross_validation(state: ReviewGraphState) -> str:
        if state["iteration"] == 1 and state.get("reextraction_requests"):
            return "extraction_agent"
        if config.evidence_stage is None:
            return END
        return "evidence_retrieval"

    def evidence_retrieval(state: ReviewGraphState) -> ReviewGraphState:
        assert config.evidence_stage is not None
        workflow_output = state["workflow_output"]
        assert workflow_output is not None
        if workflow_output.retrieval_items:
            ctx.notify("retrieve_evidence", "running")
        evidence = config.evidence_stage(request, workflow_output)
        if evidence.skipped:
            workflow_output = workflow_output.with_evidence_skipped()
            ctx.record(
                "evidence_retrieval",
                state["iteration"],
                "skipped",
                "HS 코드가 확인된 품목이 없어 법령 근거 조회를 실행하지 않았습니다.",
            )
        else:
            workflow_output = workflow_output.with_evidence_status(evidence.data_status)
            ctx.notify("retrieve_evidence", "completed")
            ctx.record(
                "evidence_retrieval",
                state["iteration"],
                {"ok": "completed", "partial": "needs_review", "failed": "failed"}[
                    evidence.data_status
                ],
                f"Data Evidence 응답 {evidence.data_status}: 근거 {len(evidence.public_evidence)}건, "
                f"요건 확인 사항 {len(evidence.evidence_findings)}건.",
            )
        return {"workflow_output": workflow_output, "evidence": evidence}

    def expert_review_agent(state: ReviewGraphState) -> ReviewGraphState:
        workflow_output = state["workflow_output"]
        evidence = state["evidence"]
        assert workflow_output is not None and evidence is not None
        findings = wf._dedupe_findings([*workflow_output.findings, *evidence.evidence_findings])
        status, execution_status, assessment = derive_statuses(
            data_status=evidence.data_status,
            workflow_status=workflow_output.workflow.status,
            has_findings=bool(findings),
        )
        comparisons_by_id = {
            comparison.comparison_id: comparison for comparison in workflow_output.comparisons
        }
        reextracted = frozenset(
            entry.document_id
            for entry in workflow_output.workflow.reextractions
            if entry.status == "completed"
        )
        unavailable = frozenset(
            entry.document_id
            for entry in workflow_output.workflow.reextractions
            if entry.status == "unavailable"
        )
        preliminary = {
            comparison_key(comparison): comparison.status
            for comparison in state.get("preliminary_comparisons", [])
        }
        findings = attach_finding_confidence(
            findings,
            comparisons_by_id=comparisons_by_id,
            preliminary_status_by_key=preliminary,
            reextracted_document_ids=reextracted,
            unavailable_document_ids=unavailable,
            data_status=evidence.data_status,
        )
        classifications = state.get("classifications", {})
        findings = [
            finding.model_copy(
                update={
                    "classification": classifications.get(
                        finding.finding_id,
                        _rule_classification(
                            finding, comparisons_by_id.get(finding.comparison_id or "")
                        ),
                    )
                }
            )
            for finding in findings
        ]
        confidence = review_confidence(
            analyses=workflow_output.document_analyses,
            comparisons=workflow_output.comparisons,
            data_status=evidence.data_status,
            retrieval_skipped=evidence.skipped,
            threshold=config.confidence_threshold,
        )
        verdict = rule_verdict(
            findings=findings,
            comparisons_by_id=comparisons_by_id,
            execution_status=execution_status,
            data_status=evidence.data_status,
            retrieval_skipped=evidence.skipped,
            review_track=confidence.review_track,
        )
        verdict = _llm_expert_review(
            ctx,
            rule=verdict,
            findings=findings,
            evidence=evidence,
            confidence=confidence,
            assessment=assessment,
        )
        ctx.record(
            "expert_review_agent",
            state["iteration"],
            "completed",
            f"신뢰도 {confidence.overall:.2f} ({confidence.review_track}), "
            f"권고 판정 {verdict.verdict} ({verdict.source}). 최종 판단은 담당자가 합니다.",
        )
        return {
            "final_findings": findings,
            "status": status,
            "execution_status": execution_status,
            "assessment": assessment,
            "confidence": confidence,
            "eligibility_verdict": verdict,
        }

    def report_generation(state: ReviewGraphState) -> ReviewGraphState:
        workflow_output = state["workflow_output"]
        evidence = state["evidence"]
        assert workflow_output is not None and evidence is not None
        findings = state["final_findings"]
        actions = wf.actions_for_findings(findings)
        ctx.notify("generate_result", "running")
        sections = build_grounded_report_sections(
            analyses=workflow_output.document_analyses,
            comparisons=workflow_output.comparisons,
            evidence=evidence.public_evidence,
            findings=findings,
            actions=actions,
            evidence_retrieval_skip_reason=workflow_output.retrieval_skip_reason,
        )
        workflow_output = workflow_output.with_report_generated()
        ctx.record(
            "report_generation",
            state["iteration"],
            "completed",
            "근거가 연결된 8구역 검토 보고서를 생성했습니다.",
        )
        return {
            "workflow_output": workflow_output,
            "actions": actions,
            "report_sections": sections,
        }

    graph: StateGraph[ReviewGraphState] = StateGraph(ReviewGraphState)
    graph.add_node("extraction_agent", extraction_agent)
    graph.add_node("cross_validation_agent", cross_validation_agent)
    graph.add_node("evidence_retrieval", evidence_retrieval)
    graph.add_node("expert_review_agent", expert_review_agent)
    graph.add_node("report_generation", report_generation)
    graph.add_edge(START, "extraction_agent")
    graph.add_edge("extraction_agent", "cross_validation_agent")
    graph.add_conditional_edges(
        "cross_validation_agent",
        route_after_cross_validation,
        {
            "extraction_agent": "extraction_agent",
            "evidence_retrieval": "evidence_retrieval",
            END: END,
        },
    )
    graph.add_edge("evidence_retrieval", "expert_review_agent")
    graph.add_edge("expert_review_agent", "report_generation")
    graph.add_edge("report_generation", END)
    return graph.compile()


def _llm_expert_review(
    ctx: _RunContext,
    *,
    rule: EligibilityVerdictV3,
    findings: list[FindingV3],
    evidence: EvidenceStageResult,
    confidence: ConfidenceV3,
    assessment: str,
) -> EligibilityVerdictV3:
    responder = ctx.config.responder
    if ctx.config.mode != "functional-uat" or responder is None:
        return rule
    payload = {
        "trade_type": ctx.request.trade_type,
        "assessment": assessment,
        "confidence": confidence.model_dump(mode="json"),
        "rule_verdict": rule.model_dump(mode="json"),
        "findings": [
            {
                "finding_id": finding.finding_id,
                "kind": finding.kind,
                "code": finding.code,
                "severity": finding.severity,
                "message": finding.message[:300],
                "confidence": finding.confidence,
                "error_category": finding.classification.error_category
                if finding.classification
                else None,
                "evidence_ids": finding.evidence_ids[:8],
            }
            for finding in findings
        ][:_MAX_LLM_CONTEXT_ITEMS],
        "evidence": [
            {
                "evidence_id": record["evidence_id"],
                "title": str(record.get("title", ""))[:200],
                "law_name": str(record.get("law_name", ""))[:100],
                "eligibility": record.get("eligibility"),
                "text": str(record.get("text", ""))[:300],
            }
            for record in evidence.evidence_records
        ][:_MAX_LLM_CONTEXT_ITEMS],
    }
    try:
        answer = responder.respond(
            payload, transcript=ctx.config.transcript, purpose="expert_review"
        )
    except Exception:
        return rule
    if not isinstance(answer, dict):
        return rule
    return merge_llm_verdict(
        rule,
        answer,
        known_finding_ids=frozenset(finding.finding_id for finding in findings),
        known_evidence_ids=frozenset(
            str(record["evidence_id"]) for record in evidence.evidence_records
        ),
    )


def run_review_graph(
    request: ServiceReviewRunRequestV3, config: ReviewGraphConfig
) -> ReviewGraphResult:
    """Run the compiled graph once and return everything the profile needs."""

    ctx = _RunContext(request=request, config=config)
    compiled = _build_graph(ctx)
    try:
        final: ReviewGraphState = compiled.invoke({}, config={"recursion_limit": 16})
    except RuntimeExecutionError:
        raise
    except Exception as exc:
        # Provider, retrieval, and input errors carry their own codes and are
        # re-raised unchanged by LangGraph; anything else is a graph failure.
        if exc.__class__.__module__.startswith("langgraph"):
            raise RuntimeExecutionError(
                "profile_review_execution_failed", "The review graph could not be completed."
            ) from None
        raise
    workflow_output = final.get("workflow_output")
    if workflow_output is None:
        raise RuntimeExecutionError(
            "profile_review_execution_failed", "The review graph produced no workflow output."
        )
    graph_trace = GraphTraceV3(
        iteration_count=final["iteration"],
        loop_reason=final.get("loop_reason") if final["iteration"] == 2 else None,
        node_runs=list(ctx.node_runs),
    )
    workflow_output = workflow_output.with_graph_trace(graph_trace)
    return ReviewGraphResult(
        workflow_output=workflow_output,
        graph_trace=graph_trace,
        findings=final.get("final_findings", workflow_output.findings),
        evidence=final.get("evidence"),
        status=final.get("status"),
        execution_status=final.get("execution_status"),
        assessment=final.get("assessment"),
        confidence=final.get("confidence"),
        eligibility_verdict=final.get("eligibility_verdict"),
        actions=final.get("actions"),
        report_sections=final.get("report_sections"),
    )


__all__ = [
    "EvidenceStage",
    "EvidenceStageResult",
    "ReviewGraphConfig",
    "ReviewGraphResult",
    "derive_statuses",
    "orchestrator_version",
    "run_review_graph",
]
