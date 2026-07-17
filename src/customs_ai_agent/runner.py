from __future__ import annotations

import copy
from os import PathLike
from typing import Any

from customs_ai_agent.adapters.llm import (
    DocumentAnalyzer,
    LLMConfigurationError,
    build_document_analyzer,
)
from customs_ai_agent.adapters.retrieval import (
    FailingRegulationRetriever,
    MockRegulationRetriever,
    RegulationRetriever,
)
from customs_ai_agent.fixtures.scenarios import get_scenario
from customs_ai_agent.graph import build_review_graph
from customs_ai_agent.nodes.common import audit
from customs_ai_agent.nodes.compliance import compliance_review
from customs_ai_agent.nodes.normalize import normalize_documents
from customs_ai_agent.nodes.pair import pair_documents
from customs_ai_agent.nodes.report import generate_xai_report
from customs_ai_agent.nodes.retrieve import retrieve_regulations
from customs_ai_agent.nodes.route import route_to_human_or_complete
from customs_ai_agent.nodes.validate import validate_cross_documents
from customs_ai_agent.schemas.models import (
    ChatAnswer,
    ReviewRunRequest,
    ReviewRunResult,
    UserCorrection,
)


def _build_initial_state(request: ReviewRunRequest) -> dict[str, Any]:
    if request.scenario_id:
        scenario = get_scenario(request.scenario_id)
        case_id = request.case_id or scenario["case_id"]
        documents = scenario["documents"]
    else:
        case_id = request.case_id or "case_manual"
        documents = [doc.model_dump(mode="json") for doc in request.documents or []]
    uploaded_codes = set(request.uploaded_requirement_document_codes)
    for document in documents:
        uploaded_codes.update(document.get("requirement_document_codes") or [])
    return {
        "case_id": case_id,
        "review_run_id": request.review_run_id,
        "review_as_of": request.review_as_of.isoformat(),
        "trade_type": request.trade_type,
        "jurisdiction": request.jurisdiction,
        "uploaded_requirement_document_codes": sorted(uploaded_codes),
        "documents": documents,
        "document_ids": [],
        "classified_documents": {},
        "model_analyses": {},
        "model_calls": [],
        "model_errors": [],
        "extracted_documents": {},
        "normalized_documents": {},
        "pairing_result": {},
        "validation_issues": [],
        "rag_evidence": [],
        "retrieval_request_id": "",
        "retrieval_request": {},
        "retrieval_status": "failed",
        "retrieval_coverage": "unknown",
        "knowledge_snapshot_id": None,
        "retrieval_results": [],
        "retrieval_errors": [],
        "regulatory_findings": [],
        "report_json": None,
        "retry_count": 0,
        "current_stage": "created",
        "requires_human_review": False,
        "final_status": None,
        "audit_trail": [],
        "llm_mode": request.llm_mode,
    }


def run_review(
    request: ReviewRunRequest | dict[str, Any],
    retriever: RegulationRetriever | None = None,
    analyzer: DocumentAnalyzer | None = None,
    env_file: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    if isinstance(request, dict):
        request = ReviewRunRequest.model_validate(request)
    if request.llm_mode == "llm":
        if retriever is None:
            raise LLMConfigurationError(
                "explicit_retriever_required",
                "llm mode requires an explicit RegulationRetriever; implicit Mock RAG is forbidden.",
            )
        active_retriever = retriever
        active_analyzer = analyzer or build_document_analyzer(env_file)
    else:
        if analyzer is not None:
            raise LLMConfigurationError(
                "analyzer_not_allowed_in_mock_mode",
                "A document analyzer cannot be injected into mock mode.",
            )
        active_retriever = retriever or MockRegulationRetriever()
        active_analyzer = None
    initial_state = _build_initial_state(request)
    graph = build_review_graph(active_retriever, active_analyzer)
    return graph.invoke(initial_state, config={"recursion_limit": 20})


def to_review_run_result(state: dict[str, Any]) -> ReviewRunResult:
    if not state.get("report_json"):
        raise ValueError("review state does not contain a completed report")
    return ReviewRunResult(
        case_id=state["case_id"],
        review_run_id=state["review_run_id"],
        review_as_of=state["review_as_of"],
        trade_type=state["trade_type"],
        jurisdiction=state["jurisdiction"],
        status=state["final_status"],
        requires_human_review=state["requires_human_review"],
        retrieval_status=state["retrieval_status"],
        retrieval_coverage=state["retrieval_coverage"],
        knowledge_snapshot_id=state.get("knowledge_snapshot_id"),
        retrieval_request=state.get("retrieval_request") or None,
        retrieval_results=state.get("retrieval_results") or [],
        retrieval_errors=state.get("retrieval_errors") or [],
        validation_issues=state.get("validation_issues") or [],
        regulatory_findings=state.get("regulatory_findings") or [],
        rag_evidence=state.get("rag_evidence") or [],
        report=state["report_json"],
        audit_trail=state.get("audit_trail") or [],
        model_calls=state.get("model_calls") or [],
        model_errors=state.get("model_errors") or [],
    )


def run_review_result(
    request: ReviewRunRequest | dict[str, Any],
    retriever: RegulationRetriever | None = None,
    analyzer: DocumentAnalyzer | None = None,
    env_file: str | PathLike[str] | None = None,
) -> ReviewRunResult:
    return to_review_run_result(
        run_review(
            request,
            retriever=retriever,
            analyzer=analyzer,
            env_file=env_file,
        )
    )


def _apply_correction(state: dict[str, Any], correction: UserCorrection) -> None:
    doc = state.get("extracted_documents", {}).get(correction.document_id)
    if not doc:
        return
    parts = correction.field_path.split(".")
    if parts[0] == "fields" and len(parts) == 2:
        field_data = doc.get("fields", {}).get(parts[1])
        if field_data:
            field_data["value"] = correction.value
            field_data["confidence"] = correction.confidence
            field_data["value_source"] = "human_correction"
    elif parts[0] == "line_items" and len(parts) == 3:
        try:
            index = int(parts[1])
        except ValueError:
            return
        line_items = doc.get("line_items") or []
        if 0 <= index < len(line_items):
            field_data = line_items[index].get(parts[2])
            if field_data:
                field_data["value"] = correction.value
                field_data["confidence"] = correction.confidence
                field_data["value_source"] = "human_correction"


def _merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    state.update(update)
    return state


def rerun_review(
    state: dict[str, Any],
    corrections: list[UserCorrection | dict[str, Any]],
    retriever: RegulationRetriever | None = None,
) -> dict[str, Any]:
    if retriever is not None:
        active_retriever = retriever
    elif state.get("knowledge_snapshot_id") == "mock-knowledge-v1":
        active_retriever = MockRegulationRetriever()
    else:
        # Never silently replace a real/failed Evidence Service with fixture data.
        active_retriever = FailingRegulationRetriever()
    next_state = copy.deepcopy(state)
    parsed = [
        correction if isinstance(correction, UserCorrection) else UserCorrection.model_validate(correction)
        for correction in corrections
    ]
    for correction in parsed:
        _apply_correction(next_state, correction)
    next_state["requires_human_review"] = False
    next_state["final_status"] = None
    next_state["audit_trail"] = audit(
        next_state,
        "rerun_review",
        "corrections_applied",
        "Applied user corrections and reran deterministic review stages.",
        {"correction_count": len(parsed)},
    )
    for node in [
        normalize_documents,
        pair_documents,
        validate_cross_documents,
        compliance_review,
        generate_xai_report,
        route_to_human_or_complete,
    ]:
        next_state = _merge(next_state, node(next_state))
        if node is validate_cross_documents:
            next_state = _merge(
                next_state,
                retrieve_regulations(next_state, active_retriever),
            )
    return next_state


def answer_question(report_or_state: dict[str, Any], question: str) -> ChatAnswer:
    report = report_or_state.get("report_json") or report_or_state
    lowered = question.casefold()
    if any(word in lowered for word in ["weather", "stock", "price", "news"]):
        return ChatAnswer(
            answer="The provided case report and evidence do not contain enough information to answer that question.",
            evidence_ids=[],
            grounded=False,
        )

    issues = report.get("issues") or []
    findings = report.get("regulatory_findings") or []
    evidence = report.get("rag_evidence") or []
    if not issues and not findings and not evidence:
        return ChatAnswer(
            answer="No case issue or regulatory evidence is available in the provided report.",
            evidence_ids=[],
            grounded=False,
        )

    evidence_ids: list[str] = []
    lines: list[str] = []
    if issues:
        top = issues[0]
        lines.append(
            f"The case needs attention because {top['issue_code']} was detected: {top['message']}"
        )
    if findings:
        finding = findings[0]
        lines.append(
            f"A regulatory finding was generated: {finding['finding_code']} ({finding['message']})."
        )
        evidence_ids.extend(finding.get("evidence_ids") or [])
    elif evidence and any(word in lowered for word in ["evidence", "regulation", "law", "rag"]):
        evidence_ids.extend(item["evidence_id"] for item in evidence[:2])
        lines.append("The answer is limited to the evidence packages included in this report.")

    return ChatAnswer(
        answer=" ".join(lines),
        evidence_ids=sorted(set(evidence_ids)),
        grounded=True,
    )
