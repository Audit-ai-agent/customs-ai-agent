from __future__ import annotations

from datetime import date
from typing import Any

from customs_ai_agent.adapters.retrieval import RegulationRetriever
from customs_ai_agent.identifiers import stable_retrieval_request_id
from customs_ai_agent.nodes.common import audit
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RetrievalErrorDetail,
    RetrievalQueryItem,
)


def _invoice_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    for doc in (state.get("normalized_documents") or {}).values():
        if doc.get("document_type") == "invoice":
            return doc.get("line_items") or []
    return []


def _query_item(item: dict[str, Any], destination_country: str) -> RetrievalQueryItem:
    origin_country = item.get("origin_country")
    if not (
        isinstance(origin_country, str)
        and len(origin_country) == 2
        and origin_country.isalpha()
    ):
        origin_country = None
    return RetrievalQueryItem(
        item_id=item["item_id"],
        description=str(item.get("description") or "Unspecified invoice item")[:2000],
        hs_code=item.get("hs_code"),
        origin_country=origin_country,
        destination_country=destination_country,
    )


def _coverage(results: list[dict[str, Any]]) -> str:
    coverages = {result.get("coverage", "unknown") for result in results}
    if coverages == {"complete"}:
        return "complete"
    if "unknown" in coverages or not coverages:
        return "unknown"
    return "partial"


def _context_errors(
    state: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    review_as_of = date.fromisoformat(state["review_as_of"])
    for result in results:
        resolved_hsk10 = result.get("resolved_hsk10")
        for evidence in result.get("evidence") or []:
            code = None
            message = None
            if evidence.get("trade_type") != state["trade_type"]:
                code = "EVIDENCE_TRADE_TYPE_MISMATCH"
                message = "Evidence trade_type did not match the review request."
            elif (
                evidence.get("source_type") == "customs_requirement"
                and evidence.get("matched_hs_code") != resolved_hsk10
            ):
                code = "EVIDENCE_HS_MISMATCH"
                message = "Customs requirement evidence did not match the resolved HSK10."
            else:
                valid_from = evidence.get("valid_from")
                valid_to = evidence.get("valid_to")
                if valid_from and date.fromisoformat(valid_from) > review_as_of:
                    code = "EVIDENCE_NOT_YET_VALID"
                    message = "Evidence was not valid on review_as_of."
                # The Evidence API exposes the last applicable calendar date.
                # Source-specific half-open intervals must be normalized by the
                # Evidence Service before crossing this contract boundary.
                elif valid_to and date.fromisoformat(valid_to) < review_as_of:
                    code = "EVIDENCE_EXPIRED"
                    message = "Evidence had expired on review_as_of."
            if code and message:
                errors.append(
                    RetrievalErrorDetail(
                        code=code,
                        message=f"{result['item_id']}/{evidence['evidence_id']}: {message}",
                        retryable=False,
                    ).model_dump(mode="json")
                )
    return errors


def _failed_update(
    state: dict[str, Any],
    *,
    request_id: str,
    code: str,
    message: str,
    request: EvidenceRetrievalRequest | None = None,
) -> dict[str, Any]:
    error = RetrievalErrorDetail(code=code, message=message, retryable=True).model_dump(
        mode="json"
    )
    return {
        "retrieval_request_id": request_id,
        "retrieval_request": request.model_dump(mode="json") if request else {},
        "retrieval_status": "failed",
        "retrieval_coverage": "unknown",
        "knowledge_snapshot_id": None,
        "retrieval_results": [],
        "retrieval_errors": [error],
        "rag_evidence": [],
        "requires_human_review": True,
        "current_stage": "retrieve_regulations",
        "audit_trail": audit(
            state,
            "retrieve_regulations",
            "failed",
            message,
            {"request_id": request_id, "error_code": code},
        ),
    }


def retrieve_regulations(
    state: dict[str, Any],
    retriever: RegulationRetriever,
) -> dict[str, Any]:
    request_id = stable_retrieval_request_id(
        str(state["review_run_id"]),
        int(state.get("retry_count") or 0),
    )
    try:
        query_items = [
            _query_item(item, state["jurisdiction"]) for item in _invoice_items(state)
        ]
    except Exception:
        return _failed_update(
            state,
            request_id=request_id,
            code="INVALID_RETRIEVAL_QUERY",
            message="Could not build a contract-valid retrieval query.",
        )
    if not query_items:
        return _failed_update(
            state,
            request_id=request_id,
            code="NO_RETRIEVAL_ITEMS",
            message="No invoice line items were available for regulatory retrieval.",
        )

    try:
        request = EvidenceRetrievalRequest(
            request_id=request_id,
            case_id=state["case_id"],
            review_run_id=state["review_run_id"],
            review_as_of=state["review_as_of"],
            jurisdiction=state["jurisdiction"],
            trade_type=state["trade_type"],
            top_k=8,
            items=query_items,
            uploaded_requirement_document_codes=(
                state.get("uploaded_requirement_document_codes") or []
            ),
            validation_issue_codes=sorted(
                {
                    issue["issue_code"]
                    for issue in state.get("validation_issues") or []
                    if issue.get("issue_code")
                }
            ),
            # Request structured requirements plus supporting law articles.
            # Supporting semantic hits are never finding-eligible by themselves.
            query_intent=f"required_{state['trade_type']}_documents_and_law_support",
        )
    except Exception:
        return _failed_update(
            state,
            request_id=request_id,
            code="INVALID_RETRIEVAL_REQUEST",
            message="Could not build a contract-valid retrieval request.",
        )
    try:
        raw_response = retriever.retrieve(request)
        response = (
            raw_response
            if isinstance(raw_response, EvidenceRetrievalResponse)
            else EvidenceRetrievalResponse.model_validate(raw_response)
        )
    except Exception:
        return _failed_update(
            state,
            request_id=request_id,
            code="RETRIEVER_EXCEPTION",
            message="Regulatory retrieval failed.",
            request=request,
        )

    if response.request_id != request_id:
        return _failed_update(
            state,
            request_id=request_id,
            code="REQUEST_ID_MISMATCH",
            message="Evidence response request_id did not match the request.",
            request=request,
        )

    requested_ids = {item.item_id for item in query_items}
    returned_ids = {result.item_id for result in response.results}
    if requested_ids != returned_ids:
        return _failed_update(
            state,
            request_id=request_id,
            code="ITEM_RESULT_MISMATCH",
            message="Evidence response did not contain exactly one result for every requested item.",
            request=request,
        )

    results = [result.model_dump(mode="json") for result in response.results]
    evidence = [item for result in results for item in result["evidence"]]
    response_errors = [error.model_dump(mode="json") for error in response.errors]
    context_errors = _context_errors(state, results)
    effective_status = "failed" if context_errors else response.status
    retrieval_coverage = "unknown" if context_errors else _coverage(results)
    fail_closed = (
        effective_status != "ok"
        or retrieval_coverage != "complete"
        or not response.knowledge_snapshot_id
        or any(result["errors"] for result in results)
        or any(
            result["hs_resolution"] not in {"exact", "unique_descendant"}
            for result in results
        )
        or any(
            result["requirements_found"]
            and not any(
                candidate.get("eligibility") == "finding"
                and candidate.get("required_document_code")
                and candidate.get("required_document_name")
                for candidate in result["evidence"]
            )
            for result in results
        )
    )
    return {
        "retrieval_request_id": request_id,
        "retrieval_request": request.model_dump(mode="json"),
        "retrieval_status": effective_status,
        "retrieval_coverage": retrieval_coverage,
        "knowledge_snapshot_id": response.knowledge_snapshot_id,
        "retrieval_results": results,
        "retrieval_errors": [*response_errors, *context_errors],
        "rag_evidence": [] if context_errors else evidence,
        "requires_human_review": state.get("requires_human_review", False) or fail_closed,
        "current_stage": "retrieve_regulations",
        "audit_trail": audit(
            state,
            "retrieve_regulations",
            "completed" if effective_status == "ok" else effective_status,
            "Retrieved item-bound regulatory evidence.",
            {
                "request_id": request_id,
                "status": effective_status,
                "coverage": retrieval_coverage,
                "result_count": len(results),
                "evidence_count": len(evidence),
                "knowledge_snapshot_id": response.knowledge_snapshot_id,
                "fail_closed": fail_closed,
                "context_error_count": len(context_errors),
            },
        ),
    }
