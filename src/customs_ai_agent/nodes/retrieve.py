from __future__ import annotations

from datetime import date
from typing import Any

from customs_ai_agent.adapters.retrieval import RegulationRetriever
from customs_ai_agent.identifiers import stable_retrieval_request_id
from customs_ai_agent.nodes.common import audit
from customs_ai_agent.nodes.normalize import is_valid_hs_code
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RagEvidence,
    RetrievalErrorDetail,
    RetrievalItemResult,
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
        isinstance(origin_country, str) and len(origin_country) == 2 and origin_country.isalpha()
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


def _is_ascii_hs_code(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all("0" <= character <= "9" for character in value)
    )


def _hs_result_matches_query(
    query: RetrievalQueryItem,
    result: RetrievalItemResult,
) -> bool:
    """Bind a successful Evidence resolution to the exact requested HS scope.

    `model_construct` can bypass every Pydantic validator, so this boundary
    deliberately rechecks primitive types, status values and candidate formats.
    A 10-digit exact result must equal the requested HSK10. A unique descendant
    is valid only for a requested 6-digit HS code and must retain that prefix.
    """

    resolution = getattr(result, "hs_resolution", None)
    resolved_hsk10 = getattr(result, "resolved_hsk10", None)
    candidate_hsk10s = getattr(result, "candidate_hsk10s", None)
    if (
        not isinstance(resolution, str)
        or resolution
        not in {
            "exact",
            "unique_descendant",
            "ambiguous",
            "invalid",
            "missing",
        }
        or not isinstance(candidate_hsk10s, list)
        or any(not _is_ascii_hs_code(code, length=10) for code in candidate_hsk10s)
        or len(candidate_hsk10s) != len(set(candidate_hsk10s))
    ):
        return False

    requested_hs = query.hs_code
    if resolution == "exact":
        if not isinstance(requested_hs, str) or not isinstance(resolved_hsk10, str):
            return False
        return (
            _is_ascii_hs_code(requested_hs, length=10)
            and _is_ascii_hs_code(resolved_hsk10, length=10)
            and resolved_hsk10 == requested_hs
            and (not candidate_hsk10s or candidate_hsk10s == [resolved_hsk10])
        )
    if resolution == "unique_descendant":
        if not isinstance(requested_hs, str) or not isinstance(resolved_hsk10, str):
            return False
        return (
            _is_ascii_hs_code(requested_hs, length=6)
            and _is_ascii_hs_code(resolved_hsk10, length=10)
            and resolved_hsk10.startswith(requested_hs)
            and (not candidate_hsk10s or candidate_hsk10s == [resolved_hsk10])
        )
    return resolved_hsk10 is None


def _response_envelope_is_well_formed(response: EvidenceRetrievalResponse) -> bool:
    snapshot_id = getattr(response, "knowledge_snapshot_id", None)
    status = getattr(response, "status", None)
    return (
        getattr(response, "schema_version", None) == "1.0"
        and isinstance(getattr(response, "request_id", None), str)
        and bool(response.request_id.strip())
        and isinstance(status, str)
        and status in {"ok", "partial", "failed"}
        and (snapshot_id is None or (isinstance(snapshot_id, str) and bool(snapshot_id.strip())))
        and isinstance(getattr(response, "results", None), list)
        and isinstance(getattr(response, "errors", None), list)
    )


def _result_envelope_is_well_formed(result: object) -> bool:
    coverage = getattr(result, "coverage", None)
    return (
        isinstance(result, RetrievalItemResult)
        and isinstance(getattr(result, "item_id", None), str)
        and bool(result.item_id.strip())
        and isinstance(coverage, str)
        and coverage in {"complete", "partial", "unknown"}
        and isinstance(getattr(result, "requirements_found", None), bool)
        and isinstance(getattr(result, "evidence", None), list)
        and isinstance(getattr(result, "errors", None), list)
    )


def _is_qualified_requirement_finding(
    evidence: dict[str, Any],
    *,
    item_id: str,
    resolved_hsk10: str | None,
) -> bool:
    try:
        parsed = RagEvidence.model_validate(evidence)
    except Exception:
        return False
    return (
        parsed.source_type == "customs_requirement"
        and parsed.eligibility == "finding"
        and parsed.item_id == item_id
        and parsed.matched_hs_code == resolved_hsk10
    )


def _context_errors(state: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    review_as_of = date.fromisoformat(state["review_as_of"])
    for result in results:
        resolved_hsk10 = result.get("resolved_hsk10")
        for evidence in result.get("evidence") or []:
            code = None
            message = None
            if (
                evidence.get("eligibility") == "finding"
                and evidence.get("source_type") != "customs_requirement"
            ):
                code = "EVIDENCE_FINDING_SOURCE_NOT_STRUCTURED"
                message = "Only structured customs requirement evidence can support a finding."
            elif evidence.get("trade_type") != state["trade_type"]:
                code = "EVIDENCE_TRADE_TYPE_MISMATCH"
                message = "Evidence trade_type did not match the review request."
            elif (
                evidence.get("source_type") == "customs_requirement"
                and evidence.get("matched_hs_code") != resolved_hsk10
            ):
                code = "EVIDENCE_HS_MISMATCH"
                message = "Customs requirement evidence did not match the resolved HSK10."
            elif evidence.get("eligibility") == "finding" and not (
                _is_qualified_requirement_finding(
                    evidence,
                    item_id=result["item_id"],
                    resolved_hsk10=resolved_hsk10,
                )
            ):
                code = "EVIDENCE_FINDING_PROVENANCE_INCOMPLETE"
                message = "Finding evidence did not contain complete structured provenance."
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
        qualified_requirement_found = any(
            _is_qualified_requirement_finding(
                evidence,
                item_id=result["item_id"],
                resolved_hsk10=resolved_hsk10,
            )
            for evidence in result.get("evidence") or []
        )
        if bool(result.get("requirements_found")) != qualified_requirement_found:
            errors.append(
                RetrievalErrorDetail(
                    code="EVIDENCE_REQUIREMENTS_FOUND_MISMATCH",
                    message=(
                        f"{result['item_id']}: requirements_found did not match "
                        "qualified structured requirement evidence."
                    ),
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
    retryable: bool = True,
) -> dict[str, Any]:
    error = RetrievalErrorDetail(
        code=code,
        message=message,
        retryable=retryable,
    ).model_dump(mode="json")
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
    invoice_items = _invoice_items(state)
    invalid_hs_item_ids = sorted(
        str(item.get("item_id") or "unknown")
        for item in invoice_items
        if (
            (
                item.get("hs_code") is not None
                or (
                    (item.get("raw_fields") or {}).get("hs_code") is not None
                    and str((item.get("raw_fields") or {}).get("hs_code")).strip()
                )
            )
            and not is_valid_hs_code(item.get("hs_code"))
        )
    )
    if invalid_hs_item_ids:
        return _failed_update(
            state,
            request_id=request_id,
            code="INVALID_HS_CODE_RETRIEVAL_BLOCKED",
            message=(
                "Regulatory retrieval was not called because invoice items have "
                f"invalid HS codes: {', '.join(invalid_hs_item_ids)}."
            ),
            retryable=False,
        )
    try:
        query_items = [_query_item(item, state["jurisdiction"]) for item in invoice_items]
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

    if not _response_envelope_is_well_formed(response) or any(
        not _result_envelope_is_well_formed(result) for result in response.results
    ):
        return _failed_update(
            state,
            request_id=request_id,
            code="INVALID_RETRIEVAL_RESPONSE",
            message="Evidence response failed the local response envelope.",
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

    query_by_item_id = {item.item_id: item for item in query_items}
    if any(
        not _hs_result_matches_query(query_by_item_id[result.item_id], result)
        for result in response.results
    ):
        return _failed_update(
            state,
            request_id=request_id,
            code="HS_RESULT_MISMATCH",
            message="Evidence HS resolution did not match the requested item HS scope.",
            request=request,
        )

    try:
        results = [result.model_dump(mode="json") for result in response.results]
        evidence = [item for result in results for item in result["evidence"]]
        response_errors = [error.model_dump(mode="json") for error in response.errors]
        context_errors = _context_errors(state, results)
    except Exception:
        return _failed_update(
            state,
            request_id=request_id,
            code="INVALID_RETRIEVAL_RESPONSE",
            message="Evidence response could not be safely normalized.",
            request=request,
        )
    effective_status = "failed" if context_errors else response.status
    retrieval_coverage = "unknown" if context_errors else _coverage(results)
    fail_closed = (
        effective_status != "ok"
        or retrieval_coverage != "complete"
        or not response.knowledge_snapshot_id
        or any(result["errors"] for result in results)
        or any(result["hs_resolution"] not in {"exact", "unique_descendant"} for result in results)
        or any(
            bool(result["requirements_found"])
            != any(
                _is_qualified_requirement_finding(
                    candidate,
                    item_id=result["item_id"],
                    resolved_hsk10=result["resolved_hsk10"],
                )
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
