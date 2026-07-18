"""Internal workflow and evaluation helpers.

Backend/service consumers must use :class:`customs_ai_agent.AgentRuntime`.
These helpers deliberately support mock scenarios and dependency injection for
tests, CLI tooling and evaluation, so they do not enforce the service release
composition boundary.
"""

from __future__ import annotations

import copy
from os import PathLike
from typing import Any, Literal, cast

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
from customs_ai_agent.nodes.redact import without_image_payloads
from customs_ai_agent.nodes.report import generate_xai_report
from customs_ai_agent.nodes.retrieve import retrieve_regulations
from customs_ai_agent.nodes.route import route_to_human_or_complete
from customs_ai_agent.nodes.validate import validate_cross_documents
from customs_ai_agent.payloads import SensitiveDocumentPayloadStore
from customs_ai_agent.schemas.models import (
    ChatAnswer,
    FieldSourceReferenceV2,
    ModelCallRecord,
    ModelErrorDetail,
    OcrProvenanceV2,
    ReviewDocumentResultV2,
    ReviewFieldV2,
    ReviewLineItemV2,
    ReviewRunRequest,
    ReviewRunRequestV2,
    ReviewRunResult,
    ReviewRunResultV2,
    UserCorrection,
)


ReviewRequest = ReviewRunRequest | ReviewRunRequestV2


def _execution_document_values(document: Any) -> dict[str, Any]:
    """Normalize the public text/image/PDF union to the internal page contract."""

    values = document.model_dump(mode="json")
    pdf_pages = values.pop("pdf_pages", None)
    if isinstance(pdf_pages, dict):
        values["image_pages"] = pdf_pages["pages"]
    return values


def _build_initial_state(request: ReviewRequest) -> dict[str, Any]:
    if request.scenario_id:
        scenario = get_scenario(request.scenario_id)
        case_id = request.case_id or scenario["case_id"]
        documents = scenario["documents"]
    else:
        case_id = request.case_id or "case_manual"
        documents = without_image_payloads(
            [_execution_document_values(doc) for doc in request.documents or []]
        )
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
        "model_conflicts": [],
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
        "sensitive_inputs_redacted": False,
    }


def _execute_review(
    request: ReviewRequest,
    retriever: RegulationRetriever | None = None,
    analyzer: DocumentAnalyzer | None = None,
    env_file: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Execute one review without taking ownership of the caller's request.

    A caller-supplied ``ReviewRunRequest`` (and its raw image payloads) remains
    caller-owned and is never mutated. The runner creates one execution-local
    payload copy, consumes it only in the model analysis node, and releases it
    before extraction, normalization, validation, Evidence retrieval, reporting,
    checkpointing, or return.
    """

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
    payload_store = SensitiveDocumentPayloadStore(
        {
            document.document_id: _execution_document_values(document)
            for document in request.documents or []
            if getattr(document, "image_pages", ())
            or getattr(document, "pdf_pages", None) is not None
        }
    )
    graph = build_review_graph(
        active_retriever,
        active_analyzer,
        payload_store=payload_store,
    )
    # Do not retain an orchestration-local reference to a caller-owned request
    # while the remainder of the graph executes.
    del request
    try:
        return graph.invoke(initial_state, config={"recursion_limit": 20})
    finally:
        # Defensive cleanup covers graph setup/runtime exceptions before the
        # classification node can consume and clear the one-shot store.
        payload_store.clear()


def run_review(
    request: ReviewRunRequest | dict[str, Any],
    retriever: RegulationRetriever | None = None,
    analyzer: DocumentAnalyzer | None = None,
    env_file: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Execute the legacy text-only V1 workflow."""

    if isinstance(request, dict):
        request = ReviewRunRequest.model_validate(request)
    return _execute_review(
        request,
        retriever=retriever,
        analyzer=analyzer,
        env_file=env_file,
    )


def run_review_v2(
    request: ReviewRunRequestV2 | dict[str, Any],
    retriever: RegulationRetriever | None = None,
    analyzer: DocumentAnalyzer | None = None,
    env_file: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Execute the additive text/image/PDF V2 workflow."""

    if isinstance(request, dict):
        request = ReviewRunRequestV2.model_validate(request)
    return _execute_review(
        request,
        retriever=retriever,
        analyzer=analyzer,
        env_file=env_file,
    )


def _result_common_values(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("report_json"):
        raise ValueError("review state does not contain a completed report")
    return {
        "case_id": state["case_id"],
        "review_run_id": state["review_run_id"],
        "review_as_of": state["review_as_of"],
        "trade_type": state["trade_type"],
        "jurisdiction": state["jurisdiction"],
        "status": state["final_status"],
        "requires_human_review": state["requires_human_review"],
        "retrieval_status": state["retrieval_status"],
        "retrieval_coverage": state["retrieval_coverage"],
        "knowledge_snapshot_id": state.get("knowledge_snapshot_id"),
        "retrieval_request": state.get("retrieval_request") or None,
        "retrieval_results": state.get("retrieval_results") or [],
        "retrieval_errors": state.get("retrieval_errors") or [],
        "validation_issues": state.get("validation_issues") or [],
        "regulatory_findings": state.get("regulatory_findings") or [],
        "rag_evidence": state.get("rag_evidence") or [],
        "report": state["report_json"],
        "audit_trail": state.get("audit_trail") or [],
    }


def _raw_wire_record(value: Any) -> dict[str, Any]:
    raw = value.model_dump(mode="python") if hasattr(value, "model_dump") else dict(value)
    return dict(raw)


def _project_v1_model_call(value: Any) -> ModelCallRecord:
    raw = _raw_wire_record(value)
    projected = {name: raw[name] for name in ModelCallRecord.model_fields if name in raw}
    return ModelCallRecord.model_validate(projected)


def _project_v1_model_error(value: Any) -> ModelErrorDetail:
    raw = _raw_wire_record(value)
    projected = {name: raw[name] for name in ModelErrorDetail.model_fields if name in raw}
    return ModelErrorDetail.model_validate(projected)


def to_review_run_result(state: dict[str, Any]) -> ReviewRunResult:
    return ReviewRunResult(
        **_result_common_values(state),
        model_calls=[_project_v1_model_call(value) for value in state.get("model_calls") or []],
        model_errors=[_project_v1_model_error(value) for value in state.get("model_errors") or []],
    )


def _public_source_reference_v2(field_data: dict[str, Any]) -> FieldSourceReferenceV2:
    evidence = dict(field_data.get("evidence") or {})
    source_modality_value = str(evidence.get("source_modality") or "text")
    if source_modality_value not in {"text", "image"}:
        raise ValueError("field evidence has an invalid source modality")
    source_modality = cast(Literal["text", "image"], source_modality_value)
    return FieldSourceReferenceV2(
        document_id=str(evidence["document_id"]),
        page=int(evidence.get("page") or 1),
        source_modality=source_modality,
        bbox=evidence.get("bbox"),
        text_start=evidence.get("text_start"),
        text_end=evidence.get("text_end"),
        image_sha256=evidence.get("image_sha256"),
        image_width_px=evidence.get("image_width_px"),
        image_height_px=evidence.get("image_height_px"),
        coordinate_space=evidence.get("coordinate_space"),
        ocr_ledger_id=evidence.get("ocr_ledger_id"),
        ocr_span_ids=list(evidence.get("ocr_span_ids") or []),
    )


def _public_field_v2(
    name: str,
    field_data: dict[str, Any],
    *,
    normalized_value: Any,
) -> ReviewFieldV2:
    confidence_basis_value = str(field_data.get("confidence_basis") or "rule_heuristic")
    allowed_confidence_bases = {
        "rule_heuristic",
        "source_quote_grounded_uncalibrated",
        "model_mapped_to_independent_ocr_uncalibrated",
    }
    if confidence_basis_value not in allowed_confidence_bases:
        raise ValueError("field result has an invalid confidence basis")
    confidence_basis = cast(
        Literal[
            "rule_heuristic",
            "source_quote_grounded_uncalibrated",
            "model_mapped_to_independent_ocr_uncalibrated",
        ],
        confidence_basis_value,
    )
    return ReviewFieldV2(
        name=name,
        value=field_data.get("value"),
        normalized_value=normalized_value,
        confidence=float(field_data.get("confidence") or 0.0),
        confidence_basis=confidence_basis,
        source=_public_source_reference_v2(field_data),
    )


def _public_document_results_v2(state: dict[str, Any]) -> list[ReviewDocumentResultV2]:
    normalized_by_id = state.get("normalized_documents") or {}
    classified_by_id = state.get("classified_documents") or {}
    results: list[ReviewDocumentResultV2] = []
    for document_id, extracted in (state.get("extracted_documents") or {}).items():
        normalized = normalized_by_id.get(document_id) or {}
        normalized_fields = normalized.get("fields") or {}
        fields = [
            _public_field_v2(
                name,
                field_data,
                normalized_value=normalized_fields.get(name),
            )
            for name, field_data in (extracted.get("fields") or {}).items()
        ]
        normalized_items = {
            str(item.get("item_id")): item for item in (normalized.get("line_items") or [])
        }
        line_items: list[ReviewLineItemV2] = []
        for raw_item in extracted.get("line_items") or []:
            item_id = str(raw_item["item_id"])
            normalized_item = normalized_items.get(item_id) or {}
            item_fields = [
                _public_field_v2(
                    name,
                    field_data,
                    normalized_value=normalized_item.get(name),
                )
                for name, field_data in raw_item.items()
                if name != "item_id" and isinstance(field_data, dict)
            ]
            line_items.append(
                ReviewLineItemV2(
                    item_id=item_id,
                    fields=item_fields,
                )
            )
        raw_ocr = extracted.get("ocr_layout")
        ocr_layout = OcrProvenanceV2.model_validate(raw_ocr) if isinstance(raw_ocr, dict) else None
        classified = classified_by_id.get(document_id) or {}
        input_modality = str(
            extracted.get("input_modality") or classified.get("input_modality") or "text"
        )
        if input_modality not in {"text", "image"}:
            raise ValueError("document result has an invalid input modality")
        typed_input_modality = cast(Literal["text", "image"], input_modality)
        results.append(
            ReviewDocumentResultV2(
                document_id=document_id,
                document_type=extracted.get("document_type") or "unknown",
                input_modality=typed_input_modality,
                extraction_status=extracted.get("extraction_status") or "failed",
                extraction_confidence=float(extracted.get("extraction_confidence") or 0.0),
                extraction_provider=(
                    extracted.get("extraction_provider") or classified.get("provider") or "rules"
                ),
                model_id=extracted.get("model_id"),
                prompt_version=extracted.get("prompt_version"),
                ocr_layout=ocr_layout,
                fields=fields,
                line_items=line_items,
            )
        )
    return results


def _finding_source_by_item(
    document_results: list[ReviewDocumentResultV2],
) -> dict[str, FieldSourceReferenceV2]:
    """Resolve each globally unique line item to its grounded HS source."""

    source_by_item: dict[str, FieldSourceReferenceV2] = {}
    observed_item_ids: set[str] = set()
    for document in document_results:
        for line_item in document.line_items:
            if line_item.item_id in observed_item_ids:
                raise ValueError("public line item IDs must be globally unique")
            observed_item_ids.add(line_item.item_id)
            hs_fields = [field for field in line_item.fields if field.name == "hs_code"]
            if len(hs_fields) == 1:
                source_by_item[line_item.item_id] = hs_fields[0].source
    return source_by_item


def _ground_v2_findings(
    findings: list[Any],
    *,
    source_by_item: dict[str, FieldSourceReferenceV2],
) -> list[dict[str, Any]]:
    grounded: list[dict[str, Any]] = []
    for finding in findings:
        payload = _raw_wire_record(finding)
        source = source_by_item.get(str(payload.get("item_id") or ""))
        if payload.get("severity") in {"critical", "high"} and source is None:
            raise ValueError("High/Critical regulatory finding has no public line-item HS source")
        payload["source"] = source.model_dump(mode="json") if source is not None else None
        grounded.append(payload)
    return grounded


def to_review_run_result_v2(state: dict[str, Any]) -> ReviewRunResultV2:
    # V1 deliberately retains its permissive legacy XAI object. Validate the
    # unmodified workflow report at the V2 boundary so an unknown nested key is
    # rejected instead of being silently discarded by the V1 compatibility model.
    document_results = _public_document_results_v2(state)
    grounded_findings = _ground_v2_findings(
        list(state.get("regulatory_findings") or []),
        source_by_item=_finding_source_by_item(document_results),
    )
    base = _result_common_values(state)
    base["regulatory_findings"] = grounded_findings
    report = copy.deepcopy(state["report_json"])
    report["regulatory_findings"] = copy.deepcopy(grounded_findings)
    base["report"] = report
    return ReviewRunResultV2(
        **base,
        model_calls=state.get("model_calls") or [],
        model_errors=state.get("model_errors") or [],
        model_conflicts=state.get("model_conflicts") or [],
        document_results=document_results,
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


def run_review_result_v2(
    request: ReviewRunRequestV2 | dict[str, Any],
    retriever: RegulationRetriever | None = None,
    analyzer: DocumentAnalyzer | None = None,
    env_file: str | PathLike[str] | None = None,
) -> ReviewRunResultV2:
    """Run the same workflow and serialize the additive multimodal V2 result.

    Raw document payloads remain execution-local; the returned model contains
    only extracted/normalized values and bounded source/OCR references.
    """

    return to_review_run_result_v2(
        run_review_v2(
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
        correction
        if isinstance(correction, UserCorrection)
        else UserCorrection.model_validate(correction)
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
