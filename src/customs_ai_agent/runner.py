"""Internal workflow and evaluation helpers.

Backend/service consumers must use :class:`customs_ai_agent.AgentRuntime`.
These helpers deliberately support mock scenarios and dependency injection for
tests, CLI tooling and evaluation, so they do not enforce the service release
composition boundary.
"""

from __future__ import annotations

import copy
import json
import math
import time
from os import PathLike
from collections.abc import Callable
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
WorkflowStageState = Literal["started", "completed", "failed"]
WorkflowStageObserver = Callable[
    [str, WorkflowStageState, dict[str, int], int | None],
    None,
]


def _execution_document_values(document: Any) -> dict[str, Any]:
    """Normalize the public text/image/PDF union to the internal page contract."""

    values = document.model_dump(mode="json")
    pdf_pages = values.pop("pdf_pages", None)
    if isinstance(pdf_pages, dict):
        values["image_pages"] = pdf_pages["pages"]
    return values


def _build_initial_state(
    request: ReviewRequest,
    *,
    initial_retrieval_attempt: int = 0,
) -> dict[str, Any]:
    if initial_retrieval_attempt < 0:
        raise ValueError("initial_retrieval_attempt cannot be negative")
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
        "retrieval_attempt": initial_retrieval_attempt,
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
    *,
    initial_retrieval_attempt: int = 0,
    stage_observer: WorkflowStageObserver | None = None,
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
    initial_state = _build_initial_state(
        request,
        initial_retrieval_attempt=initial_retrieval_attempt,
    )
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
        if stage_observer is None:
            return graph.invoke(initial_state, config={"recursion_limit": 20})
        final_state = initial_state
        task_started_ns: dict[str, int] = {}
        for mode, event in graph.stream(
            initial_state,
            config={"recursion_limit": 20},
            stream_mode=["debug", "values"],
        ):
            if not isinstance(event, dict):
                continue
            if mode == "values":
                final_state = event
                continue
            if mode != "debug":
                continue
            event_type = event.get("type")
            event_payload = event.get("payload")
            if not isinstance(event_payload, dict):
                continue
            task_id = event_payload.get("id")
            node_name = event_payload.get("name")
            if not isinstance(task_id, str) or not isinstance(node_name, str):
                continue
            if event_type == "task":
                task_started_ns[task_id] = time.perf_counter_ns()
                stage_observer(node_name, "started", {}, None)
                continue
            if event_type != "task_result":
                continue
            started_ns = task_started_ns.pop(task_id, None)
            duration_ms = (
                max(1, min(300_000, round((time.perf_counter_ns() - started_ns) / 1_000_000)))
                if started_ns is not None
                else None
            )
            partial = event_payload.get("result")
            counts_source = partial if isinstance(partial, dict) else final_state
            safe_counts = {
                "documents": len(
                    counts_source.get("documents")
                    or counts_source.get("extracted_documents")
                    or counts_source.get("normalized_documents")
                    or []
                ),
                "issues": len(counts_source.get("validation_issues") or []),
                "findings": len(counts_source.get("regulatory_findings") or []),
                "evidence": len(
                    counts_source.get("rag_evidence")
                    or counts_source.get("retrieval_results")
                    or []
                ),
            }
            state: WorkflowStageState = (
                "failed" if event_payload.get("error") is not None else "completed"
            )
            stage_observer(node_name, state, safe_counts, duration_ms)
        return final_state
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
    *,
    initial_retrieval_attempt: int = 0,
    stage_observer: WorkflowStageObserver | None = None,
) -> dict[str, Any]:
    """Execute the additive text/image/PDF V2 workflow."""

    if isinstance(request, dict):
        request = ReviewRunRequestV2.model_validate(request)
    return _execute_review(
        request,
        retriever=retriever,
        analyzer=analyzer,
        env_file=env_file,
        initial_retrieval_attempt=initial_retrieval_attempt,
        stage_observer=stage_observer,
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


def _reused_text_field(field: ReviewFieldV2, document_text: str) -> dict[str, Any]:
    source = field.source
    if (
        source.source_modality != "text"
        or source.page != 1
        or source.bbox is not None
        or source.text_start is None
        or source.text_end is None
        or source.text_start < 0
        or source.text_end > len(document_text)
    ):
        raise ValueError("parent field source is not a reusable text span")
    raw_text = document_text[source.text_start : source.text_end]
    if not raw_text:
        raise ValueError("parent field source span is empty")
    values = field.value if isinstance(field.value, list) else [field.value]
    if any(value is not None and str(value) not in raw_text for value in values):
        raise ValueError("parent field value is not grounded in its source span")
    return {
        "name": field.name,
        "value": copy.deepcopy(field.value),
        "confidence": field.confidence,
        "confidence_basis": field.confidence_basis,
        "evidence": {
            "document_id": source.document_id,
            "page": source.page,
            "bbox": None,
            "raw_text": raw_text,
            "text_start": source.text_start,
            "text_end": source.text_end,
            "confidence": field.confidence,
            "source_modality": "text",
            "image_sha256": None,
            "image_width_px": None,
            "image_height_px": None,
            "coordinate_space": None,
            "ocr_ledger_id": None,
            "ocr_span_ids": [],
        },
    }


def build_reused_extraction_state(
    request: ReviewRunRequestV2,
    parent_result: ReviewRunResultV2,
    *,
    parent_result_sha256: str,
    aggregate_reuse_key_sha256: str,
) -> dict[str, Any]:
    """Rebuild a child execution state from a hash-bound public extraction.

    This function performs no model or Evidence I/O. Callers must first bind
    the parent result/request and current extraction runtime identity.
    """

    if any(document.input_modality != "text" for document in parent_result.document_results):
        raise ValueError("only text extraction can be reused")
    if parent_result.model_errors or parent_result.model_conflicts:
        raise ValueError("failed or conflicting parent extraction cannot be reused")
    documents = request.documents
    if not documents:
        raise ValueError("reused extraction requires primary documents")
    request_documents = {document.document_id: document for document in documents}
    parent_documents = {
        document.document_id: document for document in parent_result.document_results
    }
    if set(request_documents) != set(parent_documents):
        raise ValueError("parent and child document sets differ")

    state = _build_initial_state(request)
    classified: dict[str, dict[str, Any]] = {}
    extracted: dict[str, dict[str, Any]] = {}
    for document in documents:
        parent = parent_documents[document.document_id]
        if parent.extraction_status != "completed" or parent.document_type == "unknown":
            raise ValueError("incomplete parent extraction cannot be reused")
        if parent.extraction_provider != "anthropic":
            raise ValueError("only Anthropic extraction can be reused")
        content_text = str(document.content_text or "")
        fields = {field.name: _reused_text_field(field, content_text) for field in parent.fields}
        line_items = [
            {
                "item_id": item.item_id,
                **{field.name: _reused_text_field(field, content_text) for field in item.fields},
            }
            for item in parent.line_items
        ]
        classified[document.document_id] = {
            "document_id": document.document_id,
            "filename": document.filename,
            "document_type": parent.document_type,
            "confidence": parent.extraction_confidence,
            "provider": parent.extraction_provider,
            "input_modality": "text",
        }
        extracted[document.document_id] = {
            "document_id": document.document_id,
            "filename": document.filename,
            "document_type": parent.document_type,
            "fields": fields,
            "line_items": line_items,
            "extraction_status": parent.extraction_status,
            "extraction_confidence": parent.extraction_confidence,
            "extraction_provider": parent.extraction_provider,
            "input_modality": "text",
            "model_id": parent.model_id,
            "prompt_version": parent.prompt_version,
        }
    state.update(
        {
            "document_ids": [document.document_id for document in documents],
            "classified_documents": classified,
            "extracted_documents": extracted,
            "model_calls": [
                item.model_dump(mode="json", warnings=False) for item in parent_result.model_calls
            ],
            "model_errors": [],
            "model_conflicts": [],
            "current_stage": "extract_documents",
        }
    )
    state["audit_trail"] = audit(
        state,
        "extract_documents",
        "parent_extraction_reused",
        "Reused the hash-bound parent text extraction for deterministic child review.",
        {
            "parent_result_sha256": parent_result_sha256,
            "aggregate_reuse_key_sha256": aggregate_reuse_key_sha256,
            "reused_document_count": len(extracted),
            "inherited_model_call_count": len(parent_result.model_calls),
            "actual_document_model_call_count": 0,
        },
    )
    return state


class CorrectionApplicationError(ValueError):
    """A safe, fail-closed error raised when an approved correction cannot apply."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _canonical_json_value(value: Any, *, error_code: str) -> str:
    """Return a type-preserving canonical JSON representation.

    ``UserCorrection.value`` is intentionally an open JSON value in the public
    contract. Direct workflow callers do not pass through the serialized API
    boundary, so the runner must still reject Python-only or non-finite values.
    """

    stack = [value]
    while stack:
        item = stack.pop()
        if item is None or type(item) in {bool, int, str}:
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise CorrectionApplicationError(
                    error_code,
                    "A correction contains a non-finite numeric value.",
                )
            continue
        if type(item) is list:
            stack.extend(item)
            continue
        if type(item) is dict and all(type(key) is str for key in item):
            stack.extend(item.values())
            continue
        raise CorrectionApplicationError(
            error_code,
            "A correction contains a value outside the JSON data model.",
        )
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) in {int, float}:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    return "unsupported"


def _resolve_correction_target(
    state: dict[str, Any],
    correction: UserCorrection,
) -> dict[str, Any]:
    extracted = state.get("extracted_documents")
    if not isinstance(extracted, dict):
        raise CorrectionApplicationError(
            "correction_state_invalid",
            "The extracted child state is unavailable.",
        )
    document = extracted.get(correction.document_id)
    if not isinstance(document, dict):
        raise CorrectionApplicationError(
            "correction_document_missing",
            "The approved correction document is absent from the child extraction.",
        )

    parts = correction.field_path.split(".")
    field_data: Any
    if len(parts) == 2 and parts[0] == "fields" and parts[1]:
        fields = document.get("fields")
        if not isinstance(fields, dict) or parts[1] not in fields:
            raise CorrectionApplicationError(
                "correction_field_missing",
                "The approved correction field is absent from the child extraction.",
            )
        field_data = fields[parts[1]]
    elif (
        len(parts) == 3
        and parts[0] == "line_items"
        and parts[1].isascii()
        and parts[1].isdigit()
        and parts[2]
    ):
        line_items = document.get("line_items")
        index = int(parts[1])
        if not isinstance(line_items, list) or not 0 <= index < len(line_items):
            raise CorrectionApplicationError(
                "correction_line_item_missing",
                "The approved correction line item is absent from the child extraction.",
            )
        line_item = line_items[index]
        if not isinstance(line_item, dict) or parts[2] not in line_item:
            raise CorrectionApplicationError(
                "correction_field_missing",
                "The approved correction line-item field is absent from the child extraction.",
            )
        field_data = line_item[parts[2]]
    else:
        raise CorrectionApplicationError(
            "correction_path_invalid",
            "The approved correction path is not a supported field path.",
        )

    if not isinstance(field_data, dict) or "value" not in field_data:
        raise CorrectionApplicationError(
            "correction_field_invalid",
            "The approved correction target is not a typed extracted field.",
        )
    return field_data


def _normalized_correction_value(
    state: dict[str, Any],
    correction: UserCorrection,
) -> Any:
    documents = state.get("normalized_documents")
    document = documents.get(correction.document_id) if isinstance(documents, dict) else None
    if not isinstance(document, dict):
        raise CorrectionApplicationError(
            "correction_not_persisted",
            "The corrected document is absent after normalization.",
        )
    parts = correction.field_path.split(".")
    if len(parts) == 2 and parts[0] == "fields":
        raw_fields = document.get("raw_fields")
        if not isinstance(raw_fields, dict) or parts[1] not in raw_fields:
            raise CorrectionApplicationError(
                "correction_not_persisted",
                "The corrected field is absent after normalization.",
            )
        return raw_fields[parts[1]]
    line_items = document.get("line_items")
    index = int(parts[1])
    if not isinstance(line_items, list) or not 0 <= index < len(line_items):
        raise CorrectionApplicationError(
            "correction_not_persisted",
            "The corrected line item is absent after normalization.",
        )
    line_item = line_items[index]
    raw_fields = line_item.get("raw_fields") if isinstance(line_item, dict) else None
    if not isinstance(raw_fields, dict) or parts[2] not in raw_fields:
        raise CorrectionApplicationError(
            "correction_not_persisted",
            "The corrected line-item field is absent after normalization.",
        )
    return raw_fields[parts[2]]


def _validate_corrections(corrections: list[UserCorrection]) -> None:
    targets: dict[tuple[str, str], str] = {}
    for correction in corrections:
        encoded = _canonical_json_value(
            correction.value,
            error_code="correction_value_invalid",
        )
        target = (correction.document_id, correction.field_path)
        if target in targets:
            code = (
                "duplicate_correction_target"
                if targets[target] == encoded
                else "conflicting_correction_target"
            )
            raise CorrectionApplicationError(
                code,
                "A revision contains more than one correction for the same target.",
            )
        targets[target] = encoded


def _apply_correction(state: dict[str, Any], correction: UserCorrection) -> None:
    field_data = _resolve_correction_target(state, correction)
    current_value = field_data["value"]
    _canonical_json_value(current_value, error_code="correction_field_invalid")
    expected = _canonical_json_value(
        correction.value,
        error_code="correction_value_invalid",
    )
    if current_value is not None and _json_value_kind(current_value) != _json_value_kind(
        correction.value
    ):
        raise CorrectionApplicationError(
            "correction_value_type_mismatch",
            "The correction value type does not match the extracted field type.",
        )
    field_data["value"] = copy.deepcopy(correction.value)
    field_data["confidence"] = correction.confidence
    field_data["value_source"] = "human_correction"
    if (
        _canonical_json_value(field_data["value"], error_code="correction_not_applied") != expected
        or field_data.get("confidence") != correction.confidence
        or field_data.get("value_source") != "human_correction"
    ):
        raise CorrectionApplicationError(
            "correction_not_applied",
            "The approved correction was not applied exactly.",
        )


def _verify_correction_persisted(
    state: dict[str, Any],
    correction: UserCorrection,
) -> None:
    expected = _canonical_json_value(
        correction.value,
        error_code="correction_value_invalid",
    )
    field_data = _resolve_correction_target(state, correction)
    normalized_raw_value = _normalized_correction_value(state, correction)
    if (
        _canonical_json_value(field_data["value"], error_code="correction_not_persisted")
        != expected
        or _canonical_json_value(
            normalized_raw_value,
            error_code="correction_not_persisted",
        )
        != expected
        or field_data.get("confidence") != correction.confidence
        or field_data.get("value_source") != "human_correction"
    ):
        raise CorrectionApplicationError(
            "correction_not_persisted",
            "The approved correction did not survive the child review stages exactly.",
        )


def _merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    state.update(update)
    return state


def _rerun_review(
    state: dict[str, Any],
    corrections: list[UserCorrection | dict[str, Any]],
    *,
    retriever: RegulationRetriever | None,
    finalize: bool,
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
    _validate_corrections(parsed)
    for correction in parsed:
        _apply_correction(next_state, correction)
    next_state["requires_human_review"] = False
    next_state["final_status"] = None
    for node in [
        normalize_documents,
        pair_documents,
        validate_cross_documents,
    ]:
        next_state = _merge(next_state, node(next_state))
        if node is validate_cross_documents:
            next_state = _merge(
                next_state,
                retrieve_regulations(next_state, active_retriever),
            )
    if finalize:
        for node in [compliance_review, generate_xai_report, route_to_human_or_complete]:
            next_state = _merge(next_state, node(next_state))
    for correction in parsed:
        _verify_correction_persisted(next_state, correction)
    next_state["audit_trail"] = audit(
        next_state,
        "rerun_review",
        "corrections_applied",
        (
            "Applied user corrections and completed all deterministic child review stages."
            if finalize
            else "Applied user corrections and reran deterministic stages through Evidence."
        ),
        {"correction_count": len(parsed), "finalized": finalize},
    )
    return next_state


def rerun_review_to_retrieval(
    state: dict[str, Any],
    corrections: list[UserCorrection | dict[str, Any]],
    retriever: RegulationRetriever,
) -> dict[str, Any]:
    return _rerun_review(
        state,
        corrections,
        retriever=retriever,
        finalize=False,
    )


def rerun_review(
    state: dict[str, Any],
    corrections: list[UserCorrection | dict[str, Any]],
    retriever: RegulationRetriever | None = None,
) -> dict[str, Any]:
    return _rerun_review(
        state,
        corrections,
        retriever=retriever,
        finalize=True,
    )


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
