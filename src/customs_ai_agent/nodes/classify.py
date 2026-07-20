from __future__ import annotations

import time
from typing import Any

from customs_ai_agent.adapters.llm import (
    CropReadBudget,
    DEFAULT_CLAUDE_MODEL,
    DocumentAnalyzer,
    LLMError,
    analysis_to_extracted_document,
    document_prompt_identity,
)
from customs_ai_agent.nodes.common import audit
from customs_ai_agent.nodes.redact import without_image_payloads
from customs_ai_agent.payloads import SensitiveDocumentPayloadStore


PROVIDER_WIDE_FAILURE_CODES = frozenset(
    {
        "provider_authentication_failed",
        "provider_permission_denied",
        "provider_rate_limited",
        "provider_unavailable",
        "provider_request_rejected",
        "provider_status_error",
        "provider_request_failed",
        "provider_metadata_invalid",
        "provider_model_mismatch",
        "unsafe_anthropic_logging_enabled",
        "unsafe_anthropic_logger_enabled",
        "unsafe_anthropic_transport_override",
    }
)


def _classify(text: str, filename: str) -> tuple[str, float]:
    content = f"{filename}\n{text}".upper()
    # Prefer decisive document titles before secondary cross-reference labels.
    # Packing lists legitimately carry both invoice and B/L numbers.
    if "COMMERCIAL INVOICE" in content:
        return "invoice", 0.97
    if "PACKING LIST" in content:
        return "packing_list", 0.97
    if "BILL OF LADING" in content:
        return "bill_of_lading", 0.97
    if "CERTIFICATE OF ORIGIN" in content:
        return "certificate_of_origin", 0.96
    if "PACKING NO" in content:
        return "packing_list", 0.90
    if "B/L NO" in content:
        return "bill_of_lading", 0.90
    if "INVOICE NO" in content:
        return "invoice", 0.90
    if "ORIGIN COUNTRY" in content:
        return "certificate_of_origin", 0.88
    return "unknown", 0.40


def classify_documents(
    state: dict[str, Any],
    analyzer: DocumentAnalyzer | None = None,
    *,
    payload_store: SensitiveDocumentPayloadStore | None = None,
) -> dict[str, Any]:
    """Classify/analyze documents and release raw images before returning.

    Image payloads are consumed exactly once. The only persisted analysis is a
    raw-image-free candidate, provenance ledger, and already-grounded extraction
    snapshot. The extraction node therefore cannot reacquire source bytes.
    """

    try:
        return _classify_documents(state, analyzer, payload_store=payload_store)
    finally:
        if payload_store is not None:
            payload_store.clear()


def _classify_documents(
    state: dict[str, Any],
    analyzer: DocumentAnalyzer | None = None,
    *,
    payload_store: SensitiveDocumentPayloadStore | None = None,
) -> dict[str, Any]:
    classified: dict[str, dict[str, Any]] = {}
    requires_human_review = state.get("requires_human_review", False)
    model_analyses: dict[str, dict[str, Any]] = {}
    model_calls = list(state.get("model_calls") or [])
    model_errors = list(state.get("model_errors") or [])
    model_conflicts = list(state.get("model_conflicts") or [])
    llm_mode = state.get("llm_mode") == "llm"
    circuit_failure: LLMError | None = None
    crop_budget = CropReadBudget()

    for doc in state.get("documents") or []:
        provider = "rules"
        prompt_id, prompt_version, input_modality = document_prompt_identity(doc)
        declared_type = doc.get("declared_type")
        doc_type: str | None
        confidence: float | None
        if llm_mode:
            if analyzer is None:
                raise RuntimeError("llm mode requires an injected document analyzer")
            provider = "anthropic"
            started = time.perf_counter()
            if circuit_failure is not None:
                doc_type = "unknown"
                confidence = 0.0
                requires_human_review = True
                model_errors.append(
                    {
                        "document_id": doc["document_id"],
                        "code": "provider_not_attempted_after_failure",
                        "retryable": circuit_failure.retryable,
                        "provider": "anthropic",
                        "model_id": getattr(analyzer, "model_id", DEFAULT_CLAUDE_MODEL),
                        "prompt_id": prompt_id,
                        "prompt_version": prompt_version,
                        "latency_ms": 0,
                        "provider_request_id": None,
                        "provider_status_code": None,
                        "input_modality": input_modality,
                    }
                )
            else:
                sensitive_document = (
                    payload_store.take(doc["document_id"]) if payload_store is not None else None
                )
                analysis_document = dict(sensitive_document or doc)
                if analysis_document.get("image_pages"):
                    analysis_document["_runtime_crop_budget"] = crop_budget
                try:
                    try:
                        analysis = analyzer.analyze(analysis_document)
                    except LLMError as exc:
                        doc_type = "unknown"
                        confidence = 0.0
                        requires_human_review = True
                        model_errors.append(
                            {
                                "document_id": doc["document_id"],
                                "code": exc.code,
                                "retryable": exc.retryable,
                                "provider": exc.provider,
                                "model_id": getattr(analyzer, "model_id", DEFAULT_CLAUDE_MODEL),
                                "prompt_id": prompt_id,
                                "prompt_version": prompt_version,
                                "latency_ms": round((time.perf_counter() - started) * 1000),
                                "provider_request_id": exc.provider_request_id,
                                "provider_status_code": exc.provider_status_code,
                                "input_modality": input_modality,
                            }
                        )
                        if exc.code in PROVIDER_WIDE_FAILURE_CODES:
                            circuit_failure = exc
                    else:
                        doc_type = analysis.candidate.document_type
                        confidence = 0.85 if doc_type != "unknown" else 0.0
                        stored_analysis = {
                            "candidate": analysis.candidate.model_dump(mode="json"),
                            "metadata": analysis.metadata.as_audit_dict(),
                            "input_modality": analysis.metadata.input_modality,
                            "ocr_layout_ledger": (
                                analysis.ocr_layout_ledger.model_dump(mode="json")
                                if analysis.ocr_layout_ledger is not None
                                else None
                            ),
                        }
                        try:
                            mapped_extraction = analysis_to_extracted_document(
                                analysis_document,
                                analysis,
                            )
                        except LLMError as exc:
                            requires_human_review = True
                            model_errors.append(
                                {
                                    "document_id": doc["document_id"],
                                    "code": exc.code,
                                    "retryable": exc.retryable,
                                    "provider": exc.provider,
                                    "model_id": analysis.metadata.model_id,
                                    "prompt_id": analysis.metadata.prompt_id,
                                    "prompt_version": analysis.metadata.prompt_version,
                                    "latency_ms": 0,
                                    "provider_request_id": None,
                                    "provider_status_code": None,
                                    "input_modality": analysis.metadata.input_modality,
                                }
                            )
                        else:
                            stored_analysis["mapped_extraction"] = without_image_payloads(
                                mapped_extraction
                            )
                        model_analyses[doc["document_id"]] = stored_analysis
                        model_calls.append(
                            {
                                "document_id": doc["document_id"],
                                **analysis.metadata.as_audit_dict(),
                            }
                        )
                        model_calls.extend(
                            {
                                "document_id": doc["document_id"],
                                **call.as_audit_dict(),
                            }
                            for call in analysis.additional_model_calls
                        )
                        model_errors.extend(analysis.additional_model_errors)
                        model_conflicts.extend(analysis.model_conflicts)
                        if analysis.model_conflicts:
                            requires_human_review = True
                        if declared_type and declared_type != doc_type:
                            requires_human_review = True
                except LLMError as exc:
                    # Mapping/provider failures handled above are the only
                    # expected LLM errors. This guard preserves fail-closed
                    # behavior if a future analysis step raises one.
                    doc_type = "unknown"
                    confidence = 0.0
                    requires_human_review = True
                    model_errors.append(
                        {
                            "document_id": doc["document_id"],
                            "code": exc.code,
                            "retryable": exc.retryable,
                            "provider": exc.provider,
                            "model_id": getattr(analyzer, "model_id", DEFAULT_CLAUDE_MODEL),
                            "prompt_id": prompt_id,
                            "prompt_version": prompt_version,
                            "latency_ms": round((time.perf_counter() - started) * 1000),
                            "provider_request_id": exc.provider_request_id,
                            "provider_status_code": exc.provider_status_code,
                            "input_modality": input_modality,
                        }
                    )
                finally:
                    if payload_store is not None:
                        payload_store.release(sensitive_document)
                        # ``analysis_document`` is normally a shallow envelope
                        # over the store-owned pages. Scrub it independently so
                        # a future analyzer replacement cannot leave a raw
                        # reference attached to an exception traceback.
                        payload_store.release(analysis_document)
        else:
            doc_type = declared_type
            confidence = 1.0 if doc_type else None
            if not doc_type:
                doc_type, confidence = _classify(
                    str(doc.get("content_text") or ""),
                    doc["filename"],
                )
        if doc_type == "unknown" or (confidence or 0) < 0.70:
            requires_human_review = True
        classified[doc["document_id"]] = {
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "document_type": doc_type,
            "confidence": confidence,
            "provider": provider,
            "input_modality": input_modality,
        }

    return {
        "classified_documents": classified,
        "model_analyses": model_analyses,
        "model_calls": model_calls,
        "model_errors": model_errors,
        "model_conflicts": model_conflicts,
        "current_stage": "classify_documents",
        "requires_human_review": requires_human_review,
        "audit_trail": audit(
            state,
            "classify_documents",
            "completed",
            "Classified documents.",
            {
                "document_count": len(classified),
                "provider": "anthropic" if llm_mode else "rules",
                "model_call_count": len(model_calls),
                "model_error_count": len(model_errors),
                "model_conflict_count": len(model_conflicts),
            },
        ),
    }
