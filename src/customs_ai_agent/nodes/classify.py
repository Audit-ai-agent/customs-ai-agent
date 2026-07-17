from __future__ import annotations

import time
from typing import Any

from customs_ai_agent.adapters.llm import (
    DEFAULT_CLAUDE_MODEL,
    DocumentAnalyzer,
    LLMError,
)
from customs_ai_agent.nodes.common import audit
from customs_ai_agent.prompts import (
    DOCUMENT_ANALYSIS_PROMPT_ID,
    DOCUMENT_ANALYSIS_PROMPT_VERSION,
)


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
    if "COMMERCIAL INVOICE" in content or "INVOICE NO" in content:
        return "invoice", 0.97
    if "PACKING LIST" in content or "PACKING NO" in content:
        return "packing_list", 0.97
    if "BILL OF LADING" in content or "B/L NO" in content:
        return "bill_of_lading", 0.97
    if "CERTIFICATE OF ORIGIN" in content or "ORIGIN COUNTRY" in content:
        return "certificate_of_origin", 0.96
    return "unknown", 0.40


def classify_documents(
    state: dict[str, Any],
    analyzer: DocumentAnalyzer | None = None,
) -> dict[str, Any]:
    classified: dict[str, dict[str, Any]] = {}
    requires_human_review = state.get("requires_human_review", False)
    model_analyses: dict[str, dict[str, Any]] = {}
    model_calls = list(state.get("model_calls") or [])
    model_errors = list(state.get("model_errors") or [])
    llm_mode = state.get("llm_mode") == "llm"
    circuit_failure: LLMError | None = None

    for doc in state.get("documents") or []:
        provider = "rules"
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
                        "model_id": getattr(
                            analyzer, "model_id", DEFAULT_CLAUDE_MODEL
                        ),
                        "prompt_id": DOCUMENT_ANALYSIS_PROMPT_ID,
                        "prompt_version": DOCUMENT_ANALYSIS_PROMPT_VERSION,
                        "latency_ms": 0,
                        "provider_request_id": None,
                        "provider_status_code": None,
                    }
                )
            else:
                try:
                    analysis = analyzer.analyze(doc)
                except LLMError as exc:
                    doc_type = "unknown"
                    confidence = 0.0
                    requires_human_review = True
                    model_errors.append(
                        {
                            "document_id": doc["document_id"],
                            "code": exc.code,
                            "retryable": exc.retryable,
                            "provider": "anthropic",
                            "model_id": getattr(
                                analyzer, "model_id", DEFAULT_CLAUDE_MODEL
                            ),
                            "prompt_id": DOCUMENT_ANALYSIS_PROMPT_ID,
                            "prompt_version": DOCUMENT_ANALYSIS_PROMPT_VERSION,
                            "latency_ms": round(
                                (time.perf_counter() - started) * 1000
                            ),
                            "provider_request_id": exc.provider_request_id,
                            "provider_status_code": exc.provider_status_code,
                        }
                    )
                    if exc.code in PROVIDER_WIDE_FAILURE_CODES:
                        circuit_failure = exc
                else:
                    doc_type = analysis.candidate.document_type
                    confidence = 0.85 if doc_type != "unknown" else 0.0
                    model_analyses[doc["document_id"]] = {
                        "candidate": analysis.candidate.model_dump(mode="json"),
                        "metadata": analysis.metadata.as_audit_dict(),
                    }
                    model_calls.append(
                        {
                            "document_id": doc["document_id"],
                            **analysis.metadata.as_audit_dict(),
                        }
                    )
                    if declared_type and declared_type != doc_type:
                        requires_human_review = True
        else:
            doc_type = declared_type
            confidence = 1.0 if doc_type else None
            if not doc_type:
                doc_type, confidence = _classify(doc["content_text"], doc["filename"])
        if doc_type == "unknown" or (confidence or 0) < 0.70:
            requires_human_review = True
        classified[doc["document_id"]] = {
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "document_type": doc_type,
            "confidence": confidence,
            "provider": provider,
        }

    return {
        "classified_documents": classified,
        "model_analyses": model_analyses,
        "model_calls": model_calls,
        "model_errors": model_errors,
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
            },
        ),
    }
