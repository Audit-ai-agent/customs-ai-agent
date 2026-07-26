from __future__ import annotations

from typing import Any

from customs_ai_agent.nodes.common import audit
from customs_ai_agent.policies.evidence import enforce_evidence_gate
from customs_ai_agent.schemas.models import RagEvidence, RegulatoryFinding


def _uploaded_requirement_codes(state: dict[str, Any]) -> set[str]:
    codes = {
        doc.get("document_type")
        for doc in (state.get("normalized_documents") or {}).values()
        if doc.get("document_type")
    }
    codes.update(state.get("uploaded_requirement_document_codes") or [])
    for document in state.get("documents") or []:
        codes.update(document.get("requirement_document_codes") or [])
    return codes


def _validated_evidence(evidence: Any) -> RagEvidence | None:
    try:
        return RagEvidence.model_validate(evidence)
    except Exception:
        return None


def compliance_review(state: dict[str, Any]) -> dict[str, Any]:
    raw_evidence = state.get("rag_evidence") or []
    uploaded = _uploaded_requirement_codes(state)
    verified_by_item = {
        str(item_id): {str(code) for code in codes}
        for item_id, codes in (state.get("verified_requirement_documents_by_item") or {}).items()
        if isinstance(codes, (list, tuple, set))
    }
    presented_by_item = {
        str(item_id): {str(code) for code in codes}
        for item_id, codes in (state.get("presented_requirement_documents_by_item") or {}).items()
        if isinstance(codes, (list, tuple, set))
    }
    findings: list[dict[str, Any]] = []
    requires_human = state.get("requires_human_review", False)
    rejected = 0

    retrieval_context_is_verified = (
        state.get("retrieval_status") == "ok"
        and state.get("retrieval_coverage") == "complete"
        and bool(state.get("knowledge_snapshot_id"))
    )
    qualified_evidence: list[dict[str, Any]] = []
    if raw_evidence and not retrieval_context_is_verified:
        rejected += len(raw_evidence)
        requires_human = True
    else:
        for evidence in raw_evidence:
            parsed = _validated_evidence(evidence)
            if parsed is None:
                rejected += 1
                requires_human = True
                continue
            if parsed.eligibility != "finding":
                continue
            if parsed.source_type != "customs_requirement":
                rejected += 1
                requires_human = True
                continue
            qualified_evidence.append(parsed.model_dump(mode="json"))

    available_ids = {item["evidence_id"] for item in qualified_evidence}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for evidence in qualified_evidence:
        required_code = evidence.get("required_document_code")
        required_name = evidence.get("required_document_name")
        item_id = evidence.get("item_id")
        if (
            not required_code
            or not required_name
            or not item_id
            or not evidence.get("source_record_id")
            or not evidence.get("source_uri")
            or not evidence.get("version_hash")
        ):
            requires_human = True
            rejected += 1
            continue
        if (
            required_code in uploaded
            or required_code in verified_by_item.get(item_id, set())
            or required_code in presented_by_item.get(item_id, set())
        ):
            continue
        requirement_key = (item_id, required_code)
        grouped.setdefault(requirement_key, []).append(evidence)

    for (item_id, required_code), evidence_group in grouped.items():
        first = evidence_group[0]
        required_name = first["required_document_name"]
        if any(item.get("required_document_name") != required_name for item in evidence_group):
            # A canonical document code must not carry conflicting display names.
            requires_human = True
            rejected += 1
            continue
        scores = [item["score"] for item in evidence_group if item.get("score") is not None]
        confidence = max(scores) if scores else 0.95
        finding = RegulatoryFinding(
            finding_code="REQUIRED_CERT_MISSING",
            severity="high",
            message=f"Required supporting document is missing: {required_name}.",
            item_id=item_id,
            required_document_code=required_code,
            required_document_name=required_name,
            evidence_ids=sorted({item["evidence_id"] for item in evidence_group}),
            confidence=confidence,
            confidence_basis=("retrieval_score" if scores else "structured_requirement_policy"),
        )
        if enforce_evidence_gate(finding, available_ids):
            findings.append(finding.model_dump(mode="json"))
        else:
            rejected += 1
            requires_human = True

    return {
        "regulatory_findings": findings,
        "current_stage": "compliance_review",
        "requires_human_review": requires_human,
        "audit_trail": audit(
            state,
            "compliance_review",
            "completed",
            "Generated evidence-bound compliance findings.",
            {"finding_count": len(findings), "rejected_or_low_evidence": rejected},
        ),
    }
