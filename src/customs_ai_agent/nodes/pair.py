from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from customs_ai_agent.nodes.common import audit
from customs_ai_agent.policies.case_profile import (
    CASE_PROFILE_ID,
    CASE_PROFILE_VERSION,
    required_document_types,
)

MIN_LINE_ITEM_PAIR_SCORE = 0.70

fuzz: Any = None
try:
    from rapidfuzz import fuzz as _rapidfuzz

    fuzz = _rapidfuzz
except ImportError:  # pragma: no cover
    pass


def _score(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    if fuzz:
        return fuzz.token_set_ratio(left, right) / 100.0
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def _unique_doc(
    state: dict[str, Any],
    doc_type: str,
) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    matches = [
        (doc_id, doc)
        for doc_id, doc in (state.get("normalized_documents") or {}).items()
        if doc.get("document_type") == doc_type
    ]
    if len(matches) == 1:
        return matches[0][0], matches[0][1], []
    if len(matches) > 1:
        return None, None, sorted(doc_id for doc_id, _ in matches)
    return None, None, []


def pair_documents(state: dict[str, Any]) -> dict[str, Any]:
    invoice_id, invoice, invoice_conflict = _unique_doc(state, "invoice")
    packing_id, packing, packing_conflict = _unique_doc(state, "packing_list")
    _, _, bl_conflict = _unique_doc(state, "bill_of_lading")
    _, _, co_conflict = _unique_doc(state, "certificate_of_origin")
    links: list[dict[str, Any]] = []
    conflicts = [
        {
            "document_type": document_type,
            "document_ids": document_ids,
        }
        for document_type, document_ids in (
            ("invoice", invoice_conflict),
            ("packing_list", packing_conflict),
            ("bill_of_lading", bl_conflict),
            ("certificate_of_origin", co_conflict),
        )
        if document_ids
    ]
    selected_by_type = {
        "invoice": invoice,
        "packing_list": packing,
        "bill_of_lading": _unique_doc(state, "bill_of_lading")[1],
    }
    missing_required_types = [
        document_type
        for document_type in required_document_types(str(state.get("trade_type") or "import"))
        if selected_by_type[document_type] is None
        and not any(conflict["document_type"] == document_type for conflict in conflicts)
    ]

    unmatched_invoice_item_ids: list[str] = []
    unmatched_packing_item_ids: list[str] = []
    pairing_ambiguities: list[dict[str, Any]] = []
    if invoice and packing:
        invoice_items = invoice.get("line_items") or []
        packing_items = packing.get("line_items") or []
        candidates = sorted(
            (
                (
                    _score(
                        invoice_item.get("description"),
                        packing_item.get("description"),
                    ),
                    str(invoice_item.get("item_id") or ""),
                    str(packing_item.get("item_id") or ""),
                )
                for invoice_item in invoice_items
                for packing_item in packing_items
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        eligible_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] >= MIN_LINE_ITEM_PAIR_SCORE and candidate[1] and candidate[2]
        ]
        candidates_by_invoice: dict[str, list[tuple[float, str]]] = {}
        candidates_by_packing: dict[str, list[tuple[float, str]]] = {}
        for score, invoice_item_id, packing_item_id in eligible_candidates:
            candidates_by_invoice.setdefault(invoice_item_id, []).append((score, packing_item_id))
            candidates_by_packing.setdefault(packing_item_id, []).append((score, invoice_item_id))

        for invoice_item_id, matches in sorted(candidates_by_invoice.items()):
            if len(matches) <= 1:
                continue
            ordered_matches = sorted(matches, key=lambda item: (-item[0], item[1]))
            best_score = ordered_matches[0][0]
            pairing_ambiguities.append(
                {
                    "ambiguity_type": "one_to_many_candidates",
                    "invoice_item_ids": [invoice_item_id],
                    "packing_item_ids": sorted(item[1] for item in ordered_matches),
                    "best_score": round(best_score, 4),
                    "best_score_tied": (sum(item[0] == best_score for item in ordered_matches) > 1),
                }
            )
        for packing_item_id, matches in sorted(candidates_by_packing.items()):
            if len(matches) <= 1:
                continue
            ordered_matches = sorted(matches, key=lambda item: (-item[0], item[1]))
            best_score = ordered_matches[0][0]
            pairing_ambiguities.append(
                {
                    "ambiguity_type": "many_to_one_candidates",
                    "invoice_item_ids": sorted(item[1] for item in ordered_matches),
                    "packing_item_ids": [packing_item_id],
                    "best_score": round(best_score, 4),
                    "best_score_tied": (sum(item[0] == best_score for item in ordered_matches) > 1),
                }
            )

        assigned_packing: set[str] = set()
        assigned: dict[str, tuple[str, float]] = {}
        for score, invoice_item_id, packing_item_id in eligible_candidates:
            if len(candidates_by_invoice[invoice_item_id]) != 1:
                continue
            if len(candidates_by_packing[packing_item_id]) != 1:
                continue
            assigned_packing.add(packing_item_id)
            assigned[invoice_item_id] = (packing_item_id, score)

        for invoice_item in invoice_items:
            invoice_item_id = str(invoice_item.get("item_id") or "")
            match = assigned.get(invoice_item_id)
            if not match and invoice_item_id:
                unmatched_invoice_item_ids.append(invoice_item_id)
            if not match and invoice_item_id not in candidates_by_invoice:
                low_confidence_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[1] == invoice_item_id and candidate[2]
                ]
                best_candidate = low_confidence_candidates[0] if low_confidence_candidates else None
                pairing_ambiguities.append(
                    {
                        "ambiguity_type": "below_minimum_score",
                        "invoice_item_ids": [invoice_item_id] if invoice_item_id else [],
                        "packing_item_ids": (
                            [best_candidate[2]] if best_candidate is not None else []
                        ),
                        "best_score": (
                            round(best_candidate[0], 4) if best_candidate is not None else 0.0
                        ),
                        "best_score_tied": False,
                    }
                )
            links.append(
                {
                    "invoice_item_id": invoice_item_id or None,
                    "packing_item_id": match[0] if match else None,
                    "score": round(match[1], 4) if match else 0.0,
                }
            )
        unmatched_packing_item_ids = sorted(
            str(item.get("item_id"))
            for item in packing_items
            if item.get("item_id") and str(item.get("item_id")) not in assigned_packing
        )

    result = {
        "invoice_document_id": invoice_id,
        "packing_list_document_id": packing_id,
        "line_item_links": links,
        "minimum_line_item_pair_score": MIN_LINE_ITEM_PAIR_SCORE,
        "line_item_pairing_ambiguities": pairing_ambiguities,
        "document_type_conflicts": conflicts,
        "missing_required_document_types": missing_required_types,
        "case_profile": {
            "profile_id": CASE_PROFILE_ID,
            "profile_version": CASE_PROFILE_VERSION,
        },
        "document_set_status": (
            "ambiguous_document_set"
            if conflicts
            else "incomplete_document_set"
            if missing_required_types
            else "deterministic"
        ),
        # A PL can legitimately include packing-only accessories or materials.
        # Until a business rule adjudicates them, retain them as diagnostic
        # ambiguity and never turn them into a violation.
        "unmatched_invoice_item_ids": sorted(unmatched_invoice_item_ids),
        "unmatched_invoice_item_status": ("needs_review" if unmatched_invoice_item_ids else "none"),
        "unmatched_packing_item_ids": unmatched_packing_item_ids,
        "unmatched_packing_item_status": (
            "ambiguous_not_scored" if unmatched_packing_item_ids else "none"
        ),
        "line_item_pairing_status": (
            "ambiguous"
            if pairing_ambiguities or unmatched_invoice_item_ids
            else "deterministic_with_unscored_packing_extras"
            if unmatched_packing_item_ids
            else "deterministic"
        ),
    }
    line_item_needs_review = bool(pairing_ambiguities or unmatched_invoice_item_ids)
    return {
        "pairing_result": result,
        "current_stage": "pair_documents",
        "requires_human_review": (
            state.get("requires_human_review", False)
            or bool(conflicts)
            or bool(missing_required_types)
            or line_item_needs_review
        ),
        "audit_trail": audit(
            state,
            "pair_documents",
            (
                "needs_review"
                if conflicts or missing_required_types or line_item_needs_review
                else "completed"
            ),
            (
                "Duplicate document types prevented deterministic pairing."
                if conflicts
                else "Required document types are missing from the case."
                if missing_required_types
                else "Line-item pairing was not deterministic."
                if line_item_needs_review
                else "Paired shipment and line items."
            ),
            {
                "case_profile_id": CASE_PROFILE_ID,
                "case_profile_version": CASE_PROFILE_VERSION,
                "line_item_links": len(links),
                "document_type_conflicts": len(conflicts),
                "missing_required_documents": len(missing_required_types),
                "line_item_pairing_ambiguities": len(pairing_ambiguities),
                "unmatched_invoice_items": len(unmatched_invoice_item_ids),
                "unmatched_packing_items": len(unmatched_packing_item_ids),
            },
        ),
    }
