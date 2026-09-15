"""Case-level HSK candidate extraction and ranking.

The model may only *choose* among reference codes the Backend supplies; it
extracts the goods description from document text and explains its choice.
Every malformed part of an answer is dropped, so a bad model turn degrades to
"no picks" rather than to an invented tariff code.
"""

from __future__ import annotations

import re
from typing import Any

from customs_ai_agent.schemas.v3 import (
    HsCandidateItemV1,
    HsCandidatePickV1,
    HsCandidatesRequestV1,
)

_MAX_ITEMS = 20
_MAX_SEARCH_TERMS = 10
_MAX_SEARCH_TERM_LENGTH = 80
_DIGITS = re.compile(r"[0-9]+")


def request_payload(request: HsCandidatesRequestV1) -> dict[str, Any]:
    """The exact JSON object the model receives; document text is passed as is."""

    return {
        "documents": [
            {
                "document_id": document.document_id,
                "document_type": document.document_type,
                "content_text": document.content_text[:20_000],
            }
            for document in request.documents
        ],
        "reference_candidates": [
            {
                "code": reference.code,
                "name_ko": reference.name_ko,
                "name_en": reference.name_en,
                "ancestors": reference.ancestors[:8],
            }
            for reference in request.reference_candidates
        ],
        "max_picks": request.max_picks,
    }


def _search_terms(value: object) -> list[str]:
    """Distinct, bounded lookup terms; anything that is not a short string is dropped."""

    terms: list[str] = []
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, str):
            continue
        term = " ".join(raw.split())
        if not 1 <= len(term) <= _MAX_SEARCH_TERM_LENGTH or term in terms:
            continue
        terms.append(term)
        if len(terms) >= _MAX_SEARCH_TERMS:
            break
    return terms


def validate_answer(
    answer: object, request: HsCandidatesRequestV1
) -> tuple[list[HsCandidateItemV1], list[HsCandidatePickV1]]:
    """Keep only well-formed items and picks that name a supplied reference code."""

    if not isinstance(answer, dict):
        return [], []
    known_documents = {document.document_id for document in request.documents}
    allowed_codes = {reference.code for reference in request.reference_candidates}

    items: list[HsCandidateItemV1] = []
    raw_items = answer.get("items")
    if isinstance(raw_items, list):
        for raw in raw_items[:_MAX_ITEMS]:
            if not isinstance(raw, dict):
                continue
            description = raw.get("description")
            if not isinstance(description, str) or not 1 <= len(description.strip()) <= 300:
                continue
            document_ids = raw.get("document_ids")
            cited = raw.get("cited_hs_codes")
            items.append(
                HsCandidateItemV1(
                    description=description.strip(),
                    document_ids=[
                        value
                        for value in (document_ids if isinstance(document_ids, list) else [])
                        if isinstance(value, str) and value in known_documents
                    ][:20],
                    cited_hs_codes=[
                        "".join(_DIGITS.findall(value))
                        for value in (cited if isinstance(cited, list) else [])
                        if isinstance(value, str)
                        and 6 <= len("".join(_DIGITS.findall(value))) <= 10
                    ][:10],
                    search_terms=_search_terms(raw.get("search_terms")),
                )
            )

    picks: list[HsCandidatePickV1] = []
    seen: set[str] = set()
    raw_picks = answer.get("picks")
    if isinstance(raw_picks, list):
        for raw in raw_picks[: request.max_picks * 2]:
            if not isinstance(raw, dict):
                continue
            code = raw.get("code")
            reason = raw.get("reason")
            confidence = raw.get("confidence")
            if (
                not isinstance(code, str)
                or code not in allowed_codes
                or code in seen
                or not isinstance(reason, str)
                or not 1 <= len(reason.strip()) <= 400
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                continue
            seen.add(code)
            picks.append(
                HsCandidatePickV1(
                    code=code,
                    rank=len(picks) + 1,
                    confidence=round(float(confidence), 4),
                    reason=reason.strip(),
                )
            )
            if len(picks) >= request.max_picks:
                break
    return items, picks


__all__ = ["request_payload", "validate_answer"]
