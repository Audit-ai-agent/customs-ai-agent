"""Boundary helpers for non-evidence text in a Case assistant conversation.

The Backend owns persistence of an authorized user message.  The Agent must
nevertheless protect direct callers and unsafe historical turns: pasted
document bodies are not review evidence and must never reach Data, Claude, or
the local scenario transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from customs_ai_agent.schemas.assistant import AssistantAnswerRequestV1


CASE_DOCUMENT_PLACEHOLDER = "등록되지 않은 문서 원문은 심사 근거로 사용하지 않습니다."

_DOCUMENT_HEADING = re.compile(
    r"(?:commercial\s+invoice|invoice|packing\s+list|bill\s+of\s+lading|"
    r"air\s+waybill|certificate\s+of\s+origin|export\s+declaration|"
    r"상업송장|포장명세서|선하증권|항공화물운송장|원산지증명서|수출신고(?:증)?)",
    re.IGNORECASE,
)
_FIELD_LABEL = re.compile(
    r"(?:invoice\s*(?:no|number)|packing\s*(?:no|number)|b\s*/?\s*l\s*(?:no|number)?|"
    r"awb\s*(?:no|number)?|hs\s*code|description|quantity|unit\s*price|amount|"
    r"total(?:\s+amount)?|country\s+of\s+origin|submission\s*(?:no|date|status)|"
    r"신고번호|접수번호|제출일|처리상태|품명|수량|단가|금액|총액|원산지|hs\s*코드)\s*[:=：]",
    re.IGNORECASE,
)


def is_document_shaped_case_text(value: str) -> bool:
    """Identify a pasted document body without treating short value mentions as one.

    A normal prompt may legitimately contain "B/L 번호" or an invoice number.
    The boundary therefore requires multiple lines plus a document heading or
    multiple field-value rows; it does not reject one-line business questions.
    """

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    field_rows = sum(bool(_FIELD_LABEL.search(line)) for line in lines)
    heading = any(_DOCUMENT_HEADING.search(line) for line in lines[:2])
    table_rows = sum("|" in line or "\t" in line for line in lines)
    return (heading and field_rows >= 1) or field_rows >= 2 or (len(lines) >= 4 and table_rows >= 2)


@dataclass(frozen=True)
class CaseInputSanitization:
    request: AssistantAnswerRequestV1
    current_input_was_document_shaped: bool


def sanitize_case_assistant_request(request: AssistantAnswerRequestV1) -> CaseInputSanitization:
    """Replace document-shaped current/history text before any capability call."""

    current = is_document_shaped_case_text(request.question)
    safe_turns = [
        turn.model_copy(update={"content": CASE_DOCUMENT_PLACEHOLDER})
        if turn.role == "user" and is_document_shaped_case_text(turn.content)
        else turn
        for turn in request.recent_turns
    ]
    safe_request = request.model_copy(
        update={
            "question": CASE_DOCUMENT_PLACEHOLDER if current else request.question.strip(),
            "recent_turns": safe_turns,
        }
    )
    return CaseInputSanitization(
        request=safe_request,
        current_input_was_document_shaped=current,
    )


__all__ = [
    "CASE_DOCUMENT_PLACEHOLDER",
    "CaseInputSanitization",
    "is_document_shaped_case_text",
    "sanitize_case_assistant_request",
]
