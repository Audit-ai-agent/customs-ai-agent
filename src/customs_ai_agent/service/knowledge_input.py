"""Closed input boundary for the standalone customs knowledge assistant.

The standalone assistant has no Case or document authority.  A caller must not
turn it into an alternate transport for a Case, Review Run, or uploaded source
document by pasting that material into a question or conversation history.

The check is deliberately small and deterministic.  It rejects only material
that is clearly document- or review-shaped, keeps ordinary Korean trade
questions available, and returns stable codes without retaining the input
text.  It is shared by HTTP admission and the runtime so neither path can send
rejected content to Data, Claude, or a local transcript.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_FILENAME = re.compile(
    r"(?<![A-Za-z0-9_])\w[\w.-]{0,160}"
    r"\.(?:txt|pdf|docx?|xlsx?|csv|json|xml)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_UUID_OR_HASH = re.compile(
    r"(?<![A-Za-z0-9._:-])(?:[a-f0-9]{16,}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}"
    r"-[a-f0-9]{12})(?![A-Za-z0-9._:-])",
    re.IGNORECASE,
)
_CASE_OR_REVIEW_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9._:-])(?:case|review(?:[-_\s]?run)?|run|job|document|finding|"
    r"comparison|evidence|proposal|fact|inspection)[._:-][A-Za-z0-9][A-Za-z0-9._:-]{5,}"
    r"(?![A-Za-z0-9._:-])",
    re.IGNORECASE,
)
_STRUCTURED_CASE_FIELD = re.compile(
    r"(?:[\"']?(?:case(?:s)?|review(?:[_\s-]?run)?(?:s)?|document(?:s)?|"
    r"finding(?:s)?|comparison(?:s)?|evidence(?:s)?|proposal(?:s)?|fact(?:s)?|"
    r"content(?:[_\s-]?(?:text|body))?|file(?:name|content)?|"
    r"(?:source|result|input)(?:[_\s-]?(?:hash|sha(?:[_\s-]?256)?))?|"
    r"사건|심사|검토)(?:[_\s-]?(?:id|ids|번호|실행|result|results|body|text|content))?[\"']?"
    r"\s*[:=])",
    re.IGNORECASE,
)
_SYSTEM_CASE_MARKER = re.compile(
    r"(?<![A-Za-z0-9._:-])functional(?:[-_\s]?(?:uat|matrix))(?![A-Za-z0-9._:-])",
    re.IGNORECASE,
)
_DOCUMENT_HEADING = re.compile(
    r"(?im)^\s*(?:commercial\s+invoice|invoice|packing\s+list|bill\s+of\s+lading|"
    r"air\s+waybill|certificate\s+of\s+origin|export\s+declaration|supporting\s+evidence|"
    r"상업송장|포장명세서|선하증권|항공(?:화물)?운송장|원산지(?:증명서)?|"
    r"수출신고(?:필증|증)?|보완\s*증빙|제출\s*증빙)\s*$"
)
_DOCUMENT_FIELD_LABEL = re.compile(
    r"(?im)(?:^|[|;]\s*|\n\s*)"
    r"(?:invoice(?:\s*(?:no\.?|number))?|commercial\s+invoice|packing(?:\s*list)?(?:\s*(?:no\.?|number))?|"
    r"bill\s+of\s+lading(?:\s*(?:no\.?|number))?|b\s*/?\s*l(?:\s*(?:no\.?|number))?|"
    r"air\s+waybill(?:\s*(?:no\.?|number))?|a\s*/?\s*w\s*/?\s*b(?:\s*(?:no\.?|number))?|"
    r"certificate\s+of\s+origin(?:\s*(?:no\.?|number))?|export\s+declaration(?:\s*(?:no\.?|number))?|"
    r"hs\s*(?:code)?|quantity|qty|unit(?:\s*price)?|amount|total(?:\s*amount)?|"
    r"origin(?:\s*country)?|gross\s*weight|container(?:\s*(?:no\.?|number))?|"
    r"인보이스(?:\s*번호)?|송장번호|포장명세서(?:\s*번호)?|선하증권(?:\s*번호)?|"
    r"항공(?:화물)?운송장(?:\s*번호)?|원산지(?:증명서)?(?:\s*번호)?|수출신고(?:번호|필증)?|"
    r"hs\s*코드|수량|단위|단가|금액|총액|원산지|총중량|컨테이너(?:\s*번호)?|"
    r"신고일(?:자)?|제출일(?:자)?|접수번호|처리상태|품명|목적국)\s*[:=]"
)


class KnowledgeInputBoundaryError(ValueError):
    """Content-free rejection for document- or Case-shaped knowledge input."""

    code = "knowledge_input_boundary"

    def __init__(self) -> None:
        super().__init__(self.code)


def _has_document_or_case_marker(value: str) -> bool:
    """Return whether one string carries clear document or internal Case material."""

    return bool(
        _FILENAME.search(value)
        or _UUID_OR_HASH.search(value)
        or _CASE_OR_REVIEW_IDENTIFIER.search(value)
        or _STRUCTURED_CASE_FIELD.search(value)
        or _SYSTEM_CASE_MARKER.search(value)
        or _DOCUMENT_FIELD_LABEL.search(value)
        or _DOCUMENT_HEADING.search(value)
    )


def _has_splitlines_boundary(value: str) -> bool:
    """Match every separator that Python treats as a textual line boundary.

    The non-line-breaking sentinel means a terminal separator still creates a
    second ``splitlines`` item.  Delegating to ``str.splitlines`` prevents this
    security boundary from silently omitting NEL, Unicode line/paragraph
    separators, or another separator recognised by the running Python.
    """

    return len((value + "\x00").splitlines()) > 1


def _turn_role_and_content(turn: Any) -> tuple[str | None, str | None]:
    if isinstance(turn, Mapping):
        role = turn.get("role")
        content = turn.get("content")
    else:
        role = getattr(turn, "role", None)
        content = getattr(turn, "content", None)
    return (role if isinstance(role, str) else None, content if isinstance(content, str) else None)


def knowledge_input_is_safe(question: str, recent_turns: Iterable[Any]) -> bool:
    """Validate a standalone turn without exposing its content in diagnostics.

    A new user question must be a one-line general question.  Persisted
    assistant turns may use line breaks for ordinary prose; their line breaks
    are compacted before any provider call by ``compact_knowledge_turn_text``.
    User turns with multiple lines are rejected because that is the common
    pasted-document shape.  Every role is rejected for an explicit document,
    filename, Case/Review identifier, hash, or structured internal field.
    """

    if not isinstance(question, str) or "\x00" in question:
        return False
    if _has_splitlines_boundary(question) or _has_document_or_case_marker(question):
        return False
    for turn in recent_turns:
        role, content = _turn_role_and_content(turn)
        if role not in {"user", "assistant"} or content is None or "\x00" in content:
            return False
        if _has_document_or_case_marker(content):
            return False
        if role == "user" and _has_splitlines_boundary(content):
            return False
    return True


def assert_knowledge_input_is_safe(question: str, recent_turns: Iterable[Any]) -> None:
    """Raise a stable, content-free error before any external call or recording."""

    if not knowledge_input_is_safe(question, recent_turns):
        raise KnowledgeInputBoundaryError


def compact_knowledge_turn_text(value: str) -> str:
    """Keep safe historical prose usable without forwarding its layout verbatim."""

    return " ".join(value.split())


__all__ = [
    "KnowledgeInputBoundaryError",
    "assert_knowledge_input_is_safe",
    "compact_knowledge_turn_text",
    "knowledge_input_is_safe",
]
