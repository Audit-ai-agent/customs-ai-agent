"""Small boundary checks for text shown directly to a reviewer.

The public schemas deliberately keep stable identifiers in their structured
reference fields.  They are useful to the application, but they are not useful
in a Korean sentence shown to a reviewer.  Keep that distinction at the Agent
boundary instead of changing the wire contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


# This is intentionally a narrow list of code-shaped values, rather than a
# language filter.  Official law names and document quotations stay untouched;
# only internal result identifiers, hashes, and enum/action codes are rejected
# from user-facing prose.
_INTERNAL_RESULT_MARKER = re.compile(
    r"""
    (?<![a-z0-9])
    (?:
        [a-f0-9]{64}
        | (?:sha(?:[-_]?256)?|result[_-]?sha256|input[_-]?manifest[_-]?sha256)
        | (?:comparison|finding|inspect|report|run|case|job|proposal|fact|evidence|action|human)
          [-_:][a-z0-9._:-]+
        | (?:
            case_grounded|evidence_grounded|general_customs_guidance|clarification_needed
            | insufficient_evidence|out_of_scope|needs_review|no_issue_detected
            | issues_detected|human_review_required|human_provided|validation_issue
            | regulatory_finding|local-development|functional-uat|classify_documents
            | extract_facts|normalize_facts|match_documents|validate_documents
            | reextract_documents|retrieve_evidence|generate_result|analyze_documents
            | apply_human_corrections|compare_documents|generate_report
        )
    )
    (?![a-z0-9._:-])
    """,
    re.IGNORECASE | re.VERBOSE,
)

_INTERNAL_VALUE_KEY = re.compile(
    r"(?:^|_)(?:id|ids|code|codes|status|kind|stage|action|sha256|hash|mode|version)$",
    re.IGNORECASE,
)

# Reviewer-facing prose is deliberately Korean-first.  Structured citations
# keep original document quotations separately, so a generated sentence never
# needs to repeat an English filename or an implementation-shaped label.  A
# small set of trade abbreviations remains useful and is safe to show.
_RAW_FILENAME = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z0-9][A-Za-z0-9._-]*\.(?:txt|pdf|docx|xlsx|csv)(?![A-Za-z0-9_])", re.I
)
_ENGLISH_WORD = re.compile(r"(?<![A-Za-z])([A-Za-z]{3,})(?![A-Za-z])")
_ALLOWED_REVIEWER_ABBREVIATIONS = frozenset(
    {
        "AI",
        "AWB",
        "API",
        "B",
        "BOX",
        "CBM",
        "CIF",
        "CIP",
        "C/O",
        "CPT",
        "CNY",
        "CTN",
        "CTNS",
        "DAP",
        "DDP",
        "EXW",
        "EA",
        "EUR",
        "FOB",
        "FTA",
        "GBP",
        "GROSS",
        "HS",
        "HSK",
        "JPY",
        "KG",
        "KGS",
        "KRW",
        "LLM",
        "NET",
        "OCR",
        "PCS",
        "PDF",
        "PKG",
        "PKGS",
        "TEU",
        "USD",
    }
)

_DETERMINISTIC_TERM_REPLACEMENTS = (
    ("Packing List", "포장명세서"),
    ("Commercial Invoice", "상업송장"),
    ("Invoice와", "상업송장과"),
    ("Invoice가", "상업송장이"),
    ("Invoice는", "상업송장은"),
    ("Invoice를", "상업송장을"),
    ("Invoice", "상업송장"),
    ("법령 Evidence", "법령 근거"),
    ("Evidence", "법령 근거"),
    ("Review Run", "심사 결과"),
)


def with_particle(word: str, pair: str) -> str:
    """Attach a Korean particle pair ("을를", "과와", "이가") by the final batchim.

    A word that does not end in Hangul (e.g. "B/L") keeps the vowel form.
    """

    last = word[-1:] or " "
    has_batchim = "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0
    return word + (pair[0] if has_batchim else pair[1])


def contains_internal_result_marker(value: str, *, markers: Iterable[str] = ()) -> bool:
    """Return whether reviewer prose contains an implementation-only marker."""

    if _INTERNAL_RESULT_MARKER.search(value):
        return True
    return any(marker and _marker_pattern(marker).search(value) for marker in markers)


def _marker_pattern(marker: str) -> re.Pattern[str]:
    # A marker must stand as its own token.  Plain substring matching rejected
    # ordinary prose such as "관세법 제232조" whenever a short structured value
    # like law_id "23" was present in the result.  Korean text next to the
    # marker still counts as a boundary, so "run_abc가" remains blocked.
    return re.compile(
        rf"(?<![0-9A-Za-z_-])(?<![0-9][.,]){re.escape(marker)}(?![0-9A-Za-z_-])(?![.,][0-9])"
    )


def reviewer_text_is_safe(
    values: Iterable[str],
    *,
    markers: Iterable[str] = (),
    filenames: Iterable[str] = (),
) -> bool:
    """Keep reviewer prose Korean-first and free of technical source labels.

    Original quotations stay in ``SourceV3`` and are intentionally not passed
    through this boundary.  The optional filename list catches a nonstandard
    extension as well as the ordinary ``.txt`` forms covered by the regex.
    """

    marker_values = tuple(markers)
    filename_values = tuple(value.casefold() for value in filenames if value)
    for value in values:
        if contains_internal_result_marker(value, markers=marker_values):
            return False
        lowered = value.casefold()
        if _RAW_FILENAME.search(value) or any(filename in lowered for filename in filename_values):
            return False
        if any(
            word.group(1).upper() not in _ALLOWED_REVIEWER_ABBREVIATIONS
            for word in _ENGLISH_WORD.finditer(value)
        ):
            return False
    return True


def localize_deterministic_reviewer_prose(value: str) -> str:
    """Translate implementation-owned English labels in deterministic prose only.

    Source quotations and official law names are not passed to this helper.  They
    stay verbatim in the structured source/citation fields.
    """

    localized = value
    for source, target in _DETERMINISTIC_TERM_REPLACEMENTS:
        localized = re.sub(re.escape(source), target, localized, flags=re.IGNORECASE)
    return localized


def review_result_markers(result: Any) -> frozenset[str]:
    """Collect only structured IDs, hashes, codes, and enum values from a result.

    Quotes, raw document values, and official law names are deliberately not
    collected.  That lets reviewer prose preserve those originals while still
    rejecting the current result's internal labels.
    """

    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json", warnings=False)
    else:
        payload = result
    markers: set[str] = set()

    def collect(value: Any, *, internal_value: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                collect(
                    child,
                    internal_value=(
                        internal_value
                        or (isinstance(key, str) and bool(_INTERNAL_VALUE_KEY.search(key)))
                    ),
                )
            return
        if isinstance(value, list | tuple):
            for child in value:
                collect(child, internal_value=internal_value)
            return
        if internal_value and isinstance(value, str) and value:
            markers.add(value)

    collect(payload)
    return frozenset(markers)


__all__ = [
    "contains_internal_result_marker",
    "localize_deterministic_reviewer_prose",
    "review_result_markers",
    "reviewer_text_is_safe",
]
