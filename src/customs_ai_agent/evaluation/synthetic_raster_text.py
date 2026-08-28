"""Source-only Gold projection and text-only synthetic-raster quality gates.

The structured raster Gold artifact is derived from final PDF content streams
and PDFium geometry.  This module consumes only that immutable artifact; it
does not read CSV labels, existing JSON references, OCR output, or VLM output
to manufacture a text answer key.  It intentionally evaluates text drafts
separately from the structured v2 table/geometry Gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from customs_ai_agent.evaluation.synthetic_raster import (
    RasterGoldArtifactV1,
    RasterProfileId,
    _canonical_visual_cer_text,
)
from customs_ai_agent.schemas.raster_text_extraction import RasterTextExtractionResultV1


_DOCUMENT_TYPES = ("bill_of_lading", "commercial_invoice", "packing_list")
_PROFILE_IDS = ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2: Literal["source-attested-semantic-text-v2"] = (
    "source-attested-semantic-text-v2"
)
RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3: Literal["source-attested-semantic-text-v3"] = (
    "source-attested-semantic-text-v3"
)
_MAX_SEMANTIC_TEXT_CODEPOINTS = 200_000
# This is deliberately a closed set.  Adding a visually similar glyph changes
# the metric and therefore requires a new metric revision.
_STANDALONE_FORM_MARKERS_V2 = frozenset(
    {
        "\N{BLACK SQUARE}",
        "\N{WHITE SQUARE}",
        "\N{BLACK SMALL SQUARE}",
        "\N{WHITE SMALL SQUARE}",
        "\N{BALLOT BOX}",
        "\N{BALLOT BOX WITH CHECK}",
        "\N{BALLOT BOX WITH X}",
    }
)
_STANDALONE_FORM_MARKER_PATTERN_V2 = re.compile(
    rf"(?<!\S)[{re.escape(''.join(sorted(_STANDALONE_FORM_MARKERS_V2)))}](?!\S)"
)
_UNDERSCORE_DECORATION_PATTERN_V2 = re.compile(r"_{2,}")
_WHITESPACE_PATTERN_V2 = re.compile(r"\s+")
_STANDALONE_CIRCLED_DIGIT_PATTERN_V3 = re.compile(r"(?<!\S)[\u2460-\u2469](?!\S)")


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def canonicalize_synthetic_raster_semantic_text_v2(value: str) -> str:
    """Return the fixed candidate-independent semantic text representation.

    The text-only OCR contract does not promise PDF span line breaks or form
    decoration that the recognizer cannot emit.  V2 therefore applies only
    these ordered rules to both source Gold and candidate text:

    1. NFC normalization;
    2. removal of a closed allowlist of *standalone* square/checkbox markers;
    3. removal of underscore runs of length two or greater;
    4. collapse of every whitespace run to one ASCII space and removal of the
       resulting boundary spaces.

    Single underscores, non-standalone markers, and every other punctuation
    character remain significant.  The input/output guards keep normalization
    and the subsequent edit-distance calculation fail-closed on malformed or
    unexpectedly large evidence.
    """

    if not isinstance(value, str) or len(value) > _MAX_SEMANTIC_TEXT_CODEPOINTS:
        raise ValueError("raster semantic text input is too large")
    normalized = unicodedata.normalize("NFC", value)
    if len(normalized) > _MAX_SEMANTIC_TEXT_CODEPOINTS:
        raise ValueError("raster semantic text input is too large")
    without_markers = _STANDALONE_FORM_MARKER_PATTERN_V2.sub("", normalized)
    without_decoration = _UNDERSCORE_DECORATION_PATTERN_V2.sub("", without_markers)
    canonical = _WHITESPACE_PATTERN_V2.sub(" ", without_decoration).strip(" ")
    if len(canonical) > _MAX_SEMANTIC_TEXT_CODEPOINTS:
        raise ValueError("raster semantic text output is too large")
    return canonical


def canonicalize_synthetic_raster_semantic_text_v3(value: str) -> str:
    """Apply V2 plus removal of standalone, non-emittable circled digits.

    Only one-codepoint whitespace-delimited spans in U+2460 through U+2469
    are excluded.  Joined or embedded occurrences remain significant.  This
    keeps V2 byte-for-byte unchanged while binding the narrowly attested V3
    exclusion to a new metric revision.
    """

    canonical_v2 = canonicalize_synthetic_raster_semantic_text_v2(value)
    without_standalone_circled_digits = _STANDALONE_CIRCLED_DIGIT_PATTERN_V3.sub(
        "", canonical_v2
    )
    canonical = _WHITESPACE_PATTERN_V2.sub(" ", without_standalone_circled_digits).strip(" ")
    if len(canonical) > _MAX_SEMANTIC_TEXT_CODEPOINTS:
        raise ValueError("raster semantic text output is too large")
    return canonical


def _levenshtein(reference: str, candidate: str) -> int:
    if len(reference) * len(candidate) > 10_000_000:
        raise ValueError("raster text CER input is too large")

    # Myers' bit-vector recurrence computes the same unit-cost Levenshtein
    # distance as the former row-by-row dynamic program.  Each bit represents
    # one Unicode code point in the shorter input; Python's arbitrary-width
    # integers update all of those DP cells in a handful of native operations.
    # Keeping the shorter input as the bit pattern bounds both the mask size
    # and the amount of work for highly asymmetric inputs.
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    pattern_length = len(candidate)
    if pattern_length == 0:
        return len(reference)

    character_masks: dict[str, int] = {}
    for bit_index, character in enumerate(candidate):
        character_masks[character] = character_masks.get(character, 0) | (1 << bit_index)

    full_mask = (1 << pattern_length) - 1
    positive_vertical = full_mask
    negative_vertical = 0
    distance = pattern_length
    final_bit = 1 << (pattern_length - 1)

    for character in reference:
        equal = character_masks.get(character, 0)
        vertical_or_equal = equal | negative_vertical
        horizontal = (
            (((equal & positive_vertical) + positive_vertical) ^ positive_vertical) | equal
        ) & full_mask
        positive_horizontal = (
            negative_vertical | ~(horizontal | positive_vertical)
        ) & full_mask
        negative_horizontal = positive_vertical & horizontal

        if positive_horizontal & final_bit:
            distance += 1
        elif negative_horizontal & final_bit:
            distance -= 1

        positive_horizontal = ((positive_horizontal << 1) | 1) & full_mask
        negative_horizontal = (negative_horizontal << 1) & full_mask
        positive_vertical = (
            negative_horizontal | ~(vertical_or_equal | positive_horizontal)
        ) & full_mask
        negative_vertical = positive_horizontal & vertical_or_equal

    return distance


class RasterTextGoldArtifactV1(_WireModel):
    """Compact source-only text Gold derived from one existing raster Gold file."""

    schema_version: Literal["1.0"] = "1.0"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    asset_id: str = Field(min_length=1, max_length=128)
    family_id: str = Field(min_length=1, max_length=128)
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    source_pdf_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_raster_gold_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_text: str = Field(min_length=1, max_length=200_000)
    critical_values: list[str] = Field(min_length=1, max_length=512)

def source_only_text_gold_from_raster_gold(
    gold: RasterGoldArtifactV1,
) -> RasterTextGoldArtifactV1:
    """Project source-attested spans/fields without using non-source labels."""

    # Content-stream order is not necessarily the physical order visible in
    # a raster.  Reuse the structured evaluator's source-PDFium visual order
    # exactly, so text CER does not silently reward a different ordering rule.
    canonical_text = _canonical_visual_cer_text(
        spans=[
            (span.page_number, span.bbox, span.reading_order, span.text)
            for span in gold.spans
        ]
    )
    # Existing raster field Gold is compiled from the same final source PDF
    # content stream and binds source_pdf_sha256.  Its values provide a small
    # critical-value gate without importing clean_manifest/error_labels.
    # Do not deduplicate: two independently required fields with the same
    # source value require two observed occurrences.  This remains sourced
    # solely from the final-PDF-derived field artifacts.
    critical_values = [_normalized(field.value) for field in gold.fields if field.value]
    if not critical_values:
        # Every supported template still has source spans.  Use the first
        # non-empty source token rather than silently accepting no critical
        # text condition when a template has no mapped field.
        critical_values = [next(_normalized(span.text) for span in gold.spans if span.text)]
    source_bytes = json.dumps(
        gold.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return RasterTextGoldArtifactV1(
        asset_id=gold.asset_id,
        family_id=gold.family_id,
        document_type=gold.document_type,
        profile_id=gold.profile_id,
        source_pdf_sha256=gold.source_pdf_sha256,
        source_raster_gold_sha256=hashlib.sha256(source_bytes).hexdigest(),
        canonical_text=canonical_text,
        critical_values=critical_values,
    )


@dataclass(frozen=True, slots=True)
class RasterTextEvaluationCaseV1:
    """One source Gold/candidate comparison with raster-byte provenance."""

    asset_id: str
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: RasterProfileId
    image_sha256: str
    gold: RasterTextGoldArtifactV1
    candidate: RasterTextExtractionResultV1

    def __post_init__(self) -> None:
        if (
            self.gold.asset_id != self.asset_id
            or self.gold.document_type != self.document_type
            or self.gold.profile_id != self.profile_id
            or self.candidate.document_asset_id != self.asset_id
            or self.candidate.source_sha256 != self.image_sha256
        ):
            raise ValueError("raster text evaluation case binding is invalid")


@dataclass(frozen=True, slots=True)
class RasterTextMetricCountsV1:
    character_errors: int = 0
    character_reference: int = 0
    critical_matches: int = 0
    critical_total: int = 0

    def __add__(self, other: "RasterTextMetricCountsV1") -> "RasterTextMetricCountsV1":
        return RasterTextMetricCountsV1(
            character_errors=self.character_errors + other.character_errors,
            character_reference=self.character_reference + other.character_reference,
            critical_matches=self.critical_matches + other.critical_matches,
            critical_total=self.critical_total + other.critical_total,
        )


class RasterTextGateScoreV1(_WireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0)
    critical_value_exact_match: float = Field(ge=0, le=1)


def _exact_value_tokens(value: str) -> tuple[str, ...]:
    """Return deterministic whitespace-delimited values without substrings.

    OCR values are compared as exact token sequences after Unicode/newline
    normalization.  A Gold identifier ``INV-12`` therefore cannot be
    satisfied by ``INV-123``; matching a value twice requires two separate
    non-overlapping candidate occurrences.
    """

    normalized = _normalized(value).strip()
    tokens = tuple(token for token in re.split(r"\s+", normalized) if token)
    if not tokens:
        raise ValueError("raster text critical value is empty")
    return tokens


def _critical_value_match_count(*, critical_values: Sequence[str], observed: str) -> int:
    """Count required values once each with exact boundary/occurrence rules."""

    observed_tokens = _exact_value_tokens(observed) if observed.strip() else ()
    consumed = [False] * len(observed_tokens)
    matches = 0
    # Longest sequences first makes a multi-token amount/reference incapable
    # of being consumed piecemeal by a shorter incidental field value.
    indexed = sorted(
        enumerate(_exact_value_tokens(value) for value in critical_values),
        key=lambda item: (-len(item[1]), item[1], item[0]),
    )
    for _ordinal, wanted in indexed:
        width = len(wanted)
        for start in range(0, len(observed_tokens) - width + 1):
            if (
                tuple(observed_tokens[start : start + width]) == wanted
                and not any(consumed[start : start + width])
            ):
                for index in range(start, start + width):
                    consumed[index] = True
                matches += 1
                break
    return matches


def evaluate_synthetic_raster_text_case(case: RasterTextEvaluationCaseV1) -> RasterTextMetricCountsV1:
    """Calculate source-only CER and critical-value exact matches."""

    observed = "" if case.candidate.canonical_text is None else _normalized(case.candidate.canonical_text)
    reference = _normalized(case.gold.canonical_text)
    return RasterTextMetricCountsV1(
        character_errors=_levenshtein(reference, observed),
        character_reference=len(reference),
        critical_matches=_critical_value_match_count(
            critical_values=case.gold.critical_values, observed=observed
        ),
        critical_total=len(case.gold.critical_values),
    )


def _semantic_exact_value_tokens_v2(value: str) -> tuple[str, ...]:
    canonical = canonicalize_synthetic_raster_semantic_text_v2(value)
    tokens = tuple(canonical.split(" ")) if canonical else ()
    if not tokens:
        raise ValueError("raster semantic text critical value is empty")
    return tokens


def _semantic_critical_value_match_count_v2(
    *, critical_values: Sequence[str], observed: str
) -> int:
    """Apply the V1 exact/non-overlapping matcher to V2 canonical tokens."""

    observed_canonical = canonicalize_synthetic_raster_semantic_text_v2(observed)
    observed_tokens = tuple(observed_canonical.split(" ")) if observed_canonical else ()
    consumed = [False] * len(observed_tokens)
    matches = 0
    indexed = sorted(
        enumerate(_semantic_exact_value_tokens_v2(value) for value in critical_values),
        key=lambda item: (-len(item[1]), item[1], item[0]),
    )
    for _ordinal, wanted in indexed:
        width = len(wanted)
        for start in range(0, len(observed_tokens) - width + 1):
            if (
                tuple(observed_tokens[start : start + width]) == wanted
                and not any(consumed[start : start + width])
            ):
                for index in range(start, start + width):
                    consumed[index] = True
                matches += 1
                break
    return matches


def evaluate_synthetic_raster_semantic_text_case_v2(
    case: RasterTextEvaluationCaseV1,
) -> RasterTextMetricCountsV1:
    """Calculate V2 semantic CER and still-exact critical-value matches."""

    observed_raw = "" if case.candidate.canonical_text is None else case.candidate.canonical_text
    observed = canonicalize_synthetic_raster_semantic_text_v2(observed_raw)
    reference = canonicalize_synthetic_raster_semantic_text_v2(case.gold.canonical_text)
    return RasterTextMetricCountsV1(
        character_errors=_levenshtein(reference, observed),
        character_reference=len(reference),
        critical_matches=_semantic_critical_value_match_count_v2(
            critical_values=case.gold.critical_values,
            observed=observed_raw,
        ),
        critical_total=len(case.gold.critical_values),
    )


def _semantic_exact_value_tokens_v3(value: str) -> tuple[str, ...]:
    canonical = canonicalize_synthetic_raster_semantic_text_v3(value)
    tokens = tuple(canonical.split(" ")) if canonical else ()
    if not tokens:
        raise ValueError("raster semantic text critical value is empty")
    return tokens


def _semantic_critical_value_match_count_v3(
    *, critical_values: Sequence[str], observed: str
) -> int:
    """Apply the exact/non-overlapping matcher to V3 canonical tokens."""

    observed_canonical = canonicalize_synthetic_raster_semantic_text_v3(observed)
    observed_tokens = tuple(observed_canonical.split(" ")) if observed_canonical else ()
    consumed = [False] * len(observed_tokens)
    matches = 0
    indexed = sorted(
        enumerate(_semantic_exact_value_tokens_v3(value) for value in critical_values),
        key=lambda item: (-len(item[1]), item[1], item[0]),
    )
    for _ordinal, wanted in indexed:
        width = len(wanted)
        for start in range(0, len(observed_tokens) - width + 1):
            if (
                tuple(observed_tokens[start : start + width]) == wanted
                and not any(consumed[start : start + width])
            ):
                for index in range(start, start + width):
                    consumed[index] = True
                matches += 1
                break
    return matches


def evaluate_synthetic_raster_semantic_text_case_v3(
    case: RasterTextEvaluationCaseV1,
) -> RasterTextMetricCountsV1:
    """Calculate V3 semantic CER and still-exact critical-value matches."""

    observed_raw = "" if case.candidate.canonical_text is None else case.candidate.canonical_text
    observed = canonicalize_synthetic_raster_semantic_text_v3(observed_raw)
    reference = canonicalize_synthetic_raster_semantic_text_v3(case.gold.canonical_text)
    return RasterTextMetricCountsV1(
        character_errors=_levenshtein(reference, observed),
        character_reference=len(reference),
        critical_matches=_semantic_critical_value_match_count_v3(
            critical_values=case.gold.critical_values,
            observed=observed_raw,
        ),
        critical_total=len(case.gold.critical_values),
    )


def evaluate_synthetic_raster_text_gates(
    cases: Sequence[RasterTextEvaluationCaseV1],
) -> list[RasterTextGateScoreV1]:
    """Return the exact 3×4 score matrix, or fail before activation."""

    buckets: dict[tuple[str, str], RasterTextMetricCountsV1] = {}
    for case in cases:
        key = (case.document_type, case.profile_id)
        buckets[key] = buckets.get(key, RasterTextMetricCountsV1()) + evaluate_synthetic_raster_text_case(
            case
        )
    expected = {(document_type, profile_id) for document_type in _DOCUMENT_TYPES for profile_id in _PROFILE_IDS}
    if set(buckets) != expected:
        raise ValueError("raster text evaluation requires every document type and profile")
    return [
        RasterTextGateScoreV1(
            document_type=cast_document_type(document_type),
            profile_id=cast_profile_id(profile_id),
            character_error_rate=(
                1.0
                if counts.character_reference == 0
                else counts.character_errors / counts.character_reference
            ),
            critical_value_exact_match=(
                0.0
                if counts.critical_total == 0
                else counts.critical_matches / counts.critical_total
            ),
        )
        for (document_type, profile_id), counts in sorted(buckets.items())
    ]


def evaluate_synthetic_raster_semantic_text_gates_v2(
    cases: Sequence[RasterTextEvaluationCaseV1],
) -> list[RasterTextGateScoreV1]:
    """Return the exact 3×4 V2 semantic score matrix, or fail closed."""

    buckets: dict[tuple[str, str], RasterTextMetricCountsV1] = {}
    for case in cases:
        key = (case.document_type, case.profile_id)
        buckets[key] = buckets.get(
            key, RasterTextMetricCountsV1()
        ) + evaluate_synthetic_raster_semantic_text_case_v2(case)
    expected = {
        (document_type, profile_id)
        for document_type in _DOCUMENT_TYPES
        for profile_id in _PROFILE_IDS
    }
    if set(buckets) != expected:
        raise ValueError("raster semantic text evaluation requires every document type and profile")
    return [
        RasterTextGateScoreV1(
            document_type=cast_document_type(document_type),
            profile_id=cast_profile_id(profile_id),
            character_error_rate=(
                1.0
                if counts.character_reference == 0
                else counts.character_errors / counts.character_reference
            ),
            critical_value_exact_match=(
                0.0
                if counts.critical_total == 0
                else counts.critical_matches / counts.critical_total
            ),
        )
        for (document_type, profile_id), counts in sorted(buckets.items())
    ]


def evaluate_synthetic_raster_semantic_text_gates_v3(
    cases: Sequence[RasterTextEvaluationCaseV1],
) -> list[RasterTextGateScoreV1]:
    """Return the exact 3×4 V3 semantic score matrix, or fail closed."""

    buckets: dict[tuple[str, str], RasterTextMetricCountsV1] = {}
    for case in cases:
        key = (case.document_type, case.profile_id)
        buckets[key] = buckets.get(
            key, RasterTextMetricCountsV1()
        ) + evaluate_synthetic_raster_semantic_text_case_v3(case)
    expected = {
        (document_type, profile_id)
        for document_type in _DOCUMENT_TYPES
        for profile_id in _PROFILE_IDS
    }
    if set(buckets) != expected:
        raise ValueError("raster semantic text evaluation requires every document type and profile")
    return [
        RasterTextGateScoreV1(
            document_type=cast_document_type(document_type),
            profile_id=cast_profile_id(profile_id),
            character_error_rate=(
                1.0
                if counts.character_reference == 0
                else counts.character_errors / counts.character_reference
            ),
            critical_value_exact_match=(
                0.0
                if counts.critical_total == 0
                else counts.critical_matches / counts.critical_total
            ),
        )
        for (document_type, profile_id), counts in sorted(buckets.items())
    ]


def cast_document_type(value: str) -> Literal["bill_of_lading", "commercial_invoice", "packing_list"]:
    if value not in _DOCUMENT_TYPES:
        raise ValueError("raster text document type is invalid")
    return value  # type: ignore[return-value]


def cast_profile_id(value: str) -> Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]:
    if value not in _PROFILE_IDS:
        raise ValueError("raster text profile is invalid")
    return value  # type: ignore[return-value]


def synthetic_raster_text_gate_passes(scores: Sequence[RasterTextGateScoreV1]) -> bool:
    """Text-v1's independent, non-relaxable activation Gate."""

    return len(scores) == 12 and all(
        score.character_error_rate <= 0.02 and score.critical_value_exact_match >= 0.98
        for score in scores
    )


def synthetic_raster_semantic_text_gate_passes_v2(
    scores: Sequence[RasterTextGateScoreV1],
) -> bool:
    """V2 keeps the non-relaxable V1 thresholds and complete 3×4 matrix."""

    return len(scores) == 12 and all(
        score.character_error_rate <= 0.02 and score.critical_value_exact_match >= 0.98
        for score in scores
    )


def synthetic_raster_semantic_text_gate_passes_v3(
    scores: Sequence[RasterTextGateScoreV1],
) -> bool:
    """V3 keeps the non-relaxable thresholds and complete 3×4 matrix."""

    return len(scores) == 12 and all(
        score.character_error_rate <= 0.02 and score.critical_value_exact_match >= 0.98
        for score in scores
    )


def synthetic_raster_semantic_text_preview_draft_gate_passes_v1(
    scores: Sequence[RasterTextGateScoreV1],
) -> bool:
    """Bounded preview-draft policy: only critical exact relaxes to 97%."""

    return len(scores) == 12 and all(
        score.character_error_rate <= 0.02 and score.critical_value_exact_match >= 0.97
        for score in scores
    )


__all__ = [
    "RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2",
    "RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3",
    "RasterTextEvaluationCaseV1",
    "RasterTextGateScoreV1",
    "RasterTextGoldArtifactV1",
    "RasterTextMetricCountsV1",
    "canonicalize_synthetic_raster_semantic_text_v2",
    "canonicalize_synthetic_raster_semantic_text_v3",
    "evaluate_synthetic_raster_semantic_text_case_v2",
    "evaluate_synthetic_raster_semantic_text_case_v3",
    "evaluate_synthetic_raster_semantic_text_gates_v2",
    "evaluate_synthetic_raster_semantic_text_gates_v3",
    "evaluate_synthetic_raster_text_case",
    "evaluate_synthetic_raster_text_gates",
    "source_only_text_gold_from_raster_gold",
    "synthetic_raster_semantic_text_gate_passes_v2",
    "synthetic_raster_semantic_text_gate_passes_v3",
    "synthetic_raster_semantic_text_preview_draft_gate_passes_v1",
    "synthetic_raster_text_gate_passes",
]
