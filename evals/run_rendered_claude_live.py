"""Bounded Claude Vision + independently selected OCR extraction evaluation.

This evaluator scores only semantic extraction over immutable rendered Gold.
It does not run retrieval, validation, compliance, persistence, or a system
workflow. Reports intentionally omit source text, extracted values, paths,
image payloads, prompts, secrets, and raw provider responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, cast

from customs_ai_agent.adapters import (
    ClaudeDocumentAnalyzer,
    OcrLayoutProvider,
    build_ocr_layout_provider,
    image_page_from_file,
)
from customs_ai_agent.adapters.llm import (
    ClaudeSettings,
    CropReadBudget,
    DocumentAnalysis,
    LLMError,
    ModelCallMetadata,
)
from customs_ai_agent.ocr_release import (
    OCR_RELEASE_DOCUMENT_COUNT,
    OCR_RELEASE_SAMPLE_DOCUMENT_COUNTS,
)
from customs_ai_agent.ocr_identity import (
    OCR_MODEL_ASSETS_LOCK_SHA256,
    OCR_MODEL_SET_IDS,
)
from customs_ai_agent.prompts import (
    DOCUMENT_CROP_RECONCILIATION_PROMPT_ID,
    DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION,
    DOCUMENT_VISION_ANALYSIS_PROMPT_ID,
    DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION,
    load_document_crop_reconciliation_system_prompt,
    load_document_vision_analysis_system_prompt,
)
from customs_ai_agent.schemas.llm import (
    ClaudeCropReadCandidate,
    ClaudeVisionDocumentCandidate,
    ClaudeVisionEvidence,
    ClaudeVisionFieldCandidate,
)
from customs_ai_agent.schemas.models import OcrLayoutLedger
from evals.ocr_release_gate_attestation import (
    EnvironmentAttestor,
    ModelAssetsAttestor,
    OcrReleaseGateAttestation,
    OcrReleaseGateAttestationError,
    ReleaseLockedOcrProvider,
    installed_ocr_environment_lock_sha256,
    load_canonical_ocr_release_gate,
    load_ocr_release_runtime_gate,
    verify_ocr_model_assets,
)
from evals.run_ocr_benchmark import (
    BenchmarkDocument,
    OcrBenchmarkError,
    WheelAttestor,
    installed_agent_wheel_sha256,
    load_benchmark,
)

REPORT_SCHEMA = "rendered-claude-extraction-evaluation-report/v1"
SCOPE = "RENDERED_GOLD_EXTRACTION_ONLY_NOT_WORKFLOW_OR_SYSTEM_RELEASE"
SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
ProviderName = Literal["paddle", "docling"]
ProviderLock: TypeAlias = OcrReleaseGateAttestation
LOCKED_DOCUMENT_COUNT = OCR_RELEASE_SAMPLE_DOCUMENT_COUNTS["locked"]
CO_REGRESSION_DOCUMENT_COUNT = OCR_RELEASE_SAMPLE_DOCUMENT_COUNTS["co_regression"]
RELEASE_DOCUMENT_COUNT = OCR_RELEASE_DOCUMENT_COUNT

# These release gates are code-owned. No descriptor or command-line option may
# weaken them.
EXTRACTION_THRESHOLDS: dict[str, int | float] = {
    "document_type_exact_accuracy_min": 1.0,
    "critical_identifier_exact_accuracy_min": 1.0,
    "required_field_exact_accuracy_min": 0.99,
    "hallucinated_required_field_count_max": 0,
    "ocr_grounding_failure_count_max": 0,
}

DOCUMENT_TYPE_MAP = {
    "commercial_invoice": "invoice",
    "packing_list": "packing_list",
    "bill_of_lading": "bill_of_lading",
    "certificate_of_origin": "certificate_of_origin",
}
REQUIRED_DOCUMENT_FIELDS: dict[str, frozenset[str]] = {
    "invoice": frozenset({"invoice_no", "seller", "buyer", "currency", "total_amount"}),
    "packing_list": frozenset({"packing_no", "seller", "buyer", "packages"}),
    "bill_of_lading": frozenset({"bl_no", "shipper", "consignee", "packages", "gross_weight_kg"}),
    "certificate_of_origin": frozenset(
        {
            "certificate_no",
            "exporter",
            "importer",
            "origin_country",
            "hs_codes",
            "goods_description",
        }
    ),
}
REQUIRED_LINE_FIELDS: dict[str, frozenset[str]] = {
    "invoice": frozenset({"description", "hs_code", "quantity", "amount"}),
    "packing_list": frozenset({"description", "quantity", "gross_weight_kg", "package_count"}),
}
CRITICAL_FIELD_NAMES = frozenset(
    {
        "invoice_no",
        "bl_no",
        "certificate_no",
        "container_numbers",
        "seal_numbers",
        "hs_codes",
    }
)


class Analyzer(Protocol):
    def analyze(self, document: Mapping[str, Any]) -> DocumentAnalysis: ...


ProviderFactory = Callable[[ProviderName, Path], OcrLayoutProvider]
AnalyzerFactory = Callable[[ClaudeSettings, OcrLayoutProvider], Analyzer]


class RenderedClaudeEvaluationError(RuntimeError):
    """Content-free evaluation error represented by one stable safe code."""

    def __init__(self, code: str) -> None:
        if SAFE_CODE.fullmatch(code) is None:
            code = "rendered_evaluation_internal_error"
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ExtractionGold:
    document_type: str
    document_fields: dict[str, str]
    line_items: tuple[dict[str, str], ...]

    @property
    def critical_values(self) -> dict[str, str]:
        values = {
            name: value
            for name, value in self.document_fields.items()
            if name in CRITICAL_FIELD_NAMES
        }
        for item in self.line_items:
            values.update(
                {name: value for name, value in item.items() if name in CRITICAL_FIELD_NAMES}
            )
        return values


@dataclass
class FieldCounts:
    expected: int = 0
    exact: int = 0
    wrong: int = 0
    missing: int = 0
    hallucinated_required: int = 0

    def add_expected(
        self,
        actual: Mapping[str, str],
        expected: Mapping[str, str],
        *,
        required_names: frozenset[str],
    ) -> None:
        self.expected += len(expected)
        for name, value in expected.items():
            if name not in actual:
                self.missing += 1
            elif actual[name] == value:
                self.exact += 1
            else:
                self.wrong += 1
        self.hallucinated_required += sum(
            name in required_names and name not in expected for name in actual
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RenderedClaudeEvaluationError("evaluation_file_unreadable") from exc
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


VISION_SCHEMA_SHA256 = _canonical_sha256(ClaudeVisionDocumentCandidate.model_json_schema())
CROP_SCHEMA_SHA256 = _canonical_sha256(ClaudeCropReadCandidate.model_json_schema())
VISION_PROMPT_SHA256 = _text_sha256(load_document_vision_analysis_system_prompt())
CROP_PROMPT_SHA256 = _text_sha256(load_document_crop_reconciliation_system_prompt())


def _regular_file(path: Path, code: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RenderedClaudeEvaluationError(code) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise RenderedClaudeEvaluationError(code)
    return resolved


def _safe_failure_report(
    code: str,
    *,
    provider_created: bool = False,
    provider_called: bool = False,
) -> dict[str, Any]:
    safe_code = (
        code if SAFE_CODE.fullmatch(code) is not None else ("rendered_evaluation_internal_error")
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "scope": SCOPE,
        "status": "FAIL",
        "release_score_eligible": False,
        "ocr_provider_created": provider_created,
        "anthropic_provider_called": provider_called,
        "call_accounting": {
            "attempted": 0,
            "succeeded": 0,
            "retried": 0,
        },
        "cost_budget": {"configured_max_retries": 0},
        "failures": [{"code": safe_code}],
        "thresholds": EXTRACTION_THRESHOLDS,
        "claim_limits": [
            "extraction_only",
            "no_retrieval",
            "no_compliance_decision",
            "no_operational_database",
            "not_system_release_evidence",
        ],
        "redaction": _redaction_manifest(),
    }


def _score_status(*, passed: bool, release_score_eligible: bool) -> str:
    if not passed:
        return "FAIL"
    return "PASS" if release_score_eligible else "DIAGNOSTIC_PASS"


def _redaction_manifest() -> dict[str, str]:
    return {
        "api_key": "OMITTED",
        "source_text": "OMITTED",
        "extracted_values": "OMITTED",
        "filesystem_paths": "OMITTED",
        "image_payloads": "OMITTED",
        "base64": "OMITTED",
        "prompts": "OMITTED",
        "raw_responses": "OMITTED",
        "provider_exception_messages": "OMITTED",
    }


def _field_values(layout: Mapping[str, Any]) -> dict[str, str]:
    fields = layout.get("fields")
    if not isinstance(fields, list):
        raise RenderedClaudeEvaluationError("extraction_gold_invalid")
    result: dict[str, str] = {}
    for field in fields:
        if not isinstance(field, dict) or not isinstance(field.get("field_id"), str):
            raise RenderedClaudeEvaluationError("extraction_gold_invalid")
        if field.get("rendered") is False:
            continue
        if not isinstance(field.get("raw_value"), str) or not field["raw_value"]:
            raise RenderedClaudeEvaluationError("extraction_gold_invalid")
        result[field["field_id"]] = field["raw_value"]
    return result


def _find_value(values: Mapping[str, str], marker: str) -> str | None:
    matches = [value for field_id, value in values.items() if marker in field_id]
    if len(matches) > 1:
        raise RenderedClaudeEvaluationError("extraction_gold_ambiguous")
    return matches[0] if matches else None


def _split(value: str, separator: str) -> tuple[str, str]:
    left, found, right = value.partition(separator)
    if not found or not left.strip() or not right.strip():
        raise RenderedClaudeEvaluationError("extraction_gold_invalid")
    return left.strip(), right.strip()


def _split_last(value: str) -> tuple[str, str]:
    parts = value.rsplit(maxsplit=1)
    if len(parts) != 2:
        raise RenderedClaudeEvaluationError("extraction_gold_invalid")
    return parts[0], parts[1]


def _put(
    target: dict[str, str],
    name: str,
    value: str | None,
) -> None:
    if value is None:
        return
    if not value or name in target:
        raise RenderedClaudeEvaluationError("extraction_gold_invalid")
    target[name] = value


def _invoice_gold(values: Mapping[str, str]) -> ExtractionGold:
    fields: dict[str, str] = {}
    combined = _find_value(values, "/Invoice No. and Date")
    if combined:
        invoice_no, invoice_date = _split(combined, " / ")
        _put(fields, "invoice_no", invoice_no)
        _put(fields, "invoice_date", invoice_date)
    vessel = _find_value(values, "/Vessel ~1 Flight")
    if vessel:
        vessel_name, voyage = _split(vessel, " / ")
        _put(fields, "vessel", vessel_name)
        _put(fields, "voyage_no", voyage)
    amount = _find_value(values, "/Amount")
    if amount:
        currency, total = _split(amount, " ")
        _put(fields, "currency", currency)
        _put(fields, "total_amount", total)
    terms = _find_value(values, "/Terms of Delivery and Payment")
    if terms:
        incoterms, payment = _split(terms, " / ")
        _put(fields, "incoterms", incoterms)
        _put(fields, "payment_terms", payment)
    origin = _find_value(values, "/Other References (Country of Origin)")
    if origin:
        _, country = _split(origin, ": ")
        _put(fields, "country_of_origin", country)
    for marker, name in (
        ("/Buyer (if other than consignee)", "buyer"),
        ("/Consignee (or For account & risk of Messrs)", "consignee"),
        ("/Departure Date", "departure_date"),
        ("/From", "port_of_loading"),
        ("/No. & Kind of Packages", "packages"),
        ("/Shipper ~1 Seller", "seller"),
        ("/Shipping Marks", "shipping_marks"),
        ("/To", "port_of_discharge"),
    ):
        _put(fields, name, _find_value(values, marker))
    line: dict[str, str] = {}
    for marker, name in (
        ("/Goods Description", "description"),
        ("/Unit Price", "unit_price"),
    ):
        _put(line, name, _find_value(values, marker))
    quantity = _find_value(values, "/Quantity")
    if quantity:
        number, unit = _split_last(quantity)
        _put(line, "quantity", number)
        _put(line, "unit", unit)
    if amount:
        _, line_amount = _split(amount, " ")
        _put(line, "amount", line_amount)
    return ExtractionGold("invoice", fields, (line,) if line else ())


def _packing_list_gold(values: Mapping[str, str]) -> ExtractionGold:
    fields: dict[str, str] = {}
    combined = _find_value(values, "/Invoice No. and Date")
    if combined:
        invoice_no, invoice_date = _split(combined, " / ")
        _put(fields, "invoice_no", invoice_no)
        _put(fields, "invoice_date", invoice_date)
    vessel = _find_value(values, "/Vessel ~1 Flight")
    if vessel:
        vessel_name, voyage = _split(vessel, " / ")
        _put(fields, "vessel", vessel_name)
        _put(fields, "voyage_no", voyage)
    for marker, name in (
        ("/Buyer (if other than consignee)", "buyer"),
        ("/Consignee", "consignee"),
        ("/Departure Date", "departure_date"),
        ("/From", "port_of_loading"),
        ("/Gross Weight", "total_gross_weight_kg"),
        ("/Measurement", "total_measurement_cbm"),
        ("/No. & Kind of Packages", "packages"),
        ("/Seller", "seller"),
        ("/Shipping Marks", "shipping_marks"),
        ("/To", "port_of_discharge"),
    ):
        _put(fields, name, _find_value(values, marker))
    line_pattern = re.compile(r"/Goods Details/(\d+)/(item|quantity|weight)$")
    lines: dict[int, dict[str, str]] = {}
    for field_id, value in values.items():
        match = line_pattern.search(field_id)
        if match is None:
            continue
        index = int(match.group(1))
        name = {
            "item": "description",
            "quantity": "quantity",
            "weight": "gross_weight_kg",
        }[match.group(2)]
        _put(lines.setdefault(index, {}), name, value)
    ordered = tuple(lines[index] for index in sorted(lines))
    return ExtractionGold("packing_list", fields, ordered)


def _bill_of_lading_gold(values: Mapping[str, str]) -> ExtractionGold:
    fields: dict[str, str] = {}
    seal_marks = _find_value(values, "/Seal No. ~1 Marks & No.")
    if seal_marks:
        seal, marks = _split(seal_marks, " // ")
        _put(fields, "seal_numbers", seal)
        _put(fields, "shipping_marks", marks)
    on_board = _find_value(values, "/Laden on Board (Date ~1 Signature)")
    if on_board:
        date, _ = _split(on_board, " / ")
        _put(fields, "on_board_date", date)
    for marker, name in (
        ("/B~1L No. (Bill of Lading Number)", "bl_no"),
        ("/Consignee", "consignee"),
        ("/Container No.", "container_numbers"),
        ("/Gross Weight", "gross_weight_kg"),
        ("/No. & Kinds of Containers or Packages", "packages"),
        ("/Notify Party", "notify_party"),
        ("/Ocean Vessel", "vessel"),
        ("/Place of Delivery", "place_of_delivery"),
        ("/Port of Discharge", "port_of_discharge"),
        ("/Port of Loading", "port_of_loading"),
        ("/Shipper ~1 Exporter", "shipper"),
        ("/Voyage No.", "voyage_no"),
    ):
        _put(fields, name, _find_value(values, marker))
    return ExtractionGold("bill_of_lading", fields, ())


def _certificate_gold(values: Mapping[str, str]) -> ExtractionGold:
    fields: dict[str, str] = {}
    for marker, name in (
        ("/Certificate No.", "certificate_no"),
        ("/Consignee", "importer"),
        ("/Country of Origin", "origin_country"),
        ("/Exporter", "exporter"),
        ("/Goods Details/0/hs_code", "hs_codes"),
        ("/Goods Details/0/item", "goods_description"),
        ("/Issue Date", "issue_date"),
    ):
        _put(fields, name, _find_value(values, marker))
    return ExtractionGold("certificate_of_origin", fields, ())


def _build_gold(layout: Mapping[str, Any]) -> ExtractionGold:
    document_type = layout.get("document_type")
    if document_type not in DOCUMENT_TYPE_MAP:
        raise RenderedClaudeEvaluationError("extraction_gold_invalid")
    values = _field_values(layout)
    builders: dict[str, Callable[[Mapping[str, str]], ExtractionGold]] = {
        "commercial_invoice": _invoice_gold,
        "packing_list": _packing_list_gold,
        "bill_of_lading": _bill_of_lading_gold,
        "certificate_of_origin": _certificate_gold,
    }
    gold = builders[cast(str, document_type)](values)
    if (
        gold.document_type != DOCUMENT_TYPE_MAP[document_type]
        or not gold.document_fields
        or not gold.critical_values
    ):
        raise RenderedClaudeEvaluationError("extraction_gold_invalid")
    return gold


def _estimated_input_tokens(documents: Sequence[BenchmarkDocument]) -> int:
    prompt_characters = len(load_document_vision_analysis_system_prompt())
    total = 0
    for document in documents:
        page = document.layout.get("page")
        words = document.layout.get("words")
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("claude_input_visual_tokens"), int)
            or not isinstance(words, list)
        ):
            raise RenderedClaudeEvaluationError("extraction_gold_invalid")
        text_characters = sum(
            len(str(word.get("text") or "")) for word in words if isinstance(word, dict)
        )
        # The fixed overhead covers the JSON envelope, OCR span identifiers,
        # coordinate fields, and provider message framing without exposing
        # those contents in the report.
        total += (
            int(page["claude_input_visual_tokens"])
            + (text_characters + prompt_characters + 2) // 3
            + 2_048
        )
    return total


def _default_provider_factory(
    provider: ProviderName,
    executable: Path,
) -> OcrLayoutProvider:
    return build_ocr_layout_provider(
        provider,
        python_executable=executable,
    )


def _default_analyzer_factory(
    settings: ClaudeSettings,
    provider: OcrLayoutProvider,
) -> Analyzer:
    return ClaudeDocumentAnalyzer.from_settings(
        settings,
        ocr_layout_provider=provider,
    )


def _live_settings(env_file: Path) -> ClaudeSettings:
    """Load the pinned configuration and disable all SDK-level retries."""

    configured = ClaudeSettings.from_environment(env_file)
    return ClaudeSettings(
        api_key=configured.api_key,
        model=configured.model,
        max_tokens=configured.max_tokens,
        timeout_seconds=configured.timeout_seconds,
        max_retries=0,
    )


def _safe_error_code(error: LLMError) -> str:
    return error.code if SAFE_CODE.fullmatch(error.code) else "llm_provider_error"


def _metadata_is_locked(
    metadata: ModelCallMetadata,
    *,
    call_kind: Literal["primary", "crop"],
    settings: ClaudeSettings,
    lock: ProviderLock,
) -> bool:
    if call_kind == "primary":
        expected = (
            DOCUMENT_VISION_ANALYSIS_PROMPT_ID,
            DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION,
            VISION_PROMPT_SHA256,
            "customs_claude_vision_document_candidate",
            VISION_SCHEMA_SHA256,
        )
    else:
        expected = (
            DOCUMENT_CROP_RECONCILIATION_PROMPT_ID,
            DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION,
            CROP_PROMPT_SHA256,
            "customs_claude_crop_read_candidate",
            CROP_SCHEMA_SHA256,
        )
    return (
        metadata.provider == "anthropic"
        and metadata.model_id == settings.model
        and metadata.prompt_id == expected[0]
        and metadata.prompt_version == expected[1]
        and metadata.prompt_sha256 == expected[2]
        and metadata.output_schema_id == expected[3]
        and metadata.output_schema_version == "1.0"
        and metadata.output_schema_sha256 == expected[4]
        and metadata.input_modality == ("image" if call_kind == "primary" else "image_crop")
        and metadata.ocr_provider_id == lock.provider_id
        and metadata.ocr_engine_id == lock.engine_id
        and metadata.ocr_engine_version == lock.engine_version
        and metadata.ocr_model_id == OCR_MODEL_SET_IDS[lock.provider]
        and metadata.ocr_model_assets_lock_sha256 == OCR_MODEL_ASSETS_LOCK_SHA256
    )


def _ledger_is_locked(
    ledger: OcrLayoutLedger | None,
    *,
    document_id: str,
    image_sha256: str,
    lock: ProviderLock,
) -> bool:
    if (
        ledger is None
        or ledger.document_id != document_id
        or ledger.provider_id != lock.provider_id
        or ledger.engine_id != lock.engine_id
        or ledger.engine_version != lock.engine_version
        or ledger.model_id != OCR_MODEL_SET_IDS[lock.provider]
        or ledger.model_assets_lock_sha256 != OCR_MODEL_ASSETS_LOCK_SHA256
        or len(ledger.pages) != 1
        or ledger.pages[0].image_sha256 != image_sha256
    ):
        return False
    return True


def _evidence_grounded(
    evidence: ClaudeVisionEvidence,
    *,
    value: str | None,
    ledger: OcrLayoutLedger,
) -> bool:
    spans = {span.span_id: span for page in ledger.pages for span in page.spans}
    if any(span_id not in spans for span_id in evidence.ocr_span_ids):
        return False
    selected = [spans[span_id] for span_id in evidence.ocr_span_ids]
    selected.sort(key=lambda span: (span.page, span.reading_order))
    combined = " ".join(span.text for span in selected)
    if value is None:
        return bool(evidence.source_text.strip()) and evidence.source_text in combined
    return value in evidence.source_text and value in combined


def _candidate_grounding_failures(
    candidate: ClaudeVisionDocumentCandidate,
    ledger: OcrLayoutLedger,
) -> int:
    failures = 0
    if candidate.classification_evidence is None or not _evidence_grounded(
        candidate.classification_evidence,
        value=None,
        ledger=ledger,
    ):
        failures += 1
    for field in candidate.fields:
        failures += int(
            not _evidence_grounded(
                field.evidence,
                value=field.value,
                ledger=ledger,
            )
        )
    for item in candidate.line_items:
        failures += int(
            not _evidence_grounded(
                item.row_evidence,
                value=None,
                ledger=ledger,
            )
        )
        for field in item.fields:
            failures += int(
                not _evidence_grounded(
                    field.evidence,
                    value=field.value,
                    ledger=ledger,
                )
            )
    return failures


def _field_map(
    fields: Sequence[ClaudeVisionFieldCandidate],
) -> dict[str, str]:
    return {field.name: field.value for field in fields}


def _score_candidate(
    candidate: ClaudeVisionDocumentCandidate,
    gold: ExtractionGold,
    counts: FieldCounts,
) -> tuple[int, int, int]:
    document_type_correct = int(candidate.document_type == gold.document_type)
    actual_document = _field_map(candidate.fields)
    counts.add_expected(
        actual_document,
        gold.document_fields,
        required_names=REQUIRED_DOCUMENT_FIELDS[gold.document_type],
    )
    actual_lines = [_field_map(item.fields) for item in candidate.line_items]
    required_line_names = REQUIRED_LINE_FIELDS.get(
        gold.document_type,
        frozenset(),
    )
    for index, expected in enumerate(gold.line_items):
        actual = actual_lines[index] if index < len(actual_lines) else {}
        counts.add_expected(
            actual,
            expected,
            required_names=required_line_names,
        )
    for actual in actual_lines[len(gold.line_items) :]:
        counts.hallucinated_required += sum(name in required_line_names for name in actual)
    critical_expected = gold.critical_values
    actual_all = dict(actual_document)
    for line in actual_lines:
        for name, value in line.items():
            if name in CRITICAL_FIELD_NAMES:
                actual_all[name] = value
    critical_exact = sum(actual_all.get(name) == value for name, value in critical_expected.items())
    return document_type_correct, critical_exact, len(critical_expected)


def _safe_call_record(
    metadata: ModelCallMetadata,
    *,
    call_kind: Literal["primary", "crop"],
    document_reference: str,
) -> dict[str, Any]:
    return {
        "document_reference": document_reference,
        "call_kind": call_kind,
        "provider": metadata.provider,
        "model_id": metadata.model_id,
        "prompt_id": metadata.prompt_id,
        "prompt_version": metadata.prompt_version,
        "prompt_sha256": metadata.prompt_sha256,
        "output_schema_id": metadata.output_schema_id,
        "output_schema_version": metadata.output_schema_version,
        "output_schema_sha256": metadata.output_schema_sha256,
        "input_modality": metadata.input_modality,
        "input_tokens": metadata.input_tokens,
        "output_tokens": metadata.output_tokens,
        "cache_creation_input_tokens": metadata.cache_creation_input_tokens,
        "cache_read_input_tokens": metadata.cache_read_input_tokens,
        "latency_ms": metadata.latency_ms,
        "ocr_provider_id": metadata.ocr_provider_id,
        "ocr_engine_id": metadata.ocr_engine_id,
        "ocr_engine_version": metadata.ocr_engine_version,
        "ocr_model_id": metadata.ocr_model_id,
        "ocr_model_assets_lock_sha256": metadata.ocr_model_assets_lock_sha256,
    }


def evaluate(
    *,
    env_file: Path,
    descriptor_path: Path,
    rendered_root: Path,
    ocr_release_gate_report: Path,
    ocr_release_gate_report_sha256: str,
    expected_agent_wheel_sha256: str,
    ocr_environment_lock: Path,
    ocr_environment_lock_sha256: str,
    ocr_model_assets_lock: Path,
    ocr_model_assets_lock_sha256: str,
    ocr_provider: ProviderName,
    ocr_python: Path,
    max_documents: int,
    max_calls: int,
    max_total_input_tokens: int,
    max_total_output_tokens: int,
    live: bool,
    release_mode: bool = False,
    provider_factory: ProviderFactory = _default_provider_factory,
    analyzer_factory: AnalyzerFactory = _default_analyzer_factory,
    wheel_attestor: WheelAttestor = installed_agent_wheel_sha256,
    environment_attestor: EnvironmentAttestor = installed_ocr_environment_lock_sha256,
    model_assets_attestor: ModelAssetsAttestor = verify_ocr_model_assets,
) -> tuple[int, dict[str, Any]]:
    """Evaluate a diagnostic prefix or the exact locked+C/O release denominator."""

    try:
        if (
            max_documents <= 0
            or max_calls <= 0
            or max_total_input_tokens <= 0
            or max_total_output_tokens <= 0
        ):
            raise RenderedClaudeEvaluationError("evaluation_budget_invalid")
        try:
            descriptor_sha, manifest_sha, locked_documents, _ = load_benchmark(
                descriptor_path=descriptor_path,
                rendered_root=rendered_root,
                sample="locked",
            )
        except OcrBenchmarkError as exc:
            raise RenderedClaudeEvaluationError(exc.code) from exc
        if len(locked_documents) != LOCKED_DOCUMENT_COUNT:
            raise RenderedClaudeEvaluationError("locked_release_denominator_mismatch")
        co_documents: list[BenchmarkDocument] = []
        if release_mode:
            try:
                co_descriptor_sha, co_manifest_sha, co_documents, _ = load_benchmark(
                    descriptor_path=descriptor_path,
                    rendered_root=rendered_root,
                    sample="co_regression",
                )
            except OcrBenchmarkError as exc:
                raise RenderedClaudeEvaluationError("co_regression_release_sample_invalid") from exc
            if (
                co_descriptor_sha != descriptor_sha
                or co_manifest_sha != manifest_sha
                or len(co_documents) != CO_REGRESSION_DOCUMENT_COUNT
                or len(
                    {
                        document.benchmark_document_id
                        for document in [*locked_documents, *co_documents]
                    }
                )
                != RELEASE_DOCUMENT_COUNT
            ):
                raise RenderedClaudeEvaluationError("release_denominator_mismatch")
            if max_documents != RELEASE_DOCUMENT_COUNT:
                raise RenderedClaudeEvaluationError("release_requires_exact_80_documents")
            all_documents = [*locked_documents, *co_documents]
        else:
            if max_documents > len(locked_documents):
                raise RenderedClaudeEvaluationError("max_documents_exceeds_locked_sample")
            all_documents = locked_documents
        # Gold derivation is intentionally completed for the entire locked set,
        # even when this cost-bounded run selects only a prefix.
        all_gold = [_build_gold(document.layout) for document in all_documents]
        executable = _regular_file(ocr_python, "ocr_python_unavailable")
        try:
            if release_mode:
                lock = load_canonical_ocr_release_gate(
                    report_path=ocr_release_gate_report,
                    expected_report_sha256=ocr_release_gate_report_sha256,
                    expected_provider=ocr_provider,
                    descriptor_path=descriptor_path,
                    rendered_root=rendered_root,
                    expected_agent_wheel_sha256=expected_agent_wheel_sha256,
                    environment_lock_path=ocr_environment_lock,
                    expected_environment_lock_sha256=ocr_environment_lock_sha256,
                    model_assets_lock_path=ocr_model_assets_lock,
                    expected_model_assets_lock_sha256=ocr_model_assets_lock_sha256,
                    ocr_python=executable,
                    wheel_attestor=wheel_attestor,
                    environment_attestor=environment_attestor,
                    model_assets_attestor=model_assets_attestor,
                )
            else:
                lock = load_ocr_release_runtime_gate(
                    report_path=ocr_release_gate_report,
                    expected_report_sha256=ocr_release_gate_report_sha256,
                    expected_provider=ocr_provider,
                    expected_descriptor_sha256=descriptor_sha,
                    expected_manifest_sha256=manifest_sha,
                    expected_agent_wheel_sha256=expected_agent_wheel_sha256,
                    environment_lock_path=ocr_environment_lock,
                    expected_environment_lock_sha256=ocr_environment_lock_sha256,
                    model_assets_lock_path=ocr_model_assets_lock,
                    expected_model_assets_lock_sha256=ocr_model_assets_lock_sha256,
                    ocr_python=executable,
                    wheel_attestor=wheel_attestor,
                    environment_attestor=environment_attestor,
                    model_assets_attestor=model_assets_attestor,
                )
        except OcrReleaseGateAttestationError as exc:
            raise RenderedClaudeEvaluationError(exc.code) from exc
        executable_sha256 = _sha256(executable)
        environment = _regular_file(env_file, "claude_environment_unavailable")
        selected_documents = all_documents if release_mode else all_documents[:max_documents]
        selected_gold = all_gold if release_mode else all_gold[:max_documents]
        estimated_input_tokens = _estimated_input_tokens(selected_documents)
        preflight_failures: list[str] = []
        if not live:
            preflight_failures.append("live_flag_required")
        if max_calls < max_documents:
            preflight_failures.append("max_calls_budget_too_small")
        if max_total_input_tokens < estimated_input_tokens:
            preflight_failures.append("input_token_budget_too_small")
        if preflight_failures:
            report = _safe_failure_report(sorted(preflight_failures)[0])
            report["failures"] = [{"code": code} for code in sorted(preflight_failures)]
            report["cost_budget"] = {
                "max_documents": max_documents,
                "max_calls": max_calls,
                "max_total_input_tokens": max_total_input_tokens,
                "max_total_output_tokens": max_total_output_tokens,
                "estimated_input_tokens": estimated_input_tokens,
                "configured_max_retries": 0,
            }
            return 1, report
        try:
            settings = _live_settings(environment)
        except LLMError as exc:
            return 1, _safe_failure_report(_safe_error_code(exc))
        if settings.max_tokens * max_calls > max_total_output_tokens:
            report = _safe_failure_report("configured_output_ceiling_exceeds_budget")
            report["cost_budget"] = {
                "max_documents": max_documents,
                "max_calls": max_calls,
                "max_total_input_tokens": max_total_input_tokens,
                "max_total_output_tokens": max_total_output_tokens,
                "estimated_input_tokens": estimated_input_tokens,
                "configured_max_output_tokens_per_call": settings.max_tokens,
                "configured_max_retries": settings.max_retries,
            }
            return 1, report
    except RenderedClaudeEvaluationError as exc:
        return 1, _safe_failure_report(exc.code)

    provider: OcrLayoutProvider | None = None
    provider_created = False
    calls: list[dict[str, Any]] = []
    observed_call_count = 0
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "latency_ms": 0,
    }
    failures: list[dict[str, str]] = []
    counts = FieldCounts()
    document_type_correct = 0
    critical_exact = 0
    critical_expected = 0
    grounding_failures = 0
    documents_completed = 0
    analysis_attempted = False
    main_calls_attempted = 0
    crop_budget = CropReadBudget(max_calls=max_calls - max_documents)
    try:
        try:
            provider = ReleaseLockedOcrProvider(
                provider_factory(ocr_provider, executable),
                lock,
            )
            provider_created = True
        except Exception:
            return 1, _safe_failure_report("ocr_provider_initialization_failed")
        try:
            analyzer = analyzer_factory(settings, provider)
        except Exception:
            return 1, _safe_failure_report(
                "claude_analyzer_initialization_failed",
                provider_created=True,
            )

        for document, gold in zip(
            selected_documents,
            selected_gold,
            strict=True,
        ):
            document_reference = hashlib.sha256(
                document.benchmark_document_id.encode("utf-8")
            ).hexdigest()
            try:
                page = image_page_from_file(document.image_path)
                if main_calls_attempted + crop_budget.used_calls >= max_calls:
                    failures.append(
                        {
                            "document_reference": "budget",
                            "code": "call_budget_exceeded",
                        }
                    )
                    break
                main_calls_attempted += 1
                analysis_attempted = True
                analysis = analyzer.analyze(
                    {
                        "document_id": cast(str, document.layout["document_id"]),
                        "filename": f"{document_reference}.png",
                        "declared_type": None,
                        "image_pages": [page],
                        "_runtime_crop_budget": crop_budget,
                    }
                )
            except LLMError as exc:
                failures.append(
                    {
                        "document_reference": document_reference,
                        "code": _safe_error_code(exc),
                    }
                )
                continue
            except Exception:
                failures.append(
                    {
                        "document_reference": document_reference,
                        "code": "document_analysis_failed",
                    }
                )
                continue
            candidate = analysis.candidate
            ledger = analysis.ocr_layout_ledger
            if not isinstance(candidate, ClaudeVisionDocumentCandidate):
                failures.append(
                    {
                        "document_reference": document_reference,
                        "code": "candidate_modality_mismatch",
                    }
                )
                continue
            if not _ledger_is_locked(
                ledger,
                document_id=cast(str, document.layout["document_id"]),
                image_sha256=document.image_sha256,
                lock=lock,
            ):
                failures.append(
                    {
                        "document_reference": document_reference,
                        "code": "ocr_ledger_lock_mismatch",
                    }
                )
                continue
            locked_ledger = cast(OcrLayoutLedger, ledger)
            call_metadata = (
                ("primary", analysis.metadata),
                *(("crop", metadata) for metadata in analysis.additional_model_calls),
            )
            observed_call_count += len(call_metadata)
            for _, metadata in call_metadata:
                usage["input_tokens"] += metadata.input_tokens
                usage["output_tokens"] += metadata.output_tokens
                usage["cache_creation_input_tokens"] += metadata.cache_creation_input_tokens
                usage["cache_read_input_tokens"] += metadata.cache_read_input_tokens
                usage["latency_ms"] += metadata.latency_ms
            if any(
                not _metadata_is_locked(
                    metadata,
                    call_kind=cast(Literal["primary", "crop"], kind),
                    settings=settings,
                    lock=lock,
                )
                for kind, metadata in call_metadata
            ):
                failures.append(
                    {
                        "document_reference": document_reference,
                        "code": "model_metadata_lock_mismatch",
                    }
                )
                continue
            calls.extend(
                _safe_call_record(
                    metadata,
                    call_kind=cast(Literal["primary", "crop"], kind),
                    document_reference=document_reference,
                )
                for kind, metadata in call_metadata
            )
            grounding_failures += _candidate_grounding_failures(
                candidate,
                locked_ledger,
            )
            type_score, critical_score, critical_total = _score_candidate(
                candidate,
                gold,
                counts,
            )
            document_type_correct += type_score
            critical_exact += critical_score
            critical_expected += critical_total
            documents_completed += 1
    finally:
        if provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    failures.append(
                        {
                            "document_reference": "provider",
                            "code": "ocr_provider_close_failed",
                        }
                    )

    total_calls_attempted = main_calls_attempted + crop_budget.used_calls
    if total_calls_attempted > max_calls:
        failures.append({"document_reference": "budget", "code": "call_budget_exceeded"})
    if usage["input_tokens"] > max_total_input_tokens:
        failures.append(
            {
                "document_reference": "budget",
                "code": "input_token_budget_exceeded",
            }
        )
    if usage["output_tokens"] > max_total_output_tokens:
        failures.append(
            {
                "document_reference": "budget",
                "code": "output_token_budget_exceeded",
            }
        )
    document_type_accuracy = document_type_correct / max_documents if max_documents else 0.0
    critical_accuracy = critical_exact / critical_expected if critical_expected else 0.0
    required_accuracy = counts.exact / counts.expected if counts.expected else 0.0
    checks = {
        "all_documents_completed": documents_completed == max_documents,
        "document_type_exact_accuracy": (
            document_type_accuracy >= EXTRACTION_THRESHOLDS["document_type_exact_accuracy_min"]
        ),
        "critical_identifier_exact_accuracy": (
            critical_expected > 0
            and critical_accuracy >= EXTRACTION_THRESHOLDS["critical_identifier_exact_accuracy_min"]
        ),
        "required_field_exact_accuracy": (
            counts.expected > 0
            and required_accuracy >= EXTRACTION_THRESHOLDS["required_field_exact_accuracy_min"]
        ),
        "hallucinated_required_fields": (
            counts.hallucinated_required
            <= EXTRACTION_THRESHOLDS["hallucinated_required_field_count_max"]
        ),
        "ocr_grounding": (
            grounding_failures <= EXTRACTION_THRESHOLDS["ocr_grounding_failure_count_max"]
        ),
        "cost_budget": not any(failure["document_reference"] == "budget" for failure in failures),
        "provider_metadata_locked": (
            observed_call_count == len(calls)
            and all(call["provider"] == "anthropic" for call in calls)
        ),
        "no_failures": not failures,
    }
    passed = all(checks.values())
    status = _score_status(
        passed=passed,
        release_score_eligible=release_mode,
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": SCOPE,
        "status": status,
        "release_score_eligible": release_mode,
        "evaluation_mode": "release" if release_mode else "diagnostic",
        "dataset": {
            "benchmark_id": "rendered-trade-documents-ocr-v2",
            "descriptor_sha256": lock.descriptor_sha256,
            "manifest_sha256": lock.manifest_sha256,
            "locked_document_count": len(locked_documents),
            "co_regression_document_count": len(co_documents),
            "release_document_count": len(all_documents) if release_mode else None,
            "evaluated_document_count": max_documents,
        },
        "ocr": {
            "release_gate_report_sha256": lock.report_sha256,
            "agent_wheel_sha256": lock.agent_wheel_sha256,
            "environment_lock_sha256": lock.environment_lock_sha256,
            "model_assets_lock_sha256": lock.model_assets_lock_sha256,
            "provider": lock.provider,
            "provider_id": lock.provider_id,
            "engine_id": lock.engine_id,
            "engine_version": lock.engine_version,
            "model_set_id": OCR_MODEL_SET_IDS[lock.provider],
            "python_executable_sha256": executable_sha256,
        },
        "model_contract": {
            "model_id": settings.model,
            "primary_prompt": {
                "id": DOCUMENT_VISION_ANALYSIS_PROMPT_ID,
                "version": DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION,
                "sha256": VISION_PROMPT_SHA256,
            },
            "primary_output_schema": {
                "id": "customs_claude_vision_document_candidate",
                "version": "1.0",
                "sha256": VISION_SCHEMA_SHA256,
            },
            "crop_prompt": {
                "id": DOCUMENT_CROP_RECONCILIATION_PROMPT_ID,
                "version": DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION,
                "sha256": CROP_PROMPT_SHA256,
            },
            "crop_output_schema": {
                "id": "customs_claude_crop_read_candidate",
                "version": "1.0",
                "sha256": CROP_SCHEMA_SHA256,
            },
        },
        "thresholds": EXTRACTION_THRESHOLDS,
        "metrics": {
            "document_type": {
                "correct": document_type_correct,
                "expected": max_documents,
                "exact_accuracy": document_type_accuracy,
            },
            "critical_identifiers": {
                "exact": critical_exact,
                "expected": critical_expected,
                "exact_accuracy": critical_accuracy,
            },
            "required_fields": {
                "exact": counts.exact,
                "expected": counts.expected,
                "wrong": counts.wrong,
                "missing": counts.missing,
                "exact_accuracy": required_accuracy,
            },
            "hallucinated_required_field_count": counts.hallucinated_required,
            "ocr_grounding_failure_count": grounding_failures,
        },
        "checks": checks,
        "cost_budget": {
            "max_documents": max_documents,
            "max_calls": max_calls,
            "max_total_input_tokens": max_total_input_tokens,
            "max_total_output_tokens": max_total_output_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "configured_max_output_tokens_per_call": settings.max_tokens,
            "configured_max_retries": settings.max_retries,
        },
        "call_accounting": {
            "attempted": total_calls_attempted,
            "succeeded": observed_call_count,
            "retried": 0,
            "main_attempted": main_calls_attempted,
            "crop_attempted": crop_budget.used_calls,
        },
        "usage": usage,
        "call_count": total_calls_attempted,
        "locked_call_metadata_count": len(calls),
        "calls": calls,
        "documents_completed": documents_completed,
        "ocr_provider_created": provider_created,
        "anthropic_provider_called": bool(calls) or analysis_attempted,
        "failures": failures,
        "claim_limits": [
            "extraction_only",
            "no_retrieval",
            "no_compliance_decision",
            "no_operational_database",
            "not_system_release_evidence",
        ],
        "redaction": _redaction_manifest(),
    }
    return (0 if passed else 1), report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded Claude Vision extraction over rendered Gold with a "
            "passing combined 60+20 OCR release Gate."
        )
    )
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--rendered-root", required=True, type=Path)
    parser.add_argument("--ocr-release-gate-report", required=True, type=Path)
    parser.add_argument(
        "--ocr-release-gate-report-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-agent-wheel-sha256",
        required=True,
    )
    parser.add_argument("--ocr-environment-lock", required=True, type=Path)
    parser.add_argument("--ocr-environment-lock-sha256", required=True)
    parser.add_argument("--ocr-model-assets-lock", required=True, type=Path)
    parser.add_argument("--ocr-model-assets-lock-sha256", required=True)
    parser.add_argument(
        "--ocr-provider",
        required=True,
        choices=("paddle", "docling"),
    )
    parser.add_argument("--ocr-python", required=True, type=Path)
    parser.add_argument("--max-documents", required=True, type=int)
    parser.add_argument("--max-calls", required=True, type=int)
    parser.add_argument("--max-total-input-tokens", required=True, type=int)
    parser.add_argument("--max-total-output-tokens", required=True, type=int)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to permit OCR provider creation and Anthropic API calls.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help=(
            "Require and score the exact locked 60 + C/O regression 20 denominator. "
            "Without this flag a passing bounded prefix is DIAGNOSTIC_PASS only."
        ),
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    exit_code, report = evaluate(
        env_file=args.env_file,
        descriptor_path=args.descriptor,
        rendered_root=args.rendered_root,
        ocr_release_gate_report=args.ocr_release_gate_report,
        ocr_release_gate_report_sha256=args.ocr_release_gate_report_sha256,
        expected_agent_wheel_sha256=args.expected_agent_wheel_sha256,
        ocr_environment_lock=args.ocr_environment_lock,
        ocr_environment_lock_sha256=args.ocr_environment_lock_sha256,
        ocr_model_assets_lock=args.ocr_model_assets_lock,
        ocr_model_assets_lock_sha256=args.ocr_model_assets_lock_sha256,
        ocr_provider=cast(ProviderName, args.ocr_provider),
        ocr_python=args.ocr_python,
        max_documents=args.max_documents,
        max_calls=args.max_calls,
        max_total_input_tokens=args.max_total_input_tokens,
        max_total_output_tokens=args.max_total_output_tokens,
        live=args.live,
        release_mode=args.release,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
