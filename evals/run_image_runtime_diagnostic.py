"""Opt-in, non-release image service diagnostic.

This runner exercises one hash-pinned CI/PL/B/L triplet through an isolated
evaluation-only composition: independent OCR/layout -> Claude Vision ->
Evidence HTTP. It cannot instantiate the public service ``AgentRuntime`` with
an unreleased OCR provider and is deliberately incapable of producing a
release PASS. Reports are value-free and contain only identities, hashes,
counts, statuses, and safe error codes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from dotenv import dotenv_values
from pydantic import SecretStr

from customs_ai_agent.adapters import (
    ImageInputError,
    OcrLayoutProvider,
    OcrLayoutProviderError,
    build_ocr_layout_provider,
    image_page_from_file,
)
from customs_ai_agent.adapters.llm import (
    CLAUDE_CANDIDATE_SCHEMA_VERSION,
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    LLMError,
)
from customs_ai_agent.adapters.retrieval import (
    HttpRegulationRetriever,
    RetrievalError,
    fetch_supported_review_as_of,
    validate_evidence_destination,
)
from customs_ai_agent.contracts import (
    ContractSchemaError,
    schema_sha256,
    verify_packaged_schemas,
)
from customs_ai_agent.evaluation_assets import (
    image_gold_invalidation_code,
    image_gold_semantic_invalidation_code,
)
from customs_ai_agent.prompts import (
    DOCUMENT_CROP_RECONCILIATION_PROMPT_ID,
    DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION,
    DOCUMENT_VISION_ANALYSIS_PROMPT_ID,
    DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION,
    load_document_crop_reconciliation_system_prompt,
    load_document_vision_analysis_system_prompt,
)
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.schemas.llm import (
    ClaudeCropReadCandidate,
    ClaudeVisionDocumentCandidate,
)
from customs_ai_agent.schemas.models import (
    DocumentInputV2,
    ReviewRunRequestV2,
    ReviewRunResultV2,
)


ProviderName = Literal["paddle", "docling"]
DiagnosticStatus = Literal[
    "PREFLIGHT_BLOCKED",
    "BLOCKED_EVALUATION_ASSET",
    "BLOCKED_DATA_READINESS",
    "DIAGNOSTIC_EXECUTION_FAILED",
    "DIAGNOSTIC_PASS",
]
REPORT_SCHEMA_VERSION = "image-runtime-nonrelease-diagnostic/v1"
DIAGNOSTIC_SCOPE = "NONRELEASE_IMAGE_AGENT_SERVICE_DIAGNOSTIC"
# No replacement Gold is trusted yet. Tests may pin a synthetic fixture, but
# the workspace runner remains blocked until a newly adjudicated artifact is
# explicitly reviewed and pinned here.
EXPECTED_GOLD_SHA256: str | None = None
EXPECTED_GOLD_SET_ID = "trade-images-adjudicated-gold-source-v1"
EXPECTED_SELECTION_ID = "trade-images-hash-ranked-diagnostic-v1"
FIXED_CASE_ID = "0097"
FIXED_TRIPLET_SHA256 = "cf88d63a230e2d73f6687e53fa9e66371a4ea040aca35d0994fed154dd2b33c5"
_SAFE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_EVIDENCE_ENV_KEYS = frozenset(
    {
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        "EVIDENCE_SERVICE_TOKEN",
        "EVIDENCE_SERVICE_TOKEN_HEADER",
    }
)


@dataclass(frozen=True, slots=True)
class FixedDocument:
    document_id: str
    source_document_type: str
    expected_document_type: str
    relative_name: str
    source_sha256: str


FIXED_DOCUMENTS = (
    FixedDocument(
        document_id="0097:invoice",
        source_document_type="commercial_invoice",
        expected_document_type="invoice",
        relative_name="commercial_invoice/CI_0097.jpg",
        source_sha256="bcd15c28f8ce3576d02dd1073d9b4f449f59c6a0be1bb846c919effbf603e57d",
    ),
    FixedDocument(
        document_id="0097:packing_list",
        source_document_type="packing_list",
        expected_document_type="packing_list",
        relative_name="packing_list/PL_0097.jpg",
        source_sha256="d62f33be91e66b6ed1d9a28e5d288f378a712659d42e9f44c75928c44398d57a",
    ),
    FixedDocument(
        document_id="0097:bill_of_lading",
        source_document_type="bill_of_lading",
        expected_document_type="bill_of_lading",
        relative_name="bill_of_lading/BL_0097.jpg",
        source_sha256="e2c9622c3069547a5b1c59a629262b6ababec149a1c0b80aa90ab96d028d50ae",
    ),
)


def _discover_canonical_dataset_root() -> Path:
    relative = Path("_artifacts") / "datasets" / "trade_documents_images" / "source-v1" / "raw"
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / relative
        if candidate.is_dir():
            return candidate
    # A deterministic missing location keeps imports side-effect free.  The
    # preflight will fail rather than falling back to the root reference copy.
    return Path(__file__).resolve().parent / "__canonical_dataset_not_found__"


CANONICAL_DATASET_ROOT = _discover_canonical_dataset_root()
CANONICAL_GOLD_FILE = (
    CANONICAL_DATASET_ROOT.parent / "derived" / "adjudicated-gold" / "v1" / "adjudicated-gold.json"
)


class ImageRuntimeDiagnosticError(RuntimeError):
    """Content-free failure safe to reduce to a report code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _safe_code(value: object) -> str:
    candidate = str(value or "")
    return candidate if _SAFE_CODE_PATTERN.fullmatch(candidate) else "UNSAFE_CODE_REDACTED"


def _safe_identity(value: object) -> str:
    candidate = str(value or "")
    return candidate if _SAFE_IDENTITY_PATTERN.fullmatch(candidate) else "UNSAFE_IDENTITY_REDACTED"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_sha256(loader: Any) -> str:
    return hashlib.sha256(loader().encode("utf-8")).hexdigest()


def _contract_preflight() -> tuple[dict[str, str | None], list[str]]:
    failures: list[str] = []
    request_hash: str | None = None
    result_hash: str | None = None
    try:
        verify_packaged_schemas()
        request_hash = schema_sha256("backend", "v2", "review-request")
        result_hash = schema_sha256("backend", "v2", "review-result")
    except ContractSchemaError:
        failures.append("packaged_contract_drift")
    return (
        {
            "request_schema_sha256": request_hash,
            "result_schema_sha256": result_hash,
        },
        failures,
    )


def _base_report(
    *,
    provider: ProviderName,
    contracts: Mapping[str, str | None] | None = None,
    failures: Iterable[str] = (),
    status: DiagnosticStatus = "PREFLIGHT_BLOCKED",
    phase: str = "PREFLIGHT_BLOCKED",
    readiness_status: str = "NOT_ATTEMPTED",
    diagnostic_execution_passed: bool = False,
) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "scope": DIAGNOSTIC_SCOPE,
        # Evaluation assets, generic preflight, runtime execution, and Data
        # readiness are separate blocker classes. Every non-PASS value remains
        # non-release and must not be interpreted as a release decision.
        "status": status,
        "phase": phase,
        "release_ready": False,
        "release_decision_eligible": False,
        "diagnostic_execution_passed": diagnostic_execution_passed,
        "data_readiness_status": readiness_status,
        "contracts": dict(contracts or {}),
        "dataset": {
            "dataset_id": "trade_documents_images",
            "dataset_version": "source-v1",
            "gold_sha256": EXPECTED_GOLD_SHA256,
            "case_selector_sha256": hashlib.sha256(FIXED_CASE_ID.encode("utf-8")).hexdigest(),
            "triplet_fingerprint_sha256": FIXED_TRIPLET_SHA256,
            "input_image_sha256": sorted(document.source_sha256 for document in FIXED_DOCUMENTS),
        },
        "ocr": {
            "diagnostic_provider": provider,
            "release_selected_provider_claimed": False,
            "python_executable_sha256": None,
            "identities": [],
        },
        "claude": {
            "models": [],
            "prompts": [],
            "output_schemas": [],
        },
        "counts": {
            "input_document_count": 0,
            "completed_document_count": 0,
            "ocr_ledger_count": 0,
            "claude_call_count": 0,
            "claude_main_call_count": 0,
            "claude_crop_call_count": 0,
            "model_error_count": 0,
            "model_conflict_count": 0,
            "retrieval_result_count": 0,
            "retrieval_error_count": 0,
            "evidence_count": 0,
            "validation_issue_count": 0,
            "regulatory_finding_count": 0,
        },
        "agent_result": {
            "status": None,
            "requires_human_review": None,
            "retrieval_status": None,
            "retrieval_coverage": None,
        },
        "error_codes": sorted({_safe_code(code) for code in failures}),
        "redaction": {
            "raw_document_material": "OMITTED",
            "ocr_text": "OMITTED",
            "extracted_values": "OMITTED",
            "local_file_locations": "OMITTED",
            "network_destinations": "OMITTED",
            "credentials": "OMITTED",
            "provider_raw_metadata": "OMITTED",
        },
    }


def _read_gold_and_validate_sources() -> list[Path]:
    root = CANONICAL_DATASET_ROOT
    gold_file = CANONICAL_GOLD_FILE
    try:
        resolved_root = root.resolve(strict=True)
        resolved_gold = gold_file.resolve(strict=True)
    except OSError:
        raise ImageRuntimeDiagnosticError("canonical_dataset_or_gold_missing") from None
    if (
        root.is_symlink()
        or gold_file.is_symlink()
        or not resolved_root.is_dir()
        or not resolved_gold.is_file()
    ):
        raise ImageRuntimeDiagnosticError("canonical_dataset_or_gold_invalid")
    try:
        observed_gold_sha256 = _sha256_file(resolved_gold)
        raw_gold = resolved_gold.read_bytes()
        if len(raw_gold) > 2 * 1024 * 1024:
            raise ImageRuntimeDiagnosticError("adjudicated_gold_invalid")
        gold = json.loads(raw_gold)
        invalidation = image_gold_invalidation_code(
            observed_gold_sha256
        ) or image_gold_semantic_invalidation_code(gold)
        if invalidation is not None:
            raise ImageRuntimeDiagnosticError(invalidation)
        if EXPECTED_GOLD_SHA256 is None:
            raise ImageRuntimeDiagnosticError("adjudicated_gold_replacement_not_provisioned")
        if observed_gold_sha256 != EXPECTED_GOLD_SHA256:
            raise ImageRuntimeDiagnosticError("adjudicated_gold_hash_mismatch")
    except ImageRuntimeDiagnosticError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ImageRuntimeDiagnosticError("adjudicated_gold_invalid") from None
    if (
        not isinstance(gold, dict)
        or gold.get("schema_version") != "trade-image-adjudicated-gold/v1"
        or gold.get("gold_set_id") != EXPECTED_GOLD_SET_ID
        or gold.get("selection_id") != EXPECTED_SELECTION_ID
        or gold.get("gold_eligible") is not True
        or not isinstance(gold.get("cases"), list)
    ):
        raise ImageRuntimeDiagnosticError("adjudicated_gold_contract_invalid")
    matching_cases = [
        case
        for case in gold["cases"]
        if isinstance(case, dict) and case.get("case_id") == FIXED_CASE_ID
    ]
    if len(matching_cases) != 1:
        raise ImageRuntimeDiagnosticError("fixed_adjudicated_case_missing")
    case = matching_cases[0]
    annotation_references = case.get("annotation_references")
    if (
        case.get("adjudication_status") != "adjudicated"
        or case.get("gold_eligible") is not True
        or case.get("triplet_fingerprint_sha256") != FIXED_TRIPLET_SHA256
        or not isinstance(annotation_references, list)
        or len(annotation_references) != 2
        or len(set(annotation_references)) != 2
        or not isinstance(case.get("documents"), list)
    ):
        raise ImageRuntimeDiagnosticError("fixed_adjudicated_case_invalid")
    gold_by_id = {
        item.get("document_id"): item
        for item in case["documents"]
        if isinstance(item, dict) and isinstance(item.get("document_id"), str)
    }
    if set(gold_by_id) != {item.document_id for item in FIXED_DOCUMENTS}:
        raise ImageRuntimeDiagnosticError("fixed_adjudicated_document_set_mismatch")

    image_files: list[Path] = []
    for expected in FIXED_DOCUMENTS:
        observed = gold_by_id[expected.document_id]
        if (
            observed.get("source_document_type") != expected.source_document_type
            or observed.get("expected_document_type") != expected.expected_document_type
            or observed.get("relative_path") != expected.relative_name
            or observed.get("source_image_sha256") != expected.source_sha256
        ):
            raise ImageRuntimeDiagnosticError("fixed_adjudicated_document_mismatch")
        candidate = resolved_root / Path(expected.relative_name)
        try:
            if candidate.is_symlink():
                raise ImageRuntimeDiagnosticError("fixed_source_image_invalid")
            resolved_candidate = candidate.resolve(strict=True)
        except OSError:
            raise ImageRuntimeDiagnosticError("fixed_source_image_missing") from None
        if not resolved_candidate.is_file() or not resolved_candidate.is_relative_to(resolved_root):
            raise ImageRuntimeDiagnosticError("fixed_source_image_invalid")
        try:
            observed_hash = _sha256_file(resolved_candidate)
        except OSError:
            raise ImageRuntimeDiagnosticError("fixed_source_image_unreadable") from None
        if observed_hash != expected.source_sha256:
            raise ImageRuntimeDiagnosticError("fixed_source_image_hash_mismatch")
        image_files.append(resolved_candidate)
    return image_files


def _read_evidence_settings(
    *,
    env_file: Path,
    base_url: str,
) -> tuple[str, SecretStr | None, str, frozenset[str]]:
    try:
        candidate = env_file.expanduser()
        if not candidate.is_file():
            raise OSError
        raw = dotenv_values(dotenv_path=candidate, verbose=False)
    except (OSError, UnicodeError, ValueError):
        raise ImageRuntimeDiagnosticError("env_file_unreadable") from None
    file_values = {
        name: value for name in _EVIDENCE_ENV_KEYS if isinstance((value := raw.get(name)), str)
    }

    def configured(name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, file_values.get(name, default))

    allowed = frozenset(
        value.strip()
        for value in (configured("EVIDENCE_SERVICE_ALLOWED_ORIGINS") or "").split(",")
        if value.strip()
    )
    try:
        normalized_url = validate_evidence_destination(base_url, allowed)
    except (TypeError, ValueError):
        raise ImageRuntimeDiagnosticError("unsafe_evidence_destination") from None
    token = configured("EVIDENCE_SERVICE_TOKEN")
    token_header = configured("EVIDENCE_SERVICE_TOKEN_HEADER", "X-Demo-Token")
    if not token_header or not token_header.strip():
        raise ImageRuntimeDiagnosticError("evidence_token_header_invalid")
    return (
        normalized_url,
        SecretStr(token) if token else None,
        token_header.strip(),
        allowed,
    )


def _readiness_failure_code(exc: Exception) -> str:
    if isinstance(exc, RetrievalError) and "HTTP status 503" in str(exc):
        return "evidence_readiness_unready_503"
    if isinstance(exc, RetrievalError) and "contract identity missing" in str(exc):
        return "evidence_contract_identity_missing"
    if isinstance(exc, RetrievalError) and "contract identity mismatch" in str(exc):
        return "evidence_contract_identity_mismatch"
    return "evidence_readiness_failed"


def _preflight_status(code: str) -> DiagnosticStatus:
    evaluation_asset_prefixes = (
        "adjudicated_gold_",
        "canonical_dataset_",
        "fixed_adjudicated_",
        "fixed_source_image_",
    )
    if code.startswith(evaluation_asset_prefixes):
        return "BLOCKED_EVALUATION_ASSET"
    return "PREFLIGHT_BLOCKED"


def _runtime_failure_code(exc: Exception) -> str:
    if isinstance(exc, (LLMError, OcrLayoutProviderError, RetrievalError)):
        return _safe_code(exc.code)
    if isinstance(exc, ImageInputError):
        return "image_input_invalid"
    return "image_agent_runtime_failed"


def _run_nonrelease_diagnostic_review(
    *,
    request: ReviewRunRequestV2,
    claude_settings: ClaudeSettings,
    ocr_layout_provider: OcrLayoutProvider,
    evidence_base_url: str,
    evidence_service_token: SecretStr | None,
    evidence_token_header: str,
    evidence_allowed_origins: frozenset[str],
) -> ReviewRunResultV2:
    """Evaluation-only composition that cannot be used as service runtime."""

    retriever = HttpRegulationRetriever(
        evidence_base_url,
        service_token=(
            evidence_service_token.get_secret_value()
            if evidence_service_token is not None
            else None
        ),
        token_header=evidence_token_header,
        allowed_origins=evidence_allowed_origins,
    )
    analyzer = ClaudeDocumentAnalyzer.from_settings(
        claude_settings,
        ocr_layout_provider=ocr_layout_provider,
    )
    result = run_review_result_v2(
        request,
        retriever=retriever,
        analyzer=analyzer,
    )
    try:
        return ReviewRunResultV2.model_validate(result.model_dump(mode="python", warnings=False))
    except Exception:
        raise ImageRuntimeDiagnosticError("runtime_result_contract_invalid") from None


def _group_identities(
    values: Iterable[tuple[object, ...]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    counts = Counter(values)
    grouped: list[dict[str, Any]] = []
    for identity, count in sorted(
        counts.items(),
        key=lambda item: tuple(str(value or "") for value in item[0]),
    ):
        entry = {
            key: (
                value
                if key.endswith("sha256")
                and isinstance(value, str)
                and _SHA256_PATTERN.fullmatch(value)
                else _safe_identity(value)
            )
            for key, value in zip(keys, identity, strict=True)
        }
        entry["call_count"] = count
        grouped.append(entry)
    return grouped


def _expected_call_identity(call: Any, *, model_id: str) -> bool:
    vision_prompt_hash = _prompt_sha256(load_document_vision_analysis_system_prompt)
    crop_prompt_hash = _prompt_sha256(load_document_crop_reconciliation_system_prompt)
    vision_schema_hash = _canonical_sha256(ClaudeVisionDocumentCandidate.model_json_schema())
    crop_schema_hash = _canonical_sha256(ClaudeCropReadCandidate.model_json_schema())
    common = (
        call.provider == "anthropic"
        and call.model_id == model_id
        and call.output_schema_version == CLAUDE_CANDIDATE_SCHEMA_VERSION
    )
    if call.input_modality == "image":
        return common and (
            call.prompt_id == DOCUMENT_VISION_ANALYSIS_PROMPT_ID
            and call.prompt_version == DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION
            and call.prompt_sha256 == vision_prompt_hash
            and call.output_schema_id == "customs_claude_vision_document_candidate"
            and call.output_schema_sha256 == vision_schema_hash
        )
    if call.input_modality == "image_crop":
        return common and (
            call.prompt_id == DOCUMENT_CROP_RECONCILIATION_PROMPT_ID
            and call.prompt_version == DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION
            and call.prompt_sha256 == crop_prompt_hash
            and call.output_schema_id == "customs_claude_crop_read_candidate"
            and call.output_schema_sha256 == crop_schema_hash
        )
    return False


def _summarize_result(
    result: ReviewRunResultV2,
    *,
    provider: ProviderName,
    model_id: str,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    documents = result.document_results
    calls = result.model_calls
    main_calls = [call for call in calls if call.input_modality == "image"]
    crop_calls = [call for call in calls if call.input_modality == "image_crop"]
    if len(documents) != len(FIXED_DOCUMENTS):
        failures.append("result_document_count_invalid")
    if any(
        document.input_modality != "image"
        or document.extraction_status != "completed"
        or document.ocr_layout is None
        for document in documents
    ):
        failures.append("image_extraction_incomplete")
    if len(main_calls) != len(FIXED_DOCUMENTS):
        failures.append("claude_main_call_count_invalid")
    if result.model_errors:
        failures.append("model_errors_present")
    if any(not _expected_call_identity(call, model_id=model_id) for call in calls):
        failures.append("claude_call_identity_invalid")
    if any(
        call.ocr_provider_id != provider
        or not call.ocr_ledger_id
        or not call.ocr_engine_id
        or not call.ocr_engine_version
        or not call.ocr_model_id
        or not call.ocr_model_assets_lock_sha256
        for call in calls
    ):
        failures.append("ocr_provenance_invalid")
    if not result.retrieval_results and not result.retrieval_errors:
        failures.append("evidence_query_not_observed")

    incomplete_data = (
        result.retrieval_status != "ok"
        or result.retrieval_coverage != "complete"
        or not result.knowledge_snapshot_id
        or bool(result.retrieval_errors)
        or not result.retrieval_results
        or any(item.coverage != "complete" or item.errors for item in result.retrieval_results)
    )
    if incomplete_data and (
        result.status not in {"needs_review", "blocked"} or not result.requires_human_review
    ):
        failures.append("incomplete_evidence_not_fail_closed")

    ocr_identities = _group_identities(
        (
            (
                call.ocr_provider_id,
                call.ocr_engine_id,
                call.ocr_engine_version,
                call.ocr_model_id,
                call.ocr_model_assets_lock_sha256,
            )
            for call in calls
        ),
        (
            "provider_id",
            "engine_id",
            "engine_version",
            "model_id",
            "model_assets_lock_sha256",
        ),
    )
    models = _group_identities(
        ((call.provider, call.model_id) for call in calls),
        ("provider", "model_id"),
    )
    prompts = _group_identities(
        (
            (
                call.prompt_id,
                call.prompt_version,
                call.prompt_sha256,
            )
            for call in calls
        ),
        ("prompt_id", "prompt_version", "prompt_sha256"),
    )
    output_schemas = _group_identities(
        (
            (
                call.output_schema_id,
                call.output_schema_version,
                call.output_schema_sha256,
            )
            for call in calls
        ),
        ("schema_id", "schema_version", "schema_sha256"),
    )
    summary = {
        "ocr_identities": ocr_identities,
        "models": models,
        "prompts": prompts,
        "output_schemas": output_schemas,
        "counts": {
            "input_document_count": len(FIXED_DOCUMENTS),
            "completed_document_count": sum(
                document.extraction_status == "completed" for document in documents
            ),
            "ocr_ledger_count": len(
                {
                    document.ocr_layout.ledger_id
                    for document in documents
                    if document.ocr_layout is not None
                }
            ),
            "claude_call_count": len(calls),
            "claude_main_call_count": len(main_calls),
            "claude_crop_call_count": len(crop_calls),
            "model_error_count": len(result.model_errors),
            "model_conflict_count": len(result.model_conflicts),
            "retrieval_result_count": len(result.retrieval_results),
            "retrieval_error_count": len(result.retrieval_errors),
            "evidence_count": len(result.rag_evidence),
            "validation_issue_count": len(result.validation_issues),
            "regulatory_finding_count": len(result.regulatory_findings),
        },
        "agent_result": {
            "status": _safe_code(result.status),
            "requires_human_review": result.requires_human_review,
            "retrieval_status": _safe_code(result.retrieval_status),
            "retrieval_coverage": _safe_code(result.retrieval_coverage),
        },
        "observed_error_codes": [
            *(_safe_code(error.code) for error in result.model_errors),
            *(_safe_code(error.code) for error in result.retrieval_errors),
        ],
        "incomplete_data": incomplete_data,
    }
    return summary, sorted(set(failures))


def evaluate(
    *,
    env_file: Path,
    provider: ProviderName,
    ocr_python: Path,
    evidence_base_url: str,
    review_date: date | None,
    live: bool = False,
    nonrelease_ocr_diagnostic: bool = False,
    allow_unready_diagnostic: bool = False,
) -> tuple[int, dict[str, Any]]:
    if not live or not nonrelease_ocr_diagnostic or review_date is None:
        failures = []
        if not live:
            failures.append("live_flag_required")
        if not nonrelease_ocr_diagnostic:
            failures.append("nonrelease_ocr_diagnostic_flag_required")
        if review_date is None:
            failures.append("review_date_required")
        return 1, _base_report(provider=provider, failures=failures)

    contracts, failures = _contract_preflight()
    if failures:
        return 1, _base_report(
            provider=provider,
            contracts=contracts,
            failures=failures,
        )
    try:
        image_files = _read_gold_and_validate_sources()
        if ocr_python.expanduser().is_symlink():
            raise ImageRuntimeDiagnosticError("ocr_python_unavailable")
        resolved_python = ocr_python.expanduser().resolve(strict=True)
        if not resolved_python.is_file():
            raise ImageRuntimeDiagnosticError("ocr_python_unavailable")
        python_sha256 = _sha256_file(resolved_python)
        claude_settings = ClaudeSettings.from_environment(env_file)
        normalized_url, evidence_token, token_header, allowed_origins = _read_evidence_settings(
            env_file=env_file,
            base_url=evidence_base_url,
        )
    except ImageRuntimeDiagnosticError as exc:
        return 1, _base_report(
            provider=provider,
            contracts=contracts,
            failures=[exc.code],
            status=_preflight_status(exc.code),
        )
    except (LLMError, OSError, ValueError) as exc:
        code = (
            exc.code
            if isinstance(exc, LLMError)
            else (
                "preflight_configuration_invalid"
                if isinstance(exc, ValueError)
                else "preflight_io_failed"
            )
        )
        return 1, _base_report(
            provider=provider,
            contracts=contracts,
            failures=[code],
        )

    readiness_status = "READY"
    token_value = evidence_token.get_secret_value() if evidence_token is not None else None
    try:
        supported_date = fetch_supported_review_as_of(
            normalized_url,
            service_token=token_value,
            token_header=token_header,
            allowed_origins=allowed_origins,
            contract_identity_mode="required",
        )
        if supported_date != review_date:
            raise ImageRuntimeDiagnosticError("review_date_not_supported")
    except ImageRuntimeDiagnosticError as exc:
        report = _base_report(
            provider=provider,
            contracts=contracts,
            failures=[exc.code],
            status="BLOCKED_DATA_READINESS",
            readiness_status="READY_DATE_MISMATCH",
        )
        report["ocr"]["python_executable_sha256"] = python_sha256
        return 1, report
    except Exception as exc:
        code = _readiness_failure_code(exc)
        if code != "evidence_readiness_unready_503" or not allow_unready_diagnostic:
            report = _base_report(
                provider=provider,
                contracts=contracts,
                failures=[code],
                status="BLOCKED_DATA_READINESS",
                readiness_status=(
                    "UNREADY_503" if code == "evidence_readiness_unready_503" else "FAILED"
                ),
            )
            report["ocr"]["python_executable_sha256"] = python_sha256
            return 1, report
        readiness_status = "UNREADY_503_DIAGNOSTIC"

    provider_instance: OcrLayoutProvider | None = None
    close_failure: str | None = None
    try:
        provider_instance = build_ocr_layout_provider(
            provider,
            python_executable=resolved_python,
        )
        documents = [
            DocumentInputV2(
                document_id=expected.document_id,
                filename=Path(expected.relative_name).name,
                image_pages=[image_page_from_file(source)],
            )
            for expected, source in zip(FIXED_DOCUMENTS, image_files, strict=True)
        ]
        result = _run_nonrelease_diagnostic_review(
            request=ReviewRunRequestV2(
                case_id="image-service-diagnostic",
                documents=documents,
                llm_mode="llm",
                review_run_id="image-runtime-nonrelease-diagnostic-v1",
                review_as_of=review_date,
            ),
            claude_settings=claude_settings,
            ocr_layout_provider=provider_instance,
            evidence_base_url=normalized_url,
            evidence_service_token=evidence_token,
            evidence_token_header=token_header,
            evidence_allowed_origins=allowed_origins,
        )
        if not isinstance(result, ReviewRunResultV2):
            raise ImageRuntimeDiagnosticError("runtime_result_contract_invalid")
    except Exception as exc:
        code = (
            exc.code if isinstance(exc, ImageRuntimeDiagnosticError) else _runtime_failure_code(exc)
        )
        report = _base_report(
            provider=provider,
            contracts=contracts,
            failures=[code],
            status="DIAGNOSTIC_EXECUTION_FAILED",
            phase="EXECUTION_BLOCKED",
            readiness_status=readiness_status,
        )
        report["ocr"]["python_executable_sha256"] = python_sha256
        return 1, report
    finally:
        if provider_instance is not None:
            try:
                close = getattr(provider_instance, "close", None)
                if callable(close):
                    close()
            except Exception:
                close_failure = "ocr_provider_close_failed"

    summary, execution_failures = _summarize_result(
        result,
        provider=provider,
        model_id=claude_settings.model,
    )
    if close_failure:
        execution_failures.append(close_failure)
    observed_error_codes = summary.pop("observed_error_codes")
    incomplete_data = bool(summary.pop("incomplete_data"))
    report = _base_report(
        provider=provider,
        contracts=contracts,
        failures=[*execution_failures, *observed_error_codes],
        status=(
            "BLOCKED_DATA_READINESS"
            if readiness_status != "READY" or incomplete_data
            else "DIAGNOSTIC_EXECUTION_FAILED"
        ),
        phase=(
            "DATA_READINESS_BLOCKED"
            if readiness_status != "READY" or incomplete_data
            else "COMPLETE"
        ),
        readiness_status=readiness_status,
        diagnostic_execution_passed=not execution_failures,
    )
    report["ocr"]["python_executable_sha256"] = python_sha256
    report["ocr"]["identities"] = summary["ocr_identities"]
    report["claude"]["models"] = summary["models"]
    report["claude"]["prompts"] = summary["prompts"]
    report["claude"]["output_schemas"] = summary["output_schemas"]
    report["counts"] = summary["counts"]
    report["agent_result"] = summary["agent_result"]
    if (
        readiness_status == "READY"
        and not incomplete_data
        and not execution_failures
        and not observed_error_codes
    ):
        report["status"] = "DIAGNOSTIC_PASS"
        # A diagnostic PASS remains explicitly non-release.
        return 0, report
    if readiness_status != "READY" or incomplete_data:
        report["error_codes"] = sorted(
            {
                *report["error_codes"],
                "evidence_data_incomplete",
            }
        )
    return 1, report


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one hash-pinned image triplet through OCR, Claude Vision, and "
            "Evidence HTTP. This diagnostic can never make a release decision."
        )
    )
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=("paddle", "docling"))
    parser.add_argument("--ocr-python", required=True, type=Path)
    parser.add_argument("--evidence-base-url", required=True)
    parser.add_argument("--review-date", required=True, type=_parse_date)
    parser.add_argument("--live", required=True, action="store_true")
    parser.add_argument(
        "--nonrelease-ocr-diagnostic",
        required=True,
        action="store_true",
    )
    parser.add_argument("--allow-unready-diagnostic", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, report = evaluate(
        env_file=args.env_file,
        provider=args.provider,
        ocr_python=args.ocr_python,
        evidence_base_url=args.evidence_base_url,
        review_date=args.review_date,
        live=args.live,
        nonrelease_ocr_diagnostic=args.nonrelease_ocr_diagnostic,
        allow_unready_diagnostic=args.allow_unready_diagnostic,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
