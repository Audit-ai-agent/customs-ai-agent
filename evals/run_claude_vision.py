"""Fail-closed Claude Vision evaluation over trusted synthetic image Gold.

Both the historical case-0500 artifact and the former 12-triplet AI-reviewed
Gold are invalidated and can never produce a score. No replacement Gold hash is
currently approved, so the evaluator stops before OCR, Claude, or Evidence
clients are created. A future run is possible only after independently
human-reviewed replacement Gold, its Gate, the prompt, and every pinned hash
agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from jsonschema import Draft202012Validator

from customs_ai_agent.adapters import (
    OcrLayoutProvider,
    OcrLayoutProviderError,
    build_ocr_layout_provider,
)
from customs_ai_agent.adapters.images import image_page_from_file
from customs_ai_agent.adapters.llm import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    CropReadBudget,
    DocumentAnalysis,
    LLMError,
)
from customs_ai_agent.adapters.retrieval import MockRegulationRetriever
from customs_ai_agent.evaluation_assets import (
    image_gold_invalidation_code,
    image_gold_semantic_invalidation_code,
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
    load_document_vision_analysis_system_prompt,
)
from customs_ai_agent.runner import run_review_v2
from customs_ai_agent.schemas.llm import ClaudeVisionDocumentCandidate
from customs_ai_agent.schemas.models import DocumentInputV2, ReviewRunRequestV2
from evals.ocr_release_gate_attestation import (
    EnvironmentAttestor,
    ModelAssetsAttestor,
    OcrReleaseGateAttestationError,
    ReleaseLockedOcrProvider,
    installed_ocr_environment_lock_sha256,
    load_canonical_ocr_release_gate,
    load_ocr_release_runtime_gate,
    verify_ocr_model_assets,
)
from evals.run_ocr_benchmark import (
    OcrBenchmarkError,
    WheelAttestor,
    installed_agent_wheel_sha256,
    load_benchmark,
)


DEFAULT_GOLD_PATH = (
    Path(__file__).parent / "trade_documents_images" / "v1" / "manifest_partial_gold.json"
)
RELEASE_CASE_COUNT = 12
RELEASE_DOCUMENT_COUNT = 36
DEFAULT_SELECTION_PATH = (
    Path(__file__).parent / "trade_documents_images" / "v1" / "diagnostic-selection-v1.json"
)
DEFAULT_ADJUDICATED_GOLD_SCHEMA_PATH = (
    Path(__file__).parent
    / "trade_documents_images"
    / "v1"
    / "schemas"
    / "adjudicated-gold.schema.json"
)
DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "trade_documents_images"
    / "v1"
    / "dataset.lock.json"
)
EXPECTED_INVALIDATED_ANNOTATION_SHA256 = (
    "ac86a6e248492db92e5bfa635dfbdff3b74adf8cb454bf0da7faabc77cfa33dd"
)
EXPECTED_DATASET_LOCK_SHA256 = "2177c598d1b0111cbf7dbf0a7a25c46041a523b6b3923b9ecee71d3c64141edc"
EXPECTED_SELECTION_SHA256 = "8959d3e91affb1c51fbb320cf90cea7349d069cd12b58858d27ea13fb92ba961"
EXPECTED_ADJUDICATED_GOLD_SCHEMA_SHA256 = (
    "6d6695c72cda5302b762d338fc97792b5dca48e4eb75fe55a1798362e86f9066"
)
ALLOWED_ADJUDICATION_CONCLUSIONS = {
    "source_matches_manifest_observation",
    "source_discrepancy_confirmed",
    "invalid_prior_claim",
    "unreadable_or_insufficient",
}
EXPECTED_VISION_SCHEMA_SHA256 = "0bb62c5b592ceb1147d31fa5e8e3a508d7b60f877f0ee8a8cb413de75ead3e8f"
# This is the existing 1.2.0 evaluator prompt pin from 8fe0058. Commit c28e72d
# intentionally changed the runtime prompt to 1.3.0, whose current hash is
# different. Do not silently bless that change here: a new evaluator prompt
# pin requires explicit evaluation-owner approval and fresh trusted Gold.
EXPECTED_VISION_PROMPT_SHA256 = "531ad80b62016c5bd65ea292ac584f36413dfb230b13cffb5761b6862d1705ba"
# No independently trusted replacement image Gold exists. Any adjudicated
# manifest remains ineligible until its exact reviewed bytes are pinned here.
EXPECTED_ADJUDICATED_GOLD_SHA256: str | None = None
MAX_CROP_CALLS_PER_RUN = 6
ProviderName = Literal["paddle", "docling"]
ProviderFactory = Callable[[ProviderName, Path], OcrLayoutProvider]
AnalyzerFactory = Callable[[ClaudeSettings, OcrLayoutProvider], ClaudeDocumentAnalyzer]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_or_none(path: Path) -> str | None:
    try:
        return _sha256(path)
    except OSError:
        return None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _vision_schema_sha256() -> str:
    return _canonical_sha256(ClaudeVisionDocumentCandidate.model_json_schema())


def _vision_prompt_sha256() -> str:
    return hashlib.sha256(load_document_vision_analysis_system_prompt().encode("utf-8")).hexdigest()


def _read_json_object(
    path: Path,
    *,
    failure_code: str,
    failures: list[str],
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        failures.append(failure_code)
        return {}
    if not isinstance(value, dict):
        failures.append(failure_code)
        return {}
    return value


def _read_worklist(path: Path, failures: list[str]) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        failures.append("adjudication_worklist_missing_or_invalid")
        return []
    if not values or any(not isinstance(value, dict) for value in values):
        failures.append("adjudication_worklist_missing_or_invalid")
        return []
    return values


def _adjudication_paths(
    dataset_root: Path,
    worklist_path: Path | None,
    gate_path: Path | None,
) -> tuple[Path, Path]:
    derived_root = dataset_root.parent / "derived" / "diagnostic-selection" / "v1"
    return (
        worklist_path or derived_root / "adjudications.jsonl",
        gate_path or derived_root / "gate.json",
    )


def _validate_gold_gate(
    *,
    selection: dict[str, Any],
    worklist: list[dict[str, Any]],
    gate: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    selection_id = selection.get("selection_id")
    selected_cases = selection.get("cases")
    if (
        selection.get("schema_version") != "trade-image-diagnostic-selection/v1"
        or selection_id != "trade-images-hash-ranked-diagnostic-v1"
        or selection.get("selection_role") != "diagnostic_candidate_not_gold"
        or not isinstance(selected_cases, list)
        or len(selected_cases) != 12
    ):
        return ["diagnostic_selection_contract_invalid"]
    selected_by_id = {
        case.get("case_id"): case
        for case in selected_cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    if len(selected_by_id) != 12:
        return ["diagnostic_selection_contract_invalid"]
    if len(worklist) != 12:
        failures.append("adjudication_worklist_case_set_mismatch")
        gold_ready = False
    else:
        entry_by_id = {
            entry.get("case_id"): entry
            for entry in worklist
            if isinstance(entry.get("case_id"), str)
        }
        if len(entry_by_id) != 12 or set(entry_by_id) != set(selected_by_id):
            failures.append("adjudication_worklist_case_set_mismatch")
            gold_ready = False
        else:
            gold_ready = True
            for case_id, selected in selected_by_id.items():
                entry = entry_by_id[case_id]
                if (
                    entry.get("schema_version") != "trade-image-diagnostic-adjudication/v1"
                    or entry.get("selection_id") != selection_id
                    or entry.get("triplet_fingerprint_sha256")
                    != selected.get("triplet_fingerprint_sha256")
                    or entry.get("required_independent_review_count") != 2
                ):
                    failures.append("adjudication_worklist_evidence_invalid")
                    gold_ready = False
                    continue
                if entry.get("status") != "adjudicated" or entry.get("gold_eligible") is not True:
                    gold_ready = False
                    continue
                reviews = entry.get("reviews")
                if not isinstance(reviews, list) or len(reviews) != 2:
                    failures.append("adjudication_worklist_evidence_invalid")
                    gold_ready = False
                    continue
                reviewer_ids = [review.get("reviewer_id") for review in reviews]
                review_ids = [review.get("review_id") for review in reviews]
                allowed_hashes = {
                    document["sha256"]
                    for document in selected.get("documents", {}).values()
                    if isinstance(document, dict) and "sha256" in document
                }
                if (
                    len(allowed_hashes) != 3
                    or any(
                        not isinstance(review, dict)
                        or review.get("independent_review_attested") is not True
                        or review.get("scope") != "diagnostic_full_triplet"
                        or set(review.get("reviewed_document_sha256", [])) != allowed_hashes
                        or not isinstance(review.get("annotation_reference"), str)
                        or not review["annotation_reference"]
                        or review.get("conclusion") not in ALLOWED_ADJUDICATION_CONCLUSIONS
                        for review in reviews
                    )
                    or len(set(reviewer_ids)) != 2
                    or len(set(review_ids)) != 2
                    or any(
                        not isinstance(reviewer_id, str) or not reviewer_id
                        for reviewer_id in reviewer_ids
                    )
                    or any(
                        not isinstance(review_id, str) or not review_id for review_id in review_ids
                    )
                ):
                    failures.append("adjudication_worklist_evidence_invalid")
                    gold_ready = False
                    continue
                conclusions = {review.get("conclusion") for review in reviews}
                resolution = entry.get("resolution")
                if not isinstance(resolution, dict):
                    failures.append("adjudication_worklist_resolution_invalid")
                    gold_ready = False
                    continue
                if (
                    resolution.get("final_conclusion") not in ALLOWED_ADJUDICATION_CONCLUSIONS
                    or not isinstance(resolution.get("reference"), str)
                    or not resolution["reference"]
                ):
                    failures.append("adjudication_worklist_resolution_invalid")
                    gold_ready = False
                    continue
                if len(conclusions) == 1:
                    if (
                        resolution.get("method") != "two_reviewer_consensus"
                        or resolution.get("final_conclusion") not in conclusions
                    ):
                        failures.append("adjudication_worklist_resolution_invalid")
                        gold_ready = False
                elif (
                    resolution.get("method") != "independent_adjudicator"
                    or not isinstance(resolution.get("adjudicator_id"), str)
                    or not resolution["adjudicator_id"]
                    or resolution["adjudicator_id"] in reviewer_ids
                    or resolution.get("independent_adjudicator_attested") is not True
                ):
                    failures.append("adjudication_worklist_resolution_invalid")
                    gold_ready = False

    gate_claims_ready = (
        gate.get("schema_version") == "trade-image-diagnostic-gate/v1"
        and gate.get("selection_id") == selection_id
        and gate.get("status") == "passed_two_reviewer_adjudication"
        and gate.get("gold_eligible") is True
        and gate.get("selected_case_count") == 12
        and gate.get("status_counts") == {"adjudicated": 12}
        and gate.get("blockers") == []
    )
    if gate_claims_ready != gold_ready:
        failures.append("adjudication_gate_evidence_mismatch")
    if not gold_ready or not gate_claims_ready:
        failures.append("adjudication_gate_not_gold_ready")
    return sorted(set(failures))


def _validate_gold_manifest(
    *,
    gold: dict[str, Any],
    selection: dict[str, Any],
    worklist: list[dict[str, Any]],
    selection_path: Path,
    worklist_path: Path,
    gate_path: Path,
    schema_path: Path,
) -> list[str]:
    if gold.get("schema_version") == ("trade-image-invalidated-candidate-annotations/v1"):
        if (
            gold.get("annotation_status") != "invalidated_by_label_error"
            or gold.get("gold_eligibility") != "invalidated_by_label_error"
            or gold.get("release_scoring_allowed") is not False
            or gold.get("invalidation_provenance", {}).get("reason_code")
            != "invalidated_by_label_error"
        ):
            return ["invalidated_annotation_provenance_missing"]
        return [
            "legacy_case_0500_invalidated_by_label_error",
            "adjudicated_gold_manifest_required",
        ]
    if gold.get("schema_version") != "trade-image-adjudicated-gold/v1":
        return ["adjudicated_gold_manifest_required"]
    failures: list[str] = []
    schema = _read_json_object(
        schema_path,
        failure_code="adjudicated_gold_schema_missing_or_invalid",
        failures=failures,
    )
    if schema:
        validation_errors = sorted(
            Draft202012Validator(schema).iter_errors(gold),
            key=lambda error: list(error.absolute_path),
        )
        if validation_errors:
            failures.append("adjudicated_gold_schema_validation_failed")
    if (
        gold.get("selection_id") != selection.get("selection_id")
        or gold.get("selection_sha256") != _sha256_or_none(selection_path)
        or gold.get("adjudication_worklist_sha256") != _sha256_or_none(worklist_path)
        or gold.get("adjudication_gate_sha256") != _sha256_or_none(gate_path)
        or gold.get("gold_eligible") is not True
    ):
        failures.append("adjudicated_gold_provenance_mismatch")
        return sorted(set(failures))
    selected_by_id = {case["case_id"]: case for case in selection.get("cases", [])}
    gold_cases = gold.get("cases")
    if not isinstance(gold_cases, list):
        failures.append("adjudicated_gold_case_set_mismatch")
        return sorted(set(failures))
    gold_by_id = {
        case.get("case_id"): case
        for case in gold_cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    worklist_by_id = {
        entry["case_id"]: entry
        for entry in worklist
        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str)
    }
    if (
        len(gold_by_id) != 12
        or set(gold_by_id) != set(selected_by_id)
        or set(worklist_by_id) != set(selected_by_id)
    ):
        failures.append("adjudicated_gold_case_set_mismatch")
        return sorted(set(failures))
    for case_id, selected in selected_by_id.items():
        gold_case = gold_by_id[case_id]
        if gold_case.get("triplet_fingerprint_sha256") != selected.get(
            "triplet_fingerprint_sha256"
        ):
            failures.append("adjudicated_gold_source_mismatch")
            continue
        review_references = {
            review.get("annotation_reference")
            for review in worklist_by_id[case_id].get("reviews", [])
        }
        if not review_references.issubset(set(gold_case.get("annotation_references", []))):
            failures.append("adjudicated_gold_annotation_reference_mismatch")
        selected_documents = selected.get("documents", {})
        gold_documents = gold_case.get("documents", [])
        gold_by_type = {
            document.get("source_document_type"): document
            for document in gold_documents
            if isinstance(document, dict)
        }
        if set(gold_by_type) != set(selected_documents):
            failures.append("adjudicated_gold_source_mismatch")
            continue
        for document_type, selected_document in selected_documents.items():
            gold_document = gold_by_type[document_type]
            if gold_document.get("relative_path") != selected_document.get(
                "relative_path"
            ) or gold_document.get("source_image_sha256") != selected_document.get("sha256"):
                failures.append("adjudicated_gold_source_mismatch")
    return sorted(set(failures))


def _preflight(
    *,
    dataset_root: Path,
    gold_path: Path,
    lock_path: Path,
    selection_path: Path,
    worklist_path: Path,
    gate_path: Path,
    gold_schema_path: Path,
    case_id: str | None,
    run_workflow: bool,
) -> tuple[
    list[str],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    failures: list[str] = []
    gold = _read_json_object(
        gold_path,
        failure_code="gold_manifest_missing_or_invalid",
        failures=failures,
    )
    dataset_lock = _read_json_object(
        lock_path,
        failure_code="dataset_lock_missing_or_invalid",
        failures=failures,
    )
    selection = _read_json_object(
        selection_path,
        failure_code="diagnostic_selection_missing_or_invalid",
        failures=failures,
    )
    worklist = _read_worklist(worklist_path, failures)
    gate = _read_json_object(
        gate_path,
        failure_code="adjudication_gate_missing_or_invalid",
        failures=failures,
    )
    gold_sha256 = _sha256_or_none(gold_path)
    lock_sha256 = _sha256_or_none(lock_path)
    schema_sha256 = _vision_schema_sha256()
    prompt_sha256 = _vision_prompt_sha256()
    invalidated_gold_code = image_gold_invalidation_code(gold_sha256)
    if invalidated_gold_code is not None:
        failures.append(invalidated_gold_code)
    semantic_invalidation_code = image_gold_semantic_invalidation_code(gold)
    if semantic_invalidation_code is not None:
        failures.append(semantic_invalidation_code)
    if gold.get("schema_version") == "trade-image-adjudicated-gold/v1":
        if EXPECTED_ADJUDICATED_GOLD_SHA256 is None:
            failures.append("adjudicated_gold_replacement_not_provisioned")
        elif gold_sha256 != EXPECTED_ADJUDICATED_GOLD_SHA256:
            failures.append("adjudicated_gold_hash_mismatch")
    if (
        gold.get("schema_version") == "trade-image-invalidated-candidate-annotations/v1"
        and gold_sha256 != EXPECTED_INVALIDATED_ANNOTATION_SHA256
    ):
        failures.append("invalidated_annotation_hash_mismatch")
    if lock_sha256 != EXPECTED_DATASET_LOCK_SHA256:
        failures.append("dataset_lock_hash_mismatch")
    ground_truth = dataset_lock.get("ground_truth", {})
    bounded_gold = ground_truth.get("bounded_adjudicated_gold", {})
    if (
        ground_truth.get("single_reviewer_candidate_status") != "invalidated_by_label_error"
        or ground_truth.get("release_scoring_allowed") is not False
        or not isinstance(bounded_gold, dict)
        or bounded_gold.get("status") != "invalidated_by_label_error"
        or bounded_gold.get("release_scoring_allowed_within_scope") is not False
        or bounded_gold.get("semantic_fingerprint_sha256")
        != "ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133"
    ):
        failures.append("dataset_lock_invalidation_provenance_missing")
    if _sha256_or_none(selection_path) != EXPECTED_SELECTION_SHA256:
        failures.append("diagnostic_selection_hash_mismatch")
    if _sha256_or_none(gold_schema_path) != EXPECTED_ADJUDICATED_GOLD_SCHEMA_SHA256:
        failures.append("adjudicated_gold_schema_hash_mismatch")
    if schema_sha256 != EXPECTED_VISION_SCHEMA_SHA256:
        failures.append("candidate_schema_hash_mismatch")
    if prompt_sha256 != EXPECTED_VISION_PROMPT_SHA256:
        failures.append("prompt_hash_mismatch")
    if selection and worklist and gate:
        failures.extend(
            _validate_gold_gate(
                selection=selection,
                worklist=worklist,
                gate=gate,
            )
        )
    if gold and selection and worklist:
        failures.extend(
            _validate_gold_manifest(
                gold=gold,
                selection=selection,
                worklist=worklist,
                selection_path=selection_path,
                worklist_path=worklist_path,
                gate_path=gate_path,
                schema_path=gold_schema_path,
            )
        )
    if run_workflow:
        failures.append("workflow_scoring_requires_separate_outcome_gold")
    if not dataset_root.is_dir():
        failures.append("dataset_root_missing")
    else:
        manifest_path = dataset_root / "manifest.csv"
        if not manifest_path.is_file() or _sha256(manifest_path) != dataset_lock.get(
            "source_integrity", {}
        ).get("manifest_sha256"):
            failures.append("source_manifest_hash_mismatch")

    inventory_path = dataset_root.parent / "dataset-inventory.json"
    inventory: dict[str, Any] = {}
    if not inventory_path.is_file() or _sha256(inventory_path) != dataset_lock.get(
        "source_integrity", {}
    ).get("inventory_file_sha256"):
        failures.append("inventory_hash_mismatch")
    else:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    image_index = {str(item["relative_path"]): item for item in inventory.get("images", [])}
    gold_cases = gold.get("cases", [])
    if not isinstance(gold_cases, list):
        failures.append("scored_case_not_found")
        return sorted(set(failures)), {}, dataset_lock, image_index
    eligible_cases = [
        item
        for item in gold_cases
        if isinstance(item, dict)
        and isinstance(item.get("case_id"), str)
        and isinstance(item.get("documents"), list)
        and item["documents"]
    ]
    selected_cases = (
        eligible_cases
        if case_id is None
        else [item for item in eligible_cases if item.get("case_id") == case_id]
    )
    if not selected_cases:
        failures.append("scored_case_not_found")
        return sorted(set(failures)), {}, dataset_lock, image_index

    selection_by_id = {
        str(item["case_id"]): item
        for item in selection.get("cases", [])
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    selected_documents: list[dict[str, Any]] = []
    selected_case_ids: list[str] = []
    for selected_gold_case in selected_cases:
        selected_case_id = str(selected_gold_case["case_id"])
        selected_case_ids.append(selected_case_id)
        selected_case = cast(Mapping[str, Any], selection_by_id.get(selected_case_id, {}))
        for document in selected_gold_case["documents"]:
            relative_path = str(document["relative_path"])
            image_path = (dataset_root / relative_path).resolve()
            try:
                image_path.relative_to(dataset_root.resolve())
            except ValueError:
                failures.append("document_path_escapes_dataset")
                continue
            inventory_item = image_index.get(relative_path)
            if (
                inventory_item is None
                or not image_path.is_file()
                or _sha256(image_path) != inventory_item["sha256"]
            ):
                failures.append("selected_image_hash_mismatch")
            selected_document_type = document.get("source_document_type")
            selected_document = selected_case.get("documents", {}).get(
                selected_document_type,
                {},
            )
            if selected_document.get("relative_path") != relative_path or selected_document.get(
                "sha256"
            ) != document.get("source_image_sha256"):
                failures.append("selected_image_provenance_mismatch")
            selected_documents.append(document)
    grouped_case = {
        "case_id": (
            selected_case_ids[0] if len(selected_case_ids) == 1 else "all-adjudicated-cases"
        ),
        "case_ids": selected_case_ids,
        "documents": selected_documents,
    }
    return sorted(set(failures)), grouped_case, dataset_lock, image_index


def _safe_preflight_report(
    *,
    failures: list[str],
    gold_path: Path,
    lock_path: Path,
    selection_path: Path,
    worklist_path: Path,
    gate_path: Path,
    gold_schema_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "claude-vision-evaluation-report/v1",
        "scope": "TRUSTED_REPLACEMENT_GOLD_REQUIRED_HISTORICAL_GOLD_BLOCKED",
        "status": "FAIL",
        "release_score_eligible": False,
        "provider_called": False,
        "call_accounting": {
            "attempted": 0,
            "succeeded": 0,
            "retried": 0,
        },
        "cost_budget": {"configured_max_retries": 0},
        "legacy_case_0500_provenance": "invalidated_by_label_error",
        "historical_adjudicated_gold_provenance": "invalidated_by_label_error",
        "replacement_gold_status": "not_available",
        "failures": [{"document_id": "preflight", "code": code} for code in failures],
        "gold_manifest_sha256": _sha256_or_none(gold_path),
        "gold_manifest_invalidation_code": (
            image_gold_invalidation_code(_sha256_or_none(gold_path))
            or image_gold_semantic_invalidation_code(
                _read_json_object(
                    gold_path,
                    failure_code="gold_manifest_missing_or_invalid",
                    failures=[],
                )
            )
        ),
        "expected_adjudicated_gold_sha256": EXPECTED_ADJUDICATED_GOLD_SHA256,
        "expected_invalidated_annotation_sha256": (EXPECTED_INVALIDATED_ANNOTATION_SHA256),
        "dataset_lock_sha256": _sha256_or_none(lock_path),
        "expected_dataset_lock_sha256": EXPECTED_DATASET_LOCK_SHA256,
        "diagnostic_selection_sha256": _sha256_or_none(selection_path),
        "expected_diagnostic_selection_sha256": EXPECTED_SELECTION_SHA256,
        "adjudication_worklist_sha256": _sha256_or_none(worklist_path),
        "adjudication_gate_sha256": _sha256_or_none(gate_path),
        "adjudicated_gold_schema_sha256": _sha256_or_none(gold_schema_path),
        "expected_adjudicated_gold_schema_sha256": (EXPECTED_ADJUDICATED_GOLD_SCHEMA_SHA256),
        "candidate_schema_sha256": _vision_schema_sha256(),
        "expected_candidate_schema_sha256": EXPECTED_VISION_SCHEMA_SHA256,
        "prompt_sha256": _vision_prompt_sha256(),
        "expected_prompt_sha256": EXPECTED_VISION_PROMPT_SHA256,
        "redaction": {
            "api_key": "OMITTED",
            "image_bytes": "OMITTED",
            "ocr_text": "OMITTED",
            "extracted_values": "OMITTED",
            "prompt": "OMITTED",
            "raw_response": "OMITTED",
            "absolute_paths": "OMITTED",
        },
    }


def _safe_live_gate_report(
    *,
    failures: list[str],
    gold_path: Path,
    lock_path: Path,
    selection_path: Path,
    document_count: int,
    estimated_input_tokens: int,
    max_calls: int,
    max_total_input_tokens: int,
    max_total_output_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": "claude-vision-evaluation-report/v1",
        "scope": "ADJUDICATED_SYNTHETIC_IMAGE_FIELD_EVAL_NOT_SYSTEM_RELEASE",
        "status": "FAIL",
        "release_score_eligible": False,
        "provider_called": False,
        "call_accounting": {
            "attempted": 0,
            "succeeded": 0,
            "retried": 0,
        },
        "failures": [{"document_id": "preflight", "code": code} for code in sorted(set(failures))],
        "gold_manifest_sha256": _sha256_or_none(gold_path),
        "dataset_lock_sha256": _sha256_or_none(lock_path),
        "diagnostic_selection_sha256": _sha256_or_none(selection_path),
        "documents_selected": document_count,
        "cost_budget": {
            "max_calls": max_calls,
            "max_total_input_tokens": max_total_input_tokens,
            "max_total_output_tokens": max_total_output_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "reserved_crop_calls": MAX_CROP_CALLS_PER_RUN,
            "configured_max_retries": 0,
        },
        "redaction": {
            "api_key": "OMITTED",
            "image_bytes": "OMITTED",
            "ocr_text": "OMITTED",
            "extracted_values": "OMITTED",
            "prompt": "OMITTED",
            "raw_response": "OMITTED",
            "absolute_paths": "OMITTED",
        },
    }


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
) -> ClaudeDocumentAnalyzer:
    return ClaudeDocumentAnalyzer.from_settings(
        settings,
        ocr_layout_provider=provider,
    )


def _regular_file(path: Path) -> Path | None:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or resolved.is_symlink() or path.is_symlink():
        return None
    return resolved


def _release_denominator_is_exact(
    *,
    requested_case_id: str | None,
    requested_document_ids: list[str] | None,
    max_documents: int,
    case: Mapping[str, Any],
    available_documents: Sequence[Mapping[str, Any]],
    selected_documents: Sequence[Mapping[str, Any]],
) -> bool:
    case_ids = case.get("case_ids")
    if not isinstance(case_ids, list):
        return False
    available_ids = [str(item["document_id"]) for item in available_documents]
    selected_ids = [str(item["document_id"]) for item in selected_documents]
    return (
        requested_case_id is None
        and requested_document_ids is None
        and len(case_ids) == RELEASE_CASE_COUNT
        and len(set(map(str, case_ids))) == RELEASE_CASE_COUNT
        and len(available_ids) == RELEASE_DOCUMENT_COUNT
        and len(set(available_ids)) == RELEASE_DOCUMENT_COUNT
        and max_documents == RELEASE_DOCUMENT_COUNT
        and selected_ids == available_ids
    )


def _score_status(*, passed: bool, release_score_eligible: bool) -> str:
    if not passed:
        return "FAIL"
    return "PASS" if release_score_eligible else "DIAGNOSTIC_PASS"


def evaluate(
    *,
    env_file: Path,
    dataset_root: Path,
    gold_path: Path = DEFAULT_GOLD_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    worklist_path: Path | None = None,
    gate_path: Path | None = None,
    gold_schema_path: Path = DEFAULT_ADJUDICATED_GOLD_SCHEMA_PATH,
    case_id: str | None = None,
    max_documents: int = 3,
    document_ids: list[str] | None = None,
    run_workflow: bool = False,
    live: bool = False,
    ocr_provider: ProviderName | None = None,
    ocr_python: Path | None = None,
    ocr_release_gate_report: Path | None = None,
    ocr_release_gate_report_sha256: str | None = None,
    ocr_release_descriptor_path: Path | None = None,
    ocr_release_rendered_root: Path | None = None,
    expected_agent_wheel_sha256: str | None = None,
    ocr_environment_lock: Path | None = None,
    ocr_environment_lock_sha256: str | None = None,
    ocr_model_assets_lock: Path | None = None,
    ocr_model_assets_lock_sha256: str | None = None,
    max_calls: int = 0,
    max_total_input_tokens: int = 0,
    max_total_output_tokens: int = 0,
    provider_factory: ProviderFactory = _default_provider_factory,
    analyzer_factory: AnalyzerFactory = _default_analyzer_factory,
    wheel_attestor: WheelAttestor = installed_agent_wheel_sha256,
    environment_attestor: EnvironmentAttestor = installed_ocr_environment_lock_sha256,
    model_assets_attestor: ModelAssetsAttestor = verify_ocr_model_assets,
) -> tuple[int, dict[str, Any]]:
    resolved_worklist_path, resolved_gate_path = _adjudication_paths(
        dataset_root,
        worklist_path,
        gate_path,
    )
    failures, case, dataset_lock, _image_index = _preflight(
        dataset_root=dataset_root,
        gold_path=gold_path,
        lock_path=lock_path,
        selection_path=selection_path,
        worklist_path=resolved_worklist_path,
        gate_path=resolved_gate_path,
        gold_schema_path=gold_schema_path,
        case_id=case_id,
        run_workflow=run_workflow,
    )
    if failures:
        return 1, _safe_preflight_report(
            failures=failures,
            gold_path=gold_path,
            lock_path=lock_path,
            selection_path=selection_path,
            worklist_path=resolved_worklist_path,
            gate_path=resolved_gate_path,
            gold_schema_path=gold_schema_path,
        )

    effective_case_id = str(case["case_id"])
    available_documents = case["documents"]
    available_ids = {str(item["document_id"]) for item in available_documents}
    if document_ids:
        unknown_ids = sorted(set(document_ids) - available_ids)
        if unknown_ids:
            raise ValueError("document_ids contains a document outside the scored case")
        selected_documents = [
            item for item in available_documents if str(item["document_id"]) in set(document_ids)
        ]
    else:
        if max_documents < 1 or max_documents > len(available_documents):
            raise ValueError("max_documents must select 1..all scored case documents")
        selected_documents = available_documents[:max_documents]
    release_score_eligible = _release_denominator_is_exact(
        requested_case_id=case_id,
        requested_document_ids=document_ids,
        max_documents=max_documents,
        case=case,
        available_documents=available_documents,
        selected_documents=selected_documents,
    )
    if run_workflow and len(selected_documents) != len(available_documents):
        raise ValueError("workflow evaluation requires every document in the scored case")
    documents = [
        DocumentInputV2(
            document_id=str(item["document_id"]),
            filename=Path(str(item["relative_path"])).name,
            image_pages=[
                image_page_from_file(
                    dataset_root / str(item["relative_path"]),
                    page_number=1,
                )
            ],
        )
        for item in selected_documents
    ]
    estimated_input_tokens = sum(
        sum((page.width_px * page.height_px + 749) // 750 for page in document.image_pages) + 4_096
        for document in documents
    )
    required_call_ceiling = len(documents) + MAX_CROP_CALLS_PER_RUN

    def fail_live_gate(failure_codes: list[str]) -> tuple[int, dict[str, Any]]:
        return 1, _safe_live_gate_report(
            failures=failure_codes,
            gold_path=gold_path,
            lock_path=lock_path,
            selection_path=selection_path,
            document_count=len(documents),
            estimated_input_tokens=estimated_input_tokens,
            max_calls=max_calls,
            max_total_input_tokens=max_total_input_tokens,
            max_total_output_tokens=max_total_output_tokens,
        )

    live_gate_failures: list[str] = []
    if not live:
        live_gate_failures.append("live_flag_required")
    if ocr_provider is None:
        live_gate_failures.append("selected_ocr_provider_required")
    if ocr_python is None:
        live_gate_failures.append("ocr_python_required")
    if max_calls < required_call_ceiling:
        live_gate_failures.append("max_calls_budget_too_small")
    if max_total_input_tokens < estimated_input_tokens:
        live_gate_failures.append("input_token_budget_too_small")
    if max_total_output_tokens <= 0:
        live_gate_failures.append("output_token_budget_required")
    if live:
        if ocr_release_gate_report is None:
            live_gate_failures.append("ocr_release_gate_report_required")
        if ocr_release_gate_report_sha256 is None:
            live_gate_failures.append("ocr_release_gate_report_hash_required")
        if ocr_release_descriptor_path is None:
            live_gate_failures.append("ocr_release_descriptor_required")
        if ocr_release_rendered_root is None:
            live_gate_failures.append("ocr_release_rendered_root_required")
        if expected_agent_wheel_sha256 is None:
            live_gate_failures.append("expected_agent_wheel_hash_required")
        if ocr_environment_lock is None:
            live_gate_failures.append("ocr_environment_lock_required")
        if ocr_environment_lock_sha256 is None:
            live_gate_failures.append("ocr_environment_lock_hash_required")
        if ocr_model_assets_lock is None:
            live_gate_failures.append("ocr_model_assets_lock_required")
        if ocr_model_assets_lock_sha256 is None:
            live_gate_failures.append("ocr_model_assets_lock_hash_required")
    if live_gate_failures:
        return fail_live_gate(live_gate_failures)

    # These values were checked above, but the executable boundary must remain
    # fail-closed under ``python -O`` as well. Runtime assertions would vanish.
    if (
        ocr_provider is None
        or ocr_python is None
        or ocr_release_gate_report is None
        or ocr_release_gate_report_sha256 is None
        or ocr_release_descriptor_path is None
        or ocr_release_rendered_root is None
        or expected_agent_wheel_sha256 is None
        or ocr_environment_lock is None
        or ocr_environment_lock_sha256 is None
        or ocr_model_assets_lock is None
        or ocr_model_assets_lock_sha256 is None
    ):
        return fail_live_gate(["live_gate_internal_invariant_failed"])
    try:
        descriptor_sha256, manifest_sha256, _, _ = load_benchmark(
            descriptor_path=ocr_release_descriptor_path,
            rendered_root=ocr_release_rendered_root,
            sample="locked",
        )
    except OcrBenchmarkError:
        return fail_live_gate(["ocr_release_gate_canonical_input_invalid"])
    executable = _regular_file(ocr_python)
    if executable is None:
        return fail_live_gate(["ocr_python_unavailable"])
    try:
        if release_score_eligible:
            release_lock = load_canonical_ocr_release_gate(
                report_path=ocr_release_gate_report,
                expected_report_sha256=ocr_release_gate_report_sha256,
                expected_provider=ocr_provider,
                descriptor_path=ocr_release_descriptor_path,
                rendered_root=ocr_release_rendered_root,
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
            release_lock = load_ocr_release_runtime_gate(
                report_path=ocr_release_gate_report,
                expected_report_sha256=ocr_release_gate_report_sha256,
                expected_provider=ocr_provider,
                expected_descriptor_sha256=descriptor_sha256,
                expected_manifest_sha256=manifest_sha256,
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
        return fail_live_gate([exc.code])

    settings = _live_settings(env_file)
    configured_output_ceiling = (
        settings.max_tokens * len(documents) + 1_024 * MAX_CROP_CALLS_PER_RUN
    )
    if configured_output_ceiling > max_total_output_tokens:
        return fail_live_gate(["configured_output_ceiling_exceeds_budget"])
    try:
        provider = ReleaseLockedOcrProvider(
            provider_factory(
                ocr_provider,
                executable,
            ),
            release_lock,
        )
    except OcrLayoutProviderError as exc:
        return fail_live_gate([exc.code])
    except Exception:
        return fail_live_gate(["ocr_provider_initialization_failed"])
    try:
        analyzer = analyzer_factory(settings, provider)
    except LLMError as exc:
        return fail_live_gate([exc.code])
    except Exception:
        return fail_live_gate(["claude_analyzer_initialization_failed"])
    crop_budget = CropReadBudget(max_calls=MAX_CROP_CALLS_PER_RUN)
    analyses: dict[str, DocumentAnalysis] = {}
    model_failures: list[dict[str, str]] = []
    workflow_summary: dict[str, Any] | None = None
    main_calls_attempted = 0

    def analyze_once(document: Mapping[str, Any]) -> DocumentAnalysis:
        nonlocal main_calls_attempted
        if main_calls_attempted + crop_budget.used_calls >= max_calls:
            raise LLMError(
                "call_budget_exceeded",
                "The live evaluation call budget is exhausted.",
            )
        main_calls_attempted += 1
        return analyzer.analyze(document)

    if run_workflow:

        class RecordingAnalyzer:
            @property
            def model_id(self) -> str:
                return analyzer.model_id

            def analyze(self, document: Mapping[str, Any]) -> DocumentAnalysis:
                runtime_document = dict(document)
                runtime_document["_runtime_crop_budget"] = crop_budget
                analysis = analyze_once(runtime_document)
                analyses[str(document["document_id"])] = analysis
                return analysis

        state = run_review_v2(
            ReviewRunRequestV2(
                case_id=f"synthetic-image-{effective_case_id}",
                documents=documents,
                llm_mode="llm",
            ),
            retriever=MockRegulationRetriever(),
            analyzer=RecordingAnalyzer(),
        )
        model_failures.extend(
            {
                "document_id": str(item["document_id"]),
                "code": str(item["code"]),
            }
            for item in state["model_errors"]
        )
        workflow_summary = {
            "retrieval_mode": "EXPLICIT_MOCK_NOT_LIVE_RAG",
            "status": state["final_status"],
            "requires_human_review": state["requires_human_review"],
            "validation_issue_codes": sorted(
                {str(item["issue_code"]) for item in state["validation_issues"]}
            ),
            "model_call_count": len(state["model_calls"]),
            "model_error_count": len(state["model_errors"]),
            "retrieval_status": state["retrieval_status"],
            "retrieval_coverage": state["retrieval_coverage"],
        }

    expected_by_id = {str(item["document_id"]): item for item in selected_documents}
    type_correct = 0
    expected_fields = 0
    exact_fields = 0
    wrong_fields = 0
    missing_fields = 0
    expected_line_counts = 0
    exact_line_counts = 0
    evidence_items = 0
    calls: list[dict[str, Any]] = []
    field_comparison_failures: list[dict[str, str]] = []
    line_count_failures: list[str] = []

    for document in documents:
        document_id = document.document_id
        analysis = analyses.get(document_id)
        if analysis is None and not run_workflow:
            try:
                runtime_document = document.model_dump(mode="json")
                runtime_document["_runtime_crop_budget"] = crop_budget
                analysis = analyze_once(runtime_document)
            except LLMError as exc:
                model_failures.append({"document_id": document_id, "code": exc.code})
                continue
        if analysis is None:
            continue
        if not isinstance(analysis.candidate, ClaudeVisionDocumentCandidate):
            model_failures.append(
                {"document_id": document_id, "code": "candidate_modality_mismatch"}
            )
            continue

        expected = expected_by_id[document_id]
        candidate = analysis.candidate
        type_correct += int(candidate.document_type == expected["expected_document_type"])
        actual_fields = {field.name: field.value for field in candidate.fields}
        for name, value in expected.get("expected_fields", {}).items():
            expected_fields += 1
            if name not in actual_fields:
                missing_fields += 1
                field_comparison_failures.append(
                    {
                        "document_id": document_id,
                        "field_name": name,
                        "status": "missing",
                    }
                )
            elif actual_fields[name] == value:
                exact_fields += 1
            else:
                wrong_fields += 1
                field_comparison_failures.append(
                    {
                        "document_id": document_id,
                        "field_name": name,
                        "status": "wrong_exact_value",
                    }
                )
        expected_line_counts += 1
        line_count_exact = len(candidate.line_items) == expected["expected_line_item_count"]
        exact_line_counts += int(line_count_exact)
        if not line_count_exact:
            line_count_failures.append(document_id)
        evidence_items += len(candidate.fields)
        evidence_items += sum(len(item.fields) for item in candidate.line_items)

        for metadata in (analysis.metadata, *analysis.additional_model_calls):
            calls.append(
                {
                    "document_id": document_id,
                    "provider": metadata.provider,
                    "model_id": metadata.model_id,
                    "prompt_id": metadata.prompt_id,
                    "prompt_version": metadata.prompt_version,
                    "prompt_sha256": metadata.prompt_sha256,
                    "output_schema_id": metadata.output_schema_id,
                    "output_schema_version": metadata.output_schema_version,
                    "output_schema_sha256": metadata.output_schema_sha256,
                    "request_id": metadata.request_id,
                    "message_id": metadata.message_id,
                    "stop_reason": metadata.stop_reason,
                    "input_modality": metadata.input_modality,
                    "input_tokens": metadata.input_tokens,
                    "output_tokens": metadata.output_tokens,
                    "latency_ms": metadata.latency_ms,
                    "ocr_ledger_id": metadata.ocr_ledger_id,
                    "ocr_provider_id": metadata.ocr_provider_id,
                    "ocr_engine_id": metadata.ocr_engine_id,
                    "ocr_engine_version": metadata.ocr_engine_version,
                    "ocr_model_id": metadata.ocr_model_id,
                    "ocr_model_assets_lock_sha256": metadata.ocr_model_assets_lock_sha256,
                }
            )

    document_count = len(documents)
    main_calls = [call for call in calls if call["prompt_id"] == DOCUMENT_VISION_ANALYSIS_PROMPT_ID]
    usage = {
        "input_tokens": sum(int(call["input_tokens"]) for call in calls),
        "output_tokens": sum(int(call["output_tokens"]) for call in calls),
        "latency_ms": sum(int(call["latency_ms"]) for call in calls),
    }
    total_calls_attempted = main_calls_attempted + crop_budget.used_calls
    if total_calls_attempted > max_calls:
        model_failures.append({"document_id": "budget", "code": "call_budget_exceeded"})
    if usage["input_tokens"] > max_total_input_tokens:
        model_failures.append({"document_id": "budget", "code": "input_token_budget_exceeded"})
    if usage["output_tokens"] > max_total_output_tokens:
        model_failures.append({"document_id": "budget", "code": "output_token_budget_exceeded"})
    try:
        close_provider = getattr(provider, "close", None)
        if callable(close_provider):
            close_provider()
    except Exception:
        model_failures.append({"document_id": "ocr", "code": "ocr_provider_close_failed"})
    passed = (
        not model_failures
        and len(main_calls) == document_count
        and type_correct == document_count
        and exact_fields == expected_fields
        and wrong_fields == 0
        and missing_fields == 0
        and exact_line_counts == expected_line_counts
        and evidence_items > 0
        and all(
            call["provider"] == "anthropic"
            and call["model_id"] == DEFAULT_CLAUDE_MODEL
            and (
                (
                    call["prompt_id"] == DOCUMENT_VISION_ANALYSIS_PROMPT_ID
                    and call["prompt_version"] == DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION
                    and call["input_modality"] == "image"
                )
                or (
                    call["prompt_id"] == DOCUMENT_CROP_RECONCILIATION_PROMPT_ID
                    and call["prompt_version"] == DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION
                    and call["input_modality"] == "image_crop"
                )
            )
            and call["ocr_ledger_id"]
            and call["ocr_provider_id"] == release_lock.provider_id
            and call["ocr_engine_id"] == release_lock.engine_id
            and call["ocr_engine_version"] == release_lock.engine_version
            and call["ocr_model_id"] == OCR_MODEL_SET_IDS[release_lock.provider]
            and call["ocr_model_assets_lock_sha256"] == OCR_MODEL_ASSETS_LOCK_SHA256
            for call in calls
        )
        and workflow_summary is None
    )
    report = {
        "schema_version": "claude-vision-evaluation-report/v1",
        "scope": "ADJUDICATED_SYNTHETIC_IMAGE_FIELD_EVAL_NOT_SYSTEM_RELEASE",
        "dataset_id": dataset_lock["dataset_id"],
        "dataset_version": dataset_lock["dataset_version"],
        "dataset_status": dataset_lock["status"],
        "gold_set_id": json.loads(gold_path.read_text(encoding="utf-8"))["gold_set_id"],
        "case_id": effective_case_id,
        "case_ids": [str(value) for value in case.get("case_ids", [effective_case_id])],
        "release_score_eligible": release_score_eligible,
        "evaluation_mode": "release" if release_score_eligible else "diagnostic",
        "system_release_decision_eligible": False,
        "provider_called": total_calls_attempted > 0,
        "documents_attempted": document_count,
        "documents_completed": len(main_calls),
        "release_denominator": {
            "expected_case_count": RELEASE_CASE_COUNT,
            "expected_document_count": RELEASE_DOCUMENT_COUNT,
            "available_document_count": len(available_documents),
            "selected_document_count": document_count,
        },
        "document_type_correct": type_correct,
        "field_metrics": {
            "expected": expected_fields,
            "exact": exact_fields,
            "wrong": wrong_fields,
            "missing": missing_fields,
            "unscored_returned_fields": True,
        },
        "line_item_count_metrics": {
            "expected_documents": expected_line_counts,
            "exact_documents": exact_line_counts,
            "failed_document_ids": line_count_failures,
        },
        "field_comparison_failures": field_comparison_failures,
        "evidence_items_validated": evidence_items,
        "calls": calls,
        "usage": usage,
        "cost_budget": {
            "max_calls": max_calls,
            "max_total_input_tokens": max_total_input_tokens,
            "max_total_output_tokens": max_total_output_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "configured_output_ceiling": configured_output_ceiling,
            "reserved_crop_calls": MAX_CROP_CALLS_PER_RUN,
            "configured_max_retries": settings.max_retries,
        },
        "call_accounting": {
            "attempted": total_calls_attempted,
            "succeeded": len(calls),
            "retried": 0,
            "main_attempted": main_calls_attempted,
            "crop_attempted": crop_budget.used_calls,
        },
        "ocr": {
            "release_gate_report_sha256": release_lock.report_sha256,
            "descriptor_sha256": release_lock.descriptor_sha256,
            "manifest_sha256": release_lock.manifest_sha256,
            "agent_wheel_sha256": release_lock.agent_wheel_sha256,
            "environment_lock_sha256": release_lock.environment_lock_sha256,
            "model_assets_lock_sha256": release_lock.model_assets_lock_sha256,
            "selected_provider": release_lock.provider,
            "provider_id": release_lock.provider_id,
            "engine_id": release_lock.engine_id,
            "engine_version": release_lock.engine_version,
            "model_set_id": OCR_MODEL_SET_IDS[release_lock.provider],
            "python_executable_sha256": _sha256(executable),
            "ledger_count": len(
                {str(call["ocr_ledger_id"]) for call in main_calls if call["ocr_ledger_id"]}
            ),
            "engines": sorted(
                {
                    (
                        str(call["ocr_engine_id"]),
                        str(call["ocr_engine_version"]),
                    )
                    for call in main_calls
                    if call["ocr_engine_id"] and call["ocr_engine_version"]
                }
            ),
        },
        "workflow": workflow_summary,
        "failures": model_failures,
        "status": _score_status(
            passed=passed,
            release_score_eligible=release_score_eligible,
        ),
        "claim_limits": dataset_lock["unsupported_claims"],
        "redaction": {
            "api_key": "OMITTED",
            "image_bytes": "OMITTED",
            "ocr_text": "OMITTED",
            "extracted_values": "OMITTED",
            "prompt": "OMITTED",
            "raw_response": "OMITTED",
            "absolute_paths": "OMITTED",
        },
    }
    return (0 if passed else 1), report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Claude Vision only with independently human-reviewed replacement "
            "Gold whose exact hash is approved. Current historical image Gold "
            "artifacts are always blocked before provider creation."
        ),
    )
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gold", default=DEFAULT_GOLD_PATH, type=Path)
    parser.add_argument("--dataset-lock", default=DEFAULT_LOCK_PATH, type=Path)
    parser.add_argument(
        "--selection",
        default=DEFAULT_SELECTION_PATH,
        type=Path,
    )
    parser.add_argument(
        "--adjudication-worklist",
        type=Path,
        help="Defaults to <dataset-root>/../derived/diagnostic-selection/v1/.",
    )
    parser.add_argument(
        "--adjudication-gate",
        type=Path,
        help="Defaults to <dataset-root>/../derived/diagnostic-selection/v1/.",
    )
    parser.add_argument(
        "--gold-schema",
        default=DEFAULT_ADJUDICATED_GOLD_SCHEMA_PATH,
        type=Path,
    )
    parser.add_argument("--case-id")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=3,
        help=(
            "1..36 documents. Values below the exact 36-document denominator are "
            "diagnostic-only; release PASS requires 36 with no case/document filter."
        ),
    )
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help="Evaluate only the named scored document; repeat to select multiple documents.",
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help=(
            "Fail closed: workflow scoring needs a separate adjudicated outcome Gold "
            "and is not supported by this field-extraction evaluator."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to permit Anthropic API calls.",
    )
    parser.add_argument(
        "--ocr-provider",
        choices=("paddle", "docling"),
        required=True,
        help="Must match the provider selected by the combined 60+20 OCR release Gate.",
    )
    parser.add_argument(
        "--ocr-python",
        type=Path,
        required=True,
        help="Python executable in the isolated, wheel-installed OCR environment.",
    )
    parser.add_argument(
        "--ocr-release-gate-report",
        type=Path,
        required=True,
        help="Exact combined 60 locked + 20 C/O OCR release Gate report.",
    )
    parser.add_argument(
        "--ocr-release-gate-report-sha256",
        required=True,
        help="Explicit SHA-256 pin for the combined OCR release Gate report.",
    )
    parser.add_argument(
        "--ocr-release-descriptor",
        type=Path,
        required=True,
        help="Canonical OCR benchmark descriptor referenced by the release Gate.",
    )
    parser.add_argument(
        "--ocr-release-rendered-root",
        type=Path,
        required=True,
        help="Canonical rendered benchmark root referenced by the release Gate.",
    )
    parser.add_argument(
        "--expected-agent-wheel-sha256",
        required=True,
        help="Explicit Agent wheel SHA-256 expected in the Gate and OCR environment.",
    )
    parser.add_argument("--ocr-environment-lock", type=Path, required=True)
    parser.add_argument("--ocr-environment-lock-sha256", required=True)
    parser.add_argument("--ocr-model-assets-lock", type=Path, required=True)
    parser.add_argument("--ocr-model-assets-lock-sha256", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--max-total-input-tokens", type=int, required=True)
    parser.add_argument("--max-total-output-tokens", type=int, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    exit_code, report = evaluate(
        env_file=args.env_file,
        dataset_root=args.dataset_root,
        gold_path=args.gold,
        lock_path=args.dataset_lock,
        selection_path=args.selection,
        worklist_path=args.adjudication_worklist,
        gate_path=args.adjudication_gate,
        gold_schema_path=args.gold_schema,
        case_id=args.case_id,
        max_documents=args.max_documents,
        document_ids=args.document_ids,
        run_workflow=args.workflow,
        live=args.live,
        ocr_provider=args.ocr_provider,
        ocr_python=args.ocr_python,
        ocr_release_gate_report=args.ocr_release_gate_report,
        ocr_release_gate_report_sha256=args.ocr_release_gate_report_sha256,
        ocr_release_descriptor_path=args.ocr_release_descriptor,
        ocr_release_rendered_root=args.ocr_release_rendered_root,
        expected_agent_wheel_sha256=args.expected_agent_wheel_sha256,
        ocr_environment_lock=args.ocr_environment_lock,
        ocr_environment_lock_sha256=args.ocr_environment_lock_sha256,
        ocr_model_assets_lock=args.ocr_model_assets_lock,
        ocr_model_assets_lock_sha256=args.ocr_model_assets_lock_sha256,
        max_calls=args.max_calls,
        max_total_input_tokens=args.max_total_input_tokens,
        max_total_output_tokens=args.max_total_output_tokens,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
