"""Reconcile two independent AI visual reviews into artifact-only image Gold.

The two supported annotation layouts intentionally preserve the reviewers'
visible observations.  This tool only normalizes structural aliases; it never
repairs, derives, or reconciles business values.  Any conclusion or normalized
observation disagreement fails closed and requires a separate adjudication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

try:
    from tools.build_trade_image_diagnostic_selection import (
        ADJUDICATION_SCHEMA,
        ALLOWED_CONCLUSIONS,
        DiagnosticSelectionError,
        SELECTION_COUNT,
        SELECTION_ID,
        SELECTION_SCHEMA,
        evaluate_gate,
    )
except ModuleNotFoundError:  # Support direct `python tools/<script>.py` execution.
    from build_trade_image_diagnostic_selection import (  # type: ignore[no-redef]
        ADJUDICATION_SCHEMA,
        ALLOWED_CONCLUSIONS,
        DiagnosticSelectionError,
        SELECTION_COUNT,
        SELECTION_ID,
        SELECTION_SCHEMA,
        evaluate_gate,
    )


GOLD_SCHEMA_VERSION = "trade-image-adjudicated-gold/v1"
REPORT_SCHEMA_VERSION = "trade-image-review-reconciliation-report/v1"
REVIEWER_KIND = "independent_ai_visual_review"
REVIEW_SCOPE = "diagnostic_full_triplet"
GATE_PASSED_STATUS = "passed_two_reviewer_adjudication"
RELEASE_SCORING_SCOPE = "claude_vision_field_extraction_only"
DEFAULT_GOLD_SET_ID = "trade-images-adjudicated-gold-source-v1"
EXPECTED_SELECTION_SHA256 = "8959d3e91affb1c51fbb320cf90cea7349d069cd12b58858d27ea13fb92ba961"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_PATH = (
    REPOSITORY_ROOT / "evals" / "trade_documents_images" / "v1" / "diagnostic-selection-v1.json"
)
DEFAULT_DIAGNOSTIC_SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "evals"
    / "trade_documents_images"
    / "v1"
    / "schemas"
    / "diagnostic-adjudication.schema.json"
)
DEFAULT_GOLD_SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "evals"
    / "trade_documents_images"
    / "v1"
    / "schemas"
    / "adjudicated-gold.schema.json"
)

DOCUMENT_ORDER = (
    "commercial_invoice",
    "packing_list",
    "bill_of_lading",
)
EXPECTED_DOCUMENT_TYPES = {
    "commercial_invoice": "invoice",
    "packing_list": "packing_list",
    "bill_of_lading": "bill_of_lading",
}
VISIBLE_DOCUMENT_TITLES = {
    "COMMERCIAL INVOICE": "invoice",
    "PACKING LIST": "packing_list",
    "BILL OF LADING": "bill_of_lading",
}
FORMAT_A_FIELD_ALIASES = {
    "commercial_invoice": {
        "invoice_no": "invoice_no",
        "buyer_consignee": "buyer",
    },
    "packing_list": {
        "invoice_no": "invoice_no",
        "bl_no": "bl_no",
    },
    "bill_of_lading": {
        "bl_no": "bl_no",
        "consignee": "consignee",
    },
}
FORMAT_B_FIELD_ALIASES = {
    "commercial_invoice": {
        "invoice_no": "invoice_no",
        "buyer_or_consignee": "buyer",
    },
    "packing_list": {
        "invoice_no": "invoice_no",
        "bl_no": "bl_no",
    },
    "bill_of_lading": {
        "bl_no": "bl_no",
        "consignee": "consignee",
    },
}
FORMAT_A_DOCUMENT_KEYS = {
    "source_document_type",
    "relative_path",
    "source_image_sha256",
    "readability",
    "observed_document_type",
    "observed_document_type_text",
    "observed_fields",
    "visible_line_item_count",
}
FORMAT_B_METADATA_KEYS = {
    "sha256",
    "readability",
    "document_type",
    "visible_line_item_count",
}
OUTPUT_FILENAMES = {
    "worklist": "adjudications.jsonl",
    "gate": "gate.json",
    "gold": "adjudicated-gold.json",
    "report": "reconciliation-report.json",
}
MAX_JSON_BYTES = 32 * 1024 * 1024


class ReviewReconciliationError(ValueError):
    """Raised when review evidence cannot safely produce adjudicated Gold."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n"
    ).encode()


def _is_lower_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _normalized_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ReviewReconciliationError(f"{label} must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized:
        raise ReviewReconciliationError(f"{label} must not be empty")
    return normalized


def _normalized_title(value: Any, *, label: str) -> str:
    return _normalized_text(value, label=label).upper()


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ReviewReconciliationError(f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewReconciliationError(f"cannot read {label}") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise ReviewReconciliationError(f"{label} exceeds the bounded input size")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewReconciliationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReviewReconciliationError(f"{label} root must be an object")
    return value, raw


def _read_annotation_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ReviewReconciliationError(f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewReconciliationError(f"cannot read {label}") from exc
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ReviewReconciliationError(f"{label} is empty or exceeds the bounded input size")
    try:
        lines = raw.decode("utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewReconciliationError(f"{label} is not valid UTF-8 JSONL") from exc
    if not records or any(not isinstance(record, dict) for record in records):
        raise ReviewReconciliationError(f"{label} must contain JSON objects")
    case_ids = [record.get("case_id") for record in records]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ReviewReconciliationError(f"{label} contains an invalid case ID")
    if len(set(case_ids)) != len(case_ids):
        raise ReviewReconciliationError(f"{label} contains duplicate case IDs")
    return records, raw


def _validate_selection(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ReviewReconciliationError("selection schema version is not pinned")
    if selection.get("selection_id") != SELECTION_ID:
        raise ReviewReconciliationError("selection ID is not pinned")
    if (
        selection.get("selection_role") != "diagnostic_candidate_not_gold"
        or selection.get("gold_eligible") is not False
    ):
        raise ReviewReconciliationError("selection must remain a non-Gold diagnostic candidate")
    cases = selection.get("cases")
    if not isinstance(cases, list) or len(cases) != SELECTION_COUNT:
        raise ReviewReconciliationError("selection must contain exactly 12 cases")
    seen_case_ids: set[str] = set()
    for selected in cases:
        if not isinstance(selected, dict):
            raise ReviewReconciliationError("selection contains a non-object case")
        case_id = selected.get("case_id")
        if (
            not isinstance(case_id, str)
            or len(case_id) != 4
            or not case_id.isdigit()
            or case_id in seen_case_ids
        ):
            raise ReviewReconciliationError("selection contains an invalid or duplicate case ID")
        seen_case_ids.add(case_id)
        if not _is_lower_hex_sha256(selected.get("triplet_fingerprint_sha256")):
            raise ReviewReconciliationError(f"case {case_id} has an invalid fingerprint")
        documents = selected.get("documents")
        if not isinstance(documents, dict) or set(documents) != set(DOCUMENT_ORDER):
            raise ReviewReconciliationError(f"case {case_id} must contain the selected triplet")
        for document_type in DOCUMENT_ORDER:
            document = documents[document_type]
            if not isinstance(document, dict) or not _is_lower_hex_sha256(document.get("sha256")):
                raise ReviewReconciliationError(
                    f"case {case_id} has invalid {document_type} hash evidence"
                )
            relative_path = document.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ReviewReconciliationError(
                    f"case {case_id} has an invalid {document_type} relative path"
                )
            parsed_path = PurePosixPath(relative_path)
            if parsed_path.is_absolute() or ".." in parsed_path.parts:
                raise ReviewReconciliationError(
                    f"case {case_id} has an unsafe {document_type} relative path"
                )
    return cases


def _validate_schema(value: Any, schema: dict[str, Any], *, label: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ReviewReconciliationError(f"{label} schema is invalid") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    first = errors[0]
    location = "/".join(str(part) for part in first.absolute_path) or "<root>"
    raise ReviewReconciliationError(
        f"{label} schema validation failed at {location} ({first.validator})"
    )


def _selected_hashes(selected: dict[str, Any]) -> list[str]:
    documents = selected["documents"]
    return [documents[document_type]["sha256"] for document_type in DOCUMENT_ORDER]


def _normalize_fields(
    raw_fields: dict[str, Any],
    aliases: dict[str, str],
    *,
    case_id: str,
    document_type: str,
) -> dict[str, str]:
    if set(raw_fields) != set(aliases):
        raise ReviewReconciliationError(
            f"case {case_id} {document_type} contains an unexpected field set"
        )
    result = {}
    for source_name, agent_name in aliases.items():
        result[agent_name] = _normalized_text(
            raw_fields[source_name],
            label=f"case {case_id} {document_type} {source_name}",
        )
    return dict(sorted(result.items()))


def _normalize_format_a_documents(
    raw_documents: list[Any],
    selected: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    case_id = selected["case_id"]
    by_type: dict[str, dict[str, Any]] = {}
    for raw_document in raw_documents:
        if not isinstance(raw_document, dict) or set(raw_document) != FORMAT_A_DOCUMENT_KEYS:
            raise ReviewReconciliationError(f"case {case_id} format-A document shape is invalid")
        document_type = raw_document.get("source_document_type")
        if document_type not in DOCUMENT_ORDER or document_type in by_type:
            raise ReviewReconciliationError(
                f"case {case_id} format-A document types are invalid or duplicated"
            )
        by_type[document_type] = raw_document
    if set(by_type) != set(DOCUMENT_ORDER):
        raise ReviewReconciliationError(f"case {case_id} format-A triplet is incomplete")

    normalized = {}
    for document_type in DOCUMENT_ORDER:
        raw_document = by_type[document_type]
        selected_document = selected["documents"][document_type]
        if raw_document["relative_path"] != selected_document["relative_path"]:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} path does not match selection"
            )
        if raw_document["source_image_sha256"] != selected_document["sha256"]:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} hash does not match selection"
            )
        expected_type = EXPECTED_DOCUMENT_TYPES[document_type]
        if raw_document["observed_document_type"] not in {
            document_type,
            expected_type,
        }:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} normalized type is invalid"
            )
        visible_title = _normalized_title(
            raw_document["observed_document_type_text"],
            label=f"case {case_id} {document_type} visible title",
        )
        if VISIBLE_DOCUMENT_TITLES.get(visible_title) != expected_type:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} visible title is invalid"
            )
        readability = _normalized_text(
            raw_document["readability"],
            label=f"case {case_id} {document_type} readability",
        ).lower()
        if readability != "readable":
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} is not readable enough for value-bearing Gold"
            )
        line_count = raw_document["visible_line_item_count"]
        if isinstance(line_count, bool) or not isinstance(line_count, int) or line_count < 0:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} line-item count is invalid"
            )
        raw_fields = raw_document["observed_fields"]
        if not isinstance(raw_fields, dict):
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} observed fields are invalid"
            )
        normalized[document_type] = {
            "source_document_type": document_type,
            "relative_path": selected_document["relative_path"],
            "source_image_sha256": selected_document["sha256"],
            "readability": readability,
            "expected_document_type": expected_type,
            "expected_fields": _normalize_fields(
                raw_fields,
                FORMAT_A_FIELD_ALIASES[document_type],
                case_id=case_id,
                document_type=document_type,
            ),
            "expected_line_item_count": line_count,
        }
    return normalized


def _normalize_format_b_documents(
    raw_documents: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    case_id = selected["case_id"]
    if set(raw_documents) != set(DOCUMENT_ORDER):
        raise ReviewReconciliationError(f"case {case_id} format-B triplet is incomplete")
    normalized = {}
    for document_type in DOCUMENT_ORDER:
        raw_document = raw_documents[document_type]
        aliases = FORMAT_B_FIELD_ALIASES[document_type]
        expected_keys = FORMAT_B_METADATA_KEYS | set(aliases)
        if not isinstance(raw_document, dict) or set(raw_document) != expected_keys:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} format-B document shape is invalid"
            )
        selected_document = selected["documents"][document_type]
        if raw_document["sha256"] != selected_document["sha256"]:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} hash does not match selection"
            )
        expected_type = EXPECTED_DOCUMENT_TYPES[document_type]
        visible_title = _normalized_title(
            raw_document["document_type"],
            label=f"case {case_id} {document_type} visible title",
        )
        if VISIBLE_DOCUMENT_TITLES.get(visible_title) != expected_type:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} visible title is invalid"
            )
        readability = _normalized_text(
            raw_document["readability"],
            label=f"case {case_id} {document_type} readability",
        ).lower()
        if readability != "readable":
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} is not readable enough for value-bearing Gold"
            )
        line_count = raw_document["visible_line_item_count"]
        if isinstance(line_count, bool) or not isinstance(line_count, int) or line_count < 0:
            raise ReviewReconciliationError(
                f"case {case_id} {document_type} line-item count is invalid"
            )
        raw_fields = {name: raw_document[name] for name in aliases}
        normalized[document_type] = {
            "source_document_type": document_type,
            "relative_path": selected_document["relative_path"],
            "source_image_sha256": selected_document["sha256"],
            "readability": readability,
            "expected_document_type": expected_type,
            "expected_fields": _normalize_fields(
                raw_fields,
                aliases,
                case_id=case_id,
                document_type=document_type,
            ),
            "expected_line_item_count": line_count,
        }
    return normalized


def _normalize_review_record(
    record: dict[str, Any],
    selected: dict[str, Any],
    *,
    annotation_reference: str,
) -> dict[str, Any]:
    case_id = selected["case_id"]
    if record.get("case_id") != case_id:
        raise ReviewReconciliationError(f"case {case_id} review case ID does not match")
    reviewer_id = _normalized_text(
        record.get("reviewer_id"),
        label=f"case {case_id} reviewer ID",
    )
    if record.get("reviewer_kind") != REVIEWER_KIND:
        raise ReviewReconciliationError(f"case {case_id} reviewer kind must be {REVIEWER_KIND}")
    if record.get("independent_review_attested") is not True:
        raise ReviewReconciliationError(f"case {case_id} review lacks the independence attestation")
    if record.get("scope") != REVIEW_SCOPE:
        raise ReviewReconciliationError(f"case {case_id} review scope must be {REVIEW_SCOPE}")
    conclusion = record.get("conclusion")
    if conclusion not in ALLOWED_CONCLUSIONS:
        raise ReviewReconciliationError(f"case {case_id} review conclusion is invalid")
    reviewed_hashes = record.get("reviewed_document_sha256")
    selected_hashes = _selected_hashes(selected)
    if (
        not isinstance(reviewed_hashes, list)
        or len(reviewed_hashes) != len(DOCUMENT_ORDER)
        or len(set(reviewed_hashes)) != len(reviewed_hashes)
        or set(reviewed_hashes) != set(selected_hashes)
    ):
        raise ReviewReconciliationError(
            f"case {case_id} reviewed document hashes do not match the selection"
        )
    raw_documents = record.get("documents")
    if isinstance(raw_documents, list):
        documents = _normalize_format_a_documents(raw_documents, selected)
        source_format = "review-layout-a"
    elif isinstance(raw_documents, dict):
        documents = _normalize_format_b_documents(raw_documents, selected)
        source_format = "review-layout-b"
    else:
        raise ReviewReconciliationError(f"case {case_id} review documents are invalid")
    return {
        "case_id": case_id,
        "reviewer_id": reviewer_id,
        "reviewer_kind": REVIEWER_KIND,
        "conclusion": conclusion,
        "reviewed_document_sha256": selected_hashes,
        "annotation_reference": f"{annotation_reference}#case={case_id}",
        "source_format": source_format,
        "documents": documents,
    }


def _gold_documents(
    case_id: str,
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for document_type in DOCUMENT_ORDER:
        observation = documents[document_type]
        result.append(
            {
                "document_id": f"{case_id}:{observation['expected_document_type']}",
                "source_document_type": document_type,
                "relative_path": observation["relative_path"],
                "source_image_sha256": observation["source_image_sha256"],
                "expected_document_type": observation["expected_document_type"],
                "expected_fields": observation["expected_fields"],
                "expected_line_item_count": observation["expected_line_item_count"],
            }
        )
    return result


def reconcile_review_records(
    selection: dict[str, Any],
    review_sources: list[dict[str, Any]],
    *,
    resolution_reference: str,
) -> dict[str, Any]:
    """Reconcile exactly two review sources without changing observed values."""

    selected_cases = _validate_selection(selection)
    if len(review_sources) != 2:
        raise ReviewReconciliationError("exactly two independent review sources are required")
    source_references = [source.get("annotation_reference") for source in review_sources]
    source_hashes = [source.get("sha256") for source in review_sources]
    if (
        any(not isinstance(reference, str) or not reference for reference in source_references)
        or len(set(source_references)) != 2
    ):
        raise ReviewReconciliationError("review sources must have distinct artifact references")
    if (
        any(not _is_lower_hex_sha256(digest) for digest in source_hashes)
        or len(set(source_hashes)) != 2
    ):
        raise ReviewReconciliationError("review sources must have distinct SHA-256 evidence")

    selected_case_ids = {selected["case_id"] for selected in selected_cases}
    records_by_source: list[dict[str, dict[str, Any]]] = []
    for index, source in enumerate(review_sources, start=1):
        records = source.get("records")
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ReviewReconciliationError(f"review source {index} records are invalid")
        by_case = {record.get("case_id"): record for record in records}
        if (
            len(by_case) != len(records)
            or set(by_case) != selected_case_ids
            or len(records) != SELECTION_COUNT
        ):
            raise ReviewReconciliationError(
                f"review source {index} must cover exactly the selected 12 cases"
            )
        records_by_source.append(by_case)

    worklist = []
    gold_cases = []
    conclusion_counts: Counter[str] = Counter()
    reviewer_ids: set[str] = set()
    source_formats: set[str] = set()

    for selected in selected_cases:
        case_id = selected["case_id"]
        normalized_reviews = [
            _normalize_review_record(
                records_by_source[index][case_id],
                selected,
                annotation_reference=source_references[index],
            )
            for index in range(2)
        ]
        case_reviewer_ids = {review["reviewer_id"] for review in normalized_reviews}
        if len(case_reviewer_ids) != 2:
            raise ReviewReconciliationError(f"case {case_id} must have two distinct reviewer IDs")
        conclusions = {review["conclusion"] for review in normalized_reviews}
        if len(conclusions) != 1:
            raise ReviewReconciliationError(
                f"case {case_id} conclusions disagree; independent adjudication is required"
            )
        if normalized_reviews[0]["documents"] != normalized_reviews[1]["documents"]:
            raise ReviewReconciliationError(
                f"case {case_id} normalized observations disagree; "
                "independent adjudication is required"
            )
        conclusion = normalized_reviews[0]["conclusion"]
        if conclusion == "unreadable_or_insufficient":
            raise ReviewReconciliationError(
                f"case {case_id} cannot produce value-bearing Gold from unreadable evidence"
            )

        normalized_reviews.sort(key=lambda review: review["reviewer_id"])
        schema_reviews = [
            {
                "review_id": f"{case_id}:{review['reviewer_id']}",
                "reviewer_id": review["reviewer_id"],
                "independent_review_attested": True,
                "scope": REVIEW_SCOPE,
                "conclusion": conclusion,
                "reviewed_document_sha256": review["reviewed_document_sha256"],
                "annotation_reference": review["annotation_reference"],
            }
            for review in normalized_reviews
        ]
        worklist.append(
            {
                "schema_version": ADJUDICATION_SCHEMA,
                "selection_id": SELECTION_ID,
                "case_id": case_id,
                "triplet_fingerprint_sha256": selected["triplet_fingerprint_sha256"],
                "required_independent_review_count": 2,
                "status": "adjudicated",
                "gold_eligible": True,
                "reviews": schema_reviews,
                "resolution": {
                    "method": "two_reviewer_consensus",
                    "final_conclusion": conclusion,
                    "reference": f"{resolution_reference}#case={case_id}",
                },
            }
        )
        gold_cases.append(
            {
                "case_id": case_id,
                "triplet_fingerprint_sha256": selected["triplet_fingerprint_sha256"],
                "gold_eligible": True,
                "adjudication_status": "adjudicated",
                "annotation_references": [
                    review["annotation_reference"] for review in normalized_reviews
                ],
                "documents": _gold_documents(
                    case_id,
                    normalized_reviews[0]["documents"],
                ),
            }
        )
        conclusion_counts[conclusion] += 1
        reviewer_ids.update(case_reviewer_ids)
        source_formats.update(review["source_format"] for review in normalized_reviews)

    try:
        gate = evaluate_gate(selected_cases, worklist)
    except DiagnosticSelectionError as exc:
        raise ReviewReconciliationError(
            "existing diagnostic Gate rejected reconciled evidence"
        ) from exc
    if (
        gate.get("status") != GATE_PASSED_STATUS
        or gate.get("gold_eligible") is not True
        or gate.get("status_counts") != {"adjudicated": SELECTION_COUNT}
        or gate.get("blockers") != []
    ):
        raise ReviewReconciliationError("reconciled evidence did not pass the existing Gate")
    return {
        "worklist": worklist,
        "gate": gate,
        "gold_cases": gold_cases,
        "conclusion_counts": dict(sorted(conclusion_counts.items())),
        "reviewer_ids": sorted(reviewer_ids),
        "source_formats": sorted(source_formats),
    }


def _artifact_relative_reference(path: Path, artifact_root: Path) -> str:
    root = artifact_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ReviewReconciliationError(
            "review annotations must be stored under the declared _artifacts root"
        ) from exc
    return relative.as_posix()


def _validate_artifact_output_location(
    artifact_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    root = artifact_root.resolve()
    if root.name != "_artifacts" or not root.is_dir() or artifact_root.is_symlink():
        raise ReviewReconciliationError("artifact root must be a real directory named _artifacts")
    resolved_output = output_dir.resolve()
    try:
        relative = resolved_output.relative_to(root)
    except ValueError as exc:
        raise ReviewReconciliationError(
            "value-bearing Gold output must remain under _artifacts"
        ) from exc
    if not relative.parts:
        raise ReviewReconciliationError("Gold output requires a dedicated directory")
    if output_dir.exists() and output_dir.is_symlink():
        raise ReviewReconciliationError("Gold output directory must not be a symlink")
    return root, resolved_output


def _preflight_output(path: Path, content: bytes, *, refresh: bool) -> None:
    if not path.exists():
        return
    if not path.is_file() or path.is_symlink():
        raise ReviewReconciliationError(f"output is not a regular file: {path.name}")
    if path.read_bytes() != content and not refresh:
        raise ReviewReconciliationError(
            f"output differs: {path.name}; use --refresh only after reviewing the change"
        )


def _write_generated(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary = Path(file_handle.name)
            file_handle.write(content)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_reconciliation_bundle(
    *,
    selection_path: Path,
    review_paths: list[Path],
    artifact_root: Path,
    output_dir: Path,
    diagnostic_schema_path: Path = DEFAULT_DIAGNOSTIC_SCHEMA_PATH,
    gold_schema_path: Path = DEFAULT_GOLD_SCHEMA_PATH,
    expected_selection_sha256: str = EXPECTED_SELECTION_SHA256,
    gold_set_id: str = DEFAULT_GOLD_SET_ID,
    refresh: bool = False,
) -> dict[str, Any]:
    """Validate, reconcile, and atomically materialize an artifact-only bundle."""

    if len(review_paths) != 2 or review_paths[0].resolve() == review_paths[1].resolve():
        raise ReviewReconciliationError("two distinct review annotation files are required")
    if not _is_lower_hex_sha256(expected_selection_sha256):
        raise ReviewReconciliationError("expected selection SHA-256 is invalid")
    gold_set_id = _normalized_text(gold_set_id, label="Gold set ID")
    root, resolved_output = _validate_artifact_output_location(artifact_root, output_dir)

    selection, selection_bytes = _read_json_object(selection_path, label="selection")
    selection_sha256 = _sha256_bytes(selection_bytes)
    if selection_sha256 != expected_selection_sha256:
        raise ReviewReconciliationError("selection SHA-256 does not match the pinned value")
    _validate_selection(selection)

    diagnostic_schema, _ = _read_json_object(
        diagnostic_schema_path,
        label="diagnostic adjudication schema",
    )
    gold_schema, _ = _read_json_object(
        gold_schema_path,
        label="adjudicated Gold schema",
    )
    review_sources = []
    for index, review_path in enumerate(review_paths, start=1):
        records, raw = _read_annotation_jsonl(
            review_path,
            label=f"review source {index}",
        )
        review_sources.append(
            {
                "annotation_reference": _artifact_relative_reference(review_path, root),
                "sha256": _sha256_bytes(raw),
                "records": records,
            }
        )
    if review_sources[0]["sha256"] == review_sources[1]["sha256"]:
        raise ReviewReconciliationError("review annotation files must have distinct content")

    report_reference = (resolved_output.relative_to(root) / OUTPUT_FILENAMES["report"]).as_posix()
    reconciliation = reconcile_review_records(
        selection,
        review_sources,
        resolution_reference=report_reference,
    )
    worklist = reconciliation["worklist"]
    gate = reconciliation["gate"]
    for entry in worklist:
        _validate_schema(entry, diagnostic_schema, label="diagnostic adjudication")

    worklist_bytes = _jsonl_bytes(worklist)
    gate_bytes = _json_bytes(gate)
    worklist_sha256 = _sha256_bytes(worklist_bytes)
    gate_sha256 = _sha256_bytes(gate_bytes)
    gold = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "gold_set_id": gold_set_id,
        "selection_id": selection["selection_id"],
        "selection_sha256": selection_sha256,
        "adjudication_worklist_sha256": worklist_sha256,
        "adjudication_gate_sha256": gate_sha256,
        "gold_eligible": True,
        "release_scoring_scope": RELEASE_SCORING_SCOPE,
        "cases": reconciliation["gold_cases"],
    }
    _validate_schema(gold, gold_schema, label="adjudicated Gold")
    gold_bytes = _json_bytes(gold)
    gold_sha256 = _sha256_bytes(gold_bytes)

    input_reviews = []
    for source in review_sources:
        source_reviewer_ids = sorted(
            {
                str(record["reviewer_id"])
                for record in source["records"]
                if isinstance(record.get("reviewer_id"), str)
            }
        )
        input_reviews.append(
            {
                "annotation_reference": source["annotation_reference"],
                "sha256": source["sha256"],
                "case_count": len(source["records"]),
                "reviewer_ids": source_reviewer_ids,
                "reviewer_kind": REVIEWER_KIND,
                "independence_evidence": "attested_distinct_id_and_distinct_file",
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "selection_id": selection["selection_id"],
        "selection_sha256": selection_sha256,
        "case_count": len(reconciliation["gold_cases"]),
        "gold_case_count": len(reconciliation["gold_cases"]),
        "conclusion_counts": reconciliation["conclusion_counts"],
        "reviewer_ids": reconciliation["reviewer_ids"],
        "source_formats": reconciliation["source_formats"],
        "input_reviews": input_reviews,
        "outputs": {
            "worklist": {
                "relative_path": OUTPUT_FILENAMES["worklist"],
                "sha256": worklist_sha256,
            },
            "gate": {
                "relative_path": OUTPUT_FILENAMES["gate"],
                "sha256": gate_sha256,
            },
            "gold": {
                "relative_path": OUTPUT_FILENAMES["gold"],
                "sha256": gold_sha256,
            },
        },
        "validation": {
            "selection_sha256": "passed",
            "selected_document_hashes": "passed",
            "independent_ai_visual_review_attestations": "passed",
            "distinct_reviewer_ids_per_case": "passed",
            "normalized_observation_consensus": "passed",
            "diagnostic_adjudication_schema": "passed",
            "existing_evaluate_gate": GATE_PASSED_STATUS,
            "adjudicated_gold_schema": "passed",
            "artifact_only_output_boundary": "passed",
        },
        "data_policy": {
            "gold_contains_observed_values": True,
            "report_contains_observed_values": False,
            "automatic_value_correction_performed": False,
        },
        "limitations": [
            "Reviewer independence is attested and structurally checked, not independently proven.",
            "The reviewers are independent AI visual reviews; this artifact makes no human-review claim.",
            "Consensus preserves normalized visible observations, including intentional discrepancies.",
            "This Gold scores Claude Vision field extraction only, not workflow outcomes.",
        ],
    }
    report_bytes = _json_bytes(report)
    outputs = {
        "worklist": (resolved_output / OUTPUT_FILENAMES["worklist"], worklist_bytes),
        "gate": (resolved_output / OUTPUT_FILENAMES["gate"], gate_bytes),
        "gold": (resolved_output / OUTPUT_FILENAMES["gold"], gold_bytes),
        "report": (resolved_output / OUTPUT_FILENAMES["report"], report_bytes),
    }
    for path, content in outputs.values():
        _preflight_output(path, content, refresh=refresh)
    for path, content in outputs.values():
        _write_generated(path, content)

    return {
        "paths": {name: path for name, (path, _content) in outputs.items()},
        "sha256": {name: _sha256_bytes(content) for name, (_path, content) in outputs.items()},
        "report": report,
        "worklist": worklist,
        "gate": gate,
        "gold": gold,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile two independent AI visual reviews into schema-valid, "
            "artifact-only adjudicated Gold."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-schema",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_SCHEMA_PATH,
    )
    parser.add_argument("--gold-schema", type=Path, default=DEFAULT_GOLD_SCHEMA_PATH)
    parser.add_argument(
        "--expected-selection-sha256",
        default=EXPECTED_SELECTION_SHA256,
    )
    parser.add_argument("--gold-set-id", default=DEFAULT_GOLD_SET_ID)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        bundle = build_reconciliation_bundle(
            selection_path=args.selection,
            review_paths=[args.review_a, args.review_b],
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
            diagnostic_schema_path=args.diagnostic_schema,
            gold_schema_path=args.gold_schema,
            expected_selection_sha256=args.expected_selection_sha256,
            gold_set_id=args.gold_set_id,
            refresh=args.refresh,
        )
    except ReviewReconciliationError as exc:
        print(f"review reconciliation failed: {exc}")
        return 1
    report = bundle["report"]
    print(
        "review reconciliation passed: "
        f"cases={report['case_count']}, "
        f"gate={report['validation']['existing_evaluate_gate']}, "
        f"output={bundle['paths']['gold']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
