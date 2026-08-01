"""Build a deterministic 12-triplet diagnostic selection and two-reviewer Gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tools.dataset_provenance import DatasetProvenanceError, verify_dataset_provenance


SELECTION_ID = "trade-images-hash-ranked-diagnostic-v1"
SELECTION_SCHEMA = "trade-image-diagnostic-selection/v1"
ADJUDICATION_SCHEMA = "trade-image-diagnostic-adjudication/v1"
GATE_SCHEMA = "trade-image-diagnostic-gate/v1"
SELECTION_COUNT = 12
ALLOWED_CONCLUSIONS = {
    "source_matches_manifest_observation",
    "source_discrepancy_confirmed",
    "invalid_prior_claim",
    "unreadable_or_insufficient",
}
ALLOWED_STATUSES = {
    "pending_first_review",
    "pending_second_review",
    "consensus_requires_resolution",
    "disagreement_requires_adjudicator",
    "adjudicated",
    "excluded",
}
DOCUMENT_COLUMNS = {
    "commercial_invoice": ("ci_file", "commercial_invoice"),
    "packing_list": ("pl_file", "packing_list"),
    "bill_of_lading": ("bl_file", "bill_of_lading"),
}
CORRECTION_CASES = {
    "0001": "packing_list_bl_suffix_claim_invalidated",
    "0041": "packing_list_bl_separated_suffix_claim_invalidated",
    "0500": "packing_list_bl_ee_suffix_claim_invalidated",
}
DEFAULT_FIXTURE_LOCK = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "trade_documents_images"
    / "v1"
    / "dataset.lock.json"
)


class DiagnosticSelectionError(ValueError):
    """Raised when diagnostic selection or adjudication evidence is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticSelectionError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticSelectionError(f"JSON root must be an object: {path}")
    return value


def _load_worklist(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise DiagnosticSelectionError(f"worklist is not a regular file: {path}")
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticSelectionError(f"cannot read adjudication worklist: {path}") from exc
    if not values or any(not isinstance(value, dict) for value in values):
        raise DiagnosticSelectionError("adjudication worklist must contain JSON objects")
    return values


def _load_manifest(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DiagnosticSelectionError("cannot read image source manifest") from exc
    if len(rows) != 1000:
        raise DiagnosticSelectionError("image source manifest must contain 1000 triplets")
    case_ids = [row.get("index", "") for row in rows]
    if any(not case_id.isdigit() or len(case_id) != 4 for case_id in case_ids):
        raise DiagnosticSelectionError("image source manifest case IDs are invalid")
    if len(set(case_ids)) != len(case_ids):
        raise DiagnosticSelectionError("image source manifest case IDs are duplicated")
    return rows


def _load_checksums(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative_path = line.partition("  ")
        if not separator or len(digest) != 64 or relative_path in result:
            raise DiagnosticSelectionError("image checksum file is invalid")
        result[relative_path] = digest
    return result


def _write_generated(path: Path, content: bytes, *, refresh: bool) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise DiagnosticSelectionError(f"output is not a regular file: {path}")
        if path.read_bytes() == content:
            return
        if not refresh:
            raise DiagnosticSelectionError(
                f"diagnostic output differs: {path}; use --refresh after review"
            )
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


def select_hash_ranked_cases(
    manifest_rows: list[dict[str, str]],
    image_inventory: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for row in manifest_rows:
        case_id = row["index"]
        documents = {}
        for document_type, (column, directory) in DOCUMENT_COLUMNS.items():
            relative_path = f"{directory}/{row[column]}"
            inventory = image_inventory.get(relative_path)
            if inventory is None:
                raise DiagnosticSelectionError(
                    f"manifest image is absent from inventory: {relative_path}"
                )
            documents[document_type] = {
                "relative_path": relative_path,
                "sha256": inventory["sha256"],
                "byte_size": inventory["byte_size"],
                "width_px": inventory["width"],
                "height_px": inventory["height"],
            }
        triplet_fingerprint = _sha256_bytes(_canonical_bytes(documents))
        rank_sha256 = _sha256_bytes(f"{SELECTION_ID}:{triplet_fingerprint}".encode())
        candidates.append(
            {
                "case_id": case_id,
                "rank_sha256": rank_sha256,
                "triplet_fingerprint_sha256": triplet_fingerprint,
                "documents": documents,
            }
        )
    return sorted(candidates, key=lambda item: (item["rank_sha256"], item["case_id"]))[
        :SELECTION_COUNT
    ]


def _first_correction_review(
    case_id: str,
    packing_list_sha256: str,
) -> dict[str, Any]:
    return {
        "review_id": f"correction-{case_id}-visual-review-1",
        "reviewer_id": "local-visual-reviewer-1",
        "independent_review_attested": True,
        "scope": "correction_reaudit",
        "conclusion": "invalid_prior_claim",
        "reviewed_document_sha256": [packing_list_sha256],
        "annotation_reference": (
            "fixtures/trade_documents_images/v1/dataset.lock.json"
            f"#invalidated_annotation_claims/{case_id}"
        ),
    }


def build_worklist(
    selected_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    worklist = []
    for selected in selected_cases:
        case_id = selected["case_id"]
        worklist.append(
            {
                "schema_version": ADJUDICATION_SCHEMA,
                "selection_id": SELECTION_ID,
                "case_id": case_id,
                "triplet_fingerprint_sha256": selected["triplet_fingerprint_sha256"],
                "required_independent_review_count": 2,
                "status": "pending_first_review",
                "gold_eligible": False,
                "reviews": [],
            }
        )
    return worklist


def load_or_initialize_worklist(
    path: Path,
    initial_worklist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve human review evidence; initialize only when no worklist exists."""

    if path.exists():
        return _load_worklist(path)
    _write_generated(path, _jsonl_bytes(initial_worklist), refresh=False)
    return initial_worklist


def _validate_entry_shape(entry: dict[str, Any]) -> None:
    case_id = entry.get("case_id", "<unknown>")
    if entry.get("schema_version") != ADJUDICATION_SCHEMA:
        raise DiagnosticSelectionError(f"case {case_id} adjudication schema is not pinned")
    if entry.get("selection_id") != SELECTION_ID:
        raise DiagnosticSelectionError(f"case {case_id} selection ID is not pinned")
    if entry.get("required_independent_review_count") != 2:
        raise DiagnosticSelectionError(f"case {case_id} must require two reviewers")
    if entry.get("status") not in ALLOWED_STATUSES:
        raise DiagnosticSelectionError(f"case {case_id} has an invalid status")
    if not isinstance(entry.get("gold_eligible"), bool):
        raise DiagnosticSelectionError(f"case {case_id} has invalid Gold eligibility")
    reviews = entry.get("reviews")
    if not isinstance(reviews, list) or len(reviews) > 2:
        raise DiagnosticSelectionError(f"case {case_id} must contain at most two reviews")
    required_review_keys = {
        "review_id",
        "reviewer_id",
        "independent_review_attested",
        "scope",
        "conclusion",
        "reviewed_document_sha256",
        "annotation_reference",
    }
    for review in reviews:
        if not isinstance(review, dict) or not required_review_keys.issubset(review):
            raise DiagnosticSelectionError(f"case {case_id} contains an invalid review")
        if (
            not isinstance(review["review_id"], str)
            or not review["review_id"]
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"]
            or review["independent_review_attested"] is not True
            or review["scope"] not in {"diagnostic_full_triplet", "correction_reaudit"}
            or not isinstance(review["annotation_reference"], str)
            or not review["annotation_reference"]
        ):
            raise DiagnosticSelectionError(f"case {case_id} contains an invalid review")
        hashes = review["reviewed_document_sha256"]
        if (
            not isinstance(hashes, list)
            or not hashes
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or set(digest) - set("0123456789abcdef")
                for digest in hashes
            )
        ):
            raise DiagnosticSelectionError(f"case {case_id} review has invalid image hashes")
        if review["conclusion"] not in ALLOWED_CONCLUSIONS:
            raise DiagnosticSelectionError(f"case {case_id} review has an invalid conclusion")
    resolution = entry.get("resolution")
    if resolution is None:
        return
    if not isinstance(resolution, dict):
        raise DiagnosticSelectionError(f"case {case_id} has an invalid resolution")
    if (
        resolution.get("method") not in {"two_reviewer_consensus", "independent_adjudicator"}
        or resolution.get("final_conclusion") not in ALLOWED_CONCLUSIONS
        or not isinstance(resolution.get("reference"), str)
        or not resolution["reference"]
    ):
        raise DiagnosticSelectionError(f"case {case_id} has an invalid resolution")


def derive_adjudication_status(entry: dict[str, Any]) -> str:
    _validate_entry_shape(entry)
    reviews = entry["reviews"]
    reviewer_ids = [review["reviewer_id"] for review in reviews]
    review_ids = [review["review_id"] for review in reviews]
    if len(set(review_ids)) != len(review_ids):
        raise DiagnosticSelectionError(f"case {entry['case_id']} contains duplicate review IDs")
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise DiagnosticSelectionError(f"case {entry['case_id']} contains duplicate reviewer IDs")
    if any(not review["independent_review_attested"] for review in reviews):
        raise DiagnosticSelectionError(
            f"case {entry['case_id']} has a review without independence attestation"
        )
    if not reviews:
        return "pending_first_review"
    if len(reviews) == 1:
        return "pending_second_review"
    conclusions = {review["conclusion"] for review in reviews}
    resolution = entry.get("resolution")
    if len(conclusions) == 1:
        if not isinstance(resolution, dict):
            return "consensus_requires_resolution"
        if (
            resolution.get("method") != "two_reviewer_consensus"
            or resolution.get("final_conclusion") not in conclusions
        ):
            raise DiagnosticSelectionError(
                f"case {entry['case_id']} has an invalid consensus resolution"
            )
        return (
            "excluded"
            if resolution["final_conclusion"] == "unreadable_or_insufficient"
            else "adjudicated"
        )
    if not isinstance(resolution, dict):
        return "disagreement_requires_adjudicator"
    if resolution.get("method") != "independent_adjudicator":
        raise DiagnosticSelectionError(
            f"case {entry['case_id']} requires an independent adjudicator"
        )
    adjudicator_id = resolution.get("adjudicator_id")
    if (
        not isinstance(adjudicator_id, str)
        or not adjudicator_id
        or adjudicator_id in reviewer_ids
        or resolution.get("independent_adjudicator_attested") is not True
    ):
        raise DiagnosticSelectionError(
            f"case {entry['case_id']} requires a distinct independent adjudicator"
        )
    return (
        "excluded"
        if resolution["final_conclusion"] == "unreadable_or_insufficient"
        else "adjudicated"
    )


def evaluate_gate(
    selected_cases: list[dict[str, Any]],
    worklist: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_by_id = {case["case_id"]: case for case in selected_cases}
    if any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("case_id"), str)
        or not entry["case_id"]
        for entry in worklist
    ):
        raise DiagnosticSelectionError("worklist contains an invalid case entry")
    entry_by_id = {entry["case_id"]: entry for entry in worklist}
    if (
        len(selected_by_id) != SELECTION_COUNT
        or len(worklist) != SELECTION_COUNT
        or len(entry_by_id) != len(worklist)
        or set(entry_by_id) != set(selected_by_id)
    ):
        raise DiagnosticSelectionError("worklist must contain exactly the 12 selected cases")
    derived_counts: Counter[str] = Counter()
    blockers = []
    for case_id in sorted(selected_by_id):
        selected = selected_by_id[case_id]
        entry = entry_by_id[case_id]
        if entry["triplet_fingerprint_sha256"] != selected["triplet_fingerprint_sha256"]:
            raise DiagnosticSelectionError(
                f"case {case_id} adjudication fingerprint does not match selection"
            )
        _validate_entry_shape(entry)
        allowed_hashes = {document["sha256"] for document in selected["documents"].values()}
        for review in entry["reviews"]:
            reviewed_hashes = set(review["reviewed_document_sha256"])
            if review["scope"] != "diagnostic_full_triplet" or reviewed_hashes != allowed_hashes:
                raise DiagnosticSelectionError(
                    f"case {case_id} review must cover the selected triplet"
                )
        derived_status = derive_adjudication_status(entry)
        if entry["status"] != derived_status:
            raise DiagnosticSelectionError(
                f"case {case_id} stored status does not match review evidence"
            )
        should_be_gold = derived_status == "adjudicated"
        if entry["gold_eligible"] is not should_be_gold:
            raise DiagnosticSelectionError(
                f"case {case_id} gold eligibility does not match adjudication"
            )
        derived_counts[derived_status] += 1
        if not should_be_gold:
            blockers.append({"case_id": case_id, "status": derived_status})
    return {
        "schema_version": GATE_SCHEMA,
        "selection_id": SELECTION_ID,
        "status": (
            "passed_two_reviewer_adjudication"
            if not blockers
            else "blocked_pending_independent_reviews"
        ),
        "gold_eligible": not blockers,
        "selected_case_count": len(selected_cases),
        "required_independent_review_count_per_case": 2,
        "status_counts": dict(sorted(derived_counts.items())),
        "blockers": blockers,
    }


def build_diagnostic_selection(
    dataset_root: Path,
    inventory_path: Path,
    checksums_path: Path,
    fixture_lock_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = dataset_root.resolve()
    inventory = _load_json(inventory_path)
    fixture_lock = _load_json(fixture_lock_path)
    manifest_path = root / "manifest.csv"
    if _sha256_file(manifest_path) != fixture_lock["source_integrity"]["manifest_sha256"]:
        raise DiagnosticSelectionError("source manifest does not match fixture lock")
    if _sha256_file(inventory_path) != fixture_lock["source_integrity"]["inventory_file_sha256"]:
        raise DiagnosticSelectionError("source inventory does not match fixture lock")
    if (
        _sha256_file(checksums_path)
        != fixture_lock["source_integrity"]["image_checksums_file_sha256"]
    ):
        raise DiagnosticSelectionError("source checksums do not match fixture lock")
    rows = _load_manifest(manifest_path)
    checksum_by_path = _load_checksums(checksums_path)
    image_inventory = {image["relative_path"]: image for image in inventory["images"]}
    selected = select_hash_ranked_cases(rows, image_inventory)
    for case in selected:
        for document in case["documents"].values():
            relative_path = document["relative_path"]
            if checksum_by_path.get(relative_path) != document["sha256"]:
                raise DiagnosticSelectionError(
                    f"selected image checksum evidence differs: {relative_path}"
                )
            if _sha256_file(root / relative_path) != document["sha256"]:
                raise DiagnosticSelectionError(
                    f"selected source image bytes differ: {relative_path}"
                )

    correction_provenance = []
    for case_id, claim_id in sorted(CORRECTION_CASES.items()):
        row = next(row for row in rows if row["index"] == case_id)
        relative_path = f"packing_list/{row['pl_file']}"
        image = image_inventory[relative_path]
        correction_provenance.append(
            {
                "case_id": case_id,
                "prior_claim_id": claim_id,
                "correction_status": "prior_claim_invalidated_by_direct_visual_reaudit",
                "review_status": "pending_second_review",
                "gold_eligible": False,
                "required_independent_review_count": 2,
                "completed_review_count": 1,
                "review": _first_correction_review(case_id, image["sha256"]),
            }
        )

    worklist = build_worklist(selected)
    initial_gate = evaluate_gate(selected, worklist)
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "selection_id": SELECTION_ID,
        "dataset_id": "trade_documents_images",
        "dataset_version": "source-v1",
        "descriptor_contains_business_values": False,
        "source_images_may_contain_business_values": True,
        "selection_role": "diagnostic_candidate_not_gold",
        "gold_eligible": False,
        "initial_gate_status": initial_gate["status"],
        "selection_algorithm": {
            "name": "sha256_ranked_triplet_fingerprint",
            "version": "v1",
            "rank_formula": ("sha256(selection_id + ':' + sha256(canonical document metadata))"),
            "selection_count": SELECTION_COUNT,
            "unit": "triplet_case",
        },
        "source_integrity": {
            "manifest_sha256": _sha256_file(manifest_path),
            "inventory_sha256": _sha256_file(inventory_path),
            "checksums_sha256": _sha256_file(checksums_path),
        },
        "cases": selected,
        "correction_provenance": correction_provenance,
        "adjudication": {
            "schema": (
                "evals/trade_documents_images/v1/schemas/diagnostic-adjudication.schema.json"
            ),
            "required_independent_review_count_per_case": 2,
            "workflow": ("evals/trade_documents_images/v1/diagnostic-adjudication-workflow.md"),
            "initial_gate_status": initial_gate["status"],
            "mutable_worklist_is_separate_from_selection": True,
        },
    }
    return selection, worklist, initial_gate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hash-ranked image diagnostics and enforce two-reviewer Gold Gate."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--worklist-output", required=True, type=Path)
    parser.add_argument("--gate-output", required=True, type=Path)
    parser.add_argument("--fixture-lock", type=Path, default=DEFAULT_FIXTURE_LOCK)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--require-gold-ready",
        action="store_true",
        help="Return failure unless all 12 cases have completed two-reviewer adjudication.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_dataset_provenance(
            args.fixture_lock,
            expected_raw_root=args.dataset_root,
        )
        selection, initial_worklist, _initial_gate = build_diagnostic_selection(
            args.dataset_root,
            args.inventory,
            args.checksums,
            args.fixture_lock,
        )
        worklist = load_or_initialize_worklist(
            args.worklist_output,
            initial_worklist,
        )
        gate = evaluate_gate(selection["cases"], worklist)
        _write_generated(
            args.selection_output,
            _json_bytes(selection),
            refresh=args.refresh,
        )
        _write_generated(
            args.gate_output,
            _json_bytes(gate),
            refresh=True,
        )
    except DatasetProvenanceError as exc:
        print(f"diagnostic selection failed: {exc.code}")
        return 1
    except DiagnosticSelectionError as exc:
        print(f"diagnostic selection failed: {exc}")
        return 1
    print(
        "diagnostic selection built: "
        f"{selection['selection_algorithm']['selection_count']} triplets, "
        f"gate={gate['status']}"
    )
    if args.require_gold_ready and not gate["gold_eligible"]:
        print("diagnostic Gold Gate failed: two independent reviews are incomplete")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
