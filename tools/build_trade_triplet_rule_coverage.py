"""Build the non-release relation-aware rule-coverage diagnostic bundle.

This builder never reads evaluation-v2 split membership.  It selects from the
registered public synthetic case pool with the repository-owned descriptor,
injects exactly one field mutation per positive operator, and writes only to a
new derived artifact directory.  The result is development evidence, never a
sealed or unseen release holdout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from tools.build_trade_triplet_eval_assets import (
    BUSINESS_FIELD_OPERATOR_IDS,
    inject_controlled_defects,
)


DATASET_ID = "trade_document_triplets"
DATASET_VERSION = "source-v1"
EVALUATION_VERSION = "rule-coverage-v1"
DESCRIPTOR_SCHEMA = "trade-document-rule-coverage-descriptor/v1"
MANIFEST_SCHEMA = "trade-document-rule-coverage-assets/v1"
DEFECTS_RELATIVE_PATH = Path("injected-defects/cases.jsonl")
NEGATIVE_CONTROLS_RELATIVE_PATH = Path("negative-controls/cases.jsonl")
MANIFEST_RELATIVE_PATH = Path("manifest.json")
CHECKSUMS_RELATIVE_PATH = Path("checksums.sha256")
DEFAULT_DESCRIPTOR = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "rule-coverage-v1.descriptor.json"
)
DEFAULT_DATASET_LOCK = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "dataset.lock.json"
)


class RuleCoverageBuildError(ValueError):
    """The public diagnostic input or requested output is unsafe or inconsistent."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise RuleCoverageBuildError("diagnostic input is not readable") from exc


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        )
        + "\n"
    ).encode()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleCoverageBuildError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise RuleCoverageBuildError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleCoverageBuildError(f"{label} is not readable JSONL") from exc
    if not values or not all(isinstance(value, dict) for value in values):
        raise RuleCoverageBuildError(f"{label} must contain object records")
    return values


def _regular_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RuleCoverageBuildError("diagnostic input contains an unsafe relative path")
    path = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuleCoverageBuildError("diagnostic input path is unavailable") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file() or path.is_symlink():
        raise RuleCoverageBuildError("diagnostic input is not a regular in-dataset file")
    return resolved


def _validate_descriptor(path: Path) -> tuple[dict[str, Any], str]:
    descriptor = _read_json(path, label="rule-coverage descriptor")
    selection = descriptor.get("selection")
    policy = descriptor.get("policy")
    if (
        descriptor.get("schema_version") != DESCRIPTOR_SCHEMA
        or descriptor.get("dataset_id") != DATASET_ID
        or descriptor.get("dataset_version") != DATASET_VERSION
        or descriptor.get("evaluation_version") != EVALUATION_VERSION
        or descriptor.get("scope") != "public_registered_synthetic_development_diagnostic"
        or descriptor.get("release_eligible") is not False
        or descriptor.get("positive_operator_ids") != list(BUSINESS_FIELD_OPERATOR_IDS)
        or descriptor.get("negative_control_operator_ids") != ["pl_extra_item_ambiguous_not_scored"]
        or descriptor.get("finding_grain")
        != ["issue_code", "source_document_id", "target_document_id"]
        or not isinstance(selection, dict)
        or selection.get("case_pool") != "all_registered_public_synthetic_cases"
        or selection.get("case_count") != len(BUSINESS_FIELD_OPERATOR_IDS) + 1
        or selection.get("positive_case_count") != len(BUSINESS_FIELD_OPERATOR_IDS)
        or selection.get("negative_control_case_count") != 1
        or selection.get("membership_embedded") is not False
        or selection.get("release_split_membership_consulted") is not False
        or not isinstance(selection.get("seed"), str)
        or not selection["seed"]
        or not isinstance(policy, dict)
        or policy.get("allowed_use") != "development_diagnostic_only"
        or policy.get("release_claim_allowed") is not False
        or policy.get("model_or_rule_selection_claim_allowed") is not False
        or policy.get("sealed_unseen_release_bundle_required_for_release") is not True
    ):
        raise RuleCoverageBuildError("rule-coverage descriptor contract is invalid")
    return descriptor, _sha256_file(path)


def _validate_raw_checksums(dataset_root: Path) -> str:
    checksums_path = _regular_file(dataset_root, Path("checksums.sha256"))
    rows = checksums_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 3:
        raise RuleCoverageBuildError("registered raw checksum inventory must contain three files")
    seen: set[Path] = set()
    for row in rows:
        digest, separator, raw_relative = row.partition("  ")
        relative = Path(raw_relative)
        if (
            not separator
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or relative in seen
            or relative.parts[:1] != ("raw",)
        ):
            raise RuleCoverageBuildError("registered raw checksum inventory is invalid")
        seen.add(relative)
        if _sha256_file(_regular_file(dataset_root, relative)) != digest:
            raise RuleCoverageBuildError("registered raw file hash does not match inventory")
    return _sha256_file(checksums_path)


def _validate_dataset(
    dataset_root: Path,
    dataset_lock_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    root = dataset_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuleCoverageBuildError("dataset root must be a regular directory")
    inventory_path = _regular_file(root, Path("inventory.json"))
    cases_path = _regular_file(root, Path("derived/cases.jsonl"))
    inventory = _read_json(inventory_path, label="registered inventory")
    dataset = inventory.get("dataset")
    summary = inventory.get("summary")
    derived = inventory.get("derived_artifacts")
    if (
        inventory.get("schema_version") != "trade-document-triplets-inventory/v1"
        or not isinstance(dataset, dict)
        or dataset.get("id") != DATASET_ID
        or dataset.get("version") != DATASET_VERSION
        or not isinstance(summary, dict)
        or summary.get("case_count") != 100
        or summary.get("document_count") != 300
        or summary.get("raw_bytes_preserved") is not True
        or summary.get("raw_dataset_contains_business_values") is not True
        or not isinstance(derived, dict)
        or not isinstance(derived.get("cases"), dict)
        or derived["cases"].get("relative_path") != "derived/cases.jsonl"
        or derived["cases"].get("sha256") != _sha256_file(cases_path)
        or derived["cases"].get("byte_size") != cases_path.stat().st_size
    ):
        raise RuleCoverageBuildError("registered dataset inventory contract is invalid")

    lock = _read_json(dataset_lock_path, label="dataset lock")
    lock_hash = _sha256_file(dataset_lock_path)
    inventory_lock = dataset.get("lock")
    registration = lock.get("expected_registration")
    if (
        not isinstance(inventory_lock, dict)
        or inventory_lock.get("schema_version") != "trade-document-triplets-dataset-lock/v1"
        or inventory_lock.get("sha256") != lock_hash
        or lock.get("schema_version") != "trade-document-triplets-dataset-lock/v1"
        or lock.get("dataset_id") != DATASET_ID
        or lock.get("dataset_version") != DATASET_VERSION
        or not isinstance(registration, dict)
        or registration.get("case_count") != 100
        or registration.get("document_count") != 300
    ):
        raise RuleCoverageBuildError("dataset lock does not match registered inventory")

    raw_checksums_hash = _validate_raw_checksums(root)
    cases = _read_jsonl(cases_path, label="registered cases")
    case_ids = [case.get("case_id") for case in cases]
    if (
        len(cases) != 100
        or len(set(case_ids)) != 100
        or not all(
            isinstance(case_id, str) and re.fullmatch(r"trade-triplet-\d{4}", case_id)
            for case_id in case_ids
        )
        or any(
            case.get("dataset_id") != DATASET_ID or case.get("dataset_version") != DATASET_VERSION
            for case in cases
        )
    ):
        raise RuleCoverageBuildError("registered case pool contract is invalid")
    return cases, {
        "dataset_lock_sha256": lock_hash,
        "inventory_sha256": _sha256_file(inventory_path),
        "registered_cases_sha256": _sha256_file(cases_path),
        "raw_checksums_sha256": raw_checksums_hash,
    }


def _write_bundle(
    output_root: Path,
    *,
    defects_content: bytes,
    negative_controls_content: bytes,
    manifest: dict[str, Any],
) -> None:
    output = output_root.resolve(strict=False)
    if output.exists():
        raise RuleCoverageBuildError("output bundle already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise RuleCoverageBuildError("output parent must not be a symbolic link")
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-",
        dir=output.parent,
    ) as temporary_parent:
        candidate = Path(temporary_parent) / "bundle"
        (candidate / DEFECTS_RELATIVE_PATH).parent.mkdir(parents=True)
        (candidate / DEFECTS_RELATIVE_PATH).write_bytes(defects_content)
        (candidate / NEGATIVE_CONTROLS_RELATIVE_PATH).parent.mkdir(parents=True)
        (candidate / NEGATIVE_CONTROLS_RELATIVE_PATH).write_bytes(negative_controls_content)
        manifest_content = _json_bytes(manifest)
        (candidate / MANIFEST_RELATIVE_PATH).write_bytes(manifest_content)
        checksums = (
            f"{_sha256_bytes(defects_content)}  {DEFECTS_RELATIVE_PATH.as_posix()}\n"
            f"{_sha256_bytes(negative_controls_content)}  "
            f"{NEGATIVE_CONTROLS_RELATIVE_PATH.as_posix()}\n"
            f"{_sha256_bytes(manifest_content)}  {MANIFEST_RELATIVE_PATH.as_posix()}\n"
        ).encode()
        (candidate / CHECKSUMS_RELATIVE_PATH).write_bytes(checksums)
        _validate_output_bundle(candidate)
        os.replace(candidate, output)
    _validate_output_bundle(output)


def _validate_output_bundle(output: Path) -> None:
    expected = {
        DEFECTS_RELATIVE_PATH,
        NEGATIVE_CONTROLS_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
    }
    checksums_path = _regular_file(output, CHECKSUMS_RELATIVE_PATH)
    inventory: dict[Path, str] = {}
    for row in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, raw_relative = row.partition("  ")
        relative = Path(raw_relative)
        if (
            not separator
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or relative in inventory
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise RuleCoverageBuildError("output checksum inventory is invalid")
        inventory[relative] = digest
    actual = {
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if set(inventory) != expected or actual != expected | {CHECKSUMS_RELATIVE_PATH}:
        raise RuleCoverageBuildError("output bundle inventory is incomplete or contains extras")
    for relative, digest in inventory.items():
        if _sha256_file(_regular_file(output, relative)) != digest:
            raise RuleCoverageBuildError("output bundle checksum verification failed")


def _source_case_sha256(case: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(case))


def _build_packing_extra_negative_control(case: dict[str, Any]) -> dict[str, Any]:
    source_hash = _source_case_sha256(case)
    mutated_case = copy.deepcopy(case)
    try:
        goods = mutated_case["documents"]["packing_list"]["raw_record"]["Goods Details"]
    except (KeyError, TypeError) as exc:
        raise RuleCoverageBuildError("negative-control source has no packing-list goods") from exc
    if not isinstance(goods, list) or not goods:
        raise RuleCoverageBuildError("negative-control source has no packing-list goods")
    extra_item = {
        "item": "CONTROLLED PACKING-ONLY ACCESSORY NOT PRESENT ON INVOICE",
        "quantity": "1 PCS",
        "weight": "0.01 KGS",
    }
    added_index = len(goods)
    goods.append(extra_item)
    if _source_case_sha256(case) != source_hash:
        raise RuleCoverageBuildError("negative control changed its source case")
    operator_id = "pl_extra_item_ambiguous_not_scored"
    return {
        "schema_version": "trade-document-negative-control/v1",
        "variant_id": f"{case['case_id']}--{operator_id}",
        "source_case_id": case["case_id"],
        "source_case_sha256": source_hash,
        "mutation_count": 1,
        "mutation": {
            "operator_id": operator_id,
            "operation": "add",
            "document_type": "packing_list",
            "field_path": ["Goods Details", added_index],
            "after": extra_item,
        },
        "expected_finding_delta": [],
        "expected_pairing": {
            "unmatched_packing_item_status": "ambiguous_not_scored",
            "line_item_pairing_status": "deterministic_with_unscored_packing_extras",
            "requires_human_review_from_extra_item": False,
        },
        "mutated_case": mutated_case,
    }


def build_rule_coverage_bundle(
    dataset_root: Path,
    output_root: Path,
    *,
    descriptor_path: Path = DEFAULT_DESCRIPTOR,
    dataset_lock_path: Path = DEFAULT_DATASET_LOCK,
) -> dict[str, Any]:
    source = dataset_root.resolve(strict=True)
    output = output_root.resolve(strict=False)
    raw_root = (source / "raw").resolve(strict=True)
    if output == raw_root or output.is_relative_to(raw_root):
        raise RuleCoverageBuildError("diagnostic output must not be written inside raw data")
    descriptor, descriptor_hash = _validate_descriptor(descriptor_path.resolve(strict=True))
    cases, source_hashes = _validate_dataset(source, dataset_lock_path.resolve(strict=True))
    seed = descriptor["selection"]["seed"]
    selected_case_ids = sorted(
        (str(case["case_id"]) for case in cases),
        key=lambda case_id: (
            hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest(),
            case_id,
        ),
    )[: len(BUSINESS_FIELD_OPERATOR_IDS) + 1]
    defects = inject_controlled_defects(
        cases,
        selected_case_ids[: len(BUSINESS_FIELD_OPERATOR_IDS)],
        evaluation_version=EVALUATION_VERSION,
    )
    case_by_id = {str(case["case_id"]): case for case in cases}
    negative_controls = [_build_packing_extra_negative_control(case_by_id[selected_case_ids[-1]])]
    defects_content = _jsonl_bytes(defects)
    negative_controls_content = _jsonl_bytes(negative_controls)
    defects_hash = _sha256_bytes(defects_content)
    negative_controls_hash = _sha256_bytes(negative_controls_content)
    assignments = [
        {
            "source_case_id": item["source_case_id"],
            "operator_id": item["mutation"]["operator_id"],
            "variant_id": item["variant_id"],
        }
        for item in defects
    ]
    coverage = [
        {
            "operator_id": item["mutation"]["operator_id"],
            "expected_rule_ids": item["expected_rule_ids"],
            "expected_outcome": item["expected_outcome"],
        }
        for item in defects
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "scope": descriptor["scope"],
        "source": {
            **source_hashes,
            "selection_descriptor_sha256": descriptor_hash,
            "release_split_membership_consulted": False,
        },
        "counts": {
            "controlled_defect_count": len(defects),
            "negative_control_count": len(negative_controls),
            "total_variant_count": len(defects) + len(negative_controls),
        },
        "generated_files": [
            {
                "relative_path": DEFECTS_RELATIVE_PATH.as_posix(),
                "sha256": defects_hash,
                "byte_size": len(defects_content),
            },
            {
                "relative_path": NEGATIVE_CONTROLS_RELATIVE_PATH.as_posix(),
                "sha256": negative_controls_hash,
                "byte_size": len(negative_controls_content),
            },
        ],
        "controlled_defects": {
            "relative_path": DEFECTS_RELATIVE_PATH.as_posix(),
            "sha256": defects_hash,
            "byte_size": len(defects_content),
            "generation_scope": "public_registered_synthetic_development_diagnostic",
            "release_split_membership_consulted": False,
            "selection_seed": seed,
            "case_count": len(defects),
            "case_ids": [item["source_case_id"] for item in defects],
            "operator_ids": sorted(item["mutation"]["operator_id"] for item in defects),
            "case_operator_assignments_sha256": _sha256_bytes(_canonical_bytes(assignments)),
        },
        "controlled_operator_coverage": {
            "business_fields": {
                "required_operator_ids": list(BUSINESS_FIELD_OPERATOR_IDS),
                "observed_operator_ids": [item["mutation"]["operator_id"] for item in defects],
                "operator_count": len(BUSINESS_FIELD_OPERATOR_IDS),
                "expected_outcome": "needs_review",
                "finding_grain": descriptor["finding_grain"],
                "coverage_sha256": _sha256_bytes(_canonical_bytes(coverage)),
            },
            "negative_controls": {
                "relative_path": NEGATIVE_CONTROLS_RELATIVE_PATH.as_posix(),
                "sha256": negative_controls_hash,
                "byte_size": len(negative_controls_content),
                "required_operator_ids": descriptor["negative_control_operator_ids"],
                "observed_operator_ids": [
                    item["mutation"]["operator_id"] for item in negative_controls
                ],
                "operator_count": len(descriptor["negative_control_operator_ids"]),
                "expected_finding_delta_count": 0,
            },
        },
        "release_holdout": {
            "release_eligible": False,
            "status": "not_release_evidence_public_development_diagnostic",
            "sealed_unseen_release_bundle_required": True,
        },
    }
    _write_bundle(
        output,
        defects_content=defects_content,
        negative_controls_content=negative_controls_content,
        manifest=manifest,
    )
    return {
        "status": "PASS",
        "release_eligible": False,
        "case_count": len(defects),
        "negative_control_count": len(negative_controls),
        "total_variant_count": len(defects) + len(negative_controls),
        "operator_count": len(BUSINESS_FIELD_OPERATOR_IDS),
        "defects_sha256": defects_hash,
        "negative_controls_sha256": negative_controls_hash,
        "manifest_sha256": _sha256_file(output / MANIFEST_RELATIVE_PATH),
        "checksums_sha256": _sha256_file(output / CHECKSUMS_RELATIVE_PATH),
        "selection_descriptor_sha256": descriptor_hash,
        "selected_case_ids": "OMITTED",
        "absolute_paths": "OMITTED",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--dataset-lock", type=Path, default=DEFAULT_DATASET_LOCK)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = build_rule_coverage_bundle(
            args.dataset_root,
            args.output_root,
            descriptor_path=args.descriptor,
            dataset_lock_path=args.dataset_lock,
        )
    except RuleCoverageBuildError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    except OSError:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": "filesystem_operation_failed",
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
