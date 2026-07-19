"""Create a repaired copy of the invalidated evaluation-v2 development bundle.

The historical bundle is preserved byte-for-byte.  This migration accepts only
the exact known input hashes, changes the one generator-error record, refreshes
the manifest/checksums, and writes a new sibling bundle.  It never reads or
re-derives split membership and cannot make evaluation-v2 release eligible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tools.build_trade_triplet_eval_assets import (
    EvalAssetError,
    _different_valid_container_number,
    _valid_container_number,
)


KNOWN_SOURCE_DEFECTS_SHA256 = "e4f2902b301ac20752c71ab0c96f4b7cc4e265eff44daff8275d61abedb0c762"
KNOWN_SOURCE_MANIFEST_SHA256 = "cd91e5513e57592b264f36dcb9e15cecd760c3337bf13b43238e601e17ed7470"
KNOWN_SOURCE_CHECKSUMS_SHA256 = "6cb6baec5a99e485a46e8e0b89a604187d56c6145890b312c53972441386083d"
DEFECTS_RELATIVE_PATH = Path("injected-defects/cases.jsonl")
MANIFEST_RELATIVE_PATH = Path("manifest.json")
CHECKSUMS_RELATIVE_PATH = Path("checksums.sha256")
PROVENANCE_RELATIVE_PATH = Path("repair-provenance.json")
OPERATOR_ID = "pl_container_mismatch"
RELEASE_HOLDOUT_STATUS = "release_holdout_v2_publicly_derivable"


class RepairError(ValueError):
    """The exact historical bundle cannot be safely migrated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise RepairError("repair input is not readable") from exc


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


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


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise RepairError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError("controlled defects are not readable JSONL") from exc
    if not values or not all(isinstance(value, dict) for value in values):
        raise RepairError("controlled defects must contain object records")
    return values


def _regular_file(root: Path, relative: Path) -> Path:
    path = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RepairError("repair input path is unavailable") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file() or path.is_symlink():
        raise RepairError("repair input is not a regular in-bundle file")
    return resolved


def _validate_source_tree(source: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise RepairError("source bundle contains a symbolic link")
        if not path.is_dir() and not path.is_file():
            raise RepairError("source bundle contains an unsupported file type")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _checksum_inventory(path: Path) -> dict[Path, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RepairError("checksum inventory is not readable") from exc
    inventory: dict[Path, str] = {}
    for line in lines:
        digest, separator, raw_relative = line.partition("  ")
        relative = Path(raw_relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not raw_relative
            or relative.is_absolute()
            or ".." in relative.parts
            or relative in inventory
        ):
            raise RepairError("checksum inventory is invalid")
        inventory[relative] = digest
    if not inventory:
        raise RepairError("checksum inventory is empty")
    return inventory


def _validate_bundle_inventory(
    root: Path,
    *,
    manifest: dict[str, Any],
    checksums_path: Path,
    repaired: bool,
) -> None:
    inventory = _checksum_inventory(checksums_path)
    generated = manifest.get("generated_files")
    if not isinstance(generated, list):
        raise RepairError("manifest generated-file inventory is invalid")
    manifest_inventory: dict[Path, tuple[str, int]] = {}
    for item in generated:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("relative_path"), str)
            or not isinstance(item.get("sha256"), str)
            or not isinstance(item.get("byte_size"), int)
        ):
            raise RepairError("manifest generated-file entry is invalid")
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative in manifest_inventory:
            raise RepairError("manifest generated-file path is invalid")
        manifest_inventory[relative] = (item["sha256"], item["byte_size"])
    expected = set(manifest_inventory)
    expected.add(MANIFEST_RELATIVE_PATH)
    if repaired:
        expected.add(PROVENANCE_RELATIVE_PATH)
    if set(inventory) != expected:
        raise RepairError("checksum and manifest inventories do not agree")
    actual_files = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != expected | {CHECKSUMS_RELATIVE_PATH}:
        raise RepairError("bundle contains an undeclared or missing file")
    for relative, expected_hash in inventory.items():
        path = _regular_file(root, relative)
        if _sha256_file(path) != expected_hash:
            raise RepairError("bundle file does not match the checksum inventory")
        if relative in manifest_inventory:
            manifest_hash, manifest_size = manifest_inventory[relative]
            if expected_hash != manifest_hash or path.stat().st_size != manifest_size:
                raise RepairError("manifest file metadata does not match bundle bytes")


def _repair_record(records: list[dict[str, Any]]) -> tuple[str, str]:
    matching = [
        record
        for record in records
        if isinstance(record.get("mutation"), dict)
        and record["mutation"].get("operator_id") == OPERATOR_ID
    ]
    if len(matching) != 1:
        raise RepairError("source bundle must contain exactly one container mismatch operator")
    record = matching[0]
    mutation = record["mutation"]
    before = mutation.get("before")
    after = mutation.get("after")
    if not isinstance(before, str) or after != f"{before}-CONTROLLED-MISMATCH":
        raise RepairError("source container mutation is not the known generator error")
    expected_rules = record.get("expected_rule_ids")
    adjudication = record.get("adjudication")
    evidence = adjudication.get("evidence") if isinstance(adjudication, dict) else None
    expected_evidence = {
        "operator_id": OPERATOR_ID,
        "document_type": mutation.get("document_type"),
        "field_path": mutation.get("field_path"),
        "before": before,
        "after": after,
    }
    if (
        expected_rules != ["pl_bl.container_numbers"]
        or record.get("expected_outcome") != "needs_review"
        or not isinstance(adjudication, dict)
        or adjudication.get("rule_id") != expected_rules[0]
        or adjudication.get("expected_findings") != expected_rules
        or adjudication.get("expected_outcome") != record["expected_outcome"]
        or evidence != expected_evidence
    ):
        raise RepairError("source adjudication evidence is not the known generator-error shape")
    try:
        documents = record["mutated_case"]["documents"]
        bill_container = documents["bill_of_lading"]["raw_record"]["Container No."]
        packing_record = documents["packing_list"]["raw_record"]
    except (KeyError, TypeError) as exc:
        raise RepairError("source container mutation record has an invalid shape") from exc
    if bill_container != before or packing_record.get("Container No.") != after:
        raise RepairError("source container mutation values do not match their provenance")
    setup = record.get("controlled_parent_setup")
    expected_add_path = [
        "documents",
        "packing_list",
        "raw_record",
        "Container No.",
    ]
    if setup != [{"op": "add", "path": expected_add_path, "value": before}]:
        raise RepairError("source container setup is not the known generator-error shape")
    try:
        controlled_before = _valid_container_number(before)
        replacement = _different_valid_container_number(controlled_before)
    except EvalAssetError as exc:
        raise RepairError("source container cannot produce a valid isolated mismatch") from exc
    setup[:] = [
        {
            "op": "replace",
            "path": [
                "documents",
                "bill_of_lading",
                "raw_record",
                "Container No.",
            ],
            "before": before,
            "value": controlled_before,
        },
        {
            "op": "add",
            "path": expected_add_path,
            "value": controlled_before,
        },
    ]
    mutation["before"] = controlled_before
    mutation["after"] = replacement
    evidence["before"] = controlled_before
    evidence["after"] = replacement
    documents["bill_of_lading"]["raw_record"]["Container No."] = controlled_before
    packing_record["Container No."] = replacement
    controlled_parent = copy.deepcopy(record["mutated_case"])
    controlled_parent["documents"]["packing_list"]["raw_record"]["Container No."] = (
        controlled_before
    )
    record["controlled_parent_sha256"] = _canonical_sha256(controlled_parent)
    reconstructed_source = copy.deepcopy(controlled_parent)
    del reconstructed_source["documents"]["packing_list"]["raw_record"]["Container No."]
    reconstructed_source["documents"]["bill_of_lading"]["raw_record"]["Container No."] = before
    if _canonical_sha256(reconstructed_source) != record.get("source_case_sha256"):
        raise RepairError("repaired setup does not reconstruct the pinned source case")
    return before, replacement


def _update_manifest(
    manifest: dict[str, Any],
    *,
    defects_sha256: str,
    defects_byte_size: int,
) -> None:
    controlled = manifest.get("controlled_defects")
    generated = manifest.get("generated_files")
    if (
        manifest.get("schema_version") != "trade-document-evaluation-assets/v3"
        or manifest.get("evaluation_version") != "evaluation-v2"
        or not isinstance(controlled, dict)
        or controlled.get("sha256") != KNOWN_SOURCE_DEFECTS_SHA256
        or controlled.get("relative_path") != DEFECTS_RELATIVE_PATH.as_posix()
        or not isinstance(generated, list)
    ):
        raise RepairError("source evaluation manifest is not the known diagnostic bundle")
    controlled["sha256"] = defects_sha256
    controlled["byte_size"] = defects_byte_size
    matches = [
        item
        for item in generated
        if isinstance(item, dict) and item.get("relative_path") == DEFECTS_RELATIVE_PATH.as_posix()
    ]
    if len(matches) != 1 or matches[0].get("sha256") != KNOWN_SOURCE_DEFECTS_SHA256:
        raise RepairError("source manifest does not pin the known controlled defects")
    matches[0]["sha256"] = defects_sha256
    matches[0]["byte_size"] = defects_byte_size
    release_holdout = manifest.get("release_holdout")
    if not isinstance(release_holdout, dict) or release_holdout.get("release_eligible", False):
        raise RepairError("source release-holdout boundary is invalid")
    release_holdout["status"] = RELEASE_HOLDOUT_STATUS
    release_holdout["release_eligible"] = False


def _checksums_bytes(root: Path, manifest: dict[str, Any]) -> bytes:
    generated = manifest.get("generated_files")
    if not isinstance(generated, list):
        raise RepairError("repaired manifest generated-file inventory is invalid")
    relatives = {
        Path(item["relative_path"])
        for item in generated
        if isinstance(item, dict) and isinstance(item.get("relative_path"), str)
    }
    relatives.update({MANIFEST_RELATIVE_PATH, PROVENANCE_RELATIVE_PATH})
    rows: list[str] = []
    for relative in sorted(relatives, key=lambda item: item.as_posix()):
        path = _regular_file(root, relative)
        rows.append(f"{_sha256_file(path)}  {relative.as_posix()}\n")
    return "".join(rows).encode()


def repair_bundle(source_root: Path, output_root: Path) -> dict[str, Any]:
    source = source_root.resolve(strict=True)
    output = output_root.resolve(strict=False)
    if not source.is_dir() or source.is_symlink():
        raise RepairError("source bundle must be a regular directory")
    if output.exists():
        raise RepairError("output bundle already exists")
    if output == source or output.is_relative_to(source):
        raise RepairError("output bundle must be a new sibling location")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise RepairError("output parent must not be a symbolic link")
    _validate_source_tree(source)

    defects_path = _regular_file(source, DEFECTS_RELATIVE_PATH)
    manifest_path = _regular_file(source, MANIFEST_RELATIVE_PATH)
    checksums_path = _regular_file(source, CHECKSUMS_RELATIVE_PATH)
    if (
        _sha256_file(defects_path) != KNOWN_SOURCE_DEFECTS_SHA256
        or _sha256_file(manifest_path) != KNOWN_SOURCE_MANIFEST_SHA256
        or _sha256_file(checksums_path) != KNOWN_SOURCE_CHECKSUMS_SHA256
    ):
        raise RepairError("source bundle hashes are not the exact known generator-error version")

    manifest = _read_json_object(manifest_path, label="evaluation manifest")
    _validate_bundle_inventory(
        source,
        manifest=manifest,
        checksums_path=checksums_path,
        repaired=False,
    )
    records = _read_jsonl(defects_path)
    source_value, replacement_value = _repair_record(records)
    defects_content = _jsonl_bytes(records)
    defects_sha256 = _sha256_bytes(defects_content)
    _update_manifest(
        manifest,
        defects_sha256=defects_sha256,
        defects_byte_size=len(defects_content),
    )
    manifest_content = _json_bytes(manifest)
    manifest_sha256 = _sha256_bytes(manifest_content)
    provenance = {
        "schema_version": "trade-document-evaluation-repair-provenance/v1",
        "status": "confirmed_generator_error_repaired",
        "release_eligible": False,
        "source_bundle_preserved": True,
        "operator_id": OPERATOR_ID,
        "repair_strategy": (
            "canonicalize the source B/L and controlled PL parent to the same valid "
            "ISO 6346 identifier, then mutate the PL exactly once to a different "
            "valid ISO 6346 identifier"
        ),
        "source": {
            "defects_sha256": KNOWN_SOURCE_DEFECTS_SHA256,
            "manifest_sha256": KNOWN_SOURCE_MANIFEST_SHA256,
            "checksums_sha256": KNOWN_SOURCE_CHECKSUMS_SHA256,
        },
        "output": {
            "defects_sha256": defects_sha256,
            "manifest_sha256": manifest_sha256,
        },
        "value_redaction": {
            "source_container": _sha256_bytes(source_value.encode()),
            "replacement_container": _sha256_bytes(replacement_value.encode()),
        },
    }

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-",
        dir=output.parent,
    ) as temporary_parent:
        candidate = Path(temporary_parent) / "bundle"
        shutil.copytree(source, candidate)
        (candidate / DEFECTS_RELATIVE_PATH).write_bytes(defects_content)
        (candidate / MANIFEST_RELATIVE_PATH).write_bytes(manifest_content)
        (candidate / PROVENANCE_RELATIVE_PATH).write_bytes(_json_bytes(provenance))
        (candidate / CHECKSUMS_RELATIVE_PATH).write_bytes(_checksums_bytes(candidate, manifest))
        _validate_bundle_inventory(
            candidate,
            manifest=manifest,
            checksums_path=candidate / CHECKSUMS_RELATIVE_PATH,
            repaired=True,
        )
        os.replace(candidate, output)
    _validate_bundle_inventory(
        output,
        manifest=manifest,
        checksums_path=output / CHECKSUMS_RELATIVE_PATH,
        repaired=True,
    )

    return {
        "status": "PASS",
        "source_bundle_preserved": True,
        "release_eligible": False,
        "operator_id": OPERATOR_ID,
        "defects_sha256": defects_sha256,
        "manifest_sha256": manifest_sha256,
        "output_checksums_sha256": _sha256_file(output / CHECKSUMS_RELATIVE_PATH),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = repair_bundle(args.source_root, args.output_root)
    except RepairError as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
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
