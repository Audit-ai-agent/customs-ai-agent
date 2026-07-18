"""Register the user-supplied CI/PL/B/L JSON corpus without changing its raw bytes.

This tool deliberately separates three concepts:

* raw source values, which are copied byte-for-byte and never corrected;
* observed cross-document conditions, which can be reproduced mechanically; and
* author intent, which remains unknown until it is supported by provenance or
  human adjudication.

The generated anomaly registry therefore never labels a discrepancy as an
intentional defect or a generator error on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable


DATASET_ID = "trade_document_triplets"
DATASET_VERSION = "source-v1"
CASE_COUNT = 100
CASE_ID_PREFIX = "trade-triplet"
SPLIT_SEED = "trade-document-triplets-source-v1"
SPLIT_COUNTS = {
    "development": 60,
    "validation": 20,
    "locked_test": 20,
}
EVALUATION_SPLIT_SEED = "trade-document-triplets-evaluation-v2-holdout-2026-07-18"
EVALUATION_REBALANCE_SEED = "trade-document-triplets-evaluation-v2-rebalance-2026-07-18"
EVALUATION_VERSION = "evaluation-v2"

INVENTORY_SCHEMA = "trade-document-triplets-inventory/v1"
PROVENANCE_SCHEMA = "trade-document-triplets-provenance/v1"
CASE_SCHEMA = "trade-document-triplet-case/v1"
SPLIT_SCHEMA = "trade-document-triplets-splits/v1"
EVALUATION_SPLIT_SCHEMA = "trade-document-evaluation-splits/v2"
ANOMALY_SCHEMA = "trade-document-triplets-anomaly/v1"
DATASET_LOCK_SCHEMA = "trade-document-triplets-dataset-lock/v1"
DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "dataset.lock.json"
)

UNKNOWN_INTENT_STATUSES = frozenset({"ambiguous", "coverage_gap"})


@dataclass(frozen=True)
class RawDocumentSpec:
    document_type: str
    filename: str
    required_keys: frozenset[str]


RAW_DOCUMENT_SPECS = (
    RawDocumentSpec(
        document_type="commercial_invoice",
        filename="CI_100_Final.json",
        required_keys=frozenset(
            {
                "Shipper / Seller",
                "Consignee (or For account & risk of Messrs)",
                "Departure Date",
                "Vessel / Flight",
                "From",
                "To",
                "Invoice No. and Date",
                "L/C No. and Date",
                "Buyer (if other than consignee)",
                "Other References (Country of Origin)",
                "Terms of Delivery and Payment",
                "Shipping Marks",
                "No. & Kind of Packages",
                "Goods Description",
                "Quantity",
                "Unit Price",
                "Amount",
                "Signed By",
            }
        ),
    ),
    RawDocumentSpec(
        document_type="packing_list",
        filename="PL_100_Final.json",
        required_keys=frozenset(
            {
                "Seller",
                "Consignee",
                "Departure Date",
                "Vessel / Flight",
                "From",
                "To",
                "Invoice No. and Date",
                "Buyer (if other than consignee)",
                "Other References",
                "Shipping Marks",
                "No. & Kind of Packages",
                "Goods Details",
                "Gross Weight",
                "Measurement",
                "Signed By",
            }
        ),
    ),
    RawDocumentSpec(
        document_type="bill_of_lading",
        filename="BL_100_Final.json",
        required_keys=frozenset(
            {
                "Shipper / Exporter",
                "Consignee",
                "Notify Party",
                "Ocean Vessel",
                "Port of Loading",
                "Place of Receipt",
                "Voyage No.",
                "Port of Discharge",
                "Place of Delivery",
                "Final Destination",
                "B/L No. (Bill of Lading Number)",
                "Flag",
                "Container No.",
                "Seal No. / Marks & No.",
                "No. & Kinds of Containers or Packages",
                "Description of Goods",
                "Gross Weight",
                "Freight and Charges",
                "Revenue Tons",
                "Rate",
                "Per",
                "Prepaid",
                "Collect",
                "Freight Prepaid At",
                "Freight Payable At",
                "Place and Date of Issue",
                "No. of Original B/L",
                "Laden on Board (Date / Signature)",
                "Carrier / Agent Signature",
            }
        ),
    ),
)

PL_GOODS_DETAIL_KEYS = frozenset({"item", "quantity", "weight"})


class DatasetRegistrationError(ValueError):
    """Raised when a source cannot be registered without weakening provenance."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetRegistrationError(f"JSON contains a duplicate object key: {key}")
        result[key] = value
    return result


def _load_json_array(path: Path, spec: RawDocumentSpec) -> tuple[bytes, list[dict[str, Any]]]:
    if not path.is_file() or path.is_symlink():
        raise DatasetRegistrationError(f"required regular source file is missing: {spec.filename}")

    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetRegistrationError(f"{spec.filename} must be UTF-8 JSON") from exc

    try:
        loaded = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except DatasetRegistrationError:
        raise
    except json.JSONDecodeError as exc:
        raise DatasetRegistrationError(f"{spec.filename} is invalid JSON: {exc}") from exc

    if not isinstance(loaded, list):
        raise DatasetRegistrationError(f"{spec.filename} must contain one top-level JSON array")
    if len(loaded) != CASE_COUNT:
        raise DatasetRegistrationError(
            f"{spec.filename} must contain {CASE_COUNT} records, found {len(loaded)}"
        )

    records: list[dict[str, Any]] = []
    for index, record in enumerate(loaded):
        if not isinstance(record, dict):
            raise DatasetRegistrationError(f"{spec.filename} record {index} must be a JSON object")
        keys = frozenset(record)
        if keys != spec.required_keys:
            missing = sorted(spec.required_keys - keys)
            unexpected = sorted(keys - spec.required_keys)
            raise DatasetRegistrationError(
                f"{spec.filename} record {index} has a schema mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for key, value in record.items():
            if spec.document_type == "packing_list" and key == "Goods Details":
                continue
            if not isinstance(value, str):
                raise DatasetRegistrationError(
                    f"{spec.filename} record {index} field {key!r} must be a string"
                )

        if spec.document_type == "packing_list":
            goods = record["Goods Details"]
            if not isinstance(goods, list) or not goods:
                raise DatasetRegistrationError(
                    f"{spec.filename} record {index} Goods Details must be a non-empty array"
                )
            for goods_index, detail in enumerate(goods):
                if not isinstance(detail, dict) or frozenset(detail) != PL_GOODS_DETAIL_KEYS:
                    raise DatasetRegistrationError(
                        f"{spec.filename} record {index} Goods Details item "
                        f"{goods_index} has a schema mismatch"
                    )
                if not all(isinstance(value, str) for value in detail.values()):
                    raise DatasetRegistrationError(
                        f"{spec.filename} record {index} Goods Details item "
                        f"{goods_index} fields must be strings"
                    )
        records.append(record)
    return raw_bytes, records


def _validate_source_tree(source_root: Path) -> None:
    expected = {spec.filename for spec in RAW_DOCUMENT_SPECS}
    actual_files = {path.name for path in source_root.iterdir() if path.is_file()}
    actual_directories = [path.name for path in source_root.iterdir() if path.is_dir()]
    missing = sorted(expected - actual_files)
    unexpected = sorted(actual_files - expected)
    if missing or unexpected or actual_directories:
        raise DatasetRegistrationError(
            "source directory must contain exactly the three registered JSON files; "
            f"missing={missing}, unexpected_files={unexpected}, "
            f"unexpected_directories={sorted(actual_directories)}"
        )


def _validate_roots(source_root: Path, dataset_root: Path) -> tuple[Path, Path]:
    if source_root.is_symlink():
        raise DatasetRegistrationError("source root must not be a symbolic link")
    if dataset_root.exists() and dataset_root.is_symlink():
        raise DatasetRegistrationError("dataset root must not be a symbolic link")
    source = source_root.resolve()
    target = dataset_root.resolve()
    if not source.is_dir():
        raise DatasetRegistrationError(f"source root is not a regular directory: {source}")
    if target.exists() and not target.is_dir():
        raise DatasetRegistrationError(f"dataset root is not a regular directory: {target}")
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise DatasetRegistrationError("source and dataset roots must be separate directory trees")
    _validate_source_tree(source)
    return source, target


def _copy_raw_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if _sha256_file(source) != expected_sha256:
        raise DatasetRegistrationError(f"source changed while registering: {source.name}")

    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise DatasetRegistrationError(
                f"raw destination is not a regular file: {destination.name}"
            )
        if _sha256_file(destination) != expected_sha256:
            raise DatasetRegistrationError(
                f"raw destination differs from the source and will not be overwritten: "
                f"{destination.name}"
            )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as target_handle:
            temporary_path = Path(target_handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)

        if _sha256_file(source) != expected_sha256:
            raise DatasetRegistrationError(f"source changed while being copied: {source.name}")
        if _sha256_file(temporary_path) != expected_sha256:
            raise DatasetRegistrationError(f"raw copy hash verification failed: {source.name}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_generated(path: Path, content: bytes, *, refresh_derived: bool) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise DatasetRegistrationError(f"generated output is not a regular file: {path}")
        if path.read_bytes() == content:
            return
        if not refresh_derived:
            raise DatasetRegistrationError(
                f"generated output differs: {path}; use --refresh-derived after review"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary_path = Path(file_handle.name)
            file_handle.write(content)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n"
    ).encode("utf-8")


def _load_and_validate_lock(
    lock_path: Path,
    raw_bytes_by_filename: dict[str, bytes],
    records_by_type: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], str]:
    if not lock_path.is_file() or lock_path.is_symlink():
        raise DatasetRegistrationError(f"dataset lock is missing: {lock_path}")
    lock_bytes = lock_path.read_bytes()
    try:
        lock = json.loads(
            lock_bytes.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except DatasetRegistrationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetRegistrationError(f"dataset lock is invalid UTF-8 JSON: {lock_path}") from exc
    if not isinstance(lock, dict):
        raise DatasetRegistrationError("dataset lock must be a JSON object")
    if lock.get("schema_version") != DATASET_LOCK_SCHEMA:
        raise DatasetRegistrationError("dataset lock schema_version does not match")
    if lock.get("dataset_id") != DATASET_ID or lock.get("dataset_version") != DATASET_VERSION:
        raise DatasetRegistrationError("dataset lock identity does not match this registration")
    if lock.get("defect_intent_status") != "unknown":
        raise DatasetRegistrationError(
            "source-v1 lock must not assign defect intent without adjudication"
        )
    if "contains_business_values" in lock:
        raise DatasetRegistrationError(
            "dataset lock uses the ambiguous legacy contains_business_values key"
        )
    if lock.get("descriptor_contains_business_values") is not False:
        raise DatasetRegistrationError("dataset descriptor must not contain source business values")
    if lock.get("raw_dataset_contains_business_values") is not True:
        raise DatasetRegistrationError("dataset lock must acknowledge raw business values")
    if lock.get("value_bearing_artifacts_committable") is not False:
        raise DatasetRegistrationError(
            "dataset lock must keep value-bearing artifacts out of the repository"
        )

    expected_registration = lock.get("expected_registration")
    expected_summary = {
        "case_count": CASE_COUNT,
        "document_count": CASE_COUNT * len(RAW_DOCUMENT_SPECS),
        "pairing_basis": "shared_zero_based_array_index",
        "raw_total_bytes": sum(len(content) for content in raw_bytes_by_filename.values()),
        "source_explicit_case_id": False,
        "split_counts": SPLIT_COUNTS,
    }
    if expected_registration != expected_summary:
        raise DatasetRegistrationError(
            "dataset lock registration summary does not match the supplied source"
        )

    lock_files = lock.get("raw_files")
    if not isinstance(lock_files, list):
        raise DatasetRegistrationError("dataset lock raw_files must be an array")
    expected_by_filename: dict[str, dict[str, Any]] = {}
    for item in lock_files:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise DatasetRegistrationError("dataset lock contains an invalid raw file record")
        filename = item["filename"]
        if filename in expected_by_filename:
            raise DatasetRegistrationError(
                f"dataset lock contains a duplicate raw filename: {filename}"
            )
        expected_by_filename[filename] = item

    expected_names = {spec.filename for spec in RAW_DOCUMENT_SPECS}
    if set(expected_by_filename) != expected_names:
        raise DatasetRegistrationError("dataset lock raw filename set does not match")
    for spec in RAW_DOCUMENT_SPECS:
        item = expected_by_filename[spec.filename]
        actual_sha256 = _sha256_bytes(raw_bytes_by_filename[spec.filename])
        actual_size = len(raw_bytes_by_filename[spec.filename])
        actual_count = len(records_by_type[spec.document_type])
        if item.get("document_type") != spec.document_type:
            raise DatasetRegistrationError(f"dataset lock document type mismatch: {spec.filename}")
        if item.get("sha256") != actual_sha256:
            raise DatasetRegistrationError(f"dataset lock SHA-256 mismatch: {spec.filename}")
        if item.get("byte_size") != actual_size:
            raise DatasetRegistrationError(f"dataset lock byte size mismatch: {spec.filename}")
        if item.get("record_count") != actual_count:
            raise DatasetRegistrationError(f"dataset lock record count mismatch: {spec.filename}")
    return lock, _sha256_bytes(lock_bytes)


def _schema_sha256(keys: Iterable[str]) -> str:
    canonical = json.dumps(sorted(keys), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(canonical)


def _case_id(index: int) -> str:
    return f"{CASE_ID_PREFIX}-{index + 1:04d}"


def _number(text: str) -> Decimal:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    if match is None:
        raise DatasetRegistrationError(f"cannot parse a decimal number from {text!r}")
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation as exc:
        raise DatasetRegistrationError(f"invalid decimal number in {text!r}") from exc


def _decimal_places(text: str) -> int:
    match = re.search(r"-?\d[\d,]*(?:\.(\d+))?", text)
    if match is None:
        raise DatasetRegistrationError(f"cannot determine decimal precision from {text!r}")
    return len(match.group(1) or "")


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _amount_rounding_compatible(
    quantity_text: str,
    unit_price_text: str,
    amount_text: str,
) -> tuple[bool, Decimal, Decimal]:
    quantity = _number(quantity_text)
    unit_price = _number(unit_price_text)
    amount = _number(amount_text)
    computed = quantity * unit_price
    difference = amount - computed

    unit_half_step = Decimal(5).scaleb(-(_decimal_places(unit_price_text) + 1))
    amount_half_step = Decimal(5).scaleb(-(_decimal_places(amount_text) + 1))
    possible_total_low = quantity * (unit_price - unit_half_step)
    possible_total_high = quantity * (unit_price + unit_half_step)
    represented_amount_low = amount - amount_half_step
    represented_amount_high = amount + amount_half_step
    compatible = max(possible_total_low, represented_amount_low) < min(
        possible_total_high, represented_amount_high
    )
    return compatible, computed, difference


def _new_anomaly(
    *,
    anomaly_id: str,
    scope: str,
    rule_id: str,
    status: str,
    observed_condition: str,
    expected_agent_behavior: str,
    evidence: dict[str, Any],
    possible_interpretations: list[str],
    case_id: str | None = None,
) -> dict[str, Any]:
    if status not in UNKNOWN_INTENT_STATUSES:
        raise DatasetRegistrationError(f"registration cannot assign intent status: {status}")
    record: dict[str, Any] = {
        "schema_version": ANOMALY_SCHEMA,
        "anomaly_id": anomaly_id,
        "scope": scope,
        "rule_id": rule_id,
        "status": status,
        "intent_status": "unknown",
        "observed_condition": observed_condition,
        "expected_agent_behavior": expected_agent_behavior,
        "evidence": evidence,
        "possible_interpretations": possible_interpretations,
        "scoring_eligible": False,
        "adjudication_required": True,
    }
    if case_id is not None:
        record["case_id"] = case_id
    return record


def _comparison_value(value: Any) -> str:
    if not isinstance(value, str):
        raise DatasetRegistrationError("cross-document comparison value must be a string")
    return value.strip()


def _audit_cases(
    records_by_type: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    invoices = records_by_type["commercial_invoice"]
    packing_lists = records_by_type["packing_list"]
    bills = records_by_type["bill_of_lading"]
    anomalies: list[dict[str, Any]] = []
    rule_counts: dict[str, Counter[str]] = {}

    comparison_rules: tuple[
        tuple[
            str,
            Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str],
            Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str],
        ],
        ...,
    ] = (
        (
            "ci_pl.invoice_no_and_date",
            lambda ci, pl, bl: ci["Invoice No. and Date"],
            lambda ci, pl, bl: pl["Invoice No. and Date"],
        ),
        (
            "ci_pl.seller",
            lambda ci, pl, bl: ci["Shipper / Seller"],
            lambda ci, pl, bl: pl["Seller"],
        ),
        (
            "ci_bl.shipper",
            lambda ci, pl, bl: ci["Shipper / Seller"],
            lambda ci, pl, bl: bl["Shipper / Exporter"],
        ),
        (
            "ci_pl.consignee",
            lambda ci, pl, bl: ci["Consignee (or For account & risk of Messrs)"],
            lambda ci, pl, bl: pl["Consignee"],
        ),
        (
            "ci_bl.consignee",
            lambda ci, pl, bl: ci["Consignee (or For account & risk of Messrs)"],
            lambda ci, pl, bl: bl["Consignee"],
        ),
        (
            "ci_pl.departure_date",
            lambda ci, pl, bl: ci["Departure Date"],
            lambda ci, pl, bl: pl["Departure Date"],
        ),
        (
            "ci_bl.departure_date",
            lambda ci, pl, bl: ci["Departure Date"],
            lambda ci, pl, bl: bl["Laden on Board (Date / Signature)"].split(" / ", 1)[0],
        ),
        (
            "pl_bl.vessel_and_voyage",
            lambda ci, pl, bl: pl["Vessel / Flight"],
            lambda ci, pl, bl: f"{bl['Ocean Vessel']} / {bl['Voyage No.']}",
        ),
        (
            "ci_bl.port_of_loading",
            lambda ci, pl, bl: ci["From"],
            lambda ci, pl, bl: bl["Port of Loading"],
        ),
        (
            "ci_bl.port_of_discharge",
            lambda ci, pl, bl: ci["To"],
            lambda ci, pl, bl: bl["Port of Discharge"],
        ),
        (
            "ci_pl.country_of_origin_reference",
            lambda ci, pl, bl: ci["Other References (Country of Origin)"],
            lambda ci, pl, bl: pl["Other References"],
        ),
        (
            "ci_pl.shipping_marks",
            lambda ci, pl, bl: ci["Shipping Marks"],
            lambda ci, pl, bl: pl["Shipping Marks"],
        ),
        (
            "ci_pl.package_description",
            lambda ci, pl, bl: ci["No. & Kind of Packages"],
            lambda ci, pl, bl: pl["No. & Kind of Packages"],
        ),
        (
            "ci_pl.first_goods_item",
            lambda ci, pl, bl: ci["Goods Description"],
            lambda ci, pl, bl: pl["Goods Details"][0]["item"],
        ),
        (
            "ci_pl.first_goods_quantity",
            lambda ci, pl, bl: ci["Quantity"],
            lambda ci, pl, bl: pl["Goods Details"][0]["quantity"],
        ),
        (
            "pl_bl.gross_weight",
            lambda ci, pl, bl: pl["Gross Weight"],
            lambda ci, pl, bl: bl["Gross Weight"],
        ),
        (
            "pl_bl.measurement_numeric",
            lambda ci, pl, bl: _decimal_text(_number(pl["Measurement"])),
            lambda ci, pl, bl: _decimal_text(_number(bl["Revenue Tons"])),
        ),
    )

    containment_rules: tuple[
        tuple[
            str,
            Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str],
            Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], str],
        ],
        ...,
    ] = (
        (
            "pl_bl.shipping_marks_contained",
            lambda ci, pl, bl: pl["Shipping Marks"],
            lambda ci, pl, bl: bl["Seal No. / Marks & No."],
        ),
        (
            "pl_bl.package_description_contained",
            lambda ci, pl, bl: pl["No. & Kind of Packages"],
            lambda ci, pl, bl: bl["No. & Kinds of Containers or Packages"],
        ),
        (
            "ci_bl.goods_description_contained",
            lambda ci, pl, bl: ci["Goods Description"],
            lambda ci, pl, bl: bl["Description of Goods"],
        ),
    )

    amount_not_exact_count = 0
    amount_rounding_compatible_count = 0
    additional_item_case_count = 0
    additional_item_count = 0

    for index, (ci, pl, bl) in enumerate(zip(invoices, packing_lists, bills, strict=True)):
        case_id = _case_id(index)
        for rule_id, left_getter, right_getter in comparison_rules:
            left = _comparison_value(left_getter(ci, pl, bl))
            right = _comparison_value(right_getter(ci, pl, bl))
            counts = rule_counts.setdefault(rule_id, Counter())
            if left == right:
                counts["matched"] += 1
                continue
            counts["mismatched"] += 1
            anomalies.append(
                _new_anomaly(
                    anomaly_id=f"{case_id}:{rule_id}",
                    scope="case",
                    case_id=case_id,
                    rule_id=rule_id,
                    status="ambiguous",
                    observed_condition="source_values_are_not_exactly_equal",
                    expected_agent_behavior="preserve_raw_values_and_require_review",
                    evidence={"left_raw_value": left, "right_raw_value": right},
                    possible_interpretations=[
                        "intentional test defect",
                        "unintended generator divergence",
                        "legitimate document-specific representation",
                    ],
                )
            )

        for rule_id, needle_getter, container_getter in containment_rules:
            needle = _comparison_value(needle_getter(ci, pl, bl))
            container = _comparison_value(container_getter(ci, pl, bl))
            counts = rule_counts.setdefault(rule_id, Counter())
            if needle and needle in container:
                counts["matched"] += 1
                continue
            counts["mismatched"] += 1
            anomalies.append(
                _new_anomaly(
                    anomaly_id=f"{case_id}:{rule_id}",
                    scope="case",
                    case_id=case_id,
                    rule_id=rule_id,
                    status="ambiguous",
                    observed_condition="expected_source_substring_is_not_present",
                    expected_agent_behavior="preserve_raw_values_and_require_review",
                    evidence={"expected_substring": needle, "container_raw_value": container},
                    possible_interpretations=[
                        "intentional test defect",
                        "unintended generator divergence",
                        "legitimate wording variation",
                    ],
                )
            )

        freight_counts = rule_counts.setdefault("bl.freight_payment_exclusive", Counter())
        prepaid = _comparison_value(bl["Prepaid"])
        collect = _comparison_value(bl["Collect"])
        if (prepaid == "Y" and not collect) or (collect == "Y" and not prepaid):
            freight_counts["matched"] += 1
        else:
            freight_counts["mismatched"] += 1
            anomalies.append(
                _new_anomaly(
                    anomaly_id=f"{case_id}:bl.freight_payment_exclusive",
                    scope="case",
                    case_id=case_id,
                    rule_id="bl.freight_payment_exclusive",
                    status="ambiguous",
                    observed_condition="prepaid_and_collect_are_not_mutually_exclusive_flags",
                    expected_agent_behavior="preserve_raw_values_and_require_review",
                    evidence={"prepaid": prepaid, "collect": collect},
                    possible_interpretations=[
                        "intentional test defect",
                        "unintended generator divergence",
                        "unsupported freight payment notation",
                    ],
                )
            )

        compatible, computed, difference = _amount_rounding_compatible(
            ci["Quantity"],
            ci["Unit Price"],
            ci["Amount"],
        )
        if difference != 0:
            amount_not_exact_count += 1
            if compatible:
                amount_rounding_compatible_count += 1
            anomalies.append(
                _new_anomaly(
                    anomaly_id=f"{case_id}:ci.amount_vs_displayed_unit_price",
                    scope="case",
                    case_id=case_id,
                    rule_id="ci.amount_vs_displayed_unit_price",
                    status="ambiguous",
                    observed_condition=(
                        "displayed_amount_is_not_exactly_quantity_times_displayed_unit_price"
                    ),
                    expected_agent_behavior=(
                        "preserve_source_amount_and_do_not_assign_defect_intent_without_adjudication"
                    ),
                    evidence={
                        "quantity": ci["Quantity"],
                        "unit_price": ci["Unit Price"],
                        "amount": ci["Amount"],
                        "computed_from_displayed_values": _decimal_text(computed),
                        "signed_difference": _decimal_text(difference),
                        "compatible_with_displayed_precision_rounding": compatible,
                    },
                    possible_interpretations=[
                        "an intentional arithmetic test defect",
                        "an amount calculated from a higher-precision unrounded unit price",
                        "an unintended generator arithmetic error",
                    ],
                )
            )

        goods_details = pl["Goods Details"]
        if len(goods_details) > 1:
            extra_count = len(goods_details) - 1
            additional_item_case_count += 1
            additional_item_count += extra_count
            anomalies.append(
                _new_anomaly(
                    anomaly_id=f"{case_id}:ci_pl.additional_packing_items",
                    scope="case",
                    case_id=case_id,
                    rule_id="ci_pl.additional_packing_items",
                    status="ambiguous",
                    observed_condition=(
                        "packing_list_contains_more_than_one_item_while_ci_has_one_item"
                    ),
                    expected_agent_behavior=(
                        "report_unmatched_items_as_ambiguous_without_labeling_a_violation"
                    ),
                    evidence={
                        "commercial_invoice_item_count": 1,
                        "packing_list_item_count": len(goods_details),
                        "additional_packing_item_count": extra_count,
                    },
                    possible_interpretations=[
                        "intentionally injected unmatched packing-list items",
                        "legitimate packing detail or accessories",
                        "a generator schema divergence",
                    ],
                )
            )

    anomalies.extend(
        [
            _new_anomaly(
                anomaly_id="dataset:pairing_by_shared_array_index",
                scope="dataset",
                rule_id="dataset.pairing_by_shared_array_index",
                status="ambiguous",
                observed_condition="source_has_no_explicit_case_identifier",
                expected_agent_behavior=(
                    "use_the_registered_index_mapping_only_for_this_dataset_version"
                ),
                evidence={
                    "case_count": CASE_COUNT,
                    "pairing_basis": "same_zero_based_array_index_in_three_source_files",
                },
                possible_interpretations=[
                    "the shared order is the intended triplet mapping",
                    "the arrays were independently generated and require author confirmation",
                ],
            ),
            _new_anomaly(
                anomaly_id="dataset:hs_code_coverage_gap",
                scope="dataset",
                rule_id="dataset.hs_code_coverage_gap",
                status="coverage_gap",
                observed_condition="none_of_the_registered_source_schemas_contains_an_hs_code_field",
                expected_agent_behavior=(
                    "do_not_infer_or_query_hs_specific_evidence_without_additional_input"
                ),
                evidence={"affected_case_count": CASE_COUNT},
                possible_interpretations=[
                    "HS classification is intentionally outside this dataset",
                    "the generator omitted a field required by a later evaluation",
                ],
            ),
            _new_anomaly(
                anomaly_id="dataset:generation_provenance_gap",
                scope="dataset",
                rule_id="dataset.generation_provenance_gap",
                status="coverage_gap",
                observed_condition="generator_prompt_seed_and_version_are_not_supplied",
                expected_agent_behavior="do_not_claim_generation_reproducibility",
                evidence={
                    "generator": "unknown",
                    "prompt": "unknown",
                    "seed": "unknown",
                },
                possible_interpretations=[
                    "provenance exists outside the supplied folder",
                    "the source was generated without retained provenance",
                ],
            ),
            _new_anomaly(
                anomaly_id="dataset:defect_intent_gap",
                scope="dataset",
                rule_id="dataset.defect_intent_gap",
                status="coverage_gap",
                observed_condition="no_author_supplied_defect_manifest_is_present",
                expected_agent_behavior=(
                    "exclude_unadjudicated_conditions_from_defect_precision_and_recall"
                ),
                evidence={"defect_manifest": "not_supplied"},
                possible_interpretations=[
                    "some discrepancies are intentional test defects",
                    "all discrepancies are incidental generator behavior",
                ],
            ),
            _new_anomaly(
                anomaly_id="dataset:rights_and_distribution_gap",
                scope="dataset",
                rule_id="dataset.rights_and_distribution_gap",
                status="coverage_gap",
                observed_condition="license_and_distribution_permission_are_not_supplied",
                expected_agent_behavior=(
                    "keep_raw_and_value_bearing_derivatives_out_of_the_product_repository"
                ),
                evidence={"license": "unknown", "distribution_permission": "unknown"},
                possible_interpretations=[
                    "the user owns and may distribute the generated data",
                    "the data is restricted to local evaluation",
                ],
            ),
        ]
    )

    anomalies.sort(key=lambda item: item["anomaly_id"])
    audit = {
        "comparison_rules": {
            rule_id: {
                "matched_case_count": counts["matched"],
                "mismatched_case_count": counts["mismatched"],
            }
            for rule_id, counts in sorted(rule_counts.items())
        },
        "observed_conditions": {
            "amount_not_exact_from_displayed_values_case_count": amount_not_exact_count,
            "amount_rounding_compatible_case_count": amount_rounding_compatible_count,
            "additional_packing_item_case_count": additional_item_case_count,
            "additional_packing_item_count": additional_item_count,
            "hs_code_field_case_count": 0,
        },
        "anomaly_registry": {
            "record_count": len(anomalies),
            "status_counts": dict(sorted(Counter(item["status"] for item in anomalies).items())),
            "scoring_eligible_record_count": sum(
                1 for item in anomalies if item["scoring_eligible"]
            ),
        },
    }
    return audit, anomalies


def _build_cases(
    records_by_type: dict[str, list[dict[str, Any]]],
    raw_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(CASE_COUNT):
        documents: dict[str, Any] = {}
        for spec in RAW_DOCUMENT_SPECS:
            documents[spec.document_type] = {
                "source_relative_path": f"raw/{spec.filename}",
                "source_file_sha256": raw_hashes[spec.filename],
                "source_record_index_zero_based": index,
                "raw_record": records_by_type[spec.document_type][index],
            }
        cases.append(
            {
                "schema_version": CASE_SCHEMA,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "case_id": _case_id(index),
                "documents": documents,
                "pairing": {
                    "basis": "shared_zero_based_array_index",
                    "source_explicit_case_id": False,
                    "author_intent_status": "unknown",
                    "adjudication_status": "ambiguous",
                },
            }
        )
    return cases


def _build_splits(case_ids: list[str]) -> dict[str, Any]:
    ranked = sorted(
        case_ids,
        key=lambda case_id: (
            hashlib.sha256(f"{SPLIT_SEED}:{case_id}".encode()).hexdigest(),
            case_id,
        ),
    )
    development_end = SPLIT_COUNTS["development"]
    validation_end = development_end + SPLIT_COUNTS["validation"]
    assignments = {
        "development": sorted(ranked[:development_end]),
        "validation": sorted(ranked[development_end:validation_end]),
        "locked_test": sorted(ranked[validation_end:]),
    }
    canonical_assignments = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SPLIT_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "strategy": {
            "name": "sha256_ranked_case_split",
            "version": "v1",
            "seed": SPLIT_SEED,
            "unit": "case_id",
            "variant_co_location_required": True,
        },
        "counts": {name: len(values) for name, values in assignments.items()},
        "assignments_sha256": _sha256_bytes(canonical_assignments),
        "splits": assignments,
    }


def _rank_case_ids(case_ids: Iterable[str], seed: str) -> list[str]:
    return sorted(
        case_ids,
        key=lambda case_id: (
            hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest(),
            case_id,
        ),
    )


def _build_evaluation_v2_splits(legacy_splits: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the invalidated evaluation-v2 assignment for audit/dev use.

    The source split remains an immutable historical artifact.  Its former
    locked set was evaluated before the release commit.  The v2 assignment also
    uses repository-visible deterministic seeds and is therefore permanently
    release-ineligible even though it differs from the former assignment.
    """

    legacy_assignments = legacy_splits.get("splits")
    if (
        legacy_splits.get("schema_version") != SPLIT_SCHEMA
        or not isinstance(legacy_assignments, dict)
        or set(legacy_assignments) != set(SPLIT_COUNTS)
    ):
        raise DatasetRegistrationError("legacy source-v1 split contract is invalid")
    normalized_legacy: dict[str, list[str]] = {}
    for split_name, expected_count in SPLIT_COUNTS.items():
        case_ids = legacy_assignments.get(split_name)
        if (
            not isinstance(case_ids, list)
            or len(case_ids) != expected_count
            or case_ids != sorted(case_ids)
            or not all(isinstance(case_id, str) for case_id in case_ids)
        ):
            raise DatasetRegistrationError("legacy source-v1 split membership is invalid")
        normalized_legacy[split_name] = list(case_ids)

    old_locked = set(normalized_legacy["locked_test"])
    prior_non_holdout = [
        *normalized_legacy["development"],
        *normalized_legacy["validation"],
    ]
    new_locked = set(
        _rank_case_ids(prior_non_holdout, EVALUATION_SPLIT_SEED)[: SPLIT_COUNTS["locked_test"]]
    )
    if old_locked & new_locked:
        raise DatasetRegistrationError("evaluation-v2 holdout overlaps the compromised holdout")

    development_validation_pool = sorted((set(prior_non_holdout) - new_locked) | old_locked)
    rebalanced = _rank_case_ids(development_validation_pool, EVALUATION_REBALANCE_SEED)
    assignments = {
        "development": sorted(rebalanced[: SPLIT_COUNTS["development"]]),
        "validation": sorted(rebalanced[SPLIT_COUNTS["development"] :]),
        "locked_test": sorted(new_locked),
    }
    flattened = [case_id for values in assignments.values() for case_id in values]
    legacy_flattened = [case_id for values in normalized_legacy.values() for case_id in values]
    if (
        len(flattened) != CASE_COUNT
        or len(set(flattened)) != CASE_COUNT
        or set(flattened) != set(legacy_flattened)
    ):
        raise DatasetRegistrationError("evaluation-v2 splits do not partition the source cases")
    if not old_locked <= set(assignments["development"]) | set(assignments["validation"]):
        raise DatasetRegistrationError("compromised holdout was not fully demoted")

    canonical_assignments = json.dumps(
        assignments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    legacy_content = _json_bytes(legacy_splits)
    return {
        "schema_version": EVALUATION_SPLIT_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "strategy": {
            "name": "compromised_holdout_replacement",
            "version": "v2",
            "holdout_seed": EVALUATION_SPLIT_SEED,
            "rebalance_seed": EVALUATION_REBALANCE_SEED,
            "unit": "case_id",
            "replacement_pool": "legacy_development_plus_validation_only",
            "variant_co_location_required": True,
        },
        "counts": {name: len(values) for name, values in assignments.items()},
        "assignments_sha256": _sha256_bytes(canonical_assignments),
        "splits": assignments,
        "legacy": {
            "schema_version": legacy_splits["schema_version"],
            "assignments_sha256": legacy_splits["assignments_sha256"],
            "document_sha256": _sha256_bytes(legacy_content),
            "splits": normalized_legacy,
            "holdout_status": "holdout_compromised_by_precommit_evaluation",
            "invalidated_evaluation_ids": ["EVAL-0003"],
            "former_locked_use": "development_or_validation_only",
        },
        "holdout_policy": {
            "release_eligible": False,
            "release_status": "release_holdout_v2_publicly_derivable",
            "allowed_use": "development_diagnostic_only",
            "controlled_mutation_generation_allowed_before_release": False,
            "model_or_rule_tuning_allowed": False,
            "replacement_protocol": "sealed-release-holdout-bundle-manifest/v1",
        },
    }


def register_dataset(
    source_root: Path,
    dataset_root: Path,
    *,
    refresh_derived: bool = False,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Validate, copy, and register the source dataset deterministically."""

    source, target = _validate_roots(source_root, dataset_root)

    raw_bytes_by_filename: dict[str, bytes] = {}
    records_by_type: dict[str, list[dict[str, Any]]] = {}
    for spec in RAW_DOCUMENT_SPECS:
        raw_bytes, records = _load_json_array(source / spec.filename, spec)
        raw_bytes_by_filename[spec.filename] = raw_bytes
        records_by_type[spec.document_type] = records
    lock: dict[str, Any] | None = None
    lock_sha256: str | None = None
    if lock_path is not None:
        lock, lock_sha256 = _load_and_validate_lock(
            lock_path,
            raw_bytes_by_filename,
            records_by_type,
        )

    raw_dir = target / "raw"
    if raw_dir.exists():
        unexpected_raw = sorted(
            path.name
            for path in raw_dir.iterdir()
            if path.name not in {spec.filename for spec in RAW_DOCUMENT_SPECS}
        )
        if unexpected_raw:
            raise DatasetRegistrationError(
                f"dataset raw directory contains unregistered entries: {unexpected_raw}"
            )

    raw_hashes = {
        filename: _sha256_bytes(raw_bytes) for filename, raw_bytes in raw_bytes_by_filename.items()
    }
    for spec in RAW_DOCUMENT_SPECS:
        _copy_raw_verified(
            source / spec.filename,
            raw_dir / spec.filename,
            raw_hashes[spec.filename],
        )

    audit, anomalies = _audit_cases(records_by_type)
    cases = _build_cases(records_by_type, raw_hashes)
    splits = _build_splits([case["case_id"] for case in cases])
    evaluation_v2_splits = _build_evaluation_v2_splits(splits)
    cases_content = _jsonl_bytes(cases)
    legacy_splits_content = _json_bytes(splits)
    evaluation_v2_splits_content = _json_bytes(evaluation_v2_splits)
    anomaly_content = _jsonl_bytes(anomalies)

    raw_file_records = []
    for spec in RAW_DOCUMENT_SPECS:
        raw_bytes = raw_bytes_by_filename[spec.filename]
        raw_file_records.append(
            {
                "document_type": spec.document_type,
                "relative_path": f"raw/{spec.filename}",
                "sha256": raw_hashes[spec.filename],
                "byte_size": len(raw_bytes),
                "record_count": len(records_by_type[spec.document_type]),
                "object_key_count": len(spec.required_keys),
                "object_key_schema_sha256": _schema_sha256(spec.required_keys),
            }
        )

    inventory: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA,
        "dataset": {
            "id": DATASET_ID,
            "version": DATASET_VERSION,
            "case_id_format": f"{CASE_ID_PREFIX}-NNNN",
            "lock": (
                {
                    "schema_version": lock["schema_version"],
                    "sha256": lock_sha256,
                }
                if lock is not None
                else None
            ),
        },
        "summary": {
            "case_count": CASE_COUNT,
            "document_count": CASE_COUNT * len(RAW_DOCUMENT_SPECS),
            "document_type_counts": {spec.document_type: CASE_COUNT for spec in RAW_DOCUMENT_SPECS},
            "raw_file_count": len(RAW_DOCUMENT_SPECS),
            "raw_total_bytes": sum(len(value) for value in raw_bytes_by_filename.values()),
            "raw_bytes_preserved": True,
            "source_explicit_case_id": False,
            "pairing_basis": "shared_zero_based_array_index",
            "raw_dataset_contains_business_values": True,
            "author_intent_status": "unknown",
        },
        "raw_files": raw_file_records,
        "audit": audit,
        "derived_artifacts": {
            "cases": {
                "relative_path": "derived/cases.jsonl",
                "sha256": _sha256_bytes(cases_content),
                "byte_size": len(cases_content),
            },
            "legacy_splits": {
                "relative_path": "derived/splits.json",
                "sha256": _sha256_bytes(legacy_splits_content),
                "byte_size": len(legacy_splits_content),
                "status": "holdout_compromised_by_precommit_evaluation",
            },
            "evaluation_v2_splits": {
                "relative_path": "derived/evaluation-splits/v2/splits.json",
                "sha256": _sha256_bytes(evaluation_v2_splits_content),
                "byte_size": len(evaluation_v2_splits_content),
            },
            "anomaly_registry": {
                "relative_path": "anomaly-registry.jsonl",
                "sha256": _sha256_bytes(anomaly_content),
                "byte_size": len(anomaly_content),
            },
        },
    }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "dataset": {"id": DATASET_ID, "version": DATASET_VERSION},
        "source": {
            "description": "user-supplied virtual CI/PL/B/L JSON data",
            "source_directory_name": source.name,
            "raw_file_sha256": dict(sorted(raw_hashes.items())),
        },
        "generation": {
            "generator": "unknown",
            "generator_version": "unknown",
            "prompt": "unknown",
            "seed": "unknown",
            "generated_at": "unknown",
        },
        "pairing": {
            "registered_basis": "shared_zero_based_array_index",
            "source_explicit_case_id": False,
            "author_confirmation": "unknown",
        },
        "defect_intent": {
            "status": "unknown",
            "author_supplied_manifest": False,
            "policy": (
                "Observed discrepancies remain ambiguous and are excluded from defect "
                "precision/recall until adjudicated."
            ),
        },
        "rights_and_distribution": {
            "status": "unknown_pending_confirmation",
            "raw_allowed_in_product_repository": False,
        },
        "raw_handling": {
            "immutable": True,
            "copy_mode": "byte_for_byte_with_sha256_verification",
            "source_deleted": False,
            "derived_values_may_not_overwrite_raw": True,
        },
        "registration": {
            "tool": "tools/register_trade_document_triplets.py",
            "deterministic_outputs": True,
            "dataset_lock_sha256": lock_sha256,
        },
    }

    checksums = "".join(
        f"{raw_hashes[spec.filename]}  raw/{spec.filename}\n" for spec in RAW_DOCUMENT_SPECS
    ).encode("utf-8")
    generated_files = {
        target / "inventory.json": _json_bytes(inventory),
        target / "provenance.json": _json_bytes(provenance),
        target / "checksums.sha256": checksums,
        target / "derived" / "cases.jsonl": cases_content,
        target / "derived" / "splits.json": legacy_splits_content,
        target
        / "derived"
        / "evaluation-splits"
        / "v2"
        / "splits.json": evaluation_v2_splits_content,
        target / "anomaly-registry.jsonl": anomaly_content,
    }
    for path, content in generated_files.items():
        _write_generated(path, content, refresh_derived=refresh_derived)

    for spec in RAW_DOCUMENT_SPECS:
        copied_hash = _sha256_file(raw_dir / spec.filename)
        if copied_hash != raw_hashes[spec.filename]:
            raise DatasetRegistrationError(
                f"post-registration raw verification failed: {spec.filename}"
            )
        if _sha256_file(source / spec.filename) != raw_hashes[spec.filename]:
            raise DatasetRegistrationError(
                f"source changed during post-registration verification: {spec.filename}"
            )
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register the CI/PL/B/L JSON corpus with immutable raw copies, "
            "deterministic splits, and intent-neutral observations."
        )
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Pinned value-free dataset descriptor; defaults to the repository eval lock.",
    )
    parser.add_argument(
        "--refresh-derived",
        action="store_true",
        help="Replace changed generated metadata after review; raw files are never replaced.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        inventory = register_dataset(
            args.source_root,
            args.dataset_root,
            refresh_derived=args.refresh_derived,
            lock_path=args.lock_file,
        )
    except DatasetRegistrationError as exc:
        print(f"dataset registration failed: {exc}")
        return 1

    print(
        "dataset registration passed: "
        f"{inventory['summary']['case_count']} cases, "
        f"{inventory['summary']['document_count']} documents, "
        f"{inventory['audit']['anomaly_registry']['record_count']} "
        "intent-neutral observations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
