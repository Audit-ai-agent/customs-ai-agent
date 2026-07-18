"""Evaluate deterministic cross-document rules on controlled single mutations.

This runner scores only the *delta* between each controlled variant and its
reconstructed immutable parent. Source values are never emitted in the report.
That prevents unadjudicated source anomalies from becoming false Gold labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from customs_ai_agent.nodes.common import parse_decimal
from customs_ai_agent.nodes.normalize import (
    normalize_country,
    normalize_date,
    parse_quantity,
    parse_weight_kg,
)
from customs_ai_agent.nodes.pair import pair_documents
from customs_ai_agent.nodes.validate import validate_cross_documents
from evals.release_holdout_protocol import RELEASE_INVALIDATION_CODE


REPORT_SCHEMA = "deterministic-validation-evaluation-report/v4"
DATASET_ID = "trade_document_triplets"
DATASET_VERSION = "source-v1"
EVALUATION_VERSION = "evaluation-v2"
DATASET_LOCK_SCHEMA = "trade-document-triplets-dataset-lock/v1"
SPLIT_SCHEMA = "trade-document-evaluation-splits/v2"
EVALUATION_MANIFEST_SCHEMA = "trade-document-evaluation-assets/v3"
CONTROLLED_DEFECT_SCHEMA = "trade-document-controlled-defect/v2"
CONTROLLED_DEFECT_RELATIVE_PATH = "injected-defects/cases.jsonl"
EXPECTED_SPLIT_COUNTS = {
    "development": 60,
    "validation": 20,
    "locked_test": 20,
}
CONTROLLED_CASE_SEED = "trade-document-triplets-evaluation-v2-controlled-development"
CANONICAL_DATASET_LOCK_SHA256 = "a5fcdff268fc17651f90235e9f65283e2a2913c2d1aa8466a9b96e3908f3f030"
CANONICAL_DIAGNOSTIC_SPLIT_BYTES_SHA256 = (
    "d7e6b86ef19bbae838a7f5da7b94428637481ab5c15d9599ee3af3e86bda7b64"
)
CANONICAL_CONTROLLED_DEFECTS_SHA256 = (
    "e4f2902b301ac20752c71ab0c96f4b7cc4e265eff44daff8275d61abedb0c762"
)
CANONICAL_EVALUATION_MANIFEST_SHA256 = (
    "cd91e5513e57592b264f36dcb9e15cecd760c3337bf13b43238e601e17ed7470"
)
OPERATOR_EXPECTED_ISSUES = {
    "pl_invoice_number_mismatch": frozenset({"INVOICE_NO_MISMATCH"}),
    "pl_invoice_date_mismatch": frozenset({"INVOICE_DATE_MISMATCH"}),
    "pl_seller_mismatch": frozenset({"SELLER_MISMATCH", "SHIPPER_CONSIGNEE_MISMATCH"}),
    "bl_consignee_mismatch": frozenset({"CONSIGNEE_MISMATCH"}),
    "pl_vessel_mismatch": frozenset({"VESSEL_MISMATCH"}),
    "bl_voyage_mismatch": frozenset({"VOYAGE_NO_MISMATCH"}),
    "bl_port_of_loading_mismatch": frozenset({"PORT_OF_LOADING_MISMATCH"}),
    "bl_port_of_discharge_mismatch": frozenset({"PORT_OF_DISCHARGE_MISMATCH"}),
    "pl_package_mismatch": frozenset({"PACKAGE_MISMATCH"}),
    "bl_gross_weight_mismatch": frozenset({"GROSS_WEIGHT_MISMATCH"}),
    "bl_measurement_mismatch": frozenset({"MEASUREMENT_MISMATCH"}),
    "ci_amount_mismatch": frozenset({"INV_LINE_AMOUNT_MISMATCH"}),
    "pl_container_mismatch": frozenset({"CONTAINER_NO_MISMATCH"}),
    "pl_seal_mismatch": frozenset({"SEAL_NO_MISMATCH"}),
    "bl_freight_both_present": frozenset({"FREIGHT_TERMS_INVALID"}),
    "bl_freight_neither_present": frozenset({"FREIGHT_TERMS_INVALID"}),
    "co_origin_mismatch": frozenset({"ORIGIN_MISMATCH"}),
    "co_exporter_mismatch": frozenset({"EXPORTER_MISMATCH"}),
    "co_importer_mismatch": frozenset({"IMPORTER_MISMATCH"}),
    "co_hs_code_mismatch": frozenset({"HS_CODE_MISMATCH"}),
}
DOCUMENT_KEY_BY_TYPE = {
    "commercial_invoice": "ci",
    "packing_list": "pl",
    "bill_of_lading": "bl",
    "certificate_of_origin": "co",
}


class DeterministicEvaluationError(ValueError):
    """The controlled evaluation input was not contract-valid."""


def _sha256(path: Path, *, label: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DeterministicEvaluationError(f"{label} is not readable") from exc


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeterministicEvaluationError("controlled defect input is not readable JSONL") from exc
    if not records or not all(isinstance(record, dict) for record in records):
        raise DeterministicEvaluationError("controlled defect input must contain object records")
    return records


def _byte_size(path: Path, *, label: str) -> int:
    try:
        return path.stat().st_size
    except OSError as exc:
        raise DeterministicEvaluationError(f"{label} is not readable") from exc


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeterministicEvaluationError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise DeterministicEvaluationError(f"{label} must be a JSON object")
    return value


def _validate_dataset_lock(path: Path) -> str:
    digest = _sha256(path, label="dataset lock")
    if digest != CANONICAL_DATASET_LOCK_SHA256:
        raise DeterministicEvaluationError("dataset lock hash is not the canonical release hash")
    value = _json_object(path, label="dataset lock")
    registration = value.get("expected_registration")
    if (
        value.get("schema_version") != DATASET_LOCK_SCHEMA
        or value.get("dataset_id") != DATASET_ID
        or value.get("dataset_version") != DATASET_VERSION
        or value.get("defect_intent_status") != "unknown"
        or not isinstance(registration, dict)
        or registration.get("case_count") != 100
        or registration.get("document_count") != 300
        or registration.get("split_counts") != EXPECTED_SPLIT_COUNTS
    ):
        raise DeterministicEvaluationError("dataset lock contract is invalid")
    return digest


def _validate_split_bytes(path: Path) -> str:
    """Bind diagnostic provenance without parsing or deriving any membership."""

    digest = _sha256(path, label="registered splits")
    if digest != CANONICAL_DIAGNOSTIC_SPLIT_BYTES_SHA256:
        raise DeterministicEvaluationError(
            "registered split bytes are not the canonical diagnostic input"
        )
    return digest


def _controlled_assignments(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    for record in records:
        source_case_id = record.get("source_case_id")
        variant_id = record.get("variant_id")
        mutation = record.get("mutation")
        if (
            not isinstance(source_case_id, str)
            or not isinstance(variant_id, str)
            or not isinstance(mutation, dict)
            or not isinstance(mutation.get("operator_id"), str)
        ):
            raise DeterministicEvaluationError("controlled defect identity is invalid")
        assignments.append(
            {
                "source_case_id": source_case_id,
                "operator_id": mutation["operator_id"],
                "variant_id": variant_id,
            }
        )
    return assignments


def _validate_controlled_scope(
    records: list[dict[str, Any]],
    *,
    controlled_case_ids: list[str],
    controlled_manifest: Mapping[str, Any],
    defects_sha256: str,
    defects_byte_size: int,
) -> str:
    if len(records) != len(controlled_case_ids):
        raise DeterministicEvaluationError(
            "controlled defect input must contain all 20 development cases"
        )

    assignments = _controlled_assignments(records)
    operator_ids: list[str] = []
    for record, expected_case_id, expected_operator_id in zip(
        records,
        controlled_case_ids,
        OPERATOR_EXPECTED_ISSUES,
        strict=True,
    ):
        mutation = record["mutation"]
        operator_id = mutation["operator_id"]
        expected_variant_id = f"{expected_case_id}--{operator_id}"
        mutated_case = record.get("mutated_case")
        if (
            record.get("schema_version") != CONTROLLED_DEFECT_SCHEMA
            or record.get("source_case_id") != expected_case_id
            or record.get("variant_id") != expected_variant_id
            or record.get("mutation_count") != 1
            or operator_id != expected_operator_id
            or not isinstance(mutated_case, dict)
            or mutated_case.get("case_id") != expected_case_id
        ):
            raise DeterministicEvaluationError(
                "controlled defect records do not match the hash-locked development order"
            )
        operator_ids.append(operator_id)

    expected_operator_ids = sorted(OPERATOR_EXPECTED_ISSUES)
    if sorted(set(operator_ids)) != expected_operator_ids:
        raise DeterministicEvaluationError(
            "controlled defect input does not cover every pinned mutation operator"
        )

    assignments_sha256 = _canonical_sha256(assignments)
    expected_manifest = {
        "relative_path": CONTROLLED_DEFECT_RELATIVE_PATH,
        "sha256": defects_sha256,
        "byte_size": defects_byte_size,
        "generation_split": "development",
        "release_holdout_used": False,
        "selection_seed": CONTROLLED_CASE_SEED,
        "case_count": len(records),
        "case_ids": controlled_case_ids,
        "operator_ids": expected_operator_ids,
        "case_operator_assignments_sha256": assignments_sha256,
    }
    if dict(controlled_manifest) != expected_manifest:
        raise DeterministicEvaluationError("controlled defect manifest contract is invalid")
    return assignments_sha256


def _controlled_case_ids_from_manifest(
    controlled_manifest: Mapping[str, Any],
) -> list[str]:
    case_ids = controlled_manifest.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != len(OPERATOR_EXPECTED_ISSUES)
        or case_ids != sorted(case_ids)
        or len(case_ids) != len(set(case_ids))
        or not all(
            isinstance(case_id, str) and re.fullmatch(r"trade-triplet-\d{4}", case_id) is not None
            for case_id in case_ids
        )
    ):
        raise DeterministicEvaluationError(
            "controlled defect manifest has invalid development identities"
        )
    return case_ids


def _validate_evaluation_manifest(
    path: Path,
    *,
    defects_sha256: str,
    defects_byte_size: int,
    dataset_lock_sha256: str,
    splits_sha256: str,
    records: list[dict[str, Any]],
) -> tuple[str, str, list[str]]:
    digest = _sha256(path, label="evaluation manifest")
    if digest != CANONICAL_EVALUATION_MANIFEST_SHA256:
        raise DeterministicEvaluationError(
            "evaluation manifest hash is not the canonical release hash"
        )
    value = _json_object(path, label="evaluation manifest")
    source = value.get("source")
    generated_files = value.get("generated_files")
    controlled_manifest = value.get("controlled_defects")
    coverage = value.get("controlled_operator_coverage")
    counts = value.get("counts")
    if (
        value.get("schema_version") != EVALUATION_MANIFEST_SCHEMA
        or value.get("dataset_id") != DATASET_ID
        or value.get("dataset_version") != DATASET_VERSION
        or value.get("evaluation_version") != EVALUATION_VERSION
        or not isinstance(source, dict)
        or source.get("dataset_lock_sha256") != dataset_lock_sha256
        or source.get("evaluation_split_sha256") != splits_sha256
        or not isinstance(counts, dict)
        or counts.get("controlled_defect_count") != len(OPERATOR_EXPECTED_ISSUES)
        or not isinstance(generated_files, list)
        or not isinstance(controlled_manifest, dict)
        or not isinstance(coverage, dict)
    ):
        raise DeterministicEvaluationError("evaluation manifest contract is invalid")
    controlled_case_ids = _controlled_case_ids_from_manifest(controlled_manifest)

    business_coverage = coverage.get("business_fields")
    workflow_coverage = coverage.get("workflow_boundaries")
    expected_business_coverage_sha256 = _canonical_sha256(
        [
            {
                "operator_id": record["mutation"]["operator_id"],
                "expected_rule_ids": record["expected_rule_ids"],
                "expected_outcome": record["expected_outcome"],
            }
            for record in records
        ]
    )
    if (
        not isinstance(business_coverage, dict)
        or business_coverage.get("required_operator_ids") != list(OPERATOR_EXPECTED_ISSUES)
        or business_coverage.get("observed_operator_ids") != list(OPERATOR_EXPECTED_ISSUES)
        or business_coverage.get("operator_count") != len(OPERATOR_EXPECTED_ISSUES)
        or business_coverage.get("expected_outcome") != "needs_review"
        or business_coverage.get("coverage_sha256") != expected_business_coverage_sha256
        or not isinstance(workflow_coverage, dict)
        or workflow_coverage.get("required_operator_ids")
        != [
            "required_document_missing",
            "duplicate_document_type",
            "ocr_conflict_unresolved",
            "evidence_partial",
            "evidence_failed",
            "evidence_ambiguous",
        ]
        or workflow_coverage.get("observed_operator_ids")
        != workflow_coverage.get("required_operator_ids")
        or workflow_coverage.get("operator_count") != 6
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(workflow_coverage.get("descriptor_sha256") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(workflow_coverage.get("expected_outcomes_sha256") or ""),
        )
        is None
    ):
        raise DeterministicEvaluationError(
            "controlled operator coverage manifest is incomplete; "
            "input must cover every pinned mutation operator"
        )

    controlled_files = [
        item
        for item in generated_files
        if isinstance(item, dict) and item.get("relative_path") == CONTROLLED_DEFECT_RELATIVE_PATH
    ]
    if controlled_files != [
        {
            "relative_path": CONTROLLED_DEFECT_RELATIVE_PATH,
            "sha256": defects_sha256,
            "byte_size": defects_byte_size,
        }
    ]:
        raise DeterministicEvaluationError(
            "evaluation manifest does not pin the controlled defect input"
        )
    assignments_sha256 = _validate_controlled_scope(
        records,
        controlled_case_ids=controlled_case_ids,
        controlled_manifest=controlled_manifest,
        defects_sha256=defects_sha256,
        defects_byte_size=defects_byte_size,
    )
    return digest, assignments_sha256, controlled_case_ids


def _split_invoice_reference(value: Any) -> tuple[str | None, str | None]:
    text = str(value or "").strip()
    number, separator, raw_date = text.partition(" / ")
    if not separator:
        return text or None, None
    return number.strip() or None, normalize_date(raw_date.strip())


def _split_vessel(value: Any) -> tuple[str | None, str | None]:
    text = str(value or "").strip()
    vessel, separator, voyage = text.rpartition(" / ")
    if not separator:
        return text or None, None
    return vessel.strip() or None, voyage.strip() or None


def _country(value: Any) -> str | None:
    text = str(value or "").strip()
    prefix = "Country of Origin:"
    if text.casefold().startswith(prefix.casefold()):
        text = text[len(prefix) :].strip()
    return normalize_country(text)


def _same_as_consignee(value: Any, consignee: str | None) -> str | None:
    text = str(value or "").strip()
    if text.casefold() == "same as consignee":
        return consignee
    return text or consignee


def _list_value(value: Any) -> list[str]:
    return [item.strip() for item in re.split(r"[,;]+", str(value or "")) if item.strip()]


def _invoice(document: dict[str, Any]) -> dict[str, Any]:
    invoice_no, invoice_date = _split_invoice_reference(document.get("Invoice No. and Date"))
    vessel, voyage = _split_vessel(document.get("Vessel / Flight"))
    consignee = str(document.get("Consignee (or For account & risk of Messrs)") or "").strip()
    raw_amount = document.get("Amount")
    raw_quantity = document.get("Quantity")
    raw_unit_price = document.get("Unit Price")
    quantity, unit = parse_quantity(raw_quantity)
    return {
        "document_id": "ci",
        "filename": "commercial_invoice.synthetic",
        "document_type": "invoice",
        "fields": {
            "invoice_no": invoice_no,
            "invoice_date": invoice_date,
            "departure_date": normalize_date(document.get("Departure Date")),
            "vessel": vessel,
            "voyage_no": voyage,
            "port_of_loading": document.get("From"),
            "port_of_discharge": document.get("To"),
            "seller": document.get("Shipper / Seller"),
            "buyer": _same_as_consignee(
                document.get("Buyer (if other than consignee)"),
                consignee or None,
            ),
            "consignee": consignee or None,
            "country_of_origin": _country(document.get("Other References (Country of Origin)")),
            "shipping_marks": document.get("Shipping Marks"),
            "packages": document.get("No. & Kind of Packages"),
            "total_amount": parse_decimal(raw_amount),
        },
        "raw_fields": {
            "total_amount": raw_amount,
        },
        "field_evidence": {},
        "line_items": [
            {
                "item_id": "ci-line-1",
                "description": document.get("Goods Description"),
                "hs_code": document.get("HS Code"),
                "quantity": quantity,
                "unit": unit,
                "unit_price": parse_decimal(raw_unit_price),
                "amount": parse_decimal(raw_amount),
                "origin_country": _country(document.get("Other References (Country of Origin)")),
                "raw_fields": {
                    "quantity": raw_quantity,
                    "unit_price": raw_unit_price,
                    "amount": raw_amount,
                },
                "field_evidence": {},
            }
        ],
        "extraction_confidence": 1.0,
    }


def _packing_list(document: dict[str, Any]) -> dict[str, Any]:
    invoice_no, invoice_date = _split_invoice_reference(document.get("Invoice No. and Date"))
    vessel, voyage = _split_vessel(document.get("Vessel / Flight"))
    consignee = str(document.get("Consignee") or "").strip()
    line_items = []
    for index, item in enumerate(document.get("Goods Details") or [], start=1):
        quantity, unit = parse_quantity(item.get("quantity"))
        line_items.append(
            {
                "item_id": f"pl-line-{index}",
                "description": item.get("item"),
                "quantity": quantity,
                "unit": unit,
                "gross_weight_kg": parse_weight_kg(item.get("weight")),
                "raw_fields": {
                    "quantity": item.get("quantity"),
                    "gross_weight_kg": item.get("weight"),
                },
                "field_evidence": {},
            }
        )
    return {
        "document_id": "pl",
        "filename": "packing_list.synthetic",
        "document_type": "packing_list",
        "fields": {
            "invoice_no": invoice_no,
            "invoice_date": invoice_date,
            "departure_date": normalize_date(document.get("Departure Date")),
            "vessel": vessel,
            "voyage_no": voyage,
            "port_of_loading": document.get("From"),
            "port_of_discharge": document.get("To"),
            "seller": document.get("Seller"),
            "buyer": _same_as_consignee(
                document.get("Buyer (if other than consignee)"),
                consignee or None,
            ),
            "consignee": consignee or None,
            "country_of_origin": _country(document.get("Other References")),
            "shipping_marks": document.get("Shipping Marks"),
            "packages": document.get("No. & Kind of Packages"),
            "container_numbers": _list_value(document.get("Container No.")),
            "seal_numbers": _list_value(document.get("Seal No.")),
            "total_gross_weight_kg": parse_weight_kg(document.get("Gross Weight")),
            "total_measurement_cbm": parse_decimal(document.get("Measurement")),
        },
        "raw_fields": {
            "total_gross_weight_kg": document.get("Gross Weight"),
            "total_measurement_cbm": document.get("Measurement"),
        },
        "field_evidence": {},
        "line_items": line_items,
        "extraction_confidence": 1.0,
    }


def _bill_of_lading(document: dict[str, Any]) -> dict[str, Any]:
    seal_and_marks = [
        item.strip()
        for item in str(document.get("Seal No. / Marks & No.") or "").split(" // ")
        if item.strip()
    ]
    return {
        "document_id": "bl",
        "filename": "bill_of_lading.synthetic",
        "document_type": "bill_of_lading",
        "fields": {
            "bl_no": document.get("B/L No. (Bill of Lading Number)"),
            "shipper": document.get("Shipper / Exporter"),
            "consignee": document.get("Consignee"),
            "vessel": document.get("Ocean Vessel"),
            "voyage_no": document.get("Voyage No."),
            "port_of_loading": document.get("Port of Loading"),
            "port_of_discharge": document.get("Port of Discharge"),
            "container_numbers": _list_value(document.get("Container No.")),
            "seal_numbers": seal_and_marks[:1],
            "shipping_marks": (
                " // ".join(seal_and_marks[1:]) if len(seal_and_marks) > 1 else None
            ),
            "packages": document.get("No. & Kinds of Containers or Packages"),
            "gross_weight_kg": parse_weight_kg(document.get("Gross Weight")),
            "measurement_cbm": parse_decimal(document.get("Revenue Tons")),
            "freight_prepaid": document.get("Prepaid"),
            "freight_collect": document.get("Collect"),
        },
        "raw_fields": {
            "gross_weight_kg": document.get("Gross Weight"),
            "measurement_cbm": document.get("Revenue Tons"),
            "freight_prepaid": document.get("Prepaid"),
            "freight_collect": document.get("Collect"),
        },
        "field_evidence": {},
        "line_items": [],
        "extraction_confidence": 1.0,
    }


def _certificate_of_origin(document: dict[str, Any]) -> dict[str, Any]:
    goods = document.get("Goods Details") or []
    hs_codes = [
        str(item.get("hs_code")).strip()
        for item in goods
        if isinstance(item, dict) and item.get("hs_code")
    ]
    return {
        "document_id": "co",
        "filename": "certificate_of_origin.synthetic",
        "document_type": "certificate_of_origin",
        "fields": {
            "certificate_no": document.get("Certificate No."),
            "exporter": document.get("Exporter"),
            "importer": document.get("Consignee"),
            "origin_country": _country(document.get("Country of Origin")),
            "hs_codes": hs_codes,
            "goods_description": (
                goods[0].get("item") if goods and isinstance(goods[0], dict) else None
            ),
        },
        "raw_fields": {},
        "field_evidence": {},
        "line_items": [],
        "extraction_confidence": 1.0,
    }


def _case_documents(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    documents = case.get("documents")
    if not isinstance(documents, dict):
        raise DeterministicEvaluationError("controlled case has no document mapping")
    try:
        result = {
            "ci": _invoice(documents["commercial_invoice"]["raw_record"]),
            "pl": _packing_list(documents["packing_list"]["raw_record"]),
            "bl": _bill_of_lading(documents["bill_of_lading"]["raw_record"]),
        }
        certificate = documents.get("certificate_of_origin")
        if isinstance(certificate, dict):
            result["co"] = _certificate_of_origin(certificate["raw_record"])
        return result
    except (KeyError, TypeError, AttributeError) as exc:
        raise DeterministicEvaluationError("controlled case document shape is invalid") from exc


def _issue_codes(case: dict[str, Any]) -> tuple[set[str], bool]:
    state: dict[str, Any] = {
        "normalized_documents": _case_documents(case),
        "llm_mode": "mock",
        "audit_trail": [],
        "requires_human_review": False,
    }
    state.update(pair_documents(state))
    result = validate_cross_documents(state)
    return (
        {str(issue["issue_code"]) for issue in result["validation_issues"]},
        bool(result["requires_human_review"]),
    )


def _reconstruct_parent(record: dict[str, Any]) -> dict[str, Any]:
    mutated = record.get("mutated_case")
    mutation = record.get("mutation")
    if not isinstance(mutated, dict) or not isinstance(mutation, dict):
        raise DeterministicEvaluationError("controlled defect record is missing its mutation")
    parent = copy.deepcopy(mutated)
    document_type = mutation.get("document_type")
    field_path = mutation.get("field_path")
    if (
        document_type not in DOCUMENT_KEY_BY_TYPE
        or not isinstance(field_path, list)
        or not field_path
        or not all(isinstance(part, (str, int)) for part in field_path)
    ):
        raise DeterministicEvaluationError("controlled mutation target is unsupported")
    try:
        target: Any = parent["documents"][document_type]["raw_record"]
        for part in field_path[:-1]:
            target = target[part]
        target[field_path[-1]] = mutation["before"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeterministicEvaluationError(
            "controlled mutation cannot reconstruct its parent"
        ) from exc
    if _canonical_sha256(parent) != record.get("controlled_parent_sha256"):
        raise DeterministicEvaluationError(
            "reconstructed parent does not match the pinned controlled-parent hash"
        )

    source = copy.deepcopy(parent)
    setup_operations = record.get("controlled_parent_setup")
    if not isinstance(setup_operations, list):
        raise DeterministicEvaluationError("controlled parent setup is invalid")
    for operation in reversed(setup_operations):
        if (
            not isinstance(operation, dict)
            or operation.get("op") not in {"add", "replace"}
            or not isinstance(operation.get("path"), list)
            or not operation["path"]
        ):
            raise DeterministicEvaluationError("controlled parent setup operation is invalid")
        expected_keys = (
            {"op", "path", "value"}
            if operation["op"] == "add"
            else {"op", "path", "before", "value"}
        )
        if set(operation) != expected_keys:
            raise DeterministicEvaluationError("controlled parent setup operation is invalid")
        path = operation["path"]
        try:
            target = source
            for part in path[:-1]:
                target = target[part]
            leaf = path[-1]
            if target[leaf] != operation["value"]:
                raise DeterministicEvaluationError("controlled parent setup value does not match")
            if not isinstance(target, dict):
                raise DeterministicEvaluationError(
                    "controlled parent setup removal target is invalid"
                )
            if operation["op"] == "add":
                del target[leaf]
            else:
                target[leaf] = operation["before"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeterministicEvaluationError(
                "controlled parent setup cannot reconstruct source"
            ) from exc
    if _canonical_sha256(source) != record.get("source_case_sha256"):
        raise DeterministicEvaluationError(
            "controlled parent does not reconstruct the pinned source hash"
        )
    return parent


def evaluate_controlled_records(
    records: list[dict[str, Any]],
    *,
    expected_case_ids: list[str],
) -> dict[str, Any]:
    """Score one complete 20-operator set after its outer gate is validated."""

    if len(records) != len(OPERATOR_EXPECTED_ISSUES) or len(expected_case_ids) != len(records):
        raise DeterministicEvaluationError(
            "controlled record set must contain one case per pinned business operator"
        )
    true_positive = 0
    false_positive = 0
    false_negative = 0
    cases_requiring_review = 0
    results = []
    seen_variants: set[str] = set()
    for record, expected_case_id, expected_operator_id in zip(
        records,
        sorted(expected_case_ids),
        OPERATOR_EXPECTED_ISSUES,
        strict=True,
    ):
        variant_id = str(record.get("variant_id") or "")
        if not variant_id or variant_id in seen_variants:
            raise DeterministicEvaluationError(
                "controlled variant IDs must be unique and non-empty"
            )
        seen_variants.add(variant_id)
        mutation = record.get("mutation") or {}
        operator_id = mutation.get("operator_id")
        if (
            operator_id != expected_operator_id
            or record.get("source_case_id") != expected_case_id
            or variant_id != f"{expected_case_id}--{operator_id}"
        ):
            raise DeterministicEvaluationError(
                "controlled business operator assignment is not canonical"
            )
        parent = _reconstruct_parent(record)
        baseline_codes, _ = _issue_codes(parent)
        variant_codes, requires_review = _issue_codes(record["mutated_case"])
        delta_codes = variant_codes - baseline_codes
        expected_codes = set(OPERATOR_EXPECTED_ISSUES[operator_id])
        tp = len(delta_codes & expected_codes)
        fp = len(delta_codes - expected_codes)
        fn = len(expected_codes - delta_codes)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        cases_requiring_review += int(requires_review)
        results.append(
            {
                "variant_id": variant_id,
                "operator_id": operator_id,
                "expected_issue_codes": sorted(expected_codes),
                "detected_delta_issue_codes": sorted(delta_codes),
                "requires_human_review": requires_review,
                "pass": fp == 0 and fn == 0 and requires_review,
            }
        )

    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    passed = (
        precision == 1.0
        and recall == 1.0
        and cases_requiring_review == len(records)
        and all(result["pass"] for result in results)
    )
    return {
        "passed": passed,
        "case_count": len(records),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "cases_requiring_human_review": cases_requiring_review,
        "results": results,
    }


def evaluate(
    defects_path: Path,
    *,
    manifest_path: Path,
    splits_path: Path,
    dataset_lock_path: Path,
) -> tuple[int, dict[str, Any]]:
    records = _jsonl(defects_path)
    defects_sha256 = _sha256(defects_path, label="controlled defect input")
    if defects_sha256 != CANONICAL_CONTROLLED_DEFECTS_SHA256:
        raise DeterministicEvaluationError(
            "controlled defect input hash is not the canonical diagnostic hash"
        )
    defects_byte_size = _byte_size(defects_path, label="controlled defect input")
    dataset_lock_sha256 = _validate_dataset_lock(dataset_lock_path)
    splits_sha256 = _validate_split_bytes(splits_path)
    (
        manifest_sha256,
        controlled_assignments_sha256,
        controlled_case_ids,
    ) = _validate_evaluation_manifest(
        manifest_path,
        defects_sha256=defects_sha256,
        defects_byte_size=defects_byte_size,
        dataset_lock_sha256=dataset_lock_sha256,
        splits_sha256=splits_sha256,
        records=records,
    )
    metrics = evaluate_controlled_records(
        records,
        expected_case_ids=controlled_case_ids,
    )
    passed = bool(metrics["passed"])
    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": "CONTROLLED_SINGLE_MUTATION_DEVELOPMENT",
        "status": "PASS" if passed else "FAIL",
        "input_sha256": defects_sha256,
        "evaluation_manifest_sha256": manifest_sha256,
        "dataset_lock_sha256": dataset_lock_sha256,
        "split_sha256": splits_sha256,
        "evaluation_version": EVALUATION_VERSION,
        "release_eligible": False,
        "release_holdout_status": RELEASE_INVALIDATION_CODE,
        "release_holdout_used": False,
        "controlled_assignments_sha256": controlled_assignments_sha256,
        "operator_ids": sorted(OPERATOR_EXPECTED_ISSUES),
        "case_count": metrics["case_count"],
        "true_positive": metrics["true_positive"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "cases_requiring_human_review": metrics["cases_requiring_human_review"],
        "results": metrics["results"],
        "redaction": {
            "source_values": "OMITTED",
            "absolute_paths": "OMITTED",
            "api_keys": "OMITTED",
        },
    }
    return (0 if passed else 1), report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score deterministic rules on controlled single mutations."
    )
    parser.add_argument("--defects", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--dataset-lock", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        exit_code, report = evaluate(
            args.defects,
            manifest_path=args.manifest,
            splits_path=args.splits,
            dataset_lock_path=args.dataset_lock,
        )
    except DeterministicEvaluationError as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA,
                    "scope": "CONTROLLED_SINGLE_MUTATION_DEVELOPMENT",
                    "status": "FAIL",
                    "evaluation_version": EVALUATION_VERSION,
                    "release_eligible": False,
                    "release_holdout_status": RELEASE_INVALIDATION_CODE,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
