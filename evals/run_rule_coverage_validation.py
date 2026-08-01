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
from collections import Counter
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
from tools.dataset_provenance import DatasetProvenanceError, verify_dataset_provenance


REPORT_SCHEMA = "deterministic-validation-evaluation-report/v5"
DATASET_ID = "trade_document_triplets"
DATASET_VERSION = "source-v1"
EVALUATION_VERSION = "rule-coverage-v1"
DATASET_LOCK_SCHEMA = "trade-document-triplets-dataset-lock/v1"
SELECTION_DESCRIPTOR_SCHEMA = "trade-document-rule-coverage-descriptor/v1"
EVALUATION_MANIFEST_SCHEMA = "trade-document-rule-coverage-assets/v1"
CONTROLLED_DEFECT_SCHEMA = "trade-document-controlled-defect/v2"
CONTROLLED_DEFECT_RELATIVE_PATH = "injected-defects/cases.jsonl"
NEGATIVE_CONTROL_SCHEMA = "trade-document-negative-control/v1"
NEGATIVE_CONTROL_RELATIVE_PATH = "negative-controls/cases.jsonl"
NEGATIVE_CONTROL_OPERATOR_ID = "pl_extra_item_ambiguous_not_scored"
RELEASE_STATUS = "not_release_evidence_public_development_diagnostic"
EXPECTED_SPLIT_COUNTS = {
    "development": 60,
    "validation": 20,
    "locked_test": 20,
}
CONTROLLED_CASE_SEED = "trade-document-triplets-rule-coverage-v1-public-development"
CANONICAL_DATASET_LOCK_SHA256 = "78b4da7184f500716b7a2191cec42ec98fcf89b2c16b325fbc4c1d67c31273ad"
CANONICAL_SELECTION_DESCRIPTOR_SHA256 = (
    "cd64db824947be629063533e4dbaa29e6baa31ac46b09973786140418ec87101"
)
CANONICAL_CONTROLLED_DEFECTS_SHA256 = (
    "6a1fc44a9c4896dd15b55d54877b894157d6f25bc930c4c8837b83844c1fe8df"
)
CANONICAL_NEGATIVE_CONTROLS_SHA256 = (
    "34bfd935f438857ef273a379100b02189dbe94b6384b9622ebade925ded698a7"
)
CANONICAL_EVALUATION_MANIFEST_SHA256 = (
    "e7f5800733ec042bd069dd39821e824edd327888dce89bffa1604d284382466b"
)
FindingSignature = tuple[str, str, str]
OPERATOR_EXPECTED_FINDINGS: dict[str, tuple[FindingSignature, ...]] = {
    "pl_invoice_number_mismatch": (("INVOICE_NO_MISMATCH", "ci", "pl"),),
    "pl_invoice_date_mismatch": (("INVOICE_DATE_MISMATCH", "ci", "pl"),),
    "pl_seller_mismatch": (
        ("SELLER_MISMATCH", "ci", "pl"),
        ("SHIPPER_CONSIGNEE_MISMATCH", "bl", "pl"),
    ),
    "bl_consignee_mismatch": (
        ("CONSIGNEE_MISMATCH", "bl", "pl"),
        ("CONSIGNEE_MISMATCH", "bl", "ci"),
    ),
    "pl_vessel_mismatch": (
        ("VESSEL_MISMATCH", "ci", "pl"),
        ("VESSEL_MISMATCH", "bl", "pl"),
    ),
    "bl_voyage_mismatch": (("VOYAGE_NO_MISMATCH", "bl", "pl"),),
    "bl_port_of_loading_mismatch": (("PORT_OF_LOADING_MISMATCH", "bl", "pl"),),
    "bl_port_of_discharge_mismatch": (("PORT_OF_DISCHARGE_MISMATCH", "bl", "pl"),),
    "pl_package_mismatch": (
        ("PACKAGE_MISMATCH", "ci", "pl"),
        ("PACKAGE_MISMATCH", "bl", "pl"),
    ),
    "bl_gross_weight_mismatch": (("GROSS_WEIGHT_MISMATCH", "bl", "pl"),),
    "bl_measurement_mismatch": (("MEASUREMENT_MISMATCH", "bl", "pl"),),
    "ci_amount_mismatch": (("INV_LINE_AMOUNT_MISMATCH", "ci", "ci"),),
    "pl_container_mismatch": (("CONTAINER_NO_MISMATCH", "bl", "pl"),),
    "pl_seal_mismatch": (("SEAL_NO_MISMATCH", "bl", "pl"),),
    "bl_freight_both_present": (("FREIGHT_TERMS_INVALID", "bl", "bl"),),
    "bl_freight_neither_present": (("FREIGHT_TERMS_INVALID", "bl", "bl"),),
    "co_origin_mismatch": (("ORIGIN_MISMATCH", "ci", "co"),),
    "co_exporter_mismatch": (("EXPORTER_MISMATCH", "ci", "co"),),
    "co_importer_mismatch": (("IMPORTER_MISMATCH", "ci", "co"),),
    "co_hs_code_mismatch": (("HS_CODE_MISMATCH", "ci", "co"),),
    "pl_consignee_mismatch": (
        ("CONSIGNEE_MISMATCH", "ci", "pl"),
        ("CONSIGNEE_MISMATCH", "bl", "pl"),
    ),
    "pl_departure_date_mismatch": (("DEPARTURE_DATE_MISMATCH", "ci", "pl"),),
    "ci_port_of_loading_mismatch": (("PORT_OF_LOADING_MISMATCH", "ci", "pl"),),
    "ci_port_of_discharge_mismatch": (("PORT_OF_DISCHARGE_MISMATCH", "ci", "pl"),),
    "pl_country_of_origin_mismatch": (("COUNTRY_OF_ORIGIN_MISMATCH", "ci", "pl"),),
    "pl_shipping_marks_mismatch": (("SHIPPING_MARKS_MISMATCH", "ci", "pl"),),
    "pl_line_quantity_mismatch": (("QTY_MISMATCH_INVOICE_PACKING", "ci", "pl"),),
    "pl_line_net_gross_invalid": (("NET_GROSS_WEIGHT_INVALID", "pl", "pl"),),
    "pl_line_item_unmatched": (("LINE_ITEM_UNMATCHED", "ci", "pl"),),
}
OPERATOR_EXPECTED_ISSUES = {
    operator_id: frozenset(signature[0] for signature in signatures)
    for operator_id, signatures in OPERATOR_EXPECTED_FINDINGS.items()
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


def _validate_selection_descriptor(path: Path) -> str:
    """Bind the public development selector without release-split membership."""

    digest = _sha256(path, label="selection descriptor")
    if digest != CANONICAL_SELECTION_DESCRIPTOR_SHA256:
        raise DeterministicEvaluationError(
            "selection descriptor hash is not the canonical diagnostic hash"
        )
    value = _json_object(path, label="selection descriptor")
    selection = value.get("selection")
    policy = value.get("policy")
    if (
        value.get("schema_version") != SELECTION_DESCRIPTOR_SCHEMA
        or value.get("dataset_id") != DATASET_ID
        or value.get("dataset_version") != DATASET_VERSION
        or value.get("evaluation_version") != EVALUATION_VERSION
        or value.get("scope") != "public_registered_synthetic_development_diagnostic"
        or value.get("release_eligible") is not False
        or value.get("positive_operator_ids") != list(OPERATOR_EXPECTED_FINDINGS)
        or value.get("negative_control_operator_ids") != ["pl_extra_item_ambiguous_not_scored"]
        or value.get("finding_grain") != ["issue_code", "source_document_id", "target_document_id"]
        or not isinstance(selection, dict)
        or selection.get("case_pool") != "all_registered_public_synthetic_cases"
        or selection.get("case_count") != len(OPERATOR_EXPECTED_FINDINGS) + 1
        or selection.get("positive_case_count") != len(OPERATOR_EXPECTED_FINDINGS)
        or selection.get("negative_control_case_count") != 1
        or selection.get("membership_embedded") is not False
        or selection.get("release_split_membership_consulted") is not False
        or selection.get("seed") != CONTROLLED_CASE_SEED
        or not isinstance(policy, dict)
        or policy.get("allowed_use") != "development_diagnostic_only"
        or policy.get("release_claim_allowed") is not False
        or policy.get("model_or_rule_selection_claim_allowed") is not False
        or policy.get("sealed_unseen_release_bundle_required_for_release") is not True
    ):
        raise DeterministicEvaluationError("selection descriptor contract is invalid")
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
            "controlled defect input must contain one development case per pinned operator"
        )

    assignments = _controlled_assignments(records)
    operator_ids: list[str] = []
    for record, expected_case_id, expected_operator_id in zip(
        records,
        controlled_case_ids,
        OPERATOR_EXPECTED_FINDINGS,
        strict=True,
    ):
        mutation = record["mutation"]
        operator_id = mutation["operator_id"]
        expected_variant_id = f"{expected_case_id}--{operator_id}"
        mutated_case = record.get("mutated_case")
        expected_rule_ids = record.get("expected_rule_ids")
        adjudication = record.get("adjudication")
        evidence = adjudication.get("evidence") if isinstance(adjudication, dict) else None
        expected_evidence = {
            key: mutation.get(key)
            for key in (
                "operator_id",
                "document_type",
                "field_path",
                "before",
                "after",
            )
        }
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
        if (
            not isinstance(expected_rule_ids, list)
            or not expected_rule_ids
            or not all(isinstance(rule_id, str) and rule_id for rule_id in expected_rule_ids)
            or record.get("expected_outcome") != "needs_review"
            or not isinstance(adjudication, dict)
            or adjudication.get("rule_id") != expected_rule_ids[0]
            or adjudication.get("expected_findings") != expected_rule_ids
            or adjudication.get("expected_outcome") != record["expected_outcome"]
            or evidence != expected_evidence
        ):
            raise DeterministicEvaluationError(
                "controlled adjudication evidence does not match its mutation"
            )
        operator_ids.append(operator_id)

    expected_operator_ids = sorted(OPERATOR_EXPECTED_FINDINGS)
    if sorted(set(operator_ids)) != expected_operator_ids:
        raise DeterministicEvaluationError(
            "controlled defect input does not cover every pinned mutation operator"
        )

    assignments_sha256 = _canonical_sha256(assignments)
    expected_manifest = {
        "relative_path": CONTROLLED_DEFECT_RELATIVE_PATH,
        "sha256": defects_sha256,
        "byte_size": defects_byte_size,
        "generation_scope": "public_registered_synthetic_development_diagnostic",
        "release_split_membership_consulted": False,
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
        or len(case_ids) != len(OPERATOR_EXPECTED_FINDINGS)
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
    negative_controls_sha256: str,
    negative_controls_byte_size: int,
    dataset_lock_sha256: str,
    selection_descriptor_sha256: str,
    records: list[dict[str, Any]],
) -> tuple[str, str, list[str]]:
    digest = _sha256(path, label="evaluation manifest")
    if digest != CANONICAL_EVALUATION_MANIFEST_SHA256:
        raise DeterministicEvaluationError(
            "evaluation manifest hash is not the canonical diagnostic hash"
        )
    value = _json_object(path, label="evaluation manifest")
    source = value.get("source")
    generated_files = value.get("generated_files")
    controlled_manifest = value.get("controlled_defects")
    coverage = value.get("controlled_operator_coverage")
    counts = value.get("counts")
    release_holdout = value.get("release_holdout")
    if (
        value.get("schema_version") != EVALUATION_MANIFEST_SCHEMA
        or value.get("dataset_id") != DATASET_ID
        or value.get("dataset_version") != DATASET_VERSION
        or value.get("evaluation_version") != EVALUATION_VERSION
        or value.get("scope") != "public_registered_synthetic_development_diagnostic"
        or not isinstance(source, dict)
        or set(source)
        != {
            "dataset_lock_sha256",
            "inventory_sha256",
            "raw_checksums_sha256",
            "registered_cases_sha256",
            "release_split_membership_consulted",
            "selection_descriptor_sha256",
        }
        or source.get("dataset_lock_sha256") != dataset_lock_sha256
        or source.get("selection_descriptor_sha256") != selection_descriptor_sha256
        or source.get("release_split_membership_consulted") is not False
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(source.get(name) or "")) is None
            for name in (
                "inventory_sha256",
                "raw_checksums_sha256",
                "registered_cases_sha256",
            )
        )
        or not isinstance(counts, dict)
        or counts
        != {
            "controlled_defect_count": len(OPERATOR_EXPECTED_FINDINGS),
            "negative_control_count": 1,
            "total_variant_count": len(OPERATOR_EXPECTED_FINDINGS) + 1,
        }
        or not isinstance(generated_files, list)
        or not isinstance(controlled_manifest, dict)
        or not isinstance(coverage, dict)
        or release_holdout
        != {
            "release_eligible": False,
            "sealed_unseen_release_bundle_required": True,
            "status": RELEASE_STATUS,
        }
    ):
        raise DeterministicEvaluationError("evaluation manifest contract is invalid")
    controlled_case_ids = _controlled_case_ids_from_manifest(controlled_manifest)

    business_coverage = coverage.get("business_fields")
    negative_coverage = coverage.get("negative_controls")
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
        or business_coverage.get("required_operator_ids") != list(OPERATOR_EXPECTED_FINDINGS)
        or business_coverage.get("observed_operator_ids") != list(OPERATOR_EXPECTED_FINDINGS)
        or business_coverage.get("operator_count") != len(OPERATOR_EXPECTED_FINDINGS)
        or business_coverage.get("expected_outcome") != "needs_review"
        or business_coverage.get("finding_grain")
        != ["issue_code", "source_document_id", "target_document_id"]
        or business_coverage.get("coverage_sha256") != expected_business_coverage_sha256
        or negative_coverage
        != {
            "relative_path": NEGATIVE_CONTROL_RELATIVE_PATH,
            "sha256": negative_controls_sha256,
            "byte_size": negative_controls_byte_size,
            "required_operator_ids": [NEGATIVE_CONTROL_OPERATOR_ID],
            "observed_operator_ids": [NEGATIVE_CONTROL_OPERATOR_ID],
            "operator_count": 1,
            "expected_finding_delta_count": 0,
        }
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
    negative_files = [
        item
        for item in generated_files
        if isinstance(item, dict) and item.get("relative_path") == NEGATIVE_CONTROL_RELATIVE_PATH
    ]
    if negative_files != [
        {
            "relative_path": NEGATIVE_CONTROL_RELATIVE_PATH,
            "sha256": negative_controls_sha256,
            "byte_size": negative_controls_byte_size,
        }
    ]:
        raise DeterministicEvaluationError(
            "evaluation manifest does not pin the negative-control input"
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
                "net_weight_kg": parse_weight_kg(item.get("net_weight")),
                "raw_fields": {
                    "quantity": item.get("quantity"),
                    "gross_weight_kg": item.get("weight"),
                    "net_weight_kg": item.get("net_weight"),
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


def _evaluate_case(
    case: dict[str, Any],
) -> tuple[Counter[FindingSignature], bool, dict[str, Any], bool]:
    state: dict[str, Any] = {
        "normalized_documents": _case_documents(case),
        "llm_mode": "mock",
        "audit_trail": [],
        "requires_human_review": False,
    }
    pairing_update = pair_documents(state)
    state.update(pairing_update)
    result = validate_cross_documents(state)
    return (
        Counter(
            (
                str(issue["issue_code"]),
                str(issue.get("source_document_id") or ""),
                str(issue.get("target_document_id") or ""),
            )
            for issue in result["validation_issues"]
        ),
        bool(result["requires_human_review"]),
        dict(pairing_update["pairing_result"]),
        bool(pairing_update["requires_human_review"]),
    )


def _finding_signatures(
    case: dict[str, Any],
) -> tuple[Counter[FindingSignature], bool]:
    findings, requires_review, _pairing, _pairing_requires_review = _evaluate_case(case)
    return findings, requires_review


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
        leaf = field_path[-1]
        if target[leaf] != mutation.get("after"):
            raise DeterministicEvaluationError(
                "controlled mutated value does not match mutation.after"
            )
        target[leaf] = mutation["before"]
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
    """Score one complete relation-aware operator set after its outer gate is validated."""

    if len(records) != len(OPERATOR_EXPECTED_FINDINGS) or len(expected_case_ids) != len(records):
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
        OPERATOR_EXPECTED_FINDINGS,
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
        baseline_findings, _ = _finding_signatures(parent)
        variant_findings, requires_review = _finding_signatures(record["mutated_case"])
        delta_findings = variant_findings - baseline_findings
        removed_findings = baseline_findings - variant_findings
        expected_findings = Counter(OPERATOR_EXPECTED_FINDINGS[operator_id])
        tp = sum((delta_findings & expected_findings).values())
        fp = sum((delta_findings - expected_findings).values()) + sum(removed_findings.values())
        fn = sum((expected_findings - delta_findings).values())
        true_positive += tp
        false_positive += fp
        false_negative += fn
        cases_requiring_review += int(requires_review)
        expected_signatures = sorted(expected_findings.elements())
        detected_signatures = sorted(delta_findings.elements())
        results.append(
            {
                "variant_id": variant_id,
                "operator_id": operator_id,
                "expected_issue_codes": sorted({signature[0] for signature in expected_signatures}),
                "detected_delta_issue_codes": sorted(
                    {signature[0] for signature in detected_signatures}
                ),
                "expected_findings": [
                    {
                        "issue_code": issue_code,
                        "source_document_id": source_document_id,
                        "target_document_id": target_document_id,
                    }
                    for issue_code, source_document_id, target_document_id in expected_signatures
                ],
                "detected_delta_findings": [
                    {
                        "issue_code": issue_code,
                        "source_document_id": source_document_id,
                        "target_document_id": target_document_id,
                    }
                    for issue_code, source_document_id, target_document_id in detected_signatures
                ],
                "removed_baseline_finding_count": sum(removed_findings.values()),
                "requires_human_review": requires_review,
                "pass": (fp == 0 and fn == 0 and not removed_findings and requires_review),
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


def evaluate_negative_control_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 1:
        raise DeterministicEvaluationError(
            "negative-control input must contain exactly one pinned case"
        )
    record = records[0]
    mutation = record.get("mutation")
    mutated_case = record.get("mutated_case")
    expected_pairing = record.get("expected_pairing")
    if (
        record.get("schema_version") != NEGATIVE_CONTROL_SCHEMA
        or not isinstance(mutation, dict)
        or mutation.get("operator_id") != NEGATIVE_CONTROL_OPERATOR_ID
        or mutation.get("operation") != "add"
        or mutation.get("document_type") != "packing_list"
        or not isinstance(mutation.get("field_path"), list)
        or len(mutation["field_path"]) != 2
        or mutation["field_path"][0] != "Goods Details"
        or not isinstance(mutation["field_path"][1], int)
        or not isinstance(mutation.get("after"), dict)
        or record.get("mutation_count") != 1
        or record.get("expected_finding_delta") != []
        or not isinstance(mutated_case, dict)
        or not isinstance(expected_pairing, dict)
        or expected_pairing
        != {
            "unmatched_packing_item_status": "ambiguous_not_scored",
            "line_item_pairing_status": "deterministic_with_unscored_packing_extras",
            "requires_human_review_from_extra_item": False,
        }
    ):
        raise DeterministicEvaluationError("negative-control record contract is invalid")
    case_id = record.get("source_case_id")
    if (
        not isinstance(case_id, str)
        or record.get("variant_id") != f"{case_id}--{NEGATIVE_CONTROL_OPERATOR_ID}"
        or mutated_case.get("case_id") != case_id
    ):
        raise DeterministicEvaluationError("negative-control identity is invalid")

    source = copy.deepcopy(mutated_case)
    try:
        goods = source["documents"]["packing_list"]["raw_record"]["Goods Details"]
        index = mutation["field_path"][1]
        if (
            not isinstance(goods, list)
            or index != len(goods) - 1
            or goods[index] != mutation["after"]
        ):
            raise DeterministicEvaluationError(
                "negative-control added item does not match mutation.after"
            )
        goods.pop(index)
    except (KeyError, IndexError, TypeError) as exc:
        raise DeterministicEvaluationError(
            "negative-control source cannot be reconstructed"
        ) from exc
    if _canonical_sha256(source) != record.get("source_case_sha256"):
        raise DeterministicEvaluationError(
            "negative control does not reconstruct its pinned source"
        )

    (
        baseline_findings,
        baseline_review,
        baseline_pairing,
        baseline_pairing_requires_review,
    ) = _evaluate_case(source)
    (
        variant_findings,
        variant_review,
        pairing,
        pairing_requires_review,
    ) = _evaluate_case(mutated_case)
    added_findings = variant_findings - baseline_findings
    removed_findings = baseline_findings - variant_findings
    expected_item_id = f"pl-line-{mutation['field_path'][1] + 1}"
    baseline_unmatched_ids = baseline_pairing.get("unmatched_packing_item_ids")
    unmatched_ids = pairing.get("unmatched_packing_item_ids")
    if (
        not isinstance(baseline_unmatched_ids, list)
        or not isinstance(unmatched_ids, list)
        or not all(isinstance(item_id, str) for item_id in baseline_unmatched_ids)
        or not all(isinstance(item_id, str) for item_id in unmatched_ids)
    ):
        raise DeterministicEvaluationError(
            "negative-control pairing output has invalid unmatched item IDs"
        )
    passed = (
        variant_findings == baseline_findings
        and set(unmatched_ids) - set(baseline_unmatched_ids) == {expected_item_id}
        and set(baseline_unmatched_ids).issubset(unmatched_ids)
        and len(unmatched_ids) == len(baseline_unmatched_ids) + 1
        and pairing.get("unmatched_packing_item_status") == "ambiguous_not_scored"
        and pairing.get("line_item_pairing_status") == "deterministic_with_unscored_packing_extras"
        and pairing_requires_review == baseline_pairing_requires_review
        and variant_review == baseline_review
    )
    added_count = sum(added_findings.values())
    removed_count = sum(removed_findings.values())
    return {
        "passed": passed,
        "case_count": 1,
        "automatic_violation_count": added_count,
        "finding_change_count": added_count + removed_count,
        "results": [
            {
                "variant_id": record["variant_id"],
                "operator_id": NEGATIVE_CONTROL_OPERATOR_ID,
                "finding_added_count": added_count,
                "finding_removed_count": removed_count,
                "finding_change_count": added_count + removed_count,
                "unmatched_packing_item_count_before": len(baseline_unmatched_ids),
                "unmatched_packing_item_count_after": len(unmatched_ids),
                "unmatched_packing_item_status": pairing.get("unmatched_packing_item_status"),
                "line_item_pairing_status": pairing.get("line_item_pairing_status"),
                "requires_human_review_from_extra_item": pairing_requires_review,
                "pass": passed,
            }
        ],
    }


def evaluate(
    defects_path: Path,
    *,
    negative_controls_path: Path,
    manifest_path: Path,
    selection_descriptor_path: Path,
    dataset_lock_path: Path,
) -> tuple[int, dict[str, Any]]:
    records = _jsonl(defects_path)
    defects_sha256 = _sha256(defects_path, label="controlled defect input")
    if defects_sha256 != CANONICAL_CONTROLLED_DEFECTS_SHA256:
        raise DeterministicEvaluationError(
            "controlled defect input hash is not the canonical diagnostic hash"
        )
    defects_byte_size = _byte_size(defects_path, label="controlled defect input")
    negative_records = _jsonl(negative_controls_path)
    negative_controls_sha256 = _sha256(
        negative_controls_path,
        label="negative-control input",
    )
    if negative_controls_sha256 != CANONICAL_NEGATIVE_CONTROLS_SHA256:
        raise DeterministicEvaluationError(
            "negative-control input hash is not the canonical diagnostic hash"
        )
    negative_controls_byte_size = _byte_size(
        negative_controls_path,
        label="negative-control input",
    )
    dataset_lock_sha256 = _validate_dataset_lock(dataset_lock_path)
    selection_descriptor_sha256 = _validate_selection_descriptor(selection_descriptor_path)
    (
        manifest_sha256,
        controlled_assignments_sha256,
        controlled_case_ids,
    ) = _validate_evaluation_manifest(
        manifest_path,
        defects_sha256=defects_sha256,
        defects_byte_size=defects_byte_size,
        negative_controls_sha256=negative_controls_sha256,
        negative_controls_byte_size=negative_controls_byte_size,
        dataset_lock_sha256=dataset_lock_sha256,
        selection_descriptor_sha256=selection_descriptor_sha256,
        records=records,
    )
    metrics = evaluate_controlled_records(
        records,
        expected_case_ids=controlled_case_ids,
    )
    negative_metrics = evaluate_negative_control_records(negative_records)
    passed = bool(metrics["passed"]) and bool(negative_metrics["passed"])
    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": "RELATION_AWARE_RULE_COVERAGE_DEVELOPMENT",
        "status": "PASS" if passed else "FAIL",
        "input_sha256": defects_sha256,
        "negative_control_input_sha256": negative_controls_sha256,
        "evaluation_manifest_sha256": manifest_sha256,
        "dataset_lock_sha256": dataset_lock_sha256,
        "selection_descriptor_sha256": selection_descriptor_sha256,
        "evaluation_version": EVALUATION_VERSION,
        "release_eligible": False,
        "release_status": RELEASE_STATUS,
        "release_split_membership_consulted": False,
        "controlled_assignments_sha256": controlled_assignments_sha256,
        "operator_ids": sorted(OPERATOR_EXPECTED_FINDINGS),
        "finding_grain": [
            "issue_code",
            "source_document_id",
            "target_document_id",
        ],
        "positive_operator_count": metrics["case_count"],
        "negative_control_count": negative_metrics["case_count"],
        "total_variant_count": metrics["case_count"] + negative_metrics["case_count"],
        "true_positive": metrics["true_positive"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "cases_requiring_human_review": metrics["cases_requiring_human_review"],
        "positive_results": metrics["results"],
        "negative_control_results": negative_metrics["results"],
        "automatic_violation_count_for_negative_controls": negative_metrics[
            "automatic_violation_count"
        ],
        "finding_change_count_for_negative_controls": negative_metrics["finding_change_count"],
        "redaction": {
            "source_values": "OMITTED",
            "absolute_paths": "OMITTED",
            "api_keys": "OMITTED",
        },
    }
    return (0 if passed else 1), report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score relation-aware rules and packing-extra negative controls."
    )
    parser.add_argument("--defects", required=True, type=Path)
    parser.add_argument("--negative-controls", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection-descriptor", required=True, type=Path)
    parser.add_argument("--dataset-lock", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_dataset_provenance(args.dataset_lock)
        exit_code, report = evaluate(
            args.defects,
            negative_controls_path=args.negative_controls,
            manifest_path=args.manifest,
            selection_descriptor_path=args.selection_descriptor,
            dataset_lock_path=args.dataset_lock,
        )
    except DatasetProvenanceError as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA,
                    "scope": "RELATION_AWARE_RULE_COVERAGE_DEVELOPMENT",
                    "status": "FAIL",
                    "release_eligible": False,
                    "error": exc.code,
                },
                sort_keys=True,
            )
        )
        return 2
    except DeterministicEvaluationError as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA,
                    "scope": "RELATION_AWARE_RULE_COVERAGE_DEVELOPMENT",
                    "status": "FAIL",
                    "evaluation_version": EVALUATION_VERSION,
                    "release_eligible": False,
                    "release_status": RELEASE_STATUS,
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
