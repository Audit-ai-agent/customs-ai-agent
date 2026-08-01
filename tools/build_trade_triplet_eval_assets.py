"""Build separated extraction/validation gold and controlled synthetic variants.

All value-bearing outputs are written to the caller-provided artifact directory.
The repository contains only schemas and value-free design/lock descriptors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from evals.release_holdout_protocol import RELEASE_INVALIDATION_CODE
from tools.dataset_provenance import DatasetProvenanceError, verify_dataset_provenance


DATASET_ID = "trade_document_triplets"
DATASET_VERSION = "source-v1"
EVAL_VERSION = "evaluation-v2"
EXTRACTION_SCHEMA = "trade-document-extraction-gold/v1"
VALIDATION_CASE_SCHEMA = "trade-document-validation-gold-case/v1"
ADJUDICATION_SCHEMA = "trade-document-adjudication/v1"
ADJUDICATION_EVENT_SCHEMA = "trade-document-adjudication-event/v1"
DEFECT_SCHEMA = "trade-document-controlled-defect/v2"
CO_SCHEMA = "trade-document-certificate-of-origin-fixture/v1"
MANIFEST_SCHEMA = "trade-document-evaluation-assets/v3"
CONTROLLED_DEFECT_RELATIVE_PATH = "injected-defects/cases.jsonl"
ADJUDICATION_EVENT_RELATIVE_PATH = "validation-gold/adjudication-events.jsonl"
EVALUATION_SPLIT_SCHEMA = "trade-document-evaluation-splits/v2"
EVALUATION_SPLIT_SEED = "trade-document-triplets-evaluation-v2-holdout-2026-07-18"
EVALUATION_REBALANCE_SEED = "trade-document-triplets-evaluation-v2-rebalance-2026-07-18"
CONTROLLED_CASE_SEED = "trade-document-triplets-evaluation-v2-controlled-development"
ADJUDICATION_STATES = frozenset(
    {
        "confirmed_normal",
        "confirmed_intentional_defect",
        "confirmed_generator_error",
        "ambiguous",
        "coverage_gap",
        "not_applicable",
    }
)
DEFAULT_SCHEMA_ROOT = (
    Path(__file__).resolve().parents[1] / "evals" / "trade_document_triplets" / "v1" / "schemas"
)
DEFAULT_CO_DESIGN = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "certificate_of_origin.evaluation-v2.design.json"
)
DEFAULT_WORKFLOW_DESCRIPTOR = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "workflow-boundary.descriptor.json"
)
DEFAULT_DATASET_LOCK = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "dataset.lock.json"
)
_ISO_6346_CONTAINER_PATTERN = re.compile(r"^[A-Z]{3}[UJZ]\d{7}$")
_ISO_6346_LETTER_VALUES = {
    letter: value
    for letter, value in zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        (
            10,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
        ),
        strict=True,
    )
}


class EvalAssetError(ValueError):
    """Raised when evaluation assets cannot be derived reproducibly."""


def _iso_6346_check_digit(first_ten: str) -> int:
    if re.fullmatch(r"[A-Z]{3}[UJZ]\d{6}", first_ten) is None:
        raise EvalAssetError("container base is not a valid ISO 6346 lexical form")
    total = 0
    for index, character in enumerate(first_ten):
        value = _ISO_6346_LETTER_VALUES[character] if character.isalpha() else int(character)
        total += value * (2**index)
    return (total % 11) % 10


def _valid_container_number(value: Any) -> str:
    normalized = re.sub(r"[\s-]+", "", str(value or "")).upper()
    if _ISO_6346_CONTAINER_PATTERN.fullmatch(normalized) is None:
        raise EvalAssetError("controlled container source is not a valid ISO 6346 identifier")
    first_ten = normalized[:10]
    return f"{first_ten}{_iso_6346_check_digit(first_ten)}"


def _different_valid_container_number(value: Any) -> str:
    canonical = _valid_container_number(value)
    serial = (int(canonical[4:10]) + 1) % 1_000_000
    first_ten = f"{canonical[:4]}{serial:06d}"
    replacement = f"{first_ten}{_iso_6346_check_digit(first_ten)}"
    if replacement == canonical:
        raise EvalAssetError("controlled container mutation did not change the identifier")
    return replacement


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return ("\n".join(_canonical_bytes(record).decode() for record in records) + "\n").encode()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalAssetError(f"cannot read JSON: {path}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalAssetError(f"cannot read JSONL: {path}") from exc
    if not all(isinstance(record, dict) for record in records):
        raise EvalAssetError(f"JSONL records must be objects: {path}")
    return records


def _write_generated(path: Path, content: bytes, *, refresh: bool) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise EvalAssetError(f"output is not a regular file: {path}")
        if path.read_bytes() == content:
            return
        if not refresh:
            raise EvalAssetError(f"generated output differs: {path}; use --refresh after review")
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


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _flatten_strings(value: Any, pointer: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield pointer, value
        return
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten_strings(value[key], f"{pointer}/{_escape_pointer(key)}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten_strings(item, f"{pointer}/{index}")
        return
    raise EvalAssetError(f"source record has a non-string leaf at {pointer or '/'}")


def _source_case_sha256(case: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(case))


def build_extraction_records(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create source-exact field gold without validation labels."""

    result: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        documents = []
        for document_type in (
            "commercial_invoice",
            "packing_list",
            "bill_of_lading",
        ):
            source = case["documents"][document_type]
            raw_record = source["raw_record"]
            fields = [
                {
                    "field_id": f"{document_type}:{pointer}",
                    "json_pointer": pointer,
                    "raw_value": raw_value,
                }
                for pointer, raw_value in _flatten_strings(raw_record)
            ]
            documents.append(
                {
                    "document_id": f"{case_id}:{document_type}",
                    "document_type": document_type,
                    "source_relative_path": source["source_relative_path"],
                    "source_file_sha256": source["source_file_sha256"],
                    "source_record_index_zero_based": source["source_record_index_zero_based"],
                    "source_record_sha256": _sha256_bytes(_canonical_bytes(raw_record)),
                    "fields": fields,
                }
            )
        result.append(
            {
                "schema_version": EXTRACTION_SCHEMA,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "case_id": case_id,
                "source_case_sha256": _source_case_sha256(case),
                "gold_scope": "source_exact_raw_values_only",
                "documents": documents,
            }
        )
    return result


def build_adjudications(
    anomalies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert observations to six-state records without inventing intent."""

    result = []
    for observation in anomalies:
        state = observation["status"]
        if state not in {"ambiguous", "coverage_gap"}:
            raise EvalAssetError(f"unadjudicated source observation has unsupported state: {state}")
        record = {
            "schema_version": ADJUDICATION_SCHEMA,
            "adjudication_id": observation["anomaly_id"],
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "scope": observation["scope"],
            "rule_id": observation["rule_id"],
            "state": state,
            "scoring_eligible": False,
            "intent_status": "unknown",
            "observed_condition": observation["observed_condition"],
            "expected_agent_behavior": observation["expected_agent_behavior"],
            "expected_outcome": (
                "needs_review" if observation["scope"] == "case" else "not_scored"
            ),
            "expected_findings": [],
            "evidence": observation["evidence"],
            "possible_interpretations": observation["possible_interpretations"],
            "adjudication_evidence": [],
        }
        if observation["scope"] == "case":
            record["case_id"] = observation["case_id"]
        result.append(record)
    return sorted(result, key=lambda item: item["adjudication_id"])


def build_validation_case_index(
    cases: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in adjudications:
        if record["scope"] == "case":
            by_case.setdefault(record["case_id"], []).append(record)

    result = []
    for case in cases:
        case_id = case["case_id"]
        records = sorted(
            by_case.get(case_id, []),
            key=lambda item: item["adjudication_id"],
        )
        counts = Counter(record["state"] for record in records)
        result.append(
            {
                "schema_version": VALIDATION_CASE_SCHEMA,
                "dataset_id": DATASET_ID,
                "dataset_version": DATASET_VERSION,
                "case_id": case_id,
                "source_case_sha256": _source_case_sha256(case),
                "adjudication_ids": [record["adjudication_id"] for record in records],
                "state_counts": dict(sorted(counts.items())),
                "scored_observation_count": 0,
                "expected_case_outcome": "needs_review",
            }
        )
    return result


def validate_and_fold_adjudication_events(
    adjudications: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate one append-only global chain and fold per-anomaly state.

    The event stream never rewrites the source anomaly or its original
    adjudication.  Each event binds the dataset, anomaly, previous folded
    state, and the canonical hash of the preceding global event.
    """

    by_anomaly: dict[str, dict[str, Any]] = {}
    for adjudication in adjudications:
        anomaly_id = adjudication.get("adjudication_id")
        if not isinstance(anomaly_id, str) or not anomaly_id or anomaly_id in by_anomaly:
            raise EvalAssetError("base adjudication identities must be unique and non-empty")
        if (
            adjudication.get("dataset_id") != DATASET_ID
            or adjudication.get("dataset_version") != DATASET_VERSION
            or adjudication.get("state") not in ADJUDICATION_STATES
        ):
            raise EvalAssetError("base adjudication dataset or state is invalid")
        by_anomaly[anomaly_id] = adjudication

    folded_states = {
        anomaly_id: str(adjudication["state"]) for anomaly_id, adjudication in by_anomaly.items()
    }
    folded_scoring_eligibility = {
        anomaly_id: bool(adjudication.get("scoring_eligible"))
        for anomaly_id, adjudication in by_anomaly.items()
    }
    event_counts = {anomaly_id: 0 for anomaly_id in by_anomaly}
    seen_event_ids: set[str] = set()
    previous_hash: str | None = None
    previous_time: datetime | None = None
    event_hashes: list[str] = []
    required_keys = {
        "schema_version",
        "event_id",
        "dataset_id",
        "dataset_version",
        "anomaly_id",
        "previous_state",
        "new_state",
        "evidence",
        "adjudicator_id",
        "adjudicated_at_utc",
        "explanation",
        "previous_event_sha256",
    }
    event_schema = _load_json(DEFAULT_SCHEMA_ROOT / "adjudication-event.schema.json")
    event_validator = Draft202012Validator(
        event_schema,
        format_checker=FormatChecker(),
    )
    for event in events:
        try:
            event_validator.validate(event)
        except ValidationError as exc:
            raise EvalAssetError("adjudication event failed its pinned JSON Schema") from exc
        if set(event) != required_keys:
            raise EvalAssetError("adjudication event keys do not match the append-only contract")
        event_id = event.get("event_id")
        anomaly_id = event.get("anomaly_id")
        if (
            event.get("schema_version") != ADJUDICATION_EVENT_SCHEMA
            or not isinstance(event_id, str)
            or not event_id
            or event_id in seen_event_ids
        ):
            raise EvalAssetError("adjudication event identity is invalid")
        if event.get("dataset_id") != DATASET_ID or event.get("dataset_version") != DATASET_VERSION:
            raise EvalAssetError("adjudication event dataset chain is invalid")
        if not isinstance(anomaly_id, str) or anomaly_id not in folded_states:
            raise EvalAssetError("adjudication event references an unknown anomaly")
        if event.get("previous_state") != folded_states[anomaly_id]:
            raise EvalAssetError("adjudication event previous_state breaks the anomaly chain")
        if event.get("new_state") not in ADJUDICATION_STATES:
            raise EvalAssetError("adjudication event new_state is invalid")
        if event.get("previous_event_sha256") != previous_hash:
            raise EvalAssetError("adjudication event previous_event_sha256 breaks the global chain")
        evidence = event.get("evidence")
        if (
            not isinstance(evidence, dict)
            or set(evidence) not in ({"kind", "reference"}, {"kind", "reference", "sha256"})
            or not isinstance(evidence.get("kind"), str)
            or not isinstance(evidence.get("reference"), str)
            or not evidence["reference"]
        ):
            raise EvalAssetError("adjudication event evidence is invalid")
        evidence_kind = evidence["kind"]
        if event["new_state"] == "confirmed_intentional_defect" and evidence_kind not in {
            "dataset_author_confirmation",
            "controlled_generator_manifest",
            "two_reviewer_consensus",
            "business_rule_authority",
        }:
            raise EvalAssetError("intentional-defect transition lacks scoring-safe authority")
        if event["new_state"] == "confirmed_normal" and evidence_kind not in {
            "dataset_author_confirmation",
            "two_reviewer_consensus",
            "source_correction_record",
            "business_rule_authority",
        }:
            raise EvalAssetError("normal transition lacks scoring-safe authority")
        timestamp = event.get("adjudicated_at_utc")
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise EvalAssetError("adjudication event timestamp must be explicit UTC")
        try:
            parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EvalAssetError("adjudication event timestamp is invalid") from exc
        if previous_time is not None and parsed_time < previous_time:
            raise EvalAssetError("adjudication event timestamps must be append ordered")

        event_hash = _sha256_bytes(_canonical_bytes(event))
        seen_event_ids.add(event_id)
        previous_state = folded_states[anomaly_id]
        folded_states[anomaly_id] = str(event["new_state"])
        if (
            event["new_state"]
            not in {
                "confirmed_normal",
                "confirmed_intentional_defect",
            }
            or event["new_state"] != previous_state
        ):
            # Event-only state changes never manufacture score eligibility.
            # Promotion requires a separately validated adjudication record
            # containing expected findings/outcome evidence.
            folded_scoring_eligibility[anomaly_id] = False
        event_counts[anomaly_id] += 1
        event_hashes.append(event_hash)
        previous_hash = event_hash
        previous_time = parsed_time

    return {
        "event_count": len(events),
        "chain_head_sha256": previous_hash,
        "event_hashes_sha256": _sha256_bytes(_canonical_bytes(event_hashes)),
        "initial_states_sha256": _sha256_bytes(
            _canonical_bytes(
                {
                    anomaly_id: adjudication["state"]
                    for anomaly_id, adjudication in sorted(by_anomaly.items())
                }
            )
        ),
        "folded_states_sha256": _sha256_bytes(
            _canonical_bytes(dict(sorted(folded_states.items())))
        ),
        "event_counts_sha256": _sha256_bytes(_canonical_bytes(dict(sorted(event_counts.items())))),
        "folded_scoring_eligibility_sha256": _sha256_bytes(
            _canonical_bytes(dict(sorted(folded_scoring_eligibility.items())))
        ),
        "scoring_eligible_count": sum(folded_scoring_eligibility.values()),
        "folded_states": dict(sorted(folded_states.items())),
        "folded_scoring_eligibility": dict(sorted(folded_scoring_eligibility.items())),
    }


def _replace_invoice_number(value: str) -> str:
    number, separator, remainder = value.partition(" / ")
    if not separator:
        raise EvalAssetError(f"invoice number/date format is unsupported: {value!r}")
    return f"{number}-TEST-MISMATCH{separator}{remainder}"


def _replace_invoice_date(value: str) -> str:
    number, separator, _remainder = value.partition(" / ")
    if not separator:
        raise EvalAssetError(f"invoice number/date format is unsupported: {value!r}")
    return f"{number}{separator}2099-12-31"


def _replace_vessel_name(value: str) -> str:
    vessel, separator, voyage = value.rpartition(" / ")
    if not separator:
        return f"{value} [CONTROLLED MISMATCH]"
    return f"{vessel} [CONTROLLED MISMATCH]{separator}{voyage}"


def _replace_package_count(value: str) -> str:
    match = re.search(r"\d[\d,]*", value)
    if match is None:
        raise EvalAssetError(f"package fixture field cannot be parsed: {value!r}")
    count = int(match.group(0).replace(",", "")) + 1
    return f"{value[: match.start()]}{count:,}{value[match.end() :]}"


def _add_decimal(value: str, increment: Decimal) -> str:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
    if match is None:
        raise EvalAssetError(f"numeric fixture field cannot be parsed: {value!r}")
    number = Decimal(match.group(0).replace(",", "")) + increment
    replacement = f"{number:,.2f}"
    return f"{value[: match.start()]}{replacement}{value[match.end() :]}"


def _path_parent(root: Any, path: list[str | int]) -> tuple[Any, str | int]:
    if not path:
        raise EvalAssetError("controlled mutation path must not be empty")
    current = root
    for part in path[:-1]:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError) as exc:
            raise EvalAssetError("controlled mutation path does not exist") from exc
    return current, path[-1]


def _path_value(root: Any, path: list[str | int]) -> Any:
    parent, leaf = _path_parent(root, path)
    try:
        return parent[leaf]
    except (KeyError, IndexError, TypeError) as exc:
        raise EvalAssetError("controlled mutation target does not exist") from exc


def _set_path(root: Any, path: list[str | int], value: Any) -> Any:
    parent, leaf = _path_parent(root, path)
    try:
        before = parent[leaf]
        parent[leaf] = value
    except (KeyError, IndexError, TypeError) as exc:
        raise EvalAssetError("controlled mutation target does not exist") from exc
    if before == value:
        raise EvalAssetError("controlled mutation did not change its target")
    return before


def _add_path(root: Any, path: list[str | int], value: Any) -> None:
    parent, leaf = _path_parent(root, path)
    if isinstance(parent, dict):
        if leaf in parent:
            raise EvalAssetError("controlled parent setup may only add absent fields")
        parent[leaf] = value
        return
    raise EvalAssetError("controlled parent setup supports object additions only")


BUSINESS_FIELD_OPERATOR_IDS = (
    "pl_invoice_number_mismatch",
    "pl_invoice_date_mismatch",
    "pl_seller_mismatch",
    "bl_consignee_mismatch",
    "pl_vessel_mismatch",
    "bl_voyage_mismatch",
    "bl_port_of_loading_mismatch",
    "bl_port_of_discharge_mismatch",
    "pl_package_mismatch",
    "bl_gross_weight_mismatch",
    "bl_measurement_mismatch",
    "ci_amount_mismatch",
    "pl_container_mismatch",
    "pl_seal_mismatch",
    "bl_freight_both_present",
    "bl_freight_neither_present",
    "co_origin_mismatch",
    "co_exporter_mismatch",
    "co_importer_mismatch",
    "co_hs_code_mismatch",
    "pl_consignee_mismatch",
    "pl_departure_date_mismatch",
    "ci_port_of_loading_mismatch",
    "ci_port_of_discharge_mismatch",
    "pl_country_of_origin_mismatch",
    "pl_shipping_marks_mismatch",
    "pl_line_quantity_mismatch",
    "pl_line_net_gross_invalid",
    "pl_line_item_unmatched",
)
CONTROLLED_CASE_COUNT = len(BUSINESS_FIELD_OPERATOR_IDS)
WORKFLOW_BOUNDARY_OPERATOR_IDS = (
    "required_document_missing",
    "duplicate_document_type",
    "ocr_conflict_unresolved",
    "evidence_partial",
    "evidence_failed",
    "evidence_ambiguous",
)


def _base_operator(
    operator_id: str,
) -> tuple[str, list[str | int], Callable[[str], str], list[str]] | None:
    operators: dict[
        str,
        tuple[str, list[str | int], Callable[[str], str], list[str]],
    ] = {
        "pl_invoice_number_mismatch": (
            "packing_list",
            ["Invoice No. and Date"],
            _replace_invoice_number,
            ["ci_pl.invoice_no_and_date"],
        ),
        "pl_invoice_date_mismatch": (
            "packing_list",
            ["Invoice No. and Date"],
            _replace_invoice_date,
            ["ci_pl.invoice_date"],
        ),
        "pl_seller_mismatch": (
            "packing_list",
            ["Seller"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["ci_pl.seller", "pl_bl.shipper"],
        ),
        "bl_consignee_mismatch": (
            "bill_of_lading",
            ["Consignee"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["pl_bl.consignee", "ci_bl.consignee"],
        ),
        "pl_vessel_mismatch": (
            "packing_list",
            ["Vessel / Flight"],
            _replace_vessel_name,
            ["ci_pl.vessel", "pl_bl.vessel"],
        ),
        "bl_voyage_mismatch": (
            "bill_of_lading",
            ["Voyage No."],
            lambda value: f"{value}-CONTROLLED-MISMATCH",
            ["pl_bl.voyage"],
        ),
        "bl_port_of_loading_mismatch": (
            "bill_of_lading",
            ["Port of Loading"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["pl_bl.port_of_loading"],
        ),
        "bl_port_of_discharge_mismatch": (
            "bill_of_lading",
            ["Port of Discharge"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["pl_bl.port_of_discharge"],
        ),
        "pl_package_mismatch": (
            "packing_list",
            ["No. & Kind of Packages"],
            _replace_package_count,
            ["ci_pl.packages", "pl_bl.packages"],
        ),
        "bl_gross_weight_mismatch": (
            "bill_of_lading",
            ["Gross Weight"],
            lambda value: _add_decimal(value, Decimal("100.00")),
            ["pl_bl.gross_weight"],
        ),
        "bl_measurement_mismatch": (
            "bill_of_lading",
            ["Revenue Tons"],
            lambda value: _add_decimal(value, Decimal("100.00")),
            ["pl_bl.measurement"],
        ),
        "ci_amount_mismatch": (
            "commercial_invoice",
            ["Amount"],
            lambda value: _add_decimal(value, Decimal("100000.00")),
            ["ci.amount_vs_displayed_unit_price"],
        ),
        "pl_consignee_mismatch": (
            "packing_list",
            ["Consignee"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["ci_pl.consignee", "pl_bl.consignee"],
        ),
        "pl_departure_date_mismatch": (
            "packing_list",
            ["Departure Date"],
            lambda value: "2099-12-30" if value.strip() == "2099-12-31" else "2099-12-31",
            ["ci_pl.departure_date"],
        ),
        "ci_port_of_loading_mismatch": (
            "commercial_invoice",
            ["From"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["ci_pl.port_of_loading"],
        ),
        "ci_port_of_discharge_mismatch": (
            "commercial_invoice",
            ["To"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["ci_pl.port_of_discharge"],
        ),
        "pl_country_of_origin_mismatch": (
            "packing_list",
            ["Other References"],
            lambda _value: "Country of Origin: Controlled Mismatch Origin",
            ["ci_pl.country_of_origin"],
        ),
        "pl_shipping_marks_mismatch": (
            "packing_list",
            ["Shipping Marks"],
            lambda value: f"{value} [CONTROLLED MISMATCH]",
            ["ci_pl.shipping_marks"],
        ),
        "pl_line_quantity_mismatch": (
            "packing_list",
            ["Goods Details", 0, "quantity"],
            lambda value: _add_decimal(value, Decimal("1.00")),
            ["ci_pl.line_item_quantity"],
        ),
        "pl_line_item_unmatched": (
            "packing_list",
            ["Goods Details", 0, "item"],
            lambda _value: "CONTROLLED UNRELATED ITEM WITH NO SEMANTIC OVERLAP",
            ["ci_pl.line_item_matching"],
        ),
    }
    return operators.get(operator_id)


def _setup_add(
    controlled_parent: dict[str, Any],
    setup_operations: list[dict[str, Any]],
    path: list[str | int],
    value: Any,
) -> None:
    _add_path(controlled_parent, path, value)
    setup_operations.append({"op": "add", "path": path, "value": copy.deepcopy(value)})


def _setup_replace(
    controlled_parent: dict[str, Any],
    setup_operations: list[dict[str, Any]],
    path: list[str | int],
    value: Any,
) -> None:
    before = _path_value(controlled_parent, path)
    if before == value:
        return
    parent, leaf = _path_parent(controlled_parent, path)
    parent[leaf] = value
    setup_operations.append(
        {
            "op": "replace",
            "path": path,
            "before": copy.deepcopy(before),
            "value": copy.deepcopy(value),
        }
    )


def _controlled_co_record(case: dict[str, Any], hs_code: str) -> dict[str, Any]:
    ci = case["documents"]["commercial_invoice"]["raw_record"]
    origin = _country_of_origin(ci["Other References (Country of Origin)"])
    return {
        "Certificate No.": f"CO-CONTROLLED-{case['case_id']}",
        "Exporter": ci["Shipper / Seller"],
        "Consignee": ci["Consignee (or For account & risk of Messrs)"],
        "Country of Origin": origin,
        "Goods Details": [
            {
                "item": ci["Goods Description"],
                "quantity": ci["Quantity"],
                "hs_code": hs_code,
            }
        ],
    }


def _prepare_operator(
    source_case: dict[str, Any],
    operator_id: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    list[str | int],
    str,
    list[str],
]:
    controlled_parent = copy.deepcopy(source_case)
    setup_operations: list[dict[str, Any]] = []
    base = _base_operator(operator_id)
    if base is not None:
        document_type, field_path, transform, expected_rules = base
        current = _path_value(
            controlled_parent["documents"][document_type]["raw_record"],
            field_path,
        )
        if not isinstance(current, str):
            raise EvalAssetError("controlled mutation source must be a string")
        return (
            controlled_parent,
            setup_operations,
            document_type,
            field_path,
            transform(current),
            expected_rules,
        )

    bl = controlled_parent["documents"]["bill_of_lading"]["raw_record"]
    ci = controlled_parent["documents"]["commercial_invoice"]["raw_record"]
    if operator_id == "pl_container_mismatch":
        field_path = ["Container No."]
        controlled_container = _valid_container_number(bl["Container No."])
        _setup_replace(
            controlled_parent,
            setup_operations,
            ["documents", "bill_of_lading", "raw_record", "Container No."],
            controlled_container,
        )
        _setup_add(
            controlled_parent,
            setup_operations,
            ["documents", "packing_list", "raw_record", *field_path],
            controlled_container,
        )
        return (
            controlled_parent,
            setup_operations,
            "packing_list",
            field_path,
            _different_valid_container_number(controlled_container),
            ["pl_bl.container_numbers"],
        )
    if operator_id == "pl_seal_mismatch":
        field_path = ["Seal No."]
        seal = str(bl["Seal No. / Marks & No."]).split(" // ", 1)[0].strip()
        _setup_add(
            controlled_parent,
            setup_operations,
            ["documents", "packing_list", "raw_record", *field_path],
            seal,
        )
        return (
            controlled_parent,
            setup_operations,
            "packing_list",
            field_path,
            f"{seal}-CONTROLLED-MISMATCH",
            ["pl_bl.seal_numbers"],
        )
    if operator_id == "bl_freight_both_present":
        prepaid = bool(str(bl.get("Prepaid") or "").strip())
        collect = bool(str(bl.get("Collect") or "").strip())
        if prepaid and not collect:
            target_field = "Collect"
        elif collect and not prepaid:
            target_field = "Prepaid"
        else:
            _setup_replace(
                controlled_parent,
                setup_operations,
                ["documents", "bill_of_lading", "raw_record", "Prepaid"],
                "Y",
            )
            _setup_replace(
                controlled_parent,
                setup_operations,
                ["documents", "bill_of_lading", "raw_record", "Collect"],
                "",
            )
            target_field = "Collect"
        return (
            controlled_parent,
            setup_operations,
            "bill_of_lading",
            [target_field],
            "Y",
            ["bl.freight_terms"],
        )
    if operator_id == "bl_freight_neither_present":
        prepaid = bool(str(bl.get("Prepaid") or "").strip())
        collect = bool(str(bl.get("Collect") or "").strip())
        if prepaid and not collect:
            target_field = "Prepaid"
        elif collect and not prepaid:
            target_field = "Collect"
        else:
            _setup_replace(
                controlled_parent,
                setup_operations,
                ["documents", "bill_of_lading", "raw_record", "Prepaid"],
                "Y",
            )
            _setup_replace(
                controlled_parent,
                setup_operations,
                ["documents", "bill_of_lading", "raw_record", "Collect"],
                "",
            )
            target_field = "Prepaid"
        return (
            controlled_parent,
            setup_operations,
            "bill_of_lading",
            [target_field],
            "",
            ["bl.freight_terms"],
        )
    if operator_id == "pl_line_net_gross_invalid":
        field_path = ["Goods Details", 0, "net_weight"]
        gross_weight = str(
            _path_value(
                controlled_parent["documents"]["packing_list"]["raw_record"],
                ["Goods Details", 0, "weight"],
            )
        )
        _setup_add(
            controlled_parent,
            setup_operations,
            ["documents", "packing_list", "raw_record", *field_path],
            gross_weight,
        )
        return (
            controlled_parent,
            setup_operations,
            "packing_list",
            field_path,
            _add_decimal(gross_weight, Decimal("100.00")),
            ["pl.line_item_net_gross_weight"],
        )

    hs_code = "0000000001"
    _setup_add(
        controlled_parent,
        setup_operations,
        ["documents", "commercial_invoice", "raw_record", "HS Code"],
        hs_code,
    )
    co_record = _controlled_co_record(controlled_parent, hs_code)
    co_document = {
        "source_relative_path": "controlled-extension/certificate_of_origin.json",
        "source_file_sha256": _sha256_bytes(_canonical_bytes(co_record)),
        "source_record_index_zero_based": 0,
        "raw_record": co_record,
    }
    _setup_add(
        controlled_parent,
        setup_operations,
        ["documents", "certificate_of_origin"],
        co_document,
    )
    if operator_id == "co_origin_mismatch":
        return (
            controlled_parent,
            setup_operations,
            "certificate_of_origin",
            ["Country of Origin"],
            "Controlled Mismatch Origin",
            ["co_ci.origin_country"],
        )
    if operator_id == "co_exporter_mismatch":
        return (
            controlled_parent,
            setup_operations,
            "certificate_of_origin",
            ["Exporter"],
            f"{ci['Shipper / Seller']} [CONTROLLED MISMATCH]",
            ["co_ci.exporter"],
        )
    if operator_id == "co_importer_mismatch":
        return (
            controlled_parent,
            setup_operations,
            "certificate_of_origin",
            ["Consignee"],
            f"{ci['Consignee (or For account & risk of Messrs)']} [CONTROLLED MISMATCH]",
            ["co_ci.importer"],
        )
    if operator_id == "co_hs_code_mismatch":
        return (
            controlled_parent,
            setup_operations,
            "certificate_of_origin",
            ["Goods Details", 0, "hs_code"],
            "9999999999",
            ["co_ci.hs_code"],
        )
    raise EvalAssetError(f"unsupported controlled operator: {operator_id}")


def inject_controlled_defects(
    cases: list[dict[str, Any]],
    controlled_case_ids: list[str],
    *,
    evaluation_version: str = EVAL_VERSION,
) -> list[dict[str, Any]]:
    """Create one deterministic field mutation per non-holdout development case."""

    case_by_id = {case["case_id"]: case for case in cases}
    if len(controlled_case_ids) != len(BUSINESS_FIELD_OPERATOR_IDS):
        raise EvalAssetError(
            "controlled evaluation requires exactly one case per business-field operator"
        )
    variants: list[dict[str, Any]] = []
    for operator_id, case_id in zip(
        BUSINESS_FIELD_OPERATOR_IDS,
        sorted(controlled_case_ids),
        strict=True,
    ):
        if case_id not in case_by_id:
            raise EvalAssetError(f"controlled development case is missing: {case_id}")
        original_case = case_by_id[case_id]
        source_hash = _source_case_sha256(original_case)
        (
            controlled_parent,
            setup_operations,
            document_type,
            field_path,
            replacement,
            expected_rules,
        ) = _prepare_operator(original_case, operator_id)
        controlled_parent_hash = _source_case_sha256(controlled_parent)
        mutated_case = copy.deepcopy(controlled_parent)
        before = _set_path(
            mutated_case["documents"][document_type]["raw_record"],
            field_path,
            replacement,
        )
        if _source_case_sha256(original_case) != source_hash:
            raise EvalAssetError("controlled mutation changed the source case")

        variant_id = f"{case_id}--{operator_id}"
        adjudication = {
            "schema_version": ADJUDICATION_SCHEMA,
            "adjudication_id": f"{variant_id}:expected",
            "dataset_id": DATASET_ID,
            "dataset_version": evaluation_version,
            "scope": "case",
            "case_id": variant_id,
            "rule_id": expected_rules[0],
            "state": "confirmed_intentional_defect",
            "scoring_eligible": True,
            "intent_status": "confirmed",
            "observed_condition": "one_field_was_changed_by_the_controlled_generator",
            "expected_agent_behavior": "detect_the_controlled_mismatch_and_require_review",
            "expected_outcome": "needs_review",
            "expected_findings": expected_rules,
            "evidence": {
                "operator_id": operator_id,
                "document_type": document_type,
                "field_path": field_path,
                "before": before,
                "after": replacement,
            },
            "possible_interpretations": [
                "controlled intentional defect generated for deterministic evaluation"
            ],
            "adjudication_evidence": [
                {
                    "kind": "controlled_generator_manifest",
                    "reference": "tools/build_trade_triplet_eval_assets.py",
                }
            ],
        }
        variants.append(
            {
                "schema_version": DEFECT_SCHEMA,
                "variant_id": variant_id,
                "source_case_id": case_id,
                "source_case_sha256": source_hash,
                "controlled_parent_sha256": controlled_parent_hash,
                "controlled_parent_setup": setup_operations,
                "mutation_count": 1,
                "mutation": {
                    "operator_id": operator_id,
                    "document_type": document_type,
                    "field_path": field_path,
                    "before": before,
                    "after": replacement,
                },
                "expected_rule_ids": expected_rules,
                "expected_outcome": "needs_review",
                "adjudication": adjudication,
                "mutated_case": mutated_case,
            }
        )
    return variants


def _country_of_origin(value: str) -> str:
    prefix = "Country of Origin:"
    return value[len(prefix) :].strip() if value.startswith(prefix) else value.strip()


def build_certificate_of_origin_fixtures(
    cases: list[dict[str, Any]],
    splits: dict[str, list[str]],
    design: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_selection = design["linked_case_selection"]
    if expected_selection.get("locked_test") != 0:
        raise EvalAssetError("evaluation-v2 C/O fixtures must not use release holdout cases")
    selected: list[tuple[str, str]] = []
    for split_name in ("development", "validation", "locked_test"):
        count = expected_selection[split_name]
        selected.extend((case_id, split_name) for case_id in sorted(splits[split_name])[:count])
    if len(selected) != design["case_count"]:
        raise EvalAssetError("certificate-of-origin design case count does not match selection")

    case_by_id = {case["case_id"]: case for case in cases}
    fixtures = []
    for index, (case_id, split_name) in enumerate(selected, start=1):
        case = case_by_id[case_id]
        ci = case["documents"]["commercial_invoice"]["raw_record"]
        pl = case["documents"]["packing_list"]["raw_record"]
        synthetic_hs = f"000000{index:04d}"
        record = {
            "Certificate No.": f"CO-SYNTHETIC-{index:04d}",
            "Exporter": ci["Shipper / Seller"],
            "Consignee": ci["Consignee (or For account & risk of Messrs)"],
            "Means of Transport and Route": (
                f"{ci['Vessel / Flight']} | {ci['From']} -> {ci['To']}"
            ),
            "Country of Origin": _country_of_origin(ci["Other References (Country of Origin)"]),
            "Invoice No. and Date": ci["Invoice No. and Date"],
            "Goods Details": [
                {
                    "item": ci["Goods Description"],
                    "quantity": ci["Quantity"],
                    "hs_code": synthetic_hs,
                    "origin_criterion": design["origin_criterion"],
                }
            ],
            "Gross Weight": pl["Goods Details"][0]["weight"],
            "Issuing Authority": "SYNTHETIC TEST AUTHORITY - NOT VALID",
            "Issue Date": ci["Departure Date"],
            "Signature": "SYNTHETIC FIXTURE",
            "Document Disclaimer": design["disclaimer"],
        }
        fixtures.append(
            {
                "schema_version": CO_SCHEMA,
                "fixture_id": f"co-synthetic-{index:04d}",
                "linked_source_case_id": case_id,
                "linked_source_case_sha256": _source_case_sha256(case),
                "split": split_name,
                "fixture_scope": "controlled_synthetic_extension_not_source_ground_truth",
                "synthetic_hs_code": True,
                "hs_evidence_query_allowed": False,
                "adjudication_state": "confirmed_normal",
                "record": record,
                "record_sha256": _sha256_bytes(_canonical_bytes(record)),
            }
        )
    return fixtures


def _validate_inputs(
    dataset_root: Path,
    schema_root: Path,
    co_design_path: Path,
    workflow_descriptor_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    str,
    str,
]:
    inventory = _load_json(dataset_root / "inventory.json")
    if inventory.get("schema_version") != "trade-document-triplets-inventory/v1":
        raise EvalAssetError("registered inventory schema does not match")
    if inventory["summary"].get("raw_dataset_contains_business_values") is not True:
        raise EvalAssetError("registered inventory must acknowledge raw business values")
    lock_info = inventory["dataset"].get("lock")
    if not isinstance(lock_info, dict):
        raise EvalAssetError("registered inventory has no pinned lock")
    dataset_lock_sha256 = lock_info.get("sha256")
    if (
        lock_info.get("schema_version") != "trade-document-triplets-dataset-lock/v1"
        or not isinstance(dataset_lock_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset_lock_sha256) is None
    ):
        raise EvalAssetError("registered inventory dataset lock is invalid")

    checksums: dict[str, str] = {}
    for line in (dataset_root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, separator, relative_path = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvalAssetError("registered raw checksum file is invalid")
        target = dataset_root / relative_path
        if _sha256_file(target) != digest:
            raise EvalAssetError(f"registered raw hash mismatch: {relative_path}")
        checksums[relative_path] = digest

    cases = _load_jsonl(dataset_root / "derived" / "cases.jsonl")
    anomalies = _load_jsonl(dataset_root / "anomaly-registry.jsonl")
    legacy_split_path = dataset_root / "derived" / "splits.json"
    legacy_split_document = _load_json(legacy_split_path)
    evaluation_split_path = dataset_root / "derived" / "evaluation-splits" / "v2" / "splits.json"
    split_document = _load_json(evaluation_split_path)
    if (
        split_document.get("schema_version") != EVALUATION_SPLIT_SCHEMA
        or split_document.get("dataset_id") != DATASET_ID
        or split_document.get("dataset_version") != DATASET_VERSION
        or split_document.get("evaluation_version") != EVAL_VERSION
        or split_document.get("counts") != {"development": 60, "validation": 20, "locked_test": 20}
    ):
        raise EvalAssetError("evaluation-v2 split contract is invalid")
    strategy = split_document.get("strategy")
    if (
        not isinstance(strategy, dict)
        or strategy.get("holdout_seed") != EVALUATION_SPLIT_SEED
        or strategy.get("rebalance_seed") != EVALUATION_REBALANCE_SEED
        or strategy.get("replacement_pool") != "legacy_development_plus_validation_only"
    ):
        raise EvalAssetError("evaluation-v2 split strategy is invalid")
    policy = split_document.get("holdout_policy")
    if policy != {
        "release_eligible": False,
        "release_status": "release_holdout_v2_publicly_derivable",
        "allowed_use": "development_diagnostic_only",
        "controlled_mutation_generation_allowed_before_release": False,
        "model_or_rule_tuning_allowed": False,
        "replacement_protocol": "sealed-release-holdout-bundle-manifest/v1",
    }:
        raise EvalAssetError("evaluation-v2 holdout policy is invalid")
    legacy = split_document.get("legacy")
    if (
        not isinstance(legacy, dict)
        or legacy.get("document_sha256") != _sha256_file(legacy_split_path)
        or legacy.get("holdout_status") != "holdout_compromised_by_precommit_evaluation"
        or "EVAL-0003" not in (legacy.get("invalidated_evaluation_ids") or [])
        or legacy.get("former_locked_use") != "development_or_validation_only"
        or legacy.get("splits") != legacy_split_document.get("splits")
    ):
        raise EvalAssetError("evaluation-v2 compromised holdout history is invalid")
    splits = split_document["splits"]
    all_split_ids = [case_id for split_case_ids in splits.values() for case_id in split_case_ids]
    case_ids = [case["case_id"] for case in cases]
    if len(cases) != 100 or len(set(case_ids)) != 100:
        raise EvalAssetError("registered case set must contain 100 unique cases")
    if len(all_split_ids) != 100 or set(all_split_ids) != set(case_ids):
        raise EvalAssetError("registered splits do not partition all cases exactly once")
    if split_document.get("assignments_sha256") != _sha256_bytes(_canonical_bytes(splits)):
        raise EvalAssetError("evaluation-v2 split assignment hash is invalid")
    legacy_splits = legacy["splits"]
    old_locked = set(legacy_splits["locked_test"])
    new_locked = set(splits["locked_test"])
    if old_locked & new_locked:
        raise EvalAssetError("replacement holdout contains compromised cases")
    prior_non_holdout = [
        *legacy_splits["development"],
        *legacy_splits["validation"],
    ]
    expected_new_locked = set(
        sorted(
            prior_non_holdout,
            key=lambda case_id: (
                hashlib.sha256(f"{EVALUATION_SPLIT_SEED}:{case_id}".encode()).hexdigest(),
                case_id,
            ),
        )[:20]
    )
    if new_locked != expected_new_locked:
        raise EvalAssetError("replacement holdout does not match its deterministic seed")
    if not old_locked <= set(splits["development"]) | set(splits["validation"]):
        raise EvalAssetError("compromised holdout was not fully demoted")

    required_schemas = {
        "adjudication.schema.json",
        "adjudication-event.schema.json",
        "evaluation-splits.schema.json",
        "extraction-gold.schema.json",
        "release-holdout-authorization.schema.json",
        "validation-gold-case.schema.json",
        "workflow-boundary-evaluation.schema.json",
    }
    if {path.name for path in schema_root.iterdir() if path.is_file()} != required_schemas:
        raise EvalAssetError("evaluation schema directory has an unexpected file set")
    schema_hashes = {path.name: _sha256_file(path) for path in sorted(schema_root.iterdir())}
    design = _load_json(co_design_path)
    if design.get("design_id") != "trade-document-certificate-of-origin-fixture/evaluation-v2":
        raise EvalAssetError("certificate-of-origin fixture design does not match")
    workflow_descriptor = _load_json(workflow_descriptor_path)
    if (
        workflow_descriptor.get("schema_version")
        != "trade-document-workflow-boundary-descriptor/v1"
        or workflow_descriptor.get("evaluation_version") != EVAL_VERSION
        or workflow_descriptor.get("dataset_id") != DATASET_ID
        or workflow_descriptor.get("dataset_version") != DATASET_VERSION
    ):
        raise EvalAssetError("workflow boundary descriptor does not match evaluation-v2")
    return (
        cases,
        anomalies,
        splits,
        design,
        workflow_descriptor,
        schema_hashes,
        dataset_lock_sha256,
        _sha256_file(evaluation_split_path),
    )


def build_evaluation_assets(
    dataset_root: Path,
    output_root: Path,
    *,
    schema_root: Path = DEFAULT_SCHEMA_ROOT,
    co_design_path: Path = DEFAULT_CO_DESIGN,
    workflow_descriptor_path: Path = DEFAULT_WORKFLOW_DESCRIPTOR,
    refresh: bool = False,
) -> dict[str, Any]:
    """Block regeneration of the compromised evaluation-v2 asset bundle.

    Existing hash-locked development artifacts remain usable as non-release
    diagnostics.  Regenerating them through this former split-aware builder
    would parse and re-derive repository-visible holdout membership.
    """

    raise EvalAssetError(RELEASE_INVALIDATION_CODE)

    source = dataset_root.resolve()
    output = output_root.resolve()
    raw_root = (source / "raw").resolve()
    if output == raw_root or output.is_relative_to(raw_root):
        raise EvalAssetError("evaluation outputs must never be written inside registered raw")
    (
        cases,
        anomalies,
        splits,
        design,
        workflow_descriptor,
        schema_hashes,
        dataset_lock_sha256,
        evaluation_split_sha256,
    ) = _validate_inputs(
        source,
        schema_root.resolve(),
        co_design_path.resolve(),
        workflow_descriptor_path.resolve(),
    )
    release_holdout_ids = set(splits["locked_test"])
    non_holdout_cases = [case for case in cases if case["case_id"] not in release_holdout_ids]
    non_holdout_anomalies = [
        anomaly for anomaly in anomalies if anomaly.get("case_id") not in release_holdout_ids
    ]
    extraction_records = build_extraction_records(non_holdout_cases)
    adjudications = build_adjudications(non_holdout_anomalies)
    validation_cases = build_validation_case_index(non_holdout_cases, adjudications)
    controlled_case_ids = sorted(
        splits["development"],
        key=lambda case_id: (
            hashlib.sha256(f"{CONTROLLED_CASE_SEED}:{case_id}".encode()).hexdigest(),
            case_id,
        ),
    )[:CONTROLLED_CASE_COUNT]
    if set(controlled_case_ids) & release_holdout_ids:
        raise EvalAssetError("controlled defects attempted to use release holdout cases")
    defects = inject_controlled_defects(cases, controlled_case_ids)
    certificates = build_certificate_of_origin_fixtures(cases, splits, design)
    workflow_scenarios = workflow_descriptor.get("scenarios")
    if not isinstance(workflow_scenarios, list):
        raise EvalAssetError("workflow boundary descriptor scenarios are invalid")
    workflow_operator_ids = [
        scenario.get("operator_id") if isinstance(scenario, dict) else None
        for scenario in workflow_scenarios
    ]
    if workflow_operator_ids != list(WORKFLOW_BOUNDARY_OPERATOR_IDS):
        raise EvalAssetError("workflow boundary descriptor does not cover the required operators")

    adjudication_events: list[dict[str, Any]] = []
    adjudication_event_content = b""
    adjudication_chain = validate_and_fold_adjudication_events(
        adjudications,
        adjudication_events,
    )

    controlled_defect_content = _jsonl_bytes(defects)
    content_by_relative_path = {
        "extraction-gold/cases.jsonl": _jsonl_bytes(extraction_records),
        "validation-gold/cases.jsonl": _jsonl_bytes(validation_cases),
        "validation-gold/adjudications.jsonl": _jsonl_bytes(adjudications),
        ADJUDICATION_EVENT_RELATIVE_PATH: adjudication_event_content,
        CONTROLLED_DEFECT_RELATIVE_PATH: controlled_defect_content,
        "certificate-of-origin/cases.jsonl": _jsonl_bytes(certificates),
    }
    controlled_assignments = [
        {
            "source_case_id": item["source_case_id"],
            "operator_id": item["mutation"]["operator_id"],
            "variant_id": item["variant_id"],
        }
        for item in defects
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "evaluation_version": EVAL_VERSION,
        "source": {
            "dataset_lock_sha256": dataset_lock_sha256,
            "registered_cases_sha256": _sha256_file(source / "derived" / "cases.jsonl"),
            "anomaly_registry_sha256": _sha256_file(source / "anomaly-registry.jsonl"),
            "legacy_split_sha256": _sha256_file(source / "derived" / "splits.json"),
            "evaluation_split_sha256": evaluation_split_sha256,
        },
        "controlled_defects": {
            "relative_path": CONTROLLED_DEFECT_RELATIVE_PATH,
            "sha256": _sha256_bytes(controlled_defect_content),
            "byte_size": len(controlled_defect_content),
            "generation_split": "development",
            "release_holdout_used": False,
            "selection_seed": CONTROLLED_CASE_SEED,
            "case_count": len(defects),
            "case_ids": [item["source_case_id"] for item in defects],
            "operator_ids": sorted({item["mutation"]["operator_id"] for item in defects}),
            "case_operator_assignments_sha256": _sha256_bytes(
                _canonical_bytes(controlled_assignments)
            ),
        },
        "controlled_operator_coverage": {
            "business_fields": {
                "required_operator_ids": list(BUSINESS_FIELD_OPERATOR_IDS),
                "observed_operator_ids": [item["mutation"]["operator_id"] for item in defects],
                "operator_count": len(BUSINESS_FIELD_OPERATOR_IDS),
                "expected_outcome": "needs_review",
                "coverage_sha256": _sha256_bytes(
                    _canonical_bytes(
                        [
                            {
                                "operator_id": item["mutation"]["operator_id"],
                                "expected_rule_ids": item["expected_rule_ids"],
                                "expected_outcome": item["expected_outcome"],
                            }
                            for item in defects
                        ]
                    )
                ),
            },
            "workflow_boundaries": {
                "descriptor_relative_path": (
                    "evals/trade_document_triplets/v1/workflow-boundary.descriptor.json"
                ),
                "descriptor_sha256": _sha256_file(workflow_descriptor_path),
                "required_operator_ids": list(WORKFLOW_BOUNDARY_OPERATOR_IDS),
                "observed_operator_ids": workflow_operator_ids,
                "operator_count": len(WORKFLOW_BOUNDARY_OPERATOR_IDS),
                "expected_outcomes_sha256": _sha256_bytes(
                    _canonical_bytes(
                        [
                            {
                                "operator_id": scenario["operator_id"],
                                "expected_outcome": scenario["expected_outcome"],
                                "expected_issue_codes": scenario["expected_issue_codes"],
                                "expected_blocker_codes": scenario["expected_blocker_codes"],
                            }
                            for scenario in workflow_scenarios
                        ]
                    )
                ),
            },
        },
        "release_holdout": {
            "split": "locked_test",
            "case_count": len(release_holdout_ids),
            "case_ids_sha256": _sha256_bytes(_canonical_bytes(sorted(release_holdout_ids))),
            "gold_or_controlled_fixture_generated": False,
            "status": "release_holdout_v2_publicly_derivable",
            "release_eligible": False,
        },
        "invalidated_history": {
            "evaluation_ids": ["EVAL-0003"],
            "status": "holdout_compromised_by_precommit_evaluation",
            "artifact_policy": "preserve_do_not_overwrite_or_claim_as_release_evidence",
        },
        "adjudication_event_chain": {
            "relative_path": ADJUDICATION_EVENT_RELATIVE_PATH,
            "sha256": _sha256_bytes(adjudication_event_content),
            "byte_size": len(adjudication_event_content),
            "schema_version": ADJUDICATION_EVENT_SCHEMA,
            "event_count": adjudication_chain["event_count"],
            "chain_head_sha256": adjudication_chain["chain_head_sha256"],
            "event_hashes_sha256": adjudication_chain["event_hashes_sha256"],
            "initial_states_sha256": adjudication_chain["initial_states_sha256"],
            "folded_states_sha256": adjudication_chain["folded_states_sha256"],
            "event_counts_sha256": adjudication_chain["event_counts_sha256"],
            "folded_scoring_eligibility_sha256": adjudication_chain[
                "folded_scoring_eligibility_sha256"
            ],
            "scoring_eligible_count": adjudication_chain["scoring_eligible_count"],
        },
        "separation": {
            "extraction_gold_contains_validation_labels": False,
            "validation_gold_contains_full_source_documents": False,
            "validation_gold_may_contain_minimal_observation_values": True,
            "release_holdout_excluded_from_precommit_gold": True,
            "release_holdout_excluded_from_controlled_mutations": True,
        },
        "adjudication_states": sorted(ADJUDICATION_STATES),
        "counts": {
            "extraction_case_count": len(extraction_records),
            "validation_case_count": len(validation_cases),
            "adjudication_count": len(adjudications),
            "scored_source_adjudication_count": sum(
                1 for item in adjudications if item["scoring_eligible"]
            ),
            "controlled_defect_count": len(defects),
            "certificate_of_origin_fixture_count": len(certificates),
            "release_holdout_case_count": len(release_holdout_ids),
            "workflow_boundary_case_count": len(workflow_scenarios),
        },
        "schema_sha256": dict(sorted(schema_hashes.items())),
        "design_sha256": {
            "certificate_of_origin": _sha256_file(co_design_path),
            "workflow_boundary": _sha256_file(workflow_descriptor_path),
        },
        "generated_files": [
            {
                "relative_path": relative_path,
                "sha256": _sha256_bytes(content),
                "byte_size": len(content),
            }
            for relative_path, content in sorted(content_by_relative_path.items())
        ],
    }
    manifest_content = _json_bytes(manifest)
    all_contents = {
        **content_by_relative_path,
        "manifest.json": manifest_content,
    }
    checksums_content = "".join(
        f"{_sha256_bytes(content)}  {relative_path}\n"
        for relative_path, content in sorted(all_contents.items())
    ).encode()
    all_contents["checksums.sha256"] = checksums_content
    for relative_path, content in all_contents.items():
        _write_generated(output / relative_path, content, refresh=refresh)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separated Gold, controlled defects, and C/O fixtures."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-lock", type=Path, default=DEFAULT_DATASET_LOCK)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--schema-root", type=Path, default=DEFAULT_SCHEMA_ROOT)
    parser.add_argument("--co-design", type=Path, default=DEFAULT_CO_DESIGN)
    parser.add_argument(
        "--workflow-descriptor",
        type=Path,
        default=DEFAULT_WORKFLOW_DESCRIPTOR,
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_dataset_provenance(
            args.dataset_lock,
            expected_dataset_root=args.dataset_root,
        )
        manifest = build_evaluation_assets(
            args.dataset_root,
            args.output_root,
            schema_root=args.schema_root,
            co_design_path=args.co_design,
            workflow_descriptor_path=args.workflow_descriptor,
            refresh=args.refresh,
        )
    except DatasetProvenanceError as exc:
        print(f"evaluation asset build failed: {exc.code}")
        return 1
    except EvalAssetError as exc:
        print(f"evaluation asset build failed: {exc}")
        return 1
    counts = manifest["counts"]
    print(
        "evaluation asset build passed: "
        f"{counts['extraction_case_count']} extraction cases, "
        f"{counts['adjudication_count']} source adjudications, "
        f"{counts['controlled_defect_count']} controlled defects, "
        f"{counts['certificate_of_origin_fixture_count']} C/O fixtures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
