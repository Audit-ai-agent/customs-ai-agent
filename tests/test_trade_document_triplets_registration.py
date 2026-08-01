import json
from pathlib import Path

import pytest

from tools.register_trade_document_triplets import (
    ANOMALY_SCHEMA,
    CASE_COUNT,
    DATASET_ID,
    DatasetRegistrationError,
    RAW_DOCUMENT_SPECS,
    register_dataset,
)


def _ci(index: int, *, amount: str | None = None) -> dict[str, str]:
    suffix = f"{index + 1:04d}"
    return {
        "Shipper / Seller": "Synthetic Seller",
        "Consignee (or For account & risk of Messrs)": "Synthetic Consignee",
        "Departure Date": "2025-08-13",
        "Vessel / Flight": "Synthetic Vessel / V.001E",
        "From": "Busan, South Korea",
        "To": "Los Angeles, USA",
        "Invoice No. and Date": f"KR-INV-{suffix} / 2025-08-05",
        "L/C No. and Date": f"KR-LC-{suffix} / 2025-07-09",
        "Buyer (if other than consignee)": "SAME AS CONSIGNEE",
        "Other References (Country of Origin)": "Country of Origin: Republic of Korea",
        "Terms of Delivery and Payment": "FOB / T/T 30 Days",
        "Shipping Marks": f"SYNTHETIC // C/NO. {suffix}",
        "No. & Kind of Packages": "10 Cartons",
        "Goods Description": "Synthetic Primary Item",
        "Quantity": "3 PCS",
        "Unit Price": "USD 10.00",
        "Amount": amount or "USD 30.00",
        "Signed By": "Synthetic Signer",
    }


def _pl(index: int) -> dict[str, object]:
    suffix = f"{index + 1:04d}"
    return {
        "Seller": "Synthetic Seller",
        "Consignee": "Synthetic Consignee",
        "Departure Date": "2025-08-13",
        "Vessel / Flight": "Synthetic Vessel / V.001E",
        "From": "Busan, South Korea",
        "To": "Los Angeles, USA",
        "Invoice No. and Date": f"KR-INV-{suffix} / 2025-08-05",
        "Buyer (if other than consignee)": "SAME AS CONSIGNEE",
        "Other References": "Country of Origin: Republic of Korea",
        "Shipping Marks": f"SYNTHETIC // C/NO. {suffix}",
        "No. & Kind of Packages": "10 Cartons",
        "Goods Details": [
            {
                "item": "Synthetic Primary Item",
                "quantity": "3 PCS",
                "weight": "20.00 KGS",
            },
            {
                "item": "Synthetic Additional Item",
                "quantity": "1 PCS",
                "weight": "5.00 KGS",
            },
        ],
        "Gross Weight": "30.00 KGS",
        "Measurement": "1.50 CBM",
        "Signed By": "Synthetic Signer",
    }


def _bl(index: int) -> dict[str, str]:
    suffix = f"{index + 1:04d}"
    return {
        "Shipper / Exporter": "Synthetic Seller",
        "Consignee": "Synthetic Consignee",
        "Notify Party": "SAME AS CONSIGNEE",
        "Ocean Vessel": "Synthetic Vessel",
        "Port of Loading": "Busan, South Korea",
        "Place of Receipt": "Busan CY",
        "Voyage No.": "V.001E",
        "Port of Discharge": "Los Angeles, USA",
        "Place of Delivery": "Los Angeles CY",
        "Final Destination": "Los Angeles, USA",
        "B/L No. (Bill of Lading Number)": f"BL{suffix}",
        "Flag": "Republic of Korea",
        "Container No.": f"CONT{suffix}",
        "Seal No. / Marks & No.": f"SEAL{suffix} // SYNTHETIC // C/NO. {suffix}",
        "No. & Kinds of Containers or Packages": "1x40HQ / 10 Cartons",
        "Description of Goods": "SAID TO CONTAIN: Synthetic Primary Item AND ACCESSORIES",
        "Gross Weight": "30.00 KGS",
        "Freight and Charges": "FREIGHT AS ARRANGED",
        "Revenue Tons": "1.50 RT",
        "Rate": "AS PER TARIFF",
        "Per": "RT",
        "Prepaid": "Y",
        "Collect": "",
        "Freight Prepaid At": "Busan, South Korea",
        "Freight Payable At": "N/A",
        "Place and Date of Issue": "Busan / 2025-08-13",
        "No. of Original B/L": "THREE (3)",
        "Laden on Board (Date / Signature)": "2025-08-13 / SIGNED ON BOARD",
        "Carrier / Agent Signature": "SYNTHETIC CARRIER",
    }


def _write_source(root: Path, *, packing_count: int = CASE_COUNT) -> dict[str, bytes]:
    root.mkdir()
    records_by_filename = {
        "CI_100_Final.json": [
            _ci(index, amount="USD 30.01" if index == 0 else None) for index in range(CASE_COUNT)
        ],
        "PL_100_Final.json": [_pl(index) for index in range(packing_count)],
        "BL_100_Final.json": [_bl(index) for index in range(CASE_COUNT)],
    }
    original_bytes: dict[str, bytes] = {}
    for filename, records in records_by_filename.items():
        content = (
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        (root / filename).write_bytes(content)
        original_bytes[filename] = content
    return original_bytes


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_registration_preserves_raw_bytes_and_is_deterministic(tmp_path):
    source = tmp_path / "source"
    original_bytes = _write_source(source)
    first_target = tmp_path / "registered-a"
    second_target = tmp_path / "registered-b"

    first = register_dataset(source, first_target)
    second = register_dataset(source, second_target)

    assert first == second
    assert first["summary"]["case_count"] == 100
    assert first["summary"]["document_count"] == 300
    assert first["summary"]["raw_bytes_preserved"] is True
    for spec in RAW_DOCUMENT_SPECS:
        assert (source / spec.filename).read_bytes() == original_bytes[spec.filename]
        assert (first_target / "raw" / spec.filename).read_bytes() == original_bytes[spec.filename]
        assert (second_target / "raw" / spec.filename).read_bytes() == original_bytes[spec.filename]

    generated_paths = (
        "inventory.json",
        "provenance.json",
        "checksums.sha256",
        "derived/cases.jsonl",
        "derived/splits.json",
        "derived/evaluation-splits/v2/splits.json",
        "anomaly-registry.jsonl",
    )
    for relative_path in generated_paths:
        assert (first_target / relative_path).read_bytes() == (
            second_target / relative_path
        ).read_bytes()

    cases = _jsonl(first_target / "derived" / "cases.jsonl")
    assert len(cases) == CASE_COUNT
    assert len({case["case_id"] for case in cases}) == CASE_COUNT
    assert cases[0]["case_id"] == "trade-triplet-0001"
    assert cases[-1]["case_id"] == "trade-triplet-0100"
    assert cases[0]["pairing"]["author_intent_status"] == "unknown"

    splits = json.loads((first_target / "derived" / "splits.json").read_text())
    assert splits["counts"] == {
        "development": 60,
        "locked_test": 20,
        "validation": 20,
    }
    assigned = [
        case_id for split_case_ids in splits["splits"].values() for case_id in split_case_ids
    ]
    assert len(assigned) == CASE_COUNT
    assert len(set(assigned)) == CASE_COUNT

    evaluation_v2 = json.loads(
        (first_target / "derived" / "evaluation-splits" / "v2" / "splits.json").read_text()
    )
    assert evaluation_v2["schema_version"] == "trade-document-evaluation-splits/v2"
    assert evaluation_v2["evaluation_version"] == "evaluation-v2"
    assert evaluation_v2["counts"] == splits["counts"]
    assert (
        evaluation_v2["legacy"]["holdout_status"] == "holdout_compromised_by_precommit_evaluation"
    )
    assert evaluation_v2["legacy"]["invalidated_evaluation_ids"] == ["EVAL-0003"]
    assert not (
        set(evaluation_v2["legacy"]["splits"]["locked_test"])
        & set(evaluation_v2["splits"]["locked_test"])
    )
    assert set(evaluation_v2["legacy"]["splits"]["locked_test"]) <= (
        set(evaluation_v2["splits"]["development"]) | set(evaluation_v2["splits"]["validation"])
    )
    assert evaluation_v2["holdout_policy"] == {
        "release_eligible": False,
        "release_status": "release_holdout_v2_publicly_derivable",
        "allowed_use": "development_diagnostic_only",
        "controlled_mutation_generation_allowed_before_release": False,
        "model_or_rule_tuning_allowed": False,
        "replacement_protocol": "sealed-release-holdout-bundle-manifest/v1",
    }


def test_observed_differences_never_assume_defect_intent(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    target = tmp_path / "registered"

    inventory = register_dataset(source, target)
    observations = _jsonl(target / "anomaly-registry.jsonl")

    assert all(observation["schema_version"] == ANOMALY_SCHEMA for observation in observations)
    assert {observation["status"] for observation in observations} <= {
        "ambiguous",
        "coverage_gap",
    }
    assert all(observation["intent_status"] == "unknown" for observation in observations)
    assert all(observation["scoring_eligible"] is False for observation in observations)
    assert not any(
        observation["status"] in {"confirmed_intentional_defect", "confirmed_generator_error"}
        for observation in observations
    )

    amount_observation = next(
        observation
        for observation in observations
        if observation["anomaly_id"] == "trade-triplet-0001:ci.amount_vs_displayed_unit_price"
    )
    assert amount_observation["status"] == "ambiguous"
    assert amount_observation["evidence"]["compatible_with_displayed_precision_rounding"] is True
    assert (
        inventory["audit"]["observed_conditions"]["additional_packing_item_case_count"]
        == CASE_COUNT
    )
    assert any(
        observation["anomaly_id"] == "dataset:defect_intent_gap"
        and observation["status"] == "coverage_gap"
        for observation in observations
    )


def test_registration_is_idempotent_but_never_overwrites_changed_raw(tmp_path):
    source = tmp_path / "source"
    original_bytes = _write_source(source)
    target = tmp_path / "registered"

    register_dataset(source, target)
    register_dataset(source, target)
    changed_raw = target / "raw" / "CI_100_Final.json"
    changed_raw.write_bytes(b"changed")

    with pytest.raises(DatasetRegistrationError, match="will not be overwritten"):
        register_dataset(source, target, refresh_derived=True)
    assert changed_raw.read_bytes() == b"changed"
    assert (source / "CI_100_Final.json").read_bytes() == original_bytes["CI_100_Final.json"]


def test_registration_rejects_unpaired_source_before_copy(tmp_path):
    source = tmp_path / "source"
    _write_source(source, packing_count=CASE_COUNT - 1)
    target = tmp_path / "registered"

    with pytest.raises(DatasetRegistrationError, match="must contain 100 records"):
        register_dataset(source, target)
    assert not target.exists()


def test_dataset_lock_mismatch_is_rejected_before_copy(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    target = tmp_path / "registered"
    real_lock = (
        Path(__file__).parents[1] / "evals" / "trade_document_triplets" / "v1" / "dataset.lock.json"
    )

    with pytest.raises(DatasetRegistrationError, match="dataset lock"):
        register_dataset(source, target, lock_path=real_lock)
    assert not target.exists()


def test_dataset_lock_pins_the_supplied_source_without_business_values():
    lock_path = (
        Path(__file__).parents[1] / "evals" / "trade_document_triplets" / "v1" / "dataset.lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert lock["dataset_id"] == DATASET_ID
    assert lock["lock_revision"] == 2
    assert lock["artifact_relative_path"] == ("_local/data/trade_document_triplets/source-v1")
    assert lock["defect_intent_status"] == "unknown"
    assert "contains_business_values" not in lock
    assert lock["descriptor_contains_business_values"] is False
    assert lock["raw_dataset_contains_business_values"] is True
    assert lock["value_bearing_artifacts_committable"] is False
    provenance_v2 = lock["dataset_provenance_v2"]
    assert provenance_v2["schema_version"] == "customs-dataset-provenance-binding/v2"
    assert provenance_v2["migration_id"] == "TASK-0028-dataset-canonical-relocation"
    assert provenance_v2["root_reference_preserved"] is False
    assert provenance_v2["root_reference_status"] == "retired"
    assert provenance_v2["fallback_to_legacy_paths_allowed"] is False
    assert lock["expected_registration"] == {
        "case_count": 100,
        "document_count": 300,
        "pairing_basis": "shared_zero_based_array_index",
        "raw_total_bytes": 347619,
        "source_explicit_case_id": False,
        "split_counts": {
            "development": 60,
            "locked_test": 20,
            "validation": 20,
        },
    }
    assert {item["filename"]: item["sha256"] for item in lock["raw_files"]} == {
        "CI_100_Final.json": ("6a4a43768402d7378c9f55ea8135437cd1a403b68dc2f1a04e600a22609df472"),
        "PL_100_Final.json": ("31fed773233bac0beadf4bb7e024b5c0fc9b049b54d9b9353883c5fed1bb735f"),
        "BL_100_Final.json": ("99feb8ce01b0cff6a84a223439dc4d12a98cdeb6ddaf447f44a31d298340b3ae"),
    }
