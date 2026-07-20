from __future__ import annotations

import copy
import hashlib
import json

import pytest

import evals.run_deterministic_validation as deterministic_eval
from evals.run_deterministic_validation import (
    DeterministicEvaluationError,
    OPERATOR_EXPECTED_ISSUES,
    evaluate,
)
from tools.build_trade_triplet_eval_assets import (
    BUSINESS_FIELD_OPERATOR_IDS,
    inject_controlled_defects,
)


def _source_case(index: int) -> dict:
    case_id = f"trade-triplet-{index:04d}"
    suffix = f"{index:04d}"
    ci = {
        "Shipper / Seller": "Synthetic Seller",
        "Consignee (or For account & risk of Messrs)": "Synthetic Consignee",
        "Departure Date": "2025-08-13",
        "Vessel / Flight": "Synthetic Vessel / V.001E",
        "From": "Busan, South Korea",
        "To": "Los Angeles, USA",
        "Invoice No. and Date": f"KR-INV-{suffix} / 2025-08-05",
        "Buyer (if other than consignee)": "SAME AS CONSIGNEE",
        "Other References (Country of Origin)": ("Country of Origin: Republic of Korea"),
        "Shipping Marks": f"SYNTHETIC // C/NO. {suffix}",
        "No. & Kind of Packages": "10 Cartons",
        "Goods Description": "Synthetic Primary Item",
        "Quantity": "3 PCS",
        "Unit Price": "USD 10.00",
        "Amount": "USD 30.00",
    }
    pl = {
        "Seller": "Synthetic Seller",
        "Consignee": "Synthetic Consignee",
        "Departure Date": "2025-08-13",
        "Vessel / Flight": "Synthetic Vessel / V.001E",
        "From": "Busan, South Korea",
        "To": "Los Angeles, USA",
        "Invoice No. and Date": ci["Invoice No. and Date"],
        "Buyer (if other than consignee)": "SAME AS CONSIGNEE",
        "Other References": "Country of Origin: Republic of Korea",
        "Shipping Marks": ci["Shipping Marks"],
        "No. & Kind of Packages": "10 Cartons",
        "Goods Details": [
            {
                "item": "Synthetic Primary Item",
                "quantity": "3 PCS",
                "weight": "20.00 KGS",
            }
        ],
        "Gross Weight": "30.00 KGS",
        "Measurement": "1.50 CBM",
    }
    bl = {
        "Shipper / Exporter": "Synthetic Seller",
        "Consignee": "Synthetic Consignee",
        "Ocean Vessel": "Synthetic Vessel",
        "Voyage No.": "V.001E",
        "Port of Loading": "Busan, South Korea",
        "Port of Discharge": "Los Angeles, USA",
        "B/L No. (Bill of Lading Number)": f"BL{suffix}",
        "Container No.": "HMMU9279350",
        "Seal No. / Marks & No.": f"SEAL{suffix} // {ci['Shipping Marks']}",
        "No. & Kinds of Containers or Packages": "1x40HQ / 10 Cartons",
        "Gross Weight": "30.00 KGS",
        "Revenue Tons": "1.50 RT",
        "Prepaid": "Y",
        "Collect": "",
    }
    return {
        "schema_version": "trade-document-triplet-case/v1",
        "dataset_id": "trade_document_triplets",
        "dataset_version": "source-v1",
        "case_id": case_id,
        "documents": {
            "commercial_invoice": {
                "source_relative_path": "raw/CI_100_Final.json",
                "source_file_sha256": f"{index:064x}"[-64:],
                "source_record_index_zero_based": index - 1,
                "raw_record": ci,
            },
            "packing_list": {
                "source_relative_path": "raw/PL_100_Final.json",
                "source_file_sha256": f"{index:064x}"[-64:],
                "source_record_index_zero_based": index - 1,
                "raw_record": pl,
            },
            "bill_of_lading": {
                "source_relative_path": "raw/BL_100_Final.json",
                "source_file_sha256": f"{index:064x}"[-64:],
                "source_record_index_zero_based": index - 1,
                "raw_record": bl,
            },
        },
        "pairing": {
            "basis": "shared_zero_based_array_index",
            "source_explicit_case_id": False,
            "author_intent_status": "unknown",
            "adjudication_status": "ambiguous",
        },
    }


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _sync_controlled_manifest(bundle, defects):
    _write_jsonl(bundle["defects_path"], defects)
    assignments = [
        {
            "source_case_id": item["source_case_id"],
            "operator_id": item["mutation"]["operator_id"],
            "variant_id": item["variant_id"],
        }
        for item in defects
    ]
    defects_sha256 = _sha256(bundle["defects_path"])
    defects_byte_size = bundle["defects_path"].stat().st_size
    manifest = bundle["manifest"]
    manifest["source"]["dataset_lock_sha256"] = _sha256(bundle["dataset_lock_path"])
    manifest["source"]["evaluation_split_sha256"] = _sha256(bundle["splits_path"])
    manifest["counts"]["controlled_defect_count"] = len(defects)
    manifest["generated_files"] = [
        {
            "relative_path": deterministic_eval.CONTROLLED_DEFECT_RELATIVE_PATH,
            "sha256": defects_sha256,
            "byte_size": defects_byte_size,
        }
    ]
    manifest["controlled_defects"] = {
        "relative_path": deterministic_eval.CONTROLLED_DEFECT_RELATIVE_PATH,
        "sha256": defects_sha256,
        "byte_size": defects_byte_size,
        "generation_split": "development",
        "release_holdout_used": False,
        "selection_seed": deterministic_eval.CONTROLLED_CASE_SEED,
        "case_count": len(defects),
        "case_ids": [item["source_case_id"] for item in defects],
        "operator_ids": sorted({item["mutation"]["operator_id"] for item in defects}),
        "case_operator_assignments_sha256": _canonical_sha256(assignments),
    }
    business_coverage = [
        {
            "operator_id": item["mutation"]["operator_id"],
            "expected_rule_ids": item["expected_rule_ids"],
            "expected_outcome": item["expected_outcome"],
        }
        for item in defects
    ]
    workflow_ids = [
        "required_document_missing",
        "duplicate_document_type",
        "ocr_conflict_unresolved",
        "evidence_partial",
        "evidence_failed",
        "evidence_ambiguous",
    ]
    manifest["controlled_operator_coverage"] = {
        "business_fields": {
            "required_operator_ids": list(deterministic_eval.OPERATOR_EXPECTED_ISSUES),
            "observed_operator_ids": [item["mutation"]["operator_id"] for item in defects],
            "operator_count": len(deterministic_eval.OPERATOR_EXPECTED_ISSUES),
            "expected_outcome": "needs_review",
            "coverage_sha256": _canonical_sha256(business_coverage),
        },
        "workflow_boundaries": {
            "descriptor_relative_path": "workflow-boundary.descriptor.json",
            "descriptor_sha256": "a" * 64,
            "required_operator_ids": workflow_ids,
            "observed_operator_ids": workflow_ids,
            "operator_count": 6,
            "expected_outcomes_sha256": "b" * 64,
        },
    }
    manifest["release_holdout"] = {
        "release_eligible": False,
        "status": "release_holdout_v2_publicly_derivable",
    }
    manifest["invalidated_history"] = {
        "evaluation_ids": ["EVAL-0003"],
        "status": "holdout_compromised_by_precommit_evaluation",
        "artifact_policy": "preserve_do_not_overwrite_or_claim_as_release_evidence",
    }
    _write_json(bundle["manifest_path"], manifest)
    bundle["defects"] = defects


def _pin_bundle(monkeypatch, bundle):
    monkeypatch.setattr(
        deterministic_eval,
        "CANONICAL_DATASET_LOCK_SHA256",
        _sha256(bundle["dataset_lock_path"]),
    )
    monkeypatch.setattr(
        deterministic_eval,
        "CANONICAL_DIAGNOSTIC_SPLIT_BYTES_SHA256",
        _sha256(bundle["splits_path"]),
    )
    monkeypatch.setattr(
        deterministic_eval,
        "CANONICAL_CONTROLLED_DEFECTS_SHA256",
        _sha256(bundle["defects_path"]),
    )
    monkeypatch.setattr(
        deterministic_eval,
        "CANONICAL_EVALUATION_MANIFEST_SHA256",
        _sha256(bundle["manifest_path"]),
    )


def _evidence_bundle(tmp_path, monkeypatch):
    dataset_lock = {
        "schema_version": deterministic_eval.DATASET_LOCK_SCHEMA,
        "dataset_id": deterministic_eval.DATASET_ID,
        "dataset_version": deterministic_eval.DATASET_VERSION,
        "defect_intent_status": "unknown",
        "expected_registration": {
            "case_count": 100,
            "document_count": 300,
            "split_counts": deterministic_eval.EXPECTED_SPLIT_COUNTS,
        },
    }
    root = tmp_path / "controlled-evaluation"
    bundle = {
        "defects_path": root / "evaluation" / "injected-defects" / "cases.jsonl",
        "manifest_path": root / "evaluation" / "manifest.json",
        "splits_path": root / "source-v1" / "derived" / "splits.json",
        "dataset_lock_path": root / "dataset.lock.json",
        "manifest": {
            "schema_version": deterministic_eval.EVALUATION_MANIFEST_SCHEMA,
            "dataset_id": deterministic_eval.DATASET_ID,
            "dataset_version": deterministic_eval.DATASET_VERSION,
            "evaluation_version": deterministic_eval.EVALUATION_VERSION,
            "source": {},
            "counts": {},
        },
    }
    bundle["splits_path"].parent.mkdir(parents=True, exist_ok=True)
    bundle["splits_path"].write_bytes(b"opaque-canonical-v2-diagnostic-provenance\n")
    _write_json(bundle["dataset_lock_path"], dataset_lock)
    controlled_case_ids = [f"trade-triplet-{index:04d}" for index in range(1, 21)]
    bundle["controlled_case_ids"] = controlled_case_ids
    generator_case_ids = [
        f"trade-triplet-{index:04d}" for index in range(1, len(BUSINESS_FIELD_OPERATOR_IDS) + 1)
    ]
    cases = [_source_case(int(case_id.rsplit("-", 1)[1])) for case_id in generator_case_ids]
    defects = inject_controlled_defects(cases, generator_case_ids)[:20]
    _sync_controlled_manifest(bundle, defects)
    _pin_bundle(monkeypatch, bundle)
    return bundle


def _evaluate_bundle(bundle):
    return evaluate(
        bundle["defects_path"],
        manifest_path=bundle["manifest_path"],
        splits_path=bundle["splits_path"],
        dataset_lock_path=bundle["dataset_lock_path"],
    )


def test_historical_v2_contract_remains_the_original_first_20_operators():
    assert tuple(OPERATOR_EXPECTED_ISSUES) == BUSINESS_FIELD_OPERATOR_IDS[:20]
    assert len(OPERATOR_EXPECTED_ISSUES) == 20


def test_controlled_single_mutation_evaluation_reaches_exact_precision_and_recall(
    tmp_path, monkeypatch
):
    bundle = _evidence_bundle(tmp_path, monkeypatch)

    exit_code, report = _evaluate_bundle(bundle)

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["case_count"] == 20
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["false_positive"] == 0
    assert report["false_negative"] == 0
    assert report["cases_requiring_human_review"] == 20
    assert report["input_sha256"] == _sha256(bundle["defects_path"])
    assert report["evaluation_manifest_sha256"] == _sha256(bundle["manifest_path"])
    assert report["dataset_lock_sha256"] == _sha256(bundle["dataset_lock_path"])
    assert report["split_sha256"] == _sha256(bundle["splits_path"])
    assert report["release_eligible"] is False
    assert report["release_holdout_status"] == "release_holdout_v2_publicly_derivable"
    assert "release_holdout_case_ids_sha256" not in report
    assert "split_assignments_sha256" not in report
    assert report["operator_ids"] == sorted(OPERATOR_EXPECTED_ISSUES)
    assert all(result["pass"] is True for result in report["results"])


def test_evaluation_rejects_a_mutation_that_cannot_reconstruct_pinned_parent(tmp_path, monkeypatch):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    tampered = copy.deepcopy(bundle["defects"])
    tampered[0]["source_case_sha256"] = "0" * 64
    _sync_controlled_manifest(bundle, tampered)
    _pin_bundle(monkeypatch, bundle)

    with pytest.raises(
        DeterministicEvaluationError,
        match="pinned source hash",
    ):
        _evaluate_bundle(bundle)


def test_evaluation_rejects_adjudication_evidence_that_diverges_from_mutation(
    tmp_path,
    monkeypatch,
):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    tampered = copy.deepcopy(bundle["defects"])
    tampered[0]["adjudication"]["evidence"]["after"] = "internally-inconsistent"
    _sync_controlled_manifest(bundle, tampered)
    _pin_bundle(monkeypatch, bundle)

    with pytest.raises(
        DeterministicEvaluationError,
        match="adjudication evidence",
    ):
        _evaluate_bundle(bundle)


def test_parent_reconstruction_rejects_mutated_value_that_diverges_from_after(
    tmp_path,
    monkeypatch,
):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    tampered = copy.deepcopy(bundle["defects"][0])
    mutation = tampered["mutation"]
    target = tampered["mutated_case"]["documents"][mutation["document_type"]]["raw_record"]
    for part in mutation["field_path"][:-1]:
        target = target[part]
    target[mutation["field_path"][-1]] = "internally-inconsistent"

    with pytest.raises(
        DeterministicEvaluationError,
        match="mutation.after",
    ):
        deterministic_eval._reconstruct_parent(tampered)


def test_development_set_exercises_every_pinned_business_mutation_operator(tmp_path, monkeypatch):
    bundle = _evidence_bundle(tmp_path, monkeypatch)

    exit_code, report = _evaluate_bundle(bundle)

    assert exit_code == 0
    assert report["case_count"] == 20
    assert {result["operator_id"] for result in report["results"]} == set(OPERATOR_EXPECTED_ISSUES)
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_evaluation_rejects_self_consistent_partial_development_input(tmp_path, monkeypatch):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    partial = copy.deepcopy(bundle["defects"][:-1])
    _sync_controlled_manifest(bundle, partial)
    _pin_bundle(monkeypatch, bundle)

    with pytest.raises(
        DeterministicEvaluationError,
        match="evaluation manifest contract",
    ):
        _evaluate_bundle(bundle)


def test_evaluation_rejects_self_consistent_missing_operator_coverage(tmp_path, monkeypatch):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    tampered = copy.deepcopy(bundle["defects"])
    target = tampered[12]
    target["mutation"]["operator_id"] = "pl_invoice_number_mismatch"
    target["variant_id"] = f"{target['source_case_id']}--{target['mutation']['operator_id']}"
    _sync_controlled_manifest(bundle, tampered)
    _pin_bundle(monkeypatch, bundle)

    with pytest.raises(
        DeterministicEvaluationError,
        match="every pinned mutation operator",
    ):
        _evaluate_bundle(bundle)


def test_evaluation_rejects_controlled_input_hash_tampering(tmp_path, monkeypatch):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    bundle["defects_path"].write_text(
        bundle["defects_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DeterministicEvaluationError,
        match="controlled defect input hash",
    ):
        _evaluate_bundle(bundle)


def test_evaluation_rejects_manifest_hash_tampering(tmp_path, monkeypatch):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    bundle["manifest_path"].write_text(
        bundle["manifest_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DeterministicEvaluationError,
        match="evaluation manifest hash",
    ):
        _evaluate_bundle(bundle)


def test_development_evaluator_does_not_parse_split_membership(
    tmp_path,
    monkeypatch,
):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    original_json_object = deterministic_eval._json_object

    def guarded_json_object(path, *, label):
        if path == bundle["splits_path"]:
            raise AssertionError("development evaluator parsed split membership")
        return original_json_object(path, label=label)

    monkeypatch.setattr(deterministic_eval, "_json_object", guarded_json_object)

    exit_code, report = _evaluate_bundle(bundle)

    assert exit_code == 0
    assert report["release_eligible"] is False


def test_development_evaluator_rejects_split_byte_drift_without_parsing(
    tmp_path,
    monkeypatch,
):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    bundle["splits_path"].write_bytes(bundle["splits_path"].read_bytes() + b"tampered\n")

    with pytest.raises(
        DeterministicEvaluationError,
        match="canonical diagnostic input",
    ):
        _evaluate_bundle(bundle)


def test_evaluation_rejects_dataset_lock_hash_tampering(tmp_path, monkeypatch):
    bundle = _evidence_bundle(tmp_path, monkeypatch)
    bundle["dataset_lock_path"].write_text(
        bundle["dataset_lock_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DeterministicEvaluationError,
        match="dataset lock hash",
    ):
        _evaluate_bundle(bundle)
