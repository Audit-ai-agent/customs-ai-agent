from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest

import evals.run_rule_coverage_validation as rule_eval
import tools.build_trade_triplet_rule_coverage as rule_builder
from evals.run_rule_coverage_validation import (
    DeterministicEvaluationError,
    OPERATOR_EXPECTED_FINDINGS,
)
from tools.build_trade_triplet_eval_assets import BUSINESS_FIELD_OPERATOR_IDS
from tools.build_trade_triplet_rule_coverage import (
    CHECKSUMS_RELATIVE_PATH,
    DEFECTS_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    NEGATIVE_CONTROLS_RELATIVE_PATH,
    RuleCoverageBuildError,
    build_rule_coverage_bundle,
)


class EvaluationBundle(TypedDict):
    dataset_root: Path
    dataset_lock_path: Path
    descriptor_path: Path
    output_root: Path
    defects_path: Path
    negative_controls_path: Path
    manifest_path: Path
    checksums_path: Path
    build_report: dict[str, Any]


def test_rule_coverage_cli_redacts_filesystem_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_path = "C:" + r"\Users" + r"\private-reviewer\missing-dataset"
    monkeypatch.setattr(
        rule_builder,
        "_parse_args",
        lambda: SimpleNamespace(
            dataset_root=Path("unused"),
            output_root=Path("unused"),
            descriptor=Path("unused"),
            dataset_lock=Path("unused"),
        ),
    )
    monkeypatch.setattr(
        rule_builder,
        "build_rule_coverage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(private_path)),
    )

    assert rule_builder.main() == 1
    output = capsys.readouterr().out
    assert "filesystem_operation_failed" in output
    assert private_path not in output


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
        "Other References (Country of Origin)": "Country of Origin: Republic of Korea",
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
        "dataset_id": rule_eval.DATASET_ID,
        "dataset_version": rule_eval.DATASET_VERSION,
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


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(records: list[dict]) -> bytes:
    return (
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jsonl_bytes(records))


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _descriptor() -> dict:
    return {
        "schema_version": rule_eval.SELECTION_DESCRIPTOR_SCHEMA,
        "descriptor_id": "trade-document-triplets-rule-coverage-v1",
        "dataset_id": rule_eval.DATASET_ID,
        "dataset_version": rule_eval.DATASET_VERSION,
        "evaluation_version": rule_eval.EVALUATION_VERSION,
        "scope": "public_registered_synthetic_development_diagnostic",
        "release_eligible": False,
        "selection": {
            "case_pool": "all_registered_public_synthetic_cases",
            "case_count": len(BUSINESS_FIELD_OPERATOR_IDS) + 1,
            "positive_case_count": len(BUSINESS_FIELD_OPERATOR_IDS),
            "negative_control_case_count": 1,
            "membership_embedded": False,
            "release_split_membership_consulted": False,
            "seed": rule_eval.CONTROLLED_CASE_SEED,
        },
        "finding_grain": [
            "issue_code",
            "source_document_id",
            "target_document_id",
        ],
        "positive_operator_ids": list(BUSINESS_FIELD_OPERATOR_IDS),
        "negative_control_operator_ids": [rule_eval.NEGATIVE_CONTROL_OPERATOR_ID],
        "policy": {
            "allowed_use": "development_diagnostic_only",
            "release_claim_allowed": False,
            "model_or_rule_selection_claim_allowed": False,
            "sealed_unseen_release_bundle_required_for_release": True,
        },
    }


def _registered_dataset(tmp_path: Path) -> dict[str, Path]:
    dataset_root = tmp_path / "registered-dataset"
    raw_root = dataset_root / "raw"
    raw_root.mkdir(parents=True)
    raw_paths = [
        raw_root / "CI_100_Final.json",
        raw_root / "PL_100_Final.json",
        raw_root / "BL_100_Final.json",
    ]
    for index, path in enumerate(raw_paths, start=1):
        path.write_bytes(f'[{{"synthetic_raw_marker":{index}}}]\n'.encode())
    checksums_path = dataset_root / "checksums.sha256"
    checksums_path.write_text(
        "".join(f"{_sha256(path)}  raw/{path.name}\n" for path in raw_paths),
        encoding="utf-8",
    )

    cases = [_source_case(index) for index in range(1, 101)]
    cases_path = dataset_root / "derived" / "cases.jsonl"
    _write_jsonl(cases_path, cases)
    dataset_lock_path = tmp_path / "dataset.lock.json"
    _write_json(
        dataset_lock_path,
        {
            "schema_version": rule_eval.DATASET_LOCK_SCHEMA,
            "dataset_id": rule_eval.DATASET_ID,
            "dataset_version": rule_eval.DATASET_VERSION,
            "defect_intent_status": "unknown",
            "expected_registration": {
                "case_count": 100,
                "document_count": 300,
                "split_counts": rule_eval.EXPECTED_SPLIT_COUNTS,
            },
        },
    )
    _write_json(
        dataset_root / "inventory.json",
        {
            "schema_version": "trade-document-triplets-inventory/v1",
            "dataset": {
                "id": rule_eval.DATASET_ID,
                "version": rule_eval.DATASET_VERSION,
                "lock": {
                    "schema_version": rule_eval.DATASET_LOCK_SCHEMA,
                    "sha256": _sha256(dataset_lock_path),
                },
            },
            "summary": {
                "case_count": 100,
                "document_count": 300,
                "raw_bytes_preserved": True,
                "raw_dataset_contains_business_values": True,
            },
            "derived_artifacts": {
                "cases": {
                    "relative_path": "derived/cases.jsonl",
                    "sha256": _sha256(cases_path),
                    "byte_size": cases_path.stat().st_size,
                }
            },
        },
    )
    descriptor_path = tmp_path / "rule-coverage.descriptor.json"
    _write_json(descriptor_path, _descriptor())
    return {
        "dataset_root": dataset_root,
        "dataset_lock_path": dataset_lock_path,
        "descriptor_path": descriptor_path,
    }


def _pin_bundle(monkeypatch, bundle: EvaluationBundle) -> None:
    monkeypatch.setattr(
        rule_eval,
        "CANONICAL_DATASET_LOCK_SHA256",
        _sha256(bundle["dataset_lock_path"]),
    )
    monkeypatch.setattr(
        rule_eval,
        "CANONICAL_SELECTION_DESCRIPTOR_SHA256",
        _sha256(bundle["descriptor_path"]),
    )
    monkeypatch.setattr(
        rule_eval,
        "CANONICAL_CONTROLLED_DEFECTS_SHA256",
        _sha256(bundle["defects_path"]),
    )
    monkeypatch.setattr(
        rule_eval,
        "CANONICAL_NEGATIVE_CONTROLS_SHA256",
        _sha256(bundle["negative_controls_path"]),
    )
    monkeypatch.setattr(
        rule_eval,
        "CANONICAL_EVALUATION_MANIFEST_SHA256",
        _sha256(bundle["manifest_path"]),
    )


def _bundle(tmp_path: Path, monkeypatch) -> EvaluationBundle:
    source = _registered_dataset(tmp_path)
    output_root = tmp_path / "rule-coverage-v1"
    build_report = build_rule_coverage_bundle(
        source["dataset_root"],
        output_root,
        descriptor_path=source["descriptor_path"],
        dataset_lock_path=source["dataset_lock_path"],
    )
    bundle: EvaluationBundle = {
        "dataset_root": source["dataset_root"],
        "dataset_lock_path": source["dataset_lock_path"],
        "descriptor_path": source["descriptor_path"],
        "output_root": output_root,
        "defects_path": output_root / DEFECTS_RELATIVE_PATH,
        "negative_controls_path": output_root / NEGATIVE_CONTROLS_RELATIVE_PATH,
        "manifest_path": output_root / MANIFEST_RELATIVE_PATH,
        "checksums_path": output_root / CHECKSUMS_RELATIVE_PATH,
        "build_report": build_report,
    }
    _pin_bundle(monkeypatch, bundle)
    return bundle


def _sync_bundle(
    bundle: EvaluationBundle,
    *,
    defects: list[dict] | None = None,
    negative_controls: list[dict] | None = None,
) -> None:
    defects_path = bundle["defects_path"]
    negative_path = bundle["negative_controls_path"]
    manifest_path = bundle["manifest_path"]
    descriptor_path = bundle["descriptor_path"]
    dataset_lock_path = bundle["dataset_lock_path"]
    if defects is None:
        defects = _read_jsonl(defects_path)
    else:
        _write_jsonl(defects_path, defects)
    if negative_controls is None:
        negative_controls = _read_jsonl(negative_path)
    else:
        _write_jsonl(negative_path, negative_controls)

    defects_hash = _sha256(defects_path)
    negative_hash = _sha256(negative_path)
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
    manifest = _read_json(manifest_path)
    manifest["source"]["dataset_lock_sha256"] = _sha256(dataset_lock_path)
    manifest["source"]["selection_descriptor_sha256"] = _sha256(descriptor_path)
    manifest["counts"] = {
        "controlled_defect_count": len(defects),
        "negative_control_count": len(negative_controls),
        "total_variant_count": len(defects) + len(negative_controls),
    }
    manifest["generated_files"] = [
        {
            "relative_path": rule_eval.CONTROLLED_DEFECT_RELATIVE_PATH,
            "sha256": defects_hash,
            "byte_size": defects_path.stat().st_size,
        },
        {
            "relative_path": rule_eval.NEGATIVE_CONTROL_RELATIVE_PATH,
            "sha256": negative_hash,
            "byte_size": negative_path.stat().st_size,
        },
    ]
    manifest["controlled_defects"] = {
        "relative_path": rule_eval.CONTROLLED_DEFECT_RELATIVE_PATH,
        "sha256": defects_hash,
        "byte_size": defects_path.stat().st_size,
        "generation_scope": "public_registered_synthetic_development_diagnostic",
        "release_split_membership_consulted": False,
        "selection_seed": rule_eval.CONTROLLED_CASE_SEED,
        "case_count": len(defects),
        "case_ids": [item["source_case_id"] for item in defects],
        "operator_ids": sorted(item["mutation"]["operator_id"] for item in defects),
        "case_operator_assignments_sha256": _canonical_sha256(assignments),
    }
    manifest["controlled_operator_coverage"]["business_fields"] = {
        "required_operator_ids": list(BUSINESS_FIELD_OPERATOR_IDS),
        "observed_operator_ids": [item["mutation"]["operator_id"] for item in defects],
        "operator_count": len(BUSINESS_FIELD_OPERATOR_IDS),
        "expected_outcome": "needs_review",
        "finding_grain": [
            "issue_code",
            "source_document_id",
            "target_document_id",
        ],
        "coverage_sha256": _canonical_sha256(coverage),
    }
    manifest["controlled_operator_coverage"]["negative_controls"] = {
        "relative_path": rule_eval.NEGATIVE_CONTROL_RELATIVE_PATH,
        "sha256": negative_hash,
        "byte_size": negative_path.stat().st_size,
        "required_operator_ids": [rule_eval.NEGATIVE_CONTROL_OPERATOR_ID],
        "observed_operator_ids": [item["mutation"]["operator_id"] for item in negative_controls],
        "operator_count": 1,
        "expected_finding_delta_count": 0,
    }
    _write_json(manifest_path, manifest)


def _evaluate_bundle(bundle: EvaluationBundle):
    return rule_eval.evaluate(
        bundle["defects_path"],
        negative_controls_path=bundle["negative_controls_path"],
        manifest_path=bundle["manifest_path"],
        selection_descriptor_path=bundle["descriptor_path"],
        dataset_lock_path=bundle["dataset_lock_path"],
    )


def test_rule_coverage_bundle_scores_29_positive_and_one_negative_control(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    negative_variant_id = _read_jsonl(bundle["negative_controls_path"])[0]["variant_id"]

    exit_code, report = _evaluate_bundle(bundle)

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["positive_operator_count"] == 29
    assert report["negative_control_count"] == 1
    assert report["total_variant_count"] == 30
    assert report["true_positive"] == 34
    assert report["false_positive"] == 0
    assert report["false_negative"] == 0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["cases_requiring_human_review"] == 29
    assert report["automatic_violation_count_for_negative_controls"] == 0
    assert report["release_eligible"] is False
    assert report["release_split_membership_consulted"] is False
    assert report["operator_ids"] == sorted(OPERATOR_EXPECTED_FINDINGS)
    assert report["finding_grain"] == [
        "issue_code",
        "source_document_id",
        "target_document_id",
    ]
    package = next(
        result
        for result in report["positive_results"]
        if result["operator_id"] == "pl_package_mismatch"
    )
    assert package["detected_delta_findings"] == [
        {
            "issue_code": "PACKAGE_MISMATCH",
            "source_document_id": "bl",
            "target_document_id": "pl",
        },
        {
            "issue_code": "PACKAGE_MISMATCH",
            "source_document_id": "ci",
            "target_document_id": "pl",
        },
    ]
    assert report["negative_control_results"] == [
        {
            "variant_id": negative_variant_id,
            "operator_id": rule_eval.NEGATIVE_CONTROL_OPERATOR_ID,
            "finding_added_count": 0,
            "finding_removed_count": 0,
            "finding_change_count": 0,
            "unmatched_packing_item_count_before": 0,
            "unmatched_packing_item_count_after": 1,
            "unmatched_packing_item_status": "ambiguous_not_scored",
            "line_item_pairing_status": "deterministic_with_unscored_packing_extras",
            "requires_human_review_from_extra_item": False,
            "pass": True,
        }
    ]


def test_rule_coverage_rejects_adjudication_evidence_diverging_from_mutation(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    defects = _read_jsonl(bundle["defects_path"])
    defects[0]["adjudication"]["evidence"]["after"] = "internally-inconsistent"
    _sync_bundle(bundle, defects=defects)
    _pin_bundle(monkeypatch, bundle)

    with pytest.raises(
        DeterministicEvaluationError,
        match="adjudication evidence",
    ):
        _evaluate_bundle(bundle)


def test_rule_coverage_rejects_mutated_leaf_diverging_from_mutation_after(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    defect = _read_jsonl(bundle["defects_path"])[0]
    mutation = defect["mutation"]
    target = defect["mutated_case"]["documents"][mutation["document_type"]]["raw_record"]
    for part in mutation["field_path"][:-1]:
        target = target[part]
    target[mutation["field_path"][-1]] = "internally-inconsistent"

    with pytest.raises(
        DeterministicEvaluationError,
        match="mutation.after",
    ):
        rule_eval._reconstruct_parent(defect)


def test_rule_coverage_rejects_negative_control_mutation_evidence_tampering(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    negative = _read_jsonl(bundle["negative_controls_path"])
    negative[0]["mutation"]["after"]["item"] = "does not match mutated document"
    _sync_bundle(bundle, negative_controls=negative)
    _pin_bundle(monkeypatch, bundle)

    with pytest.raises(
        DeterministicEvaluationError,
        match="does not match mutation.after",
    ):
        _evaluate_bundle(bundle)


def test_negative_control_detects_findings_removed_by_the_variant(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    negative = _read_jsonl(bundle["negative_controls_path"])
    removed_finding = ("BASELINE_ONLY_FINDING", "ci", "pl")
    evaluations = iter(
        [
            (
                Counter({removed_finding: 1}),
                False,
                {"unmatched_packing_item_ids": []},
                False,
            ),
            (
                Counter(),
                False,
                {
                    "unmatched_packing_item_ids": ["pl-line-2"],
                    "unmatched_packing_item_status": "ambiguous_not_scored",
                    "line_item_pairing_status": ("deterministic_with_unscored_packing_extras"),
                },
                False,
            ),
        ]
    )
    monkeypatch.setattr(rule_eval, "_evaluate_case", lambda _case: next(evaluations))

    metrics = rule_eval.evaluate_negative_control_records(negative)

    assert metrics["passed"] is False
    assert metrics["automatic_violation_count"] == 0
    assert metrics["finding_change_count"] == 1
    assert metrics["results"][0]["finding_removed_count"] == 1


def test_negative_control_adds_exactly_one_unmatched_packing_item(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    record = _read_jsonl(bundle["negative_controls_path"])[0]
    source = copy.deepcopy(record["mutated_case"])
    source["documents"]["packing_list"]["raw_record"]["Goods Details"].pop()

    _, _, baseline_pairing, _ = rule_eval._evaluate_case(source)
    _, _, variant_pairing, _ = rule_eval._evaluate_case(record["mutated_case"])

    assert len(variant_pairing["unmatched_packing_item_ids"]) == (
        len(baseline_pairing["unmatched_packing_item_ids"]) + 1
    )
    assert variant_pairing["unmatched_packing_item_status"] == "ambiguous_not_scored"


def test_rule_coverage_rejects_release_eligible_descriptor_even_when_repinned(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    descriptor = _read_json(bundle["descriptor_path"])
    descriptor["release_eligible"] = True
    _write_json(bundle["descriptor_path"], descriptor)
    _sync_bundle(bundle)
    _pin_bundle(monkeypatch, bundle)

    with pytest.raises(
        DeterministicEvaluationError,
        match="selection descriptor contract",
    ):
        _evaluate_bundle(bundle)


def test_rule_coverage_rejects_negative_control_hash_drift(
    tmp_path,
    monkeypatch,
):
    bundle = _bundle(tmp_path, monkeypatch)
    negative_path = bundle["negative_controls_path"]
    negative_path.write_bytes(negative_path.read_bytes() + b"\n")

    with pytest.raises(
        DeterministicEvaluationError,
        match="negative-control input hash",
    ):
        _evaluate_bundle(bundle)


def test_builder_creates_deterministic_non_release_bundle_without_split_file(
    tmp_path,
):
    source = _registered_dataset(tmp_path)
    raw_hashes_before = {
        path.name: _sha256(path) for path in (source["dataset_root"] / "raw").iterdir()
    }

    first = build_rule_coverage_bundle(
        source["dataset_root"],
        tmp_path / "bundle-one",
        descriptor_path=source["descriptor_path"],
        dataset_lock_path=source["dataset_lock_path"],
    )
    second = build_rule_coverage_bundle(
        source["dataset_root"],
        tmp_path / "bundle-two",
        descriptor_path=source["descriptor_path"],
        dataset_lock_path=source["dataset_lock_path"],
    )

    assert first["case_count"] == 29
    assert first["negative_control_count"] == 1
    assert first["total_variant_count"] == 30
    assert first["release_eligible"] is False
    assert first["selected_case_ids"] == "OMITTED"
    assert first["absolute_paths"] == "OMITTED"
    assert first["defects_sha256"] == second["defects_sha256"]
    assert first["negative_controls_sha256"] == second["negative_controls_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert {
        path.relative_to(tmp_path / "bundle-one").as_posix()
        for path in (tmp_path / "bundle-one").rglob("*")
        if path.is_file()
    } == {
        DEFECTS_RELATIVE_PATH.as_posix(),
        NEGATIVE_CONTROLS_RELATIVE_PATH.as_posix(),
        MANIFEST_RELATIVE_PATH.as_posix(),
        CHECKSUMS_RELATIVE_PATH.as_posix(),
    }
    assert {
        path.name: _sha256(path) for path in (source["dataset_root"] / "raw").iterdir()
    } == raw_hashes_before
    assert not (source["dataset_root"] / "derived" / "splits.json").exists()


def test_builder_rejects_existing_output_without_modifying_it(tmp_path):
    source = _registered_dataset(tmp_path)
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(RuleCoverageBuildError, match="already exists"):
        build_rule_coverage_bundle(
            source["dataset_root"],
            output,
            descriptor_path=source["descriptor_path"],
            dataset_lock_path=source["dataset_lock_path"],
        )

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert list(output.iterdir()) == [marker]


def test_builder_rejects_raw_checksum_drift(tmp_path):
    source = _registered_dataset(tmp_path)
    raw_path = source["dataset_root"] / "raw" / "PL_100_Final.json"
    raw_path.write_bytes(raw_path.read_bytes() + b"tampered")

    with pytest.raises(
        RuleCoverageBuildError,
        match="raw file hash",
    ):
        build_rule_coverage_bundle(
            source["dataset_root"],
            tmp_path / "must-not-exist",
            descriptor_path=source["descriptor_path"],
            dataset_lock_path=source["dataset_lock_path"],
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_builder_rejects_release_claim_descriptor(tmp_path):
    source = _registered_dataset(tmp_path)
    descriptor = _read_json(source["descriptor_path"])
    descriptor["policy"]["release_claim_allowed"] = True
    _write_json(source["descriptor_path"], descriptor)

    with pytest.raises(
        RuleCoverageBuildError,
        match="descriptor contract",
    ):
        build_rule_coverage_bundle(
            source["dataset_root"],
            tmp_path / "must-not-exist",
            descriptor_path=source["descriptor_path"],
            dataset_lock_path=source["dataset_lock_path"],
        )

    assert not (tmp_path / "must-not-exist").exists()
