import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.inventory_trade_images import DatasetValidationError, build_inventory


FIELDNAMES = [
    "index",
    "shipment_id",
    "invoice_no",
    "bl_no",
    "consignee",
    "num_items",
    "ci_file",
    "pl_file",
    "bl_file",
]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 48), color).save(path, format="JPEG", quality=90)


def _row(index: int) -> dict[str, str]:
    suffix = f"{index:04d}"
    return {
        "index": str(index),
        "shipment_id": f"SHIP-SECRET-{suffix}",
        "invoice_no": f"INV-SECRET-{suffix}",
        "bl_no": f"BL-SECRET-{suffix}",
        "consignee": f"PRIVATE CONSIGNEE {suffix}",
        "num_items": "1",
        "ci_file": f"commercial_invoice/CI_{suffix}.jpg",
        "pl_file": f"packing_list/PL_{suffix}.jpg",
        "bl_file": f"bill_of_lading/BL_{suffix}.jpg",
    }


def _create_dataset(root: Path, *, shipments: int = 2) -> list[dict[str, str]]:
    rows = [_row(index) for index in range(1, shipments + 1)]
    color_seed = 1
    for row in rows:
        for column in ("ci_file", "pl_file", "bl_file"):
            _write_image(
                root / row[column],
                (color_seed, color_seed * 2, color_seed * 3),
            )
            color_seed += 1

    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _rewrite_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_build_inventory_writes_privacy_safe_deterministic_evidence(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _create_dataset(dataset_root)
    inventory_path = tmp_path / "evidence" / "inventory.json"
    checksums_path = tmp_path / "evidence" / "checksums.sha256"

    inventory = build_inventory(dataset_root, inventory_path, checksums_path)

    assert inventory["summary"] == {
        "shipment_count": 2,
        "image_count": 6,
        "document_type_counts": {
            "bill_of_lading": 2,
            "commercial_invoice": 2,
            "packing_list": 2,
        },
        "media_type_counts": {"image/jpeg": 6},
        "dimension_counts": [{"width": 32, "height": 48, "count": 6}],
        "total_bytes": sum(
            (dataset_root / record["relative_path"]).stat().st_size
            for record in inventory["images"]
        ),
        "duplicate_sha256_groups": 0,
        "unreferenced_image_count": 0,
        "contains_image_content": False,
        "contains_manifest_business_values": False,
    }
    stored = inventory_path.read_text(encoding="utf-8")
    assert json.loads(stored) == inventory
    assert "SHIP-SECRET" not in stored
    assert "INV-SECRET" not in stored
    assert "PRIVATE CONSIGNEE" not in stored
    checksum_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 7
    assert checksum_lines[0].endswith("  manifest.csv")
    assert all("SECRET" not in line for line in checksum_lines)


def test_manifest_basenames_resolve_inside_document_type_directories(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rows = _create_dataset(dataset_root, shipments=1)
    for row in rows:
        for column in ("ci_file", "pl_file", "bl_file"):
            row[column] = Path(row[column]).name
    _rewrite_manifest(dataset_root, rows)

    inventory = build_inventory(
        dataset_root,
        tmp_path / "inventory.json",
        tmp_path / "checksums.sha256",
    )

    assert {item["relative_path"] for item in inventory["images"]} == {
        "commercial_invoice/CI_0001.jpg",
        "packing_list/PL_0001.jpg",
        "bill_of_lading/BL_0001.jpg",
    }


def test_duplicate_shipment_id_is_rejected(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    rows = _create_dataset(dataset_root)
    rows[1]["shipment_id"] = rows[0]["shipment_id"]
    _rewrite_manifest(dataset_root, rows)

    with pytest.raises(DatasetValidationError, match="duplicate shipment_id"):
        build_inventory(
            dataset_root,
            tmp_path / "inventory.json",
            tmp_path / "checksums.sha256",
        )


def test_missing_referenced_image_is_rejected(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _create_dataset(dataset_root)
    (dataset_root / "packing_list" / "PL_0001.jpg").unlink()

    with pytest.raises(DatasetValidationError, match="referenced image is missing"):
        build_inventory(
            dataset_root,
            tmp_path / "inventory.json",
            tmp_path / "checksums.sha256",
        )


def test_duplicate_image_content_is_rejected(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _create_dataset(dataset_root)
    source = dataset_root / "commercial_invoice" / "CI_0001.jpg"
    duplicate = dataset_root / "packing_list" / "PL_0001.jpg"
    duplicate.write_bytes(source.read_bytes())

    with pytest.raises(DatasetValidationError, match="duplicate image content"):
        build_inventory(
            dataset_root,
            tmp_path / "inventory.json",
            tmp_path / "checksums.sha256",
        )


def test_corrupt_image_is_rejected(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _create_dataset(dataset_root)
    (dataset_root / "bill_of_lading" / "BL_0001.jpg").write_bytes(b"not-a-jpeg")

    with pytest.raises(DatasetValidationError, match="cannot decode image"):
        build_inventory(
            dataset_root,
            tmp_path / "inventory.json",
            tmp_path / "checksums.sha256",
        )


def test_unreferenced_image_is_rejected(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _create_dataset(dataset_root)
    _write_image(dataset_root / "commercial_invoice" / "EXTRA.jpg", (200, 201, 202))

    with pytest.raises(DatasetValidationError, match="unreferenced image"):
        build_inventory(
            dataset_root,
            tmp_path / "inventory.json",
            tmp_path / "checksums.sha256",
        )


def test_invalidated_suffix_annotations_cannot_reenter_partial_gold():
    repository_root = Path(__file__).parents[1]
    partial_gold = json.loads(
        (
            repository_root
            / "evals"
            / "trade_documents_images"
            / "v1"
            / "manifest_partial_gold.json"
        ).read_text(encoding="utf-8")
    )
    dataset_lock = json.loads(
        (
            repository_root / "fixtures" / "trade_documents_images" / "v1" / "dataset.lock.json"
        ).read_text(encoding="utf-8")
    )

    case_0500 = partial_gold["cases"][0]
    packing_list = next(
        document
        for document in case_0500["documents"]
        if document["document_id"] == "0500-packing-list"
    )
    assert packing_list["observed_candidate_fields"]["bl_no"] == "ONEU26931026"
    assert not packing_list["observed_candidate_fields"]["bl_no"].endswith("EE")
    assert partial_gold["annotation_status"] == "invalidated_by_label_error"
    assert partial_gold["gold_eligibility"] == "invalidated_by_label_error"
    assert partial_gold["release_scoring_allowed"] is False
    assert "gold_set_id" not in partial_gold
    assert partial_gold["required_independent_review_count"] == 2
    assert case_0500["annotation_status"] == "invalidated_by_label_error"
    assert case_0500["gold_eligible"] is False
    assert dataset_lock["known_source_anomalies"] == []
    assert (
        dataset_lock["ground_truth"]["gold_gate_status"]
        == "bounded_12_case_gold_passed_full_corpus_candidate_unverified"
    )
    assert (
        dataset_lock["ground_truth"]["single_reviewer_candidate_status"]
        == "invalidated_by_label_error"
    )
    assert dataset_lock["ground_truth"]["release_scoring_allowed"] is False
    assert {
        correction["case_id"] for correction in dataset_lock["invalidated_annotation_claims"]
    } == {"0001", "0041", "0500", "1000"}
    assert all(
        correction["review_status"] == "pending_second_review"
        and correction["completed_independent_review_count"] == 1
        and correction["gold_eligible"] is False
        for correction in dataset_lock["invalidated_annotation_claims"]
    )


def test_restored_source_registration_is_canonical_only_and_scope_limited():
    repository_root = Path(__file__).parents[1]
    dataset_lock = json.loads(
        (
            repository_root / "fixtures" / "trade_documents_images" / "v1" / "dataset.lock.json"
        ).read_text(encoding="utf-8")
    )

    assert dataset_lock["lock_revision"] == 2
    storage = dataset_lock["source_storage"]
    assert (
        storage["canonical_workspace_relative_path"]
        == "_artifacts/datasets/trade_documents_images/source-v1/raw"
    )
    assert storage["runner_access_policy"] == "canonical_only"
    assert storage["root_reference_workspace_relative_path"] == "trade_documents_images"
    assert storage["root_reference_preserved"] is True
    assert storage["root_reference_runner_eligible"] is False
    assert storage["fallback_to_root_allowed"] is False

    integrity = dataset_lock["source_integrity"]
    assert integrity["source_file_count"] == 3002
    assert integrity["source_bytes"] == 260248817
    assert integrity["image_count"] == 3000
    assert integrity["image_bytes"] == 260132937
    assert integrity["manifest_bytes"] == 114955
    assert integrity["source_readme_bytes"] == 925
    assert (
        integrity["manifest_sha256"]
        == "a1ebe0e18953b6712f044ee4d7dd3ede8ad426ae73ed3158dd0eab38e2111965"
    )
    assert (
        integrity["source_readme_sha256"]
        == "80edd388bd40c68b6c1c4cae6ddb6b9538502bc2a85d801707844ba1e6d591f0"
    )
    assert integrity["root_to_canonical_parity"] == {
        "status": "byte_for_byte_identical",
        "comparison_scope": "all_relative_paths_sizes_and_sha256",
        "source_deleted": False,
        "verification_pass_count": 2,
        "mismatch_count": 0,
    }
    assert integrity["checksums_coverage"] == {
        "pinned_entry_count": 3001,
        "includes_images": True,
        "includes_manifest": True,
        "includes_source_readme": False,
        "source_readme_verified_separately": True,
    }

    provenance = dataset_lock["provenance"]
    assert provenance["public_redistribution_rights"] == "not_confirmed"
    assert provenance["raw_allowed_in_git_or_public_portfolio"] is False

    ground_truth = dataset_lock["ground_truth"]
    assert ground_truth["full_corpus_status"] == "candidate_unverified"
    assert ground_truth["referenced_master_file"] == "shipments_master.json"
    assert ground_truth["referenced_master_present"] is False
    assert ground_truth["full_field_gold_available"] is False
    bounded_gold = ground_truth["bounded_adjudicated_gold"]
    assert bounded_gold["status"] == "passed_two_reviewer_adjudication"
    assert bounded_gold["case_count"] == 12
    assert bounded_gold["document_count"] == 36
    assert bounded_gold["reviewer_kind"] == "independent_ai_visual_review"
    assert bounded_gold["human_review_completed"] is False
    assert bounded_gold["release_scoring_allowed_within_scope"] is True
    assert bounded_gold["scoring_scope"] == "claude_vision_field_extraction_only"

    discrepancies = dataset_lock["known_source_observation_discrepancies"]
    assert discrepancies["source_matches_manifest_case_count"] == 6
    assert discrepancies["source_discrepancy_case_count"] == 6
    assert discrepancies["business_defect_status"] == "ambiguous_not_scored_as_defect"
    assert discrepancies["automatic_source_correction_performed"] is False
