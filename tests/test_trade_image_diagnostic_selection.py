import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.build_trade_image_diagnostic_selection import (
    ADJUDICATION_SCHEMA,
    DiagnosticSelectionError,
    SELECTION_COUNT,
    SELECTION_ID,
    build_worklist,
    derive_adjudication_status,
    evaluate_gate,
    load_or_initialize_worklist,
    select_hash_ranked_cases,
)


def _manifest_and_inventory(case_count: int = 15):
    manifest = []
    inventory = {}
    for index in range(1, case_count + 1):
        case_id = f"{index:04d}"
        row = {
            "index": case_id,
            "ci_file": f"CI_{case_id}.jpg",
            "pl_file": f"PL_{case_id}.jpg",
            "bl_file": f"BL_{case_id}.jpg",
        }
        manifest.append(row)
        for ordinal, (directory, filename) in enumerate(
            (
                ("commercial_invoice", row["ci_file"]),
                ("packing_list", row["pl_file"]),
                ("bill_of_lading", row["bl_file"]),
            ),
            start=1,
        ):
            relative_path = f"{directory}/{filename}"
            inventory[relative_path] = {
                "relative_path": relative_path,
                "sha256": f"{index * 3 + ordinal:064x}",
                "byte_size": 1000 + index,
                "width": 1000,
                "height": 1414,
            }
    return manifest, inventory


def _review(
    selected_case,
    reviewer_id: str,
    conclusion: str = "source_matches_manifest_observation",
):
    return {
        "review_id": f"{selected_case['case_id']}-{reviewer_id}",
        "reviewer_id": reviewer_id,
        "independent_review_attested": True,
        "scope": "diagnostic_full_triplet",
        "conclusion": conclusion,
        "reviewed_document_sha256": [
            document["sha256"] for document in selected_case["documents"].values()
        ],
        "annotation_reference": f"reviews/{selected_case['case_id']}/{reviewer_id}",
    }


def _schema():
    return json.loads(
        (
            Path(__file__).parents[1]
            / "evals"
            / "trade_documents_images"
            / "v1"
            / "schemas"
            / "diagnostic-adjudication.schema.json"
        ).read_text(encoding="utf-8")
    )


def test_hash_ranked_selection_is_deterministic_and_keeps_triplets_together():
    manifest, inventory = _manifest_and_inventory()

    first = select_hash_ranked_cases(manifest, inventory)
    second = select_hash_ranked_cases(list(reversed(manifest)), inventory)

    assert first == second
    assert len(first) == SELECTION_COUNT
    assert len({case["case_id"] for case in first}) == SELECTION_COUNT
    assert all(
        set(case["documents"])
        == {
            "commercial_invoice",
            "packing_list",
            "bill_of_lading",
        }
        for case in first
    )


def test_pending_worklist_is_schema_valid_and_blocks_gold():
    manifest, inventory = _manifest_and_inventory()
    selected = select_hash_ranked_cases(manifest, inventory)
    worklist = build_worklist(selected)

    validator = Draft202012Validator(_schema())
    for entry in worklist:
        validator.validate(entry)
    gate = evaluate_gate(selected, worklist)

    assert gate["gold_eligible"] is False
    assert gate["status"] == "blocked_pending_independent_reviews"
    assert gate["status_counts"] == {"pending_first_review": SELECTION_COUNT}


def test_one_review_remains_pending_and_cannot_be_declared_gold():
    manifest, inventory = _manifest_and_inventory()
    selected = select_hash_ranked_cases(manifest, inventory)
    worklist = build_worklist(selected)
    selected_case = selected[0]
    entry = next(item for item in worklist if item["case_id"] == selected_case["case_id"])
    entry["reviews"] = [_review(selected_case, "reviewer-a")]
    entry["status"] = "pending_second_review"

    Draft202012Validator(_schema()).validate(entry)
    gate = evaluate_gate(selected, worklist)

    assert gate["gold_eligible"] is False
    assert any(
        blocker
        == {
            "case_id": selected_case["case_id"],
            "status": "pending_second_review",
        }
        for blocker in gate["blockers"]
    )


def test_two_matching_reviews_require_explicit_consensus_then_pass():
    manifest, inventory = _manifest_and_inventory()
    selected = select_hash_ranked_cases(manifest, inventory)
    worklist = build_worklist(selected)
    for entry in worklist:
        selected_case = next(case for case in selected if case["case_id"] == entry["case_id"])
        entry["reviews"] = [
            _review(selected_case, "reviewer-a"),
            _review(selected_case, "reviewer-b"),
        ]
        entry["status"] = "consensus_requires_resolution"
        assert derive_adjudication_status(entry) == "consensus_requires_resolution"
        entry["resolution"] = {
            "method": "two_reviewer_consensus",
            "final_conclusion": "source_matches_manifest_observation",
            "reference": f"resolutions/{entry['case_id']}",
        }
        entry["status"] = "adjudicated"
        entry["gold_eligible"] = True
        Draft202012Validator(_schema()).validate(entry)

    gate = evaluate_gate(selected, worklist)

    assert gate["gold_eligible"] is True
    assert gate["status"] == "passed_two_reviewer_adjudication"
    assert gate["status_counts"] == {"adjudicated": SELECTION_COUNT}


def test_disagreement_and_duplicate_reviewer_are_fail_closed():
    manifest, inventory = _manifest_and_inventory()
    selected = select_hash_ranked_cases(manifest, inventory)
    worklist = build_worklist(selected)
    selected_case = selected[0]
    entry = next(item for item in worklist if item["case_id"] == selected_case["case_id"])
    entry["reviews"] = [
        _review(selected_case, "reviewer-a"),
        _review(selected_case, "reviewer-b", "source_discrepancy_confirmed"),
    ]
    entry["status"] = "disagreement_requires_adjudicator"

    assert derive_adjudication_status(entry) == "disagreement_requires_adjudicator"
    assert evaluate_gate(selected, worklist)["gold_eligible"] is False

    duplicate = copy.deepcopy(entry)
    duplicate["reviews"][1]["reviewer_id"] = "reviewer-a"
    with pytest.raises(DiagnosticSelectionError, match="duplicate reviewer"):
        derive_adjudication_status(duplicate)


def test_disagreement_requires_a_distinct_independent_adjudicator():
    manifest, inventory = _manifest_and_inventory()
    selected = select_hash_ranked_cases(manifest, inventory)
    entry = build_worklist(selected)[0]
    entry["reviews"] = [
        _review(selected[0], "reviewer-a"),
        _review(selected[0], "reviewer-b", "source_discrepancy_confirmed"),
    ]
    entry["resolution"] = {
        "method": "independent_adjudicator",
        "adjudicator_id": "adjudicator-c",
        "independent_adjudicator_attested": True,
        "final_conclusion": "unreadable_or_insufficient",
        "reference": "resolutions/independent-0001",
    }
    entry["status"] = "excluded"

    assert derive_adjudication_status(entry) == "excluded"
    Draft202012Validator(_schema()).validate(entry)

    entry["resolution"]["adjudicator_id"] = "reviewer-a"
    with pytest.raises(DiagnosticSelectionError, match="distinct independent"):
        derive_adjudication_status(entry)


def test_review_cannot_reference_image_outside_selected_triplet():
    manifest, inventory = _manifest_and_inventory()
    selected = select_hash_ranked_cases(manifest, inventory)
    worklist = build_worklist(selected)
    entry = worklist[0]
    entry["reviews"] = [_review(selected[0], "reviewer-a")]
    entry["reviews"][0]["reviewed_document_sha256"] = ["f" * 64]
    entry["status"] = "pending_second_review"

    with pytest.raises(DiagnosticSelectionError, match="cover the selected triplet"):
        evaluate_gate(selected, worklist)


def test_existing_human_worklist_is_loaded_without_overwrite(tmp_path):
    manifest, inventory = _manifest_and_inventory()
    selected = select_hash_ranked_cases(manifest, inventory)
    initial = build_worklist(selected)
    existing = copy.deepcopy(initial)
    existing[0]["reviews"] = [_review(selected[0], "reviewer-a")]
    existing[0]["status"] = "pending_second_review"
    payload = (
        "\n".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in existing)
        + "\n"
    )
    worklist_path = tmp_path / "adjudications.jsonl"
    worklist_path.write_text(payload, encoding="utf-8")

    loaded = load_or_initialize_worklist(worklist_path, initial)

    assert loaded == existing
    assert worklist_path.read_text(encoding="utf-8") == payload


def test_schema_version_is_pinned():
    schema = _schema()
    assert schema["properties"]["schema_version"]["const"] == ADJUDICATION_SCHEMA
    assert SELECTION_ID
