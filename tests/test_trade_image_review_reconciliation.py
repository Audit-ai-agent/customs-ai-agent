import hashlib
import json
from pathlib import Path

import pytest

from customs_ai_agent.evaluation_assets import image_gold_semantic_fingerprint
from evals import run_claude_vision
from tools import build_trade_image_adjudicated_gold as reconciliation
from tools.build_trade_image_adjudicated_gold import (
    DEFAULT_DIAGNOSTIC_SCHEMA_PATH,
    DEFAULT_GOLD_SCHEMA_PATH,
    ReviewReconciliationError,
    build_reconciliation_bundle,
)


DOCUMENT_ORDER = (
    "commercial_invoice",
    "packing_list",
    "bill_of_lading",
)


def _digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _selection():
    cases = []
    for index in range(12):
        case_id = f"{index:04d}"
        documents = {
            document_type: {
                "relative_path": f"{document_type}/SYNTHETIC_{case_id}.png",
                "sha256": _digest(f"{case_id}:{document_type}"),
            }
            for document_type in DOCUMENT_ORDER
        }
        cases.append(
            {
                "case_id": case_id,
                "triplet_fingerprint_sha256": _digest(f"{case_id}:triplet"),
                "documents": documents,
            }
        )
    return {
        "schema_version": "trade-image-diagnostic-selection/v1",
        "selection_id": "trade-images-hash-ranked-diagnostic-v1",
        "selection_role": "diagnostic_candidate_not_gold",
        "gold_eligible": False,
        "cases": cases,
    }


def _observed_values(case_id, *, discrepancy):
    invoice_no = f"SYN-INV-{case_id}"
    bl_no = f"SYN-BL-{case_id}"
    packing_bl_no = f"{bl_no}-INTENTIONAL-DIFFERENCE" if discrepancy else bl_no
    buyer = f"SYNTHETIC BUYER {case_id}"
    return {
        "invoice_no": invoice_no,
        "bl_no": bl_no,
        "packing_bl_no": packing_bl_no,
        "buyer": buyer,
    }


def _review_a(selection):
    records = []
    for index, selected in enumerate(selection["cases"]):
        case_id = selected["case_id"]
        values = _observed_values(case_id, discrepancy=index < 2)
        records.append(
            {
                "case_id": case_id,
                "reviewer_id": "synthetic-ai-reviewer-a",
                "reviewer_kind": "independent_ai_visual_review",
                "independent_review_attested": True,
                "scope": "diagnostic_full_triplet",
                "reviewed_document_sha256": [
                    selected["documents"][document_type]["sha256"]
                    for document_type in DOCUMENT_ORDER
                ],
                "conclusion": (
                    "source_discrepancy_confirmed"
                    if index < 2
                    else "source_matches_manifest_observation"
                ),
                "documents": [
                    {
                        "source_document_type": "commercial_invoice",
                        "relative_path": selected["documents"]["commercial_invoice"][
                            "relative_path"
                        ],
                        "source_image_sha256": selected["documents"]["commercial_invoice"][
                            "sha256"
                        ],
                        "readability": "readable",
                        "observed_document_type": "commercial_invoice",
                        "observed_document_type_text": "COMMERCIAL INVOICE",
                        "observed_fields": {
                            "invoice_no": values["invoice_no"],
                            "buyer_consignee": values["buyer"],
                        },
                        "visible_line_item_count": 1,
                    },
                    {
                        "source_document_type": "packing_list",
                        "relative_path": selected["documents"]["packing_list"]["relative_path"],
                        "source_image_sha256": selected["documents"]["packing_list"]["sha256"],
                        "readability": "readable",
                        "observed_document_type": "packing_list",
                        "observed_document_type_text": "PACKING LIST",
                        "observed_fields": {
                            "invoice_no": values["invoice_no"],
                            "bl_no": values["packing_bl_no"],
                        },
                        "visible_line_item_count": 1,
                    },
                    {
                        "source_document_type": "bill_of_lading",
                        "relative_path": selected["documents"]["bill_of_lading"]["relative_path"],
                        "source_image_sha256": selected["documents"]["bill_of_lading"]["sha256"],
                        "readability": "readable",
                        "observed_document_type": "bill_of_lading",
                        "observed_document_type_text": "BILL OF LADING",
                        "observed_fields": {
                            "bl_no": values["bl_no"],
                            "consignee": values["buyer"],
                        },
                        "visible_line_item_count": 0,
                    },
                ],
                "limitations": ["SYNTHETIC FIXTURE"],
                "notes": "SYNTHETIC FIXTURE",
            }
        )
    return records


def _review_b(selection):
    records = []
    for index, selected in enumerate(selection["cases"]):
        case_id = selected["case_id"]
        values = _observed_values(case_id, discrepancy=index < 2)
        records.append(
            {
                "case_id": case_id,
                "reviewer_id": "synthetic-ai-reviewer-b",
                "reviewer_kind": "independent_ai_visual_review",
                "independent_review_attested": True,
                "scope": "diagnostic_full_triplet",
                "reviewed_document_sha256": [
                    selected["documents"][document_type]["sha256"]
                    for document_type in reversed(DOCUMENT_ORDER)
                ],
                "conclusion": (
                    "source_discrepancy_confirmed"
                    if index < 2
                    else "source_matches_manifest_observation"
                ),
                "documents": {
                    "commercial_invoice": {
                        "sha256": selected["documents"]["commercial_invoice"]["sha256"],
                        "readability": "READABLE",
                        "document_type": "commercial invoice",
                        "invoice_no": values["invoice_no"],
                        "buyer_or_consignee": f"  {values['buyer']}  ",
                        "visible_line_item_count": 1,
                    },
                    "packing_list": {
                        "sha256": selected["documents"]["packing_list"]["sha256"],
                        "readability": "readable",
                        "document_type": "PACKING LIST",
                        "invoice_no": values["invoice_no"],
                        "bl_no": values["packing_bl_no"],
                        "visible_line_item_count": 1,
                    },
                    "bill_of_lading": {
                        "sha256": selected["documents"]["bill_of_lading"]["sha256"],
                        "readability": "readable",
                        "document_type": "BILL OF LADING",
                        "bl_no": values["bl_no"],
                        "consignee": values["buyer"],
                        "visible_line_item_count": 0,
                    },
                },
                "limitations": ["SYNTHETIC FIXTURE"],
                "notes": ["SYNTHETIC FIXTURE"],
            }
        )
    return records


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _fixture_files(tmp_path, *, mutate=None):
    selection = _selection()
    review_a = _review_a(selection)
    review_b = _review_b(selection)
    if mutate is not None:
        mutate(review_a, review_b)

    repository = tmp_path / "synthetic-repository"
    repository.mkdir()
    selection_path = repository / "selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_root = tmp_path / "_artifacts"
    review_a_path = artifact_root / "reviews" / "a.jsonl"
    review_b_path = artifact_root / "reviews" / "b.jsonl"
    _write_jsonl(review_a_path, review_a)
    _write_jsonl(review_b_path, review_b)
    return {
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "artifact_root": artifact_root,
        "review_paths": [review_a_path, review_b_path],
        "output_dir": artifact_root / "derived" / "adjudicated-gold" / "v1",
    }


def _build(files, **overrides):
    arguments = {
        "selection_path": files["selection_path"],
        "review_paths": files["review_paths"],
        "artifact_root": files["artifact_root"],
        "output_dir": files["output_dir"],
        "expected_selection_sha256": files["selection_sha256"],
    }
    arguments.update(overrides)
    return build_reconciliation_bundle(**arguments)


def test_reconciles_two_ai_review_layouts_and_preserves_intentional_discrepancy(
    tmp_path,
):
    files = _fixture_files(tmp_path)

    bundle = _build(files)

    assert bundle["gate"]["status"] == "passed_two_reviewer_adjudication"
    assert bundle["gate"]["status_counts"] == {"adjudicated": 12}
    assert bundle["report"]["conclusion_counts"] == {
        "source_discrepancy_confirmed": 2,
        "source_matches_manifest_observation": 10,
    }
    assert bundle["report"]["data_policy"] == {
        "gold_contains_observed_values": True,
        "report_contains_observed_values": False,
        "automatic_value_correction_performed": False,
    }
    first_case = bundle["gold"]["cases"][0]
    documents = {document["source_document_type"]: document for document in first_case["documents"]}
    assert documents["commercial_invoice"]["expected_fields"] == {
        "buyer": "SYNTHETIC BUYER 0000",
        "invoice_no": "SYN-INV-0000",
    }
    assert (
        documents["packing_list"]["expected_fields"]["bl_no"]
        == "SYN-BL-0000-INTENTIONAL-DIFFERENCE"
    )
    assert documents["bill_of_lading"]["expected_fields"]["bl_no"] == "SYN-BL-0000"
    assert "SYN-INV" not in json.dumps(bundle["report"])
    assert all(path.is_relative_to(files["artifact_root"]) for path in bundle["paths"].values())

    worklist = [
        json.loads(line)
        for line in bundle["paths"]["worklist"].read_text(encoding="utf-8").splitlines()
    ]
    assert (
        run_claude_vision._validate_gold_gate(
            selection=files["selection"],
            worklist=worklist,
            gate=bundle["gate"],
        )
        == []
    )
    assert (
        run_claude_vision._validate_gold_manifest(
            gold=bundle["gold"],
            selection=files["selection"],
            worklist=worklist,
            selection_path=files["selection_path"],
            worklist_path=bundle["paths"]["worklist"],
            gate_path=bundle["paths"]["gate"],
            schema_path=DEFAULT_GOLD_SCHEMA_PATH,
        )
        == []
    )

    repeated = _build(files)
    assert repeated["sha256"] == bundle["sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda review_a, review_b: review_b[0].update(reviewer_id=review_a[0]["reviewer_id"]),
            "distinct reviewer IDs",
        ),
        (
            lambda _review_a, review_b: review_b[0].update(reviewer_kind="visual_review"),
            "reviewer kind",
        ),
        (
            lambda _review_a, review_b: review_b[0].update(independent_review_attested=False),
            "independence attestation",
        ),
        (
            lambda _review_a, review_b: review_b[0].update(scope="partial_triplet"),
            "review scope",
        ),
        (
            lambda _review_a, review_b: review_b[0]["reviewed_document_sha256"].__setitem__(
                0, "0" * 64
            ),
            "hashes do not match",
        ),
        (
            lambda _review_a, review_b: review_b[0].update(conclusion="invalid_prior_claim"),
            "conclusions disagree",
        ),
        (
            lambda _review_a, review_b: review_b[0]["documents"]["packing_list"].update(
                bl_no="SYN-MISMATCH"
            ),
            "normalized observations disagree",
        ),
    ],
)
def test_reconciliation_fails_closed_on_invalid_or_disagreeing_reviews(
    tmp_path,
    mutation,
    message,
):
    files = _fixture_files(tmp_path, mutate=mutation)

    with pytest.raises(ReviewReconciliationError, match=message):
        _build(files)


def test_known_invalidated_output_hash_cannot_be_rebuilt(
    monkeypatch,
    tmp_path,
):
    files = _fixture_files(tmp_path)
    monkeypatch.setattr(
        reconciliation,
        "image_gold_invalidation_code",
        lambda _digest: "adjudicated_gold_invalidated_by_label_error",
    )

    with pytest.raises(ReviewReconciliationError, match="explicitly invalidated"):
        _build(files)


def test_semantically_invalid_gold_cannot_bypass_with_output_dir_or_gold_id(
    monkeypatch,
    tmp_path,
):
    files = _fixture_files(tmp_path)
    original = _build(files)
    invalid_fingerprint = image_gold_semantic_fingerprint(original["gold"])
    assert invalid_fingerprint is not None

    def semantic_invalidation(candidate):
        if image_gold_semantic_fingerprint(candidate) == invalid_fingerprint:
            return "adjudicated_gold_invalidated_by_label_error"
        return None

    monkeypatch.setattr(
        reconciliation,
        "image_gold_semantic_invalidation_code",
        semantic_invalidation,
    )
    files["output_dir"] = files["artifact_root"] / "different-output-root" / "adjudicated-gold"

    with pytest.raises(ReviewReconciliationError, match="explicitly invalidated"):
        _build(files, gold_set_id="renamed-gold-set")


def test_unreadable_reviews_cannot_create_value_bearing_gold(tmp_path):
    def mark_unreadable(review_a, review_b):
        review_a[0]["conclusion"] = "unreadable_or_insufficient"
        review_b[0]["conclusion"] = "unreadable_or_insufficient"
        for document in review_a[0]["documents"]:
            document["readability"] = "unreadable"
        for document in review_b[0]["documents"].values():
            document["readability"] = "unreadable"

    files = _fixture_files(tmp_path, mutate=mark_unreadable)

    with pytest.raises(ReviewReconciliationError, match="not readable enough"):
        _build(files)


def test_selection_digest_is_pinned(tmp_path):
    files = _fixture_files(tmp_path)

    with pytest.raises(ReviewReconciliationError, match="selection SHA-256"):
        _build(files, expected_selection_sha256="0" * 64)


def test_value_bearing_gold_cannot_be_written_outside_artifacts(tmp_path):
    files = _fixture_files(tmp_path)
    outside_output = tmp_path / "synthetic-repository" / "generated-gold"

    with pytest.raises(ReviewReconciliationError, match="must remain under _artifacts"):
        _build(files, output_dir=outside_output)


def test_both_adjudication_schemas_are_valid_json_schema():
    for path in (DEFAULT_DIAGNOSTIC_SCHEMA_PATH, DEFAULT_GOLD_SCHEMA_PATH):
        assert Path(path).is_file()
