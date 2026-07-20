import json

from customs_ai_agent.evaluation_assets import (
    INVALIDATED_IMAGE_GOLD,
    INVALIDATED_IMAGE_GOLD_SEMANTICS,
    image_gold_invalidation_code,
    image_gold_semantic_fingerprint,
    image_gold_semantic_invalidation_code,
)


def test_invalidated_image_gold_registry_is_exact_hash_bound() -> None:
    invalidated = "75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6"

    assert (
        image_gold_invalidation_code(invalidated) == "adjudicated_gold_invalidated_by_label_error"
    )
    assert (
        image_gold_invalidation_code(invalidated.upper())
        == "adjudicated_gold_invalidated_by_label_error"
    )
    assert image_gold_invalidation_code("0" * 64) is None
    assert tuple(INVALIDATED_IMAGE_GOLD) == (invalidated,)


def _semantic_gold_fixture() -> dict:
    return {
        "schema_version": "trade-image-adjudicated-gold/v1",
        "gold_set_id": "metadata-is-not-semantic",
        "adjudication_worklist_sha256": "1" * 64,
        "adjudication_gate_sha256": "2" * 64,
        "cases": [
            {
                "case_id": "0001",
                "triplet_fingerprint_sha256": "3" * 64,
                "annotation_references": ["output/a", "output/b"],
                "documents": [
                    {
                        "document_id": "0001:packing_list",
                        "source_document_type": "packing_list",
                        "relative_path": "first/location.png",
                        "source_image_sha256": "4" * 64,
                        "expected_document_type": "packing_list",
                        "expected_fields": {
                            "invoice_no": "INV-1",
                            "bl_no": "B/L-1",
                        },
                        "expected_line_item_count": 2,
                    }
                ],
            }
        ],
    }


def test_semantic_fingerprint_ignores_json_format_and_nonvalue_metadata() -> None:
    original = _semantic_gold_fixture()
    reserialized = json.loads(
        json.dumps(
            original,
            ensure_ascii=False,
            indent=7,
            sort_keys=False,
        )
    )
    reserialized["gold_set_id"] = "renamed-gold"
    reserialized["adjudication_worklist_sha256"] = "5" * 64
    reserialized["adjudication_gate_sha256"] = "6" * 64
    case = reserialized["cases"][0]
    case["annotation_references"] = ["different/root/a", "different/root/b"]
    case["documents"][0]["relative_path"] = "different/source/location.png"

    original_fingerprint = image_gold_semantic_fingerprint(original)
    reserialized_fingerprint = image_gold_semantic_fingerprint(reserialized)

    assert original_fingerprint is not None
    assert reserialized_fingerprint == original_fingerprint
    assert image_gold_semantic_invalidation_code(original) is None

    reserialized["cases"][0]["documents"][0]["expected_fields"]["bl_no"] = "CHANGED"
    assert image_gold_semantic_fingerprint(reserialized) != original_fingerprint


def test_invalidated_semantic_registry_is_hash_bound() -> None:
    invalidated = "ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133"

    assert tuple(INVALIDATED_IMAGE_GOLD_SEMANTICS) == (invalidated,)
