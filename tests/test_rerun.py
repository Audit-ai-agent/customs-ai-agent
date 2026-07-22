import pytest

from customs_ai_agent.runner import (
    CorrectionApplicationError,
    rerun_review,
    run_review,
    to_review_run_result_v2,
)
from customs_ai_agent.schemas.models import ReviewRunRequest, UserCorrection


def test_rerun_applies_user_correction_and_revalidates():
    state = run_review(ReviewRunRequest(scenario_id="invoice_total_mismatch"))
    assert state["final_status"] == "needs_review"

    corrected = rerun_review(
        state,
        [
            UserCorrection(
                document_id="doc_inv",
                field_path="fields.total_amount",
                value="12000",
            )
        ],
    )

    assert corrected["validation_issues"] == []
    assert corrected["final_status"] == "completed"
    assert corrected["extracted_documents"]["doc_inv"]["fields"]["total_amount"]["value"] == (
        "12000"
    )
    assert corrected["normalized_documents"]["doc_inv"]["raw_fields"]["total_amount"] == ("12000")
    assert corrected["audit_trail"][-1]["event"] == "corrections_applied"


@pytest.mark.parametrize(
    ("document_id", "field_path", "value", "error_code"),
    [
        ("missing", "fields.total_amount", "12000", "correction_document_missing"),
        ("doc_inv", "unsupported.total_amount", "12000", "correction_path_invalid"),
        ("doc_inv", "fields.missing", "12000", "correction_field_missing"),
        ("doc_inv", "line_items.9.quantity", "1200 KG", "correction_line_item_missing"),
        ("doc_inv", "line_items.0.missing", "1200 KG", "correction_field_missing"),
        ("doc_inv", "fields.total_amount", 12000, "correction_value_type_mismatch"),
    ],
)
def test_rerun_fails_closed_when_correction_cannot_apply_exactly(
    document_id: str,
    field_path: str,
    value: object,
    error_code: str,
) -> None:
    original = run_review(ReviewRunRequest(scenario_id="invoice_total_mismatch"))

    with pytest.raises(CorrectionApplicationError) as captured:
        rerun_review(
            original,
            [
                UserCorrection(
                    document_id=document_id,
                    field_path=field_path,
                    value=value,
                )
            ],
        )

    assert captured.value.code == error_code
    assert all(item.get("event") != "corrections_applied" for item in original["audit_trail"])


@pytest.mark.parametrize(
    ("second_value", "error_code"),
    [
        ("12000", "duplicate_correction_target"),
        ("13000", "conflicting_correction_target"),
    ],
)
def test_rerun_rejects_duplicate_or_conflicting_correction_targets(
    second_value: str,
    error_code: str,
) -> None:
    state = run_review(ReviewRunRequest(scenario_id="invoice_total_mismatch"))
    corrections = [
        UserCorrection(
            document_id="doc_inv",
            field_path="fields.total_amount",
            value="12000",
        ),
        UserCorrection(
            document_id="doc_inv",
            field_path="fields.total_amount",
            value=second_value,
        ),
    ]

    with pytest.raises(CorrectionApplicationError) as captured:
        rerun_review(state, corrections)

    assert captured.value.code == error_code


def test_line_item_correction_is_exact_in_normalized_and_public_child_result() -> None:
    state = run_review(ReviewRunRequest(scenario_id="happy_path"))

    corrected = rerun_review(
        state,
        [
            UserCorrection(
                document_id="doc_inv",
                field_path="line_items.0.quantity",
                value="1000",
            )
        ],
    )
    public = to_review_run_result_v2(corrected)
    invoice = next(item for item in public.document_results if item.document_id == "doc_inv")
    quantity = next(item for item in invoice.line_items[0].fields if item.name == "quantity")

    assert (
        corrected["normalized_documents"]["doc_inv"]["line_items"][0]["raw_fields"]["quantity"]
        == "1000"
    )
    assert quantity.value == "1000"
    assert quantity.normalized_value == "1000"
    assert {item.issue_code for item in public.validation_issues} >= {
        "INV_LINE_AMOUNT_MISMATCH",
        "QTY_MISMATCH_INVOICE_PACKING",
    }
