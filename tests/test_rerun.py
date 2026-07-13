from customs_ai_agent.runner import rerun_review, run_review
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
