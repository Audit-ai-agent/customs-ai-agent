from __future__ import annotations

from customs_ai_agent.adapters.retrieval import MockRegulationRetriever
from customs_ai_agent.runner import run_review
from customs_ai_agent.schemas.models import ReviewRunRequest


class _NeverCalledRetriever:
    def __init__(self) -> None:
        self.called = False

    def retrieve(self, _request):
        self.called = True
        raise AssertionError("Evidence retrieval must not receive an ambiguous Invoice")


def _invoice(document_id: str, invoice_no: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "filename": f"{document_id}.txt",
        "content_text": "\n".join(
            [
                "COMMERCIAL INVOICE",
                f"INVOICE NO: {invoice_no}",
                "SELLER: Example Seller",
                "BUYER: Example Buyer",
                "CURRENCY: USD",
                "TOTAL AMOUNT: 100",
                "ITEM: Steel bolts | HS: 731815 | QTY: 10 | AMOUNT: 100",
            ]
        ),
    }


def test_duplicate_document_type_is_not_silently_first_selected():
    request = ReviewRunRequest(
        case_id="case_duplicate_invoice",
        documents=[
            _invoice("invoice_a", "INV-A"),
            _invoice("invoice_b", "INV-B"),
        ],
        llm_mode="mock",
    )

    retriever = _NeverCalledRetriever()
    state = run_review(request, retriever=retriever)

    assert retriever.called is False
    assert state["pairing_result"]["invoice_document_id"] is None
    assert state["pairing_result"]["document_type_conflicts"] == [
        {
            "document_type": "invoice",
            "document_ids": ["invoice_a", "invoice_b"],
        }
    ]
    assert state["pairing_result"]["document_set_status"] == "ambiguous_document_set"
    duplicate_issues = [
        issue
        for issue in state["validation_issues"]
        if issue["issue_code"] == "DUPLICATE_DOCUMENT_TYPE"
    ]
    assert len(duplicate_issues) == 1
    assert duplicate_issues[0]["observed_value"] == "invoice_a, invoice_b"
    assert state["retrieval_request"] == {}
    assert state["retrieval_status"] == "failed"
    assert state["retrieval_errors"] == [
        {
            "code": "AMBIGUOUS_INVOICE_RETRIEVAL_BLOCKED",
            "message": (
                "Regulatory retrieval was not called because the Case contains "
                "multiple invoice candidates."
            ),
            "retryable": False,
        }
    ]
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"


def test_duplicate_bill_of_lading_is_also_an_ambiguous_document_set():
    bill = {
        "document_id": "bl_a",
        "filename": "bl_a.txt",
        "content_text": "\n".join(
            [
                "BILL OF LADING",
                "B/L NO: BL-A",
                "SHIPPER: Example Seller",
                "CONSIGNEE: Example Buyer",
                "PACKAGES: 10 CARTONS",
                "GROSS WEIGHT: 100 KG",
            ]
        ),
    }
    duplicate = dict(bill)
    duplicate.update(
        {
            "document_id": "bl_b",
            "filename": "bl_b.txt",
            "content_text": str(bill["content_text"]).replace("BL-A", "BL-B"),
        }
    )
    request = ReviewRunRequest(
        case_id="case_duplicate_bl",
        documents=[bill, duplicate],
        llm_mode="mock",
    )

    state = run_review(request, retriever=MockRegulationRetriever())

    assert state["pairing_result"]["document_set_status"] == "ambiguous_document_set"
    assert state["pairing_result"]["document_type_conflicts"] == [
        {
            "document_type": "bill_of_lading",
            "document_ids": ["bl_a", "bl_b"],
        }
    ]
    assert any(
        issue["issue_code"] == "DUPLICATE_DOCUMENT_TYPE" and issue["observed_value"] == "bl_a, bl_b"
        for issue in state["validation_issues"]
    )
    assert state["final_status"] == "needs_review"
