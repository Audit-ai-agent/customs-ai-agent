from customs_ai_agent.identifiers import (
    MAX_EVIDENCE_ITEM_ID_LENGTH,
    stable_line_item_id,
    stable_retrieval_request_id,
)
from customs_ai_agent.schemas.models import RetrievalQueryItem


def test_short_line_item_id_preserves_existing_contract_shape():
    assert stable_line_item_id("doc_inv", 1) == "doc_inv_line_1"


def test_long_document_ids_produce_stable_distinct_contract_valid_item_ids():
    first_document_id = "a" * 128
    second_document_id = "a" * 127 + "b"

    first = stable_line_item_id(first_document_id, 200)
    repeated = stable_line_item_id(first_document_id, 200)
    second = stable_line_item_id(second_document_id, 200)

    assert first == repeated
    assert first != second
    assert len(first) <= MAX_EVIDENCE_ITEM_ID_LENGTH
    RetrievalQueryItem(
        item_id=first,
        description="Synthetic item",
        destination_country="KR",
    )


def test_long_review_run_id_produces_stable_contract_bounded_request_id():
    review_run_id = "r" * 128

    request_id = stable_retrieval_request_id(review_run_id, 999)

    assert request_id == stable_retrieval_request_id(review_run_id, 999)
    assert len(request_id) <= MAX_EVIDENCE_ITEM_ID_LENGTH
    assert request_id.endswith(":retrieve:999")
