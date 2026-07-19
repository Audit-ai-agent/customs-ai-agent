import re

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
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", request_id)


def test_safe_review_run_id_remains_readable_and_http_header_compatible():
    request_id = stable_retrieval_request_id("runtime-live-eval-v1", 0)

    assert request_id == "runtime-live-eval-v1-retrieve-0"
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", request_id)


def test_colon_allowed_by_agent_wire_is_hashed_for_stricter_evidence_http_header():
    first = stable_retrieval_request_id("backend:review:1", 0)
    repeated = stable_retrieval_request_id("backend:review:1", 0)
    retry = stable_retrieval_request_id("backend:review:1", 1)

    assert first == repeated
    assert first != retry
    assert first.startswith("retrieval-")
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", first)
