from __future__ import annotations

import pytest

from customs_ai_agent.prompts import (
    load_grounded_assistant_system_prompt,
    load_grounded_report_system_prompt,
)
from customs_ai_agent.service.user_text import (
    contains_internal_result_marker,
    localize_deterministic_reviewer_prose,
    review_result_markers,
    reviewer_text_is_safe,
)


@pytest.mark.parametrize(
    ("value", "markers"),
    [
        ("comparison-4f2a9c", ()),
        ("finding-8c2d11", ()),
        ("inspect_document_consistency", ()),
        ("needs_review", ()),
        ("case_grounded", ()),
        ("결과 해시는 " + "a" * 64 + "입니다.", ()),
        ("CCCT_DOC_IMPORT_CONFIRMATION", ("CCCT_DOC_IMPORT_CONFIRMATION",)),
    ],
)
def test_reviewer_text_blocks_internal_result_markers(value: str, markers: tuple[str, ...]) -> None:
    assert contains_internal_result_marker(value, markers=markers)


def test_reviewer_text_keeps_document_citations_and_official_law_names() -> None:
    values = (
        "상업송장 번호 INV-001의 수량을 포장명세서와 함께 확인해 주세요.",
        "Invoice No: INV_001은 원문에 적힌 문서번호입니다.",
        "수입식품안전관리 특별법에 따른 확인 항목을 검토해 주세요.",
        "선하증권 B/L No. BL-002와 컨테이너 번호 ABCD1234567을 확인했습니다.",
    )

    assert reviewer_text_is_safe(values)
    assert not any(contains_internal_result_marker(value) for value in values)


def test_deterministic_prose_uses_korean_terms_without_mutating_structured_citations() -> None:
    prose = localize_deterministic_reviewer_prose(
        "Invoice와 Packing List의 법령 Evidence를 Review Run에서 확인해야 합니다."
    )

    assert prose == "상업송장과 포장명세서의 법령 근거를 심사 결과에서 확인해야 합니다."


def test_result_marker_collection_keeps_law_names_and_document_quotes_out_of_the_guard() -> None:
    markers = review_result_markers(
        {
            "finding": {
                "finding_id": "finding-123",
                "code": "CCCT_DOC_IMPORT_CONFIRMATION",
                "message": "수입식품안전관리 특별법에 따른 확인이 필요합니다.",
            },
            "evidence": {
                "law_name": "수입식품안전관리 특별법",
                "source_quote": "Invoice No: INV_001",
            },
        }
    )

    assert "finding-123" in markers
    assert "CCCT_DOC_IMPORT_CONFIRMATION" in markers
    assert "수입식품안전관리 특별법" not in markers
    assert "Invoice No: INV_001" not in markers
    assert contains_internal_result_marker("CCCT_DOC_IMPORT_CONFIRMATION", markers=markers)
    assert not contains_internal_result_marker("Invoice No: INV_001", markers=markers)


def test_grounded_prompts_require_korean_reviewer_prose_and_structured_references() -> None:
    assistant_prompt = load_grounded_assistant_system_prompt()
    report_prompt = load_grounded_report_system_prompt()

    assert "자연스러운 한국어" in assistant_prompt
    assert "comparison-/finding-" in assistant_prompt
    assert "원문을 바꾸지 말고 보존" in assistant_prompt
    assert "자연스러운 한국어" in report_prompt
    assert "fact_ids, comparison_ids, evidence_ids" in report_prompt
    assert "원문을 바꾸지 말고 보존" in report_prompt
