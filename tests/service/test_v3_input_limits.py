from __future__ import annotations

import json
from dataclasses import replace

import pytest

from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.input import inspect_v3_auxiliary_request
from customs_ai_agent.service.settings import ServiceSettings


def _nested_request(run_id: str, documents: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_id": "case-1",
        "documents": documents,
        "llm_mode": "llm",
        "review_as_of": "2026-07-22",
        "review_run_id": run_id,
    }


def _revision(parent_documents: list[dict[str, object]], revised_documents=None) -> bytes:
    return json.dumps(
        {
            "parent_request": _nested_request("parent-run", parent_documents),
            "revised_request": _nested_request(
                "child-run",
                parent_documents if revised_documents is None else revised_documents,
            ),
        },
        separators=(",", ":"),
    ).encode()


def _document(text: str = "invoice") -> dict[str, object]:
    return {
        "document_id": "doc-1",
        "filename": "invoice.txt",
        "content_text": text,
        "image_pages": [],
        "pdf_pages": None,
    }


def test_revision_applies_effective_document_limit_to_parent_and_child(
    service_settings: ServiceSettings,
) -> None:
    settings = replace(service_settings, max_documents=1)
    second = {**_document(), "document_id": "doc-2"}
    for payload in (
        _revision([_document(), second], [_document()]),
        _revision([_document()], [_document(), second]),
    ):
        with pytest.raises(ServiceHttpError) as captured:
            inspect_v3_auxiliary_request(payload, operation="revision", settings=settings)
        assert captured.value.status_code == 422
        assert captured.value.code == "request_invalid"


def test_revision_applies_total_text_limit_to_parent_and_child(
    service_settings: ServiceSettings,
) -> None:
    settings = replace(service_settings, max_text_characters=8)
    for payload in (
        _revision([_document("123456789")], [_document("ok")]),
        _revision([_document("ok")], [_document("123456789")]),
    ):
        with pytest.raises(ServiceHttpError) as captured:
            inspect_v3_auxiliary_request(payload, operation="revision", settings=settings)
        assert captured.value.status_code == 413
        assert captured.value.code == "request_too_large"


def test_revision_accepts_backend_empty_visual_placeholders(
    service_settings: ServiceSettings,
) -> None:
    inspected = inspect_v3_auxiliary_request(
        _revision([_document()]),
        operation="revision",
        settings=service_settings,
    )
    assert inspected.replay_id == "child-run"
