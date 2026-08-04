from __future__ import annotations

from customs_ai_agent.contracts import available_schemas, verify_packaged_schemas


def test_only_live_http_schema_bytes_are_packaged() -> None:
    assert available_schemas() == (
        ("backend", "v3", "review-request"),
        ("backend", "v3", "review-result"),
        ("evidence", "v3", "rag-retrieval"),
        ("evidence", "v3", "readiness"),
    )
    assert set(verify_packaged_schemas()) == set(available_schemas())
