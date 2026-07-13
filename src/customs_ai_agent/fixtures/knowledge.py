from __future__ import annotations

from typing import Any


KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "evidence_id": "ev_hs_731815_general",
        "source_type": "hs_code",
        "title": "HS reference for bolts",
        "law_name": "HS Tariff Schedule",
        "article_no": None,
        "chunk_id": "hs_731815_001",
        "text": "HS 7318.15 covers threaded screws and bolts. No special certificate is represented in this mock evidence.",
        "score": 0.86,
        "source_uri": "mock://hs/731815",
        "effective_date": "2026-01-01",
        "version_hash": "mock-731815-v1",
        "hs_codes": ["731815"],
        "keywords": ["stainless bolts", "bolts"],
        "required_document": None,
    },
    {
        "evidence_id": "ev_hs_030354_low",
        "source_type": "notice",
        "title": "Mock import requirement for frozen fish",
        "law_name": "Mock Customs Confirmation Notice",
        "article_no": "mock-article-12",
        "chunk_id": "notice_030354_001",
        "text": "Frozen fish items may require an additional quarantine certificate, but this mock evidence has only moderate confidence.",
        "score": 0.62,
        "source_uri": "mock://notice/030354",
        "effective_date": "2026-01-01",
        "version_hash": "mock-030354-v1",
        "hs_codes": ["030354"],
        "keywords": ["frozen mackerel", "mackerel", "fish"],
        "required_document": "quarantine_certificate",
    },
]
