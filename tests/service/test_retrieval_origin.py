"""Origins on Data retrieval items are ISO-2 codes or absent, never free text.

Regression: a Certificate of Origin spelling out "REPUBLIC OF KOREA" used to
reach the Evidence API verbatim, which rejects non-ISO origins and failed the
whole review's legal-evidence retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

from customs_ai_agent.service.workflow import _is_country_code, _retrieval_origin


@dataclass
class _Fact:
    normalized_value: str | None


def test_is_country_code_accepts_only_two_uppercase_letters() -> None:
    assert _is_country_code("KR")
    assert not _is_country_code("REPUBLIC OF KOREA")
    assert not _is_country_code("kr")
    assert not _is_country_code("KOR")


def test_retrieval_origin_degrades_free_text_to_none() -> None:
    assert _retrieval_origin(_Fact("KR")) == "KR"
    assert _retrieval_origin(_Fact("REPUBLIC OF KOREA")) is None
    assert _retrieval_origin(_Fact(None)) is None
    assert _retrieval_origin(None) is None
