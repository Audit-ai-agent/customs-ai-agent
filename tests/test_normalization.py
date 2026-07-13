from decimal import Decimal

from customs_ai_agent.nodes.normalize import (
    is_valid_hs_code,
    normalize_country,
    normalize_currency,
    normalize_date,
    normalize_hs_code,
    parse_weight_kg,
)


def test_basic_normalizers():
    assert normalize_currency("US$") == "USD"
    assert parse_weight_kg("1.2 MT") == Decimal("1200.0")
    assert parse_weight_kg("1200 KG") == Decimal("1200")
    assert normalize_hs_code("0303.54-0000") == "0303540000"
    assert is_valid_hs_code("030354")
    assert is_valid_hs_code("0303540000")
    assert normalize_country("Norway") == "NO"
    assert normalize_date("Jun 1 2026") == "2026-06-01"
