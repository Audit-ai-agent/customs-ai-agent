from decimal import Decimal

from customs_ai_agent.nodes.normalize import (
    is_ambiguous_numeric_date,
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


def test_hs_normalizer_only_removes_allowlisted_display_separators():
    assert normalize_hs_code("7318 15") == "731815"
    assert normalize_hs_code("7318-15-1000") == "7318151000"
    assert normalize_hs_code("ABC731815") is None
    assert normalize_hs_code("7318/15") is None
    assert normalize_hs_code("7318\t15") is None
    assert normalize_hs_code("７３１８１５") is None
    assert not is_valid_hs_code("１２３４５６")


def test_numeric_dates_require_an_unambiguous_interpretation():
    assert is_ambiguous_numeric_date("05/06/2025")
    assert normalize_date("05/06/2025") is None
    assert is_ambiguous_numeric_date("05-06-2025")
    assert normalize_date("05-06-2025") is None
    assert is_ambiguous_numeric_date("05.06.2025")
    assert normalize_date("05.06.2025") is None
    assert not is_ambiguous_numeric_date("13/06/2025")
    assert normalize_date("13/06/2025") == "2025-06-13"
    assert normalize_date("13-06-2025") == "2025-06-13"
    assert normalize_date("13.06.2025") == "2025-06-13"
    assert normalize_date("06/13/2025") == "2025-06-13"
    assert normalize_date("05/05/2025") == "2025-05-05"
    assert normalize_date("2025-06-05") == "2025-06-05"
    assert normalize_date("Jun 5 2025") == "2025-06-05"
