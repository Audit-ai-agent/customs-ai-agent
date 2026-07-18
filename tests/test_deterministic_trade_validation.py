from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any

from customs_ai_agent.nodes.pair import pair_documents
from customs_ai_agent.nodes.validate import validate_cross_documents


def _document(
    document_id: str,
    document_type: str,
    *,
    fields: dict[str, Any],
    line_items: list[dict[str, Any]] | None = None,
    raw_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "filename": f"{document_id}.png",
        "document_type": document_type,
        "fields": fields,
        "raw_fields": raw_fields or {},
        "field_evidence": {},
        "line_items": line_items or [],
        "extraction_confidence": 1.0,
    }


def _clean_documents() -> dict[str, dict[str, Any]]:
    invoice_amount = Decimal("717754.27")
    return {
        "ci": _document(
            "ci",
            "invoice",
            fields={
                "invoice_no": "KR-INV-0001",
                "invoice_date": "2025-08-05",
                "departure_date": "2025-08-13",
                "vessel": "HMM Algeciras / V.350E",
                "port_of_loading": "Gwangyang, South Korea",
                "port_of_discharge": "Los Angeles, USA",
                "seller": "SK Hynix Inc.",
                "buyer": "Foxconn Technology Group",
                "consignee": "Foxconn Technology Group",
                "country_of_origin": "KR",
                "shipping_marks": "FOXCONN // LOS ANGELES // C/NO. 1-31",
                "packages": "31 Cartons",
                "total_amount": invoice_amount,
            },
            raw_fields={"total_amount": "USD 717,754.27"},
            line_items=[
                {
                    "item_id": "ci-line-1",
                    "description": "High-Bandwidth Memory (HBM3) Chips",
                    "hs_code": "854232",
                    "origin_country": "KR",
                    "quantity": Decimal("160"),
                    "unit_price": Decimal("4485.96"),
                    "amount": invoice_amount,
                    "raw_fields": {
                        "quantity": "160 PCS",
                        "unit_price": "USD 4,485.96",
                        "amount": "USD 717,754.27",
                    },
                    "field_evidence": {},
                }
            ],
        ),
        "pl": _document(
            "pl",
            "packing_list",
            fields={
                "invoice_no": "KR-INV-0001",
                "invoice_date": "2025-08-05",
                "departure_date": "2025-08-13",
                "vessel": "HMM Algeciras / V.350E",
                "port_of_loading": "Gwangyang, South Korea",
                "port_of_discharge": "Los Angeles, USA",
                "seller": "SK Hynix Inc.",
                "buyer": "Foxconn Technology Group",
                "consignee": "Foxconn Technology Group",
                "country_of_origin": "KR",
                "shipping_marks": "FOXCONN // LOS ANGELES // C/NO. 1-31",
                "container_numbers": ["HMMU9279352"],
                "seal_numbers": ["KR-58959"],
                "packages": "31 Cartons",
                "total_gross_weight_kg": Decimal("4198.03"),
                "total_measurement_cbm": Decimal("20.99"),
            },
            raw_fields={
                "total_gross_weight_kg": "4,198.03 KGS",
                "total_measurement_cbm": "20.99 CBM",
            },
            line_items=[
                {
                    "item_id": "pl-line-1",
                    "description": "High-Bandwidth Memory (HBM3) Chips",
                    "quantity": Decimal("160"),
                    "gross_weight_kg": Decimal("3165.35"),
                    "raw_fields": {
                        "quantity": "160 PCS",
                        "gross_weight_kg": "3,165.35 KGS",
                    },
                    "field_evidence": {},
                },
                {
                    "item_id": "pl-line-accessory",
                    "description": "Packing accessory",
                    "quantity": Decimal("1"),
                    "gross_weight_kg": Decimal("1"),
                    "raw_fields": {
                        "quantity": "1 PCS",
                        "gross_weight_kg": "1.00 KGS",
                    },
                    "field_evidence": {},
                },
            ],
        ),
        "bl": _document(
            "bl",
            "bill_of_lading",
            fields={
                "bl_no": "HMMU3790301681",
                "shipper": "SK Hynix Inc.",
                "consignee": "Foxconn Technology Group",
                "vessel": "HMM Algeciras",
                "voyage_no": "V.350E",
                "port_of_loading": "Gwangyang, South Korea",
                "port_of_discharge": "Los Angeles, USA",
                "container_numbers": ["HMMU9279352"],
                "seal_numbers": ["KR-58959"],
                "packages": "1x40HQ / 31 Cartons",
                "gross_weight_kg": Decimal("4198.03"),
                "measurement_cbm": Decimal("20.99"),
                "freight_prepaid": "Y",
            },
            raw_fields={
                "gross_weight_kg": "4,198.03 KGS",
                "measurement_cbm": "20.99 RT",
                "freight_prepaid": "Y",
            },
        ),
        "co": _document(
            "co",
            "certificate_of_origin",
            fields={
                "certificate_no": "CO-SYN-0001",
                "exporter": "SK Hynix Inc.",
                "importer": "Foxconn Technology Group",
                "origin_country": "KR",
                "hs_codes": ["854232"],
                "goods_description": "High-Bandwidth Memory (HBM3) Chips",
            },
        ),
    }


def _validate(documents: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    state: dict[str, Any] = {
        "normalized_documents": documents,
        "llm_mode": "mock",
        "audit_trail": [],
        "requires_human_review": False,
    }
    pairing = pair_documents(state)
    state.update(pairing)
    validation = validate_cross_documents(state)
    return pairing["pairing_result"], validation


def _pair(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return pair_documents(
        {
            "normalized_documents": documents,
            "llm_mode": "mock",
            "audit_trail": [],
            "requires_human_review": False,
        }
    )


def _codes(validation: dict[str, Any]) -> set[str]:
    return {str(issue["issue_code"]) for issue in validation["validation_issues"]}


def test_full_clean_trade_rules_pass_and_packing_only_item_stays_ambiguous():
    documents = _clean_documents()
    update = _pair(documents)
    pairing, validation = _validate(documents)

    assert validation["validation_issues"] == []
    assert update["requires_human_review"] is False
    assert pairing["unmatched_invoice_item_ids"] == []
    assert pairing["unmatched_packing_item_ids"] == ["pl-line-accessory"]
    assert pairing["unmatched_packing_item_status"] == "ambiguous_not_scored"
    assert pairing["line_item_pairing_status"] == ("deterministic_with_unscored_packing_extras")
    assert pairing["line_item_pairing_ambiguities"] == []


def test_pairing_score_threshold_is_fail_closed_and_inclusive(
    monkeypatch: Any,
):
    documents = _clean_documents()
    documents["pl"]["line_items"] = documents["pl"]["line_items"][:1]

    monkeypatch.setattr(
        "customs_ai_agent.nodes.pair._score",
        lambda _left, _right: 0.6999,
    )
    pairing, validation = _validate(documents)

    assert pairing["line_item_links"] == [
        {
            "invoice_item_id": "ci-line-1",
            "packing_item_id": None,
            "score": 0.0,
        }
    ]
    assert pairing["unmatched_invoice_item_ids"] == ["ci-line-1"]
    assert pairing["unmatched_invoice_item_status"] == "needs_review"
    assert pairing["unmatched_packing_item_ids"] == ["pl-line-1"]
    assert pairing["unmatched_packing_item_status"] == "ambiguous_not_scored"
    assert pairing["line_item_pairing_ambiguities"] == [
        {
            "ambiguity_type": "below_minimum_score",
            "invoice_item_ids": ["ci-line-1"],
            "packing_item_ids": ["pl-line-1"],
            "best_score": 0.6999,
            "best_score_tied": False,
        }
    ]
    assert "LINE_ITEM_UNMATCHED" in _codes(validation)
    assert validation["requires_human_review"] is True

    monkeypatch.setattr(
        "customs_ai_agent.nodes.pair._score",
        lambda _left, _right: 0.70,
    )
    accepted_pairing, accepted_validation = _validate(documents)

    assert accepted_pairing["line_item_links"][0]["packing_item_id"] == "pl-line-1"
    assert accepted_pairing["line_item_links"][0]["score"] == 0.70
    assert accepted_pairing["unmatched_invoice_item_ids"] == []
    assert accepted_pairing["unmatched_packing_item_ids"] == []
    assert accepted_pairing["line_item_pairing_ambiguities"] == []
    assert "LINE_ITEM_UNMATCHED" not in _codes(accepted_validation)


def test_one_to_many_candidates_do_not_select_a_unique_top_score(
    monkeypatch: Any,
):
    documents = _clean_documents()

    def candidate_score(_invoice_description: str | None, packing_description: str | None):
        return 0.95 if packing_description == "Packing accessory" else 0.80

    monkeypatch.setattr("customs_ai_agent.nodes.pair._score", candidate_score)
    update = _pair(documents)
    pairing = update["pairing_result"]

    assert pairing["line_item_links"][0]["packing_item_id"] is None
    assert pairing["unmatched_invoice_item_ids"] == ["ci-line-1"]
    assert pairing["unmatched_packing_item_ids"] == [
        "pl-line-1",
        "pl-line-accessory",
    ]
    assert pairing["line_item_pairing_ambiguities"] == [
        {
            "ambiguity_type": "one_to_many_candidates",
            "invoice_item_ids": ["ci-line-1"],
            "packing_item_ids": ["pl-line-1", "pl-line-accessory"],
            "best_score": 0.95,
            "best_score_tied": False,
        }
    ]
    assert update["requires_human_review"] is True


def test_many_to_one_tie_does_not_depend_on_greedy_input_order(
    monkeypatch: Any,
):
    documents = _clean_documents()
    documents["pl"]["line_items"] = documents["pl"]["line_items"][:1]
    second_invoice_item = deepcopy(documents["ci"]["line_items"][0])
    second_invoice_item["item_id"] = "ci-line-2"
    second_invoice_item["description"] = "Second invoice item"
    documents["ci"]["line_items"].append(second_invoice_item)
    monkeypatch.setattr(
        "customs_ai_agent.nodes.pair._score",
        lambda _left, _right: 0.90,
    )

    update = _pair(documents)
    pairing = update["pairing_result"]

    assert all(link["packing_item_id"] is None for link in pairing["line_item_links"])
    assert pairing["unmatched_invoice_item_ids"] == ["ci-line-1", "ci-line-2"]
    assert pairing["unmatched_packing_item_ids"] == ["pl-line-1"]
    assert pairing["line_item_pairing_ambiguities"] == [
        {
            "ambiguity_type": "many_to_one_candidates",
            "invoice_item_ids": ["ci-line-1", "ci-line-2"],
            "packing_item_ids": ["pl-line-1"],
            "best_score": 0.90,
            "best_score_tied": True,
        }
    ]
    assert update["requires_human_review"] is True


def test_invoice_line_amount_uses_displayed_unit_price_precision_interval():
    inside = _clean_documents()
    inside["ci"]["line_items"][0]["amount"] = Decimal("717754.40")
    inside["ci"]["line_items"][0]["raw_fields"]["amount"] = "USD 717,754.40"
    inside["ci"]["fields"]["total_amount"] = Decimal("717754.40")
    inside["ci"]["raw_fields"]["total_amount"] = "USD 717,754.40"
    _, inside_result = _validate(inside)
    assert "INV_LINE_AMOUNT_MISMATCH" not in _codes(inside_result)

    outside = _clean_documents()
    outside["ci"]["line_items"][0]["amount"] = Decimal("717755.00")
    outside["ci"]["line_items"][0]["raw_fields"]["amount"] = "USD 717,755.00"
    outside["ci"]["fields"]["total_amount"] = Decimal("717755.00")
    outside["ci"]["raw_fields"]["total_amount"] = "USD 717,755.00"
    _, outside_result = _validate(outside)
    assert "INV_LINE_AMOUNT_MISMATCH" in _codes(outside_result)


def test_cross_document_party_hs_and_freight_conflicts_are_explicit():
    documents = deepcopy(_clean_documents())
    documents["pl"]["fields"]["seller"] = "Different Seller"
    documents["co"]["fields"]["hs_codes"] = ["999999"]
    documents["bl"]["fields"]["freight_collect"] = "Y"

    _, result = _validate(documents)

    assert {
        "SELLER_MISMATCH",
        "HS_CODE_MISMATCH",
        "FREIGHT_TERMS_INVALID",
    }.issubset(_codes(result))
    assert result["requires_human_review"] is True


def test_freight_terms_require_exactly_one_prepaid_or_collect_expression():
    neither = deepcopy(_clean_documents())
    neither["bl"]["fields"].pop("freight_prepaid")
    _, neither_result = _validate(neither)
    assert "FREIGHT_TERMS_INVALID" in _codes(neither_result)
    assert neither_result["requires_human_review"] is True

    text_neither = deepcopy(_clean_documents())
    text_neither["bl"]["fields"].pop("freight_prepaid")
    text_neither["bl"]["fields"]["freight_terms"] = "TO BE AGREED"
    _, text_neither_result = _validate(text_neither)
    assert "FREIGHT_TERMS_INVALID" in _codes(text_neither_result)

    text_prepaid = deepcopy(_clean_documents())
    text_prepaid["bl"]["fields"].pop("freight_prepaid")
    text_prepaid["bl"]["fields"]["freight_terms"] = "FREIGHT PREPAID"
    _, text_prepaid_result = _validate(text_prepaid)
    assert "FREIGHT_TERMS_INVALID" not in _codes(text_prepaid_result)

    text_collect = deepcopy(_clean_documents())
    text_collect["bl"]["fields"].pop("freight_prepaid")
    text_collect["bl"]["fields"]["freight_terms"] = "FREIGHT COLLECT"
    _, text_collect_result = _validate(text_collect)
    assert "FREIGHT_TERMS_INVALID" not in _codes(text_collect_result)

    text_both = deepcopy(_clean_documents())
    text_both["bl"]["fields"].pop("freight_prepaid")
    text_both["bl"]["fields"]["freight_terms"] = "FREIGHT PREPAID / COLLECT"
    _, text_both_result = _validate(text_both)
    assert "FREIGHT_TERMS_INVALID" in _codes(text_both_result)


def test_missing_required_document_types_are_explicit_and_fail_closed():
    for document_key, expected_type in (
        ("ci", "invoice"),
        ("pl", "packing_list"),
        ("bl", "bill_of_lading"),
    ):
        documents = deepcopy(_clean_documents())
        documents.pop(document_key)
        pairing, result = _validate(documents)
        assert pairing["document_set_status"] == "incomplete_document_set"
        assert pairing["missing_required_document_types"] == [expected_type]
        assert "REQUIRED_DOCUMENT_MISSING" in _codes(result)
        assert result["requires_human_review"] is True


def test_missing_comparison_input_is_not_silently_treated_as_consistent():
    documents = deepcopy(_clean_documents())
    documents["pl"]["fields"].pop("port_of_loading")
    documents["pl"]["fields"].pop("container_numbers")

    _, result = _validate(documents)

    missing = [
        issue
        for issue in result["validation_issues"]
        if issue["issue_code"] == "VALIDATION_INPUT_MISSING"
    ]
    assert missing
    assert any("port_of_loading" in str(issue["expected_value"]) for issue in missing)
    assert any("container_numbers" in str(issue["expected_value"]) for issue in missing)
    assert result["requires_human_review"] is True


def test_weight_comparison_uses_source_precision_instead_of_fixed_one_kg():
    inside = deepcopy(_clean_documents())
    inside["pl"]["fields"]["total_gross_weight_kg"] = Decimal("4198.04")
    inside["pl"]["raw_fields"]["total_gross_weight_kg"] = "4,198.04 KGS"
    _, inside_result = _validate(inside)
    assert "GROSS_WEIGHT_MISMATCH" not in _codes(inside_result)

    outside = deepcopy(_clean_documents())
    outside["pl"]["fields"]["total_gross_weight_kg"] = Decimal("4198.05")
    outside["pl"]["raw_fields"]["total_gross_weight_kg"] = "4,198.05 KGS"
    _, outside_result = _validate(outside)
    assert "GROSS_WEIGHT_MISMATCH" in _codes(outside_result)
