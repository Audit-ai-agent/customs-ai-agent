from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Sequence, cast

import pytest

from customs_ai_agent.evaluation.document_ocr_quality import CandidateSpanV1, SyntheticDocumentType
from customs_ai_agent.evaluation.native_pdf_text_layer_fields import (
    project_native_pdf_text_layer_fields,
)
from customs_ai_agent.schemas.document_extraction import BoundingBoxV1


def _span(
    *,
    reading_order: int,
    text: str,
    x: float,
    y: float,
    page_number: int = 1,
    width: float = 0.18,
    height: float = 0.015,
) -> CandidateSpanV1:
    return CandidateSpanV1(
        page_number=page_number,
        reading_order=reading_order,
        text=text,
        bbox=BoundingBoxV1(x=x, y=y, width=width, height=height),
    )


def _same_row_pairs(pairs: list[tuple[str, str]]) -> list[CandidateSpanV1]:
    spans: list[CandidateSpanV1] = []
    for index, (label, value) in enumerate(pairs):
        y = 0.05 + (index * 0.055)
        spans.extend(
            (
                _span(reading_order=(index * 2) + 1, text=label, x=0.08, y=y),
                _span(reading_order=(index * 2) + 2, text=value, x=0.40, y=y),
            )
        )
    return spans


@pytest.mark.parametrize(
    ("document_type", "pairs", "expected"),
    [
        (
            "bill_of_lading",
            [
                ("Shipper", "Example Exporter"),
                ("Consignee", "Example Consignee"),
                ("Port of Loading", "Port A"),
                ("Port of Discharge", "Port B"),
                ("Final Destination", "Port C"),
                ("Vessel / Flight", "Example Vessel"),
                ("Voyage No", "V-001"),
                ("B/L No", "BL-001"),
                ("Container No", "CONT-001"),
                ("Seal No", "SEAL-001"),
                ("Goods Description", "Example Goods"),
                ("Gross Weight", "10 KG"),
                ("No of Containers", "1"),
            ],
            [
                "exporter_name",
                "consignee",
                "load_port",
                "disch_port",
                "final_dest",
                "vessel",
                "voyage",
                "bl_no",
                "container_no",
                "seal_no",
                "good",
                "gross_weight",
                "containers",
            ],
        ),
        (
            "invoice",
            [
                ("Exporter", "Example Exporter"),
                ("Notify Party", "Example Importer"),
                ("Consignee", "Example Consignee"),
                ("Port of Loading", "Port A"),
                ("Port of Discharge", "Port B"),
                ("Vessel", "Example Vessel"),
                ("Invoice Number", "INV-001"),
                ("Invoice Date", "2026-08-16"),
                ("Letter of Credit", "LC-001"),
                ("Goods Description", "Example Goods"),
                ("Unit", "PCS"),
                ("Quantity", "2"),
                ("Unit Price", "3.00"),
                ("Amount", "6.00"),
                ("Total", "6.00"),
            ],
            [
                "exporter_name",
                "importer_name",
                "consignee",
                "load_port",
                "disch_port",
                "vessel",
                "inv_no",
                "inv_date",
                "lc_no",
                "good",
                "unit",
                "qty",
                "unit_price",
                "amount",
                "total",
            ],
        ),
        (
            "packing_list",
            [
                ("Exporter", "Example Exporter"),
                ("Consignee", "Example Consignee"),
                ("No. of Invoice", "INV-001"),
                ("No. of L / C", "LC-001"),
                ("Goods Description", "Example Goods"),
                ("Unit", "PCS"),
                ("Quantity", "2"),
                ("No of Cartons / PKGS", "3 CTNS"),
                ("Net Weight", "10 KG"),
                ("Gross Weight", "12 KG"),
                ("CBM", "0.20"),
                ("Net/Gross Weight", "10 KG / 12 KG"),
            ],
            [
                "exporter_name",
                "consignee",
                "inv_no",
                "lc_no",
                "good",
                "unit",
                "qty",
                "cartons",
                "net_weight",
                "gross_weight",
                "cbm",
                "net_gross_weight",
            ],
        ),
    ],
)
def test_projects_all_visible_fields_from_one_unambiguous_same_row_pair_per_label(
    document_type: str,
    pairs: list[tuple[str, str]],
    expected: list[str],
) -> None:
    fields = project_native_pdf_text_layer_fields(
        document_type=cast(SyntheticDocumentType, document_type),
        spans=_same_row_pairs(pairs),
    )

    assert [field.field_name for field in fields] == expected


def test_projects_a_single_below_label_value_pair() -> None:
    fields = project_native_pdf_text_layer_fields(
        document_type="invoice",
        spans=[
            _span(reading_order=1, text="Invoice Number", x=0.20, y=0.20),
            _span(reading_order=2, text="INV-001", x=0.22, y=0.245),
        ],
    )

    assert [(field.field_name, field.value) for field in fields] == [("inv_no", "INV-001")]


def test_rejects_a_field_when_the_unique_nearby_value_anchor_is_duplicated() -> None:
    fields = project_native_pdf_text_layer_fields(
        document_type="invoice",
        spans=[
            _span(reading_order=1, text="Invoice Number", x=0.10, y=0.20),
            _span(reading_order=2, text="INV-ONE", x=0.40, y=0.20),
            _span(reading_order=3, text="INV-TWO", x=0.40, y=0.20),
        ],
    )

    assert fields == []


def test_uses_only_one_geometrically_nearby_value_when_callback_order_has_an_unrelated_fragment() -> (
    None
):
    fields = project_native_pdf_text_layer_fields(
        document_type="invoice",
        spans=[
            _span(reading_order=1, text="Invoice Number", x=0.10, y=0.20),
            _span(reading_order=2, text="Unrelated page fragment", x=0.80, y=0.80),
            _span(reading_order=3, text="INV-ONE", x=0.40, y=0.20),
        ],
    )

    assert [(field.field_name, field.value) for field in fields] == [("inv_no", "INV-ONE")]


def test_rejects_a_field_when_two_different_values_are_geometrically_nearby() -> None:
    fields = project_native_pdf_text_layer_fields(
        document_type="invoice",
        spans=[
            _span(reading_order=1, text="Invoice Number", x=0.10, y=0.20),
            _span(reading_order=2, text="INV-ONE", x=0.40, y=0.20),
            _span(reading_order=3, text="INV-TWO", x=0.42, y=0.20),
        ],
    )

    assert fields == []


def test_rejects_a_field_when_its_visible_label_is_duplicated() -> None:
    fields = project_native_pdf_text_layer_fields(
        document_type="packing_list",
        spans=[
            _span(reading_order=1, text="No. of Invoice", x=0.10, y=0.10),
            _span(reading_order=2, text="INV-ONE", x=0.40, y=0.10),
            _span(reading_order=3, text="No. of Invoice", x=0.10, y=0.30),
            _span(reading_order=4, text="INV-TWO", x=0.40, y=0.30),
        ],
    )

    assert fields == []


def test_rejects_missing_geometry_and_cross_page_values() -> None:
    no_geometry = CandidateSpanV1(page_number=1, reading_order=1, text="Invoice Number")
    value = _span(reading_order=2, text="INV-001", x=0.40, y=0.10)
    cross_page_label = _span(reading_order=1, text="Invoice Number", x=0.10, y=0.10)
    cross_page_value = _span(
        reading_order=2,
        text="INV-001",
        x=0.40,
        y=0.10,
        page_number=2,
    )

    assert (
        project_native_pdf_text_layer_fields(document_type="invoice", spans=[no_geometry, value])
        == []
    )
    assert (
        project_native_pdf_text_layer_fields(
            document_type="invoice", spans=[cross_page_label, cross_page_value]
        )
        == []
    )


def test_rejects_a_known_label_as_another_label_value() -> None:
    fields = project_native_pdf_text_layer_fields(
        document_type="invoice",
        spans=[
            _span(reading_order=1, text="Invoice Number", x=0.10, y=0.10),
            _span(reading_order=2, text="Invoice Date", x=0.40, y=0.10),
        ],
    )

    assert fields == []


def test_splits_only_an_exact_geometric_quantity_and_ascii_unit_pair() -> None:
    fields = project_native_pdf_text_layer_fields(
        document_type="invoice",
        spans=[
            _span(reading_order=1, text="Quantity", x=0.10, y=0.10),
            _span(reading_order=2, text="2 PCS", x=0.40, y=0.10),
        ],
    )

    assert [(field.field_name, field.value) for field in fields] == [
        ("unit", "PCS"),
        ("qty", "2"),
    ]


@dataclass(frozen=True, slots=True)
class _InternalAnchoredSpan:
    page_number: int
    reading_order: int
    text: str
    anchor_x: float
    anchor_y: float
    bbox: None = None


def test_accepts_a_proven_internal_anchor_when_a_rectangle_is_not_available() -> None:
    spans = [
        _InternalAnchoredSpan(
            page_number=1,
            reading_order=1,
            text="Invoice Number",
            anchor_x=0.10,
            anchor_y=0.10,
        ),
        _InternalAnchoredSpan(
            page_number=1,
            reading_order=2,
            text="INV-001",
            anchor_x=0.40,
            anchor_y=0.10,
        ),
    ]

    fields = project_native_pdf_text_layer_fields(
        document_type="invoice",
        spans=cast("Sequence[CandidateSpanV1]", spans),
    )

    assert [(field.field_name, field.value) for field in fields] == [("inv_no", "INV-001")]


def test_has_only_candidate_document_type_and_span_inputs() -> None:
    assert tuple(inspect.signature(project_native_pdf_text_layer_fields).parameters) == (
        "document_type",
        "spans",
    )
