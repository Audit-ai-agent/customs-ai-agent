from __future__ import annotations

import ast
import inspect

from customs_ai_agent.evaluation.document_ocr_quality import CandidateSpanV1
from customs_ai_agent.evaluation.native_pdf_vector_form_fields import (
    NativePdfVectorVisibleFieldSpec,
    project_native_pdf_vector_form_fields,
)
from customs_ai_agent.evaluation.native_pdf_vector_grid import NativePdfVectorFormRegion


_INVOICE_NUMBER = NativePdfVectorVisibleFieldSpec(
    field_name="inv_no",
    normalizer="identifier-v1",
    labels=("Invoice Number",),
)
_CONSIGNEE = NativePdfVectorVisibleFieldSpec(
    field_name="consignee",
    normalizer="text-v1",
    labels=("Consignee",),
)


def _span(*, text: str, x: float, y: float, order: int) -> CandidateSpanV1:
    return CandidateSpanV1(
        page_number=1,
        reading_order=order,
        text=text,
        anchor_x=x,
        anchor_y=y,
    )


def _region(
    *, x: float = 0.05, y: float = 0.05, width: float = 0.80, height: float = 0.40
) -> NativePdfVectorFormRegion:
    return NativePdfVectorFormRegion(page_number=1, x=x, y=y, width=width, height=height)


def test_projects_one_same_row_visible_label_value_pair() -> None:
    fields = project_native_pdf_vector_form_fields(
        spans=(
            _span(text="Invoice Number", x=0.10, y=0.10, order=50),
            _span(text="INV-0007", x=0.50, y=0.10, order=1),
        ),
        regions=(_region(),),
        specifications=(_INVOICE_NUMBER,),
    )

    assert [(field.field_name, field.value) for field in fields] == [("inv_no", "INV-0007")]


def test_projects_one_below_multiline_value_without_a_coordinate_profile() -> None:
    fields = project_native_pdf_vector_form_fields(
        spans=(
            _span(text="Consignee", x=0.10, y=0.10, order=3),
            _span(text="Receiver", x=0.10, y=0.22, order=1),
            _span(text="Company", x=0.10, y=0.30, order=2),
        ),
        regions=(_region(),),
        specifications=(_CONSIGNEE,),
    )

    assert [(field.field_name, field.value) for field in fields] == [
        ("consignee", "Receiver\nCompany")
    ]


def test_omits_a_field_when_same_row_and_below_values_are_both_possible() -> None:
    fields = project_native_pdf_vector_form_fields(
        spans=(
            _span(text="Invoice Number", x=0.10, y=0.10, order=1),
            _span(text="INV-A", x=0.50, y=0.10, order=2),
            _span(text="INV-B", x=0.10, y=0.25, order=3),
        ),
        regions=(_region(),),
        specifications=(_INVOICE_NUMBER,),
    )

    assert fields == []


def test_omits_duplicate_labels_and_overlapping_region_ownership() -> None:
    duplicate = project_native_pdf_vector_form_fields(
        spans=(
            _span(text="Invoice Number", x=0.10, y=0.10, order=1),
            _span(text="INV-A", x=0.50, y=0.10, order=2),
            _span(text="Invoice Number", x=0.10, y=0.60, order=3),
            _span(text="INV-B", x=0.50, y=0.60, order=4),
        ),
        regions=(_region(y=0.05), _region(y=0.55)),
        specifications=(_INVOICE_NUMBER,),
    )
    overlapping = project_native_pdf_vector_form_fields(
        spans=(
            _span(text="Invoice Number", x=0.10, y=0.10, order=1),
            _span(text="INV-A", x=0.50, y=0.10, order=2),
        ),
        regions=(_region(), _region(x=0.08, y=0.08, width=0.50, height=0.20)),
        specifications=(_INVOICE_NUMBER,),
    )

    assert duplicate == []
    assert overlapping == []


def test_module_has_no_truth_or_fixed_run_index_boundary() -> None:
    import customs_ai_agent.evaluation.native_pdf_vector_form_fields as form_fields

    source = inspect.getsource(form_fields)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth" not in imported_modules
    assert "customs_ai_agent.evaluation.document_gold" not in imported_modules
    assert "reading_order" not in source
    assert tuple(inspect.signature(project_native_pdf_vector_form_fields).parameters) == (
        "spans",
        "regions",
        "specifications",
    )
