"""Regression checks for the pinned optional Docling preview profile.

The standard Agent test environment intentionally does not install heavyweight
document-preview extras.  The declaration test always runs; the import test
runs whenever an operator explicitly installs the Docling profile.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import tomllib
from pathlib import Path

import pytest

import customs_ai_agent.service.document_preview_runtime as runtime_module

_COMPATIBILITY_PINS = {
    "docling-core[chunking]": "2.13.1",
    "docling-ibm-models": "3.1.0",
    "docling-parse": "3.0.0",
    "pillow": "10.4.0",
}


def _declared_docling_extra() -> set[str]:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)
    extras = pyproject["project"]["optional-dependencies"]
    return set(extras["document-preview-docling"])


def test_docling_preview_profile_declares_legacy_compatible_pins() -> None:
    declared = _declared_docling_extra()
    for distribution, version in _COMPATIBILITY_PINS.items():
        assert f"{distribution}=={version}" in declared


def test_installed_docling_profile_imports_document_converter_with_legacy_bbox() -> None:
    if importlib.util.find_spec("docling") is None:
        pytest.skip("the optional document-preview-docling profile is not installed")

    expected_versions = {
        "docling": "2.15.1",
        "docling-core": "2.13.1",
        "docling-ibm-models": "3.1.0",
        "docling-parse": "3.0.0",
        "pillow": "10.4.0",
    }
    for distribution, version in expected_versions.items():
        assert importlib.metadata.version(distribution) == version

    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling_core.types.legacy_doc.base import BoundingBox
    from docling.document_converter import DocumentConverter, PdfFormatOption

    assert BoundingBox is not None
    assert DocumentConverter is not None
    assert PdfFormatOption(backend=PyPdfiumDocumentBackend).backend is PyPdfiumDocumentBackend


def test_installed_docling_profile_exposes_direct_structured_table_grid() -> None:
    """The pinned object model supports the no-Markdown extraction boundary."""

    if importlib.util.find_spec("docling") is None:
        pytest.skip("the optional document-preview-docling profile is not installed")

    from docling_core.types.doc import (
        BoundingBox,
        CoordOrigin,
        DocItemLabel,
        DoclingDocument,
        PageItem,
        ProvenanceItem,
        Size,
        TableCell,
        TableData,
    )

    bbox = BoundingBox(l=10, t=20, r=60, b=40, coord_origin=CoordOrigin.TOPLEFT)
    provenance = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, 1))
    table_data = TableData(
        num_rows=2,
        num_cols=1,
        table_cells=[
            TableCell(
                text="Item",
                bbox=bbox,
                start_row_offset_idx=0,
                end_row_offset_idx=1,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            ),
            TableCell(
                text="Widget",
                bbox=bbox,
                start_row_offset_idx=1,
                end_row_offset_idx=2,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
            ),
        ],
    )
    document = DoclingDocument(
        name="structured-test",
        pages={1: PageItem(page_no=1, size=Size(width=100, height=100))},
    )
    document.add_text(label=DocItemLabel.TEXT, text="Invoice", prov=provenance)
    document.add_table(data=table_data, prov=provenance)

    extraction = runtime_module._extract_docling_structured_document(document)

    assert [span.text for span in extraction.pages[0].spans] == ["Invoice", "Item", "Widget"]
    assert [(cell.row_index, cell.column_index, cell.text) for cell in extraction.table_cells] == [
        (0, 0, "Item"),
        (1, 0, "Widget"),
    ]
