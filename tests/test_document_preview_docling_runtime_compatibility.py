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

    from docling_core.types.legacy_doc.base import BoundingBox
    from docling.document_converter import DocumentConverter

    assert BoundingBox is not None
    assert DocumentConverter is not None
