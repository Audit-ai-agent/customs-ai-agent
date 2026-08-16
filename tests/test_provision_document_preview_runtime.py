from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import provision_document_preview_runtime as provision_tool


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _docling_configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": "docling",
        "extractor_revision": "docling-test-v1",
        "options": {
            "artifacts_directory": "models/docling",
            "pdf_backend": "pypdfium2",
            "ocr_enabled": False,
            "device": "cpu",
            "num_threads": 1,
            "do_table_structure": True,
            "table_structure_mode": "fast",
        },
    }


def _pdf_configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": "pdf_text_layer",
        "extractor_revision": "pypdf-5.3.1-text-layer-v1",
        "options": {},
    }


def _inventory(kind: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": kind,
        "packages": [
            {
                "name": "pypdf",
                "version": "test-v1",
                "license_id": "test-license",
                "source_uri": "https://example.test/pypdf/",
            }
        ],
        "artifacts": [],
    }


def test_docling_provision_runs_native_preflight_after_sidecar_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_path = root / "staging" / "configuration.json"
    inventory_path = root / "staging" / "inventory.json"
    _write_json(configuration_path, _docling_configuration())
    _write_json(inventory_path, _inventory("docling"))
    calls: list[str] = []

    monkeypatch.setattr(provision_tool, "_check_free_space", lambda *_args: None)
    monkeypatch.setattr(
        provision_tool,
        "write_runtime_sidecars",
        lambda **_kwargs: ("a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        provision_tool,
        "preflight_provisioned_docling_native_runtime",
        lambda *, document_preview_root, configuration: calls.append(
            f"{document_preview_root.name}:{configuration.engine_kind}"
        ),
    )

    assert provision_tool.provision_runtime(
        document_preview_root=root,
        configuration_path=configuration_path,
        inventory_path=inventory_path,
        cache_directory=None,
        minimum_free_gib=12.0,
    ) == ("a" * 64, "b" * 64)
    assert calls == ["document-preview:docling"]


def test_non_docling_provision_skips_native_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_path = root / "staging" / "configuration.json"
    inventory_path = root / "staging" / "inventory.json"
    _write_json(configuration_path, _pdf_configuration())
    _write_json(inventory_path, _inventory("pdf_text_layer"))

    monkeypatch.setattr(provision_tool, "_check_free_space", lambda *_args: None)
    monkeypatch.setattr(
        provision_tool,
        "write_runtime_sidecars",
        lambda **_kwargs: ("a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        provision_tool,
        "preflight_provisioned_docling_native_runtime",
        lambda **_kwargs: pytest.fail("non-Docling provision must not run Docling preflight"),
    )

    provision_tool.provision_runtime(
        document_preview_root=root,
        configuration_path=configuration_path,
        inventory_path=inventory_path,
        cache_directory=None,
        minimum_free_gib=12.0,
    )
