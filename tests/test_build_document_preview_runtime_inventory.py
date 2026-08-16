from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import pytest

from customs_ai_agent.service import document_preview_runtime as runtime_module
from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    parse_model_inventory,
    parse_runtime_configuration,
    verify_configured_model_directory_artifacts,
)
from tools.build_document_preview_runtime_inventory import (
    build_runtime_inventory,
    main,
    write_staging_inventory,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _package_version(_name: str) -> str:
    return "test-v1"


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace" / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _configuration(root: Path, engine_kind: str) -> tuple[Path, dict[str, object]]:
    if engine_kind == "pdf_text_layer":
        options: dict[str, object] = {}
        extractor_revision = "pdf-text-test-v1"
    elif engine_kind == "paddleocr":
        options = {
            "det_model_directory": "models/paddle/det",
            "rec_model_directory": "models/paddle/rec",
            "cls_model_directory": "models/paddle/cls",
            "ocr_version": "PP-OCRv4",
            "use_gpu": False,
            "cpu_threads": 1,
        }
        extractor_revision = "paddle-test-v1"
    else:
        options = {
            "artifacts_directory": "models/docling",
            "pdf_backend": "pypdfium2",
            "ocr_enabled": False,
            "device": "cpu",
            "num_threads": 1,
            "do_table_structure": True,
            "table_structure_mode": "fast",
        }
        extractor_revision = "docling-test-v1"
    value: dict[str, object] = {
        "schema_version": "1.0",
        "engine_kind": engine_kind,
        "extractor_revision": extractor_revision,
        "options": options,
    }
    path = root / "staging" / f"configuration-{engine_kind}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(_canonical(value))
    return path, value


def _configured_directories(configuration: dict[str, object]) -> list[str]:
    options = configuration["options"]
    assert isinstance(options, dict)
    return sorted(
        value
        for key, value in options.items()
        if key.endswith("_directory") and isinstance(value, str)
    )


def _write_model_files(root: Path, configuration: dict[str, object]) -> None:
    for index, relative_directory in enumerate(_configured_directories(configuration), start=1):
        path = root / relative_directory / "weights.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"test-model-{index}".encode("utf-8"))


def _provenance(root: Path, engine_kind: str, configuration: dict[str, object]) -> Path:
    package_names = sorted(runtime_module._required_packages(engine_kind))
    value: dict[str, object] = {
        "schema_version": "1.0",
        "engine_kind": engine_kind,
        "packages": [
            {
                "name": name,
                "license_id": "Test-License",
                "source_uri": f"https://example.test/packages/{name}",
            }
            for name in package_names
        ],
        "artifact_directories": [
            {
                "relative_directory": directory,
                "license_id": "Test-Model-License",
                "source_uri": f"https://example.test/models/{index}",
            }
            for index, directory in enumerate(_configured_directories(configuration), start=1)
        ],
    }
    path = root / "staging" / f"provenance-{engine_kind}.json"
    path.write_bytes(_canonical(value))
    return path


@pytest.mark.parametrize("engine_kind", ("pdf_text_layer", "paddleocr", "docling"))
def test_builds_exact_runtime_inventory_for_every_supported_engine(
    tmp_path: Path,
    engine_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    configuration_path, configuration_value = _configuration(root, engine_kind)
    _write_model_files(root, configuration_value)
    provenance_path = _provenance(root, engine_kind, configuration_value)
    original_configuration = configuration_path.read_bytes()
    if engine_kind == "paddleocr":
        # This builder test supplies synthetic package versions and a synthetic
        # model tree, not a real locked OpenCV installation.  Dedicated runtime
        # tests exercise the RECORD-backed cv2 verifier directly.
        monkeypatch.setattr(
            runtime_module,
            "_recorded_cv2_tree_matches_designated_provider",
            lambda: None,
        )

    inventory_value, configuration_sha256 = build_runtime_inventory(
        document_preview_root=root,
        configuration_path=configuration_path,
        provenance_path=provenance_path,
        package_version_reader=_package_version,
    )

    assert configuration_sha256 == _sha(_canonical(configuration_value))
    assert configuration_path.read_bytes() == original_configuration
    assert inventory_value["engine_kind"] == engine_kind
    packages = inventory_value["packages"]
    artifacts = inventory_value["artifacts"]
    assert isinstance(packages, list)
    assert isinstance(artifacts, list)
    assert {item["name"] for item in packages if isinstance(item, dict)} == set(
        runtime_module._required_packages(engine_kind)
    )
    assert len(artifacts) == len(_configured_directories(configuration_value))
    inventory = parse_model_inventory(_canonical(inventory_value))
    configuration = parse_runtime_configuration(_canonical(configuration_value))
    verify_configured_model_directory_artifacts(root, configuration, inventory)

    inventory_sha256 = write_staging_inventory(
        root=root,
        output_path=root / "staging" / f"inventory-{engine_kind}.json",
        inventory_value=inventory_value,
    )
    assert inventory_sha256 == _sha(_canonical(inventory_value))
    assert (root / "staging" / f"inventory-{engine_kind}.json").read_bytes() == _canonical(
        inventory_value
    )


def test_rejects_extra_artifact_directory_provenance(tmp_path: Path) -> None:
    root = _root(tmp_path)
    configuration_path, configuration_value = _configuration(root, "paddleocr")
    _write_model_files(root, configuration_value)
    provenance_path = _provenance(root, "paddleocr", configuration_value)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert isinstance(provenance["artifact_directories"], list)
    provenance["artifact_directories"].append(
        {
            "relative_directory": "models/unconfigured",
            "license_id": "Test-Model-License",
            "source_uri": "https://example.test/models/unconfigured",
        }
    )
    provenance_path.write_bytes(_canonical(provenance))

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_inventory_provenance_invalid",
    ):
        build_runtime_inventory(
            document_preview_root=root,
            configuration_path=configuration_path,
            provenance_path=provenance_path,
            package_version_reader=_package_version,
        )


def test_runtime_rejects_an_extra_file_added_after_inventory_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    configuration_path, configuration_value = _configuration(root, "paddleocr")
    _write_model_files(root, configuration_value)
    provenance_path = _provenance(root, "paddleocr", configuration_value)
    # The synthetic inventory build cannot resolve the real OpenCV wheel in
    # the default CI profile; direct runtime tests cover that exact boundary.
    monkeypatch.setattr(
        runtime_module,
        "_recorded_cv2_tree_matches_designated_provider",
        lambda: None,
    )
    inventory_value, _ = build_runtime_inventory(
        document_preview_root=root,
        configuration_path=configuration_path,
        provenance_path=provenance_path,
        package_version_reader=_package_version,
    )
    rogue = root / "models" / "paddle" / "det" / "unexpected.bin"
    rogue.write_bytes(b"unexpected")

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_model_artifact_invalid",
    ):
        verify_configured_model_directory_artifacts(
            root,
            parse_runtime_configuration(_canonical(configuration_value)),
            parse_model_inventory(_canonical(inventory_value)),
        )


def test_staging_writer_cannot_replace_a_protected_configuration(tmp_path: Path) -> None:
    root = _root(tmp_path)
    configuration_path, configuration_value = _configuration(root, "pdf_text_layer")
    provenance_path = _provenance(root, "pdf_text_layer", configuration_value)
    inventory_value, _ = build_runtime_inventory(
        document_preview_root=root,
        configuration_path=configuration_path,
        provenance_path=provenance_path,
        package_version_reader=_package_version,
    )
    protected_configuration = root / "staging" / "inventory-configuration.json"
    protected_configuration.write_bytes(_canonical(configuration_value))

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_inventory_write_failed",
    ):
        write_staging_inventory(
            root=root,
            output_path=protected_configuration,
            inventory_value=inventory_value,
            protected_paths=(protected_configuration,),
        )
    assert protected_configuration.read_bytes() == _canonical(configuration_value)


def test_rejects_symlinked_model_artifact(tmp_path: Path) -> None:
    root = _root(tmp_path)
    configuration_path, configuration_value = _configuration(root, "docling")
    _write_model_files(root, configuration_value)
    model_directory = root / "models" / "docling"
    link = model_directory / "linked.bin"
    try:
        os.symlink(model_directory / "weights.bin", link)
    except (NotImplementedError, OSError):
        pytest.skip("the current Windows test account cannot create file symlinks")
    provenance_path = _provenance(root, "docling", configuration_value)

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_model_artifact_invalid",
    ):
        build_runtime_inventory(
            document_preview_root=root,
            configuration_path=configuration_path,
            provenance_path=provenance_path,
            package_version_reader=_package_version,
        )


def test_builder_executes_under_the_no_network_guard(tmp_path: Path) -> None:
    root = _root(tmp_path)
    configuration_path, configuration_value = _configuration(root, "pdf_text_layer")
    provenance_path = _provenance(root, "pdf_text_layer", configuration_value)
    failures: list[str] = []

    def networked_package_reader(_name: str) -> str:
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=0.1)
        except OSError as error:
            failures.append(str(error))
            raise
        return "test-v1"

    with pytest.raises(
        DocumentPreviewRuntimeError,
        match="document_preview_runtime_dependency_unavailable",
    ):
        build_runtime_inventory(
            document_preview_root=root,
            configuration_path=configuration_path,
            provenance_path=provenance_path,
            package_version_reader=networked_package_reader,
        )
    assert any("local preview runtime forbids network access" in item for item in failures)


def test_cli_requires_explicit_local_build_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    configuration_path, configuration_value = _configuration(root, "pdf_text_layer")
    provenance_path = _provenance(root, "pdf_text_layer", configuration_value)

    assert (
        main(
            [
                "--document-preview-root",
                str(root),
                "--configuration",
                str(configuration_path),
                "--provenance",
                str(provenance_path),
            ]
        )
        == 2
    )

    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "code": "confirmation_required",
    }
