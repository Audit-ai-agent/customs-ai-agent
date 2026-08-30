from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import socket
from typing import Any, Callable, Sequence

import pytest

from tools import raster_text_oci_metadata as metadata


_DETECTOR_NAME = "PP-OCRv5_mobile_det"
_RECOGNIZER_NAME = "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
_MODEL_FILENAMES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
_CPYTHON_LICENSE_TEXT = "Fixture CPython license evidence.\nhttps://www.python.org/psf/\n"


@dataclass(frozen=True, slots=True)
class MetadataFixture:
    root: Path
    profile: dict[str, Any]
    policy: Path
    parent_finder: Callable[[str], Sequence[importlib.metadata.Distribution]]


def _write_json(path: Path, value: object, *, trailing_lf: bool = False) -> str:
    raw = metadata.canonical_json_bytes(value) + (b"\n" if trailing_lf else b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _model_snapshot(artifacts: list[dict[str, object]], prefix: str) -> str:
    by_path = {str(item["relative_path"]): item for item in artifacts}
    rows = [
        {
            "byte_size": by_path[f"{prefix}/{name}"]["byte_size"],
            "relative_path": (
                f"models/{prefix}/{name}" if prefix == "text-det" else f"{prefix}/{name}"
            ),
            "sha256": by_path[f"{prefix}/{name}"]["sha256"],
        }
        for name in _MODEL_FILENAMES
    ]
    return hashlib.sha256(metadata.canonical_json_bytes({"artifacts": rows})).hexdigest()


def _write_distribution(
    root: Path,
    *,
    name: str,
    version: str,
    license_expression: str | None,
) -> None:
    dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    fields = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"Project-URL: Source, https://github.com/example/{name}",
    ]
    if license_expression is not None:
        fields.append(f"License-Expression: {license_expression}")
    (dist_info / "METADATA").write_text("\n".join(fields) + "\n", encoding="utf-8")
    (dist_info / "RECORD").write_text(
        f"{dist_info.name}/METADATA,,\n{dist_info.name}/RECORD,,\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> MetadataFixture:
    root = tmp_path / "document-preview"
    sidecar_root = Path("runtime/sidecar")
    module_root = root / sidecar_root / "module-root"
    interpreter_root = root / sidecar_root / "interpreter"
    _write_distribution(
        module_root,
        name="fixture-sidecar",
        version="1.0",
        license_expression="MIT",
    )
    parent_metadata = tmp_path / "parent-metadata"
    _write_distribution(
        parent_metadata,
        name="fixture-parent",
        version="2.0",
        license_expression="Apache-2.0",
    )
    parent_distributions = tuple(importlib.metadata.distributions(path=[str(parent_metadata)]))

    header = interpreter_root / "include/python3.11/patchlevel.h"
    header.parent.mkdir(parents=True)
    header.write_text(
        "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 11\n#define PY_MICRO_VERSION 15\n",
        encoding="utf-8",
    )
    (interpreter_root / "LICENSE.txt").write_bytes(_CPYTHON_LICENSE_TEXT.encode("utf-8"))

    parent_inventory_path = root / sidecar_root / "inventories/parent.json"
    parent_inventory_sha = _write_json(
        parent_inventory_path,
        {"packages": [{"name": "fixture-parent", "version": "2.0"}]},
    )
    sidecar_inventory_path = root / sidecar_root / "inventories/sidecar.json"
    sidecar_inventory_sha = _write_json(
        sidecar_inventory_path,
        {"packages": [{"name": "fixture-sidecar", "version": "1.0"}]},
    )

    model_root = Path("runtime/models")
    detector_hashes = [f"{index + 10:064x}" for index in range(4)]
    recognizer_hashes = [f"{index + 20:064x}" for index in range(4)]
    model_artifacts = [
        *[
            {
                "byte_size": index + 1,
                "relative_path": f"text-det/{name}",
                "sha256": digest,
            }
            for index, (name, digest) in enumerate(zip(_MODEL_FILENAMES, detector_hashes))
        ],
        *[
            {
                "byte_size": index + 5,
                "relative_path": f"text-rec/{name}",
                "sha256": digest,
            }
            for index, (name, digest) in enumerate(zip(_MODEL_FILENAMES, recognizer_hashes))
        ],
    ]
    detector_snapshot = _model_snapshot(model_artifacts, "text-det")
    recognizer_snapshot = _model_snapshot(model_artifacts, "text-rec")
    model_inventory_path = root / model_root / "inventories/models.json"
    model_inventory_sha = _write_json(
        model_inventory_path,
        {
            "artifacts": model_artifacts,
            "text_detection_model_name": _DETECTOR_NAME,
            "text_recognition_model_name": _RECOGNIZER_NAME,
        },
    )
    source_inventory_path = root / "model-inventories/source.json"
    upstream_recognizer_hashes = [f"{index + 30:064x}" for index in range(4)]
    upstream_recognizer_source = (
        "https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec/"
        "tree/267c36e24c331595590fe7bd72bde2436fd286f2"
    )
    source_rows = [
        *[
            {
                "license_id": "Apache-2.0",
                "relative_path": f"models/paddleocr3/det/file-{index}",
                "sha256": digest,
                "source_uri": "https://github.com/example/fixture-detector",
            }
            for index, digest in enumerate(detector_hashes)
        ],
        *[
            {
                "license_id": "Apache-2.0",
                "relative_path": f"models/paddleocr3/rec/{name}",
                "sha256": digest,
                "source_uri": upstream_recognizer_source,
            }
            for name, digest in zip(_MODEL_FILENAMES, upstream_recognizer_hashes)
        ],
    ]
    source_inventory_sha = _write_json(
        source_inventory_path,
        {"artifacts": source_rows},
        trailing_lf=True,
    )
    mobile_manifest_path = root / model_root / "provenance/mobile-manifest.json"
    _write_json(
        mobile_manifest_path,
        {
            "source_model_inventory_relative_path": "model-inventories/source.json",
            "source_model_inventory_sha256": source_inventory_sha,
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        },
    )
    upstream_artifacts = {
        f"models/text-rec/{name}": digest
        for name, digest in zip(_MODEL_FILENAMES, upstream_recognizer_hashes)
    }
    receipt: dict[str, object] = {
        "initialization": "pinned_public_inference_graph_warm_start",
        "public_mobile_source_inventory_sha256": source_inventory_sha,
        "public_recognizer_artifact_sha256s": upstream_artifacts,
        "receipt_sha256": "",
        "tuning_scope": "head",
    }
    receipt_sha = hashlib.sha256(metadata.canonical_json_bytes(receipt)).hexdigest()
    receipt["receipt_sha256"] = receipt_sha
    receipt_path = root / model_root / "provenance/training-receipt.json"
    _write_json(receipt_path, receipt)
    combined_manifest_path = root / model_root / "manifest.json"
    combined_manifest: dict[str, object] = {
        "files": [
            {
                "byte_size": mobile_manifest_path.stat().st_size,
                "path": "provenance/mobile-manifest.json",
                "sha256": hashlib.sha256(mobile_manifest_path.read_bytes()).hexdigest(),
            },
            {
                "byte_size": receipt_path.stat().st_size,
                "path": "provenance/training-receipt.json",
                "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            },
        ],
        "mobile_manifest_provenance_relative_path": "provenance/mobile-manifest.json",
        "model_inventory_relative_path": "inventories/models.json",
        "recognizer_receipt_provenance_relative_path": "provenance/training-receipt.json",
        "source_detector_model_snapshot_sha256": detector_snapshot,
        "source_recognizer_receipt_sha256": receipt_sha,
        "source_recognizer_model_snapshot_sha256": recognizer_snapshot,
    }
    combined_manifest_sha = _write_json(combined_manifest_path, combined_manifest)
    profile: dict[str, Any] = {
        "mobile_model_bundle_manifest_sha256": combined_manifest_sha,
        "mobile_model_bundle_root_relative_path": model_root.as_posix(),
        "model_inventory_path": (model_root / "inventories/models.json").as_posix(),
        "model_inventory_sha256": model_inventory_sha,
        "parent_renderer_package_inventory_path": parent_inventory_path.relative_to(
            root
        ).as_posix(),
        "parent_renderer_package_inventory_sha256": parent_inventory_sha,
        "raster_text_profile_revision": "synthetic-raster-text-v1:" + "a" * 24,
        "sidecar_interpreter_root": (sidecar_root / "interpreter").as_posix(),
        "sidecar_interpreter_root_tree_sha256": "c" * 64,
        "sidecar_module_root": (sidecar_root / "module-root").as_posix(),
        "sidecar_package_inventory_path": sidecar_inventory_path.relative_to(root).as_posix(),
        "sidecar_package_inventory_sha256": sidecar_inventory_sha,
    }
    policy = tmp_path / "runtime-license-policy.json"
    _write_json(
        policy,
        {
            "cpython": [
                {
                    "license_sha256": hashlib.sha256(
                        _CPYTHON_LICENSE_TEXT.encode("utf-8")
                    ).hexdigest(),
                    "source_uri": "https://github.com/python/cpython/tree/v3.11.15",
                    "version": "3.11.15",
                }
            ],
            "distribution_scope": "private-ghcr-only",
            "fine_tuned_models": [
                {
                    "combined_model_manifest_sha256": combined_manifest_sha,
                    "distribution_scope": "private-ghcr-only",
                    "license_id": "NOASSERTION",
                    "model_name": _RECOGNIZER_NAME,
                    "model_snapshot_sha256": recognizer_snapshot,
                    "source_uri": "NOASSERTION",
                    "upstream_base_artifacts_sha256": hashlib.sha256(
                        metadata.canonical_json_bytes(upstream_artifacts)
                    ).hexdigest(),
                    "upstream_base_license_id": "Apache-2.0",
                    "upstream_base_model_name": "en_PP-OCRv5_mobile_rec",
                    "upstream_base_source_inventory_sha256": source_inventory_sha,
                    "upstream_base_source_uri": upstream_recognizer_source,
                    "upstream_training_receipt_sha256": receipt_sha,
                }
            ],
            "schema_version": "1.0",
        },
        trailing_lf=True,
    )

    def parent_finder(_name: str) -> Sequence[importlib.metadata.Distribution]:
        return parent_distributions

    return MetadataFixture(root, profile, policy, parent_finder)


def _build(fixture: MetadataFixture) -> metadata.RuntimeMetadataArtifacts:
    return metadata.build_runtime_metadata(
        fixture.root,
        fixture.profile,
        policy_path=fixture.policy,
        distribution_finder=fixture.parent_finder,
    )


def _write_generated(
    fixture: MetadataFixture, artifacts: metadata.RuntimeMetadataArtifacts
) -> tuple[Path, Path]:
    deployment = fixture.root / "_deployment"
    deployment.mkdir(exist_ok=True)
    license_path = deployment / "third-party-licenses.json"
    sbom_path = deployment / "runtime.spdx.json"
    license_path.write_bytes(artifacts.license_manifest)
    sbom_path.write_bytes(artifacts.sbom)
    (deployment / "runtime-license-policy.json").write_bytes(artifacts.policy)
    for relative, payload in artifacts.generated_evidence:
        path = fixture.root.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return license_path, sbom_path


def test_metadata_is_deterministic_offline_and_exact_1_to_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def network_forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    first = _build(fixture)
    second = _build(fixture)
    assert first == second

    licenses = json.loads(first.license_manifest)
    sbom = json.loads(first.sbom)
    assert licenses["distribution_scope"] == "private-ghcr-only"
    expected = {
        (item.scope, item.name.casefold(), item.version)
        for item in metadata.actual_runtime_components(fixture.root, fixture.profile)
    }
    license_set = {
        (item["scope"], item["name"].casefold(), item["version"]) for item in licenses["components"]
    }
    spdx_set = {
        (
            item["summary"].removeprefix("customs-runtime-scope:"),
            item["name"].casefold(),
            item["versionInfo"],
        )
        for item in sbom["packages"]
    }
    assert len(expected) == 5
    assert expected == license_set == spdx_set
    assert sbom["spdxVersion"] == "SPDX-2.3"
    recognizer_license = next(
        item for item in licenses["components"] if item["name"] == _RECOGNIZER_NAME
    )
    recognizer_spdx = next(item for item in sbom["packages"] if item["name"] == _RECOGNIZER_NAME)
    assert recognizer_license["license_id"] == "NOASSERTION"
    assert recognizer_license["source_uri"] == "NOASSERTION"
    assert recognizer_spdx["licenseDeclared"] == "NOASSERTION"
    assert recognizer_spdx["licenseConcluded"] == "NOASSERTION"
    assert recognizer_spdx["downloadLocation"] == "NOASSERTION"
    assert first.source_evidence_paths == (PurePosixPath("model-inventories/source.json"),)

    license_path, sbom_path = _write_generated(fixture, first)
    metadata.validate_metadata_component_sets(
        fixture.root,
        fixture.profile,
        license_manifest_path=license_path,
        sbom_path=sbom_path,
        distribution_finder=fixture.parent_finder,
    )


def test_missing_or_new_finetuned_model_hash_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    policy = json.loads(fixture.policy.read_bytes())
    policy["fine_tuned_models"] = []
    _write_json(fixture.policy, policy, trailing_lf=True)
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_finetuned_model_license_policy_missing"

    fixture = _fixture(tmp_path / "new-hash")
    inventory_path = fixture.root / fixture.profile["model_inventory_path"]
    inventory = json.loads(inventory_path.read_bytes())
    recognizer_record = next(
        item
        for item in inventory["artifacts"]
        if item["relative_path"] == "text-rec/inference.pdiparams"
    )
    recognizer_record["sha256"] = "f" * 64
    fixture.profile["model_inventory_sha256"] = _write_json(inventory_path, inventory)
    new_snapshot = _model_snapshot(inventory["artifacts"], "text-rec")
    manifest_path = fixture.root / "runtime/models/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["source_recognizer_model_snapshot_sha256"] = new_snapshot
    fixture.profile["mobile_model_bundle_manifest_sha256"] = _write_json(manifest_path, manifest)
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_finetuned_model_license_policy_missing"


def test_noassertion_policy_is_private_only_and_upstream_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    policy = json.loads(fixture.policy.read_bytes())
    policy["distribution_scope"] = "public"
    policy["fine_tuned_models"][0]["distribution_scope"] = "public"
    _write_json(fixture.policy, policy, trailing_lf=True)
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_license_policy_invalid"

    fixture = _fixture(tmp_path / "upstream-drift")
    source_path = fixture.root / "model-inventories/source.json"
    source = json.loads(source_path.read_bytes())
    recognizer_row = next(
        row
        for row in source["artifacts"]
        if str(row["relative_path"]).startswith("models/paddleocr3/rec/")
    )
    recognizer_row["license_id"] = "MIT"
    source_path.write_bytes(metadata.canonical_json_bytes(source) + b"\n")
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_model_license_evidence_invalid"


def test_coordinated_fake_model_snapshot_and_policy_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest_path = fixture.root / "runtime/models/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["source_recognizer_model_snapshot_sha256"] = "f" * 64
    combined_sha = _write_json(manifest_path, manifest)
    fixture.profile["mobile_model_bundle_manifest_sha256"] = combined_sha
    policy = json.loads(fixture.policy.read_bytes())
    policy["fine_tuned_models"][0]["combined_model_manifest_sha256"] = combined_sha
    policy["fine_tuned_models"][0]["model_snapshot_sha256"] = "f" * 64
    _write_json(fixture.policy, policy, trailing_lf=True)
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_model_inventory_invalid"


def test_package_inventory_and_license_drift_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    artifacts = _build(fixture)
    license_path, sbom_path = _write_generated(fixture, artifacts)

    sidecar_metadata = next(
        (fixture.root / "runtime/sidecar/module-root").glob("*.dist-info/METADATA")
    )
    sidecar_metadata.write_text(
        sidecar_metadata.read_text(encoding="utf-8").replace(
            "License-Expression: MIT", "License-Expression: Apache-2.0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata.validate_metadata_component_sets(
            fixture.root,
            fixture.profile,
            license_manifest_path=license_path,
            sbom_path=sbom_path,
            distribution_finder=fixture.parent_finder,
        )
    assert exc_info.value.code == "runtime_license_evidence_mismatch"

    fixture = _fixture(tmp_path / "inventory-drift")
    inventory_path = fixture.root / fixture.profile["sidecar_package_inventory_path"]
    inventory = json.loads(inventory_path.read_bytes())
    inventory["packages"].append({"name": "undeclared-runtime", "version": "9"})
    fixture.profile["sidecar_package_inventory_sha256"] = _write_json(inventory_path, inventory)
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_package_metadata_set_mismatch"


def test_parent_and_sidecar_inventory_identity_overlap_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    inventory_path = fixture.root / fixture.profile["sidecar_package_inventory_path"]
    inventory = json.loads(inventory_path.read_bytes())
    inventory["packages"].append({"name": "fixture-parent", "version": "2.0"})
    fixture.profile["sidecar_package_inventory_sha256"] = _write_json(inventory_path, inventory)

    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_component_identity_duplicate"


def test_coordinated_fake_license_and_source_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    artifacts = _build(fixture)
    license_path, sbom_path = _write_generated(fixture, artifacts)
    licenses = json.loads(license_path.read_bytes())
    sbom = json.loads(sbom_path.read_bytes())
    detector_license = next(
        item for item in licenses["components"] if item["name"] == _DETECTOR_NAME
    )
    detector_license["license_id"] = "MIT"
    detector_license["source_uri"] = "https://github.com/example/forged-detector"
    detector_spdx = next(item for item in sbom["packages"] if item["name"] == _DETECTOR_NAME)
    detector_spdx["licenseConcluded"] = "MIT"
    detector_spdx["licenseDeclared"] = "MIT"
    detector_spdx["downloadLocation"] = "https://github.com/example/forged-detector"
    license_path.write_bytes(metadata.canonical_json_bytes(licenses))
    sbom_path.write_bytes(metadata.canonical_json_bytes(sbom))
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata.validate_metadata_component_sets(
            fixture.root,
            fixture.profile,
            license_manifest_path=license_path,
            sbom_path=sbom_path,
            distribution_finder=fixture.parent_finder,
        )
    assert exc_info.value.code == "runtime_license_evidence_mismatch"

    detector_license["source_uri"] = "https://source.example.invalid/model"
    license_path.write_bytes(metadata.canonical_json_bytes(licenses))
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata.validate_metadata_component_sets(
            fixture.root,
            fixture.profile,
            license_manifest_path=license_path,
            sbom_path=sbom_path,
            distribution_finder=fixture.parent_finder,
        )
    assert exc_info.value.code == "runtime_component_source_uri_invalid"


def test_unknown_or_ambiguous_package_license_without_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    sidecar_metadata = next(
        (fixture.root / "runtime/sidecar/module-root").glob("*.dist-info/METADATA")
    )
    sidecar_metadata.write_text(
        sidecar_metadata.read_text(encoding="utf-8").replace(
            "License-Expression: MIT\n", "License: mystery-license\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_package_license_missing"


def test_legacy_license_uses_british_spelling_license_file_as_exact_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    dist_info = next((fixture.root / "runtime/sidecar/module-root").glob("*.dist-info"))
    metadata_path = dist_info / "METADATA"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8").replace(
            "License-Expression: MIT\n", "License: MPL-2.0 AND MIT\n"
        ),
        encoding="utf-8",
    )
    licence_text = "Exact legacy licence evidence.\n"
    licence_path = dist_info / "licenses/LICENCE"
    licence_path.parent.mkdir()
    licence_path.write_bytes(licence_text.encode("utf-8"))
    (dist_info / "RECORD").write_text(
        "../../../bin/fixture-sidecar,,\n"
        f"{dist_info.name}/METADATA,,\n"
        f"{dist_info.name}/RECORD,,\n"
        f"{dist_info.name}/licenses/LICENCE,,\n",
        encoding="utf-8",
    )

    artifacts = _build(fixture)
    licenses = json.loads(artifacts.license_manifest)
    sidecar = next(item for item in licenses["components"] if item["name"] == "fixture-sidecar")
    assert sidecar["license_id"].startswith("LicenseRef-PackageEvidence-")
    sbom = json.loads(artifacts.sbom)
    extracted = {
        item["licenseId"]: item["extractedText"] for item in sbom["hasExtractedLicensingInfos"]
    }
    assert licence_text in extracted[sidecar["license_id"]]


def test_license_directory_and_embedded_sbom_are_exact_package_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    dist_info = next((fixture.root / "runtime/sidecar/module-root").glob("*.dist-info"))
    metadata_path = dist_info / "METADATA"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8").replace(
            "License-Expression: MIT\n", "License: legacy-value-is-not-spdx\n"
        ),
        encoding="utf-8",
    )
    terms = dist_info / "licenses/Apache-2.0.txt"
    terms.parent.mkdir()
    terms.write_text("Exact bundled license terms.\n", encoding="utf-8")
    sbom = dist_info / "sboms/package.cyclonedx.json"
    sbom.parent.mkdir()
    sbom.write_bytes(
        metadata.canonical_json_bytes(
            {
                "bomFormat": "CycloneDX",
                "components": [
                    {
                        "licenses": [{"expression": "Apache-2.0"}],
                        "name": "fixture-sidecar",
                        "version": "1.0",
                    }
                ],
            }
        )
    )
    (dist_info / "RECORD").write_text(
        f"{dist_info.name}/METADATA,,\n"
        f"{dist_info.name}/RECORD,,\n"
        f"{dist_info.name}/licenses/Apache-2.0.txt,,\n"
        f"{dist_info.name}/sboms/package.cyclonedx.json,,\n",
        encoding="utf-8",
    )

    artifacts = _build(fixture)
    sidecar = next(
        item
        for item in json.loads(artifacts.license_manifest)["components"]
        if item["name"] == "fixture-sidecar"
    )
    assert sidecar["license_id"] == "Apache-2.0"


def test_license_less_matching_sbom_component_does_not_override_metadata(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    dist_info = next((fixture.root / "runtime/sidecar/module-root").glob("*.dist-info"))
    sbom = dist_info / "sboms/wheel.cyclonedx.json"
    sbom.parent.mkdir()
    sbom.write_bytes(
        metadata.canonical_json_bytes(
            {
                "bomFormat": "CycloneDX",
                "components": [{"name": "fixture-sidecar", "version": "1.0"}],
            }
        )
    )
    (dist_info / "RECORD").write_text(
        f"{dist_info.name}/METADATA,,\n"
        f"{dist_info.name}/RECORD,,\n"
        f"{dist_info.name}/sboms/wheel.cyclonedx.json,,\n",
        encoding="utf-8",
    )

    sidecar = next(
        item
        for item in json.loads(_build(fixture).license_manifest)["components"]
        if item["name"] == "fixture-sidecar"
    )
    assert sidecar["license_id"] == "MIT"


def test_distribution_cannot_claim_another_dist_info_license(tmp_path: Path) -> None:
    module_root = tmp_path / "module-root"
    _write_distribution(
        module_root,
        name="package-a",
        version="1.0",
        license_expression=None,
    )
    _write_distribution(
        module_root,
        name="package-b",
        version="1.0",
        license_expression="MIT",
    )
    package_a = module_root / "package_a-1.0.dist-info"
    package_b = module_root / "package_b-1.0.dist-info"
    (package_b / "LICENSE").write_text("Package B only.\n", encoding="utf-8")
    (package_a / "RECORD").write_text(
        f"{package_a.name}/RECORD,,\n{package_b.name}/METADATA,,\n{package_b.name}/LICENSE,,\n",
        encoding="utf-8",
    )
    distribution = next(
        item
        for item in importlib.metadata.distributions(path=[str(module_root)])
        if item.metadata["Name"] == "package-a"
    )
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata._distribution_evidence(distribution)
    assert exc_info.value.code == "runtime_package_metadata_invalid"


def test_distribution_cannot_borrow_license_through_intermediate_symlink(
    tmp_path: Path,
) -> None:
    module_root = tmp_path / "module-root"
    _write_distribution(
        module_root,
        name="package-a",
        version="1.0",
        license_expression=None,
    )
    _write_distribution(
        module_root,
        name="package-b",
        version="1.0",
        license_expression="MIT",
    )
    package_a = module_root / "package_a-1.0.dist-info"
    package_b = module_root / "package_b-1.0.dist-info"
    (package_b / "LICENSE").write_text("Package B only.\n", encoding="utf-8")
    try:
        (package_a / "licenses").symlink_to(
            Path("..") / package_b.name,
            target_is_directory=True,
        )
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")
    (package_a / "RECORD").write_text(
        f"{package_a.name}/METADATA,,\n"
        f"{package_a.name}/RECORD,,\n"
        f"{package_a.name}/licenses/LICENSE,,\n",
        encoding="utf-8",
    )
    distribution = next(
        item
        for item in importlib.metadata.distributions(path=[str(module_root)])
        if item.metadata["Name"] == "package-a"
    )

    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata._distribution_evidence(distribution)
    assert exc_info.value.code == "runtime_package_license_invalid"


def test_source_registry_fallback_is_safe_and_identity_bound() -> None:
    evidence = metadata.DistributionEvidence(
        name="Example_Package",
        version="1.2+local",
        metadata_sha256="a" * 64,
        license_expressions=("MIT",),
        legacy_licenses=(),
        project_urls=("Source, http://legacy.example.org/project",),
        home_pages=(),
        sbom_license_evidence=(),
        license_files=(),
    )
    assert (
        metadata._metadata_source_uri(evidence)
        == "https://pypi.org/project/example-package/1.2%2Blocal/"
    )

    forged = metadata.DistributionEvidence(
        name=evidence.name,
        version=evidence.version,
        metadata_sha256=evidence.metadata_sha256,
        license_expressions=evidence.license_expressions,
        legacy_licenses=(),
        project_urls=("Source, https://github.com/example/source\nforged",),
        home_pages=(),
        sbom_license_evidence=(),
        license_files=(),
    )
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata._metadata_source_uri(forged)
    assert exc_info.value.code == "runtime_component_source_uri_invalid"


def test_parent_evidence_source_and_runtime_checksums_cannot_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifacts = _build(fixture)
    license_path, sbom_path = _write_generated(fixture, artifacts)
    evidence_path = fixture.root / "_deployment/parent-package-evidence.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence["distributions"][0]["project_urls"] = [
        "Source, https://github.com/example/forged-parent"
    ]
    evidence_path.write_bytes(metadata.canonical_json_bytes(evidence))
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata.validate_metadata_component_sets(
            fixture.root,
            fixture.profile,
            license_manifest_path=license_path,
            sbom_path=sbom_path,
            distribution_finder=fixture.parent_finder,
        )
    assert exc_info.value.code == "runtime_parent_package_evidence_mismatch"

    fixture = _fixture(tmp_path / "checksum-drift")
    artifacts = _build(fixture)
    license_path, sbom_path = _write_generated(fixture, artifacts)
    licenses = json.loads(license_path.read_bytes())
    sbom = json.loads(sbom_path.read_bytes())
    runtime_license = next(item for item in licenses["components"] if item["scope"] == "runtime")
    runtime_license["checksum_sha256"] = "f" * 64
    runtime_spdx = next(
        item for item in sbom["packages"] if item["summary"] == "customs-runtime-scope:runtime"
    )
    runtime_spdx["checksums"][0]["checksumValue"] = "f" * 64
    license_path.write_bytes(metadata.canonical_json_bytes(licenses))
    sbom_path.write_bytes(metadata.canonical_json_bytes(sbom))
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata.validate_metadata_component_sets(
            fixture.root,
            fixture.profile,
            license_manifest_path=license_path,
            sbom_path=sbom_path,
            distribution_finder=fixture.parent_finder,
        )
    assert exc_info.value.code == "runtime_license_evidence_mismatch"


def test_cpython_license_text_requires_exact_hash_bound_policy(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    license_path = fixture.root / "runtime/sidecar/interpreter/LICENSE.txt"
    license_path.write_bytes((_CPYTHON_LICENSE_TEXT + "unapproved change\n").encode("utf-8"))
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_cpython_license_policy_missing"


@pytest.mark.parametrize(
    "value",
    (
        "https://github.com/foo\nbar",
        "https://github.com/foo\tbar",
        "https://github.com/foo bar",
        "https://github.com/foo\\bar",
        "https://GITHUB.com/example/source",
        "https://github.com/example/%2fsource",
    ),
)
def test_noncanonical_or_control_character_source_uri_fails_closed(value: str) -> None:
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata._valid_source_uri(value)
    assert exc_info.value.code == "runtime_component_source_uri_invalid"


def test_nonfinite_policy_json_fails_with_stable_error(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.policy.write_bytes(b'{"cpython":NaN,"fine_tuned_models":[],"schema_version":"1.0"}\n')
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        _build(fixture)
    assert exc_info.value.code == "runtime_license_policy_invalid"


def test_sbom_and_license_sets_must_match_actual_inventory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    artifacts = _build(fixture)
    license_path, sbom_path = _write_generated(fixture, artifacts)
    licenses = json.loads(license_path.read_bytes())
    licenses["components"] = licenses["components"][:-1]
    license_path.write_bytes(metadata.canonical_json_bytes(licenses))
    with pytest.raises(metadata.RuntimeMetadataError) as exc_info:
        metadata.validate_metadata_component_sets(
            fixture.root,
            fixture.profile,
            license_manifest_path=license_path,
            sbom_path=sbom_path,
            distribution_finder=fixture.parent_finder,
        )
    assert exc_info.value.code == "runtime_license_component_set_mismatch"


def test_exporter_defaults_to_automatic_metadata_and_rejects_manual_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    exporter = importlib.import_module("export_raster_text_oci_runtime")
    parsed = exporter._parser().parse_args(
        [
            "--document-preview-root",
            str(tmp_path / "source"),
            "--destination",
            str(tmp_path / "output"),
        ]
    )
    assert parsed.runtime_sbom is None
    assert parsed.runtime_license_manifest is None

    generated = b'{"automatic":true}'
    compatible = tmp_path / "compatible.json"
    compatible.write_bytes(generated)
    assert exporter._metadata_payload(compatible, generated) == generated
    compatible.write_bytes(b'{"automatic":fals}')
    with pytest.raises(exporter.RuntimeExportError) as exc_info:
        exporter._metadata_payload(compatible, generated)
    assert exc_info.value.code == "deployment_metadata_input_mismatch"

    monkeypatch.setattr(exporter.platform, "system", lambda: "Linux")
    monkeypatch.setattr(exporter.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(exporter.sys, "version_info", (3, 12))
    with pytest.raises(exporter.RuntimeExportError) as exc_info:
        exporter._assert_linux_amd64_profile(tmp_path, object())
    assert exc_info.value.code == "linux_cp311_builder_required"
