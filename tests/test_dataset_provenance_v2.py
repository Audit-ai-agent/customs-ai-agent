import hashlib
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.dataset_provenance import (
    DatasetProvenanceError,
    build_relocation_manifest,
    verify_dataset_provenance,
)


ROOT = Path(__file__).parents[1]
CONTRACT_ROOT = ROOT / "fixtures" / "dataset_provenance" / "v2"


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def test_repository_v2_contracts_are_schema_valid_and_hash_bound():
    binding_path = CONTRACT_ROOT / "relocation-binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_schema = json.loads(
        (CONTRACT_ROOT / "schemas" / "relocation-binding.schema.json").read_text(encoding="utf-8")
    )
    provenance_schema = json.loads(
        (CONTRACT_ROOT / "schemas" / "dataset-provenance.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(binding_schema).validate(binding)
    binding_hash = _sha256(binding_path.read_bytes())
    template_by_id = {
        "trade_document_triplets": CONTRACT_ROOT / "trade-document-triplets.provenance.json",
        "trade_documents_images": CONTRACT_ROOT / "trade-documents-images.provenance.json",
    }
    for entry in binding["datasets"]:
        template = template_by_id[entry["dataset_id"]]
        Draft202012Validator(provenance_schema).validate(
            json.loads(template.read_text(encoding="utf-8"))
        )
        assert _sha256(template.read_bytes()) == entry["provenance_sha256"]
    lock_paths = (
        ROOT / "evals" / "trade_document_triplets" / "v1" / "dataset.lock.json",
        ROOT / "fixtures" / "trade_documents_images" / "v1" / "dataset.lock.json",
    )
    for lock_path in lock_paths:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        assert lock["dataset_provenance_v2"]["binding_sha256"] == binding_hash


def _workspace_fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("test workspace\n", encoding="utf-8")
    repository = workspace / "_local" / "worktrees" / "TASK-TEST" / "ai-agent"
    contract_root = repository / "fixtures" / "dataset_provenance" / "v2"
    shutil.copytree(CONTRACT_ROOT / "schemas", contract_root / "schemas")
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")

    dataset_root = workspace / "_local" / "data" / "trade_document_triplets" / "source-v1"
    raw_root = dataset_root / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "one.json").write_bytes(b"one")
    (raw_root / "two.json").write_bytes(b"two-two")
    checksums = dataset_root / "checksums.sha256"
    checksums.write_text(
        "".join(
            f"{_sha256((raw_root / name).read_bytes())}  raw/{name}\n"
            for name in ("one.json", "two.json")
        ),
        encoding="utf-8",
    )
    inventory = dataset_root / "inventory.json"
    inventory.write_text("{}\n", encoding="utf-8")
    integrity = {
        "raw_file_count": 2,
        "raw_bytes": 10,
        "raw_checksums_workspace_relative_path": (
            "_local/data/trade_document_triplets/source-v1/checksums.sha256"
        ),
        "raw_checksums_sha256": _sha256(checksums.read_bytes()),
        "inventory_workspace_relative_path": (
            "_local/data/trade_document_triplets/source-v1/inventory.json"
        ),
        "inventory_sha256": _sha256(inventory.read_bytes()),
    }
    provenance = {
        "schema_version": "customs-dataset-provenance/v2",
        "provenance_revision": 2,
        "dataset": {"id": "trade_document_triplets", "version": "source-v1"},
        "classification": "test synthetic fixture",
        "canonical_storage": {
            "workspace_relative_path": ("_local/data/trade_document_triplets/source-v1"),
            "raw_workspace_relative_path": ("_local/data/trade_document_triplets/source-v1/raw"),
            "tracked_in_git": False,
            "immutable_raw": True,
            "runner_access_policy": "canonical_only_after_relocation_verification",
        },
        "origin": {
            "description": "test source",
            "generator": {"name": "unknown", "version": "unknown", "seed": "unknown"},
            "legacy_source_reference": {
                "workspace_relative_path": "legacy-triplets",
                "status": "retired",
                "preserved": False,
                "runner_eligible": False,
            },
            "legacy_canonical_registration": {
                "workspace_relative_path": (
                    "_artifacts/datasets/trade_document_triplets/source-v1"
                ),
                "status": "retired",
                "preserved": False,
                "runner_eligible": False,
                "provenance_v1_sha256": "1" * 64,
            },
        },
        "integrity": integrity,
        "truth_and_rights": {
            "ground_truth_status": "test_only",
            "generation_provenance_status": "unknown",
            "public_redistribution_rights": "not_confirmed",
            "raw_allowed_in_git_or_public_portfolio": False,
        },
        "migration": {
            "migration_id": "TASK-0028-dataset-canonical-relocation",
            "contract_issued_at": "2026-08-01T07:58:13+09:00",
            "relocation_manifest_workspace_relative_path": (
                "_local/data/dataset-relocation-manifest.v1.json"
            ),
            "completion_policy": "manifest_verified_and_both_legacy_paths_absent",
        },
    }
    provenance_path = dataset_root / "provenance.json"
    provenance_path.write_bytes(_json_bytes(provenance))
    entry = {
        "dataset_id": "trade_document_triplets",
        "dataset_version": "source-v1",
        "canonical_dataset_workspace_relative_path": (
            "_local/data/trade_document_triplets/source-v1"
        ),
        "canonical_raw_workspace_relative_path": (
            "_local/data/trade_document_triplets/source-v1/raw"
        ),
        "provenance_workspace_relative_path": (
            "_local/data/trade_document_triplets/source-v1/provenance.json"
        ),
        "provenance_sha256": _sha256(provenance_path.read_bytes()),
        "legacy_source_workspace_relative_path": "legacy-triplets",
        "legacy_canonical_workspace_relative_path": (
            "_artifacts/datasets/trade_document_triplets/source-v1"
        ),
        "raw_integrity": integrity,
    }
    binding = {
        "schema_version": "customs-dataset-relocation-binding/v1",
        "migration_id": "TASK-0028-dataset-canonical-relocation",
        "workspace_layout_version": "local-data/v1",
        "contract_issued_at": "2026-08-01T07:58:13+09:00",
        "manifest_workspace_relative_path": ("_local/data/dataset-relocation-manifest.v1.json"),
        "datasets": [entry],
    }
    binding_path = contract_root / "relocation-binding.json"
    binding_path.write_bytes(_json_bytes(binding))
    binding_sha256 = _sha256(binding_path.read_bytes())
    lock = {
        "dataset_id": "trade_document_triplets",
        "dataset_version": "source-v1",
        "dataset_provenance_v2": {
            "schema_version": "customs-dataset-provenance-binding/v2",
            "migration_id": "TASK-0028-dataset-canonical-relocation",
            "binding_repository_relative_path": (
                "fixtures/dataset_provenance/v2/relocation-binding.json"
            ),
            "binding_sha256": binding_sha256,
            "relocation_manifest_workspace_relative_path": (
                "_local/data/dataset-relocation-manifest.v1.json"
            ),
            "canonical_dataset_workspace_relative_path": entry[
                "canonical_dataset_workspace_relative_path"
            ],
            "canonical_raw_workspace_relative_path": entry["canonical_raw_workspace_relative_path"],
            "provenance_workspace_relative_path": entry["provenance_workspace_relative_path"],
            "legacy_source_workspace_relative_path": entry["legacy_source_workspace_relative_path"],
            "legacy_canonical_workspace_relative_path": entry[
                "legacy_canonical_workspace_relative_path"
            ],
            "root_reference_preserved": False,
            "root_reference_status": "retired",
            "fallback_to_legacy_paths_allowed": False,
        },
    }
    lock_path = repository / "fixtures" / "test" / "dataset.lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(_json_bytes(lock))
    manifest_entry = {
        "dataset_id": entry["dataset_id"],
        "dataset_version": entry["dataset_version"],
        "canonical_dataset_workspace_relative_path": entry[
            "canonical_dataset_workspace_relative_path"
        ],
        "canonical_raw_workspace_relative_path": entry["canonical_raw_workspace_relative_path"],
        "provenance_sha256": entry["provenance_sha256"],
        "raw_file_count": integrity["raw_file_count"],
        "raw_bytes": integrity["raw_bytes"],
        "raw_checksums_sha256": integrity["raw_checksums_sha256"],
        "inventory_sha256": integrity["inventory_sha256"],
        "legacy_source_reference_removed": True,
        "legacy_canonical_registration_removed": True,
        "canonical_hash_verification_passed": True,
    }
    manifest = {
        "schema_version": "customs-dataset-relocation-manifest/v1",
        "migration_id": "TASK-0028-dataset-canonical-relocation",
        "relocation_status": "completed_and_verified",
        "verified_at": "2026-08-01T08:00:00+09:00",
        "binding_repository_relative_path": (
            "fixtures/dataset_provenance/v2/relocation-binding.json"
        ),
        "binding_sha256": binding_sha256,
        "datasets": [manifest_entry],
    }
    manifest_path = workspace / "_local" / "data" / "dataset-relocation-manifest.v1.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return {
        "workspace": workspace,
        "lock": lock_path,
        "raw": raw_root,
        "provenance": provenance_path,
        "manifest": manifest_path,
    }


def test_verified_relocation_resolves_only_the_canonical_raw_tree(tmp_path):
    fixture = _workspace_fixture(tmp_path)

    result = verify_dataset_provenance(
        fixture["lock"],
        workspace_root=fixture["workspace"],
        expected_raw_root=fixture["raw"],
    )

    assert result.raw_root == fixture["raw"]
    assert result.provenance_path == fixture["provenance"]
    assert result.relocation_manifest_path == fixture["manifest"]


def test_manifest_builder_reconstructs_the_exact_verified_contract(tmp_path):
    fixture = _workspace_fixture(tmp_path)
    expected = json.loads(fixture["manifest"].read_text(encoding="utf-8"))

    observed = build_relocation_manifest(
        fixture["lock"].parents[2],
        fixture["workspace"],
        verified_at="2026-08-01T08:00:00+09:00",
    )

    assert observed == expected


def test_missing_relocation_manifest_fails_closed(tmp_path):
    fixture = _workspace_fixture(tmp_path)
    fixture["manifest"].unlink()

    with pytest.raises(DatasetProvenanceError) as error:
        verify_dataset_provenance(fixture["lock"], workspace_root=fixture["workspace"])

    assert error.value.code == "dataset_provenance_regular_file_required"


def test_tampered_provenance_fails_closed(tmp_path):
    fixture = _workspace_fixture(tmp_path)
    fixture["provenance"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(DatasetProvenanceError) as error:
        verify_dataset_provenance(fixture["lock"], workspace_root=fixture["workspace"])

    assert error.value.code == "dataset_provenance_v2_schema_invalid"


def test_same_size_raw_content_tampering_fails_closed(tmp_path):
    fixture = _workspace_fixture(tmp_path)
    (fixture["raw"] / "one.json").write_bytes(b"two")

    with pytest.raises(DatasetProvenanceError) as error:
        verify_dataset_provenance(fixture["lock"], workspace_root=fixture["workspace"])

    assert error.value.code == "dataset_canonical_raw_checksum_mismatch"


def test_retired_root_cannot_coexist_with_verified_canonical_data(tmp_path):
    fixture = _workspace_fixture(tmp_path)
    (fixture["workspace"] / "legacy-triplets").mkdir()

    with pytest.raises(DatasetProvenanceError) as error:
        verify_dataset_provenance(fixture["lock"], workspace_root=fixture["workspace"])

    assert error.value.code == "dataset_legacy_path_not_retired"


def test_runner_cannot_substitute_another_dataset_root(tmp_path):
    fixture = _workspace_fixture(tmp_path)
    fallback = fixture["workspace"] / "fallback"
    fallback.mkdir()

    with pytest.raises(DatasetProvenanceError) as error:
        verify_dataset_provenance(
            fixture["lock"],
            workspace_root=fixture["workspace"],
            expected_raw_root=fallback,
        )

    assert error.value.code == "dataset_runner_raw_root_mismatch"
