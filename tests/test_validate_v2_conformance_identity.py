from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from customs_ai_agent.contracts.evidence_runtime_release import canonical_json_bytes
import tools.validate_v2_conformance_identity as validator_module
from tools.validate_v2_conformance_identity import (
    ConformanceIdentityValidationError,
    build_parser,
    validate_conformance_identity,
)


ROOT_LOCK = Path("contracts/evidence/runtime-release-v2.lock.json")
PACKAGE_LOCK = Path("src/customs_ai_agent/contracts/evidence/runtime-release-v2.lock.json")
ROOT_RETRIEVAL = Path("contracts/evidence/v1/rag_retrieval_v1.schema.json")
PACKAGE_RETRIEVAL = Path("src/customs_ai_agent/contracts/evidence/v1.schema.json")
ROOT_READINESS = Path("contracts/evidence/runtime-release-v2.schema.json")
PACKAGE_READINESS = Path("src/customs_ai_agent/contracts/evidence/runtime-release-v2.schema.json")


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _provider_lock(
    *,
    retrieval_schema: bytes,
    readiness_schema: bytes,
) -> dict[str, str]:
    source_manifest_sha256 = "a" * 64
    payload = {
        "contract": "evidence-retrieval",
        "schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": "1" * 40,
        "schema_sha256": hashlib.sha256(retrieval_schema).hexdigest(),
        "source_manifest_sha256": source_manifest_sha256,
    }
    provider_build_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {
        "contract": "evidence-runtime-release-provider-lock",
        "schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": "1" * 40,
        "provider_build_sha256": provider_build_sha256,
        "retrieval_schema_sha256": hashlib.sha256(retrieval_schema).hexdigest(),
        "readiness_schema_sha256": hashlib.sha256(readiness_schema).hexdigest(),
        "provider_source_manifest_sha256": source_manifest_sha256,
    }


def _runtime_identity(provider_lock: dict[str, str]) -> dict[str, str]:
    payload = {
        "contract": "evidence-runtime-release",
        "schema_version": "2.0",
        "retrieval_schema_version": "1.0",
        "provider_release": provider_lock["provider_release"],
        "provider_revision": provider_lock["provider_revision"],
        "provider_build_sha256": provider_lock["provider_build_sha256"],
        "retrieval_schema_sha256": provider_lock["retrieval_schema_sha256"],
        "provider_source_manifest_sha256": provider_lock["provider_source_manifest_sha256"],
        "knowledge_generation_id": "2" * 64,
        "knowledge_generation_attestation_sha256": "3" * 64,
        "knowledge_source_manifest_sha256": "4" * 64,
        "model_manifest_sha256": "5" * 64,
        "index_config_sha256": "6" * 64,
        "evaluation_report_sha256": "7" * 64,
        "seal_attestation_sha256": "8" * 64,
        "knowledge_snapshot_id": "test-only-conformance-snapshot",
        "supported_review_as_of": "2026-07-10",
    }
    payload["runtime_release_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _unreleased_lock(provider_lock: dict[str, str]) -> dict[str, object]:
    return {
        "contract": "evidence-runtime-release-consumer-lock",
        "schema_version": "1.0",
        "canonical_owner": "customs-data-platform",
        "consumer": "customs-ai-agent",
        "provider_release": provider_lock["provider_release"],
        "retrieval_schema_version": "1.0",
        "retrieval_schema_sha256": provider_lock["retrieval_schema_sha256"],
        "readiness_schema_sha256": provider_lock["readiness_schema_sha256"],
        "release_state": "unreleased",
        "identity": None,
        "unreleased_reason": "No real sealed Data release is approved.",
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "agent-repository"
    external = tmp_path / "external-inputs"
    repository.mkdir()
    external.mkdir()
    retrieval_schema = b'{"type":"object","title":"retrieval"}\n'
    readiness_schema = b'{"type":"object","title":"readiness"}\n'
    provider_lock = _provider_lock(
        retrieval_schema=retrieval_schema,
        readiness_schema=readiness_schema,
    )
    for path in (ROOT_RETRIEVAL, PACKAGE_RETRIEVAL):
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(retrieval_schema)
    for path in (ROOT_READINESS, PACKAGE_READINESS):
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(readiness_schema)
    for path in (ROOT_LOCK, PACKAGE_LOCK):
        _write_json(repository / path, _unreleased_lock(provider_lock))
    provider_lock_path = external / "provider-runtime-lock.json"
    identity_path = external / "runtime-identity.json"
    _write_json(provider_lock_path, provider_lock)
    _write_json(identity_path, _runtime_identity(provider_lock))

    _run_git(repository, "init", "-q")
    _run_git(repository, "config", "user.name", "Conformance Test")
    _run_git(repository, "config", "user.email", "conformance@example.invalid")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "-q", "-m", "fixture")
    return {
        "repository": repository,
        "external": external,
        "provider_lock": provider_lock,
        "provider_lock_path": provider_lock_path,
        "identity_path": identity_path,
        "manifest_path": external / "validation-manifest.json",
        "base_commit": _run_git(repository, "rev-parse", "HEAD"),
    }


def _validate(fixture: dict[str, Any]):
    return validate_conformance_identity(
        repository=fixture["repository"],
        provider_runtime_lock_path=fixture["provider_lock_path"],
        runtime_identity_path=fixture["identity_path"],
        manifest_output=fixture["manifest_path"],
        expected_base_commit=fixture["base_commit"],
        test_only_conformance=True,
    )


def test_validates_identity_and_writes_only_external_non_deployable_manifest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    repository: Path = fixture["repository"]
    before_tree = {
        path: (repository / path).read_bytes()
        for path in (
            ROOT_LOCK,
            PACKAGE_LOCK,
            ROOT_RETRIEVAL,
            PACKAGE_RETRIEVAL,
            ROOT_READINESS,
            PACKAGE_READINESS,
        )
    }

    validation = _validate(fixture)

    manifest_bytes = fixture["manifest_path"].read_bytes()
    parsed_manifest = json.loads(manifest_bytes)
    assert parsed_manifest == validation.manifest
    assert validation.identity.to_wire() == json.loads(fixture["identity_path"].read_bytes())
    assert validation.manifest["conformance_only"] is True
    assert validation.manifest["deployable"] is False
    assert validation.manifest["push_allowed"] is False
    assert validation.manifest["tag_allowed"] is False
    assert validation.manifest["release_allowed"] is False
    assert validation.manifest["runtime_overlay"] == "evaluator-process-memory-only"
    assert validation.manifest["canonical_lock_release_state"] == "unreleased"
    assert validation.manifest["repository_mutated"] is False
    assert validation.manifest["repository_clean_before"] is True
    assert validation.manifest["repository_clean_after"] is True
    assert validation.manifest["modified_agent_file_count"] == 0
    assert validation.manifest["forbidden_actions"] == ["deploy", "push", "release", "tag"]
    assert str(tmp_path) not in manifest_bytes.decode("utf-8")
    assert "path" not in parsed_manifest
    assert _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert _run_git(repository, "diff", "--name-only") == ""
    assert {path: (repository / path).read_bytes() for path in before_tree} == before_tree
    assert json.loads((repository / ROOT_LOCK).read_bytes())["release_state"] == "unreleased"
    assert json.loads((repository / PACKAGE_LOCK).read_bytes())["release_state"] == "unreleased"


def test_requires_explicit_cli_acknowledgement() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--base-commit",
                "1" * 40,
                "--provider-runtime-lock",
                "provider.json",
                "--runtime-identity",
                "identity.json",
                "--manifest-output",
                "manifest.json",
            ]
        )


def test_function_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(
        ConformanceIdentityValidationError,
        match="test_only_acknowledgement_required",
    ):
        validate_conformance_identity(
            repository=fixture["repository"],
            provider_runtime_lock_path=fixture["provider_lock_path"],
            runtime_identity_path=fixture["identity_path"],
            manifest_output=fixture["manifest_path"],
            expected_base_commit=fixture["base_commit"],
            test_only_conformance=False,
        )

    assert not fixture["manifest_path"].exists()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("dirty", "clean_worktree_required"),
        ("base", "base_commit_mismatch"),
        ("provider_extra", "provider_runtime_lock_invalid"),
        ("identity_mismatch", "runtime_identity_provider_mismatch"),
        ("identity_hash", "runtime_identity_invalid"),
        ("schema", "retrieval_schema_mismatch"),
        ("consumer_copy", "consumer_lock_copies_mismatch"),
        ("consumer_released", "consumer_lock_not_unreleased"),
        ("manifest_exists", "manifest_output_exists"),
        ("manifest_internal", "manifest_output_must_be_external"),
        ("manifest_git_repository", "manifest_output_git_repository_forbidden"),
    ],
)
def test_rejects_unsafe_or_unattested_validation(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _fixture(tmp_path)
    repository: Path = fixture["repository"]
    if mutation == "dirty":
        (repository / "unrelated.txt").write_text("dirty", encoding="utf-8")
    elif mutation == "base":
        fixture["base_commit"] = "f" * 40
    elif mutation == "provider_extra":
        payload = json.loads(fixture["provider_lock_path"].read_bytes())
        payload["unexpected"] = True
        _write_json(fixture["provider_lock_path"], payload)
    elif mutation == "identity_mismatch":
        payload = json.loads(fixture["identity_path"].read_bytes())
        payload["provider_revision"] = "f" * 40
        hash_input = {
            key: value for key, value in payload.items() if key != "runtime_release_sha256"
        }
        payload["runtime_release_sha256"] = hashlib.sha256(
            canonical_json_bytes(hash_input)
        ).hexdigest()
        _write_json(fixture["identity_path"], payload)
    elif mutation == "identity_hash":
        payload = json.loads(fixture["identity_path"].read_bytes())
        payload["runtime_release_sha256"] = "f" * 64
        _write_json(fixture["identity_path"], payload)
    elif mutation == "schema":
        (repository / PACKAGE_RETRIEVAL).write_bytes(b'{"type":"array"}\n')
        _run_git(repository, "add", PACKAGE_RETRIEVAL.as_posix())
        _run_git(repository, "commit", "-q", "-m", "schema mutation")
        fixture["base_commit"] = _run_git(repository, "rev-parse", "HEAD")
    elif mutation == "consumer_copy":
        (repository / PACKAGE_LOCK).write_bytes(b"{}\n")
        _run_git(repository, "add", PACKAGE_LOCK.as_posix())
        _run_git(repository, "commit", "-q", "-m", "consumer copy mutation")
        fixture["base_commit"] = _run_git(repository, "rev-parse", "HEAD")
    elif mutation == "consumer_released":
        payload = json.loads((repository / ROOT_LOCK).read_bytes())
        payload["release_state"] = "released"
        payload["identity"] = _runtime_identity(fixture["provider_lock"])
        payload.pop("unreleased_reason")
        for path in (ROOT_LOCK, PACKAGE_LOCK):
            _write_json(repository / path, payload)
        _run_git(repository, "add", ROOT_LOCK.as_posix(), PACKAGE_LOCK.as_posix())
        _run_git(repository, "commit", "-q", "-m", "released mutation")
        fixture["base_commit"] = _run_git(repository, "rev-parse", "HEAD")
    elif mutation == "manifest_exists":
        fixture["manifest_path"].write_text("existing", encoding="utf-8")
    elif mutation == "manifest_internal":
        fixture["manifest_path"] = repository / "validation-manifest.json"
    elif mutation == "manifest_git_repository":
        other_repository = fixture["external"] / "other-repository"
        other_repository.mkdir()
        _run_git(other_repository, "init", "-q")
        fixture["manifest_path"] = other_repository / "validation-manifest.json"

    with pytest.raises(ConformanceIdentityValidationError, match=expected_code):
        _validate(fixture)

    if mutation != "manifest_exists":
        assert not fixture["manifest_path"].exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation requires host policy")
def test_rejects_symlinked_external_identity(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    real_identity = fixture["identity_path"]
    linked_identity = fixture["external"] / "identity-link.json"
    linked_identity.symlink_to(real_identity)
    fixture["identity_path"] = linked_identity

    with pytest.raises(ConformanceIdentityValidationError, match="runtime_identity_invalid"):
        _validate(fixture)

    assert not fixture["manifest_path"].exists()


def test_manifest_create_race_never_deletes_an_existing_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"other-process-owned")

    with pytest.raises(ConformanceIdentityValidationError, match="manifest_write_failed"):
        validator_module._write_external_manifest(manifest, b"validator-owned")

    assert manifest.read_bytes() == b"other-process-owned"
