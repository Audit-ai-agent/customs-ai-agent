"""Validate an evaluator-only Evidence v2 conformance identity without mutation.

The production Agent lock remains ``unreleased`` in both canonical checkout
locations and in every wheel. This tool validates external Data conformance
artifacts against one exact, clean Agent commit and writes only an external
non-deployable manifest. It never writes an Agent file, builds a distribution,
invokes a provider, or creates a production release lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseIdentity,
    canonical_json_bytes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT_CONSUMER_LOCK = Path("contracts/evidence/runtime-release-v2.lock.json")
PACKAGED_CONSUMER_LOCK = Path(
    "src/customs_ai_agent/contracts/evidence/runtime-release-v2.lock.json"
)
ROOT_RETRIEVAL_SCHEMA = Path("contracts/evidence/v1/rag_retrieval_v1.schema.json")
PACKAGED_RETRIEVAL_SCHEMA = Path("src/customs_ai_agent/contracts/evidence/v1.schema.json")
ROOT_READINESS_SCHEMA = Path("contracts/evidence/runtime-release-v2.schema.json")
PACKAGED_READINESS_SCHEMA = Path(
    "src/customs_ai_agent/contracts/evidence/runtime-release-v2.schema.json"
)

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_LOCK_FIELDS = {
    "contract",
    "schema_version",
    "provider_release",
    "provider_revision",
    "provider_build_sha256",
    "retrieval_schema_sha256",
    "readiness_schema_sha256",
    "provider_source_manifest_sha256",
}
_UNRELEASED_LOCK_FIELDS = {
    "contract",
    "schema_version",
    "canonical_owner",
    "consumer",
    "provider_release",
    "retrieval_schema_version",
    "retrieval_schema_sha256",
    "readiness_schema_sha256",
    "release_state",
    "identity",
    "unreleased_reason",
}
_REPOSITORY_INPUTS = (
    ROOT_CONSUMER_LOCK,
    PACKAGED_CONSUMER_LOCK,
    ROOT_RETRIEVAL_SCHEMA,
    PACKAGED_RETRIEVAL_SCHEMA,
    ROOT_READINESS_SCHEMA,
    PACKAGED_READINESS_SCHEMA,
)


class ConformanceIdentityValidationError(RuntimeError):
    """A content-free conformance validation failure safe for operator output."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ConformanceIdentityValidation:
    identity: EvidenceRuntimeReleaseIdentity
    manifest: dict[str, object]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_object(payload: bytes, *, error_code: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError
            value[key] = item
        return value

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise ConformanceIdentityValidationError(error_code) from None
    if not isinstance(value, dict):
        raise ConformanceIdentityValidationError(error_code)
    return value


def _serialized_document(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def _git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise ConformanceIdentityValidationError("git_unavailable")
    return executable


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603 - resolved executable and fixed argv boundary
        [_git_executable(), "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and result.returncode:
        raise ConformanceIdentityValidationError("git_state_invalid")
    return result


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_regular_file(path: Path, *, error_code: str) -> bytes:
    if path.is_symlink():
        raise ConformanceIdentityValidationError(error_code)
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError
        return resolved.read_bytes()
    except OSError:
        raise ConformanceIdentityValidationError(error_code) from None


def _read_repository_file(
    repository: Path,
    relative_path: Path,
    *,
    error_code: str,
) -> bytes:
    candidate = repository / relative_path
    current = candidate
    while current != repository:
        if current.is_symlink():
            raise ConformanceIdentityValidationError(error_code)
        current = current.parent
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise ConformanceIdentityValidationError(error_code) from None
    if not _is_within(resolved, repository):
        raise ConformanceIdentityValidationError(error_code)
    return _read_regular_file(candidate, error_code=error_code)


def _read_external_input(path: Path, repository: Path, *, error_code: str) -> bytes:
    if path.is_symlink():
        raise ConformanceIdentityValidationError(error_code)
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ConformanceIdentityValidationError(error_code) from None
    if _is_within(resolved, repository):
        raise ConformanceIdentityValidationError(error_code)
    return _read_regular_file(resolved, error_code=error_code)


def _external_manifest_output(path: Path, repository: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise ConformanceIdentityValidationError("manifest_output_exists")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError:
        raise ConformanceIdentityValidationError("manifest_parent_invalid") from None
    if not parent.is_dir() or _is_within(parent, repository):
        raise ConformanceIdentityValidationError("manifest_output_must_be_external")
    git_probe = _git(parent, "rev-parse", "--git-dir", check=False)
    if git_probe.returncode == 0:
        raise ConformanceIdentityValidationError("manifest_output_git_repository_forbidden")
    return parent / path.name


def _assert_exact_clean_head(repository: Path, expected_base_commit: str) -> str:
    if _GIT_SHA.fullmatch(expected_base_commit) is None:
        raise ConformanceIdentityValidationError("base_commit_invalid")
    top_level = _git(repository, "rev-parse", "--show-toplevel").stdout.strip()
    try:
        if Path(top_level).resolve(strict=True) != repository:
            raise ConformanceIdentityValidationError("repository_invalid")
    except OSError:
        raise ConformanceIdentityValidationError("repository_invalid") from None
    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    if head != expected_base_commit:
        raise ConformanceIdentityValidationError("base_commit_mismatch")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if status:
        raise ConformanceIdentityValidationError("clean_worktree_required")
    return head


def _verified_provider_lock(payload: dict[str, Any]) -> dict[str, str]:
    if (
        set(payload) != _PROVIDER_LOCK_FIELDS
        or payload.get("contract") != "evidence-runtime-release-provider-lock"
        or payload.get("schema_version") != "1.0"
        or payload.get("provider_release") != "evidence-api-v1.2.0"
    ):
        raise ConformanceIdentityValidationError("provider_runtime_lock_invalid")
    revision = payload.get("provider_revision")
    if not isinstance(revision, str) or _GIT_SHA.fullmatch(revision) is None:
        raise ConformanceIdentityValidationError("provider_runtime_lock_invalid")
    for field_name in (
        "provider_build_sha256",
        "retrieval_schema_sha256",
        "readiness_schema_sha256",
        "provider_source_manifest_sha256",
    ):
        value = payload.get(field_name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ConformanceIdentityValidationError("provider_runtime_lock_invalid")
    build_input = {
        "contract": "evidence-retrieval",
        "schema_version": payload["schema_version"],
        "provider_release": payload["provider_release"],
        "provider_revision": revision,
        "schema_sha256": payload["retrieval_schema_sha256"],
        "source_manifest_sha256": payload["provider_source_manifest_sha256"],
    }
    if _sha256(canonical_json_bytes(build_input)) != payload["provider_build_sha256"]:
        raise ConformanceIdentityValidationError("provider_runtime_lock_invalid")
    return payload  # type: ignore[return-value]


def _verified_identity(
    payload: dict[str, Any],
    provider_lock: Mapping[str, str],
) -> EvidenceRuntimeReleaseIdentity:
    try:
        identity = EvidenceRuntimeReleaseIdentity.from_mapping(payload)
    except ValueError:
        raise ConformanceIdentityValidationError("runtime_identity_invalid") from None
    comparisons = {
        "provider_release": identity.provider_release,
        "provider_revision": identity.provider_revision,
        "provider_build_sha256": identity.provider_build_sha256,
        "retrieval_schema_sha256": identity.retrieval_schema_sha256,
        "provider_source_manifest_sha256": identity.provider_source_manifest_sha256,
    }
    if any(provider_lock[field] != observed for field, observed in comparisons.items()):
        raise ConformanceIdentityValidationError("runtime_identity_provider_mismatch")
    return identity


def _verified_local_contracts(
    repository: Path,
    provider_lock: Mapping[str, str],
) -> tuple[bytes, bytes]:
    retrieval_root = _read_repository_file(
        repository,
        ROOT_RETRIEVAL_SCHEMA,
        error_code="retrieval_schema_invalid",
    )
    retrieval_package = _read_repository_file(
        repository,
        PACKAGED_RETRIEVAL_SCHEMA,
        error_code="retrieval_schema_invalid",
    )
    readiness_root = _read_repository_file(
        repository,
        ROOT_READINESS_SCHEMA,
        error_code="readiness_schema_invalid",
    )
    readiness_package = _read_repository_file(
        repository,
        PACKAGED_READINESS_SCHEMA,
        error_code="readiness_schema_invalid",
    )
    if (
        retrieval_root != retrieval_package
        or _sha256(retrieval_root) != provider_lock["retrieval_schema_sha256"]
    ):
        raise ConformanceIdentityValidationError("retrieval_schema_mismatch")
    if (
        readiness_root != readiness_package
        or _sha256(readiness_root) != provider_lock["readiness_schema_sha256"]
    ):
        raise ConformanceIdentityValidationError("readiness_schema_mismatch")
    for payload, error_code in (
        (retrieval_root, "retrieval_schema_invalid"),
        (readiness_root, "readiness_schema_invalid"),
    ):
        _strict_json_object(payload, error_code=error_code)
    return retrieval_root, readiness_root


def _verified_unreleased_consumer_locks(
    repository: Path,
    provider_lock: Mapping[str, str],
) -> bytes:
    root_bytes = _read_repository_file(
        repository,
        ROOT_CONSUMER_LOCK,
        error_code="consumer_lock_invalid",
    )
    package_bytes = _read_repository_file(
        repository,
        PACKAGED_CONSUMER_LOCK,
        error_code="consumer_lock_invalid",
    )
    if root_bytes != package_bytes:
        raise ConformanceIdentityValidationError("consumer_lock_copies_mismatch")
    payload = _strict_json_object(root_bytes, error_code="consumer_lock_invalid")
    if (
        set(payload) != _UNRELEASED_LOCK_FIELDS
        or payload.get("contract") != "evidence-runtime-release-consumer-lock"
        or payload.get("schema_version") != "1.0"
        or payload.get("canonical_owner") != "customs-data-platform"
        or payload.get("consumer") != "customs-ai-agent"
        or payload.get("provider_release") != provider_lock["provider_release"]
        or payload.get("retrieval_schema_version") != "1.0"
        or payload.get("retrieval_schema_sha256") != provider_lock["retrieval_schema_sha256"]
        or payload.get("readiness_schema_sha256") != provider_lock["readiness_schema_sha256"]
        or payload.get("release_state") != "unreleased"
        or payload.get("identity") is not None
        or not isinstance(payload.get("unreleased_reason"), str)
        or not payload["unreleased_reason"].strip()
    ):
        raise ConformanceIdentityValidationError("consumer_lock_not_unreleased")
    return root_bytes


def _repository_snapshot(repository: Path) -> dict[str, str]:
    return {
        path.as_posix(): _sha256(
            _read_repository_file(
                repository,
                path,
                error_code="repository_input_changed",
            )
        )
        for path in _REPOSITORY_INPUTS
    }


def _manifest(
    *,
    base_commit: str,
    provider_lock: Mapping[str, str],
    provider_lock_bytes: bytes,
    identity_bytes: bytes,
    identity: EvidenceRuntimeReleaseIdentity,
    consumer_lock_bytes: bytes,
    repository_snapshot: Mapping[str, str],
) -> dict[str, object]:
    return {
        "artifact_type": "agent-v2-provider-conformance-identity-validation",
        "schema_version": "1.0",
        "conformance_only": True,
        "deployable": False,
        "push_allowed": False,
        "tag_allowed": False,
        "release_allowed": False,
        "runtime_overlay": "evaluator-process-memory-only",
        "canonical_lock_release_state": "unreleased",
        "repository_mutated": False,
        "repository_clean_before": True,
        "repository_clean_after": True,
        "modified_agent_file_count": 0,
        "forbidden_actions": ["deploy", "push", "release", "tag"],
        "base_commit": base_commit,
        "provider_release": provider_lock["provider_release"],
        "provider_revision": provider_lock["provider_revision"],
        "provider_build_sha256": provider_lock["provider_build_sha256"],
        "provider_source_manifest_sha256": provider_lock["provider_source_manifest_sha256"],
        "provider_runtime_lock_sha256": _sha256(provider_lock_bytes),
        "runtime_identity_document_sha256": _sha256(identity_bytes),
        "retrieval_schema_sha256": provider_lock["retrieval_schema_sha256"],
        "readiness_schema_sha256": provider_lock["readiness_schema_sha256"],
        "runtime_release_sha256": identity.runtime_release_sha256,
        "canonical_unreleased_consumer_lock_sha256": _sha256(consumer_lock_bytes),
        "repository_input_set_sha256": _sha256(canonical_json_bytes(dict(repository_snapshot))),
    }


def _write_external_manifest(path: Path, payload: bytes) -> None:
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            path.unlink(missing_ok=True)
        raise ConformanceIdentityValidationError("manifest_write_failed") from None


def validate_conformance_identity(
    *,
    repository: Path,
    provider_runtime_lock_path: Path,
    runtime_identity_path: Path,
    manifest_output: Path,
    expected_base_commit: str,
    test_only_conformance: bool,
) -> ConformanceIdentityValidation:
    """Validate one local-only identity while preserving every Agent file."""

    if test_only_conformance is not True:
        raise ConformanceIdentityValidationError("test_only_acknowledgement_required")
    try:
        repository = repository.resolve(strict=True)
    except OSError:
        raise ConformanceIdentityValidationError("repository_invalid") from None
    base_commit = _assert_exact_clean_head(repository, expected_base_commit)
    manifest_output = _external_manifest_output(manifest_output, repository)
    before_snapshot = _repository_snapshot(repository)

    provider_lock_bytes = _read_external_input(
        provider_runtime_lock_path,
        repository,
        error_code="provider_runtime_lock_invalid",
    )
    identity_bytes = _read_external_input(
        runtime_identity_path,
        repository,
        error_code="runtime_identity_invalid",
    )
    provider_lock = _verified_provider_lock(
        _strict_json_object(
            provider_lock_bytes,
            error_code="provider_runtime_lock_invalid",
        )
    )
    identity = _verified_identity(
        _strict_json_object(identity_bytes, error_code="runtime_identity_invalid"),
        provider_lock,
    )
    _verified_local_contracts(repository, provider_lock)
    consumer_lock_bytes = _verified_unreleased_consumer_locks(repository, provider_lock)

    _assert_exact_clean_head(repository, expected_base_commit)
    after_snapshot = _repository_snapshot(repository)
    if after_snapshot != before_snapshot:
        raise ConformanceIdentityValidationError("repository_input_changed")
    manifest = _manifest(
        base_commit=base_commit,
        provider_lock=provider_lock,
        provider_lock_bytes=provider_lock_bytes,
        identity_bytes=identity_bytes,
        identity=identity,
        consumer_lock_bytes=consumer_lock_bytes,
        repository_snapshot=before_snapshot,
    )
    _write_external_manifest(manifest_output, _serialized_document(manifest))
    try:
        _assert_exact_clean_head(repository, expected_base_commit)
        if _repository_snapshot(repository) != before_snapshot:
            raise ConformanceIdentityValidationError("repository_input_changed")
    except ConformanceIdentityValidationError:
        manifest_output.unlink(missing_ok=True)
        raise
    return ConformanceIdentityValidation(identity=identity, manifest=manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-only-conformance",
        action="store_true",
        required=True,
        help="Required acknowledgement that the validated identity is evaluator-only.",
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--provider-runtime-lock", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validation = validate_conformance_identity(
            repository=args.repository,
            provider_runtime_lock_path=args.provider_runtime_lock,
            runtime_identity_path=args.runtime_identity,
            manifest_output=args.manifest_output,
            expected_base_commit=args.base_commit,
            test_only_conformance=args.test_only_conformance,
        )
    except ConformanceIdentityValidationError as exc:
        print(f"conformance_identity_validation_failed={exc.code}")
        return 1
    print(f"conformance_only={str(validation.manifest['conformance_only']).lower()}")
    print(f"deployable={str(validation.manifest['deployable']).lower()}")
    print(f"repository_mutated={str(validation.manifest['repository_mutated']).lower()}")
    print(f"runtime_release_sha256={validation.identity.runtime_release_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
