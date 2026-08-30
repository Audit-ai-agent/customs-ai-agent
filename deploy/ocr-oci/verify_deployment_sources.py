"""Static fail-closed checks for the asset-free OCR OCI deployment sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


class DeploymentSourceError(RuntimeError):
    pass


def _reject_nonfinite_json(_value: str) -> None:
    raise ValueError


def _require(text: str, pattern: str, code: str) -> None:
    if re.search(pattern, text, flags=re.MULTILINE) is None:
        raise DeploymentSourceError(code)


def verify(repository_root: Path) -> None:
    root = repository_root.resolve(strict=True)
    dockerfile = (root / "deploy/ocr-oci/Dockerfile").read_text(encoding="utf-8")
    runtime_dockerfile = (root / "deploy/ocr-oci/runtime-context.Dockerfile").read_text(
        encoding="utf-8"
    )
    placeholder_dockerfile = (root / "deploy/ocr-oci/placeholder/Dockerfile").read_text(
        encoding="utf-8"
    )
    workflow = (root / ".github/workflows/ocr-oci-publish.yml").read_text(encoding="utf-8")
    deployment_readme = (root / "deploy/ocr-oci/README.md").read_text(encoding="utf-8")
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    schema_path = root / "deploy/ocr-oci/runtime-bundle-manifest.schema.json"
    schema = json.loads(
        schema_path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise DeploymentSourceError("runtime_manifest_schema_invalid")
    try:
        maximum_files = schema["properties"]["files"]["maxItems"]
        roles = schema["$defs"]["fileRecord"]["properties"]["role"]["enum"]
    except (KeyError, TypeError):
        raise DeploymentSourceError("runtime_manifest_schema_invalid") from None
    if maximum_files != 24_999:
        raise DeploymentSourceError("runtime_manifest_file_limit_invalid")
    if not isinstance(roles, list) or "component-evidence" not in roles:
        raise DeploymentSourceError("runtime_manifest_schema_invalid")
    policy_path = root / "deploy/ocr-oci/runtime-license-policy.json"
    policy_raw = policy_path.read_bytes()
    policy = json.loads(policy_raw, parse_constant=_reject_nonfinite_json)
    canonical_policy = (
        json.dumps(
            policy,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    expected_policy = {
        "cpython": [
            {
                "license_sha256": (
                    "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf"
                ),
                "source_uri": "https://github.com/python/cpython/tree/v3.11.15",
                "version": "3.11.15",
            }
        ],
        "distribution_scope": "private-ghcr-only",
        "fine_tuned_models": [
            {
                "combined_model_manifest_sha256": (
                    "3b276c1a83740f73949d319095034bfdba01285aed3318d1f4cdcb7190e18185"
                ),
                "distribution_scope": "private-ghcr-only",
                "license_id": "NOASSERTION",
                "model_name": "source_attested_en_PP-OCRv5_mobile_rec_head_v2",
                "model_snapshot_sha256": (
                    "2a41badb75dc0b66c8597429e585a440a0d25e8a52c771df73b8e0c90ac908ad"
                ),
                "source_uri": "NOASSERTION",
                "upstream_base_artifacts_sha256": (
                    "0cfa8addcff71284ac996182e9edb39f7ee95bf7d651acfc39458cf89bc9a656"
                ),
                "upstream_base_license_id": "Apache-2.0",
                "upstream_base_model_name": "en_PP-OCRv5_mobile_rec",
                "upstream_base_source_inventory_sha256": (
                    "57d566b507b7fff67ef0cb3eb605671d862facb2f8aa4765f74bab546baf130d"
                ),
                "upstream_base_source_uri": (
                    "https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec/"
                    "tree/267c36e24c331595590fe7bd72bde2436fd286f2"
                ),
                "upstream_training_receipt_sha256": (
                    "b09e8a208407af78e6e1b3535a25abacce3a765f952d73dbfbc68193cdaa42a6"
                ),
            }
        ],
        "schema_version": "1.0",
    }
    if policy_raw != canonical_policy or policy != expected_policy:
        raise DeploymentSourceError("runtime_license_policy_invalid")

    first_rule = next(
        (
            line.strip()
            for line in dockerignore.splitlines()
            if line.strip() and not line.startswith("#")
        ),
        None,
    )
    if first_rule != "**":
        raise DeploymentSourceError("docker_context_not_allowlisted")
    for required in (
        "!pyproject.toml",
        "!uv.lock",
        "!src/**",
        "!deploy/ocr-oci/**",
        "!tools/raster_text_oci_evidence.py",
        "!tools/raster_text_oci_metadata.py",
        "!tools/verify_raster_text_oci_runtime.py",
    ):
        if required not in dockerignore.splitlines():
            raise DeploymentSourceError("docker_context_allowlist_incomplete")

    _require(dockerfile, r"^# syntax=docker/dockerfile:1\.10$", "docker_syntax_unpinned")
    _require(
        dockerfile,
        r"^ARG PYTHON_IMAGE=python:3\.11\.[0-9]+-slim-bookworm@sha256:[a-f0-9]{64}$",
        "python_base_not_digest_pinned",
    )
    _require(
        dockerfile,
        r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:[^@\s]+@sha256:[a-f0-9]{64}$",
        "uv_base_not_digest_pinned",
    )
    _require(
        dockerfile,
        r"^ARG RASTER_RUNTIME_IMAGE=scratch$",
        "runtime_image_fail_closed_default_missing",
    )
    for literal in (
        "FROM ${RASTER_RUNTIME_IMAGE} AS raster-runtime",
        "COPY --from=raster-runtime --link --chown=0:0",
        "UV_PROJECT_ENVIRONMENT=/opt/customs/app/.venv",
        "#!/opt/customs/app/.venv/bin/python",
        "tools/verify_raster_text_oci_runtime.py",
        "verify_agent_runtime_binding.py",
        "raster_text_oci_evidence.py",
        "raster_text_oci_metadata.py",
        "expected-runtime-license-policy.json",
        "--expected-license-policy /opt/customs/deploy/expected-runtime-license-policy.json",
        "RUN --network=none",
        "install -d -o 10001 -g 10001 -m 0755 /opt/customs/deploy/verified",
        "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_AUTO_ACTIVATION=source-attested-synthetic-raster-text-v1",
        "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_PATH=/opt/customs/_local/data/document-preview/approvals/raster-text-active.json",
        "USER 10001:10001",
        "HEALTHCHECK",
        'ENTRYPOINT ["/opt/customs/app/.venv/bin/customs-ai-agent-serve"]',
    ):
        if literal not in dockerfile:
            raise DeploymentSourceError("release_dockerfile_boundary_incomplete")
    if re.search(r"^\s*(?:ADD\s+https?://|COPY\s+\.\s)", dockerfile, flags=re.MULTILINE):
        raise DeploymentSourceError("unbounded_docker_copy_or_remote_add")
    if "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_SHA256" in dockerfile:
        raise DeploymentSourceError("auto_activation_sha_environment_forbidden")
    if (
        re.search(
            r"FROM runtime-material AS runtime-verified.*?USER 10001:10001\s+"
            r"RUN --network=none",
            dockerfile,
            flags=re.DOTALL,
        )
        is None
    ):
        raise DeploymentSourceError("ocr_warm_verification_not_nonroot")
    package_block = re.search(
        r"apt-get install --yes --no-install-recommends\s+(.*?)\s+&&",
        dockerfile,
        flags=re.DOTALL,
    )
    if package_block is None:
        raise DeploymentSourceError("ocr_native_runtime_packages_missing")
    packages = set(package_block.group(1).replace("\\", " ").split())
    if packages != {
        "libgl1=1.6.0-1",
        "libglib2.0-0=2.74.6-2+deb12u9",
        "libgomp1=12.2.0-14+deb12u1",
    }:
        raise DeploymentSourceError("ocr_native_runtime_packages_invalid")
    for literal in (
        "rm -rf /var/lib/apt/lists/*",
        "libgomp.so.1",
        "libGL.so.1",
        "libgthread-2.0.so.0",
    ):
        if literal not in dockerfile:
            raise DeploymentSourceError("ocr_native_runtime_link_check_missing")

    if (
        "FROM scratch" not in runtime_dockerfile
        or "COPY --chown=0:0 document-preview/" not in runtime_dockerfile
    ):
        raise DeploymentSourceError("runtime_context_dockerfile_invalid")
    if re.search(r"^\s*(?:RUN|ADD)\s", runtime_dockerfile, flags=re.MULTILINE):
        raise DeploymentSourceError("runtime_context_dockerfile_not_data_only")
    if (
        placeholder_dockerfile.count("FROM scratch") != 1
        or "ARG SOURCE_REPOSITORY" not in placeholder_dockerfile
        or 'org.opencontainers.image.source="${SOURCE_REPOSITORY}"' not in placeholder_dockerfile
        or 'io.customs.artifact.kind="asset-free-ghcr-bootstrap-v1"' not in placeholder_dockerfile
        or re.search(r"^\s*(?:ADD|COPY|RUN)\s", placeholder_dockerfile, flags=re.MULTILINE)
        is not None
    ):
        raise DeploymentSourceError("ghcr_placeholder_not_asset_free")

    for literal in (
        "workflow_dispatch:",
        "operation:",
        "- bootstrap",
        "- publish",
        "tags:",
        "actions: read",
        "packages: write",
        "attestations: write",
        "confirm_source_sha:",
        "confirm_private_distribution:",
        "ref: refs/heads/main",
        '"${GITHUB_REF_TYPE}" != "branch"',
        '"${GITHUB_RUN_ATTEMPT}" != "1"',
        '"${GITHUB_ACTOR}" != "${GITHUB_TRIGGERING_ACTOR}"',
        '"${DISPATCH_CONFIRM_SOURCE_SHA}" != "${GITHUB_SHA}"',
        '"${DISPATCH_CONFIRM_PRIVATE_DISTRIBUTION}" != "true"',
        '"$(git rev-parse HEAD)" == "${GITHUB_SHA}"',
        "verify_github_release_actor.py",
        "deploy/ocr-oci/placeholder/Dockerfile",
        "asset-free-bootstrap-v1",
        "--provenance=false",
        "--sbom=false",
        "--allow-missing-repository-for-source-attested-bootstrap",
        "--max-attempts 6",
        "--retry-delay-seconds 2",
        "provenance: mode=max",
        "sbom: true",
        "RASTER_RUNTIME_IMAGE=${{ steps.release.outputs.runtime_image }}",
        "SOURCE_REVISION=${{ github.sha }}",
        "^sha256:[a-f0-9]{64}$",
    ):
        if literal not in workflow:
            raise DeploymentSourceError("publish_workflow_boundary_incomplete")
    if (
        workflow.count("verify_ghcr_private_package.py") != 4
        or workflow.count('--repository "${GITHUB_REPOSITORY}"') != 5
        or workflow.count("--allow-missing-repository-for-source-attested-bootstrap") != 1
        or workflow.count("--max-attempts 6") != 1
        or workflow.count("--retry-delay-seconds 2") != 1
    ):
        raise DeploymentSourceError("publish_package_link_verification_incomplete")
    if (
        re.search(
            r"^  (?:push|pull_request|pull_request_target|schedule):\s*$",
            workflow,
            flags=re.MULTILINE,
        )
        is not None
        or "ocr-oci-v" in workflow
    ):
        raise DeploymentSourceError("publish_non_manual_trigger_forbidden")
    if re.search(r"^\s+environment:\s*", workflow, flags=re.MULTILINE) is not None:
        raise DeploymentSourceError("publish_paid_environment_dependency_forbidden")
    if "ocr-oci-v" in deployment_readme:
        raise DeploymentSourceError("publish_documentation_tag_trigger_forbidden")
    # This build copies the private OCR/model runtime into intermediate layers.
    # An Actions cache is a weaker boundary than the private GHCR package and
    # mode=max would export those intermediate bytes. Keep every cache backend
    # out of this asset-bearing workflow; source-only builds may cache elsewhere.
    if any(
        marker in workflow for marker in ("cache-from", "cache-to", "type=gha", "actions/cache@")
    ):
        raise DeploymentSourceError("asset_bearing_publish_cache_forbidden")

    exporter = (root / "tools/export_raster_text_oci_runtime.py").read_text(encoding="utf-8")
    runtime_verifier = (root / "tools/verify_raster_text_oci_runtime.py").read_text(
        encoding="utf-8"
    )
    agent_verifier = (root / "deploy/ocr-oci/verify_agent_runtime_binding.py").read_text(
        encoding="utf-8"
    )
    runtime_publisher = (root / "deploy/ocr-oci/publish_runtime_context.sh").read_text(
        encoding="utf-8"
    )
    privacy_verifier = (root / "deploy/ocr-oci/verify_ghcr_private_package.py").read_text(
        encoding="utf-8"
    )
    actor_verifier = (root / "deploy/ocr-oci/verify_github_release_actor.py").read_text(
        encoding="utf-8"
    )
    readiness = (root / "deploy/ocr-oci/ocr_readiness.py").read_text(encoding="utf-8")
    if (
        "RasterTextActivationPointerV12" not in exporter
        or "RasterTextActivationPointerV11" in exporter
        or 'RuntimeExportError("schema12_active_pointer_required")' not in exporter
        or "build_runtime_metadata(" not in exporter
        or "deployment_metadata_input_mismatch" not in exporter
        or "runtime-license-policy.json" not in exporter
        or "--expected-license-policy" not in runtime_publisher
        or "deploy/ocr-oci/runtime-license-policy.json" not in runtime_publisher
        or "runtime_license_policy_mismatch" not in runtime_verifier
        or "_MAX_FILE_COUNT = 25_000" not in runtime_verifier
        or "validate_metadata_component_sets(" not in runtime_verifier
        or 'value.get("schema_version") != "12.0"' not in runtime_verifier
        or 'value.get("schema_version") != "9.0"' not in runtime_verifier
        or 'pointer.get("schema_version") != "12.0"' not in readiness
        or "_release_manifest_records(" not in runtime_verifier
        or 'elf_header[:7] != b"\\x7fELF\\x02\\x01\\x01"' not in runtime_verifier
        or "start_service_session=True" not in agent_verifier
        or "binding.verify_runtime_state()" not in agent_verifier
        or "verify_v9_direct_evidence_closure(" not in exporter
        or "verify_v9_direct_evidence_closure(" not in agent_verifier
        or "deploy/ocr-oci/verify_agent_runtime_binding.py" not in runtime_publisher
        or runtime_publisher.count("verify_ghcr_private_package.py") != 2
        or "api.github.com" not in privacy_verifier
        or 'metadata["visibility"] != "private"' not in privacy_verifier
        or 'metadata["version_count"]' not in privacy_verifier
        or "isinstance(version_count, bool)" not in privacy_verifier
        or 'if "repository" not in metadata:' not in privacy_verifier
        or "allow_missing_repository" not in privacy_verifier
        or "_GhcrMetadataPending" not in privacy_verifier
        or "time.sleep(retry_delay_seconds)" not in privacy_verifier
        or "_MAX_ATTEMPTS = 6" not in privacy_verifier
        or "/packages/container/" not in privacy_verifier
        or runtime_publisher.count('--repository "${source_repository}"') != 2
        or 'metadata.get("private") is not True' not in actor_verifier
        or 'metadata.get("visibility") != "private"' not in actor_verifier
        or 'metadata.get("default_branch") != "main"' not in actor_verifier
        or 'permission.get("permission") != "admin"' not in actor_verifier
        or 'user.get("type") != "User"' not in actor_verifier
    ):
        raise DeploymentSourceError("current_native_ocr_oci_boundary_incomplete")

    actor_check = workflow.find("verify_github_release_actor.py")
    placeholder_push = workflow.find("deploy/ocr-oci/placeholder/Dockerfile")
    final_build = workflow.find("Build and publish immutable Agent plus OCR image")
    if not 0 <= actor_check < placeholder_push < final_build:
        raise DeploymentSourceError("publish_verification_order_invalid")

    for relative in (
        "deploy/ocr-oci/create_app_license_manifest.py",
        "deploy/ocr-oci/ocr_readiness.py",
        "deploy/ocr-oci/selftest_github_release_actor.py",
        "deploy/ocr-oci/selftest_ocr_readiness.py",
        "deploy/ocr-oci/selftest_ghcr_private_package.py",
        "deploy/ocr-oci/selftest_runtime_contract.py",
        "deploy/ocr-oci/verify_ghcr_private_package.py",
        "deploy/ocr-oci/verify_github_release_actor.py",
        "deploy/ocr-oci/verify_agent_runtime_binding.py",
        "deploy/ocr-oci/verify_deployment_sources.py",
        "tools/export_raster_text_oci_runtime.py",
        "tools/raster_text_oci_evidence.py",
        "tools/raster_text_oci_metadata.py",
        "tools/verify_raster_text_oci_runtime.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        compile(source, relative, "exec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    try:
        verify(args.repository_root)
    except (
        DeploymentSourceError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        code = str(exc) if isinstance(exc, DeploymentSourceError) else "deployment_source_invalid"
        print(code, file=sys.stderr)
        return 1
    print("ocr_oci_deployment_sources_valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
