"""Dependency-free positive and negative tests for the OCI runtime verifier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from verify_raster_text_oci_runtime import (  # noqa: E402
    RuntimeContextError,
    verify_runtime_context,
)

_BINDING = "a" * 64
_PROFILE = f"synthetic-raster-text-v1:{'b' * 24}"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write(root: Path, relative: str, payload: bytes) -> tuple[str, int]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o644)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _build_valid(root: Path) -> None:
    values: dict[str, tuple[str, int, str]] = {}
    for relative, payload, role in (
        ("profiles/profile.json", b"{}", "runtime-profile"),
        ("evidence/sealed.json", b"{}", "activation-evidence"),
        ("evidence/revalidation-execution.json", b"{}", "activation-evidence"),
        ("evidence/revalidation-smoke.json", b"{}", "activation-evidence"),
        ("evidence/selection.json", b"{}", "activation-evidence"),
        (
            "runtime/inventories/packages.json",
            _canonical({"packages": [{"name": "test-python", "version": "1.0"}]}),
            "inventory",
        ),
        ("runtime/sidecar/bin/python3.11", b"\x7fELF-fake", "sidecar-runtime"),
        ("runtime/models/text-rec/model.bin", b"model", "model"),
    ):
        digest, size = _write(root, relative, payload)
        values[relative] = (digest, size, role)

    pointer = {
        "activation_kind": "source_attested_synthetic_raster_text",
        "current_runner_implementation_sha256": _BINDING,
        "raster_text_profile_revision": _PROFILE,
        "revalidation_candidate_execution_path": "evidence/revalidation-execution.json",
        "revalidation_candidate_execution_sha256": values["evidence/revalidation-execution.json"][
            0
        ],
        "revalidation_smoke_receipt_path": "evidence/revalidation-smoke.json",
        "revalidation_smoke_receipt_sha256": values["evidence/revalidation-smoke.json"][0],
        "runtime_configuration_path": "profiles/profile.json",
        "runtime_configuration_sha256": values["profiles/profile.json"][0],
        "runtime_validation_revision": "activation-full-runtime-compact-v1",
        "schema_version": "11.0",
        "sealed_evaluation_path": "evidence/sealed.json",
        "sealed_evaluation_sha256": values["evidence/sealed.json"][0],
        "source_recognizer_selection_receipt_path": "evidence/selection.json",
        "source_recognizer_selection_receipt_sha256": values["evidence/selection.json"][0],
        "status": "activated",
    }
    pointer_payload = _canonical(pointer)
    pointer_digest, pointer_size = _write(
        root, "approvals/raster-text-active.json", pointer_payload
    )
    values["approvals/raster-text-active.json"] = (
        pointer_digest,
        pointer_size,
        "active-pointer",
    )

    spdx = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "dataLicense": "CC0-1.0",
        "name": "test-runtime",
        "packages": [{"SPDXID": "SPDXRef-Package", "name": "test"}],
        "spdxVersion": "SPDX-2.3",
    }
    spdx_digest, spdx_size = _write(root, "_deployment/runtime.spdx.json", _canonical(spdx))
    values["_deployment/runtime.spdx.json"] = (spdx_digest, spdx_size, "sbom")
    licenses = {
        "components": [
            {
                "license_id": "Apache-2.0",
                "name": "test-model",
                "scope": "model",
                "source_uri": "https://example.invalid/source",
                "version": "1.0",
            },
            {
                "license_id": "Apache-2.0",
                "name": "test-python",
                "scope": "python-package",
                "source_uri": "https://example.invalid/source",
                "version": "1.0",
            },
        ],
        "schema_version": "1.0",
    }
    license_digest, license_size = _write(
        root,
        "_deployment/third-party-licenses.json",
        _canonical(licenses),
    )
    values["_deployment/third-party-licenses.json"] = (
        license_digest,
        license_size,
        "licenses",
    )

    records = [
        {
            "byte_size": size,
            "relative_path": relative,
            "role": role,
            "sha256": digest,
        }
        for relative, (digest, size, role) in sorted(values.items())
    ]
    manifest = {
        "active_pointer": {
            "raster_text_profile_revision": _PROFILE,
            "relative_path": "approvals/raster-text-active.json",
            "sha256": pointer_digest,
        },
        "artifact_kind": "customs-raster-text-runtime-oci",
        "files": records,
        "license_manifest": {
            "relative_path": "_deployment/third-party-licenses.json",
            "sha256": license_digest,
        },
        "platform": {"architecture": "amd64", "os": "linux", "python_abi": "cp311"},
        "runtime_binding_sha256": _BINDING,
        "sbom": {
            "relative_path": "_deployment/runtime.spdx.json",
            "sha256": spdx_digest,
        },
        "schema_version": "1.0",
    }
    _write(root, "_deployment/raster-text-runtime-oci-v1.json", _canonical(manifest))


def _expect_failure(root: Path, expected: str) -> None:
    try:
        verify_runtime_context(root)
    except RuntimeContextError as exc:
        if exc.code != expected:
            raise AssertionError(f"expected {expected}, got {exc.code}") from exc
        return
    raise AssertionError(f"expected {expected}")


def _copy(source: Path, parent: Path, name: str) -> Path:
    destination = parent / name
    shutil.copytree(source, destination)
    return destination


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ocr-oci-contract-") as temporary:
        parent = Path(temporary)
        valid = parent / "valid"
        valid.mkdir()
        _build_valid(valid)
        result = verify_runtime_context(valid)
        if result["raster_text_profile_revision"] != _PROFILE:
            raise AssertionError("valid profile mismatch")

        drift = _copy(valid, parent, "drift")
        (drift / "runtime/models/text-rec/model.bin").write_bytes(b"changed")
        _expect_failure(drift, "runtime_file_digest_mismatch")

        unlisted = _copy(valid, parent, "unlisted")
        _write(unlisted, "evidence/unlisted.json", b"{}")
        _expect_failure(unlisted, "runtime_manifest_file_set_mismatch")

        corpus = _copy(valid, parent, "corpus")
        leak_digest, leak_size = _write(corpus, "evidence/leak.png", b"not-an-image")
        manifest_path = corpus / "_deployment/raster-text-runtime-oci-v1.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"].append(
            {
                "byte_size": leak_size,
                "relative_path": "evidence/leak.png",
                "role": "activation-evidence",
                "sha256": leak_digest,
            }
        )
        manifest["files"].sort(key=lambda item: item["relative_path"])
        manifest_path.write_bytes(_canonical(manifest))
        _expect_failure(corpus, "evaluation_corpus_forbidden")

        incomplete_license = _copy(valid, parent, "incomplete-license")
        license_path = incomplete_license / "_deployment/third-party-licenses.json"
        license_value = json.loads(license_path.read_text(encoding="utf-8"))
        license_value["components"][1]["name"] = "unrelated-python"
        new_license_payload = _canonical(license_value)
        license_path.write_bytes(new_license_payload)
        new_license_sha = hashlib.sha256(new_license_payload).hexdigest()
        incomplete_manifest_path = (
            incomplete_license / "_deployment/raster-text-runtime-oci-v1.json"
        )
        incomplete_manifest = json.loads(incomplete_manifest_path.read_text(encoding="utf-8"))
        incomplete_manifest["license_manifest"]["sha256"] = new_license_sha
        for record in incomplete_manifest["files"]:
            if record["relative_path"] == "_deployment/third-party-licenses.json":
                record["sha256"] = new_license_sha
                record["byte_size"] = len(new_license_payload)
        incomplete_manifest_path.write_bytes(_canonical(incomplete_manifest))
        _expect_failure(incomplete_license, "license_manifest_incomplete")

        if os.name != "nt":
            linked = _copy(valid, parent, "linked")
            model = linked / "runtime/models/text-rec/model.bin"
            model.unlink()
            model.symlink_to(linked / "evidence/sealed.json")
            _expect_failure(linked, "runtime_symlink_forbidden")

    print("ocr_oci_runtime_contract_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
