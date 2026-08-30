"""Positive and negative contract tests for the OCI runtime verifiers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from verify_raster_text_oci_runtime import (  # noqa: E402
    RuntimeContextError,
    verify_runtime_context,
)
from customs_ai_agent.service.raster_text_preview_runtime import (  # noqa: E402
    parse_raster_text_activation_pointer,
)
from raster_text_oci_evidence import (  # noqa: E402
    RasterTextOciEvidenceError,
    verify_v9_direct_evidence_closure,
)
from raster_text_oci_metadata import build_runtime_metadata  # noqa: E402

_BINDING = "a" * 64
_PROFILE = f"synthetic-raster-text-v1:{'b' * 24}"
_ENGINE = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_METRIC = "source-attested-semantic-text-v3"
_EVIDENCE = "source-attested-public-finetuned-staged-train-holdout-v2"
_QUALITY = "preview-draft-critical97-cer2-v1"
_GENERATION = "current-runner-native-full-gate-v1"
_MODEL_FILENAMES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
_CPYTHON_LICENSE_TEXT = b"Fixture CPython license.\nhttps://www.python.org/psf/\n"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_pointer(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _model_snapshot(values: dict[str, tuple[str, int, str]], prefix: str) -> str:
    rows = [
        {
            "byte_size": values[f"runtime/models/{prefix}/{name}"][1],
            "relative_path": (
                f"models/{prefix}/{name}" if prefix == "text-det" else f"{prefix}/{name}"
            ),
            "sha256": values[f"runtime/models/{prefix}/{name}"][0],
        }
        for name in _MODEL_FILENAMES
    ]
    return hashlib.sha256(_canonical({"artifacts": rows})).hexdigest()


def _write(root: Path, relative: str, payload: bytes) -> tuple[str, int]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    current = root
    current.chmod(0o755)
    for part in path.parent.relative_to(root).parts:
        current = current / part
        current.chmod(0o755)
    path.write_bytes(payload)
    path.chmod(0o644)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _build_valid(
    root: Path,
    *,
    precompiled_relative: str = ("interpreter/lib/python3.11/__pycache__/pathlib.cpython-311.pyc"),
) -> None:
    parent_site = root.parent / "parent-site"
    parent_dist = parent_site / "test_parent_python-2.0.dist-info"
    parent_dist.mkdir(parents=True, exist_ok=True)
    (parent_dist / "METADATA").write_text(
        "Metadata-Version: 2.4\n"
        "Name: test-parent-python\n"
        "Version: 2.0\n"
        "License-Expression: Apache-2.0\n"
        "Project-URL: Source, https://github.com/example/test-parent-python\n",
        encoding="utf-8",
    )
    (parent_dist / "RECORD").write_text(
        "test_parent_python-2.0.dist-info/METADATA,,\ntest_parent_python-2.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    parent_site_text = str(parent_site)
    if parent_site_text not in sys.path:
        sys.path.insert(0, parent_site_text)
    values: dict[str, tuple[str, int, str]] = {}
    for relative, payload, role in (
        (
            "evidence/selection.json",
            b'{"artifact_kind":"test-receipt"}',
            "activation-evidence",
        ),
        (
            "evidence/projection.json",
            b'{"artifact_kind":"test-receipt"}',
            "activation-evidence",
        ),
        (
            "evidence/stage1-decision.json",
            b'{"artifact_kind":"test-receipt"}',
            "activation-evidence",
        ),
        (
            "evidence/source-gold.json",
            b'{"artifact_kind":"test-receipt"}',
            "activation-evidence",
        ),
        (
            "evidence/calibration.json",
            b'{"artifact_kind":"test-receipt"}',
            "activation-evidence",
        ),
        (
            "evidence/candidate-execution.json",
            b'{"artifact_kind":"test-receipt"}',
            "activation-evidence",
        ),
        (
            "evidence/warm-smoke.json",
            b'{"artifact_kind":"test-receipt"}',
            "activation-evidence",
        ),
        (
            "runtime/sidecar/inventories/packages.json",
            _canonical({"packages": [{"name": "test-python", "version": "1.0"}]}),
            "inventory",
        ),
        (
            "runtime/sidecar/inventories/parent-packages.json",
            _canonical({"packages": [{"name": "test-parent-python", "version": "2.0"}]}),
            "inventory",
        ),
        (
            "runtime/sidecar/interpreter/bin/python3.11",
            b"\x7fELF\x02\x01\x01" + b"\x00" * 9 + b"\x03\x00\x3e\x00",
            "sidecar-runtime",
        ),
        (
            "runtime/sidecar/interpreter/include/python3.11/patchlevel.h",
            b"#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 11\n"
            b"#define PY_MICRO_VERSION 15\n",
            "sidecar-runtime",
        ),
        (
            "runtime/sidecar/interpreter/LICENSE.txt",
            _CPYTHON_LICENSE_TEXT,
            "sidecar-runtime",
        ),
        (
            f"runtime/sidecar/{precompiled_relative}",
            b"fixture-manifest-bound-bytecode",
            "sidecar-runtime",
        ),
        (
            "runtime/sidecar/module-root/test_python-1.0.dist-info/METADATA",
            b"Metadata-Version: 2.4\nName: test-python\nVersion: 1.0\n"
            b"License-Expression: MIT\n"
            b"Project-URL: Source, https://github.com/example/test-python\n",
            "sidecar-runtime",
        ),
        (
            "runtime/sidecar/module-root/test_python-1.0.dist-info/RECORD",
            b"test_python-1.0.dist-info/METADATA,,\ntest_python-1.0.dist-info/RECORD,,\n",
            "sidecar-runtime",
        ),
    ):
        digest, size = _write(root, relative, payload)
        values[relative] = (digest, size, role)
    for prefix in ("text-det", "text-rec"):
        for index, name in enumerate(_MODEL_FILENAMES):
            relative = f"runtime/models/{prefix}/{name}"
            payload = f"{prefix}-{index}".encode("ascii")
            digest, size = _write(root, relative, payload)
            values[relative] = (digest, size, "model")
    (root / "runtime/sidecar/interpreter/bin/python3.11").chmod(0o555)

    sidecar_manifest = {
        "bundle_id": "raster-text-direct-paddle-bundle-v1",
        "files": [
            {
                "byte_size": values[f"runtime/sidecar/{relative}"][1],
                "path": relative,
                "sha256": values[f"runtime/sidecar/{relative}"][0],
            }
            for relative in sorted(
                path.removeprefix("runtime/sidecar/")
                for path in values
                if path.startswith("runtime/sidecar/")
            )
        ],
        "runtime_architecture": "amd64",
        "runtime_operating_system": "linux",
        "runtime_python_abi": "cp311",
        "schema_version": "2.0",
        "sidecar_interpreter_relative_path": "interpreter/bin/python3.11",
    }
    sidecar_manifest_digest, sidecar_manifest_size = _write(
        root,
        "runtime/sidecar/manifest.json",
        _canonical(sidecar_manifest),
    )
    values["runtime/sidecar/manifest.json"] = (
        sidecar_manifest_digest,
        sidecar_manifest_size,
        "inventory",
    )

    detector_paths = [f"text-det/{name}" for name in _MODEL_FILENAMES]
    recognizer_paths = [f"text-rec/{name}" for name in _MODEL_FILENAMES]
    model_inventory = {
        "artifacts": [
            {
                "byte_size": values[f"runtime/models/{relative}"][1],
                "relative_path": relative,
                "sha256": values[f"runtime/models/{relative}"][0],
            }
            for relative in [*detector_paths, *recognizer_paths]
        ],
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": ("source_attested_en_PP-OCRv5_mobile_rec_head_v2"),
    }
    model_inventory_digest, model_inventory_size = _write(
        root,
        "runtime/models/inventories/models.json",
        _canonical(model_inventory),
    )
    values["runtime/models/inventories/models.json"] = (
        model_inventory_digest,
        model_inventory_size,
        "inventory",
    )
    upstream_recognizer_hashes = {
        f"models/text-rec/{name}": f"{index + 40:064x}"
        for index, name in enumerate(_MODEL_FILENAMES)
    }
    upstream_source_uri = (
        "https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec/"
        "tree/267c36e24c331595590fe7bd72bde2436fd286f2"
    )
    source_inventory = {
        "artifacts": [
            *[
                {
                    "license_id": "Apache-2.0",
                    "relative_path": f"models/paddleocr3/det/model-{index}.bin",
                    "sha256": values[f"runtime/models/{relative}"][0],
                    "source_uri": "https://github.com/example/test-detector",
                }
                for index, relative in enumerate(detector_paths)
            ],
            *[
                {
                    "license_id": "Apache-2.0",
                    "relative_path": f"models/paddleocr3/rec/{name}",
                    "sha256": upstream_recognizer_hashes[f"models/text-rec/{name}"],
                    "source_uri": upstream_source_uri,
                }
                for name in _MODEL_FILENAMES
            ],
        ]
    }
    source_inventory_digest, source_inventory_size = _write(
        root,
        "model-inventories/source.json",
        _canonical(source_inventory),
    )
    values["model-inventories/source.json"] = (
        source_inventory_digest,
        source_inventory_size,
        "component-evidence",
    )
    mobile_manifest = {
        "source_model_inventory_relative_path": "model-inventories/source.json",
        "source_model_inventory_sha256": source_inventory_digest,
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
    }
    mobile_manifest_digest, mobile_manifest_size = _write(
        root,
        "runtime/models/provenance/mobile-manifest.json",
        _canonical(mobile_manifest),
    )
    values["runtime/models/provenance/mobile-manifest.json"] = (
        mobile_manifest_digest,
        mobile_manifest_size,
        "inventory",
    )
    training_receipt = {
        "initialization": "pinned_public_inference_graph_warm_start",
        "public_mobile_source_inventory_sha256": source_inventory_digest,
        "public_recognizer_artifact_sha256s": upstream_recognizer_hashes,
        "receipt_sha256": "",
        "tuning_scope": "head",
    }
    training_receipt["receipt_sha256"] = hashlib.sha256(_canonical(training_receipt)).hexdigest()
    training_receipt_digest, training_receipt_size = _write(
        root,
        "runtime/models/provenance/training-receipt.json",
        _canonical(training_receipt),
    )
    values["runtime/models/provenance/training-receipt.json"] = (
        training_receipt_digest,
        training_receipt_size,
        "inventory",
    )

    model_manifest = {
        "bundle_id": "test-model-bundle",
        "files": [
            {
                "byte_size": values[f"runtime/models/{relative}"][1],
                "path": relative,
                "sha256": values[f"runtime/models/{relative}"][0],
            }
            for relative in sorted(
                path.removeprefix("runtime/models/")
                for path in values
                if path.startswith("runtime/models/")
            )
        ],
        "mobile_manifest_provenance_relative_path": "provenance/mobile-manifest.json",
        "model_inventory_relative_path": "inventories/models.json",
        "recognizer_receipt_provenance_relative_path": "provenance/training-receipt.json",
        "schema_version": "1.0",
        "source_detector_model_snapshot_sha256": _model_snapshot(values, "text-det"),
        "source_recognizer_receipt_sha256": training_receipt["receipt_sha256"],
        "source_recognizer_model_snapshot_sha256": _model_snapshot(values, "text-rec"),
    }
    model_manifest_digest, model_manifest_size = _write(
        root,
        "runtime/models/manifest.json",
        _canonical(model_manifest),
    )
    values["runtime/models/manifest.json"] = (
        model_manifest_digest,
        model_manifest_size,
        "inventory",
    )

    profile = {
        "bundle_manifest_sha256": sidecar_manifest_digest,
        "bundle_root_relative_path": "runtime/sidecar",
        "engine_kind": _ENGINE,
        "mobile_model_bundle_manifest_sha256": model_manifest_digest,
        "mobile_model_bundle_root_relative_path": "runtime/models",
        "model_inventory_path": "runtime/models/inventories/models.json",
        "model_inventory_sha256": model_inventory_digest,
        "parent_renderer_package_inventory_path": (
            "runtime/sidecar/inventories/parent-packages.json"
        ),
        "parent_renderer_package_inventory_sha256": values[
            "runtime/sidecar/inventories/parent-packages.json"
        ][0],
        "raster_text_profile_revision": _PROFILE,
        "schema_version": "19.0",
        "sidecar_interpreter_root": "runtime/sidecar/interpreter",
        "sidecar_interpreter_root_tree_sha256": "c" * 64,
        "sidecar_interpreter_path": "runtime/sidecar/interpreter/bin/python3.11",
        "sidecar_interpreter_sha256": values["runtime/sidecar/interpreter/bin/python3.11"][0],
        "sidecar_module_root": "runtime/sidecar/module-root",
        "sidecar_package_inventory_path": "runtime/sidecar/inventories/packages.json",
        "sidecar_package_inventory_sha256": values["runtime/sidecar/inventories/packages.json"][0],
        "source_recognizer_parent_selection_receipt_sha256": "6" * 64,
        "source_recognizer_receipt_sha256": "3" * 64,
        "source_recognizer_release_sha256": "2" * 64,
        "source_recognizer_selection_receipt_relative_path": "evidence/selection.json",
        "source_recognizer_selection_receipt_sha256": values["evidence/selection.json"][0],
        "source_recognizer_training_intent_sha256": "4" * 64,
    }
    profile_digest, profile_size = _write(root, "profiles/profile.json", _canonical(profile))
    values["profiles/profile.json"] = (profile_digest, profile_size, "runtime-profile")

    scores = [
        {
            "character_error_rate": 0.02,
            "critical_value_exact_match": 0.97,
            "document_type": document_type,
            "profile_id": profile_id,
        }
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]
    sealed = {
        "artifact_kind": "synthetic_raster_semantic_text_fresh_staged_evaluation",
        "asset_count": 2_400,
        "calibration_receipt_path": "evidence/calibration.json",
        "calibration_receipt_sha256": values["evidence/calibration.json"][0],
        "candidate_execution_path": "evidence/candidate-execution.json",
        "candidate_execution_sha256": values["evidence/candidate-execution.json"][0],
        "current_runner_implementation_sha256": _BINDING,
        "dataset_id": "synthetic-raster-text-v1",
        "evidence_generation_revision": _GENERATION,
        "evidence_revision": _EVIDENCE,
        "extractor_revision": "v19-linux-extractor",
        "family_count": 100,
        "hard_timeout_seconds": 30,
        "latency_gate_revision": "warm-p95-10s-v1",
        "latency_p95_seconds": 7.8,
        "metric_revision": _METRIC,
        "minimum_page_confidence": 0.0,
        "projection_kind": "sealed-stage2-100-family-v1",
        "projection_receipt_path": "evidence/projection.json",
        "projection_receipt_sha256": values["evidence/projection.json"][0],
        "quality_gate_revision": _QUALITY,
        "raster_text_profile_revision": _PROFILE,
        "runtime_configuration_sha256": profile_digest,
        "schema_version": "9.0",
        "scores": scores,
        "selection_receipt_path": "evidence/selection.json",
        "selection_receipt_sha256": values["evidence/selection.json"][0],
        "service_warm_latency_p95_seconds": 7.9,
        "source_gold_receipt_path": "evidence/source-gold.json",
        "source_gold_receipt_sha256": values["evidence/source-gold.json"][0],
        "source_recognizer_parent_selection_receipt_sha256": "6" * 64,
        "source_recognizer_receipt_sha256": "3" * 64,
        "source_recognizer_release_sha256": "2" * 64,
        "source_recognizer_training_intent_sha256": "4" * 64,
        "source_truth_kind": "pdf_content_stream_text_v1",
        "split": "fresh-sealed",
        "stage1_decision": "continue",
        "stage1_decision_receipt_path": "evidence/stage1-decision.json",
        "stage1_decision_receipt_sha256": values["evidence/stage1-decision.json"][0],
        "warm_service_smoke_receipt_path": "evidence/warm-smoke.json",
        "warm_service_smoke_receipt_sha256": values["evidence/warm-smoke.json"][0],
    }
    sealed_digest, sealed_size = _write(root, "evidence/sealed.json", _canonical(sealed))
    values["evidence/sealed.json"] = (sealed_digest, sealed_size, "activation-evidence")

    pointer = {
        "activation_kind": "source_attested_synthetic_raster_text",
        "current_runner_implementation_sha256": _BINDING,
        "dataset_id": "synthetic-raster-text-v1",
        "engine_kind": _ENGINE,
        "evidence_generation_revision": _GENERATION,
        "evidence_revision": _EVIDENCE,
        "latency_gate_revision": "warm-p95-10s-v1",
        "metric_revision": _METRIC,
        "quality_gate_revision": _QUALITY,
        "raster_text_profile_revision": _PROFILE,
        "runtime_configuration_path": "profiles/profile.json",
        "runtime_configuration_sha256": profile_digest,
        "runtime_validation_revision": "activation-full-runtime-compact-v1",
        "schema_version": "12.0",
        "sealed_evaluation_path": "evidence/sealed.json",
        "sealed_evaluation_sha256": sealed_digest,
        "source_recognizer_parent_selection_receipt_sha256": "6" * 64,
        "source_recognizer_receipt_sha256": "3" * 64,
        "source_recognizer_release_sha256": "2" * 64,
        "source_recognizer_selection_receipt_path": "evidence/selection.json",
        "source_recognizer_selection_receipt_sha256": values["evidence/selection.json"][0],
        "source_recognizer_training_intent_sha256": "4" * 64,
        "status": "activated",
    }
    pointer_payload = _canonical_pointer(pointer)
    pointer_digest, pointer_size = _write(
        root, "approvals/raster-text-active.json", pointer_payload
    )
    values["approvals/raster-text-active.json"] = (
        pointer_digest,
        pointer_size,
        "active-pointer",
    )

    policy = {
        "cpython": [
            {
                "license_sha256": hashlib.sha256(_CPYTHON_LICENSE_TEXT).hexdigest(),
                "source_uri": "https://github.com/python/cpython/tree/v3.11.15",
                "version": "3.11.15",
            }
        ],
        "distribution_scope": "private-ghcr-only",
        "fine_tuned_models": [
            {
                "combined_model_manifest_sha256": model_manifest_digest,
                "distribution_scope": "private-ghcr-only",
                "license_id": "NOASSERTION",
                "model_name": "source_attested_en_PP-OCRv5_mobile_rec_head_v2",
                "model_snapshot_sha256": _model_snapshot(values, "text-rec"),
                "source_uri": "NOASSERTION",
                "upstream_base_artifacts_sha256": hashlib.sha256(
                    _canonical(upstream_recognizer_hashes)
                ).hexdigest(),
                "upstream_base_license_id": "Apache-2.0",
                "upstream_base_model_name": "en_PP-OCRv5_mobile_rec",
                "upstream_base_source_inventory_sha256": source_inventory_digest,
                "upstream_base_source_uri": upstream_source_uri,
                "upstream_training_receipt_sha256": training_receipt["receipt_sha256"],
            }
        ],
        "schema_version": "1.0",
    }
    policy_payload = _canonical(policy) + b"\n"
    policy_digest, policy_size = _write(
        root,
        "_deployment/runtime-license-policy.json",
        policy_payload,
    )
    values["_deployment/runtime-license-policy.json"] = (
        policy_digest,
        policy_size,
        "component-evidence",
    )
    generated = build_runtime_metadata(
        root,
        profile,
        policy_path=root / "_deployment/runtime-license-policy.json",
    )
    for relative, payload in generated.generated_evidence:
        digest, size = _write(root, relative.as_posix(), payload)
        values[relative.as_posix()] = (digest, size, "component-evidence")
    spdx_digest, spdx_size = _write(
        root,
        "_deployment/runtime.spdx.json",
        generated.sbom,
    )
    values["_deployment/runtime.spdx.json"] = (spdx_digest, spdx_size, "sbom")
    license_digest, license_size = _write(
        root,
        "_deployment/third-party-licenses.json",
        generated.license_manifest,
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


def _expect_failure(
    root: Path,
    expected: str,
    *,
    expected_license_policy: Path | None = None,
) -> None:
    try:
        verify_runtime_context(
            root,
            expected_license_policy=expected_license_policy,
        )
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
        expected_policy = parent / "expected-policy.json"
        expected_policy.write_bytes(
            (valid / "_deployment/runtime-license-policy.json").read_bytes()
        )
        result = verify_runtime_context(
            valid,
            expected_license_policy=expected_policy,
        )
        if result["raster_text_profile_revision"] != _PROFILE:
            raise AssertionError("valid profile mismatch")
        parent_evidence_relative = "_deployment/parent-package-evidence.json"
        manifest_value = json.loads(
            (valid / "_deployment/raster-text-runtime-oci-v1.json").read_bytes()
        )
        parent_records = [
            record
            for record in manifest_value["files"]
            if record["relative_path"] == parent_evidence_relative
        ]
        if (
            len(parent_records) != 1
            or parent_records[0]["role"] != "component-evidence"
            or not (valid / parent_evidence_relative).is_file()
        ):
            raise AssertionError("parent package evidence integration missing")
        pointer_raw = (valid / "approvals/raster-text-active.json").read_bytes()
        pointer = parse_raster_text_activation_pointer(pointer_raw)
        direct_evidence = verify_v9_direct_evidence_closure(root=valid, pointer=pointer)
        if direct_evidence != {
            PurePosixPath(relative)
            for relative in (
                "evidence/calibration.json",
                "evidence/candidate-execution.json",
                "evidence/projection.json",
                "evidence/sealed.json",
                "evidence/selection.json",
                "evidence/source-gold.json",
                "evidence/stage1-decision.json",
                "evidence/warm-smoke.json",
            )
        }:
            raise AssertionError("direct V9 evidence closure mismatch")

        wrong_policy = parent / "wrong-policy.json"
        wrong_policy.write_bytes(b'{"cpython":[],"fine_tuned_models":[],"schema_version":"1.0"}\n')
        _expect_failure(
            valid,
            "runtime_license_policy_mismatch",
            expected_license_policy=wrong_policy,
        )

        missing_parent_evidence = _copy(valid, parent, "missing-parent-evidence")
        (missing_parent_evidence / parent_evidence_relative).unlink()
        _expect_failure(
            missing_parent_evidence,
            "runtime_manifest_file_set_mismatch",
        )

        drifted_parent_evidence = _copy(valid, parent, "drifted-parent-evidence")
        drifted_parent_path = drifted_parent_evidence / parent_evidence_relative
        drifted_parent_value = json.loads(drifted_parent_path.read_bytes())
        drifted_parent_value["distributions"][0]["project_urls"] = [
            "Source, https://github.com/example/forged-parent"
        ]
        drifted_parent_raw = _canonical(drifted_parent_value)
        drifted_parent_path.write_bytes(drifted_parent_raw)
        drifted_manifest_path = (
            drifted_parent_evidence / "_deployment/raster-text-runtime-oci-v1.json"
        )
        drifted_manifest = json.loads(drifted_manifest_path.read_bytes())
        for record in drifted_manifest["files"]:
            if record["relative_path"] == parent_evidence_relative:
                record["sha256"] = hashlib.sha256(drifted_parent_raw).hexdigest()
                record["byte_size"] = len(drifted_parent_raw)
        drifted_manifest_path.write_bytes(_canonical(drifted_manifest))
        _expect_failure(
            drifted_parent_evidence,
            "runtime_parent_package_evidence_mismatch",
        )

        missing_policy = _copy(valid, parent, "missing-policy")
        (missing_policy / "_deployment/runtime-license-policy.json").unlink()
        missing_policy_manifest_path = (
            missing_policy / "_deployment/raster-text-runtime-oci-v1.json"
        )
        missing_policy_manifest = json.loads(
            missing_policy_manifest_path.read_text(encoding="utf-8")
        )
        missing_policy_manifest["files"] = [
            record
            for record in missing_policy_manifest["files"]
            if record["relative_path"] != "_deployment/runtime-license-policy.json"
        ]
        missing_policy_manifest_path.write_bytes(_canonical(missing_policy_manifest))
        _expect_failure(missing_policy, "runtime_license_policy_invalid")

        direct_evidence_drift = _copy(valid, parent, "direct-evidence-drift")
        (direct_evidence_drift / "evidence/stage1-decision.json").write_bytes(b"changed")
        try:
            verify_v9_direct_evidence_closure(
                root=direct_evidence_drift,
                pointer=parse_raster_text_activation_pointer(
                    (direct_evidence_drift / "approvals/raster-text-active.json").read_bytes()
                ),
            )
        except RasterTextOciEvidenceError:
            pass
        else:
            raise AssertionError("direct V9 evidence drift accepted")

        legacy_pointer = _copy(valid, parent, "legacy-pointer")
        legacy_pointer_path = legacy_pointer / "approvals/raster-text-active.json"
        legacy_pointer_value = json.loads(legacy_pointer_path.read_text(encoding="utf-8"))
        legacy_pointer_value["schema_version"] = "11.0"
        legacy_pointer_payload = _canonical_pointer(legacy_pointer_value)
        legacy_pointer_path.write_bytes(legacy_pointer_payload)
        legacy_pointer_sha = hashlib.sha256(legacy_pointer_payload).hexdigest()
        legacy_manifest_path = legacy_pointer / "_deployment/raster-text-runtime-oci-v1.json"
        legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        legacy_manifest["active_pointer"]["sha256"] = legacy_pointer_sha
        for record in legacy_manifest["files"]:
            if record["relative_path"] == "approvals/raster-text-active.json":
                record["sha256"] = legacy_pointer_sha
                record["byte_size"] = len(legacy_pointer_payload)
        legacy_manifest_path.write_bytes(_canonical(legacy_manifest))
        _expect_failure(legacy_pointer, "active_pointer_invalid")

        legacy_sealed = _copy(valid, parent, "legacy-sealed")
        legacy_sealed_path = legacy_sealed / "evidence/sealed.json"
        legacy_sealed_value = json.loads(legacy_sealed_path.read_text(encoding="utf-8"))
        legacy_sealed_value["schema_version"] = "8.0"
        legacy_sealed_payload = _canonical(legacy_sealed_value)
        legacy_sealed_path.write_bytes(legacy_sealed_payload)
        legacy_sealed_sha = hashlib.sha256(legacy_sealed_payload).hexdigest()
        changed_pointer_path = legacy_sealed / "approvals/raster-text-active.json"
        changed_pointer = json.loads(changed_pointer_path.read_text(encoding="utf-8"))
        changed_pointer["sealed_evaluation_sha256"] = legacy_sealed_sha
        changed_pointer_payload = _canonical_pointer(changed_pointer)
        changed_pointer_path.write_bytes(changed_pointer_payload)
        changed_pointer_sha = hashlib.sha256(changed_pointer_payload).hexdigest()
        changed_manifest_path = legacy_sealed / "_deployment/raster-text-runtime-oci-v1.json"
        changed_manifest = json.loads(changed_manifest_path.read_text(encoding="utf-8"))
        changed_manifest["active_pointer"]["sha256"] = changed_pointer_sha
        for record in changed_manifest["files"]:
            if record["relative_path"] == "evidence/sealed.json":
                record["sha256"] = legacy_sealed_sha
                record["byte_size"] = len(legacy_sealed_payload)
            elif record["relative_path"] == "approvals/raster-text-active.json":
                record["sha256"] = changed_pointer_sha
                record["byte_size"] = len(changed_pointer_payload)
        changed_manifest_path.write_bytes(_canonical(changed_manifest))
        _expect_failure(legacy_sealed, "sealed_evaluation_invalid")

        drift = _copy(valid, parent, "drift")
        (drift / "runtime/models/text-rec/inference.pdiparams").write_bytes(b"changed")
        _expect_failure(drift, "runtime_file_digest_mismatch")

        unlisted = _copy(valid, parent, "unlisted")
        _write(unlisted, "evidence/unlisted.json", b"{}")
        _expect_failure(unlisted, "runtime_manifest_file_set_mismatch")

        extra_component_evidence = _copy(valid, parent, "extra-component-evidence")
        extra_digest, extra_size = _write(
            extra_component_evidence,
            "model-inventories/extra.json",
            b"{}",
        )
        extra_manifest_path = (
            extra_component_evidence / "_deployment/raster-text-runtime-oci-v1.json"
        )
        extra_manifest = json.loads(extra_manifest_path.read_text(encoding="utf-8"))
        extra_manifest["files"].append(
            {
                "byte_size": extra_size,
                "relative_path": "model-inventories/extra.json",
                "role": "component-evidence",
                "sha256": extra_digest,
            }
        )
        extra_manifest["files"].sort(key=lambda item: item["relative_path"])
        extra_manifest_path.write_bytes(_canonical(extra_manifest))
        _expect_failure(
            extra_component_evidence,
            "runtime_component_evidence_set_mismatch",
        )

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

        unbound_cache = _copy(valid, parent, "unbound-cache")
        cache_digest, cache_size = _write(
            unbound_cache,
            "runtime/models/text-det/__pycache__/extra.cpython-311.pyc",
            b"unbound-bytecode",
        )
        cache_manifest_path = unbound_cache / "_deployment/raster-text-runtime-oci-v1.json"
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_manifest["files"].append(
            {
                "byte_size": cache_size,
                "relative_path": ("runtime/models/text-det/__pycache__/extra.cpython-311.pyc"),
                "role": "model",
                "sha256": cache_digest,
            }
        )
        cache_manifest["files"].sort(key=lambda item: item["relative_path"])
        cache_manifest_path.write_bytes(_canonical(cache_manifest))
        _expect_failure(unbound_cache, "runtime_cache_or_secret_forbidden")

        declared_non_bytecode = parent / "declared-non-bytecode-cache"
        declared_non_bytecode.mkdir()
        _build_valid(
            declared_non_bytecode,
            precompiled_relative="interpreter/lib/python3.11/__pycache__/payload.txt",
        )
        _expect_failure(declared_non_bytecode, "runtime_cache_or_secret_forbidden")

        empty_receipt = _copy(valid, parent, "empty-receipt")
        empty_digest, empty_size = _write(empty_receipt, "evidence/extra-receipt.json", b"{}")
        empty_manifest_path = empty_receipt / "_deployment/raster-text-runtime-oci-v1.json"
        empty_manifest = json.loads(empty_manifest_path.read_text(encoding="utf-8"))
        empty_manifest["files"].append(
            {
                "byte_size": empty_size,
                "relative_path": "evidence/extra-receipt.json",
                "role": "activation-evidence",
                "sha256": empty_digest,
            }
        )
        empty_manifest["files"].sort(key=lambda item: item["relative_path"])
        empty_manifest_path.write_bytes(_canonical(empty_manifest))
        _expect_failure(empty_receipt, "activation_evidence_invalid")

        disguised_sidecar_resource = _copy(valid, parent, "disguised-sidecar-resource")
        resource_digest, resource_size = _write(
            disguised_sidecar_resource,
            "runtime/sidecar/tcl/demos/leak.png",
            b"not-a-package-resource",
        )
        resource_manifest_path = (
            disguised_sidecar_resource / "_deployment/raster-text-runtime-oci-v1.json"
        )
        resource_manifest = json.loads(resource_manifest_path.read_text(encoding="utf-8"))
        resource_manifest["files"].append(
            {
                "byte_size": resource_size,
                "relative_path": "runtime/sidecar/tcl/demos/leak.png",
                "role": "sidecar-runtime",
                "sha256": resource_digest,
            }
        )
        resource_manifest["files"].sort(key=lambda item: item["relative_path"])
        resource_manifest_path.write_bytes(_canonical(resource_manifest))
        _expect_failure(disguised_sidecar_resource, "runtime_platform_manifest_invalid")

        malformed_role = _copy(valid, parent, "malformed-role")
        malformed_manifest_path = malformed_role / "_deployment/raster-text-runtime-oci-v1.json"
        malformed_manifest = json.loads(malformed_manifest_path.read_text(encoding="utf-8"))
        malformed_manifest["files"][0]["role"] = []
        malformed_manifest_path.write_bytes(_canonical(malformed_manifest))
        _expect_failure(malformed_role, "runtime_context_invalid")

        disguised_corpus = _copy(valid, parent, "disguised-corpus")
        leak_digest, leak_size = _write(
            disguised_corpus,
            "misc/synthetic-raster-v1/leak.pdf",
            b"not-a-document",
        )
        disguised_manifest_path = disguised_corpus / "_deployment/raster-text-runtime-oci-v1.json"
        disguised_manifest = json.loads(disguised_manifest_path.read_text(encoding="utf-8"))
        disguised_manifest["files"].append(
            {
                "byte_size": leak_size,
                "relative_path": "misc/synthetic-raster-v1/leak.pdf",
                "role": "model",
                "sha256": leak_digest,
            }
        )
        disguised_manifest["files"].sort(key=lambda item: item["relative_path"])
        disguised_manifest_path.write_bytes(_canonical(disguised_manifest))
        _expect_failure(disguised_corpus, "evaluation_corpus_forbidden")

        misbound_role = _copy(valid, parent, "misbound-role")
        extra_digest, extra_size = _write(misbound_role, "misc/extra.bin", b"extra")
        misbound_manifest_path = misbound_role / "_deployment/raster-text-runtime-oci-v1.json"
        misbound_manifest = json.loads(misbound_manifest_path.read_text(encoding="utf-8"))
        misbound_manifest["files"].append(
            {
                "byte_size": extra_size,
                "relative_path": "misc/extra.bin",
                "role": "model",
                "sha256": extra_digest,
            }
        )
        misbound_manifest["files"].sort(key=lambda item: item["relative_path"])
        misbound_manifest_path.write_bytes(_canonical(misbound_manifest))
        _expect_failure(misbound_role, "runtime_role_path_invalid")

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
        _expect_failure(
            incomplete_license,
            "runtime_license_component_set_mismatch",
        )

        if os.name != "nt":
            writable_root = _copy(valid, parent, "writable-root")
            writable_root.chmod(0o777)
            _expect_failure(writable_root, "runtime_directory_permissions_invalid")

            writable_manifest = _copy(valid, parent, "writable-manifest")
            (writable_manifest / "_deployment/raster-text-runtime-oci-v1.json").chmod(0o666)
            _expect_failure(writable_manifest, "runtime_file_permissions_invalid")

            writable_directory = _copy(valid, parent, "writable-directory")
            (writable_directory / "approvals").chmod(0o777)
            _expect_failure(
                writable_directory,
                "runtime_directory_permissions_invalid",
            )

            unreadable = _copy(valid, parent, "unreadable")
            (unreadable / "runtime/models/text-rec/inference.pdiparams").chmod(0o600)
            _expect_failure(unreadable, "runtime_file_permissions_invalid")

            unsearchable = _copy(valid, parent, "unsearchable")
            (unsearchable / "runtime/models").chmod(0o700)
            _expect_failure(unsearchable, "runtime_directory_permissions_invalid")

            linked = _copy(valid, parent, "linked")
            model = linked / "runtime/models/text-rec/inference.pdiparams"
            model.unlink()
            model.symlink_to(linked / "evidence/sealed.json")
            _expect_failure(linked, "runtime_symlink_forbidden")

    print("ocr_oci_runtime_contract_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
