"""Score one provenance-bound synthetic-raster candidate split locally.

This command accepts no OCR result JSON, score, model hash, package hash, or
profile label from the operator. It re-reads the content-addressed candidate
bundle/execution receipt, revalidates the current local PP-StructureV3
configuration through the shared evaluation resolver, and rebuilds every
candidate-vs-Gold case from the immutable synthetic-raster receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Sequence

from customs_ai_agent.evaluation.synthetic_raster import (
    RasterAssetReceiptV1,
    RasterEvaluationCaseV1,
    RasterGoldArtifactV1,
    RenderedRasterBytesV1,
    SyntheticRasterCompilerUnavailable,
    SyntheticRasterReceiptV1,
    derive_synthetic_raster_confidence_thresholds,
    evaluate_synthetic_raster_gates,
    synthetic_raster_case_manifest_sha256,
    synthetic_raster_gate_passes,
    verify_rendered_raster_output,
)
from customs_ai_agent.evaluation.synthetic_raster_candidates import (
    SyntheticRasterCandidateError,
    read_canonical_raster_candidate_bundle,
    read_canonical_raster_candidate_execution,
    validate_raster_candidate_bundle_against_receipt,
    validate_raster_candidate_execution,
)
from customs_ai_agent.service.raster_document_preview_runtime import (
    PpStructureV3OfflineProfileV2,
    RasterCandidateEngineResolutionV1,
    resolve_ppstructurev3_raster_candidate,
)


_MAX_CASE_BYTES = 32 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024
_MAX_RASTER_ASSET_BYTES = 32 * 1024 * 1024
_CandidateSplit = Literal["calibration", "sealed"]
_SYNTHETIC_RASTER_PROFILE_REVISION = r"^(?:synthetic-raster-v1|synthetic-raster-v1:[a-f0-9]{24})$"


def evaluation_implementation_sha256() -> str:
    """Bind an activation evidence path to this evaluator's exact bytes.

    The digest is used only for local content-addressed receipt naming.  It
    never accepts a caller-provided score, model value, OCR JSON, or Gold.
    """

    try:
        source = Path(__file__).read_bytes()
    except OSError as error:
        raise ValueError("local evaluator source is unavailable") from error
    return hashlib.sha256(source).hexdigest()


def _preview_root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise ValueError("document preview root is invalid")
        root = path.resolve(strict=True)
        if (
            not root.is_dir()
            or root.is_symlink()
            or tuple(part.casefold() for part in root.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise ValueError("document preview root is invalid")
        return root
    except OSError as error:
        raise ValueError("document preview root is invalid") from error


def _relative_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts or ":" in value or "\\" in value:
        raise ValueError("local artifact path is invalid")
    destination = root
    for part in relative.parts:
        destination = destination / part
        if destination.is_symlink():
            raise ValueError("local artifact path is invalid")
    resolved = destination.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("local artifact path is invalid") from error
    return destination


def _read_raw(path: Path, *, limit: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("local evaluation input is invalid")
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError("local evaluation input is invalid") from error
    if not raw or len(raw) > limit:
        raise ValueError("local evaluation input is invalid")
    return raw


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _load_synthetic_receipt(root: Path, *, relative_path: str) -> tuple[SyntheticRasterReceiptV1, str]:
    raw = _read_raw(_relative_path(root, relative_path), limit=32 * 1024 * 1024)
    try:
        receipt = SyntheticRasterReceiptV1.model_validate_json(raw)
    except ValueError as error:
        raise ValueError("synthetic raster receipt is invalid") from error
    if raw != _canonical_json(receipt.model_dump(mode="json", warnings=False)):
        raise ValueError("synthetic raster receipt is not canonical")
    expected_families = {"train": 750, "development": 100, "calibration": 50, "sealed": 100}
    families_by_split = {
        split: {asset.family_id for asset in receipt.assets if asset.split == split}
        for split in expected_families
    }
    if len(receipt.assets) != 24_000 or {
        split: len(families_by_split[split]) for split in expected_families
    } != expected_families:
        raise ValueError("synthetic raster receipt is incomplete")
    # A legacy receipt can be parsed for historical inspection, but cannot
    # become a new calibration/sealed evidence source because it lacks the
    # actual source-tree and build-input bindings added for raster v1.
    if (
        receipt.source_receipt_relative_path is None
        or receipt.build_inputs_relative_path is None
        or receipt.build_inputs_sha256 is None
    ):
        raise ValueError("synthetic raster receipt lacks build provenance")
    return receipt, hashlib.sha256(raw).hexdigest()


def _verify_raster_asset_against_receipt(
    *, root: Path, asset: RasterAssetReceiptV1, pdfium_version: str
) -> None:
    """Re-read the raster bytes that were actually passed to the candidate run."""

    try:
        raw = _read_raw(
            _relative_path(root, asset.asset_relative_path), limit=_MAX_RASTER_ASSET_BYTES
        )
        observed = verify_rendered_raster_output(
            RenderedRasterBytesV1(
                asset_id=asset.asset_id,
                profile_id=asset.profile_id,
                content=raw,
                width=asset.width,
                height=asset.height,
            ),
            pdfium_version=pdfium_version,
        )
    except (OSError, SyntheticRasterCompilerUnavailable, ValueError):
        raise ValueError("raster evaluation asset does not bind the receipt") from None
    if (
        observed.content_sha256 != asset.image_sha256
        or observed.media_type != asset.media_type
        or observed.byte_size != asset.byte_size
        or observed.width != asset.width
        or observed.height != asset.height
    ):
        raise ValueError("raster evaluation asset does not bind the receipt")


def _gold_for_asset(*, root: Path, asset: RasterAssetReceiptV1) -> RasterGoldArtifactV1:
    gold_raw = _read_raw(_relative_path(root, asset.gold_relative_path), limit=_MAX_CASE_BYTES)
    if hashlib.sha256(gold_raw).hexdigest() != asset.gold_sha256:
        raise ValueError("raster evaluation Gold hash does not bind the receipt")
    try:
        gold = RasterGoldArtifactV1.model_validate_json(gold_raw)
    except ValueError as error:
        raise ValueError("raster evaluation Gold artifact is invalid") from error
    if (
        gold.asset_id != asset.asset_id
        or gold.family_id != asset.family_id
        or gold.document_type != asset.document_type
        or gold.profile_id != asset.profile_id
        or gold.source_pdf_sha256 != asset.source_pdf_sha256
    ):
        raise ValueError("raster evaluation Gold artifact does not bind the receipt")
    return gold


def _verify_cases_against_receipt(
    *,
    root: Path,
    cases: Sequence[RasterEvaluationCaseV1],
    receipt: SyntheticRasterReceiptV1,
    split: str,
) -> None:
    """Legacy unit-test seam for Gold binding; the CLI never accepts cases.

    Keeping this small pure verifier avoids weakening the new command-line
    contract: production evaluation exclusively rebuilds cases from a
    canonical candidate bundle and receipt-bound Gold below.
    """

    expected_assets = {
        (asset.asset_id, asset.profile_id): asset
        for asset in receipt.assets
        if asset.split == split
    }
    observed_keys = [(case.asset_id, case.profile_id) for case in cases]
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != set(expected_assets):
        raise ValueError("raster evaluation cases do not cover the receipt split exactly")
    for case in cases:
        asset = expected_assets[(case.asset_id, case.profile_id)]
        if (
            case.source_pdf_sha256 != asset.source_pdf_sha256
            or case.image_sha256 != asset.image_sha256
            or case.document_type != asset.document_type
        ):
            raise ValueError("raster evaluation case does not bind the receipt asset")
        _verify_raster_asset_against_receipt(root=root, asset=asset, pdfium_version=receipt.pdfium_version)
        gold = _gold_for_asset(root=root, asset=asset)
        if (
            _canonical_json([item.model_dump(mode="json", warnings=False) for item in case.gold_spans])
            != _canonical_json([item.model_dump(mode="json", warnings=False) for item in gold.spans])
            or _canonical_json(
                [item.model_dump(mode="json", warnings=False) for item in case.gold_fields]
            )
            != _canonical_json([item.model_dump(mode="json", warnings=False) for item in gold.fields])
            or _canonical_json(
                [item.model_dump(mode="json", warnings=False) for item in case.gold_table_cells]
            )
            != _canonical_json(
                [item.model_dump(mode="json", warnings=False) for item in gold.table_cells]
            )
        ):
            raise ValueError("raster evaluation case Gold does not bind the receipt artifact")


def _rebuild_cases(
    *,
    root: Path,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    split: _CandidateSplit,
    candidate_bundle_relative_path: str,
    candidate_bundle_sha256: str,
) -> tuple[list[RasterEvaluationCaseV1], object, bytes]:
    try:
        bundle, bundle_raw, bundle_digest = read_canonical_raster_candidate_bundle(
            document_preview_root=root,
            relative_path=candidate_bundle_relative_path,
            expected_sha256=candidate_bundle_sha256,
        )
        if bundle_digest != candidate_bundle_sha256:
            raise SyntheticRasterCandidateError("raster_candidate_bundle_hash_mismatch")
        expected_assets = validate_raster_candidate_bundle_against_receipt(
            bundle=bundle,
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            split=split,
        )
    except SyntheticRasterCandidateError:
        raise ValueError("raster evaluation candidate bundle is invalid") from None
    observations = {(item.asset_id, item.profile_id): item for item in bundle.candidates}
    cases: list[RasterEvaluationCaseV1] = []
    for key, asset in sorted(expected_assets.items()):
        _verify_raster_asset_against_receipt(root=root, asset=asset, pdfium_version=receipt.pdfium_version)
        gold = _gold_for_asset(root=root, asset=asset)
        observation = observations[key]
        cases.append(
            RasterEvaluationCaseV1(
                asset_id=asset.asset_id,
                document_type=asset.document_type,
                profile_id=asset.profile_id,
                source_pdf_sha256=asset.source_pdf_sha256,
                image_sha256=asset.image_sha256,
                gold_spans=gold.spans,
                gold_fields=gold.fields,
                gold_table_cells=gold.table_cells,
                candidate=observation.candidate,
            )
        )
    return cases, bundle, bundle_raw


def _write_create_only(root: Path, *, relative_path: str, value: object) -> tuple[Path, str]:
    destination = _relative_path(root, relative_path)
    if destination.is_symlink():
        raise ValueError("local artifact path is invalid")
    payload = _canonical_json(value)
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError("local evaluation artifact is too large")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _relative_path(root, relative_path)
    try:
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Reruns may encounter the same immutable local evidence.  Reuse is
        # valid only when its complete canonical bytes match this calculation;
        # never replace a result from a different candidate/runtime chain.
        if _read_raw(destination, limit=_MAX_ARTIFACT_BYTES) != payload:
            raise ValueError("local evaluation artifact immutable conflict") from None
        return destination, hashlib.sha256(payload).hexdigest()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination, hashlib.sha256(payload).hexdigest()


def _runtime_resolution(
    *, root: Path, runtime_configuration_relative_path: str
) -> RasterCandidateEngineResolutionV1:
    resolution = resolve_ppstructurev3_raster_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or resolution.profile is None
        or resolution.runtime_configuration_sha256 is None
    ):
        raise ValueError("raster evaluation runtime profile is unavailable")
    if re.fullmatch(_SYNTHETIC_RASTER_PROFILE_REVISION, resolution.profile.raster_profile_revision) is None:
        raise ValueError("raster evaluation runtime profile is invalid")
    return resolution


def _runtime_artifact_schema_version(runtime: RasterCandidateEngineResolutionV1) -> Literal["1.0", "2.0"]:
    """Select the receipt shape from the verified runtime profile, never CLI input."""

    if runtime.profile is None:
        raise ValueError("raster evaluation runtime profile is unavailable")
    return "2.0" if isinstance(runtime.profile, PpStructureV3OfflineProfileV2) else "1.0"


def _runtime_inventory_bindings(runtime: RasterCandidateEngineResolutionV1) -> dict[str, str]:
    """Bind one legacy or split-environment profile to evaluation evidence."""

    if runtime.profile is None:
        raise ValueError("raster evaluation runtime profile is unavailable")
    if isinstance(runtime.profile, PpStructureV3OfflineProfileV2):
        return {
            "parent_renderer_package_inventory_sha256": (
                runtime.profile.parent_renderer_package_inventory_sha256
            ),
            "sidecar_package_inventory_sha256": runtime.profile.sidecar_package_inventory_sha256,
            "model_inventory_sha256": runtime.profile.model_inventory_sha256,
        }
    return {
        "package_inventory_sha256": runtime.profile.package_inventory_sha256,
        "model_inventory_sha256": runtime.profile.model_inventory_sha256,
    }


def _candidate_provenance(
    *,
    root: Path,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    split: _CandidateSplit,
    runtime: RasterCandidateEngineResolutionV1,
    candidate_bundle_relative_path: str,
    candidate_execution_relative_path: str,
) -> tuple[list[RasterEvaluationCaseV1], dict[str, object]]:
    if runtime.profile is None or runtime.runtime_configuration_sha256 is None:
        raise ValueError("raster evaluation runtime profile is unavailable")
    try:
        bundle, bundle_raw, bundle_sha256 = read_canonical_raster_candidate_bundle(
            document_preview_root=root,
            relative_path=candidate_bundle_relative_path,
        )
        execution, _execution_raw, execution_sha256 = read_canonical_raster_candidate_execution(
            document_preview_root=root,
            relative_path=candidate_execution_relative_path,
        )
        validate_raster_candidate_bundle_against_receipt(
            bundle=bundle,
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            split=split,
        )
        validate_raster_candidate_execution(
            execution=execution,
            bundle=bundle,
            bundle_raw=bundle_raw,
            bundle_sha256=bundle_sha256,
            runtime_configuration_sha256=runtime.runtime_configuration_sha256,
        )
    except SyntheticRasterCandidateError:
        raise ValueError("raster evaluation candidate provenance is invalid") from None
    if (
        execution.split != split
        or execution.synthetic_raster_receipt_sha256 != receipt_sha256
        or execution.raster_profile_revision != runtime.profile.raster_profile_revision
        or execution.extractor_revision != runtime.profile.extractor_revision
        or bundle.extractor_revision != runtime.profile.extractor_revision
    ):
        raise ValueError("raster evaluation candidate provenance is invalid")
    cases, replayed_bundle, replayed_raw = _rebuild_cases(
        root=root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        split=split,
        candidate_bundle_relative_path=candidate_bundle_relative_path,
        candidate_bundle_sha256=bundle_sha256,
    )
    if replayed_bundle != bundle or replayed_raw != bundle_raw:
        raise ValueError("raster evaluation candidate provenance is invalid")
    return cases, {
        "raster_profile_revision": runtime.profile.raster_profile_revision,
        "runtime_configuration_sha256": runtime.runtime_configuration_sha256,
        **_runtime_inventory_bindings(runtime),
        "extractor_revision": runtime.profile.extractor_revision,
        "candidate_bundle_relative_path": candidate_bundle_relative_path,
        "candidate_bundle_sha256": bundle_sha256,
        "candidate_execution_relative_path": candidate_execution_relative_path,
        "candidate_execution_sha256": execution_sha256,
    }


def _calibration_artifact(arguments: argparse.Namespace, *, root: Path) -> tuple[dict[str, object], str]:
    runtime = _runtime_resolution(
        root=root, runtime_configuration_relative_path=arguments.runtime_configuration_relative_path
    )
    receipt, receipt_sha256 = _load_synthetic_receipt(
        root, relative_path=arguments.synthetic_raster_receipt_relative_path
    )
    cases, bindings = _candidate_provenance(
        root=root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        split="calibration",
        runtime=runtime,
        candidate_bundle_relative_path=arguments.candidate_bundle_relative_path,
        candidate_execution_relative_path=arguments.candidate_execution_relative_path,
    )
    payload: dict[str, object] = {
        "schema_version": _runtime_artifact_schema_version(runtime),
        "dataset_id": "synthetic-raster-v1",
        "split": "calibration",
        **bindings,
        "synthetic_raster_receipt_path": arguments.synthetic_raster_receipt_relative_path,
        "synthetic_raster_receipt_sha256": receipt_sha256,
        "calibration_case_manifest_sha256": synthetic_raster_case_manifest_sha256(cases),
        "thresholds": derive_synthetic_raster_confidence_thresholds(cases),
    }
    _destination, digest = _write_create_only(
        root, relative_path=arguments.output_relative_path, value=payload
    )
    # ``status`` is an in-memory CLI/orchestrator result, not part of the
    # persisted receipt.  The runtime's strict Pydantic receipt models forbid
    # undeclared fields, so adding it to local evidence would make a passing
    # calibration impossible to activate.
    return {"status": "calibrated", **payload}, digest


def evaluate_synthetic_raster_split(
    *,
    document_preview_root: Path,
    runtime_configuration_relative_path: str,
    synthetic_raster_receipt_relative_path: str,
    candidate_bundle_relative_path: str,
    candidate_execution_relative_path: str,
    output_relative_path: str,
    split: _CandidateSplit,
    calibration_threshold_receipt_relative_path: str | None = None,
) -> tuple[dict[str, object], str]:
    """Evaluate one fixed candidate split and create-or-verify its receipt.

    This is the callable counterpart of the CLI for the one-shot local
    activation wrapper.  It accepts only local relative paths and never a
    score, model hash, threshold, Gold, or OCR result from the caller.
    """

    root = _preview_root(document_preview_root)
    arguments = argparse.Namespace(
        runtime_configuration_relative_path=runtime_configuration_relative_path,
        synthetic_raster_receipt_relative_path=synthetic_raster_receipt_relative_path,
        candidate_bundle_relative_path=candidate_bundle_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        output_relative_path=output_relative_path,
        calibration_threshold_receipt_path=calibration_threshold_receipt_relative_path,
    )
    if split == "calibration":
        if calibration_threshold_receipt_relative_path is not None:
            raise ValueError("calibration evaluation cannot consume a threshold receipt")
        return _calibration_artifact(arguments, root=root)
    if calibration_threshold_receipt_relative_path is None:
        raise ValueError("sealed evaluation requires a calibration threshold receipt")
    return _sealed_artifact(arguments, root=root)


def _load_calibration_receipt(*, root: Path, relative_path: str) -> tuple[dict[str, Any], str]:
    raw = _read_raw(_relative_path(root, relative_path), limit=_MAX_ARTIFACT_BYTES)
    try:
        calibration = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("calibration threshold receipt is invalid") from error
    if not isinstance(calibration, dict) or raw != _canonical_json(calibration):
        raise ValueError("calibration threshold receipt is invalid")
    return calibration, hashlib.sha256(raw).hexdigest()


def _sealed_artifact(arguments: argparse.Namespace, *, root: Path) -> tuple[dict[str, object], str]:
    runtime = _runtime_resolution(
        root=root, runtime_configuration_relative_path=arguments.runtime_configuration_relative_path
    )
    receipt, receipt_sha256 = _load_synthetic_receipt(
        root, relative_path=arguments.synthetic_raster_receipt_relative_path
    )
    cases, bindings = _candidate_provenance(
        root=root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        split="sealed",
        runtime=runtime,
        candidate_bundle_relative_path=arguments.candidate_bundle_relative_path,
        candidate_execution_relative_path=arguments.candidate_execution_relative_path,
    )
    calibration, calibration_hash = _load_calibration_receipt(
        root=root, relative_path=arguments.calibration_threshold_receipt_path
    )
    required_calibration = {
        "schema_version": _runtime_artifact_schema_version(runtime),
        "dataset_id": "synthetic-raster-v1",
        "split": "calibration",
        "raster_profile_revision": bindings["raster_profile_revision"],
        "runtime_configuration_sha256": bindings["runtime_configuration_sha256"],
        **_runtime_inventory_bindings(runtime),
        "extractor_revision": bindings["extractor_revision"],
        "synthetic_raster_receipt_path": arguments.synthetic_raster_receipt_relative_path,
        "synthetic_raster_receipt_sha256": receipt_sha256,
    }
    if (
        any(calibration.get(key) != value for key, value in required_calibration.items())
        or not isinstance(calibration.get("calibration_case_manifest_sha256"), str)
        or not isinstance(calibration.get("candidate_bundle_relative_path"), str)
        or not isinstance(calibration.get("candidate_bundle_sha256"), str)
        or not isinstance(calibration.get("candidate_execution_relative_path"), str)
        or not isinstance(calibration.get("candidate_execution_sha256"), str)
    ):
        raise ValueError("calibration threshold receipt is invalid")
    scores = evaluate_synthetic_raster_gates(cases)
    passed = synthetic_raster_gate_passes(scores)
    artifact: dict[str, object] = {
        "schema_version": _runtime_artifact_schema_version(runtime),
        "dataset_id": "synthetic-raster-v1",
        "split": "sealed",
        "status": "passed" if passed else "failed",
        **bindings,
        "synthetic_raster_receipt_path": arguments.synthetic_raster_receipt_relative_path,
        "synthetic_raster_receipt_sha256": receipt_sha256,
        "sealed_case_manifest_sha256": synthetic_raster_case_manifest_sha256(cases),
        "calibration_case_manifest_sha256": calibration["calibration_case_manifest_sha256"],
        "calibration_threshold_receipt_path": arguments.calibration_threshold_receipt_path,
        "calibration_threshold_receipt_sha256": calibration_hash,
        "gate_scores": [score.model_dump(mode="json", warnings=False) for score in scores],
    }
    if not passed:
        artifact["failure_code"] = "sealed_raster_gate_failed"
    _destination, digest = _write_create_only(
        root, relative_path=arguments.output_relative_path, value=artifact
    )
    return artifact, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--runtime-configuration-relative-path", required=True)
    parser.add_argument("--synthetic-raster-receipt-relative-path", required=True)
    parser.add_argument("--candidate-bundle-relative-path", required=True)
    parser.add_argument("--candidate-execution-relative-path", required=True)
    parser.add_argument("--output-relative-path", required=True)
    subparsers = parser.add_subparsers(dest="kind", required=True)
    subparsers.add_parser("calibration")
    sealed = subparsers.add_parser("sealed")
    sealed.add_argument("--calibration-threshold-receipt-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        root = _preview_root(arguments.document_preview_root)
        if arguments.kind == "calibration":
            artifact, digest = _calibration_artifact(arguments, root=root)
        else:
            artifact, digest = _sealed_artifact(arguments, root=root)
    except (OSError, ValueError, SyntheticRasterCandidateError):
        print(
            json.dumps(
                {"status": "failed", "code": "synthetic_raster_evaluation_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            {"status": artifact["status"], "artifact_sha256": digest, "kind": arguments.kind},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] != "failed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
