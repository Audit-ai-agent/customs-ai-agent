"""One local, non-human activation transaction for ``synthetic-raster-v1``.

This is deliberately an orchestration boundary, not an alternative evaluator:
it accepts no score, OCR JSON, model name/hash, confirmation flag, or network
option.  It never provisions/downloads a missing bundle.  It can derive a
new immutable local release only from a fully verified existing local bundle
when checked-in first-party source/configuration changes; from then on it
creates/verifies the fixed build inputs, produces the source-attested dataset,
runs PP-StructureV3 over the calibration and sealed splits, evaluates all 12
gates, then atomically replaces the active pointer *only* after every
material has been replayed.

The final pointer writer is the sole mutable activation step.  Any failure
before it leaves an existing pointer untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys
from typing import Any, Iterator

from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.evaluation.raster_offline_bundle import (
    RasterOfflineBundleError,
    prepare_raster_runtime_profile_v2,
    verify_parent_renderer_inventory,
)
from customs_ai_agent.evaluation.raster_offline_bundle_bootstrap import (
    RasterOfflineBundleBootstrapError,
    ensure_raster_offline_bundle_release_v1,
)
from customs_ai_agent.evaluation.synthetic_raster import (
    SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
    SyntheticRasterCompilerUnavailable,
    compile_synthetic_raster_dataset,
    ensure_fixed_synthetic_raster_build_inputs,
    read_verified_synthetic_raster_dataset,
)


_ACTIVATION_LOCK_NAME = ".synthetic-raster-v1-auto-activation.lock"


class SyntheticRasterAutoActivationError(RuntimeError):
    """Content-free non-human activation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SyntheticRasterAutoActivationResultV1:
    """Small, content-free result suitable for a local CLI transcript."""

    raster_profile_revision: str
    activation_pointer_sha256: str
    sealed_evaluation_relative_path: str
    sealed_evaluation_sha256: str


@dataclass(frozen=True, slots=True)
class _CandidateArtifacts:
    bundle_relative_path: str
    bundle_sha256: str
    execution_relative_path: str
    execution_sha256: str


@contextmanager
def _deny_parent_network_connections() -> Iterator[None]:
    """Deny network connections for the complete local activation transaction.

    The PP-Structure child already applies its own socket guard, but the
    parent process builds the raster corpus, derives releases, and imports the
    evaluator/runner before that child starts.  Those operations are designed
    to be local-only; making the restriction executable prevents a future
    helper or dependency from silently widening the activation boundary.
    """

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def _deny_connect(_socket: socket.socket, _address: object) -> None:
        raise OSError("network access denied")

    def _deny_connect_ex(_socket: socket.socket, _address: object) -> int:
        return getattr(socket, "EAI_FAIL", 1)

    def _deny_create_connection(*_args: object, **_kwargs: object) -> socket.socket:
        raise OSError("network access denied")

    setattr(socket.socket, "connect", _deny_connect)
    setattr(socket.socket, "connect_ex", _deny_connect_ex)
    setattr(socket, "create_connection", _deny_create_connection)
    try:
        yield
    finally:
        setattr(socket.socket, "connect", original_connect)
        setattr(socket.socket, "connect_ex", original_connect_ex)
        setattr(socket, "create_connection", original_create_connection)


@contextmanager
def _activation_lock(document_preview_root: Path) -> Iterator[None]:
    """Serialize whole transactions without requiring stale-marker cleanup."""

    lock_path = document_preview_root / _ACTIVATION_LOCK_NAME
    try:
        with exclusive_local_process_lock(lock_path):
            yield
    except LocalProcessLockBusy:
        raise SyntheticRasterAutoActivationError("raster_auto_activation_locked") from None
    except LocalProcessLockInvalid:
        raise SyntheticRasterAutoActivationError("raster_auto_activation_lock_invalid") from None


def _profile_suffix(raster_profile_revision: str) -> str:
    try:
        suffix = raster_profile_revision.rsplit(":", 1)[1]
    except IndexError:
        raise SyntheticRasterAutoActivationError("raster_auto_activation_profile_invalid") from None
    if len(suffix) != 24 or any(char not in "0123456789abcdef" for char in suffix):
        raise SyntheticRasterAutoActivationError("raster_auto_activation_profile_invalid")
    return suffix


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _required_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SyntheticRasterAutoActivationError("raster_auto_activation_candidate_invalid")
    return value


def _evaluation_relative_path(
    *,
    raster_profile_revision: str,
    runtime_configuration_sha256: str,
    split: str,
    candidate: _CandidateArtifacts,
    evaluator_implementation_sha256: str,
    calibration_receipt_sha256: str | None = None,
) -> str:
    """Return an immutable evaluation path from all result-affecting bytes.

    A failed sealed receipt stays preserved, but it can never prevent a later
    corrected candidate/evaluator from being measured.  Identical material
    resolves to exactly the same path and is create-or-verify reusable.
    """

    suffix = _profile_suffix(raster_profile_revision)
    if split not in {"calibration", "sealed"}:
        raise SyntheticRasterAutoActivationError("raster_auto_activation_evaluation_invalid")
    run_material: dict[str, object] = {
        "schema": "synthetic-raster-activation-run-v1",
        "raster_profile_revision": raster_profile_revision,
        "runtime_configuration_sha256": _required_sha256(runtime_configuration_sha256),
        "split": split,
        "candidate_bundle_sha256": candidate.bundle_sha256,
        "candidate_execution_sha256": candidate.execution_sha256,
        "evaluator_implementation_sha256": _required_sha256(evaluator_implementation_sha256),
    }
    if split == "sealed":
        if calibration_receipt_sha256 is None:
            raise SyntheticRasterAutoActivationError("raster_auto_activation_evaluation_invalid")
        run_material["calibration_receipt_sha256"] = _required_sha256(calibration_receipt_sha256)
    elif calibration_receipt_sha256 is not None:
        raise SyntheticRasterAutoActivationError("raster_auto_activation_evaluation_invalid")
    return (
        f"synthetic-raster-v1/activation-evaluations/{suffix}/{split}/{_sha256(run_material)}.json"
    )


def _candidate_result_artifacts(
    result: object,
    *,
    raster_profile_revision: str,
    runtime_configuration_sha256: str,
) -> _CandidateArtifacts:
    if not isinstance(result, dict):
        raise SyntheticRasterAutoActivationError("raster_auto_activation_candidate_invalid")
    bundle = result.get("candidate_bundle_relative_path")
    execution = result.get("candidate_execution_relative_path")
    if (
        not isinstance(bundle, str)
        or not isinstance(execution, str)
        or result.get("raster_profile_revision") != raster_profile_revision
        or result.get("runtime_configuration_sha256") != runtime_configuration_sha256
    ):
        raise SyntheticRasterAutoActivationError("raster_auto_activation_candidate_invalid")
    return _CandidateArtifacts(
        bundle_relative_path=bundle,
        bundle_sha256=_required_sha256(result.get("candidate_bundle_sha256")),
        execution_relative_path=execution,
        execution_sha256=_required_sha256(result.get("candidate_execution_sha256")),
    )


def _sealed_result_sha256(artifact: object, digest: object) -> str:
    if not isinstance(artifact, dict) or artifact.get("status") != "passed":
        raise SyntheticRasterAutoActivationError("sealed_raster_gate_failed")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SyntheticRasterAutoActivationError("raster_auto_activation_evaluation_invalid")
    return digest


def _load_checked_in_synthetic_raster_tooling() -> tuple[Any, Any]:
    """Load the exact local evaluator and candidate runner from this checkout.

    The public activation entrypoint is a directly executable script under
    ``tools/``.  Python then places that directory, rather than the checkout
    root, on ``sys.path``.  Bind the two required files to the source checkout
    explicitly so the package-level orchestrator works for that normal
    invocation too, and reject a similarly named module from somewhere else.
    """

    try:
        checkout_root = Path(__file__).resolve(strict=True).parents[3]
        tools_root = checkout_root / "tools"
        expected_evaluator = tools_root / "evaluate_synthetic_raster.py"
        expected_runner = tools_root / "run_ppstructurev3_synthetic_raster_candidates.py"
        for path in (tools_root, expected_evaluator, expected_runner):
            if path.is_symlink() or not path.exists():
                raise ValueError("local synthetic raster tooling is unavailable")
        if (
            not tools_root.is_dir()
            or not expected_evaluator.is_file()
            or not expected_runner.is_file()
        ):
            raise ValueError("local synthetic raster tooling is unavailable")
        checkout_value = str(checkout_root)
        if not sys.path or sys.path[0] != checkout_value:
            sys.path.insert(0, checkout_value)
        evaluator = importlib.import_module("tools.evaluate_synthetic_raster")
        runner = importlib.import_module("tools.run_ppstructurev3_synthetic_raster_candidates")
        if Path(str(evaluator.__file__)).resolve(strict=True) != expected_evaluator.resolve(
            strict=True
        ) or Path(str(runner.__file__)).resolve(strict=True) != expected_runner.resolve(
            strict=True
        ):
            raise ValueError("local synthetic raster tooling is unavailable")
    except (ImportError, OSError, ValueError):
        raise SyntheticRasterAutoActivationError(
            "raster_auto_activation_tooling_unavailable"
        ) from None
    return evaluator, runner


def _auto_activate_synthetic_raster_v1(
    *, document_preview_root: Path
) -> SyntheticRasterAutoActivationResultV1:
    """Run the whole automatic source-attested raster activation transaction.

    The evaluation and activation tools are local modules and accept no
    operator-provided model or evidence values.  A missing bundle remains an
    availability failure; this path never installs or downloads packages or
    models.  It may only derive a new content-addressed release from an
    already verified local bundle.
    """

    try:
        bundle = ensure_raster_offline_bundle_release_v1(
            document_preview_root=document_preview_root
        )
    except RasterOfflineBundleBootstrapError as error:
        raise SyntheticRasterAutoActivationError(error.code) from None
    except RasterOfflineBundleError as error:
        raise SyntheticRasterAutoActivationError(error.code) from None
    with _activation_lock(bundle.document_preview_root):
        try:
            # Step 1. The normal Agent only verifies PDFium/Pillow.  The
            # PP-Structure sidecar verifies its own packages immediately
            # before import in each candidate execution.
            verify_parent_renderer_inventory(bundle)
            build_inputs = ensure_fixed_synthetic_raster_build_inputs(
                document_preview_root=bundle.document_preview_root
            )

            # Step 2. This performs the 6,000-source preflight before staging
            # and publishes only the complete 24,000-asset dataset.
            try:
                compile_synthetic_raster_dataset(
                    document_preview_root=bundle.document_preview_root,
                    build_inputs_relative_path=build_inputs.build_inputs_relative_path,
                )
            except ValueError:
                # The detailed parser/PDFium failure may contain local source
                # structure.  The non-human wrapper reports only its durable
                # phase, while the builder leaves dataset/pointer untouched.
                raise SyntheticRasterAutoActivationError(
                    "synthetic_raster_dataset_build_failed"
                ) from None
            # ``compile_synthetic_raster_dataset`` already replays all
            # material before publishing a new staging tree and, when a final
            # dataset already exists, returns only after the same full replay.
            # Repeating it immediately here added a complete 48,000-artifact
            # pass without a later activation boundary.  Candidate and
            # pointer-boundary replays remain mandatory below.
            # Runtime evidence lives beside an already published dataset.  It
            # must not create ``synthetic-raster-v1/`` early, because an
            # unfinished directory at that name intentionally blocks dataset
            # publish/reuse as a fail-closed partial output.
            profile = prepare_raster_runtime_profile_v2(
                bundle,
                pdfium_version=build_inputs.renderer_configuration.pdfium_version,
            )

            # Imports occur only after the bundle/profile has been verified;
            # neither runner accepts Gold/OCR/model/operator input values.
            evaluator, candidate_runner = _load_checked_in_synthetic_raster_tooling()
            evaluate_synthetic_raster_split = evaluator.evaluate_synthetic_raster_split
            evaluation_implementation_sha256 = evaluator.evaluation_implementation_sha256
            run_ppstructurev3_synthetic_raster_candidates = (
                candidate_runner.run_ppstructurev3_synthetic_raster_candidates
            )
            from customs_ai_agent.service.raster_document_preview_runtime import (
                write_source_attested_synthetic_raster_preview_pointer,
            )

            calibration_result = run_ppstructurev3_synthetic_raster_candidates(
                document_preview_root=bundle.document_preview_root,
                runtime_configuration_relative_path=profile.runtime_configuration_relative_path,
                synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                split="calibration",
            )
            calibration_candidate = _candidate_result_artifacts(
                calibration_result,
                raster_profile_revision=profile.raster_profile_revision,
                runtime_configuration_sha256=profile.runtime_configuration_sha256,
            )
            evaluator_sha256 = evaluation_implementation_sha256()
            calibration_path = _evaluation_relative_path(
                raster_profile_revision=profile.raster_profile_revision,
                runtime_configuration_sha256=profile.runtime_configuration_sha256,
                split="calibration",
                candidate=calibration_candidate,
                evaluator_implementation_sha256=evaluator_sha256,
            )
            calibration_artifact, _calibration_sha256 = evaluate_synthetic_raster_split(
                document_preview_root=bundle.document_preview_root,
                runtime_configuration_relative_path=profile.runtime_configuration_relative_path,
                synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                candidate_bundle_relative_path=calibration_candidate.bundle_relative_path,
                candidate_execution_relative_path=calibration_candidate.execution_relative_path,
                output_relative_path=calibration_path,
                split="calibration",
            )
            if calibration_artifact.get("status") != "calibrated":
                raise SyntheticRasterAutoActivationError(
                    "raster_auto_activation_calibration_invalid"
                )
            sealed_result = run_ppstructurev3_synthetic_raster_candidates(
                document_preview_root=bundle.document_preview_root,
                runtime_configuration_relative_path=profile.runtime_configuration_relative_path,
                synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                split="sealed",
            )
            sealed_candidate = _candidate_result_artifacts(
                sealed_result,
                raster_profile_revision=profile.raster_profile_revision,
                runtime_configuration_sha256=profile.runtime_configuration_sha256,
            )
            sealed_path = _evaluation_relative_path(
                raster_profile_revision=profile.raster_profile_revision,
                runtime_configuration_sha256=profile.runtime_configuration_sha256,
                split="sealed",
                candidate=sealed_candidate,
                evaluator_implementation_sha256=evaluator_sha256,
                calibration_receipt_sha256=_calibration_sha256,
            )
            sealed_artifact, sealed_sha256 = evaluate_synthetic_raster_split(
                document_preview_root=bundle.document_preview_root,
                runtime_configuration_relative_path=profile.runtime_configuration_relative_path,
                synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                candidate_bundle_relative_path=sealed_candidate.bundle_relative_path,
                candidate_execution_relative_path=sealed_candidate.execution_relative_path,
                output_relative_path=sealed_path,
                split="sealed",
                calibration_threshold_receipt_relative_path=calibration_path,
            )
            sealed_sha256 = _sealed_result_sha256(sealed_artifact, sealed_sha256)

            # Step 5. Re-read all source/build/raster/Gold bytes and the
            # bundle immediately before the only mutable operation.
            # Re-run release selection as well as byte verification.  If the
            # checked-in sidecar/configuration changed during a long sealed
            # run, this derives/selects a distinct release and the identity
            # comparison below stops this older evidence from being activated.
            rechecked_bundle = ensure_raster_offline_bundle_release_v1(
                document_preview_root=bundle.document_preview_root
            )
            verify_parent_renderer_inventory(rechecked_bundle)
            if (
                rechecked_bundle.raster_profile_revision != profile.raster_profile_revision
                or rechecked_bundle.manifest_sha256 != bundle.manifest_sha256
            ):
                raise SyntheticRasterAutoActivationError("offline_bundle_drift")
            read_verified_synthetic_raster_dataset(
                document_preview_root=bundle.document_preview_root,
                expected_build_inputs_relative_path=build_inputs.build_inputs_relative_path,
            )
            pointer_sha256, pointer = write_source_attested_synthetic_raster_preview_pointer(
                document_preview_root=bundle.document_preview_root,
                runtime_configuration_relative_path=profile.runtime_configuration_relative_path,
                sealed_evaluation_relative_path=sealed_path,
            )
            if (
                pointer.raster_profile_revision != profile.raster_profile_revision
                or pointer.sealed_evaluation_sha256 != sealed_sha256
            ):
                raise SyntheticRasterAutoActivationError("raster_auto_activation_pointer_invalid")
            return SyntheticRasterAutoActivationResultV1(
                raster_profile_revision=profile.raster_profile_revision,
                activation_pointer_sha256=pointer_sha256,
                sealed_evaluation_relative_path=sealed_path,
                sealed_evaluation_sha256=sealed_sha256,
            )
        except SyntheticRasterAutoActivationError:
            raise
        except RasterOfflineBundleBootstrapError as error:
            raise SyntheticRasterAutoActivationError(error.code) from None
        except RasterOfflineBundleError as error:
            raise SyntheticRasterAutoActivationError(error.code) from None
        except SyntheticRasterCompilerUnavailable:
            raise SyntheticRasterAutoActivationError("synthetic_raster_build_unavailable") from None
        except (OSError, ValueError, RuntimeError):
            # Keep the CLI transcript free of source bytes, OCR text, native
            # sidecar diagnostics, paths, and implementation stack details.
            raise SyntheticRasterAutoActivationError("raster_auto_activation_failed") from None


def auto_activate_synthetic_raster_v1(
    *, document_preview_root: Path
) -> SyntheticRasterAutoActivationResultV1:
    """Run a strictly local transaction with parent and child network denial."""

    with _deny_parent_network_connections():
        return _auto_activate_synthetic_raster_v1(document_preview_root=document_preview_root)


__all__ = [
    "SyntheticRasterAutoActivationError",
    "SyntheticRasterAutoActivationResultV1",
    "auto_activate_synthetic_raster_v1",
]
