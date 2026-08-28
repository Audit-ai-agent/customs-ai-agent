"""Run hash-pinned PP-StructureV3 against one synthetic-raster split.

This is an *evaluation-only* runner.  It reads raster bytes and a local
runtime configuration, but never reads Gold artifacts, never downloads a
model, and cannot write an approval pointer.  Its only durable outputs are a
content-addressed candidate bundle and a content-free execution receipt below
the local document-preview root.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterator, Literal, Sequence

from customs_ai_agent.evaluation.synthetic_raster import (
    RasterAssetReceiptV1,
    RasterProfileId,
    RenderedRasterBytesV1,
    SyntheticRasterCompilerUnavailable,
    SyntheticRasterReceiptV1,
    verify_rendered_raster_output,
)
from customs_ai_agent.evaluation.synthetic_raster_candidates import (
    RasterCandidateBundleV1,
    RasterCandidateObservationV1,
    RasterCandidateShardPlanV1,
    SyntheticRasterCandidateError,
    build_raster_candidate_execution,
    build_raster_candidate_shard,
    build_raster_candidate_shard_checkpoint,
    build_raster_candidate_shard_plan,
    candidate_bundle_bytes,
    candidate_shard_bytes,
    candidate_shard_plan_sha256,
    iter_raster_candidate_shard_asset_bindings,
    iter_raster_candidate_shard_definitions,
    read_verified_raster_candidate_shard_checkpoint,
    validate_raster_candidate_bundle_against_receipt,
    validate_raster_candidate_shard_plan_against_receipt,
    write_canonical_raster_candidate_bundle,
    write_canonical_raster_candidate_execution,
    write_canonical_raster_candidate_shard,
    write_canonical_raster_candidate_shard_checkpoint,
    write_canonical_raster_candidate_shard_plan,
)
from customs_ai_agent.schemas.raster_document_extraction import (
    RasterDocumentExtractionResultV2,
    RasterDocumentExtractionRequestMetadataV2,
    RasterPreviewInputV2,
)
from customs_ai_agent.service.document_input import inspect_raster_document_structure
from customs_ai_agent.service.raster_document_extraction import (
    MediaAwareRasterPreviewEngine,
    RasterizedPageV2,
    extract_raster_document_preview,
    raster_preview_candidate_to_result,
)
from customs_ai_agent.service.raster_document_preview_runtime import (
    RasterCandidateEngineResolutionV1,
    PpStructureV3CommandExecutorV2,
    PpStructureV3CandidateBatchSessionV2,
    PpStructureV3EvaluationBatchInputV2,
    PpStructureV3OfflineProfileV2,
    PpStructureV3OfflineProvider,
    resolve_ppstructurev3_raster_candidate,
)


_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_MAX_ASSET_BYTES = 32 * 1024 * 1024
_CandidateSplit = Literal["calibration", "sealed"]
_SYNTHETIC_RASTER_PROFILE_REVISION = r"^(?:synthetic-raster-v1|synthetic-raster-v1:[a-f0-9]{24})$"
_CANDIDATE_SHARD_SIZE = 24
_CANDIDATE_SHARD_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[1]
    / "src"
    / "customs_ai_agent"
    / "evaluation"
    / "synthetic_raster_candidates.py",
    Path(__file__).resolve().parents[1]
    / "src"
    / "customs_ai_agent"
    / "service"
    / "document_input.py",
    Path(__file__).resolve().parents[1]
    / "src"
    / "customs_ai_agent"
    / "service"
    / "raster_document_extraction.py",
    Path(__file__).resolve().parents[1]
    / "src"
    / "customs_ai_agent"
    / "service"
    / "raster_document_preview_runtime.py",
    Path(__file__).resolve().parents[1]
    / "src"
    / "customs_ai_agent"
    / "schemas"
    / "raster_document_extraction.py",
)


class SyntheticRasterCandidateRunnerError(RuntimeError):
    """Content-free candidate-runner error suitable for the CLI boundary."""


@dataclass(frozen=True, slots=True)
class _EvaluationInputLimits:
    """The same strict admission limits used by the raster service path."""

    max_document_bytes: int = 16 * 1024 * 1024
    max_document_pages: int = 20
    max_document_image_pixels: int = 40_000_000


@dataclass(frozen=True, slots=True)
class _EvaluationEngineContext:
    """The sole evaluation-only view of a verified persistent sidecar."""

    engine: MediaAwareRasterPreviewEngine
    session: PpStructureV3CandidateBatchSessionV2
    profile: PpStructureV3OfflineProfileV2


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _preview_root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise SyntheticRasterCandidateRunnerError("raster_candidate_root_invalid")
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise SyntheticRasterCandidateRunnerError("raster_candidate_root_invalid")
        return root
    except SyntheticRasterCandidateRunnerError:
        raise
    except OSError:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_root_invalid") from None


def _relative_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in value
        or "\\" in value
    ):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_path_invalid")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise SyntheticRasterCandidateRunnerError("raster_candidate_path_invalid")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_path_invalid") from None
    return path


def _read_regular(path: Path, *, limit: int, code: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise SyntheticRasterCandidateRunnerError(code)
        raw = path.read_bytes()
    except SyntheticRasterCandidateRunnerError:
        raise
    except OSError:
        raise SyntheticRasterCandidateRunnerError(code) from None
    if not raw or len(raw) > limit:
        raise SyntheticRasterCandidateRunnerError(code)
    return raw


def _load_receipt(
    *, root: Path, relative_path: str
) -> tuple[SyntheticRasterReceiptV1, str]:
    raw = _read_regular(
        _relative_path(root, relative_path), limit=_MAX_RECEIPT_BYTES, code="raster_receipt_invalid"
    )
    try:
        receipt = SyntheticRasterReceiptV1.model_validate_json(raw)
    except ValueError:
        raise SyntheticRasterCandidateRunnerError("raster_receipt_invalid") from None
    if raw != _canonical_json(receipt.model_dump(mode="json", warnings=False)):
        raise SyntheticRasterCandidateRunnerError("raster_receipt_not_canonical")
    expected_families = {"train": 750, "development": 100, "calibration": 50, "sealed": 100}
    families_by_split = {
        split: {asset.family_id for asset in receipt.assets if asset.split == split}
        for split in expected_families
    }
    if len(receipt.assets) != 24_000 or {
        split: len(families_by_split[split]) for split in expected_families
    } != expected_families:
        raise SyntheticRasterCandidateRunnerError("raster_receipt_incomplete")
    if (
        receipt.source_receipt_relative_path is None
        or receipt.build_inputs_relative_path is None
        or receipt.build_inputs_sha256 is None
    ):
        raise SyntheticRasterCandidateRunnerError("raster_receipt_build_provenance_missing")
    return receipt, hashlib.sha256(raw).hexdigest()


def _verified_asset_bytes(
    *, root: Path, asset: RasterAssetReceiptV1, pdfium_version: str
) -> bytes:
    raw = _read_regular(
        _relative_path(root, asset.asset_relative_path),
        limit=_MAX_ASSET_BYTES,
        code="raster_candidate_asset_invalid",
    )
    try:
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
    except (SyntheticRasterCompilerUnavailable, ValueError):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_asset_invalid") from None
    if (
        observed.content_sha256 != asset.image_sha256
        or observed.byte_size != asset.byte_size
        or observed.media_type != asset.media_type
        or observed.width != asset.width
        or observed.height != asset.height
    ):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_asset_binding_invalid")
    return raw


def _request_id(asset: RasterAssetReceiptV1) -> str:
    return f"synthetic-raster-{asset.profile_id}-{asset.image_sha256[:24]}"


@contextmanager
def _evaluation_engine(
    resolution: RasterCandidateEngineResolutionV1,
) -> Iterator[_EvaluationEngineContext]:
    """Expose only the persistent, v2-only sidecar to synthetic evaluation.

    A one-shot service executor is intentionally not an evaluation fallback:
    it would rehash and reload the eight models for every asset, which makes
    the sealed Gate operationally infeasible and creates a different runtime
    shape from the one the bundle was built to attest.
    """

    profile = resolution.profile
    engine = resolution.engine
    if (
        not isinstance(profile, PpStructureV3OfflineProfileV2)
        or not isinstance(engine, MediaAwareRasterPreviewEngine)
        or not isinstance(engine.provider, PpStructureV3OfflineProvider)
        or engine.provider.profile != profile
        or engine.provider.thresholds is not None
        or not isinstance(engine.provider.executor, PpStructureV3CommandExecutorV2)
    ):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_runtime_unavailable")
    executor = engine.provider.executor
    try:
        with executor.open_evaluation_session(profile=profile) as session:
            yield _EvaluationEngineContext(
                engine=MediaAwareRasterPreviewEngine(
                    input_adapter=engine.input_adapter,
                    provider=PpStructureV3OfflineProvider(
                        profile=profile,
                        executor=session,
                        thresholds=None,
                    ),
                ),
                session=session,
                profile=profile,
            )
    except SyntheticRasterCandidateRunnerError:
        raise
    except Exception:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_execution_failed") from None


def _evaluation_item_id(asset: RasterAssetReceiptV1) -> str:
    """Return a compact unique key that does not rely on image SHA alone."""

    material = _canonical_json(
        {
            "asset_id": asset.asset_id,
            "profile_id": asset.profile_id,
            "image_sha256": asset.image_sha256,
        }
    )
    return f"candidate-{hashlib.sha256(material).hexdigest()}"


def _raster_preview_input(
    *, asset: RasterAssetReceiptV1, raw: bytes, raster_profile_revision: str
) -> RasterPreviewInputV2:
    return RasterPreviewInputV2(
        metadata=RasterDocumentExtractionRequestMetadataV2(
            document_asset_id=asset.asset_id,
            extraction_request_id=_request_id(asset),
            source_sha256=asset.image_sha256,
            media_type=asset.media_type,
            raster_profile_revision=raster_profile_revision,
        ),
        byte_size=len(raw),
        content=raw,
    )


def _candidate_runner_implementation_sha256() -> str:
    """Bind resumable shards to every parent-side result-producing module.

    The offline bundle already pins the child interpreter, model, and sidecar
    source.  These files cover the checked-in parent code that admits bytes,
    prepares pages, projects ledgers, and serializes candidate observations.
    A missing, linked, or unexpectedly large source file is an availability
    failure rather than a reason to reuse a previous OCR checkpoint.
    """

    digest = hashlib.sha256()
    try:
        for source in _CANDIDATE_SHARD_IMPLEMENTATION_FILES:
            if source.is_symlink() or not source.is_file():
                raise OSError
            raw = source.read_bytes()
            if not raw or len(raw) > 8 * 1024 * 1024:
                raise OSError
            digest.update(source.name.encode("ascii"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(raw).digest())
    except OSError:
        raise SyntheticRasterCandidateRunnerError(
            "raster_candidate_runner_implementation_unavailable"
        ) from None
    return digest.hexdigest()


def _candidate_observation(
    *,
    asset: RasterAssetReceiptV1,
    candidate: RasterDocumentExtractionResultV2,
    profile: PpStructureV3OfflineProfileV2,
) -> RasterCandidateObservationV1:
    """Keep native outages out of immutable evaluation evidence."""

    if (
        candidate.status == "unavailable"
        or candidate.extractor_revision != profile.extractor_revision
    ):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_execution_failed")
    try:
        return RasterCandidateObservationV1(
            asset_id=asset.asset_id,
            profile_id=asset.profile_id,
            candidate=candidate,
        )
    except (TypeError, ValueError):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_execution_failed") from None


def _evaluate_assets(
    *,
    root: Path,
    receipt: SyntheticRasterReceiptV1,
    assets: Sequence[RasterAssetReceiptV1],
    evaluation_context: _EvaluationEngineContext,
    limits: _EvaluationInputLimits,
) -> list[RasterCandidateObservationV1]:
    """Evaluate one complete shard without writing a partial durable result."""

    profile = evaluation_context.profile
    observations: list[RasterCandidateObservationV1] = []
    if profile.evaluation_batch_size == 1:
        # Preserve replay of an already-issued legacy release exactly.
        for asset in assets:
            raw = _verified_asset_bytes(
                root=root, asset=asset, pdfium_version=receipt.pdfium_version
            )
            try:
                structure = inspect_raster_document_structure(
                    raw, asset.media_type, limits  # type: ignore[arg-type]
                )
                candidate = extract_raster_document_preview(
                    evaluation_context.engine,
                    raster_input=_raster_preview_input(
                        asset=asset,
                        raw=raw,
                        raster_profile_revision=profile.raster_profile_revision,
                    ),
                    structure=structure,
                    settings=limits,  # type: ignore[arg-type]
                )
            except Exception:
                raise SyntheticRasterCandidateRunnerError(
                    "raster_candidate_execution_failed"
                ) from None
            observations.append(
                _candidate_observation(asset=asset, candidate=candidate, profile=profile)
            )
        return observations
    if profile.evaluation_batch_size != 4:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_runtime_unavailable")

    for offset in range(0, len(assets), 4):
        selected = assets[offset : offset + 4]
        prepared: list[
            tuple[RasterAssetReceiptV1, RasterPreviewInputV2, tuple[RasterizedPageV2, ...]]
        ] = []
        try:
            for asset in selected:
                raw = _verified_asset_bytes(
                    root=root, asset=asset, pdfium_version=receipt.pdfium_version
                )
                if len(raw) > limits.max_document_bytes:
                    raise ValueError("evaluation asset exceeds service byte limit")
                structure = inspect_raster_document_structure(
                    raw, asset.media_type, limits  # type: ignore[arg-type]
                )
                # The synthetic corpus is created without an EXIF rotation. A
                # rotated JPEG cannot enter a geometry batch, just as the
                # service routes it to review.
                if (
                    asset.media_type == "image/jpeg"
                    and structure.jpeg_exif_orientation not in {None, 1}
                ):
                    raise ValueError("evaluation jpeg orientation is unsupported")
                raster_input = _raster_preview_input(
                    asset=asset,
                    raw=raw,
                    raster_profile_revision=profile.raster_profile_revision,
                )
                pages = evaluation_context.engine.prepare_pages(raster_input, structure)
                if len(pages) != 1:
                    raise ValueError("evaluation batch requires a single page")
                prepared.append((asset, raster_input, pages))
            batch_items = tuple(
                PpStructureV3EvaluationBatchInputV2(
                    item_id=_evaluation_item_id(asset),
                    pages=pages,
                    source_sha256=raster_input.metadata.source_sha256,
                    raster_profile_revision=raster_input.metadata.raster_profile_revision,
                )
                for asset, raster_input, pages in prepared
            )
            ledgers = evaluation_context.session.extract_layout_batch(
                batch_items, profile=profile
            )
            if len(ledgers) != len(prepared):
                raise ValueError("evaluation batch result count is invalid")
            for (asset, raster_input, pages), ledger in zip(prepared, ledgers, strict=True):
                preview = evaluation_context.engine.preview_from_ledger(
                    raster_input=raster_input, pages=pages, ledger=ledger
                )
                observations.append(
                    _candidate_observation(
                        asset=asset,
                        candidate=raster_preview_candidate_to_result(
                            raster_input=raster_input, candidate=preview
                        ),
                        profile=profile,
                    )
                )
        except SyntheticRasterCandidateRunnerError:
            raise
        except Exception:
            # A valid deterministic raster should be admitted just like the
            # service route. Do not convert a batch defect into cosmetic
            # all-zero candidate evidence.
            raise SyntheticRasterCandidateRunnerError(
                "raster_candidate_execution_failed"
            ) from None
    return observations


def _assets_for_shard(
    *,
    plan: RasterCandidateShardPlanV1,
    shard_ordinal: int,
    assets_by_key: dict[tuple[str, RasterProfileId], RasterAssetReceiptV1],
) -> tuple[RasterAssetReceiptV1, ...]:
    """Resolve a plan's safe asset bindings back to the verified receipt rows."""

    selected: list[RasterAssetReceiptV1] = []
    try:
        bindings = iter_raster_candidate_shard_asset_bindings(
            plan=plan, shard_ordinal=shard_ordinal
        )
        for binding in bindings:
            asset = assets_by_key[(binding.asset_id, binding.profile_id)]
            if (
                asset.source_pdf_sha256 != binding.source_pdf_sha256
                or asset.image_sha256 != binding.image_sha256
                or asset.media_type != binding.media_type
                or asset.byte_size != binding.byte_size
                or asset.width != binding.width
                or asset.height != binding.height
            ):
                raise ValueError("planned asset binding changed")
            selected.append(asset)
    except (KeyError, SyntheticRasterCandidateError, ValueError):
        raise SyntheticRasterCandidateRunnerError(
            "raster_candidate_shard_plan_binding_invalid"
        ) from None
    return tuple(selected)


def run_ppstructurev3_synthetic_raster_candidates(
    *,
    document_preview_root: Path,
    runtime_configuration_relative_path: str,
    synthetic_raster_receipt_relative_path: str,
    split: _CandidateSplit,
    resolver=resolve_ppstructurev3_raster_candidate,
) -> dict[str, object]:
    """Create one exact-coverage candidate bundle from the real local engine.

    ``resolver`` is injectable solely for local component tests.  The default
    is the PP-StructureV3 resolver which rechecks package/model/sidecar hashes
    and refuses all network/download paths before this loop begins.
    """

    root = _preview_root(document_preview_root)
    resolution: RasterCandidateEngineResolutionV1 = resolver(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or resolution.engine is None
        or resolution.profile is None
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_runtime_unavailable")
    if not isinstance(resolution.profile, PpStructureV3OfflineProfileV2):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_runtime_unavailable")
    profile = resolution.profile
    if (
        re.fullmatch(
            _SYNTHETIC_RASTER_PROFILE_REVISION, profile.raster_profile_revision
        )
        is None
    ):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_profile_revision_invalid")
    receipt, receipt_sha256 = _load_receipt(
        root=root, relative_path=synthetic_raster_receipt_relative_path
    )
    if receipt.pdfium_version != profile.pdfium_version:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_pdfium_version_mismatch")
    assets = sorted(
        (asset for asset in receipt.assets if asset.split == split),
        key=lambda asset: (asset.asset_id, asset.profile_id),
    )
    if not assets:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_split_empty")
    limits = _EvaluationInputLimits()
    runner_implementation_sha256 = _candidate_runner_implementation_sha256()
    try:
        shard_plan = build_raster_candidate_shard_plan(
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            split=split,
            runtime_configuration_sha256=resolution.runtime_configuration_sha256,
            raster_profile_revision=profile.raster_profile_revision,
            extractor_revision=profile.extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            shard_size=_CANDIDATE_SHARD_SIZE,
            evaluation_batch_size=profile.evaluation_batch_size,
        )
        validate_raster_candidate_shard_plan_against_receipt(
            plan=shard_plan, receipt=receipt, receipt_sha256=receipt_sha256, split=split
        )
        (
            shard_plan_relative_path,
            shard_plan_sha256,
            _shard_plan_byte_size,
        ) = write_canonical_raster_candidate_shard_plan(
            document_preview_root=root, plan=shard_plan
        )
        if shard_plan_sha256 != candidate_shard_plan_sha256(shard_plan):
            raise SyntheticRasterCandidateError("raster_candidate_shard_plan_invalid")
    except SyntheticRasterCandidateError:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_shard_plan_invalid") from None

    assets_by_key = {(asset.asset_id, asset.profile_id): asset for asset in assets}
    completed_by_ordinal: dict[int, tuple[RasterCandidateObservationV1, ...]] = {}
    pending_ordinals: list[int] = []
    reused_shard_count = 0
    for definition in iter_raster_candidate_shard_definitions(shard_plan):
        try:
            _checkpoint, durable_shard = read_verified_raster_candidate_shard_checkpoint(
                document_preview_root=root,
                plan=shard_plan,
                shard_ordinal=definition.shard_ordinal,
            )
        except SyntheticRasterCandidateError as error:
            if str(error) != "raster_candidate_shard_checkpoint_unavailable":
                raise SyntheticRasterCandidateRunnerError(
                    "raster_candidate_shard_checkpoint_invalid"
                ) from None
            pending_ordinals.append(definition.shard_ordinal)
            continue
        completed_by_ordinal[definition.shard_ordinal] = tuple(durable_shard.candidates)
        reused_shard_count += 1

    # A checkpoint is written only after every asset in its fixed shard has
    # completed.  A crash therefore loses at most 24 observations (six native
    # batch-4 calls) and can never produce an incomplete candidate bundle.
    if pending_ordinals:
        with _evaluation_engine(resolution) as evaluation_context:
            for shard_ordinal in pending_ordinals:
                selected = _assets_for_shard(
                    plan=shard_plan,
                    shard_ordinal=shard_ordinal,
                    assets_by_key=assets_by_key,
                )
                observations = _evaluate_assets(
                    root=root,
                    receipt=receipt,
                    assets=selected,
                    evaluation_context=evaluation_context,
                    limits=limits,
                )
                try:
                    shard = build_raster_candidate_shard(
                        plan=shard_plan,
                        shard_ordinal=shard_ordinal,
                        candidates=observations,
                    )
                    shard_relative_path, _shard_sha256, _shard_byte_size = (
                        write_canonical_raster_candidate_shard(
                            document_preview_root=root, shard=shard
                        )
                    )
                    checkpoint = build_raster_candidate_shard_checkpoint(
                        plan=shard_plan,
                        shard=shard,
                        candidate_shard_relative_path_value=shard_relative_path,
                        candidate_shard_raw=candidate_shard_bytes(shard),
                    )
                    write_canonical_raster_candidate_shard_checkpoint(
                        document_preview_root=root, checkpoint=checkpoint
                    )
                    # Re-read from the final checkpoint rather than retaining
                    # in-memory output; a durable receipt is the only resume
                    # evidence accepted by the eventual merged bundle.
                    _checkpoint, durable_shard = read_verified_raster_candidate_shard_checkpoint(
                        document_preview_root=root,
                        plan=shard_plan,
                        shard_ordinal=shard_ordinal,
                    )
                except SyntheticRasterCandidateError:
                    raise SyntheticRasterCandidateRunnerError(
                        "raster_candidate_shard_artifact_invalid"
                    ) from None
                completed_by_ordinal[shard_ordinal] = tuple(durable_shard.candidates)

    try:
        observations = [
            observation
            for definition in iter_raster_candidate_shard_definitions(shard_plan)
            for observation in completed_by_ordinal[definition.shard_ordinal]
        ]
    except KeyError:
        raise SyntheticRasterCandidateRunnerError(
            "raster_candidate_shard_coverage_invalid"
        ) from None
    if len(observations) != len(assets):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_shard_coverage_invalid")

    # A model/package/sidecar/pointer-equivalent replacement after process
    # start cannot be accepted merely because the child completed.  Replay
    # the immutable input set and re-resolve the exact v2 runtime before any
    # candidate artifact is created.
    post_receipt, post_receipt_sha256 = _load_receipt(
        root=root, relative_path=synthetic_raster_receipt_relative_path
    )
    if post_receipt_sha256 != receipt_sha256 or post_receipt != receipt:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_receipt_drift")
    for asset in assets:
        _verified_asset_bytes(root=root, asset=asset, pdfium_version=receipt.pdfium_version)
    post_resolution: RasterCandidateEngineResolutionV1 = resolver(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        post_resolution.status != "available"
        or post_resolution.profile is None
        or post_resolution.runtime_configuration_sha256 != resolution.runtime_configuration_sha256
        or post_resolution.profile != profile
        or _candidate_runner_implementation_sha256() != runner_implementation_sha256
    ):
        raise SyntheticRasterCandidateRunnerError("raster_candidate_runtime_drift")
    bundle = RasterCandidateBundleV1(
        split=split,
        synthetic_raster_receipt_sha256=receipt_sha256,
        raster_profile_revision=profile.raster_profile_revision,
        extractor_revision=profile.extractor_revision,
        candidates=observations,
    )
    try:
        validate_raster_candidate_bundle_against_receipt(
            bundle=bundle, receipt=receipt, receipt_sha256=receipt_sha256, split=split
        )
        bundle_relative_path, bundle_sha256, bundle_byte_size = (
            write_canonical_raster_candidate_bundle(document_preview_root=root, bundle=bundle)
        )
        execution = build_raster_candidate_execution(
            bundle=bundle,
            candidate_bundle_relative_path_value=bundle_relative_path,
            candidate_bundle_raw=candidate_bundle_bytes(bundle),
            runtime_configuration_sha256=resolution.runtime_configuration_sha256,
        )
        execution_relative_path, execution_sha256 = write_canonical_raster_candidate_execution(
            document_preview_root=root, execution=execution
        )
    except SyntheticRasterCandidateError:
        raise SyntheticRasterCandidateRunnerError("raster_candidate_artifact_invalid") from None
    return {
        "status": "synthetic_raster_candidates_written",
        "split": split,
        "candidate_count": len(observations),
        "candidate_bundle_relative_path": bundle_relative_path,
        "candidate_bundle_sha256": bundle_sha256,
        "candidate_bundle_byte_size": bundle_byte_size,
        "candidate_execution_relative_path": execution_relative_path,
        "candidate_execution_sha256": execution_sha256,
        "candidate_shard_plan_relative_path": shard_plan_relative_path,
        "candidate_shard_plan_sha256": shard_plan_sha256,
        "candidate_shard_count": len(iter_raster_candidate_shard_definitions(shard_plan)),
        "reused_candidate_shard_count": reused_shard_count,
        "runtime_configuration_sha256": resolution.runtime_configuration_sha256,
        "raster_profile_revision": profile.raster_profile_revision,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--runtime-configuration-relative-path", required=True)
    parser.add_argument("--synthetic-raster-receipt-relative-path", required=True)
    parser.add_argument("--split", choices=["calibration", "sealed"], required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_ppstructurev3_synthetic_raster_candidates(
            document_preview_root=arguments.document_preview_root,
            runtime_configuration_relative_path=arguments.runtime_configuration_relative_path,
            synthetic_raster_receipt_relative_path=arguments.synthetic_raster_receipt_relative_path,
            split=arguments.split,
        )
    except (OSError, ValueError, SyntheticRasterCandidateRunnerError):
        print(
            json.dumps(
                {"status": "failed", "code": "synthetic_raster_candidate_run_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
