"""Immutable local candidate artifacts for ``synthetic-raster-v1``.

The raster Gold chain and a model's OCR/layout output must never share a
file.  This module therefore stores only the candidate side in a canonical,
content-addressed bundle, then records the exact local runtime configuration
which produced that bundle in a small execution receipt.  Gold is re-read
only by the evaluator from the synthetic-raster receipt.

All paths are deliberately restricted to the Git-excluded
``_local/data/document-preview`` root.  These artifacts may contain synthetic
OCR text, but no source bytes, model errors, or public transcript data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.synthetic_raster import (
    RasterAssetReceiptV1,
    RasterProfileId,
    SyntheticRasterReceiptV1,
    synthetic_raster_receipt_sha256,
)
from customs_ai_agent.schemas.raster_document_extraction import RasterDocumentExtractionResultV2


_SHA256 = r"^[a-f0-9]{64}$"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_RASTER_PROFILE_REVISION = r"^(?:synthetic-raster-v1|synthetic-raster-v1:[a-f0-9]{24})$"
_MAX_CANDIDATE_BUNDLE_BYTES = 128 * 1024 * 1024
_MAX_EXECUTION_RECEIPT_BYTES = 1024 * 1024
_MAX_CANDIDATE_SHARD_PLAN_BYTES = 32 * 1024 * 1024
_MAX_CANDIDATE_SHARD_BYTES = 32 * 1024 * 1024
_MAX_CANDIDATE_SHARD_CHECKPOINT_BYTES = 1024 * 1024
_MAX_CANDIDATE_SHARD_SIZE = 64
_CandidateSplit = Literal["calibration", "sealed"]
_SHARD_ID = r"^shard-[0-9]{6}-[a-f0-9]{24}$"


class SyntheticRasterCandidateError(ValueError):
    """Content-free local candidate/evaluation provenance failure."""


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _require_sha256(value: str, *, code: str) -> str:
    if not isinstance(value, str) or re.fullmatch(_SHA256, value) is None:
        raise SyntheticRasterCandidateError(code)
    return value


def _require_candidate_split(value: str) -> _CandidateSplit:
    if value not in {"calibration", "sealed"}:
        raise SyntheticRasterCandidateError("raster_candidate_shard_split_invalid")
    return value  # type: ignore[return-value]


def _deterministic_shard_id(
    *,
    split: _CandidateSplit,
    synthetic_raster_receipt_sha256: str,
    runtime_configuration_sha256: str,
    raster_profile_revision: str,
    extractor_revision: str,
    runner_implementation_sha256: str,
    shard_size: int,
    evaluation_batch_size: int,
    shard_ordinal: int,
    asset_bindings: Sequence[Mapping[str, object]],
) -> str:
    """Derive a shard key from every material that can change its OCR work.

    The full plan hash cannot be used here because a plan contains these IDs,
    which would create a circular identity.  The same binding fields are
    instead included directly in this compact shard material.
    """

    material = {
        "schema": "synthetic-raster-candidate-shard-id-v1",
        "split": split,
        "synthetic_raster_receipt_sha256": synthetic_raster_receipt_sha256,
        "runtime_configuration_sha256": runtime_configuration_sha256,
        "raster_profile_revision": raster_profile_revision,
        "extractor_revision": extractor_revision,
        "runner_implementation_sha256": runner_implementation_sha256,
        "shard_size": shard_size,
        "evaluation_batch_size": evaluation_batch_size,
        "shard_ordinal": shard_ordinal,
        "asset_bindings": list(asset_bindings),
    }
    return f"shard-{shard_ordinal:06d}-{hashlib.sha256(_canonical_json(material)).hexdigest()[:24]}"


class RasterCandidateObservationV1(_WireModel):
    """One OCR/layout result bound to one immutable raster input asset."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    profile_id: RasterProfileId
    candidate: RasterDocumentExtractionResultV2

    @model_validator(mode="after")
    def _candidate_binds_its_asset(self) -> "RasterCandidateObservationV1":
        if self.candidate.document_asset_id != self.asset_id:
            raise ValueError("raster candidate observation asset binding is invalid")
        return self


class RasterCandidateShardAssetBindingV1(_WireModel):
    """Non-Gold, non-byte identity for one planned raster OCR input."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    profile_id: RasterProfileId
    source_pdf_sha256: str = Field(pattern=_SHA256)
    image_sha256: str = Field(pattern=_SHA256)
    media_type: Literal["image/png", "image/jpeg", "application/pdf"]
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)


class RasterCandidateShardDefinitionV1(_WireModel):
    """One deterministic, independently restartable portion of a split."""

    shard_ordinal: int = Field(ge=0, le=100_000)
    shard_id: str = Field(pattern=_SHARD_ID)
    asset_bindings: list[RasterCandidateShardAssetBindingV1] = Field(
        min_length=1, max_length=_MAX_CANDIDATE_SHARD_SIZE
    )

    @model_validator(mode="after")
    def _asset_pairs_are_unique(self) -> "RasterCandidateShardDefinitionV1":
        pairs = [(item.asset_id, item.profile_id) for item in self.asset_bindings]
        if len(pairs) != len(set(pairs)):
            raise ValueError("raster candidate shard asset/profile pairs must be unique")
        return self


class RasterCandidateShardPlanV1(_WireModel):
    """Immutable ordered work plan for a candidate split.

    The plan intentionally contains input identity metadata only.  It cannot
    carry Gold values, rendered bytes, OCR output, or a mutable progress flag.
    Each shard completion is recorded separately below this plan identity.
    """

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_candidate_shard_plan"] = (
        "synthetic_raster_candidate_shard_plan"
    )
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    engine_kind: Literal["ppstructurev3"] = "ppstructurev3"
    split: _CandidateSplit
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_profile_revision: str = Field(
        default="synthetic-raster-v1", pattern=_RASTER_PROFILE_REVISION
    )
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    shard_size: int = Field(ge=1, le=_MAX_CANDIDATE_SHARD_SIZE)
    evaluation_batch_size: Literal[1, 4]
    shards: list[RasterCandidateShardDefinitionV1] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def _plan_is_ordered_and_deterministic(self) -> "RasterCandidateShardPlanV1":
        if self.shard_size % self.evaluation_batch_size != 0:
            raise ValueError("raster candidate shard size must align with evaluation batch size")
        ordinals = [shard.shard_ordinal for shard in self.shards]
        if ordinals != list(range(len(self.shards))):
            raise ValueError("raster candidate shard ordinals must be contiguous")
        flattened = [binding for shard in self.shards for binding in shard.asset_bindings]
        pairs = [(binding.asset_id, binding.profile_id) for binding in flattened]
        if len(pairs) != len(set(pairs)) or pairs != sorted(pairs):
            raise ValueError("raster candidate shard plan asset order is invalid")
        for index, shard in enumerate(self.shards):
            if len(shard.asset_bindings) > self.shard_size or (
                index < len(self.shards) - 1 and len(shard.asset_bindings) != self.shard_size
            ):
                raise ValueError("raster candidate shard plan partition is invalid")
            expected_shard_id = _deterministic_shard_id(
                split=self.split,
                synthetic_raster_receipt_sha256=self.synthetic_raster_receipt_sha256,
                runtime_configuration_sha256=self.runtime_configuration_sha256,
                raster_profile_revision=self.raster_profile_revision,
                extractor_revision=self.extractor_revision,
                runner_implementation_sha256=self.runner_implementation_sha256,
                shard_size=self.shard_size,
                evaluation_batch_size=self.evaluation_batch_size,
                shard_ordinal=shard.shard_ordinal,
                asset_bindings=[
                    binding.model_dump(mode="json", warnings=False)
                    for binding in shard.asset_bindings
                ],
            )
            if shard.shard_id != expected_shard_id:
                raise ValueError("raster candidate shard ID is not deterministic")
        return self


class RasterCandidateShardV1(_WireModel):
    """Candidate OCR/layout output for exactly one planned shard."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_candidate_shard"] = "synthetic_raster_candidate_shard"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    engine_kind: Literal["ppstructurev3"] = "ppstructurev3"
    split: _CandidateSplit
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_profile_revision: str = Field(
        default="synthetic-raster-v1", pattern=_RASTER_PROFILE_REVISION
    )
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    candidate_shard_plan_sha256: str = Field(pattern=_SHA256)
    shard_ordinal: int = Field(ge=0, le=100_000)
    shard_id: str = Field(pattern=_SHARD_ID)
    candidates: list[RasterCandidateObservationV1] = Field(
        min_length=1, max_length=_MAX_CANDIDATE_SHARD_SIZE
    )

    @model_validator(mode="after")
    def _candidate_keys_are_unique_and_runtime_bound(self) -> "RasterCandidateShardV1":
        keys = [(item.asset_id, item.profile_id) for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("raster candidate shard asset/profile pairs must be unique")
        for observation in self.candidates:
            candidate = observation.candidate
            if candidate.raster_profile_revision != self.raster_profile_revision:
                raise ValueError("raster candidate shard profile binding is invalid")
            if (
                candidate.status != "unavailable"
                and candidate.extractor_revision != self.extractor_revision
            ):
                raise ValueError("raster candidate shard extractor binding is invalid")
            if candidate.status == "unavailable" and (
                candidate.extractor_revision is not None
                or candidate.canonical_text_sha256 is not None
                or candidate.pages
                or candidate.field_candidates
                or candidate.tables
            ):
                raise ValueError("unavailable raster candidate cannot claim OCR provenance")
        return self


class RasterCandidateShardCheckpointV1(_WireModel):
    """Create-only completion binding for one planned candidate shard.

    Its path is derived from the plan identity and shard ID, rather than from
    progress order.  A restart can therefore distinguish an absent completion
    (safe to execute) from a conflicting or malformed completion (fail closed).
    """

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_candidate_shard_checkpoint"] = (
        "synthetic_raster_candidate_shard_checkpoint"
    )
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    engine_kind: Literal["ppstructurev3"] = "ppstructurev3"
    split: _CandidateSplit
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_profile_revision: str = Field(
        default="synthetic-raster-v1", pattern=_RASTER_PROFILE_REVISION
    )
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    candidate_shard_plan_sha256: str = Field(pattern=_SHA256)
    shard_ordinal: int = Field(ge=0, le=100_000)
    shard_id: str = Field(pattern=_SHARD_ID)
    candidate_shard_relative_path: str = Field(min_length=1, max_length=320)
    candidate_shard_sha256: str = Field(pattern=_SHA256)
    candidate_shard_byte_size: int = Field(ge=1, le=_MAX_CANDIDATE_SHARD_BYTES)
    candidate_count: int = Field(ge=1, le=_MAX_CANDIDATE_SHARD_SIZE)


class RasterCandidateBundleV1(_WireModel):
    """Raw local candidate output, with no Gold values or metrics."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_candidate_bundle"] = (
        "synthetic_raster_candidate_bundle"
    )
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    engine_kind: Literal["ppstructurev3"] = "ppstructurev3"
    split: _CandidateSplit
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    # v1 retained its fixed label.  The automatic v2 release uses an opaque
    # fingerprint suffix so a profile replacement cannot share a job/result
    # identity with an earlier runtime.
    raster_profile_revision: str = Field(
        default="synthetic-raster-v1", pattern=_RASTER_PROFILE_REVISION
    )
    extractor_revision: str = Field(min_length=1, max_length=256)
    candidates: list[RasterCandidateObservationV1] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def _candidate_keys_are_unique_and_revision_bound(self) -> "RasterCandidateBundleV1":
        keys = [(item.asset_id, item.profile_id) for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("raster candidate bundle asset/profile pairs must be unique")
        for observation in self.candidates:
            candidate = observation.candidate
            if candidate.raster_profile_revision != self.raster_profile_revision:
                raise ValueError("raster candidate bundle profile binding is invalid")
            if (
                candidate.status != "unavailable"
                and candidate.extractor_revision != self.extractor_revision
            ):
                raise ValueError("raster candidate bundle extractor binding is invalid")
            if candidate.status == "unavailable" and (
                candidate.extractor_revision is not None
                or candidate.canonical_text_sha256 is not None
                or candidate.pages
                or candidate.field_candidates
                or candidate.tables
            ):
                raise ValueError("unavailable raster candidate cannot claim OCR provenance")
        return self


class RasterCandidateExecutionV1(_WireModel):
    """Content-free binding between one bundle and one verified local profile."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_candidate_execution"] = (
        "synthetic_raster_candidate_execution"
    )
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    engine_kind: Literal["ppstructurev3"] = "ppstructurev3"
    split: _CandidateSplit
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_profile_revision: str = Field(
        default="synthetic-raster-v1", pattern=_RASTER_PROFILE_REVISION
    )
    extractor_revision: str = Field(min_length=1, max_length=256)
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_bundle_byte_size: int = Field(ge=1, le=_MAX_CANDIDATE_BUNDLE_BYTES)
    candidate_count: int = Field(ge=1, le=100_000)

    @model_validator(mode="after")
    def _bundle_path_is_content_addressed(self) -> "RasterCandidateExecutionV1":
        expected = candidate_bundle_relative_path(
            split=self.split, bundle_sha256=self.candidate_bundle_sha256
        )
        if self.candidate_bundle_relative_path != expected:
            raise ValueError("raster candidate execution bundle path is not canonical")
        return self


def candidate_bundle_bytes(bundle: RasterCandidateBundleV1) -> bytes:
    return _canonical_json(bundle.model_dump(mode="json", warnings=False))


def candidate_bundle_sha256(bundle: RasterCandidateBundleV1) -> str:
    return hashlib.sha256(candidate_bundle_bytes(bundle)).hexdigest()


def candidate_execution_bytes(execution: RasterCandidateExecutionV1) -> bytes:
    return _canonical_json(execution.model_dump(mode="json", warnings=False))


def candidate_execution_sha256(execution: RasterCandidateExecutionV1) -> str:
    return hashlib.sha256(candidate_execution_bytes(execution)).hexdigest()


def candidate_bundle_relative_path(*, split: _CandidateSplit, bundle_sha256: str) -> str:
    if not isinstance(bundle_sha256, str) or len(bundle_sha256) != 64:
        raise SyntheticRasterCandidateError("raster_candidate_bundle_hash_invalid")
    return f"synthetic-raster-v1/candidate-bundles/{split}/{bundle_sha256}.json"


def candidate_execution_relative_path(*, split: _CandidateSplit, execution_sha256: str) -> str:
    if not isinstance(execution_sha256, str) or len(execution_sha256) != 64:
        raise SyntheticRasterCandidateError("raster_candidate_execution_hash_invalid")
    return f"synthetic-raster-v1/candidate-executions/{split}/{execution_sha256}.json"


def _preview_root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise SyntheticRasterCandidateError("raster_candidate_root_invalid")
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise SyntheticRasterCandidateError("raster_candidate_root_invalid")
        return root
    except SyntheticRasterCandidateError:
        raise
    except OSError:
        raise SyntheticRasterCandidateError("raster_candidate_root_invalid") from None


def _relative_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in value
        or "\\" in value
    ):
        raise SyntheticRasterCandidateError("raster_candidate_path_invalid")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise SyntheticRasterCandidateError("raster_candidate_path_invalid")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SyntheticRasterCandidateError("raster_candidate_path_invalid") from None
    return candidate


def _read_regular(path: Path, *, limit: int, code: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise SyntheticRasterCandidateError(code)
        raw = path.read_bytes()
    except SyntheticRasterCandidateError:
        raise
    except OSError:
        raise SyntheticRasterCandidateError(code) from None
    if not raw or len(raw) > limit:
        raise SyntheticRasterCandidateError(code)
    return raw


def _write_hash_named(root: Path, *, relative_path: str, payload: bytes, code: str) -> Path:
    destination = _relative_path(root, relative_path)
    if destination.is_symlink() or destination.suffix != ".json":
        raise SyntheticRasterCandidateError(code)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Recheck after mkdir so a pre-existing linked parent cannot be used
        # as a write-through path even when it appeared during this call.
        destination = _relative_path(root, relative_path)
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except SyntheticRasterCandidateError:
        raise
    except FileExistsError:
        # Content-addressed paths are safely reusable only if the already
        # present bytes are exactly the canonical payload this invocation
        # would have written.  A different byte stream is an immutable
        # conflict, not a best-effort overwrite.
        if _read_regular(destination, limit=len(payload), code=code) != payload:
            raise SyntheticRasterCandidateError(code) from None
        return destination
    except OSError:
        raise SyntheticRasterCandidateError(code) from None
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
    return destination


def write_canonical_raster_candidate_bundle(
    *, document_preview_root: Path, bundle: RasterCandidateBundleV1
) -> tuple[str, str, int]:
    """Write a create-only bundle to its sole hash-derived local path."""

    root = _preview_root(document_preview_root)
    payload = candidate_bundle_bytes(bundle)
    if len(payload) > _MAX_CANDIDATE_BUNDLE_BYTES:
        raise SyntheticRasterCandidateError("raster_candidate_bundle_too_large")
    digest = hashlib.sha256(payload).hexdigest()
    relative = candidate_bundle_relative_path(split=bundle.split, bundle_sha256=digest)
    _write_hash_named(
        root,
        relative_path=relative,
        payload=payload,
        code="raster_candidate_bundle_immutable_conflict",
    )
    return relative, digest, len(payload)


def read_canonical_raster_candidate_bundle(
    *,
    document_preview_root: Path,
    relative_path: str,
    expected_sha256: str | None = None,
) -> tuple[RasterCandidateBundleV1, bytes, str]:
    """Read one raw candidate only when its bytes and path are canonical."""

    root = _preview_root(document_preview_root)
    raw = _read_regular(
        _relative_path(root, relative_path),
        limit=_MAX_CANDIDATE_BUNDLE_BYTES,
        code="raster_candidate_bundle_invalid",
    )
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SyntheticRasterCandidateError("raster_candidate_bundle_hash_mismatch")
    try:
        bundle = RasterCandidateBundleV1.model_validate_json(raw)
    except (ValidationError, UnicodeDecodeError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_bundle_invalid") from None
    if raw != candidate_bundle_bytes(bundle):
        raise SyntheticRasterCandidateError("raster_candidate_bundle_not_canonical")
    if relative_path != candidate_bundle_relative_path(split=bundle.split, bundle_sha256=digest):
        raise SyntheticRasterCandidateError("raster_candidate_bundle_path_invalid")
    return bundle, raw, digest


def build_raster_candidate_execution(
    *,
    bundle: RasterCandidateBundleV1,
    candidate_bundle_relative_path_value: str,
    candidate_bundle_raw: bytes,
    runtime_configuration_sha256: str,
) -> RasterCandidateExecutionV1:
    """Bind a canonical candidate result to the verified runtime config hash."""

    if not runtime_configuration_sha256 or len(runtime_configuration_sha256) != 64:
        raise SyntheticRasterCandidateError("raster_candidate_execution_configuration_invalid")
    canonical = candidate_bundle_bytes(bundle)
    if candidate_bundle_raw != canonical:
        raise SyntheticRasterCandidateError("raster_candidate_execution_bundle_invalid")
    bundle_digest = hashlib.sha256(canonical).hexdigest()
    expected_path = candidate_bundle_relative_path(split=bundle.split, bundle_sha256=bundle_digest)
    if candidate_bundle_relative_path_value != expected_path:
        raise SyntheticRasterCandidateError("raster_candidate_execution_bundle_invalid")
    return RasterCandidateExecutionV1(
        engine_kind="ppstructurev3",
        split=bundle.split,
        synthetic_raster_receipt_sha256=bundle.synthetic_raster_receipt_sha256,
        runtime_configuration_sha256=runtime_configuration_sha256,
        raster_profile_revision=bundle.raster_profile_revision,
        extractor_revision=bundle.extractor_revision,
        candidate_bundle_relative_path=candidate_bundle_relative_path_value,
        candidate_bundle_sha256=bundle_digest,
        candidate_bundle_byte_size=len(canonical),
        candidate_count=len(bundle.candidates),
    )


def write_canonical_raster_candidate_execution(
    *, document_preview_root: Path, execution: RasterCandidateExecutionV1
) -> tuple[str, str]:
    root = _preview_root(document_preview_root)
    payload = candidate_execution_bytes(execution)
    if len(payload) > _MAX_EXECUTION_RECEIPT_BYTES:
        raise SyntheticRasterCandidateError("raster_candidate_execution_too_large")
    digest = hashlib.sha256(payload).hexdigest()
    relative = candidate_execution_relative_path(split=execution.split, execution_sha256=digest)
    _write_hash_named(
        root,
        relative_path=relative,
        payload=payload,
        code="raster_candidate_execution_immutable_conflict",
    )
    return relative, digest


def read_canonical_raster_candidate_execution(
    *,
    document_preview_root: Path,
    relative_path: str,
    expected_sha256: str | None = None,
) -> tuple[RasterCandidateExecutionV1, bytes, str]:
    root = _preview_root(document_preview_root)
    raw = _read_regular(
        _relative_path(root, relative_path),
        limit=_MAX_EXECUTION_RECEIPT_BYTES,
        code="raster_candidate_execution_invalid",
    )
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SyntheticRasterCandidateError("raster_candidate_execution_hash_mismatch")
    try:
        execution = RasterCandidateExecutionV1.model_validate_json(raw)
    except (ValidationError, UnicodeDecodeError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_execution_invalid") from None
    if raw != candidate_execution_bytes(execution):
        raise SyntheticRasterCandidateError("raster_candidate_execution_not_canonical")
    if relative_path != candidate_execution_relative_path(
        split=execution.split, execution_sha256=digest
    ):
        raise SyntheticRasterCandidateError("raster_candidate_execution_path_invalid")
    return execution, raw, digest


def validate_raster_candidate_bundle_against_receipt(
    *,
    bundle: RasterCandidateBundleV1,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    split: _CandidateSplit,
) -> Mapping[tuple[str, RasterProfileId], RasterAssetReceiptV1]:
    """Require exact split coverage and immutable raster-input bindings.

    A wrong classification or OCR text is intentionally retained and scored;
    only a wrong asset/profile/runtime provenance fails this validation.
    """

    if (
        bundle.split != split
        or bundle.synthetic_raster_receipt_sha256 != receipt_sha256
        or bundle.dataset_id != receipt.dataset_id
        or re.fullmatch(_RASTER_PROFILE_REVISION, bundle.raster_profile_revision) is None
    ):
        raise SyntheticRasterCandidateError("raster_candidate_bundle_receipt_binding_invalid")
    expected = {
        (asset.asset_id, asset.profile_id): asset
        for asset in receipt.assets
        if asset.split == split
    }
    observed = {(item.asset_id, item.profile_id): item for item in bundle.candidates}
    if len(observed) != len(bundle.candidates) or set(observed) != set(expected):
        raise SyntheticRasterCandidateError("raster_candidate_bundle_coverage_invalid")
    for key, observation in observed.items():
        asset = expected[key]
        candidate = observation.candidate
        if (
            candidate.source_sha256 != asset.image_sha256
            or candidate.media_type != asset.media_type
            or candidate.byte_size != asset.byte_size
            or candidate.raster_profile_revision != bundle.raster_profile_revision
        ):
            raise SyntheticRasterCandidateError("raster_candidate_bundle_asset_binding_invalid")
    return expected


def validate_raster_candidate_execution(
    *,
    execution: RasterCandidateExecutionV1,
    bundle: RasterCandidateBundleV1,
    bundle_raw: bytes,
    bundle_sha256: str,
    runtime_configuration_sha256: str,
) -> None:
    """Replay the small execution receipt without rerunning OCR."""

    if (
        bundle_raw != candidate_bundle_bytes(bundle)
        or hashlib.sha256(bundle_raw).hexdigest() != bundle_sha256
    ):
        raise SyntheticRasterCandidateError("raster_candidate_execution_bundle_invalid")
    expected = build_raster_candidate_execution(
        bundle=bundle,
        candidate_bundle_relative_path_value=candidate_bundle_relative_path(
            split=bundle.split, bundle_sha256=bundle_sha256
        ),
        candidate_bundle_raw=bundle_raw,
        runtime_configuration_sha256=runtime_configuration_sha256,
    )
    if execution != expected:
        raise SyntheticRasterCandidateError("raster_candidate_execution_binding_invalid")


def _asset_binding_from_receipt(
    asset: RasterAssetReceiptV1,
) -> RasterCandidateShardAssetBindingV1:
    """Copy only safe input identity metadata from a raster receipt row."""

    return RasterCandidateShardAssetBindingV1(
        asset_id=asset.asset_id,
        profile_id=asset.profile_id,
        source_pdf_sha256=asset.source_pdf_sha256,
        image_sha256=asset.image_sha256,
        media_type=asset.media_type,
        byte_size=asset.byte_size,
        width=asset.width,
        height=asset.height,
    )


def _validated_shard_plan(plan: RasterCandidateShardPlanV1) -> RasterCandidateShardPlanV1:
    try:
        return RasterCandidateShardPlanV1.model_validate(
            plan.model_dump(mode="json", warnings=False)
        )
    except (AttributeError, TypeError, ValidationError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_invalid") from None


def _validated_shard(shard: RasterCandidateShardV1) -> RasterCandidateShardV1:
    try:
        return RasterCandidateShardV1.model_validate(shard.model_dump(mode="json", warnings=False))
    except (AttributeError, TypeError, ValidationError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_invalid") from None


def _validated_shard_checkpoint(
    checkpoint: RasterCandidateShardCheckpointV1,
) -> RasterCandidateShardCheckpointV1:
    try:
        return RasterCandidateShardCheckpointV1.model_validate(
            checkpoint.model_dump(mode="json", warnings=False)
        )
    except (AttributeError, TypeError, ValidationError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_invalid") from None


def candidate_shard_plan_bytes(plan: RasterCandidateShardPlanV1) -> bytes:
    return _canonical_json(plan.model_dump(mode="json", warnings=False))


def candidate_shard_plan_sha256(plan: RasterCandidateShardPlanV1) -> str:
    return hashlib.sha256(candidate_shard_plan_bytes(plan)).hexdigest()


def candidate_shard_bytes(shard: RasterCandidateShardV1) -> bytes:
    return _canonical_json(shard.model_dump(mode="json", warnings=False))


def candidate_shard_sha256(shard: RasterCandidateShardV1) -> str:
    return hashlib.sha256(candidate_shard_bytes(shard)).hexdigest()


def candidate_shard_checkpoint_bytes(checkpoint: RasterCandidateShardCheckpointV1) -> bytes:
    return _canonical_json(checkpoint.model_dump(mode="json", warnings=False))


def candidate_shard_checkpoint_sha256(checkpoint: RasterCandidateShardCheckpointV1) -> str:
    return hashlib.sha256(candidate_shard_checkpoint_bytes(checkpoint)).hexdigest()


def candidate_shard_plan_relative_path(*, split: _CandidateSplit, plan_sha256: str) -> str:
    _require_candidate_split(split)
    _require_sha256(plan_sha256, code="raster_candidate_shard_plan_hash_invalid")
    return f"synthetic-raster-v1/candidate-shard-plans/{split}/{plan_sha256}.json"


def candidate_shard_relative_path(
    *, split: _CandidateSplit, plan_sha256: str, shard_id: str, shard_sha256: str
) -> str:
    _require_candidate_split(split)
    _require_sha256(plan_sha256, code="raster_candidate_shard_plan_hash_invalid")
    _require_sha256(shard_sha256, code="raster_candidate_shard_hash_invalid")
    if not isinstance(shard_id, str) or re.fullmatch(_SHARD_ID, shard_id) is None:
        raise SyntheticRasterCandidateError("raster_candidate_shard_id_invalid")
    # Keep a Windows-safe path depth/length.  The complete plan SHA and shard
    # ID remain in the canonical payload and are revalidated on every read;
    # a prefix collision therefore becomes an immutable conflict, never a
    # cross-run reuse.  The ordinal directory is enough to prevent one plan's
    # independent shards from sharing a location.
    shard_ordinal = shard_id[6:12]
    return (
        f"synthetic-raster-v1/candidate-shards/{split[0]}/{plan_sha256[:24]}/"
        f"{shard_ordinal}/{shard_sha256}.json"
    )


def candidate_shard_checkpoint_relative_path(
    *, split: _CandidateSplit, plan_sha256: str, shard_id: str
) -> str:
    """Return the one create-only resume marker path for a planned shard."""

    _require_candidate_split(split)
    _require_sha256(plan_sha256, code="raster_candidate_shard_plan_hash_invalid")
    if not isinstance(shard_id, str) or re.fullmatch(_SHARD_ID, shard_id) is None:
        raise SyntheticRasterCandidateError("raster_candidate_shard_id_invalid")
    return (
        f"synthetic-raster-v1/candidate-shard-checkpoints/{split[0]}/{plan_sha256[:24]}/"
        f"{shard_id[6:12]}.json"
    )


def iter_raster_candidate_shard_definitions(
    plan: RasterCandidateShardPlanV1,
) -> tuple[RasterCandidateShardDefinitionV1, ...]:
    """Expose the deterministic shard order without exposing OCR or Gold data."""

    return tuple(_validated_shard_plan(plan).shards)


def raster_candidate_shard_definition(
    *, plan: RasterCandidateShardPlanV1, shard_ordinal: int
) -> RasterCandidateShardDefinitionV1:
    verified = _validated_shard_plan(plan)
    if not isinstance(shard_ordinal, int) or isinstance(shard_ordinal, bool):
        raise SyntheticRasterCandidateError("raster_candidate_shard_definition_unavailable")
    try:
        definition = verified.shards[shard_ordinal]
    except IndexError:
        raise SyntheticRasterCandidateError(
            "raster_candidate_shard_definition_unavailable"
        ) from None
    if definition.shard_ordinal != shard_ordinal:
        raise SyntheticRasterCandidateError("raster_candidate_shard_definition_unavailable")
    return definition


def iter_raster_candidate_shard_asset_bindings(
    *, plan: RasterCandidateShardPlanV1, shard_ordinal: int | None = None
) -> tuple[RasterCandidateShardAssetBindingV1, ...]:
    """Return planned input bindings in canonical evaluation order."""

    definitions = (
        (raster_candidate_shard_definition(plan=plan, shard_ordinal=shard_ordinal),)
        if shard_ordinal is not None
        else iter_raster_candidate_shard_definitions(plan)
    )
    return tuple(binding for definition in definitions for binding in definition.asset_bindings)


def validate_raster_candidate_shard_plan_against_receipt(
    *,
    plan: RasterCandidateShardPlanV1,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    split: _CandidateSplit,
) -> tuple[RasterCandidateShardAssetBindingV1, ...]:
    """Require exact split coverage before any shard can be accepted."""

    verified = _validated_shard_plan(plan)
    _require_sha256(receipt_sha256, code="raster_candidate_shard_plan_receipt_invalid")
    if receipt_sha256 != synthetic_raster_receipt_sha256(receipt):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_receipt_invalid")
    if (
        verified.split != split
        or verified.synthetic_raster_receipt_sha256 != receipt_sha256
        or verified.dataset_id != receipt.dataset_id
    ):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_receipt_binding_invalid")
    expected = tuple(
        _asset_binding_from_receipt(asset)
        for asset in sorted(
            (asset for asset in receipt.assets if asset.split == split),
            key=lambda asset: (asset.asset_id, asset.profile_id),
        )
    )
    observed = iter_raster_candidate_shard_asset_bindings(plan=verified)
    if not expected or observed != expected:
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_coverage_invalid")
    return expected


def build_raster_candidate_shard_plan(
    *,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    split: _CandidateSplit,
    runtime_configuration_sha256: str,
    raster_profile_revision: str,
    extractor_revision: str,
    runner_implementation_sha256: str,
    shard_size: int,
    evaluation_batch_size: Literal[1, 4],
) -> RasterCandidateShardPlanV1:
    """Create the sole deterministic shard partition for one run identity."""

    _require_sha256(receipt_sha256, code="raster_candidate_shard_plan_receipt_invalid")
    _require_sha256(
        runtime_configuration_sha256, code="raster_candidate_shard_plan_runtime_invalid"
    )
    _require_sha256(runner_implementation_sha256, code="raster_candidate_shard_plan_runner_invalid")
    if receipt_sha256 != synthetic_raster_receipt_sha256(receipt):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_receipt_invalid")
    bindings = tuple(
        _asset_binding_from_receipt(asset)
        for asset in sorted(
            (asset for asset in receipt.assets if asset.split == split),
            key=lambda asset: (asset.asset_id, asset.profile_id),
        )
    )
    if not bindings:
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_split_empty")
    try:
        shards = [
            RasterCandidateShardDefinitionV1(
                shard_ordinal=ordinal,
                shard_id=_deterministic_shard_id(
                    split=split,
                    synthetic_raster_receipt_sha256=receipt_sha256,
                    runtime_configuration_sha256=runtime_configuration_sha256,
                    raster_profile_revision=raster_profile_revision,
                    extractor_revision=extractor_revision,
                    runner_implementation_sha256=runner_implementation_sha256,
                    shard_size=shard_size,
                    evaluation_batch_size=evaluation_batch_size,
                    shard_ordinal=ordinal,
                    asset_bindings=[
                        binding.model_dump(mode="json", warnings=False)
                        for binding in bindings[offset : offset + shard_size]
                    ],
                ),
                asset_bindings=list(bindings[offset : offset + shard_size]),
            )
            for ordinal, offset in enumerate(range(0, len(bindings), shard_size))
        ]
        plan = RasterCandidateShardPlanV1(
            split=split,
            synthetic_raster_receipt_sha256=receipt_sha256,
            runtime_configuration_sha256=runtime_configuration_sha256,
            raster_profile_revision=raster_profile_revision,
            extractor_revision=extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            shard_size=shard_size,
            evaluation_batch_size=evaluation_batch_size,
            shards=shards,
        )
    except (TypeError, ValidationError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_invalid") from None
    validate_raster_candidate_shard_plan_against_receipt(
        plan=plan, receipt=receipt, receipt_sha256=receipt_sha256, split=split
    )
    return plan


def write_canonical_raster_candidate_shard_plan(
    *, document_preview_root: Path, plan: RasterCandidateShardPlanV1
) -> tuple[str, str, int]:
    root = _preview_root(document_preview_root)
    verified = _validated_shard_plan(plan)
    payload = candidate_shard_plan_bytes(verified)
    if len(payload) > _MAX_CANDIDATE_SHARD_PLAN_BYTES:
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_too_large")
    digest = hashlib.sha256(payload).hexdigest()
    relative = candidate_shard_plan_relative_path(split=verified.split, plan_sha256=digest)
    _write_hash_named(
        root,
        relative_path=relative,
        payload=payload,
        code="raster_candidate_shard_plan_immutable_conflict",
    )
    return relative, digest, len(payload)


def read_canonical_raster_candidate_shard_plan(
    *,
    document_preview_root: Path,
    relative_path: str,
    expected_sha256: str | None = None,
) -> tuple[RasterCandidateShardPlanV1, bytes, str]:
    root = _preview_root(document_preview_root)
    raw = _read_regular(
        _relative_path(root, relative_path),
        limit=_MAX_CANDIDATE_SHARD_PLAN_BYTES,
        code="raster_candidate_shard_plan_invalid",
    )
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_hash_mismatch")
    try:
        plan = RasterCandidateShardPlanV1.model_validate_json(raw)
    except (ValidationError, UnicodeDecodeError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_invalid") from None
    if raw != candidate_shard_plan_bytes(plan):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_not_canonical")
    if relative_path != candidate_shard_plan_relative_path(split=plan.split, plan_sha256=digest):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_path_invalid")
    return plan, raw, digest


def build_raster_candidate_shard(
    *,
    plan: RasterCandidateShardPlanV1,
    shard_ordinal: int,
    candidates: Sequence[RasterCandidateObservationV1],
) -> RasterCandidateShardV1:
    """Bind a completed shard's candidate observations to its immutable plan."""

    verified_plan = _validated_shard_plan(plan)
    definition = raster_candidate_shard_definition(plan=verified_plan, shard_ordinal=shard_ordinal)
    try:
        shard = RasterCandidateShardV1(
            split=verified_plan.split,
            synthetic_raster_receipt_sha256=verified_plan.synthetic_raster_receipt_sha256,
            runtime_configuration_sha256=verified_plan.runtime_configuration_sha256,
            raster_profile_revision=verified_plan.raster_profile_revision,
            extractor_revision=verified_plan.extractor_revision,
            runner_implementation_sha256=verified_plan.runner_implementation_sha256,
            candidate_shard_plan_sha256=candidate_shard_plan_sha256(verified_plan),
            shard_ordinal=definition.shard_ordinal,
            shard_id=definition.shard_id,
            candidates=list(candidates),
        )
    except (TypeError, ValidationError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_invalid") from None
    validate_raster_candidate_shard_against_plan(plan=verified_plan, shard=shard)
    return shard


def validate_raster_candidate_shard_against_plan(
    *, plan: RasterCandidateShardPlanV1, shard: RasterCandidateShardV1
) -> RasterCandidateShardDefinitionV1:
    """Reject swapped, stale, partial, or differently ordered shard output."""

    verified_plan = _validated_shard_plan(plan)
    verified_shard = _validated_shard(shard)
    definition = raster_candidate_shard_definition(
        plan=verified_plan, shard_ordinal=verified_shard.shard_ordinal
    )
    if (
        verified_shard.split != verified_plan.split
        or verified_shard.synthetic_raster_receipt_sha256
        != verified_plan.synthetic_raster_receipt_sha256
        or verified_shard.runtime_configuration_sha256 != verified_plan.runtime_configuration_sha256
        or verified_shard.raster_profile_revision != verified_plan.raster_profile_revision
        or verified_shard.extractor_revision != verified_plan.extractor_revision
        or verified_shard.runner_implementation_sha256 != verified_plan.runner_implementation_sha256
        or verified_shard.candidate_shard_plan_sha256 != candidate_shard_plan_sha256(verified_plan)
        or verified_shard.shard_id != definition.shard_id
    ):
        raise SyntheticRasterCandidateError("raster_candidate_shard_plan_binding_invalid")
    expected = tuple(definition.asset_bindings)
    observed = tuple((item.asset_id, item.profile_id) for item in verified_shard.candidates)
    if observed != tuple((item.asset_id, item.profile_id) for item in expected):
        raise SyntheticRasterCandidateError("raster_candidate_shard_coverage_invalid")
    for observation, binding in zip(verified_shard.candidates, expected, strict=True):
        candidate = observation.candidate
        if (
            candidate.source_sha256 != binding.image_sha256
            or candidate.media_type != binding.media_type
            or candidate.byte_size != binding.byte_size
        ):
            raise SyntheticRasterCandidateError("raster_candidate_shard_asset_binding_invalid")
    return definition


def write_canonical_raster_candidate_shard(
    *, document_preview_root: Path, shard: RasterCandidateShardV1
) -> tuple[str, str, int]:
    root = _preview_root(document_preview_root)
    verified = _validated_shard(shard)
    payload = candidate_shard_bytes(verified)
    if len(payload) > _MAX_CANDIDATE_SHARD_BYTES:
        raise SyntheticRasterCandidateError("raster_candidate_shard_too_large")
    digest = hashlib.sha256(payload).hexdigest()
    relative = candidate_shard_relative_path(
        split=verified.split,
        plan_sha256=verified.candidate_shard_plan_sha256,
        shard_id=verified.shard_id,
        shard_sha256=digest,
    )
    _write_hash_named(
        root,
        relative_path=relative,
        payload=payload,
        code="raster_candidate_shard_immutable_conflict",
    )
    return relative, digest, len(payload)


def read_canonical_raster_candidate_shard(
    *,
    document_preview_root: Path,
    relative_path: str,
    expected_sha256: str | None = None,
) -> tuple[RasterCandidateShardV1, bytes, str]:
    root = _preview_root(document_preview_root)
    raw = _read_regular(
        _relative_path(root, relative_path),
        limit=_MAX_CANDIDATE_SHARD_BYTES,
        code="raster_candidate_shard_invalid",
    )
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SyntheticRasterCandidateError("raster_candidate_shard_hash_mismatch")
    try:
        shard = RasterCandidateShardV1.model_validate_json(raw)
    except (ValidationError, UnicodeDecodeError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_invalid") from None
    if raw != candidate_shard_bytes(shard):
        raise SyntheticRasterCandidateError("raster_candidate_shard_not_canonical")
    if relative_path != candidate_shard_relative_path(
        split=shard.split,
        plan_sha256=shard.candidate_shard_plan_sha256,
        shard_id=shard.shard_id,
        shard_sha256=digest,
    ):
        raise SyntheticRasterCandidateError("raster_candidate_shard_path_invalid")
    return shard, raw, digest


def build_raster_candidate_shard_checkpoint(
    *,
    plan: RasterCandidateShardPlanV1,
    shard: RasterCandidateShardV1,
    candidate_shard_relative_path_value: str,
    candidate_shard_raw: bytes,
) -> RasterCandidateShardCheckpointV1:
    """Create the exact one-shard resume attestation after a durable shard write."""

    verified_plan = _validated_shard_plan(plan)
    verified_shard = _validated_shard(shard)
    definition = validate_raster_candidate_shard_against_plan(
        plan=verified_plan, shard=verified_shard
    )
    canonical = candidate_shard_bytes(verified_shard)
    if candidate_shard_raw != canonical:
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_shard_invalid")
    digest = hashlib.sha256(canonical).hexdigest()
    expected_path = candidate_shard_relative_path(
        split=verified_shard.split,
        plan_sha256=verified_shard.candidate_shard_plan_sha256,
        shard_id=verified_shard.shard_id,
        shard_sha256=digest,
    )
    if candidate_shard_relative_path_value != expected_path:
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_shard_invalid")
    return RasterCandidateShardCheckpointV1(
        split=verified_plan.split,
        synthetic_raster_receipt_sha256=verified_plan.synthetic_raster_receipt_sha256,
        runtime_configuration_sha256=verified_plan.runtime_configuration_sha256,
        raster_profile_revision=verified_plan.raster_profile_revision,
        extractor_revision=verified_plan.extractor_revision,
        runner_implementation_sha256=verified_plan.runner_implementation_sha256,
        candidate_shard_plan_sha256=candidate_shard_plan_sha256(verified_plan),
        shard_ordinal=definition.shard_ordinal,
        shard_id=definition.shard_id,
        candidate_shard_relative_path=expected_path,
        candidate_shard_sha256=digest,
        candidate_shard_byte_size=len(canonical),
        candidate_count=len(verified_shard.candidates),
    )


def validate_raster_candidate_shard_checkpoint_against_plan(
    *, plan: RasterCandidateShardPlanV1, checkpoint: RasterCandidateShardCheckpointV1
) -> RasterCandidateShardDefinitionV1:
    """Verify the completion marker's identity before following its shard path."""

    verified_plan = _validated_shard_plan(plan)
    verified_checkpoint = _validated_shard_checkpoint(checkpoint)
    definition = raster_candidate_shard_definition(
        plan=verified_plan, shard_ordinal=verified_checkpoint.shard_ordinal
    )
    if (
        verified_checkpoint.split != verified_plan.split
        or verified_checkpoint.synthetic_raster_receipt_sha256
        != verified_plan.synthetic_raster_receipt_sha256
        or verified_checkpoint.runtime_configuration_sha256
        != verified_plan.runtime_configuration_sha256
        or verified_checkpoint.raster_profile_revision != verified_plan.raster_profile_revision
        or verified_checkpoint.extractor_revision != verified_plan.extractor_revision
        or verified_checkpoint.runner_implementation_sha256
        != verified_plan.runner_implementation_sha256
        or verified_checkpoint.candidate_shard_plan_sha256
        != candidate_shard_plan_sha256(verified_plan)
        or verified_checkpoint.shard_id != definition.shard_id
        or verified_checkpoint.candidate_count != len(definition.asset_bindings)
    ):
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_binding_invalid")
    expected_path = candidate_shard_relative_path(
        split=verified_checkpoint.split,
        plan_sha256=verified_checkpoint.candidate_shard_plan_sha256,
        shard_id=verified_checkpoint.shard_id,
        shard_sha256=verified_checkpoint.candidate_shard_sha256,
    )
    if verified_checkpoint.candidate_shard_relative_path != expected_path:
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_path_invalid")
    return definition


def write_canonical_raster_candidate_shard_checkpoint(
    *, document_preview_root: Path, checkpoint: RasterCandidateShardCheckpointV1
) -> tuple[str, str]:
    """Write one fixed resume marker; conflicting bytes cannot replace it."""

    root = _preview_root(document_preview_root)
    verified = _validated_shard_checkpoint(checkpoint)
    payload = candidate_shard_checkpoint_bytes(verified)
    if len(payload) > _MAX_CANDIDATE_SHARD_CHECKPOINT_BYTES:
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_too_large")
    relative = candidate_shard_checkpoint_relative_path(
        split=verified.split,
        plan_sha256=verified.candidate_shard_plan_sha256,
        shard_id=verified.shard_id,
    )
    _write_hash_named(
        root,
        relative_path=relative,
        payload=payload,
        code="raster_candidate_shard_checkpoint_immutable_conflict",
    )
    return relative, hashlib.sha256(payload).hexdigest()


def read_canonical_raster_candidate_shard_checkpoint(
    *,
    document_preview_root: Path,
    relative_path: str,
    expected_sha256: str | None = None,
) -> tuple[RasterCandidateShardCheckpointV1, bytes, str]:
    root = _preview_root(document_preview_root)
    raw = _read_regular(
        _relative_path(root, relative_path),
        limit=_MAX_CANDIDATE_SHARD_CHECKPOINT_BYTES,
        code="raster_candidate_shard_checkpoint_invalid",
    )
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_hash_mismatch")
    try:
        checkpoint = RasterCandidateShardCheckpointV1.model_validate_json(raw)
    except (ValidationError, UnicodeDecodeError, ValueError):
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_invalid") from None
    if raw != candidate_shard_checkpoint_bytes(checkpoint):
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_not_canonical")
    if relative_path != candidate_shard_checkpoint_relative_path(
        split=checkpoint.split,
        plan_sha256=checkpoint.candidate_shard_plan_sha256,
        shard_id=checkpoint.shard_id,
    ):
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_path_invalid")
    return checkpoint, raw, digest


def read_verified_raster_candidate_shard_checkpoint(
    *,
    document_preview_root: Path,
    plan: RasterCandidateShardPlanV1,
    shard_ordinal: int,
) -> tuple[RasterCandidateShardCheckpointV1, RasterCandidateShardV1]:
    """Read a completed shard, or report only a genuinely absent checkpoint.

    ``raster_candidate_shard_checkpoint_unavailable`` means the fixed marker
    does not exist and the runner may execute that shard.  Every other error
    denotes malformed, stale, swapped, or conflicting durable evidence and
    must stop the run instead of being treated as a cache miss.
    """

    root = _preview_root(document_preview_root)
    verified_plan = _validated_shard_plan(plan)
    definition = raster_candidate_shard_definition(plan=verified_plan, shard_ordinal=shard_ordinal)
    plan_sha256 = candidate_shard_plan_sha256(verified_plan)
    relative = candidate_shard_checkpoint_relative_path(
        split=verified_plan.split, plan_sha256=plan_sha256, shard_id=definition.shard_id
    )
    checkpoint_path = _relative_path(root, relative)
    if not checkpoint_path.exists():
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_unavailable")
    checkpoint, _checkpoint_raw, _checkpoint_sha256 = (
        read_canonical_raster_candidate_shard_checkpoint(
            document_preview_root=root, relative_path=relative
        )
    )
    validate_raster_candidate_shard_checkpoint_against_plan(
        plan=verified_plan, checkpoint=checkpoint
    )
    shard, shard_raw, shard_sha256 = read_canonical_raster_candidate_shard(
        document_preview_root=root,
        relative_path=checkpoint.candidate_shard_relative_path,
        expected_sha256=checkpoint.candidate_shard_sha256,
    )
    if (
        len(shard_raw) != checkpoint.candidate_shard_byte_size
        or shard_sha256 != checkpoint.candidate_shard_sha256
    ):
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_shard_invalid")
    validate_raster_candidate_shard_against_plan(plan=verified_plan, shard=shard)
    if len(shard.candidates) != checkpoint.candidate_count:
        raise SyntheticRasterCandidateError("raster_candidate_shard_checkpoint_shard_invalid")
    return checkpoint, shard


__all__ = [
    "RasterCandidateBundleV1",
    "RasterCandidateExecutionV1",
    "RasterCandidateObservationV1",
    "RasterCandidateShardAssetBindingV1",
    "RasterCandidateShardCheckpointV1",
    "RasterCandidateShardDefinitionV1",
    "RasterCandidateShardPlanV1",
    "RasterCandidateShardV1",
    "SyntheticRasterCandidateError",
    "build_raster_candidate_execution",
    "build_raster_candidate_shard",
    "build_raster_candidate_shard_checkpoint",
    "build_raster_candidate_shard_plan",
    "candidate_bundle_bytes",
    "candidate_bundle_relative_path",
    "candidate_bundle_sha256",
    "candidate_execution_bytes",
    "candidate_execution_relative_path",
    "candidate_execution_sha256",
    "candidate_shard_bytes",
    "candidate_shard_checkpoint_bytes",
    "candidate_shard_checkpoint_relative_path",
    "candidate_shard_checkpoint_sha256",
    "candidate_shard_plan_bytes",
    "candidate_shard_plan_relative_path",
    "candidate_shard_plan_sha256",
    "candidate_shard_relative_path",
    "candidate_shard_sha256",
    "read_canonical_raster_candidate_bundle",
    "read_canonical_raster_candidate_execution",
    "read_canonical_raster_candidate_shard",
    "read_canonical_raster_candidate_shard_checkpoint",
    "read_canonical_raster_candidate_shard_plan",
    "read_verified_raster_candidate_shard_checkpoint",
    "iter_raster_candidate_shard_asset_bindings",
    "iter_raster_candidate_shard_definitions",
    "raster_candidate_shard_definition",
    "validate_raster_candidate_bundle_against_receipt",
    "validate_raster_candidate_execution",
    "validate_raster_candidate_shard_against_plan",
    "validate_raster_candidate_shard_checkpoint_against_plan",
    "validate_raster_candidate_shard_plan_against_receipt",
    "write_canonical_raster_candidate_bundle",
    "write_canonical_raster_candidate_execution",
    "write_canonical_raster_candidate_shard",
    "write_canonical_raster_candidate_shard_checkpoint",
    "write_canonical_raster_candidate_shard_plan",
]
