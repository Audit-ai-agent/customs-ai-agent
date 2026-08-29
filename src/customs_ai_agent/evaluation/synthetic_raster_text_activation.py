"""Local-only source-attested activation for the text-only raster OCR lane.

This module deliberately has no dependency on the PP-Structure pointer or
table/geometry Gate.  It creates compact source-Gold receipts, runs the exact
manifest-pinned direct OCR candidate, fixes a confidence floor from
calibration, measures sealed data, and calls the text pointer writer only at
the very end.  It accepts neither an operator score nor an OCR/Gold payload.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
from statistics import NormalDist
import subprocess
import sys
import threading
import time
from typing import BinaryIO, Callable, cast, Final, Iterator, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.evaluation.raster_offline_bundle import (
    RasterOfflineBundleError,
    current_raster_runtime_platform_binding,
    load_verified_raster_offline_bundle,
    load_verified_raster_offline_bundle_release,
    write_or_verify_canonical_local_json,
)
from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    RasterTextDirectPaddleBundleError,
    load_verified_raster_text_direct_paddle_bundle_v1,
    materialize_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle import (
    RasterTextMobilePaddleBundleError,
    load_verified_raster_text_direct_mobile_paddle_bundle_v1,
    materialize_verified_raster_text_direct_mobile_paddle_bundle_v1,
    prepare_raster_text_direct_mobile_runtime_profile_v15,
)
from customs_ai_agent.evaluation.raster_text_public_finetuned_paddle_bundle import (
    RasterTextPublicFineTunedPaddleBundleError,
    VerifiedRasterTextPublicFineTunedPaddleBundleV1,
    load_verified_raster_text_public_finetuned_paddle_bundle_v1,
    materialize_verified_raster_text_public_finetuned_paddle_bundle_v1,
    prepare_raster_text_public_finetuned_one_pass_runtime_profile_v19,
    prepare_raster_text_public_finetuned_runtime_profile_v19,
)
from customs_ai_agent.evaluation.raster_text_public_voyage_replay_paddle_bundle import (
    RasterTextPublicVoyageReplayPaddleBundleError,
    VerifiedRasterTextPublicVoyageReplayPaddleBundleV1,
    load_verified_raster_text_public_voyage_replay_paddle_bundle_v1,
    materialize_verified_raster_text_public_voyage_replay_paddle_bundle_v1,
    prepare_raster_text_public_voyage_replay_runtime_profile_v20,
)
from customs_ai_agent.evaluation.raster_text_server_paddle_bundle import (
    RasterTextServerPaddleBundleError,
    load_verified_raster_text_direct_server_paddle_bundle_v1,
    materialize_verified_raster_text_direct_server_paddle_bundle_v1,
    prepare_raster_text_direct_server_runtime_profile_v18,
)
from customs_ai_agent.evaluation.source_attested_bl_repair_audit import (
    BL_REPAIR_AUDIT_DATASET_ID,
    BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH,
    BlRepairAuditCandidateBundleV1,
    BlRepairAuditRasterBindingV1,
    BlRepairAuditReceiptV1,
    bl_repair_audit_evaluation_sha256,
    bl_repair_audit_receipt_sha256,
    compile_bl_repair_audit_dataset,
    evaluate_bl_repair_audit_candidates,
    verify_bl_repair_audit_candidate_bundle,
    verify_bl_repair_audit_dataset,
    verify_bl_repair_audit_evaluation,
    write_bl_repair_audit_candidate_bundle,
    write_bl_repair_audit_evaluation,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_finetuning import (
    PublicFineTuneReceiptV2,
    SourceAttestedPublicRecognizerFineTuneError,
    VerifiedPublicFineTunedRecognizerReleaseV1,
    load_verified_public_finetuned_recognizer_release_v1,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_voyage_replay import (
    VoyageReplayError,
    VoyageReplayReceiptV3,
    VerifiedVoyageReplayReleaseV1,
    load_verified_voyage_replay_release_v1,
)
from customs_ai_agent.evaluation.synthetic_raster import (
    RASTER_PROFILES,
    SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
    SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH,
    RasterAssetReceiptV1,
    RasterGoldArtifactV1,
    ResolvedSyntheticRasterBuildInputsV1,
    SyntheticRasterCompilerUnavailable,
    SyntheticRasterReceiptV1,
    SyntheticRasterSourceReceiptV1,
    compile_synthetic_raster_dataset,
    deterministic_family_splits,
    ensure_fixed_synthetic_raster_build_inputs,
    synthetic_raster_source_receipt_sha256,
    validate_fixed_synthetic_raster_source_receipt,
    verify_rendered_raster_output,
    verify_synthetic_raster_receipt_materials,
    RenderedRasterBytesV1,
)
from customs_ai_agent.evaluation.synthetic_raster_text import (
    RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
    RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    RasterTextEvaluationCaseV1,
    RasterTextGateScoreV1,
    RasterTextGoldArtifactV1,
    RasterTextMetricCountsV1,
    canonicalize_synthetic_raster_semantic_text_v3,
    evaluate_synthetic_raster_semantic_text_case_v2,
    evaluate_synthetic_raster_semantic_text_case_v3,
    evaluate_synthetic_raster_semantic_text_gates_v2,
    evaluate_synthetic_raster_semantic_text_gates_v3,
    evaluate_synthetic_raster_text_case,
    evaluate_synthetic_raster_text_gates,
    source_only_text_gold_from_raster_gold,
    synthetic_raster_semantic_text_gate_passes_v2,
    synthetic_raster_semantic_text_gate_passes_v3,
    synthetic_raster_semantic_text_preview_draft_gate_passes_v1,
    synthetic_raster_text_gate_passes,
)
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionRequestMetadataV1,
    RasterTextExtractionResultV1,
    RasterTextInputV1,
    raster_text_result_json,
)
from customs_ai_agent.service.document_input import inspect_raster_document_structure
from customs_ai_agent.service.raster_text_extraction import (
    MediaAwareRasterTextEngine,
    extract_raster_text_preview,
)
from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextCalibrationReceiptV1,
    RasterTextCalibrationReceiptV4,
    RasterTextCalibrationReceiptV5,
    RasterTextCalibrationReceiptV6,
    RasterTextCalibrationReceiptV7,
    RasterTextActivationPointerV6,
    RasterTextActivationPointerV7,
    RasterTextActivationPointerV8,
    RasterTextActivationPointerV9,
    RasterTextActivationPointerV11,
    RasterTextActivationPointerV12,
    RasterTextOfflineProfile,
    RasterTextOfflineProfileV14,
    RasterTextOfflineProfileV15,
    RasterTextOfflineProfileV18,
    RasterTextOfflineProfileV19,
    RasterTextOfflineProfileV20,
    RasterTextSealedEvaluationV1,
    RasterTextSealedEvaluationV4,
    RasterTextSealedEvaluationV5,
    RasterTextSealedEvaluationV6,
    RasterTextSealedEvaluationV7,
    RasterTextSealedEvaluationV8,
    RasterTextSealedEvaluationV9,
    RasterTextSealedProfileScoreV1,
    RasterTextSealedProfileScoreV4,
    RasterTextSealedProfileScoreV5,
    RasterTextSealedProfileScoreV6,
    RasterTextSealedProfileScoreV7,
    RasterTextSealedProfileScoreV8,
    RasterTextWarmServiceSmokeReceiptV1,
    parse_raster_text_offline_profile,
    raster_text_exact_result_set_sha256,
    raster_text_service_startup_timeout_seconds,
    resolve_paddleocr3_raster_text_candidate,
    write_source_attested_synthetic_raster_text_pointer,
)


_SHA256 = r"^[a-f0-9]{64}$"
_Scope = Literal["smoke", "calibration", "sealed"]
_FreshScope = Literal["fresh-calibration", "fresh-sealed"]
_FreshSelectionRevision = Literal[
    "source-receipt-train-holdout-v1",
    "source-receipt-train-holdout-v2",
]
_FreshProjectionKind = Literal[
    "calibration-10-family-v1",
    "sealed-stage1-50-family-v1",
    "sealed-stage2-100-family-v1",
]
_FreshSealedStageDecision = Literal["early_pass", "early_fail", "continue"]
_Preview97QualityGateRevision = Literal["preview-draft-critical97-cer2-v1"]
_CandidateScope = Literal["smoke", "calibration", "sealed", "fresh-calibration", "fresh-sealed"]
_SemanticMetricRevision = Literal[
    "source-attested-semantic-text-v2",
    "source-attested-semantic-text-v3",
]
_ACTIVATION_LOCK_NAME = ".synthetic-raster-text-v1-auto-activation.lock"
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
# The immutable 24,000-asset raster receipt is intentionally larger than the
# compact activation/evaluation receipts.  Keep this ceiling aligned with the
# raster compiler's own local-artifact bound: it admits the complete known
# receipt (about 20 MiB) but still fail-closes oversized replacement files.
_MAX_DATASET_RECEIPT_BYTES = 32 * 1024 * 1024
_MAX_ASSET_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_PDF_BYTES = 32 * 1024 * 1024
_MAX_CANDIDATE_CHECKPOINT_BYTES = 2 * 1024 * 1024
_BL_REPAIR_AUDIT_CHECKPOINT_DIRECTORY = f"{BL_REPAIR_AUDIT_DATASET_ID}/candidate-checkpoints"
_V19_ENGINE_KIND: Final = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_V19B_ENGINE_KIND: Final = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
)
_V20_ENGINE_KIND: Final = (
    "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
# Immutable runner that produced the complete V19 calibration/staged-sealed
# chain retained by TASK-0067.  Preview-97 promotion may replay this one
# historical chain only; callers cannot nominate an arbitrary old runner.
_V19_PREVIEW97_HISTORICAL_RUNNER_IMPLEMENTATION_SHA256: Final = (
    "3eb9959fb437808d7174838f35b31f2a36c8d71616a1d1793e570797cdb51fa7"
)
_V19_PREVIEW97_QUALITY_GATE_REVISION: Final = "preview-draft-critical97-cer2-v1"
_V19_PREVIEW97_EVIDENCE_REVALIDATION_REVISION: Final = "historical-runner-current-runtime-smoke-v1"
_V19_CURRENT_NATIVE_EVIDENCE_GENERATION_REVISION: Final = "current-runner-native-full-gate-v1"
# These are the one audited public recognizer/model release retained by the
# best V19 one-pass profile.  The Linux activation command accepts neither a
# model selector nor a hash input: deployment must provide these exact,
# content-addressed, platform-independent model materials alongside a new
# schema-2 Linux direct runtime bundle.
_V19_PUBLIC_FINETUNED_RECOGNIZER_RELEASE_SHA256: Final = (
    "51f4324be43c1840a574cdb4b9febb58ec175df80de79cd95a0ef9ff33e0bea5"
)
_V19_PUBLIC_FINETUNED_MODEL_BUNDLE_SHA256: Final = (
    "3b276c1a83740f73949d319095034bfdba01285aed3318d1f4cdcb7190e18185"
)
_V19_PREVIEW97_PROFILE_RELATIVE_PATH: Final = (
    "raster-text-runtime-profiles-v1/9eb7f34b40da6ba6c7316e8b/profile.json"
)
_V19_PREVIEW97_CALIBRATION_RECEIPT_RELATIVE_PATH: Final = (
    "synthetic-raster-text-v1/calibration-v7/"
    "e239ceb1604f032c9cd9035e87e28ea9f57070574a83aaebea60081c62d779c9.json"
)
_V19_PREVIEW97_SEALED_SOURCE_GOLD_RELATIVE_PATH: Final = (
    "synthetic-raster-text-v1/fresh-source-gold-receipts/fresh-sealed/"
    "19046b40055da9d08067c541d848a0f1e8ba3d53c1ceaa1a8bd96c0722b350f1.json"
)
_V19_PREVIEW97_SEALED_EXECUTION_RELATIVE_PATH: Final = (
    "synthetic-raster-text-v1/fresh-projected-candidate-executions/fresh-sealed/"
    "b70d19bada1c0b73198fc9d739fa0f05336108bb324fff2df5983cdd56829bfa.json"
)
_V19_PREVIEW97_STAGE1_DECISION_RELATIVE_PATH: Final = (
    "synthetic-raster-text-v1/fresh-sealed-stage-decisions/"
    "5ed2a42550141ee45c94e7cc64f1a8fd434e7a722c98a1d416e621734abf56ee.json"
)
_V19_PREVIEW97_HISTORICAL_WARM_SMOKE_RELATIVE_PATH: Final = (
    "synthetic-raster-text-v1/warm-service-smoke/9eb7f34b40da6ba6c7316e8b/"
    "737885a82cbc4525fc8721e33ce70da64571e19303d067d8eafe8258e0f0d2e4.json"
)
_V19_FINE_TUNE_TOOL_RELATIVE_PATH = (
    Path("tools") / "finetune_source_attested_public_raster_text_recognizer.py"
)
# This is an offline model-training budget, not the service OCR request limit.
# V19 still has a separately enforced 30-second per-request ceiling and a
# 10-second p95 activation Gate.
_V19_FINE_TUNE_SUBPROCESS_TIMEOUT_SECONDS = 60 * 60
_V19_FINE_TUNE_SUBPROCESS_TERMINATE_SECONDS = 5
_MAX_V19_FINE_TUNE_CHILD_STDOUT_BYTES = 4 * 1024
_SAFE_V19_FINE_TUNE_CHILD_CODE = re.compile(r"^[a-z0-9_]{1,128}$")
_VOYAGE_REPLAY_TOOL_RELATIVE_PATH = (
    Path("tools") / "finetune_source_attested_public_raster_text_voyage_replay.py"
)
# This bounds the offline corpus replay/training child, not one OCR request.
# V20's immutable serving profile separately fixes 30 seconds per request and
# activation rejects any measured p95 above 10 seconds.
_VOYAGE_REPLAY_SUBPROCESS_TIMEOUT_SECONDS = 30 * 60
_VOYAGE_REPLAY_SUBPROCESS_TERMINATE_SECONDS = 5
_MAX_VOYAGE_REPLAY_CHILD_STDOUT_BYTES = 4 * 1024
_SAFE_VOYAGE_REPLAY_CHILD_CODE = re.compile(r"^[a-z0-9_]{1,128}$")
_FRESH_HOLDOUT_SELECTION_REVISION: Literal["source-receipt-train-holdout-v1"] = (
    "source-receipt-train-holdout-v1"
)
_FRESH_HOLDOUT_SELECTION_SEED_SHA256 = hashlib.sha256(
    b"synthetic-raster-text-v1:fresh-train-holdout:v1"
).hexdigest()
_FRESH_HOLDOUT_SELECTION_REVISION_V2: Literal["source-receipt-train-holdout-v2"] = (
    "source-receipt-train-holdout-v2"
)
_FRESH_HOLDOUT_SELECTION_SEED_SHA256_V2 = hashlib.sha256(
    b"synthetic-raster-text-v1:fresh-train-holdout:v2"
).hexdigest()
_FRESH_PROJECTION_REVISION = "source-bound-complete-family-hash-projection-v2"
_FRESH_PROJECTION_SEED_SHA256 = hashlib.sha256(
    b"synthetic-raster-text-v1:fresh-evaluation-projection:v2"
).hexdigest()
_FRESH_CALIBRATION_FAMILY_COUNT = 10
_FRESH_SEALED_STAGE1_FAMILY_COUNT = 50
_FRESH_SEALED_STAGE2_FAMILY_COUNT = 100
_FRESH_SEALED_SIMULTANEOUS_ALPHA = 0.05
# Two quality claims (character accuracy and critical-value accuracy) are made
# for each of the fixed 3 x 4 strata.  Bonferroni keeps the complete set of 24
# one-sided Wilson claims at no less than 95% family-wise confidence.
_FRESH_SEALED_SIMULTANEOUS_CLAIM_COUNT = 24

# The automatic lane deliberately has one bounded native configuration.  Keep
# these values together so a measured V15 comparison can change the candidate
# without introducing a CLI knob or an unrecorded runtime fallback.
_AUTO_ACTIVATION_V15_CPU_THREADS: Literal[4, 6] = 6
_AUTO_ACTIVATION_V15_RECOGNITION_BATCH_SIZE: Literal[8, 16] = 8
_AUTO_ACTIVATION_V15_OPENCV_THREADS: Literal[1] = 1
_AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS = 10.0


class SyntheticRasterTextActivationError(RuntimeError):
    """Content-free error suitable for the one-command wrapper."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest_model(value: BaseModel) -> str:
    return hashlib.sha256(
        _canonical_json(value.model_dump(mode="json", warnings=False))
    ).hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise SyntheticRasterTextActivationError("raster_text_activation_path_invalid")
    return path


def _root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise OSError
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(item.casefold() for item in root.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise OSError
        return root
    except OSError:
        raise SyntheticRasterTextActivationError("raster_text_activation_root_invalid") from None


def _path(root: Path, relative: str) -> Path:
    parts = _relative_path(relative).parts
    candidate = root
    try:
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if resolved.is_symlink() or not resolved.is_file():
            raise OSError
        return resolved
    except (OSError, ValueError):
        raise SyntheticRasterTextActivationError("raster_text_activation_path_invalid") from None


def _read(root: Path, relative: str, *, limit: int, code: str) -> bytes:
    try:
        path = _path(root, relative)
        raw = path.read_bytes()
        if not raw or len(raw) > limit:
            raise OSError
        return raw
    except SyntheticRasterTextActivationError:
        raise
    except OSError:
        raise SyntheticRasterTextActivationError(code) from None


@dataclass(frozen=True, slots=True)
class _InputLimits:
    max_document_bytes: int = _MAX_ASSET_BYTES
    max_document_pages: int = 20
    max_document_image_pixels: int = 40_000_000


class RasterTextFreshScopeSelectionReceiptV1(_WireModel):
    """Immutable fresh holdouts selected only from the unused train families."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_scope_selection"] = (
        "synthetic_raster_text_fresh_scope_selection"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    selection_revision: Literal["source-receipt-train-holdout-v1"] = (
        "source-receipt-train-holdout-v1"
    )
    source_split: Literal["train"] = "train"
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    source_receipt_relative_path: Literal["synthetic-raster-v1/source-receipt.json"] = (
        "synthetic-raster-v1/source-receipt.json"
    )
    source_receipt_sha256: str = Field(pattern=_SHA256)
    selection_seed_sha256: str = Field(pattern=_SHA256)
    fresh_calibration_family_ids: list[str] = Field(min_length=50, max_length=50)
    fresh_sealed_family_ids: list[str] = Field(min_length=100, max_length=100)

    @model_validator(mode="after")
    def _families_are_sorted_unique_and_disjoint(
        self,
    ) -> "RasterTextFreshScopeSelectionReceiptV1":
        calibration = self.fresh_calibration_family_ids
        sealed = self.fresh_sealed_family_ids
        if (
            calibration != sorted(calibration)
            or sealed != sorted(sealed)
            or len(calibration) != len(set(calibration))
            or len(sealed) != len(set(sealed))
            or set(calibration) & set(sealed)
            or self.selection_seed_sha256 != _FRESH_HOLDOUT_SELECTION_SEED_SHA256
        ):
            raise ValueError("fresh raster text family selection is invalid")
        return self


class RasterTextFreshScopeSelectionReceiptV2(_WireModel):
    """Second result-blind sealed holdout from the V1 train-family remainder."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_scope_selection"] = (
        "synthetic_raster_text_fresh_scope_selection"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    selection_revision: Literal["source-receipt-train-holdout-v2"] = (
        "source-receipt-train-holdout-v2"
    )
    source_split: Literal["train"] = "train"
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    source_receipt_relative_path: Literal["synthetic-raster-v1/source-receipt.json"] = (
        "synthetic-raster-v1/source-receipt.json"
    )
    source_receipt_sha256: str = Field(pattern=_SHA256)
    parent_selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    parent_selection_receipt_sha256: str = Field(pattern=_SHA256)
    selection_seed_sha256: str = Field(pattern=_SHA256)
    excluded_family_count: Literal[400] = 400
    excluded_family_ids_sha256: str = Field(pattern=_SHA256)
    eligible_family_count: Literal[600] = 600
    eligible_family_ids_sha256: str = Field(pattern=_SHA256)
    fresh_calibration_family_ids: list[str] = Field(min_length=50, max_length=50)
    fresh_sealed_family_ids: list[str] = Field(min_length=100, max_length=100)

    @model_validator(mode="after")
    def _families_and_parent_are_canonical(
        self,
    ) -> "RasterTextFreshScopeSelectionReceiptV2":
        calibration = self.fresh_calibration_family_ids
        sealed = self.fresh_sealed_family_ids
        expected_parent_path = (
            "synthetic-raster-text-v1/fresh-scope-selections/"
            f"{self.parent_selection_receipt_sha256}.json"
        )
        if (
            calibration != sorted(calibration)
            or sealed != sorted(sealed)
            or len(calibration) != len(set(calibration))
            or len(sealed) != len(set(sealed))
            or set(calibration) & set(sealed)
            or self.selection_seed_sha256 != _FRESH_HOLDOUT_SELECTION_SEED_SHA256_V2
            or self.parent_selection_receipt_relative_path != expected_parent_path
        ):
            raise ValueError("fresh raster text V2 family selection is invalid")
        _relative_path(self.parent_selection_receipt_relative_path)
        return self


RasterTextFreshScopeSelectionReceipt = (
    RasterTextFreshScopeSelectionReceiptV1 | RasterTextFreshScopeSelectionReceiptV2
)


class RasterTextFreshEvaluationProjectionReceiptV1(_WireModel):
    """Result-blind complete-family projection over one full fresh scope.

    The full source-Gold receipt remains the checkpoint identity input.  This
    receipt only selects a deterministic, result-blind family subset, so already-created
    observations are never rebound to a different Gold digest or silently
    cherry-picked after their OCR result is known.
    """

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_evaluation_projection"] = (
        "synthetic_raster_text_fresh_evaluation_projection"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    projection_revision: Literal["source-bound-complete-family-hash-projection-v2"] = (
        "source-bound-complete-family-hash-projection-v2"
    )
    projection_seed_sha256: str = Field(pattern=_SHA256)
    source_scope: _FreshScope
    projection_kind: _FreshProjectionKind
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    full_source_gold_receipt_relative_path: str = Field(min_length=1, max_length=240)
    full_source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    family_ids: list[str] = Field(min_length=10, max_length=100)
    family_count: int = Field(ge=10, le=100)
    asset_count: int = Field(ge=240, le=2_400)
    per_stratum_asset_count: int = Field(ge=20, le=200)
    parent_stage1_decision_relative_path: str | None = Field(
        default=None, min_length=1, max_length=240
    )
    parent_stage1_decision_sha256: str | None = Field(default=None, pattern=_SHA256)

    @model_validator(mode="after")
    def _fixed_projection_shape(self) -> "RasterTextFreshEvaluationProjectionReceiptV1":
        expected = {
            "calibration-10-family-v1": ("fresh-calibration", 10, 240, 20),
            "sealed-stage1-50-family-v1": ("fresh-sealed", 50, 1_200, 100),
            "sealed-stage2-100-family-v1": ("fresh-sealed", 100, 2_400, 200),
        }[self.projection_kind]
        has_parent = self.parent_stage1_decision_relative_path is not None
        if (
            (self.source_scope, self.family_count, self.asset_count, self.per_stratum_asset_count)
            != expected
            or self.projection_revision != _FRESH_PROJECTION_REVISION
            or self.projection_seed_sha256 != _FRESH_PROJECTION_SEED_SHA256
            or self.family_ids != sorted(self.family_ids)
            or len(self.family_ids) != len(set(self.family_ids))
            or len(self.family_ids) != self.family_count
            or has_parent != (self.parent_stage1_decision_sha256 is not None)
            or (self.projection_kind == "sealed-stage2-100-family-v1") != has_parent
        ):
            raise ValueError("fresh raster text projection is invalid")
        _relative_path(self.selection_receipt_relative_path)
        _relative_path(self.full_source_gold_receipt_relative_path)
        if self.parent_stage1_decision_relative_path is not None:
            _relative_path(self.parent_stage1_decision_relative_path)
        return self


class RasterTextFreshSealedStageStratumV1(_WireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_errors: int = Field(ge=0)
    character_reference: int = Field(ge=1)
    critical_matches: int = Field(ge=0)
    critical_total: int = Field(ge=1)
    full_character_reference: int = Field(ge=1)
    full_critical_total: int = Field(ge=1)
    character_accuracy_lower_bound: float = Field(ge=0, le=1)
    critical_accuracy_lower_bound: float = Field(ge=0, le=1)
    point_gate_passed: bool
    full_gate_recoverable: bool

    @model_validator(mode="after")
    def _counts_are_nested(self) -> "RasterTextFreshSealedStageStratumV1":
        if (
            self.character_errors > self.character_reference
            or self.critical_matches > self.critical_total
            or self.character_reference > self.full_character_reference
            or self.critical_total > self.full_critical_total
            or self.character_errors > self.full_character_reference
            or self.critical_matches > self.full_critical_total
        ):
            raise ValueError("fresh raster text staged counts are invalid")
        return self


class RasterTextFreshSealedStageDecisionReceiptV1(_WireModel):
    """Content-free stage-1 decision which can only pass, fail, or expand."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_sealed_stage_decision"] = (
        "synthetic_raster_text_fresh_sealed_stage_decision"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    stage: Literal["stage1-50-family"] = "stage1-50-family"
    source_gold_receipt_relative_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_relative_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    projection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=_SHA256)
    simultaneous_method: Literal["bonferroni-wilson-one-sided-v1"] = (
        "bonferroni-wilson-one-sided-v1"
    )
    family_wise_confidence: float = Field(default=0.95, ge=0.95, le=0.95)
    simultaneous_claim_count: Literal[24] = 24
    decision: _FreshSealedStageDecision
    strata: list[RasterTextFreshSealedStageStratumV1] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _decision_matches_strata(self) -> "RasterTextFreshSealedStageDecisionReceiptV1":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in RASTER_PROFILES
        }
        keys = [(item.document_type, item.profile_id) for item in self.strata]
        early_pass = all(
            item.point_gate_passed
            and item.character_accuracy_lower_bound >= 0.98
            and item.critical_accuracy_lower_bound >= 0.98
            for item in self.strata
        )
        early_fail = any(not item.full_gate_recoverable for item in self.strata)
        expected_decision: _FreshSealedStageDecision = (
            "early_pass" if early_pass else "early_fail" if early_fail else "continue"
        )
        if (
            set(keys) != expected
            or keys != sorted(keys)
            or self.decision != expected_decision
            or self.simultaneous_claim_count != _FRESH_SEALED_SIMULTANEOUS_CLAIM_COUNT
            or self.family_wise_confidence != 0.95
        ):
            raise ValueError("fresh raster text staged decision is invalid")
        for value in (
            self.source_gold_receipt_relative_path,
            self.candidate_execution_relative_path,
            self.projection_receipt_relative_path,
        ):
            _relative_path(value)
        return self


class RasterTextFreshSealedStageDecisionReceiptV2(_WireModel):
    """Stage-1 decision with CER 2% and critical exact 97% made explicit."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_sealed_stage_decision"] = (
        "synthetic_raster_text_fresh_sealed_stage_decision"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    quality_gate_revision: _Preview97QualityGateRevision
    stage: Literal["stage1-50-family"] = "stage1-50-family"
    source_gold_receipt_relative_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_relative_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    projection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=_SHA256)
    simultaneous_method: Literal["bonferroni-wilson-one-sided-v1"] = (
        "bonferroni-wilson-one-sided-v1"
    )
    family_wise_confidence: float = Field(default=0.95, ge=0.95, le=0.95)
    simultaneous_claim_count: Literal[24] = 24
    decision: _FreshSealedStageDecision
    strata: list[RasterTextFreshSealedStageStratumV1] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _decision_matches_preview97_strata(
        self,
    ) -> "RasterTextFreshSealedStageDecisionReceiptV2":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in RASTER_PROFILES
        }
        keys = [(item.document_type, item.profile_id) for item in self.strata]
        early_pass = all(
            item.point_gate_passed
            and item.character_accuracy_lower_bound >= 0.98
            and item.critical_accuracy_lower_bound >= 0.97
            for item in self.strata
        )
        early_fail = any(not item.full_gate_recoverable for item in self.strata)
        expected_decision: _FreshSealedStageDecision = (
            "early_pass" if early_pass else "early_fail" if early_fail else "continue"
        )
        if (
            set(keys) != expected
            or keys != sorted(keys)
            or self.decision != expected_decision
            or self.simultaneous_claim_count != _FRESH_SEALED_SIMULTANEOUS_CLAIM_COUNT
            or self.family_wise_confidence != 0.95
        ):
            raise ValueError("fresh raster text preview-97 staged decision is invalid")
        for value in (
            self.source_gold_receipt_relative_path,
            self.candidate_execution_relative_path,
            self.projection_receipt_relative_path,
        ):
            _relative_path(value)
        return self


RasterTextFreshSealedStageDecisionReceipt = (
    RasterTextFreshSealedStageDecisionReceiptV1 | RasterTextFreshSealedStageDecisionReceiptV2
)


class RasterTextSourceGoldBindingV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128)
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    family_id: str = Field(min_length=1, max_length=128)
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    source_pdf_sha256: str = Field(pattern=_SHA256)
    image_sha256: str = Field(pattern=_SHA256)
    source_raster_gold_relative_path: str = Field(min_length=1, max_length=240)
    source_raster_gold_sha256: str = Field(pattern=_SHA256)
    derived_text_gold_sha256: str = Field(pattern=_SHA256)


class RasterTextSourceGoldReceiptV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_source_gold_receipt"] = (
        "synthetic_raster_text_source_gold_receipt"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    scope: _Scope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    bindings: list[RasterTextSourceGoldBindingV1] = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def _unique_coverage(self) -> "RasterTextSourceGoldReceiptV1":
        keys = [(item.asset_id, item.profile_id) for item in self.bindings]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("raster text source Gold receipt coverage is invalid")
        return self


class RasterTextFreshSourceGoldReceiptV1(_WireModel):
    """Source-only Gold bindings for one immutable fresh train holdout."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_source_gold_receipt"] = (
        "synthetic_raster_text_fresh_source_gold_receipt"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    scope: _FreshScope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    source_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    bindings: list[RasterTextSourceGoldBindingV1] = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def _unique_coverage(self) -> "RasterTextFreshSourceGoldReceiptV1":
        keys = [(item.asset_id, item.profile_id) for item in self.bindings]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("fresh raster text source Gold receipt coverage is invalid")
        _relative_path(self.selection_receipt_relative_path)
        return self


class RasterTextCandidateObservationV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128)
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    # Cold model construction finishes before the evaluation-session ready
    # frame.  This measures one already-warm OCR response, so it shares the
    # production request ceiling rather than the one-time setup allowance.
    elapsed_seconds: float = Field(gt=0, le=30)
    candidate: RasterTextExtractionResultV1

    @model_validator(mode="after")
    def _binding(self) -> "RasterTextCandidateObservationV1":
        if (
            self.candidate.document_asset_id != self.asset_id
            or self.candidate.status == "unavailable"
        ):
            raise ValueError("raster text candidate binding is invalid")
        return self


class RasterTextCandidateCheckpointIdentityV1(_WireModel):
    """All immutable inputs that make one asset observation reusable."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _Scope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    asset_id: str = Field(min_length=1, max_length=128)
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    image_sha256: str = Field(pattern=_SHA256)
    gold_sha256: str = Field(pattern=_SHA256)


class RasterTextCandidateCheckpointV1(_WireModel):
    """Create-only local evidence for one completed warm OCR request."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_candidate_checkpoint"] = (
        "synthetic_raster_text_candidate_checkpoint"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    identity: RasterTextCandidateCheckpointIdentityV1
    identity_sha256: str = Field(pattern=_SHA256)
    elapsed_seconds: float = Field(gt=0, le=30)
    observation_sha256: str = Field(pattern=_SHA256)
    observation: RasterTextCandidateObservationV1

    @model_validator(mode="after")
    def _binding(self) -> "RasterTextCandidateCheckpointV1":
        if (
            self.identity_sha256 != _digest_model(self.identity)
            or self.observation_sha256 != _digest_model(self.observation)
            or self.elapsed_seconds != self.observation.elapsed_seconds
            or self.identity.asset_id != self.observation.asset_id
            or self.identity.profile_id != self.observation.profile_id
        ):
            raise ValueError("raster text candidate checkpoint binding is invalid")
        return self


class RasterTextFreshCandidateCheckpointIdentityV1(_WireModel):
    """Checkpoint identity with the additional immutable holdout selection."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _FreshScope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    asset_id: str = Field(min_length=1, max_length=128)
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    image_sha256: str = Field(pattern=_SHA256)
    gold_sha256: str = Field(pattern=_SHA256)


class RasterTextFreshCandidateCheckpointV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_candidate_checkpoint"] = (
        "synthetic_raster_text_fresh_candidate_checkpoint"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    identity: RasterTextFreshCandidateCheckpointIdentityV1
    identity_sha256: str = Field(pattern=_SHA256)
    elapsed_seconds: float = Field(gt=0, le=30)
    observation_sha256: str = Field(pattern=_SHA256)
    observation: RasterTextCandidateObservationV1

    @model_validator(mode="after")
    def _binding(self) -> "RasterTextFreshCandidateCheckpointV1":
        if (
            self.identity_sha256 != _digest_model(self.identity)
            or self.observation_sha256 != _digest_model(self.observation)
            or self.elapsed_seconds != self.observation.elapsed_seconds
            or self.identity.asset_id != self.observation.asset_id
            or self.identity.profile_id != self.observation.profile_id
        ):
            raise ValueError("fresh raster text candidate checkpoint binding is invalid")
        return self


class BlRepairAuditCandidateCheckpointIdentityV1(_WireModel):
    """Complete immutable identity for one independent B/L audit request."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-bl-repair-audit-v1"] = BL_REPAIR_AUDIT_DATASET_ID
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    asset_id: str = Field(min_length=1, max_length=128)
    family_id: str = Field(min_length=1, max_length=128)
    fixture_kind: Literal["clean", "error"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    source_pdf_sha256: str = Field(pattern=_SHA256)
    image_sha256: str = Field(pattern=_SHA256)
    gold_sha256: str = Field(pattern=_SHA256)
    media_type: Literal["image/png", "image/jpeg", "application/pdf"]
    byte_size: int = Field(ge=1, le=_MAX_ASSET_BYTES)


class BlRepairAuditCandidateCheckpointV1(_WireModel):
    """Create-only result for one completed, warm, at-most-30-second request."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_bl_repair_audit_candidate_checkpoint"] = (
        "source_attested_bl_repair_audit_candidate_checkpoint"
    )
    dataset_id: Literal["synthetic-bl-repair-audit-v1"] = BL_REPAIR_AUDIT_DATASET_ID
    identity: BlRepairAuditCandidateCheckpointIdentityV1
    identity_sha256: str = Field(pattern=_SHA256)
    elapsed_seconds: float = Field(gt=0, le=30)
    candidate_sha256: str = Field(pattern=_SHA256)
    candidate: RasterTextExtractionResultV1

    @model_validator(mode="after")
    def _binding(self) -> "BlRepairAuditCandidateCheckpointV1":
        if (
            self.identity_sha256 != _digest_model(self.identity)
            or self.candidate_sha256 != _digest_model(self.candidate)
            or self.identity.asset_id != self.candidate.document_asset_id
            or self.identity.image_sha256 != self.candidate.source_sha256
        ):
            raise ValueError("B/L repair audit candidate checkpoint binding is invalid")
        return self


@dataclass(frozen=True, slots=True)
class BlRepairAuditCandidateRunV1:
    dataset_receipt_relative_path: str
    dataset_receipt_sha256: str
    candidate_bundle_relative_path: str
    candidate_bundle_sha256: str
    evaluation_relative_path: str
    evaluation_sha256: str
    raster_text_profile_revision: str
    runtime_configuration_sha256: str


class RasterTextCandidateBundleV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_candidate_bundle"] = (
        "synthetic_raster_text_candidate_bundle"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _Scope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    candidates: list[RasterTextCandidateObservationV1] = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def _coverage(self) -> "RasterTextCandidateBundleV1":
        keys = [(item.asset_id, item.profile_id) for item in self.candidates]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("raster text candidate coverage is invalid")
        if any(
            item.candidate.raster_text_profile_revision != self.raster_text_profile_revision
            or item.candidate.extractor_revision != self.extractor_revision
            or item.candidate.canonical_text is None
            for item in self.candidates
        ):
            raise ValueError("raster text candidate provenance is invalid")
        return self


class RasterTextCandidateExecutionV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_candidate_execution"] = (
        "synthetic_raster_text_candidate_execution"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _Scope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_count: int = Field(ge=1, le=4_000)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30


class RasterTextFreshCandidateBundleV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_candidate_bundle"] = (
        "synthetic_raster_text_fresh_candidate_bundle"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _FreshScope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    candidates: list[RasterTextCandidateObservationV1] = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def _coverage(self) -> "RasterTextFreshCandidateBundleV1":
        keys = [(item.asset_id, item.profile_id) for item in self.candidates]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("fresh raster text candidate coverage is invalid")
        if any(
            item.candidate.raster_text_profile_revision != self.raster_text_profile_revision
            or item.candidate.extractor_revision != self.extractor_revision
            or item.candidate.canonical_text is None
            for item in self.candidates
        ):
            raise ValueError("fresh raster text candidate provenance is invalid")
        _relative_path(self.selection_receipt_relative_path)
        return self


class RasterTextFreshCandidateExecutionV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_candidate_execution"] = (
        "synthetic_raster_text_fresh_candidate_execution"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _FreshScope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_gold_receipt_relative_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_count: int = Field(ge=1, le=4_000)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _paths_are_local(self) -> "RasterTextFreshCandidateExecutionV1":
        _relative_path(self.selection_receipt_relative_path)
        _relative_path(self.source_gold_receipt_relative_path)
        _relative_path(self.candidate_bundle_relative_path)
        return self


class RasterTextFreshProjectedCandidateBundleV1(_WireModel):
    """Candidate subset whose observations retain their full-scope identity."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_projected_candidate_bundle"] = (
        "synthetic_raster_text_fresh_projected_candidate_bundle"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _FreshScope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    projection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    runner_implementation_sha256: str = Field(pattern=_SHA256)
    candidates: list[RasterTextCandidateObservationV1] = Field(min_length=240, max_length=2_400)

    @model_validator(mode="after")
    def _coverage(self) -> "RasterTextFreshProjectedCandidateBundleV1":
        keys = [(item.asset_id, item.profile_id) for item in self.candidates]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("fresh projected raster text candidate coverage is invalid")
        if any(
            item.candidate.raster_text_profile_revision != self.raster_text_profile_revision
            or item.candidate.extractor_revision != self.extractor_revision
            or item.candidate.canonical_text is None
            for item in self.candidates
        ):
            raise ValueError("fresh projected raster text candidate provenance is invalid")
        for value in (
            self.selection_receipt_relative_path,
            self.projection_receipt_relative_path,
        ):
            _relative_path(value)
        return self


class RasterTextFreshProjectedCandidateExecutionV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_fresh_projected_candidate_execution"] = (
        "synthetic_raster_text_fresh_projected_candidate_execution"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: _FreshScope
    synthetic_raster_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_gold_receipt_relative_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    projection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=_SHA256)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_count: Literal[240, 1_200, 2_400]
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _paths_are_local(self) -> "RasterTextFreshProjectedCandidateExecutionV1":
        for value in (
            self.selection_receipt_relative_path,
            self.source_gold_receipt_relative_path,
            self.projection_receipt_relative_path,
            self.candidate_bundle_relative_path,
        ):
            _relative_path(value)
        return self


@dataclass(frozen=True, slots=True)
class RasterTextCandidateRunV1:
    source_gold_receipt_relative_path: str
    source_gold_receipt_sha256: str
    candidate_execution_relative_path: str
    candidate_execution_sha256: str
    raster_text_profile_revision: str
    runtime_configuration_sha256: str
    latency_p95_seconds: float


class RasterTextSmokeQualityScreenV1(_WireModel):
    """Immutable early-rejection evidence for the fixed 24-asset smoke set.

    This screen is intentionally not approval evidence: it can only stop a
    weak candidate before the 240-asset calibration and staged sealed work.
    The sealed split remains the sole quality basis for pointer activation.
    """

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["synthetic_raster_text_smoke_quality_screen"] = (
        "synthetic_raster_text_smoke_quality_screen"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    scope: Literal["smoke"] = "smoke"
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    scores: list[RasterTextGateScoreV1] = Field(min_length=12, max_length=12)
    gate_passed: bool

    @model_validator(mode="after")
    def _exact_screen_matrix(self) -> "RasterTextSmokeQualityScreenV1":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if {(item.document_type, item.profile_id) for item in self.scores} != expected:
            raise ValueError("raster text smoke quality coverage is invalid")
        if self.gate_passed != synthetic_raster_text_gate_passes(self.scores):
            raise ValueError("raster text smoke quality result is invalid")
        return self


class RasterTextSmokeQualityScreenV2(_WireModel):
    """Early-rejection evidence bound to the semantic text metric revision."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_smoke_quality_screen"] = (
        "synthetic_raster_semantic_text_smoke_quality_screen"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v2"]
    scope: Literal["smoke"] = "smoke"
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    scores: list[RasterTextGateScoreV1] = Field(min_length=12, max_length=12)
    gate_passed: bool

    @model_validator(mode="after")
    def _exact_screen_matrix(self) -> "RasterTextSmokeQualityScreenV2":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if {(item.document_type, item.profile_id) for item in self.scores} != expected:
            raise ValueError("raster semantic text smoke quality coverage is invalid")
        if self.gate_passed != synthetic_raster_semantic_text_gate_passes_v2(self.scores):
            raise ValueError("raster semantic text smoke quality result is invalid")
        return self


class RasterTextSmokeQualityScreenV3(_WireModel):
    """Early-rejection evidence bound to semantic text metric V3."""

    schema_version: Literal["3.0"] = "3.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_smoke_quality_screen"] = (
        "synthetic_raster_semantic_text_smoke_quality_screen"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    scope: Literal["smoke"] = "smoke"
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    scores: list[RasterTextGateScoreV1] = Field(min_length=12, max_length=12)
    gate_passed: bool

    @model_validator(mode="after")
    def _exact_screen_matrix(self) -> "RasterTextSmokeQualityScreenV3":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if {(item.document_type, item.profile_id) for item in self.scores} != expected:
            raise ValueError("raster semantic text smoke quality coverage is invalid")
        if self.gate_passed != synthetic_raster_semantic_text_gate_passes_v3(self.scores):
            raise ValueError("raster semantic text smoke quality result is invalid")
        return self


class RasterTextSmokeQualityScreenV4(_WireModel):
    """Preview-97 early rejection screen; never sufficient for activation."""

    schema_version: Literal["4.0"] = "4.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_smoke_quality_screen"] = (
        "synthetic_raster_semantic_text_smoke_quality_screen"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    quality_gate_revision: _Preview97QualityGateRevision
    scope: Literal["smoke"] = "smoke"
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    scores: list[RasterTextGateScoreV1] = Field(min_length=12, max_length=12)
    gate_passed: bool

    @model_validator(mode="after")
    def _exact_preview97_screen_matrix(self) -> "RasterTextSmokeQualityScreenV4":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if {(item.document_type, item.profile_id) for item in self.scores} != expected:
            raise ValueError("raster preview-97 smoke quality coverage is invalid")
        if self.gate_passed != synthetic_raster_semantic_text_preview_draft_gate_passes_v1(
            self.scores
        ):
            raise ValueError("raster preview-97 smoke quality result is invalid")
        return self


class RasterTextCalibrationReceiptV2(_WireModel):
    """Confidence selection evidence bound to semantic metric V2."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_calibration"] = (
        "synthetic_raster_semantic_text_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["calibration"] = "calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v2"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV2":
        _relative_path(self.source_gold_receipt_path)
        _relative_path(self.candidate_execution_path)
        return self


class RasterTextCalibrationReceiptV3(_WireModel):
    """Confidence selection evidence bound to semantic metric V3."""

    schema_version: Literal["3.0"] = "3.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_calibration"] = (
        "synthetic_raster_semantic_text_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["calibration"] = "calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV3":
        _relative_path(self.source_gold_receipt_path)
        _relative_path(self.candidate_execution_path)
        return self


class RasterTextSealedProfileScoreV2(_WireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedProfileScoreV3(_WireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedEvaluationV2(_WireModel):
    """Passing sealed evidence bound to semantic metric V2."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_sealed_evaluation"] = (
        "synthetic_raster_semantic_text_sealed_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["sealed"] = "sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v2"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=_SHA256)
    latency_p95_seconds: float = Field(gt=0, le=30)
    service_warm_latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30
    minimum_page_confidence: float = Field(ge=0, le=1)
    scores: list[RasterTextSealedProfileScoreV2] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _all_profiles_are_passed(self) -> "RasterTextSealedEvaluationV2":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if {(score.document_type, score.profile_id) for score in self.scores} != expected:
            raise ValueError("raster semantic text sealed profiles are incomplete")
        for value in (
            self.source_gold_receipt_path,
            self.calibration_receipt_path,
            self.candidate_execution_path,
            self.warm_service_smoke_receipt_path,
        ):
            _relative_path(value)
        return self


class RasterTextSealedEvaluationV3(_WireModel):
    """Passing sealed evidence bound to semantic metric V3."""

    schema_version: Literal["3.0"] = "3.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_sealed_evaluation"] = (
        "synthetic_raster_semantic_text_sealed_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["sealed"] = "sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=_SHA256)
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=_SHA256)
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=_SHA256)
    latency_p95_seconds: float = Field(gt=0, le=30)
    service_warm_latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30
    minimum_page_confidence: float = Field(ge=0, le=1)
    scores: list[RasterTextSealedProfileScoreV3] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _all_profiles_are_passed(self) -> "RasterTextSealedEvaluationV3":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if {(score.document_type, score.profile_id) for score in self.scores} != expected:
            raise ValueError("raster semantic text sealed profiles are incomplete")
        for value in (
            self.source_gold_receipt_path,
            self.calibration_receipt_path,
            self.candidate_execution_path,
            self.warm_service_smoke_receipt_path,
        ):
            _relative_path(value)
        return self


def _semantic_metric_generation(
    metric_revision: _SemanticMetricRevision | None,
) -> Literal["v2", "v3"] | None:
    if metric_revision is None:
        return None
    if metric_revision == RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2:
        return "v2"
    if metric_revision == RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3:
        return "v3"
    raise SyntheticRasterTextActivationError("raster_text_metric_revision_invalid")


def _uses_semantic_metric_v2(
    metric_revision: Literal["source-attested-semantic-text-v2"] | None,
) -> bool:
    """Preserve the V2-only helper contract for existing callers."""

    return _semantic_metric_generation(metric_revision) == "v2"


def _dataset_family_splits(receipt: SyntheticRasterReceiptV1) -> dict[str, str]:
    split_by_family: dict[str, str] = {}
    asset_count_by_family: dict[str, int] = {}
    for asset in receipt.assets:
        existing = split_by_family.setdefault(asset.family_id, asset.split)
        if existing != asset.split:
            raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
        asset_count_by_family[asset.family_id] = asset_count_by_family.get(asset.family_id, 0) + 1
    expected_counts = {
        "train": 750,
        "development": 100,
        "calibration": 50,
        "sealed": 100,
    }
    observed_counts = {
        split: sum(value == split for value in split_by_family.values())
        for split in expected_counts
    }
    if (
        len(split_by_family) != 1_000
        or observed_counts != expected_counts
        or set(asset_count_by_family.values()) != {24}
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return split_by_family


def _build_fresh_scope_selection(
    *,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    source_receipt_sha256: str,
) -> RasterTextFreshScopeSelectionReceiptV1:
    split_by_family = _dataset_family_splits(receipt)
    train_families = [family_id for family_id, split in split_by_family.items() if split == "train"]
    ranked = sorted(
        train_families,
        key=lambda family_id: (
            hashlib.sha256(
                (
                    f"{_FRESH_HOLDOUT_SELECTION_SEED_SHA256}:"
                    f"{receipt_sha256}:{source_receipt_sha256}:{family_id}"
                ).encode("ascii")
            ).hexdigest(),
            family_id,
        ),
    )
    selection = RasterTextFreshScopeSelectionReceiptV1(
        synthetic_raster_receipt_sha256=receipt_sha256,
        source_receipt_sha256=source_receipt_sha256,
        selection_seed_sha256=_FRESH_HOLDOUT_SELECTION_SEED_SHA256,
        fresh_calibration_family_ids=sorted(ranked[:50]),
        fresh_sealed_family_ids=sorted(ranked[50:150]),
    )
    excluded = {family_id for family_id, split in split_by_family.items() if split != "train"}
    if (
        set(selection.fresh_calibration_family_ids) & excluded
        or set(selection.fresh_sealed_family_ids) & excluded
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return selection


def _fresh_family_ids_sha256(family_ids: Sequence[str]) -> str:
    """Hash one canonical sorted, duplicate-free family-id list."""

    canonical = sorted(family_ids)
    if canonical != list(family_ids) or len(canonical) != len(set(canonical)):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _build_fresh_scope_selection_v2(
    *,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    source_receipt_sha256: str,
) -> RasterTextFreshScopeSelectionReceiptV2:
    """Reserve a new sealed set after excluding every V1/non-train family."""

    split_by_family = _dataset_family_splits(receipt)
    parent = _build_fresh_scope_selection(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        source_receipt_sha256=source_receipt_sha256,
    )
    parent_sha256 = _digest_model(parent)
    parent_relative_path = f"synthetic-raster-text-v1/fresh-scope-selections/{parent_sha256}.json"
    parent_families = set(parent.fresh_calibration_family_ids) | set(parent.fresh_sealed_family_ids)
    non_train_families = {
        family_id for family_id, split in split_by_family.items() if split != "train"
    }
    excluded_family_ids = sorted(non_train_families | parent_families)
    eligible_family_ids = sorted(set(split_by_family) - set(excluded_family_ids))
    if (
        len(parent_families) != 150
        or len(non_train_families) != 250
        or parent_families & non_train_families
        or len(excluded_family_ids) != 400
        or len(eligible_family_ids) != 600
        or any(split_by_family[family_id] != "train" for family_id in eligible_family_ids)
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    ranked = sorted(
        eligible_family_ids,
        key=lambda family_id: (
            hashlib.sha256(
                (
                    f"{_FRESH_HOLDOUT_SELECTION_SEED_SHA256_V2}:"
                    f"{receipt_sha256}:{source_receipt_sha256}:"
                    f"{parent_sha256}:{family_id}"
                ).encode("ascii")
            ).hexdigest(),
            family_id,
        ),
    )
    selection = RasterTextFreshScopeSelectionReceiptV2(
        synthetic_raster_receipt_sha256=receipt_sha256,
        source_receipt_sha256=source_receipt_sha256,
        parent_selection_receipt_relative_path=parent_relative_path,
        parent_selection_receipt_sha256=parent_sha256,
        selection_seed_sha256=_FRESH_HOLDOUT_SELECTION_SEED_SHA256_V2,
        excluded_family_ids_sha256=_fresh_family_ids_sha256(excluded_family_ids),
        eligible_family_ids_sha256=_fresh_family_ids_sha256(eligible_family_ids),
        fresh_calibration_family_ids=list(parent.fresh_calibration_family_ids),
        fresh_sealed_family_ids=sorted(ranked[:100]),
    )
    if (
        set(selection.fresh_calibration_family_ids) != set(parent.fresh_calibration_family_ids)
        or set(selection.fresh_sealed_family_ids) & set(excluded_family_ids)
        or not set(selection.fresh_sealed_family_ids).issubset(eligible_family_ids)
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return selection


def _build_fresh_scope_selection_for_revision(
    *,
    receipt: SyntheticRasterReceiptV1,
    receipt_sha256: str,
    source_receipt_sha256: str,
    selection_revision: _FreshSelectionRevision,
) -> RasterTextFreshScopeSelectionReceipt:
    if selection_revision == _FRESH_HOLDOUT_SELECTION_REVISION:
        return _build_fresh_scope_selection(
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            source_receipt_sha256=source_receipt_sha256,
        )
    if selection_revision == _FRESH_HOLDOUT_SELECTION_REVISION_V2:
        return _build_fresh_scope_selection_v2(
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            source_receipt_sha256=source_receipt_sha256,
        )
    raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")


def _fresh_scope_assets(
    *,
    receipt: SyntheticRasterReceiptV1,
    selection: RasterTextFreshScopeSelectionReceipt,
    scope: _FreshScope,
) -> list[RasterAssetReceiptV1]:
    split_by_family = _dataset_family_splits(receipt)
    if isinstance(selection, RasterTextFreshScopeSelectionReceiptV2):
        expected_v2 = _build_fresh_scope_selection_v2(
            receipt=receipt,
            receipt_sha256=selection.synthetic_raster_receipt_sha256,
            source_receipt_sha256=selection.source_receipt_sha256,
        )
        if selection != expected_v2:
            raise SyntheticRasterTextActivationError("raster_text_fresh_scope_coverage_invalid")
    family_ids = set(
        selection.fresh_calibration_family_ids
        if scope == "fresh-calibration"
        else selection.fresh_sealed_family_ids
    )
    expected_family_count = 50 if scope == "fresh-calibration" else 100
    expected_per_stratum = expected_family_count * 2
    if len(family_ids) != expected_family_count or any(
        split_by_family.get(family_id) != "train" for family_id in family_ids
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_scope_coverage_invalid")
    result = [asset for asset in receipt.assets if asset.family_id in family_ids]
    family_counts = {
        family_id: sum(asset.family_id == family_id for asset in result) for family_id in family_ids
    }
    strata = {
        (document_type, profile_id): sum(
            asset.document_type == document_type and asset.profile_id == profile_id
            for asset in result
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    }
    if (
        len(result) != expected_family_count * 24
        or set(family_counts.values()) != {24}
        or set(strata.values()) != {expected_per_stratum}
        or any(asset.split != "train" for asset in result)
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_scope_coverage_invalid")
    return sorted(result, key=lambda item: (item.asset_id, item.profile_id))


def _scope_assets(receipt: SyntheticRasterReceiptV1, scope: _Scope) -> list[RasterAssetReceiptV1]:
    if scope == "smoke":
        family = min(asset.family_id for asset in receipt.assets if asset.split == "calibration")
        result = [asset for asset in receipt.assets if asset.family_id == family]
        if len(result) != 24:
            raise SyntheticRasterTextActivationError("raster_text_smoke_coverage_invalid")
    else:
        result = [asset for asset in receipt.assets if asset.split == scope]
        expected = 1_200 if scope == "calibration" else 2_400
        if len(result) != expected:
            raise SyntheticRasterTextActivationError("raster_text_scope_coverage_invalid")
    return sorted(result, key=lambda item: (item.asset_id, item.profile_id))


def _existing_synthetic_raster_dataset(root: Path) -> bool:
    """Return whether the fixed output directory already exists, fail-closed."""

    candidate = root / "synthetic-raster-v1"
    try:
        if candidate.is_symlink():
            raise OSError
        if not candidate.exists():
            return False
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if resolved.is_symlink() or not resolved.is_dir():
            raise OSError
        return True
    except (OSError, ValueError):
        raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid") from None


def _verify_smoke_source_pdf(
    *,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1,
    source_relative_path: str,
    expected_sha256: str,
    expected_byte_size: int,
) -> None:
    """Hash one fixed smoke-family source PDF without trusting a receipt cache."""

    try:
        if expected_byte_size < 1 or expected_byte_size > _MAX_SOURCE_PDF_BYTES:
            raise OSError
        source_root = build_inputs.source_root
        if source_root.is_symlink() or not source_root.is_dir():
            raise OSError
        candidate = source_root
        for part in _relative_path(source_relative_path).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        source_path = candidate.resolve(strict=True)
        source_path.relative_to(source_root)
        if source_path.is_symlink() or not source_path.is_file():
            raise OSError
        raw = source_path.read_bytes()
        if len(raw) != expected_byte_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise OSError
    except (OSError, ValueError, SyntheticRasterTextActivationError):
        raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid") from None


def _verify_existing_dataset_smoke_materials(
    *,
    root: Path,
    receipt: SyntheticRasterReceiptV1,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1,
) -> None:
    """Verify only the deterministic 24-item smoke family before its OCR run.

    This is deliberately not an approval verifier.  The original all-split
    replay still runs after smoke and before calibration.  It exists only to
    avoid spending that replay on a candidate that immediately fails smoke.
    """

    try:
        if (
            build_inputs.document_preview_root != root
            or receipt.source_dataset_id != "synthetic-customs-pdf-2000"
            or receipt.source_dataset_revision != receipt.source_receipt_sha256
            or receipt.source_receipt_relative_path != SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH
            or receipt.build_inputs_relative_path != build_inputs.build_inputs_relative_path
            or receipt.build_inputs_sha256 != build_inputs.build_inputs_sha256
            or receipt.renderer_configuration_sha256 != build_inputs.renderer_configuration_sha256
            or receipt.renderer_revision != build_inputs.renderer_configuration.renderer_revision
            or receipt.pdfium_version != build_inputs.renderer_configuration.pdfium_version
            or receipt.font_manifest_sha256 != build_inputs.font_manifest_sha256
            or receipt.font_resource_policy != build_inputs.font_manifest.font_resource_policy
            or receipt.geometry_compiler_revision
            != build_inputs.renderer_configuration.geometry_compiler_revision
            or receipt.split_seed_sha256 != build_inputs.split_seed_sha256
            or receipt.render_recipe_sha256 != build_inputs.render_recipe_sha256
            or receipt.render_seed_sha256 != build_inputs.render_seed_sha256
            or [profile.model_dump(mode="json") for profile in receipt.profiles]
            != [profile.model_dump(mode="json") for profile in RASTER_PROFILES.values()]
        ):
            raise ValueError
    except (AttributeError, ValueError):
        raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid") from None

    source_raw = _read(
        root,
        SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH,
        limit=_MAX_DATASET_RECEIPT_BYTES,
        code="raster_text_dataset_material_invalid",
    )
    try:
        source_receipt = SyntheticRasterSourceReceiptV1.model_validate_json(source_raw)
        if (
            source_raw != _canonical_json(source_receipt.model_dump(mode="json", warnings=False))
            or synthetic_raster_source_receipt_sha256(source_receipt)
            != receipt.source_receipt_sha256
            or source_receipt.font_manifest_sha256 != receipt.font_manifest_sha256
        ):
            raise ValueError
        validate_fixed_synthetic_raster_source_receipt(source_receipt)
        split_by_family = deterministic_family_splits(
            sorted({source.family_id for source in source_receipt.assets}),
            split_seed_sha256=build_inputs.split_seed_sha256,
        )
        smoke_family = min(
            family_id for family_id, split in split_by_family.items() if split == "calibration"
        )
        smoke_sources = [
            source for source in source_receipt.assets if source.family_id == smoke_family
        ]
        if len(smoke_sources) != 6:
            raise ValueError
    except (ValidationError, ValueError):
        raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid") from None

    source_by_id = {source.asset_id: source for source in smoke_sources}
    expected_pairs = {
        (source.asset_id, profile_id) for source in smoke_sources for profile_id in RASTER_PROFILES
    }
    assets = _scope_assets(receipt, "smoke")
    if {(asset.asset_id, asset.profile_id) for asset in assets} != expected_pairs:
        raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid")

    for source in smoke_sources:
        _verify_smoke_source_pdf(
            build_inputs=build_inputs,
            source_relative_path=source.source_relative_path,
            expected_sha256=source.source_pdf_sha256,
            expected_byte_size=source.byte_size,
        )
    for asset in assets:
        source_asset = source_by_id.get(asset.asset_id)
        profile = RASTER_PROFILES.get(asset.profile_id)
        if (
            source_asset is None
            or profile is None
            or asset.family_id != source_asset.family_id
            or asset.document_type != source_asset.document_type
            or asset.source_pdf_sha256 != source_asset.source_pdf_sha256
            or asset.split != "calibration"
            or asset.media_type != profile.media_type
            or asset.coordinate_transform != receipt.coordinate_transform
            or asset.render_recipe_sha256 != receipt.render_recipe_sha256
            or asset.render_seed_sha256 != receipt.render_seed_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid")
        _verified_asset(root, asset, pdfium_version=receipt.pdfium_version)
        text_gold = _gold_for_asset(root, asset)
        if (
            text_gold.asset_id != asset.asset_id
            or text_gold.family_id != asset.family_id
            or text_gold.document_type != asset.document_type
            or text_gold.profile_id != asset.profile_id
            or text_gold.source_pdf_sha256 != asset.source_pdf_sha256
            or text_gold.source_raster_gold_sha256 != asset.gold_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid")


def _load_dataset_receipt(
    root: Path,
    relative_path: str,
    *,
    scope: _Scope,
    preflight_smoke_only: bool = False,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1 | None = None,
) -> tuple[SyntheticRasterReceiptV1, str]:
    raw = _read(
        root,
        relative_path,
        limit=_MAX_DATASET_RECEIPT_BYTES,
        code="raster_text_dataset_receipt_invalid",
    )
    try:
        receipt = SyntheticRasterReceiptV1.model_validate_json(raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_dataset_receipt_invalid") from None
    if (
        raw != _canonical_json(receipt.model_dump(mode="json", warnings=False))
        or len(receipt.assets) != 24_000
    ):
        raise SyntheticRasterTextActivationError("raster_text_dataset_receipt_invalid")
    if preflight_smoke_only:
        if scope != "smoke" or build_inputs is None:
            raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid")
        _verify_existing_dataset_smoke_materials(
            root=root,
            receipt=receipt,
            build_inputs=build_inputs,
        )
    else:
        try:
            verify_synthetic_raster_receipt_materials(
                document_preview_root=root,
                receipt=receipt,
                required_splits=("calibration" if scope == "smoke" else scope,),
            )
        except (SyntheticRasterCompilerUnavailable, ValueError):
            raise SyntheticRasterTextActivationError(
                "raster_text_dataset_material_invalid"
            ) from None
    return receipt, hashlib.sha256(raw).hexdigest()


def _load_fresh_dataset_receipts(
    root: Path,
    synthetic_raster_receipt_relative_path: str,
) -> tuple[
    SyntheticRasterReceiptV1,
    str,
    SyntheticRasterSourceReceiptV1,
    str,
]:
    if synthetic_raster_receipt_relative_path != SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    raster_raw = _read(
        root,
        synthetic_raster_receipt_relative_path,
        limit=_MAX_DATASET_RECEIPT_BYTES,
        code="raster_text_fresh_selection_invalid",
    )
    try:
        receipt = SyntheticRasterReceiptV1.model_validate_json(raster_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid") from None
    raster_sha256 = hashlib.sha256(raster_raw).hexdigest()
    if (
        raster_raw != _canonical_json(receipt.model_dump(mode="json", warnings=False))
        or len(receipt.assets) != 24_000
        or receipt.source_receipt_relative_path != SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    source_raw = _read(
        root,
        SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_selection_invalid",
    )
    try:
        source_receipt = SyntheticRasterSourceReceiptV1.model_validate_json(source_raw)
        validate_fixed_synthetic_raster_source_receipt(source_receipt)
    except (ValidationError, ValueError):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid") from None
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        source_raw != _canonical_json(source_receipt.model_dump(mode="json", warnings=False))
        or source_sha256 != receipt.source_receipt_sha256
        or source_receipt.font_manifest_sha256 != receipt.font_manifest_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    _dataset_family_splits(receipt)
    return receipt, raster_sha256, source_receipt, source_sha256


def _write_fresh_scope_selection(
    *,
    root: Path,
    selection: RasterTextFreshScopeSelectionReceipt,
) -> tuple[str, str]:
    digest = _digest_model(selection)
    relative = f"synthetic-raster-text-v1/fresh-scope-selections/{digest}.json"
    try:
        written, written_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=selection.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_selection_write_failed"
        ) from None
    if written != relative or written_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_write_failed")
    return relative, digest


def _read_fresh_scope_selection(
    *,
    root: Path,
    relative_path: str,
    expected: RasterTextFreshScopeSelectionReceiptV1,
) -> tuple[RasterTextFreshScopeSelectionReceiptV1, str]:
    expected_digest = _digest_model(expected)
    expected_relative = f"synthetic-raster-text-v1/fresh-scope-selections/{expected_digest}.json"
    if relative_path != expected_relative:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    raw = _read(
        root,
        relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_selection_invalid",
    )
    try:
        observed = RasterTextFreshScopeSelectionReceiptV1.model_validate_json(raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid") from None
    if (
        raw != _canonical_json(observed.model_dump(mode="json", warnings=False))
        or hashlib.sha256(raw).hexdigest() != expected_digest
        or observed != expected
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return observed, expected_digest


def _read_fresh_scope_selection_v2(
    *,
    root: Path,
    relative_path: str,
    expected: RasterTextFreshScopeSelectionReceiptV2,
) -> tuple[RasterTextFreshScopeSelectionReceiptV2, str]:
    expected_digest = _digest_model(expected)
    expected_relative = f"synthetic-raster-text-v1/fresh-scope-selections/{expected_digest}.json"
    if relative_path != expected_relative:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    raw = _read(
        root,
        relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_selection_invalid",
    )
    try:
        observed = RasterTextFreshScopeSelectionReceiptV2.model_validate_json(raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid") from None
    if (
        raw != _canonical_json(observed.model_dump(mode="json", warnings=False))
        or hashlib.sha256(raw).hexdigest() != expected_digest
        or observed != expected
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return observed, expected_digest


def _fresh_scope_selection_revision_from_path(
    *, root: Path, relative_path: str
) -> _FreshSelectionRevision:
    raw = _read(
        root,
        relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_selection_invalid",
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid") from None
    if not isinstance(value, dict):
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    discriminator = (value.get("schema_version"), value.get("selection_revision"))
    if discriminator == ("1.0", _FRESH_HOLDOUT_SELECTION_REVISION):
        return _FRESH_HOLDOUT_SELECTION_REVISION
    if discriminator == ("2.0", _FRESH_HOLDOUT_SELECTION_REVISION_V2):
        return _FRESH_HOLDOUT_SELECTION_REVISION_V2
    raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")


@dataclass(frozen=True, slots=True)
class _FreshScopeContext:
    receipt: SyntheticRasterReceiptV1
    receipt_sha256: str
    source_receipt: SyntheticRasterSourceReceiptV1
    source_receipt_sha256: str
    selection: RasterTextFreshScopeSelectionReceipt
    selection_relative_path: str
    selection_sha256: str
    assets: list[RasterAssetReceiptV1]


def _fresh_scope_context(
    *,
    root: Path,
    synthetic_raster_receipt_relative_path: str,
    scope: _FreshScope,
    selection_revision: _FreshSelectionRevision = _FRESH_HOLDOUT_SELECTION_REVISION,
    existing_selection_relative_path: str | None = None,
) -> _FreshScopeContext:
    receipt, receipt_sha256, source_receipt, source_receipt_sha256 = _load_fresh_dataset_receipts(
        root, synthetic_raster_receipt_relative_path
    )
    selection = _build_fresh_scope_selection_for_revision(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        source_receipt_sha256=source_receipt_sha256,
        selection_revision=selection_revision,
    )
    if isinstance(selection, RasterTextFreshScopeSelectionReceiptV2):
        parent = _build_fresh_scope_selection(
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            source_receipt_sha256=source_receipt_sha256,
        )
        if existing_selection_relative_path is None:
            parent_relative_path, parent_sha256 = _write_fresh_scope_selection(
                root=root,
                selection=parent,
            )
        else:
            parent_relative_path = selection.parent_selection_receipt_relative_path
            try:
                _parent, parent_sha256 = _read_fresh_scope_selection(
                    root=root,
                    relative_path=parent_relative_path,
                    expected=parent,
                )
            except SyntheticRasterTextActivationError:
                raise SyntheticRasterTextActivationError(
                    "raster_text_fresh_selection_invalid"
                ) from None
        if (
            parent_relative_path != selection.parent_selection_receipt_relative_path
            or parent_sha256 != selection.parent_selection_receipt_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    if existing_selection_relative_path is None:
        selection_relative_path, selection_sha256 = _write_fresh_scope_selection(
            root=root,
            selection=selection,
        )
    else:
        selection_relative_path = existing_selection_relative_path
        if isinstance(selection, RasterTextFreshScopeSelectionReceiptV1):
            selection, selection_sha256 = _read_fresh_scope_selection(
                root=root,
                relative_path=existing_selection_relative_path,
                expected=selection,
            )
        else:
            selection, selection_sha256 = _read_fresh_scope_selection_v2(
                root=root,
                relative_path=existing_selection_relative_path,
                expected=selection,
            )
    assets = _fresh_scope_assets(receipt=receipt, selection=selection, scope=scope)
    source_by_asset_id = {asset.asset_id: asset for asset in source_receipt.assets}
    for asset in assets:
        source = source_by_asset_id.get(asset.asset_id)
        if (
            source is None
            or source.family_id != asset.family_id
            or source.document_type != asset.document_type
            or source.source_pdf_sha256 != asset.source_pdf_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return _FreshScopeContext(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        source_receipt=source_receipt,
        source_receipt_sha256=source_receipt_sha256,
        selection=selection,
        selection_relative_path=selection_relative_path,
        selection_sha256=selection_sha256,
        assets=assets,
    )


def _fresh_projection_family_ids(
    *,
    selection: RasterTextFreshScopeSelectionReceipt,
    projection_kind: _FreshProjectionKind,
) -> list[str]:
    """Select a fixed, result-blind hash-ranked family subset."""

    if projection_kind == "calibration-10-family-v1":
        source = selection.fresh_calibration_family_ids
        count = _FRESH_CALIBRATION_FAMILY_COUNT
    elif projection_kind == "sealed-stage1-50-family-v1":
        source = selection.fresh_sealed_family_ids
        count = _FRESH_SEALED_STAGE1_FAMILY_COUNT
    else:
        source = selection.fresh_sealed_family_ids
        count = _FRESH_SEALED_STAGE2_FAMILY_COUNT
    if source != sorted(source) or len(source) < count:
        raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
    if projection_kind == "sealed-stage2-100-family-v1":
        # Stage 2 is the complete sealed population.  Sorting is canonical only.
        return list(source)
    ranked = sorted(
        source,
        key=lambda family_id: (
            hashlib.sha256(
                (f"{_FRESH_PROJECTION_SEED_SHA256}:{projection_kind}:{family_id}").encode("utf-8")
            ).hexdigest(),
            family_id,
        ),
    )
    # Membership follows the hash rank; wire ordering remains canonical.
    return sorted(ranked[:count])


def _fresh_projection_assets(
    *,
    context: _FreshScopeContext,
    projection: RasterTextFreshEvaluationProjectionReceiptV1,
) -> list[RasterAssetReceiptV1]:
    expected_family_ids = _fresh_projection_family_ids(
        selection=context.selection,
        projection_kind=projection.projection_kind,
    )
    expected_scope: _FreshScope = (
        "fresh-calibration"
        if projection.projection_kind == "calibration-10-family-v1"
        else "fresh-sealed"
    )
    if (
        context.selection_relative_path != projection.selection_receipt_relative_path
        or context.selection_sha256 != projection.selection_receipt_sha256
        or context.receipt_sha256 != projection.synthetic_raster_receipt_sha256
        or projection.source_scope != expected_scope
        or projection.family_ids != expected_family_ids
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
    family_ids = set(expected_family_ids)
    assets = sorted(
        (asset for asset in context.assets if asset.family_id in family_ids),
        key=lambda item: (item.asset_id, item.profile_id),
    )
    family_counts = {
        family_id: sum(asset.family_id == family_id for asset in assets) for family_id in family_ids
    }
    strata = {
        (document_type, profile_id): sum(
            asset.document_type == document_type and asset.profile_id == profile_id
            for asset in assets
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    }
    if (
        len(assets) != projection.asset_count
        or set(family_counts.values()) != {24}
        or set(strata.values()) != {projection.per_stratum_asset_count}
        or any(asset.split != "train" for asset in assets)
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
    return assets


def _fresh_stage_decision_expected_path(digest: str) -> str:
    return f"synthetic-raster-text-v1/fresh-sealed-stage-decisions/{digest}.json"


def _read_fresh_sealed_stage_decision(
    *, root: Path, relative_path: str, expected_sha256: str | None = None
) -> tuple[RasterTextFreshSealedStageDecisionReceipt, str]:
    raw = _read(
        root,
        relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_stage_decision_invalid",
    )
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        if value.get("schema_version") == "1.0":
            decision: RasterTextFreshSealedStageDecisionReceipt = (
                RasterTextFreshSealedStageDecisionReceiptV1.model_validate(value)
            )
        elif value.get("schema_version") == "2.0":
            decision = RasterTextFreshSealedStageDecisionReceiptV2.model_validate(value)
        else:
            raise ValueError
    except (json.JSONDecodeError, ValidationError, ValueError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_stage_decision_invalid"
        ) from None
    if (
        raw != _canonical_json(decision.model_dump(mode="json", warnings=False))
        or relative_path != _fresh_stage_decision_expected_path(digest)
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    return decision, digest


def _build_fresh_evaluation_projection(
    *,
    root: Path,
    context: _FreshScopeContext,
    source_gold_receipt_relative_path: str,
    source_gold_receipt_sha256: str,
    projection_kind: _FreshProjectionKind,
    parent_stage1_decision_relative_path: str | None = None,
) -> RasterTextFreshEvaluationProjectionReceiptV1:
    expected_scope: _FreshScope = (
        "fresh-calibration" if projection_kind == "calibration-10-family-v1" else "fresh-sealed"
    )
    family_ids = _fresh_projection_family_ids(
        selection=context.selection, projection_kind=projection_kind
    )
    parent_sha256: str | None = None
    if parent_stage1_decision_relative_path is not None:
        decision, parent_sha256 = _read_fresh_sealed_stage_decision(
            root=root,
            relative_path=parent_stage1_decision_relative_path,
        )
        if (
            decision.decision != "continue"
            or decision.source_gold_receipt_relative_path != source_gold_receipt_relative_path
            or decision.source_gold_receipt_sha256 != source_gold_receipt_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
    family_count = len(family_ids)
    projection = RasterTextFreshEvaluationProjectionReceiptV1(
        source_scope=expected_scope,
        projection_kind=projection_kind,
        projection_seed_sha256=_FRESH_PROJECTION_SEED_SHA256,
        synthetic_raster_receipt_sha256=context.receipt_sha256,
        selection_receipt_relative_path=context.selection_relative_path,
        selection_receipt_sha256=context.selection_sha256,
        full_source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        full_source_gold_receipt_sha256=source_gold_receipt_sha256,
        family_ids=family_ids,
        family_count=family_count,
        asset_count=family_count * 24,
        per_stratum_asset_count=family_count * 2,
        parent_stage1_decision_relative_path=(parent_stage1_decision_relative_path),
        parent_stage1_decision_sha256=parent_sha256,
    )
    _fresh_projection_assets(context=context, projection=projection)
    return projection


def _fresh_projection_expected_path(
    projection: RasterTextFreshEvaluationProjectionReceiptV1,
) -> tuple[str, str]:
    digest = _digest_model(projection)
    relative = (
        "synthetic-raster-text-v1/fresh-evaluation-projections/"
        f"{projection.source_scope}/{digest}.json"
    )
    return relative, digest


def _write_fresh_evaluation_projection(
    *,
    root: Path,
    context: _FreshScopeContext,
    source_gold_receipt_relative_path: str,
    source_gold_receipt_sha256: str,
    projection_kind: _FreshProjectionKind,
    parent_stage1_decision_relative_path: str | None = None,
) -> tuple[RasterTextFreshEvaluationProjectionReceiptV1, str, str]:
    projection = _build_fresh_evaluation_projection(
        root=root,
        context=context,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_gold_receipt_sha256,
        projection_kind=projection_kind,
        parent_stage1_decision_relative_path=(parent_stage1_decision_relative_path),
    )
    relative, digest = _fresh_projection_expected_path(projection)
    try:
        written, written_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=projection.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_projection_write_failed"
        ) from None
    if written != relative or written_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_projection_write_failed")
    return projection, relative, digest


def _read_fresh_evaluation_projection(
    *,
    root: Path,
    context: _FreshScopeContext,
    relative_path: str,
    expected_sha256: str,
    source_gold_receipt_relative_path: str,
    source_gold_receipt_sha256: str,
) -> RasterTextFreshEvaluationProjectionReceiptV1:
    raw = _read(
        root,
        relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_projection_invalid",
    )
    try:
        observed = RasterTextFreshEvaluationProjectionReceiptV1.model_validate_json(raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid") from None
    expected = _build_fresh_evaluation_projection(
        root=root,
        context=context,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_gold_receipt_sha256,
        projection_kind=observed.projection_kind,
        parent_stage1_decision_relative_path=(observed.parent_stage1_decision_relative_path),
    )
    expected_relative, expected_digest = _fresh_projection_expected_path(expected)
    if (
        raw != _canonical_json(observed.model_dump(mode="json", warnings=False))
        or relative_path != expected_relative
        or expected_sha256 != expected_digest
        or hashlib.sha256(raw).hexdigest() != expected_digest
        or observed != expected
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
    return observed


def write_fresh_raster_text_scope_selection_receipt(
    *,
    document_preview_root: Path,
    synthetic_raster_receipt_relative_path: str = (SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
    selection_revision: _FreshSelectionRevision = _FRESH_HOLDOUT_SELECTION_REVISION,
) -> tuple[str, str]:
    """Create or verify one fixed-generation source-bound train holdout."""

    root = _root(document_preview_root)
    receipt, receipt_sha256, _source_receipt, source_receipt_sha256 = _load_fresh_dataset_receipts(
        root, synthetic_raster_receipt_relative_path
    )
    selection = _build_fresh_scope_selection_for_revision(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        source_receipt_sha256=source_receipt_sha256,
        selection_revision=selection_revision,
    )
    if isinstance(selection, RasterTextFreshScopeSelectionReceiptV2):
        parent = _build_fresh_scope_selection(
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            source_receipt_sha256=source_receipt_sha256,
        )
        parent_relative_path, parent_sha256 = _write_fresh_scope_selection(
            root=root,
            selection=parent,
        )
        if (
            parent_relative_path != selection.parent_selection_receipt_relative_path
            or parent_sha256 != selection.parent_selection_receipt_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    return _write_fresh_scope_selection(root=root, selection=selection)


def _gold_for_asset(root: Path, asset: RasterAssetReceiptV1) -> RasterTextGoldArtifactV1:
    raw = _read(
        root,
        asset.gold_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_source_gold_invalid",
    )
    try:
        gold = RasterGoldArtifactV1.model_validate_json(raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_source_gold_invalid") from None
    if (
        raw != _canonical_json(gold.model_dump(mode="json", warnings=False))
        or hashlib.sha256(raw).hexdigest() != asset.gold_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_source_gold_invalid")
    try:
        return source_only_text_gold_from_raster_gold(gold)
    except (StopIteration, ValueError):
        raise SyntheticRasterTextActivationError("raster_text_source_gold_invalid") from None


def write_source_attested_raster_text_gold_receipt(
    *,
    document_preview_root: Path,
    synthetic_raster_receipt_relative_path: str,
    scope: _Scope,
    preflight_smoke_only: bool = False,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1 | None = None,
) -> tuple[str, str]:
    """Write a text-free Gold receipt re-derived only from raster source Gold."""

    root = _root(document_preview_root)
    receipt, receipt_sha256 = _load_dataset_receipt(
        root,
        synthetic_raster_receipt_relative_path,
        scope=scope,
        preflight_smoke_only=preflight_smoke_only,
        build_inputs=build_inputs,
    )
    bindings: list[RasterTextSourceGoldBindingV1] = []
    for asset in _scope_assets(receipt, scope):
        text_gold = _gold_for_asset(root, asset)
        bindings.append(
            RasterTextSourceGoldBindingV1(
                asset_id=asset.asset_id,
                profile_id=asset.profile_id,
                family_id=asset.family_id,
                document_type=asset.document_type,
                source_pdf_sha256=asset.source_pdf_sha256,
                image_sha256=asset.image_sha256,
                source_raster_gold_relative_path=asset.gold_relative_path,
                source_raster_gold_sha256=asset.gold_sha256,
                derived_text_gold_sha256=_digest_model(text_gold),
            )
        )
    artifact = RasterTextSourceGoldReceiptV1(
        scope=scope, synthetic_raster_receipt_sha256=receipt_sha256, bindings=bindings
    )
    digest = _digest_model(artifact)
    relative = f"synthetic-raster-text-v1/source-gold-receipts/{scope}/{digest}.json"
    try:
        written, written_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError("raster_text_source_gold_write_failed") from None
    if written != relative or written_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_source_gold_write_failed")
    return relative, digest


def _write_source_attested_fresh_raster_text_gold_receipt(
    *,
    root: Path,
    context: _FreshScopeContext,
    scope: _FreshScope,
) -> tuple[str, str]:
    bindings: list[RasterTextSourceGoldBindingV1] = []
    for asset in context.assets:
        text_gold = _gold_for_asset(root, asset)
        bindings.append(
            RasterTextSourceGoldBindingV1(
                asset_id=asset.asset_id,
                profile_id=asset.profile_id,
                family_id=asset.family_id,
                document_type=asset.document_type,
                source_pdf_sha256=asset.source_pdf_sha256,
                image_sha256=asset.image_sha256,
                source_raster_gold_relative_path=asset.gold_relative_path,
                source_raster_gold_sha256=asset.gold_sha256,
                derived_text_gold_sha256=_digest_model(text_gold),
            )
        )
    artifact = RasterTextFreshSourceGoldReceiptV1(
        scope=scope,
        synthetic_raster_receipt_sha256=context.receipt_sha256,
        source_receipt_sha256=context.source_receipt_sha256,
        selection_receipt_relative_path=context.selection_relative_path,
        selection_receipt_sha256=context.selection_sha256,
        bindings=bindings,
    )
    digest = _digest_model(artifact)
    relative = f"synthetic-raster-text-v1/fresh-source-gold-receipts/{scope}/{digest}.json"
    try:
        written, written_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_source_gold_write_failed"
        ) from None
    if written != relative or written_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_source_gold_write_failed")
    return relative, digest


def write_source_attested_fresh_raster_text_gold_receipt(
    *,
    document_preview_root: Path,
    synthetic_raster_receipt_relative_path: str,
    scope: _FreshScope,
    selection_revision: _FreshSelectionRevision = _FRESH_HOLDOUT_SELECTION_REVISION,
) -> tuple[str, str]:
    """Write source-only Gold for one deterministically selected fresh scope."""

    root = _root(document_preview_root)
    context = _fresh_scope_context(
        root=root,
        synthetic_raster_receipt_relative_path=(synthetic_raster_receipt_relative_path),
        scope=scope,
        selection_revision=selection_revision,
    )
    return _write_source_attested_fresh_raster_text_gold_receipt(
        root=root,
        context=context,
        scope=scope,
    )


_RUNNER_IMPLEMENTATION_FILES: tuple[tuple[str, Path], ...] = (
    # This is deliberately a fixed, reviewable manifest rather than an
    # import walk or a glob.  Every file below can change candidate input,
    # source-only Gold, Gate/floor calculation, or the active text runtime.
    ("evaluation/synthetic_raster_text_activation.py", Path(__file__).resolve()),
    (
        "evaluation/synthetic_raster_text.py",
        Path(__file__).resolve().with_name("synthetic_raster_text.py"),
    ),
    (
        "evaluation/synthetic_raster.py",
        Path(__file__).resolve().with_name("synthetic_raster.py"),
    ),
    (
        "evaluation/source_attested_synthetic_pdf_truth.py",
        Path(__file__).resolve().with_name("source_attested_synthetic_pdf_truth.py"),
    ),
    (
        "evaluation/raster_offline_bundle.py",
        Path(__file__).resolve().with_name("raster_offline_bundle.py"),
    ),
    (
        "evaluation/raster_text_direct_paddle_bundle.py",
        Path(__file__).resolve().with_name("raster_text_direct_paddle_bundle.py"),
    ),
    (
        "evaluation/raster_text_mobile_paddle_bundle.py",
        Path(__file__).resolve().with_name("raster_text_mobile_paddle_bundle.py"),
    ),
    (
        "service/raster_text_extraction.py",
        Path(__file__).resolve().parents[1] / "service" / "raster_text_extraction.py",
    ),
    (
        "service/raster_text_preview_runtime.py",
        Path(__file__).resolve().parents[1] / "service" / "raster_text_preview_runtime.py",
    ),
    (
        "service/raster_text_direct_paddle_sidecar.py",
        Path(__file__).resolve().parents[1] / "service" / "raster_text_direct_paddle_sidecar.py",
    ),
    (
        "service/raster_document_preview_runtime.py",
        Path(__file__).resolve().parents[1] / "service" / "raster_document_preview_runtime.py",
    ),
    (
        "service/document_input.py",
        Path(__file__).resolve().parents[1] / "service" / "document_input.py",
    ),
    (
        "schemas/raster_text_extraction.py",
        Path(__file__).resolve().parents[1] / "schemas" / "raster_text_extraction.py",
    ),
)


def _runner_implementation_sha256() -> str:
    digest = hashlib.sha256()
    try:
        for relative_path, file in _RUNNER_IMPLEMENTATION_FILES:
            if file.is_symlink() or not file.is_file():
                raise OSError
            raw = file.read_bytes()
            if not raw or len(raw) > 8 * 1024 * 1024:
                raise OSError
            digest.update(relative_path.encode("ascii"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(raw).digest())
    except OSError:
        raise SyntheticRasterTextActivationError(
            "raster_text_runner_implementation_unavailable"
        ) from None
    return digest.hexdigest()


def _bl_repair_audit_runner_implementation_sha256() -> str:
    """Bind audit execution to both the OCR runner and its source-Gold compiler."""

    digest = hashlib.sha256()
    digest.update(b"source-attested-bl-repair-audit-runner-v1\0")
    digest.update(bytes.fromhex(_runner_implementation_sha256()))
    audit_module = Path(__file__).resolve().with_name("source_attested_bl_repair_audit.py")
    try:
        if audit_module.is_symlink() or not audit_module.is_file():
            raise OSError
        raw = audit_module.read_bytes()
        if not raw or len(raw) > 8 * 1024 * 1024:
            raise OSError
    except OSError:
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_runner_implementation_unavailable"
        ) from None
    digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def _raster_text_request_id(asset: RasterAssetReceiptV1) -> str:
    return f"raster-text-{asset.profile_id}-{asset.image_sha256[:24]}"


def _candidate_checkpoint_identity(
    *,
    scope: _CandidateScope,
    synthetic_raster_receipt_sha256: str,
    source_gold_receipt_sha256: str,
    selection_receipt_sha256: str | None = None,
    runtime_configuration_sha256: str,
    raster_text_profile_revision: str,
    extractor_revision: str,
    runner_implementation_sha256: str,
    asset: RasterAssetReceiptV1,
) -> RasterTextCandidateCheckpointIdentityV1 | RasterTextFreshCandidateCheckpointIdentityV1:
    if scope in ("fresh-calibration", "fresh-sealed"):
        if selection_receipt_sha256 is None:
            raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
        return RasterTextFreshCandidateCheckpointIdentityV1(
            scope=scope,
            synthetic_raster_receipt_sha256=synthetic_raster_receipt_sha256,
            source_gold_receipt_sha256=source_gold_receipt_sha256,
            selection_receipt_sha256=selection_receipt_sha256,
            runtime_configuration_sha256=runtime_configuration_sha256,
            raster_text_profile_revision=raster_text_profile_revision,
            extractor_revision=extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            asset_id=asset.asset_id,
            profile_id=asset.profile_id,
            image_sha256=asset.image_sha256,
            gold_sha256=asset.gold_sha256,
        )
    if selection_receipt_sha256 is not None:
        raise SyntheticRasterTextActivationError("raster_text_candidate_checkpoint_invalid")
    return RasterTextCandidateCheckpointIdentityV1(
        scope=cast(_Scope, scope),
        synthetic_raster_receipt_sha256=synthetic_raster_receipt_sha256,
        source_gold_receipt_sha256=source_gold_receipt_sha256,
        runtime_configuration_sha256=runtime_configuration_sha256,
        raster_text_profile_revision=raster_text_profile_revision,
        extractor_revision=extractor_revision,
        runner_implementation_sha256=runner_implementation_sha256,
        asset_id=asset.asset_id,
        profile_id=asset.profile_id,
        image_sha256=asset.image_sha256,
        gold_sha256=asset.gold_sha256,
    )


def _candidate_checkpoint_relative_path(
    identity: (
        RasterTextCandidateCheckpointIdentityV1 | RasterTextFreshCandidateCheckpointIdentityV1
    ),
) -> tuple[str, str]:
    identity_sha256 = _digest_model(identity)
    return (
        "synthetic-raster-text-v1/candidate-checkpoints/"
        f"{identity.scope}/{identity_sha256[:2]}/{identity_sha256}.json",
        identity_sha256,
    )


def _candidate_checkpoint_path_if_present(root: Path, relative_path: str) -> Path | None:
    """Return one bounded regular checkpoint path without following symlinks."""

    try:
        parts = _relative_path(relative_path).parts
        candidate = root
        observed_stat = None
        for index, part in enumerate(parts):
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
            try:
                observed_stat = candidate.stat()
            except FileNotFoundError:
                return None
            if index < len(parts) - 1 and not stat.S_ISDIR(observed_stat.st_mode):
                raise OSError
        if (
            observed_stat is None
            or not stat.S_ISREG(observed_stat.st_mode)
            or observed_stat.st_size <= 0
            or observed_stat.st_size > _MAX_CANDIDATE_CHECKPOINT_BYTES
        ):
            raise OSError
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if resolved.is_symlink() or not resolved.is_file():
            raise OSError
        return resolved
    except (OSError, ValueError, SyntheticRasterTextActivationError):
        raise SyntheticRasterTextActivationError(
            "raster_text_candidate_checkpoint_invalid"
        ) from None


def _validate_candidate_provenance(
    *,
    candidate: RasterTextExtractionResultV1,
    asset: RasterAssetReceiptV1,
    raster_text_profile_revision: str,
    extractor_revision: str,
    code: str,
) -> None:
    if (
        candidate.document_asset_id != asset.asset_id
        or candidate.extraction_request_id != _raster_text_request_id(asset)
        or candidate.source_sha256 != asset.image_sha256
        or candidate.media_type != asset.media_type
        or candidate.byte_size != asset.byte_size
        or candidate.raster_text_profile_revision != raster_text_profile_revision
        or candidate.extractor_revision != extractor_revision
        or candidate.status == "unavailable"
        or candidate.canonical_text is None
    ):
        raise SyntheticRasterTextActivationError(code)


def _candidate_input_equivalence_key(
    asset: RasterAssetReceiptV1,
) -> tuple[str, str, str, int]:
    """Identify byte-identical OCR inputs without consulting Gold or output.

    The synthetic source set can contain clean/error records whose rendered
    bytes are exactly identical.  OCR depends on the bytes, media decoder and
    immutable runtime profile, not on the logical asset id or its Gold.  The
    runtime profile is already bound outside this key by the checkpoint
    identity, while ``profile_id`` keeps the rendering contract explicit.
    """

    return (
        asset.profile_id,
        asset.media_type,
        asset.image_sha256,
        asset.byte_size,
    )


def _candidate_content_sha256(candidate: RasterTextExtractionResultV1) -> str:
    """Hash provider content while excluding request-specific identifiers."""

    return hashlib.sha256(
        _canonical_json(
            candidate.model_dump(
                mode="json",
                warnings=False,
                exclude={"document_asset_id", "extraction_request_id"},
            )
        )
    ).hexdigest()


def _rebind_equivalent_candidate_observation(
    *,
    source: RasterTextCandidateObservationV1,
    asset: RasterAssetReceiptV1,
    raster_text_profile_revision: str,
    extractor_revision: str,
) -> RasterTextCandidateObservationV1:
    """Bind a byte-identical result to another logical synthetic asset.

    This is evaluation-only memoization.  Every public result field is
    revalidated after rebinding, and the original measured latency is copied
    conservatively instead of claiming a near-zero cache hit.
    """

    try:
        candidate = RasterTextExtractionResultV1.model_validate(
            source.candidate.model_dump(
                mode="json",
                warnings=False,
                exclude={"document_asset_id", "extraction_request_id"},
            )
            | {
                "document_asset_id": asset.asset_id,
                "extraction_request_id": _raster_text_request_id(asset),
            }
        )
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_candidate_execution_failed") from None
    _validate_candidate_provenance(
        candidate=candidate,
        asset=asset,
        raster_text_profile_revision=raster_text_profile_revision,
        extractor_revision=extractor_revision,
        code="raster_text_candidate_execution_failed",
    )
    if _candidate_content_sha256(candidate) != _candidate_content_sha256(source.candidate):
        raise SyntheticRasterTextActivationError("raster_text_candidate_execution_failed")
    return RasterTextCandidateObservationV1(
        asset_id=asset.asset_id,
        profile_id=asset.profile_id,
        elapsed_seconds=source.elapsed_seconds,
        candidate=candidate,
    )


def _read_candidate_checkpoint(
    *,
    root: Path,
    path: Path,
    expected_identity: (
        RasterTextCandidateCheckpointIdentityV1 | RasterTextFreshCandidateCheckpointIdentityV1
    ),
    asset: RasterAssetReceiptV1,
) -> RasterTextCandidateObservationV1:
    try:
        expected_relative, _identity_sha256 = _candidate_checkpoint_relative_path(expected_identity)
        expected_path = root.joinpath(*PurePosixPath(expected_relative).parts)
        if path != expected_path:
            raise OSError
        before = path.stat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CANDIDATE_CHECKPOINT_BYTES
        ):
            raise OSError
        raw = path.read_bytes()
        after = path.stat()
        if (
            path.is_symlink()
            or before.st_size != len(raw)
            or after.st_size != len(raw)
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
        ):
            raise OSError
        checkpoint: RasterTextCandidateCheckpointV1 | RasterTextFreshCandidateCheckpointV1
        if isinstance(expected_identity, RasterTextFreshCandidateCheckpointIdentityV1):
            checkpoint = RasterTextFreshCandidateCheckpointV1.model_validate_json(raw)
        else:
            checkpoint = RasterTextCandidateCheckpointV1.model_validate_json(raw)
        if (
            raw != _canonical_json(checkpoint.model_dump(mode="json", warnings=False))
            or checkpoint.identity != expected_identity
        ):
            raise ValueError
        _validate_candidate_provenance(
            candidate=checkpoint.observation.candidate,
            asset=asset,
            raster_text_profile_revision=expected_identity.raster_text_profile_revision,
            extractor_revision=expected_identity.extractor_revision,
            code="raster_text_candidate_checkpoint_invalid",
        )
        return checkpoint.observation
    except SyntheticRasterTextActivationError:
        raise
    except (OSError, ValidationError, ValueError):
        raise SyntheticRasterTextActivationError(
            "raster_text_candidate_checkpoint_invalid"
        ) from None


def _write_candidate_checkpoint(
    *,
    root: Path,
    identity: (
        RasterTextCandidateCheckpointIdentityV1 | RasterTextFreshCandidateCheckpointIdentityV1
    ),
    observation: RasterTextCandidateObservationV1,
) -> None:
    relative_path, identity_sha256 = _candidate_checkpoint_relative_path(identity)
    checkpoint: RasterTextCandidateCheckpointV1 | RasterTextFreshCandidateCheckpointV1
    if isinstance(identity, RasterTextFreshCandidateCheckpointIdentityV1):
        checkpoint = RasterTextFreshCandidateCheckpointV1(
            identity=identity,
            identity_sha256=identity_sha256,
            elapsed_seconds=observation.elapsed_seconds,
            observation_sha256=_digest_model(observation),
            observation=observation,
        )
    else:
        checkpoint = RasterTextCandidateCheckpointV1(
            identity=identity,
            identity_sha256=identity_sha256,
            elapsed_seconds=observation.elapsed_seconds,
            observation_sha256=_digest_model(observation),
            observation=observation,
        )
    checkpoint_value = checkpoint.model_dump(mode="json", warnings=False)
    checkpoint_payload = _canonical_json(checkpoint_value)
    if not checkpoint_payload or len(checkpoint_payload) > _MAX_CANDIDATE_CHECKPOINT_BYTES:
        raise SyntheticRasterTextActivationError("raster_text_candidate_checkpoint_write_failed")
    try:
        written, digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative_path,
            value=checkpoint_value,
        )
    except RuntimeError as error:
        if getattr(error, "code", None) == "offline_bundle_output_conflict":
            raise SyntheticRasterTextActivationError(
                "raster_text_candidate_checkpoint_invalid"
            ) from None
        raise SyntheticRasterTextActivationError(
            "raster_text_candidate_checkpoint_write_failed"
        ) from None
    except (OSError, ValueError):
        raise SyntheticRasterTextActivationError(
            "raster_text_candidate_checkpoint_write_failed"
        ) from None
    if written != relative_path or digest != hashlib.sha256(checkpoint_payload).hexdigest():
        raise SyntheticRasterTextActivationError("raster_text_candidate_checkpoint_write_failed")


def _verified_asset(root: Path, asset: RasterAssetReceiptV1, *, pdfium_version: str) -> bytes:
    raw = _read(
        root,
        asset.asset_relative_path,
        limit=_MAX_ASSET_BYTES,
        code="raster_text_candidate_asset_invalid",
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
        raise SyntheticRasterTextActivationError("raster_text_candidate_asset_invalid") from None
    if (
        observed.content_sha256 != asset.image_sha256
        or observed.media_type != asset.media_type
        or observed.byte_size != asset.byte_size
    ):
        raise SyntheticRasterTextActivationError("raster_text_candidate_asset_invalid")
    return raw


def _raster_text_input_for_asset(
    *,
    asset: RasterAssetReceiptV1,
    raw: bytes,
    raster_text_profile_revision: str,
) -> RasterTextInputV1:
    """Build the one deterministic request identity used by eval and service.

    Exact equality must cover all public result fields, including request
    provenance.  Both protocols therefore receive the same strict metadata
    for a given immutable raster asset.
    """

    return RasterTextInputV1(
        metadata=RasterTextExtractionRequestMetadataV1(
            document_asset_id=asset.asset_id,
            extraction_request_id=_raster_text_request_id(asset),
            source_sha256=asset.image_sha256,
            media_type=asset.media_type,
            raster_text_profile_revision=raster_text_profile_revision,
        ),
        byte_size=len(raw),
        content=raw,
    )


def _bl_repair_audit_request_id(binding: BlRepairAuditRasterBindingV1) -> str:
    return f"bl-repair-audit-{binding.profile_id}-{binding.image_sha256[:24]}"


def _is_exact_bl_repair_audit_profile(
    profile: RasterTextOfflineProfileV19,
) -> bool:
    """Allow only the two independently audited runtime tuples.

    V20 inherits V19's bounded CPU controls, so an ``isinstance`` check alone
    would accidentally admit a future subclass or a V19/V20 engine cross
    pairing.  Exact concrete type, schema and engine checks keep old V19b
    evidence valid while making the one-pass V20 lane fail closed.
    """

    return (
        type(profile) is RasterTextOfflineProfileV19
        and profile.schema_version == "19.0"
        and profile.engine_kind == _V19B_ENGINE_KIND
    ) or (
        type(profile) is RasterTextOfflineProfileV20
        and profile.schema_version == "20.0"
        and profile.engine_kind == _V20_ENGINE_KIND
    )


def _bl_repair_audit_checkpoint_identity(
    *,
    receipt_sha256: str,
    runtime_configuration_sha256: str,
    profile: RasterTextOfflineProfileV19,
    runner_implementation_sha256: str,
    binding: BlRepairAuditRasterBindingV1,
) -> BlRepairAuditCandidateCheckpointIdentityV1:
    return BlRepairAuditCandidateCheckpointIdentityV1(
        dataset_receipt_sha256=receipt_sha256,
        runtime_configuration_sha256=runtime_configuration_sha256,
        raster_text_profile_revision=profile.raster_text_profile_revision,
        extractor_revision=profile.extractor_revision,
        runner_implementation_sha256=runner_implementation_sha256,
        asset_id=binding.asset_id,
        family_id=binding.family_id,
        fixture_kind=binding.fixture_kind,
        profile_id=binding.profile_id,
        source_pdf_sha256=binding.source_pdf_sha256,
        image_sha256=binding.image_sha256,
        gold_sha256=binding.gold_sha256,
        media_type=binding.media_type,
        byte_size=binding.byte_size,
    )


def _bl_repair_audit_checkpoint_relative_path(
    identity: BlRepairAuditCandidateCheckpointIdentityV1,
) -> tuple[str, str]:
    identity_sha256 = _digest_model(identity)
    return (
        f"{_BL_REPAIR_AUDIT_CHECKPOINT_DIRECTORY}/{identity_sha256[:2]}/{identity_sha256}.json",
        identity_sha256,
    )


def _bl_repair_audit_checkpoint_path_if_present(*, root: Path, relative_path: str) -> Path | None:
    try:
        return _candidate_checkpoint_path_if_present(root, relative_path)
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_checkpoint_invalid"
        ) from None


def _verify_bl_repair_audit_binding_material(
    *,
    root: Path,
    receipt: BlRepairAuditReceiptV1,
    binding: BlRepairAuditRasterBindingV1,
) -> bytes:
    """Rehash and decode one audit input and rehash its source-derived Gold."""

    raw = _read(
        root,
        binding.asset_relative_path,
        limit=_MAX_ASSET_BYTES,
        code="bl_repair_audit_material_invalid",
    )
    gold_raw = _read(
        root,
        binding.gold_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="bl_repair_audit_material_invalid",
    )
    if (
        hashlib.sha256(raw).hexdigest() != binding.image_sha256
        or len(raw) != binding.byte_size
        or hashlib.sha256(gold_raw).hexdigest() != binding.gold_sha256
    ):
        raise SyntheticRasterTextActivationError("bl_repair_audit_material_invalid")
    try:
        observed = verify_rendered_raster_output(
            RenderedRasterBytesV1(
                asset_id=binding.asset_id,
                profile_id=binding.profile_id,
                content=raw,
                width=binding.width,
                height=binding.height,
            ),
            pdfium_version=receipt.pdfium_version,
        )
        gold = RasterGoldArtifactV1.model_validate_json(gold_raw)
    except (SyntheticRasterCompilerUnavailable, ValidationError, ValueError):
        raise SyntheticRasterTextActivationError("bl_repair_audit_material_invalid") from None
    if (
        observed.content_sha256 != binding.image_sha256
        or observed.media_type != binding.media_type
        or observed.byte_size != binding.byte_size
        or gold.asset_id != binding.asset_id
        or gold.family_id != binding.family_id
        or gold.profile_id != binding.profile_id
        or gold.source_pdf_sha256 != binding.source_pdf_sha256
    ):
        raise SyntheticRasterTextActivationError("bl_repair_audit_material_invalid")
    return raw


def _bl_repair_audit_input(
    *,
    binding: BlRepairAuditRasterBindingV1,
    raw: bytes,
    raster_text_profile_revision: str,
) -> RasterTextInputV1:
    return RasterTextInputV1(
        metadata=RasterTextExtractionRequestMetadataV1(
            document_asset_id=binding.asset_id,
            extraction_request_id=_bl_repair_audit_request_id(binding),
            source_sha256=binding.image_sha256,
            media_type=binding.media_type,
            raster_text_profile_revision=raster_text_profile_revision,
        ),
        byte_size=len(raw),
        content=raw,
    )


def _validate_bl_repair_audit_candidate(
    *,
    candidate: RasterTextExtractionResultV1,
    binding: BlRepairAuditRasterBindingV1,
    profile: RasterTextOfflineProfileV19,
    code: str,
) -> None:
    if (
        candidate.document_asset_id != binding.asset_id
        or candidate.extraction_request_id != _bl_repair_audit_request_id(binding)
        or candidate.source_sha256 != binding.image_sha256
        or candidate.media_type != binding.media_type
        or candidate.byte_size != binding.byte_size
        or candidate.raster_text_profile_revision != profile.raster_text_profile_revision
        or candidate.extractor_revision != profile.extractor_revision
        or candidate.status == "unavailable"
        or candidate.canonical_text is None
    ):
        raise SyntheticRasterTextActivationError(code)


def _bl_repair_audit_candidate_content_sha256(
    candidate: RasterTextExtractionResultV1,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            candidate.model_dump(
                mode="json",
                warnings=False,
                exclude={"document_asset_id", "extraction_request_id"},
            )
        )
    ).hexdigest()


def _rebind_bl_repair_audit_candidate(
    *,
    source: RasterTextExtractionResultV1,
    binding: BlRepairAuditRasterBindingV1,
    profile: RasterTextOfflineProfileV19,
) -> RasterTextExtractionResultV1:
    try:
        candidate = RasterTextExtractionResultV1.model_validate(
            source.model_dump(
                mode="json",
                warnings=False,
                exclude={"document_asset_id", "extraction_request_id"},
            )
            | {
                "document_asset_id": binding.asset_id,
                "extraction_request_id": _bl_repair_audit_request_id(binding),
            }
        )
    except ValidationError:
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_execution_failed"
        ) from None
    _validate_bl_repair_audit_candidate(
        candidate=candidate,
        binding=binding,
        profile=profile,
        code="bl_repair_audit_candidate_execution_failed",
    )
    if _bl_repair_audit_candidate_content_sha256(
        candidate
    ) != _bl_repair_audit_candidate_content_sha256(source):
        raise SyntheticRasterTextActivationError("bl_repair_audit_candidate_execution_failed")
    return candidate


def _read_bl_repair_audit_checkpoint(
    *,
    root: Path,
    path: Path,
    expected_identity: BlRepairAuditCandidateCheckpointIdentityV1,
    binding: BlRepairAuditRasterBindingV1,
    profile: RasterTextOfflineProfileV19,
) -> BlRepairAuditCandidateCheckpointV1:
    try:
        expected_relative, _identity_sha256 = _bl_repair_audit_checkpoint_relative_path(
            expected_identity
        )
        expected_path = root.joinpath(*PurePosixPath(expected_relative).parts)
        before = path.stat()
        if (
            path != expected_path
            or path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CANDIDATE_CHECKPOINT_BYTES
        ):
            raise OSError
        raw = path.read_bytes()
        after = path.stat()
        if (
            path.is_symlink()
            or before.st_size != len(raw)
            or after.st_size != len(raw)
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
        ):
            raise OSError
        checkpoint = BlRepairAuditCandidateCheckpointV1.model_validate_json(raw)
        if (
            raw != _canonical_json(checkpoint.model_dump(mode="json", warnings=False))
            or checkpoint.identity != expected_identity
        ):
            raise ValueError
        _validate_bl_repair_audit_candidate(
            candidate=checkpoint.candidate,
            binding=binding,
            profile=profile,
            code="bl_repair_audit_candidate_checkpoint_invalid",
        )
        return checkpoint
    except SyntheticRasterTextActivationError:
        raise
    except (OSError, ValidationError, ValueError):
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_checkpoint_invalid"
        ) from None


def _write_bl_repair_audit_checkpoint(
    *,
    root: Path,
    identity: BlRepairAuditCandidateCheckpointIdentityV1,
    elapsed_seconds: float,
    candidate: RasterTextExtractionResultV1,
) -> None:
    relative_path, identity_sha256 = _bl_repair_audit_checkpoint_relative_path(identity)
    checkpoint = BlRepairAuditCandidateCheckpointV1(
        identity=identity,
        identity_sha256=identity_sha256,
        elapsed_seconds=elapsed_seconds,
        candidate_sha256=_digest_model(candidate),
        candidate=candidate,
    )
    value = checkpoint.model_dump(mode="json", warnings=False)
    payload = _canonical_json(value)
    if not payload or len(payload) > _MAX_CANDIDATE_CHECKPOINT_BYTES:
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_checkpoint_write_failed"
        )
    try:
        written, digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative_path,
            value=value,
        )
    except RuntimeError as error:
        if getattr(error, "code", None) == "offline_bundle_output_conflict":
            raise SyntheticRasterTextActivationError(
                "bl_repair_audit_candidate_checkpoint_invalid"
            ) from None
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_checkpoint_write_failed"
        ) from None
    except (OSError, ValueError):
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_checkpoint_write_failed"
        ) from None
    if written != relative_path or digest != hashlib.sha256(payload).hexdigest():
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_checkpoint_write_failed"
        )


def _bl_repair_audit_equivalence_key(
    binding: BlRepairAuditRasterBindingV1,
) -> tuple[str, str, str, int]:
    return (
        binding.profile_id,
        binding.media_type,
        binding.image_sha256,
        binding.byte_size,
    )


def _run_bl_repair_audit_observations(
    *,
    root: Path,
    receipt: BlRepairAuditReceiptV1,
    receipt_sha256: str,
    runtime_configuration_sha256: str,
    profile: RasterTextOfflineProfileV19,
    engine: MediaAwareRasterTextEngine,
    bindings: Sequence[BlRepairAuditRasterBindingV1],
) -> list[RasterTextExtractionResultV1]:
    """Run or resume one audit with a single persistent warm child session."""

    provider = getattr(engine, "provider", None)
    open_session = getattr(provider, "open_evaluation_session", None)
    input_adapter = getattr(engine, "input_adapter", None)
    if not callable(open_session) or input_adapter is None:
        raise SyntheticRasterTextActivationError("bl_repair_audit_candidate_runtime_unavailable")
    runner_sha256 = _bl_repair_audit_runner_implementation_sha256()
    candidates_by_key: dict[tuple[str, str], RasterTextExtractionResultV1] = {}
    equivalent_candidates: dict[
        tuple[str, str, str, int], tuple[RasterTextExtractionResultV1, float]
    ] = {}
    missing: list[
        tuple[BlRepairAuditRasterBindingV1, BlRepairAuditCandidateCheckpointIdentityV1]
    ] = []
    for binding in bindings:
        _verify_bl_repair_audit_binding_material(
            root=root,
            receipt=receipt,
            binding=binding,
        )
        identity = _bl_repair_audit_checkpoint_identity(
            receipt_sha256=receipt_sha256,
            runtime_configuration_sha256=runtime_configuration_sha256,
            profile=profile,
            runner_implementation_sha256=runner_sha256,
            binding=binding,
        )
        relative_path, _identity_sha256 = _bl_repair_audit_checkpoint_relative_path(identity)
        checkpoint_path = _bl_repair_audit_checkpoint_path_if_present(
            root=root, relative_path=relative_path
        )
        if checkpoint_path is None:
            missing.append((binding, identity))
            continue
        checkpoint = _read_bl_repair_audit_checkpoint(
            root=root,
            path=checkpoint_path,
            expected_identity=identity,
            binding=binding,
            profile=profile,
        )
        equivalent_key = _bl_repair_audit_equivalence_key(binding)
        equivalent = equivalent_candidates.get(equivalent_key)
        if equivalent is not None and _bl_repair_audit_candidate_content_sha256(
            equivalent[0]
        ) != _bl_repair_audit_candidate_content_sha256(checkpoint.candidate):
            raise SyntheticRasterTextActivationError("bl_repair_audit_candidate_checkpoint_invalid")
        equivalent_candidates.setdefault(
            equivalent_key, (checkpoint.candidate, checkpoint.elapsed_seconds)
        )
        candidates_by_key[(binding.asset_id, binding.image_sha256)] = checkpoint.candidate

    missing_groups: dict[
        tuple[str, str, str, int],
        list[tuple[BlRepairAuditRasterBindingV1, BlRepairAuditCandidateCheckpointIdentityV1]],
    ] = {}
    for binding, identity in missing:
        key = _bl_repair_audit_equivalence_key(binding)
        equivalent = equivalent_candidates.get(key)
        if equivalent is None:
            missing_groups.setdefault(key, []).append((binding, identity))
            continue
        candidate = _rebind_bl_repair_audit_candidate(
            source=equivalent[0], binding=binding, profile=profile
        )
        _write_bl_repair_audit_checkpoint(
            root=root,
            identity=identity,
            elapsed_seconds=equivalent[1],
            candidate=candidate,
        )
        candidates_by_key[(binding.asset_id, binding.image_sha256)] = candidate

    try:
        if missing_groups:
            with open_session(parent_materials_verified=True) as session:
                evaluation_engine = MediaAwareRasterTextEngine(
                    input_adapter=input_adapter,
                    provider=session,
                )
                for key, group in missing_groups.items():
                    binding, identity = group[0]
                    raw = _verify_bl_repair_audit_binding_material(
                        root=root,
                        receipt=receipt,
                        binding=binding,
                    )
                    started = time.monotonic()
                    result = extract_raster_text_preview(
                        evaluation_engine,
                        raster_input=_bl_repair_audit_input(
                            binding=binding,
                            raw=raw,
                            raster_text_profile_revision=(profile.raster_text_profile_revision),
                        ),
                        structure=inspect_raster_document_structure(
                            raw,
                            binding.media_type,
                            _InputLimits(),  # type: ignore[arg-type]
                        ),
                        settings=_InputLimits(),  # type: ignore[arg-type]
                    )
                    elapsed = time.monotonic() - started
                    if elapsed <= 0 or elapsed > 30:
                        raise SyntheticRasterTextActivationError(
                            "bl_repair_audit_candidate_execution_failed"
                        )
                    _validate_bl_repair_audit_candidate(
                        candidate=result,
                        binding=binding,
                        profile=profile,
                        code="bl_repair_audit_candidate_execution_failed",
                    )
                    _write_bl_repair_audit_checkpoint(
                        root=root,
                        identity=identity,
                        elapsed_seconds=elapsed,
                        candidate=result,
                    )
                    candidates_by_key[(binding.asset_id, binding.image_sha256)] = result
                    equivalent_candidates[key] = (result, elapsed)
                    for duplicate, duplicate_identity in group[1:]:
                        _verify_bl_repair_audit_binding_material(
                            root=root,
                            receipt=receipt,
                            binding=duplicate,
                        )
                        duplicate_candidate = _rebind_bl_repair_audit_candidate(
                            source=result,
                            binding=duplicate,
                            profile=profile,
                        )
                        _write_bl_repair_audit_checkpoint(
                            root=root,
                            identity=duplicate_identity,
                            elapsed_seconds=elapsed,
                            candidate=duplicate_candidate,
                        )
                        candidates_by_key[(duplicate.asset_id, duplicate.image_sha256)] = (
                            duplicate_candidate
                        )
    except SyntheticRasterTextActivationError:
        raise
    except Exception:
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_execution_failed"
        ) from None
    if _bl_repair_audit_runner_implementation_sha256() != runner_sha256:
        raise SyntheticRasterTextActivationError("bl_repair_audit_candidate_runner_drift")
    try:
        return sorted(
            (candidates_by_key[(binding.asset_id, binding.image_sha256)] for binding in bindings),
            key=lambda item: (item.document_asset_id, item.source_sha256),
        )
    except KeyError:
        raise SyntheticRasterTextActivationError(
            "bl_repair_audit_candidate_execution_failed"
        ) from None


def _verify_bl_repair_audit_activation_evidence(
    *,
    root: Path,
    runtime_configuration_relative_path: str,
    evidence: BlRepairAuditCandidateRunV1,
) -> None:
    """Replay every audit binding, bundle, score, and exact V19b/V20 runtime."""

    try:
        receipt = verify_bl_repair_audit_dataset(document_preview_root=root)
        if (
            evidence.dataset_receipt_relative_path != BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH
            or bl_repair_audit_receipt_sha256(receipt) != evidence.dataset_receipt_sha256
        ):
            raise ValueError
        evaluation = verify_bl_repair_audit_evaluation(
            document_preview_root=root,
            evaluation_relative_path=evidence.evaluation_relative_path,
            evaluation_sha256=evidence.evaluation_sha256,
            expected_candidate_profile_revision=(evidence.raster_text_profile_revision),
            expected_runtime_configuration_sha256=(evidence.runtime_configuration_sha256),
            require_gate_passed=True,
        )
        if (
            bl_repair_audit_evaluation_sha256(evaluation) != evidence.evaluation_sha256
            or evaluation.candidate_bundle_relative_path != evidence.candidate_bundle_relative_path
            or evaluation.candidate_bundle_sha256 != evidence.candidate_bundle_sha256
        ):
            raise ValueError
        bundle = verify_bl_repair_audit_candidate_bundle(
            document_preview_root=root,
            candidate_bundle_relative_path=evidence.candidate_bundle_relative_path,
            candidate_bundle_sha256=evidence.candidate_bundle_sha256,
            expected_candidate_profile_revision=(evidence.raster_text_profile_revision),
            expected_runtime_configuration_sha256=(evidence.runtime_configuration_sha256),
        )
    except (OSError, ValidationError, ValueError):
        raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid") from None
    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or not isinstance(resolution.profile, RasterTextOfflineProfileV19)
        or not _is_exact_bl_repair_audit_profile(resolution.profile)
        or resolution.profile.raster_text_profile_revision != evidence.raster_text_profile_revision
        or resolution.profile.extractor_revision != evaluation.extractor_revision
        or resolution.runtime_configuration_sha256 != evidence.runtime_configuration_sha256
    ):
        raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid")
    runner_sha256 = _bl_repair_audit_runner_implementation_sha256()
    candidates = {(item.document_asset_id, item.source_sha256): item for item in bundle.candidates}
    if len(candidates) != len(receipt.rasters):
        raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid")
    for binding in receipt.rasters:
        identity = _bl_repair_audit_checkpoint_identity(
            receipt_sha256=evidence.dataset_receipt_sha256,
            runtime_configuration_sha256=evidence.runtime_configuration_sha256,
            profile=resolution.profile,
            runner_implementation_sha256=runner_sha256,
            binding=binding,
        )
        relative_path, _identity_sha256 = _bl_repair_audit_checkpoint_relative_path(identity)
        checkpoint_path = _bl_repair_audit_checkpoint_path_if_present(
            root=root,
            relative_path=relative_path,
        )
        if checkpoint_path is None:
            raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid")
        try:
            checkpoint = _read_bl_repair_audit_checkpoint(
                root=root,
                path=checkpoint_path,
                expected_identity=identity,
                binding=binding,
                profile=resolution.profile,
            )
        except SyntheticRasterTextActivationError:
            raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid") from None
        if checkpoint.candidate != candidates.get((binding.asset_id, binding.image_sha256)):
            raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid")


def run_source_attested_bl_repair_audit_candidates(
    *,
    document_preview_root: Path,
    runtime_configuration_relative_path: str,
) -> BlRepairAuditCandidateRunV1:
    """Compile, run/resume, score, and fully replay the fixed 800-asset audit."""

    root = _root(document_preview_root)
    try:
        receipt = compile_bl_repair_audit_dataset(document_preview_root=root)
    except (OSError, ValidationError, ValueError):
        raise SyntheticRasterTextActivationError("bl_repair_audit_dataset_invalid") from None
    # The compiler is create-or-fully-verify and returns only after replaying
    # all 200 source PDFs, 800 rasters, and 800 source-derived Gold files.
    receipt_sha256 = bl_repair_audit_receipt_sha256(receipt)
    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or resolution.engine is None
        or not isinstance(resolution.profile, RasterTextOfflineProfileV19)
        or not _is_exact_bl_repair_audit_profile(resolution.profile)
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterTextActivationError("bl_repair_audit_candidate_runtime_unavailable")
    profile = resolution.profile
    candidates = _run_bl_repair_audit_observations(
        root=root,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        runtime_configuration_sha256=resolution.runtime_configuration_sha256,
        profile=profile,
        engine=cast(MediaAwareRasterTextEngine, resolution.engine),
        bindings=receipt.rasters,
    )
    try:
        bundle = BlRepairAuditCandidateBundleV1(
            dataset_receipt_sha256=receipt_sha256,
            candidate_profile_revision=profile.raster_text_profile_revision,
            runtime_configuration_sha256=resolution.runtime_configuration_sha256,
            extractor_revision=profile.extractor_revision,
            candidates=candidates,
        )
        bundle_path, bundle_sha256 = write_bl_repair_audit_candidate_bundle(
            document_preview_root=root,
            receipt=receipt,
            bundle=bundle,
        )
        verified_bundle = verify_bl_repair_audit_candidate_bundle(
            document_preview_root=root,
            candidate_bundle_relative_path=bundle_path,
            candidate_bundle_sha256=bundle_sha256,
            expected_candidate_profile_revision=profile.raster_text_profile_revision,
            expected_runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
        )
        if verified_bundle != bundle:
            raise ValueError
        evaluation = evaluate_bl_repair_audit_candidates(
            document_preview_root=root,
            receipt=receipt,
            bundle=bundle,
        )
        evaluation_path, evaluation_sha256 = write_bl_repair_audit_evaluation(
            document_preview_root=root,
            receipt=receipt,
            evaluation=evaluation,
        )
        verified_evaluation = verify_bl_repair_audit_evaluation(
            document_preview_root=root,
            evaluation_relative_path=evaluation_path,
            evaluation_sha256=evaluation_sha256,
            expected_candidate_profile_revision=profile.raster_text_profile_revision,
            expected_runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            require_gate_passed=False,
        )
        if verified_evaluation != evaluation:
            raise ValueError
    except SyntheticRasterTextActivationError:
        raise
    except (OSError, ValidationError, ValueError):
        raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid") from None
    if not evaluation.gate_passed:
        raise SyntheticRasterTextActivationError("bl_repair_audit_gate_failed")
    run = BlRepairAuditCandidateRunV1(
        dataset_receipt_relative_path=BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH,
        dataset_receipt_sha256=receipt_sha256,
        candidate_bundle_relative_path=bundle_path,
        candidate_bundle_sha256=bundle_sha256,
        evaluation_relative_path=evaluation_path,
        evaluation_sha256=evaluation_sha256,
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_sha256=resolution.runtime_configuration_sha256,
    )
    _verify_bl_repair_audit_activation_evidence(
        root=root,
        runtime_configuration_relative_path=runtime_configuration_relative_path,
        evidence=run,
    )
    return run


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise SyntheticRasterTextActivationError("raster_text_candidate_execution_failed")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]


def run_paddleocr3_synthetic_raster_text_candidates(
    *,
    document_preview_root: Path,
    runtime_configuration_relative_path: str,
    synthetic_raster_receipt_relative_path: str,
    scope: _CandidateScope,
    source_gold_receipt_relative_path: str | None = None,
    preflight_smoke_only: bool = False,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1 | None = None,
    fresh_projection_kind: _FreshProjectionKind | None = None,
    parent_stage1_decision_relative_path: str | None = None,
    fresh_selection_revision: _FreshSelectionRevision = (_FRESH_HOLDOUT_SELECTION_REVISION),
) -> RasterTextCandidateRunV1:
    """Run exactly one fixed scope with no Gold text or model arguments."""

    root = _root(document_preview_root)
    fresh_context: _FreshScopeContext | None = None
    if scope in ("fresh-calibration", "fresh-sealed"):
        if preflight_smoke_only or build_inputs is not None:
            raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
        fresh_context = _fresh_scope_context(
            root=root,
            synthetic_raster_receipt_relative_path=(synthetic_raster_receipt_relative_path),
            scope=cast(_FreshScope, scope),
            selection_revision=fresh_selection_revision,
        )
        receipt = fresh_context.receipt
        receipt_sha256 = fresh_context.receipt_sha256
        ordered_assets = list(fresh_context.assets)
    else:
        if (
            fresh_projection_kind is not None
            or parent_stage1_decision_relative_path is not None
            or fresh_selection_revision != _FRESH_HOLDOUT_SELECTION_REVISION
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
        receipt, receipt_sha256 = _load_dataset_receipt(
            root,
            synthetic_raster_receipt_relative_path,
            scope=cast(_Scope, scope),
            preflight_smoke_only=preflight_smoke_only,
            build_inputs=build_inputs,
        )
        ordered_assets = list(_scope_assets(receipt, cast(_Scope, scope)))
    if source_gold_receipt_relative_path is None:
        if fresh_context is not None:
            source_gold_receipt_relative_path, source_gold_sha256 = (
                _write_source_attested_fresh_raster_text_gold_receipt(
                    root=root,
                    context=fresh_context,
                    scope=cast(_FreshScope, scope),
                )
            )
        else:
            source_gold_receipt_relative_path, source_gold_sha256 = (
                write_source_attested_raster_text_gold_receipt(
                    document_preview_root=root,
                    synthetic_raster_receipt_relative_path=(synthetic_raster_receipt_relative_path),
                    scope=cast(_Scope, scope),
                    preflight_smoke_only=preflight_smoke_only,
                    build_inputs=build_inputs,
                )
            )
    else:
        source_gold_raw = _read(
            root,
            source_gold_receipt_relative_path,
            limit=_MAX_RECEIPT_BYTES,
            code="raster_text_source_gold_invalid",
        )
        source_gold_sha256 = hashlib.sha256(source_gold_raw).hexdigest()
        if fresh_context is not None:
            try:
                fresh_source_gold = RasterTextFreshSourceGoldReceiptV1.model_validate_json(
                    source_gold_raw
                )
            except ValidationError:
                raise SyntheticRasterTextActivationError(
                    "raster_text_source_gold_invalid"
                ) from None
            expected_assets = {
                (asset.asset_id, asset.profile_id): asset for asset in ordered_assets
            }
            observed_bindings = {
                (binding.asset_id, binding.profile_id): binding
                for binding in fresh_source_gold.bindings
            }
            if (
                fresh_source_gold.scope != scope
                or fresh_source_gold.synthetic_raster_receipt_sha256 != receipt_sha256
                or fresh_source_gold.source_receipt_sha256 != fresh_context.source_receipt_sha256
                or fresh_source_gold.selection_receipt_relative_path
                != fresh_context.selection_relative_path
                or fresh_source_gold.selection_receipt_sha256 != fresh_context.selection_sha256
                or set(observed_bindings) != set(expected_assets)
                or source_gold_raw
                != _canonical_json(fresh_source_gold.model_dump(mode="json", warnings=False))
                or any(
                    binding.family_id != expected_assets[key].family_id
                    or binding.document_type != expected_assets[key].document_type
                    or binding.source_pdf_sha256 != expected_assets[key].source_pdf_sha256
                    or binding.image_sha256 != expected_assets[key].image_sha256
                    or binding.source_raster_gold_relative_path
                    != expected_assets[key].gold_relative_path
                    or binding.source_raster_gold_sha256 != expected_assets[key].gold_sha256
                    or binding.derived_text_gold_sha256
                    != _digest_model(_gold_for_asset(root, expected_assets[key]))
                    for key, binding in observed_bindings.items()
                )
            ):
                raise SyntheticRasterTextActivationError("raster_text_source_gold_invalid")
        else:
            try:
                source_gold = RasterTextSourceGoldReceiptV1.model_validate_json(source_gold_raw)
            except ValidationError:
                raise SyntheticRasterTextActivationError(
                    "raster_text_source_gold_invalid"
                ) from None
            if (
                source_gold.scope != scope
                or source_gold.synthetic_raster_receipt_sha256 != receipt_sha256
                or source_gold_raw
                != _canonical_json(source_gold.model_dump(mode="json", warnings=False))
            ):
                raise SyntheticRasterTextActivationError("raster_text_source_gold_invalid")
    projection: RasterTextFreshEvaluationProjectionReceiptV1 | None = None
    projection_relative_path: str | None = None
    projection_sha256: str | None = None
    if fresh_context is not None and fresh_projection_kind is not None:
        projection, projection_relative_path, projection_sha256 = (
            _write_fresh_evaluation_projection(
                root=root,
                context=fresh_context,
                source_gold_receipt_relative_path=(source_gold_receipt_relative_path),
                source_gold_receipt_sha256=source_gold_sha256,
                projection_kind=fresh_projection_kind,
                parent_stage1_decision_relative_path=(parent_stage1_decision_relative_path),
            )
        )
        ordered_assets = _fresh_projection_assets(context=fresh_context, projection=projection)
    elif parent_stage1_decision_relative_path is not None:
        raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root, configuration_relative_path=runtime_configuration_relative_path
    )
    if (
        resolution.status != "available"
        or resolution.engine is None
        or resolution.profile is None
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterTextActivationError("raster_text_candidate_runtime_unavailable")
    profile = resolution.profile
    limits = _InputLimits()
    provider = getattr(resolution.engine, "provider", None)
    open_session = getattr(provider, "open_evaluation_session", None)
    input_adapter = getattr(resolution.engine, "input_adapter", None)
    if not callable(open_session) or input_adapter is None:
        raise SyntheticRasterTextActivationError("raster_text_candidate_runtime_unavailable")
    runner_implementation_sha256 = _runner_implementation_sha256()
    observations_by_key: dict[tuple[str, str], RasterTextCandidateObservationV1] = {}
    equivalent_observations: dict[tuple[str, str, str, int], RasterTextCandidateObservationV1] = {}
    missing: list[
        tuple[
            RasterAssetReceiptV1,
            RasterTextCandidateCheckpointIdentityV1 | RasterTextFreshCandidateCheckpointIdentityV1,
        ]
    ] = []
    for asset in ordered_assets:
        identity = _candidate_checkpoint_identity(
            scope=scope,
            synthetic_raster_receipt_sha256=receipt_sha256,
            source_gold_receipt_sha256=source_gold_sha256,
            selection_receipt_sha256=(
                None if fresh_context is None else fresh_context.selection_sha256
            ),
            runtime_configuration_sha256=resolution.runtime_configuration_sha256,
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            asset=asset,
        )
        checkpoint_relative_path, _identity_sha256 = _candidate_checkpoint_relative_path(identity)
        checkpoint_path = _candidate_checkpoint_path_if_present(root, checkpoint_relative_path)
        if checkpoint_path is None:
            missing.append((asset, identity))
            continue
        # A cached OCR observation is reusable only after the current raster
        # material and every immutable binding have been checked again.
        _verified_asset(root, asset, pdfium_version=receipt.pdfium_version)
        observation = _read_candidate_checkpoint(
            root=root,
            path=checkpoint_path,
            expected_identity=identity,
            asset=asset,
        )
        equivalent_key = _candidate_input_equivalence_key(asset)
        equivalent = equivalent_observations.get(equivalent_key)
        if equivalent is not None and _candidate_content_sha256(
            equivalent.candidate
        ) != _candidate_content_sha256(observation.candidate):
            raise SyntheticRasterTextActivationError("raster_text_candidate_checkpoint_invalid")
        equivalent_observations.setdefault(equivalent_key, observation)
        observations_by_key[(asset.asset_id, asset.profile_id)] = observation

    # Reuse only byte-identical, profile-identical OCR content.  Every target
    # raster is rehashed and receives its own provenance-bound checkpoint.
    # The separately measured warm service smoke still invokes all 24 inputs,
    # so this optimization cannot hide request-path or latency regressions.
    missing_groups: dict[
        tuple[str, str, str, int],
        list[
            tuple[
                RasterAssetReceiptV1,
                RasterTextCandidateCheckpointIdentityV1
                | RasterTextFreshCandidateCheckpointIdentityV1,
            ]
        ],
    ] = {}
    for asset, identity in missing:
        equivalent_key = _candidate_input_equivalence_key(asset)
        equivalent = equivalent_observations.get(equivalent_key)
        if equivalent is None:
            missing_groups.setdefault(equivalent_key, []).append((asset, identity))
            continue
        _verified_asset(root, asset, pdfium_version=receipt.pdfium_version)
        observation = _rebind_equivalent_candidate_observation(
            source=equivalent,
            asset=asset,
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
        )
        _write_candidate_checkpoint(
            root=root,
            identity=identity,
            observation=observation,
        )
        observations_by_key[(asset.asset_id, asset.profile_id)] = observation
    # This is the evaluation-only persistent child.  It loads the same pinned
    # PP-OCR models once, but every input/result remains individually bound
    # and timed.  The separately measured customer path uses the service
    # protocol below; the two protocols must agree exactly before activation.
    try:
        # ``resolve_paddleocr3_raster_text_candidate`` has just completed the
        # exhaustive parent interpreter/module verification for this exact
        # immutable configuration.  Reuse that attestation at session launch
        # rather than hashing the same full import trees again.  The provider
        # performs a direct-file binding check here and the isolated child
        # keeps its configuration/package/model checks before ready and per
        # candidate frame.
        if missing_groups:
            with open_session(parent_materials_verified=True) as session:
                evaluation_engine = MediaAwareRasterTextEngine(
                    input_adapter=input_adapter, provider=session
                )
                for equivalent_key, group in missing_groups.items():
                    asset, identity = group[0]
                    raw = _verified_asset(root, asset, pdfium_version=receipt.pdfium_version)
                    started = time.monotonic()
                    result = extract_raster_text_preview(
                        evaluation_engine,
                        raster_input=_raster_text_input_for_asset(
                            asset=asset,
                            raw=raw,
                            raster_text_profile_revision=profile.raster_text_profile_revision,
                        ),
                        structure=inspect_raster_document_structure(
                            raw,
                            asset.media_type,
                            limits,  # type: ignore[arg-type]
                        ),
                        settings=limits,  # type: ignore[arg-type]
                    )
                    elapsed = time.monotonic() - started
                    if elapsed <= 0 or elapsed > 30:
                        raise SyntheticRasterTextActivationError(
                            "raster_text_candidate_execution_failed"
                        )
                    _validate_candidate_provenance(
                        candidate=result,
                        asset=asset,
                        raster_text_profile_revision=(profile.raster_text_profile_revision),
                        extractor_revision=profile.extractor_revision,
                        code="raster_text_candidate_execution_failed",
                    )
                    observation = RasterTextCandidateObservationV1(
                        asset_id=asset.asset_id,
                        profile_id=asset.profile_id,
                        elapsed_seconds=elapsed,
                        candidate=result,
                    )
                    _write_candidate_checkpoint(
                        root=root,
                        identity=identity,
                        observation=observation,
                    )
                    observations_by_key[(asset.asset_id, asset.profile_id)] = observation
                    equivalent_observations[equivalent_key] = observation
                    for duplicate_asset, duplicate_identity in group[1:]:
                        _verified_asset(
                            root,
                            duplicate_asset,
                            pdfium_version=receipt.pdfium_version,
                        )
                        duplicate_observation = _rebind_equivalent_candidate_observation(
                            source=observation,
                            asset=duplicate_asset,
                            raster_text_profile_revision=(profile.raster_text_profile_revision),
                            extractor_revision=profile.extractor_revision,
                        )
                        _write_candidate_checkpoint(
                            root=root,
                            identity=duplicate_identity,
                            observation=duplicate_observation,
                        )
                        observations_by_key[
                            (duplicate_asset.asset_id, duplicate_asset.profile_id)
                        ] = duplicate_observation
    except SyntheticRasterTextActivationError:
        raise
    except Exception:
        raise SyntheticRasterTextActivationError("raster_text_candidate_execution_failed") from None
    observations = [
        observations_by_key[(asset.asset_id, asset.profile_id)] for asset in ordered_assets
    ]
    if _runner_implementation_sha256() != runner_implementation_sha256:
        raise SyntheticRasterTextActivationError("raster_text_candidate_runner_drift")
    bundle: (
        RasterTextCandidateBundleV1
        | RasterTextFreshCandidateBundleV1
        | RasterTextFreshProjectedCandidateBundleV1
    )
    if projection is not None:
        if projection_relative_path is None or projection_sha256 is None:
            raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
        bundle = RasterTextFreshProjectedCandidateBundleV1(
            scope=projection.source_scope,
            synthetic_raster_receipt_sha256=receipt_sha256,
            selection_receipt_relative_path=(
                fresh_context.selection_relative_path  # type: ignore[union-attr]
            ),
            selection_receipt_sha256=(
                fresh_context.selection_sha256  # type: ignore[union-attr]
            ),
            source_gold_receipt_sha256=source_gold_sha256,
            projection_receipt_relative_path=projection_relative_path,
            projection_receipt_sha256=projection_sha256,
            runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            candidates=observations,
        )
        bundle_directory = "fresh-projected-candidate-bundles"
    elif fresh_context is not None:
        bundle = RasterTextFreshCandidateBundleV1(
            scope=cast(_FreshScope, scope),
            synthetic_raster_receipt_sha256=receipt_sha256,
            selection_receipt_relative_path=(fresh_context.selection_relative_path),
            selection_receipt_sha256=fresh_context.selection_sha256,
            source_gold_receipt_sha256=source_gold_sha256,
            runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            candidates=observations,
        )
        bundle_directory = "fresh-candidate-bundles"
    else:
        bundle = RasterTextCandidateBundleV1(
            scope=cast(_Scope, scope),
            synthetic_raster_receipt_sha256=receipt_sha256,
            runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            candidates=observations,
        )
        bundle_directory = "candidate-bundles"
    bundle_sha256 = _digest_model(bundle)
    bundle_path = f"synthetic-raster-text-v1/{bundle_directory}/{scope}/{bundle_sha256}.json"
    try:
        written, digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=bundle_path,
            value=bundle.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError("raster_text_candidate_write_failed") from None
    if written != bundle_path or digest != bundle_sha256:
        raise SyntheticRasterTextActivationError("raster_text_candidate_write_failed")
    execution: (
        RasterTextCandidateExecutionV1
        | RasterTextFreshCandidateExecutionV1
        | RasterTextFreshProjectedCandidateExecutionV1
    )
    if projection is not None:
        if projection_relative_path is None or projection_sha256 is None:
            raise SyntheticRasterTextActivationError("raster_text_fresh_projection_invalid")
        execution = RasterTextFreshProjectedCandidateExecutionV1(
            scope=projection.source_scope,
            synthetic_raster_receipt_sha256=receipt_sha256,
            selection_receipt_relative_path=(
                fresh_context.selection_relative_path  # type: ignore[union-attr]
            ),
            selection_receipt_sha256=(
                fresh_context.selection_sha256  # type: ignore[union-attr]
            ),
            source_gold_receipt_relative_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_gold_sha256,
            projection_receipt_relative_path=projection_relative_path,
            projection_receipt_sha256=projection_sha256,
            runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
            candidate_bundle_relative_path=bundle_path,
            candidate_bundle_sha256=bundle_sha256,
            candidate_count=len(observations),  # type: ignore[arg-type]
            latency_p95_seconds=_p95([item.elapsed_seconds for item in observations]),
        )
        execution_directory = "fresh-projected-candidate-executions"
    elif fresh_context is not None:
        execution = RasterTextFreshCandidateExecutionV1(
            scope=cast(_FreshScope, scope),
            synthetic_raster_receipt_sha256=receipt_sha256,
            selection_receipt_relative_path=(fresh_context.selection_relative_path),
            selection_receipt_sha256=fresh_context.selection_sha256,
            source_gold_receipt_relative_path=(source_gold_receipt_relative_path),
            source_gold_receipt_sha256=source_gold_sha256,
            runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
            candidate_bundle_relative_path=bundle_path,
            candidate_bundle_sha256=bundle_sha256,
            candidate_count=len(observations),
            latency_p95_seconds=_p95([item.elapsed_seconds for item in observations]),
        )
        execution_directory = "fresh-candidate-executions"
    else:
        execution = RasterTextCandidateExecutionV1(
            scope=cast(_Scope, scope),
            synthetic_raster_receipt_sha256=receipt_sha256,
            runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            raster_text_profile_revision=profile.raster_text_profile_revision,
            extractor_revision=profile.extractor_revision,
            candidate_bundle_relative_path=bundle_path,
            candidate_bundle_sha256=bundle_sha256,
            candidate_count=len(observations),
            latency_p95_seconds=_p95([item.elapsed_seconds for item in observations]),
        )
        execution_directory = "candidate-executions"
    execution_sha256 = _digest_model(execution)
    execution_path = (
        f"synthetic-raster-text-v1/{execution_directory}/{scope}/{execution_sha256}.json"
    )
    try:
        written, digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=execution_path,
            value=execution.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError("raster_text_candidate_write_failed") from None
    if written != execution_path or digest != execution_sha256:
        raise SyntheticRasterTextActivationError("raster_text_candidate_write_failed")
    return RasterTextCandidateRunV1(
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_gold_sha256,
        candidate_execution_relative_path=execution_path,
        candidate_execution_sha256=execution_sha256,
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_sha256=resolution.runtime_configuration_sha256,
        latency_p95_seconds=execution.latency_p95_seconds,
    )


def run_warm_service_raster_text_smoke(
    *,
    document_preview_root: Path,
    runtime_configuration_relative_path: str,
    synthetic_raster_receipt_relative_path: str,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
) -> tuple[str, str]:
    """Measure the real warm service protocol against exact final candidates.

    This function intentionally resolves a profile directly from its immutable
    configuration and never reads or writes an approval pointer.  The same
    persistent service child used after activation receives the same decoded
    input and deterministic request identity as the final smoke candidate.
    """

    root = _root(document_preview_root)
    receipt, receipt_sha256 = _load_dataset_receipt(
        root, synthetic_raster_receipt_relative_path, scope="smoke"
    )
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_warm_smoke_candidate_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_warm_smoke_candidate_invalid",
    )
    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or resolution.engine is None
        or resolution.profile is None
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_runtime_unavailable")
    profile = resolution.profile
    try:
        source, execution, bundle, _cases = verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=source_gold_receipt_relative_path,
            candidate_execution_relative_path=candidate_execution_relative_path,
            scope="smoke",
            expected_raster_text_profile_revision=profile.raster_text_profile_revision,
            expected_runtime_configuration_sha256=(resolution.runtime_configuration_sha256),
            expected_extractor_revision=profile.extractor_revision,
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_warm_smoke_candidate_invalid"
        ) from None
    if (
        source.scope != "smoke"
        or source.synthetic_raster_receipt_sha256 != receipt_sha256
        or execution.scope != "smoke"
        or execution.synthetic_raster_receipt_sha256 != receipt_sha256
        or bundle.scope != "smoke"
        or bundle.synthetic_raster_receipt_sha256 != receipt_sha256
        or execution.runtime_configuration_sha256 != resolution.runtime_configuration_sha256
        or bundle.runtime_configuration_sha256 != resolution.runtime_configuration_sha256
        or execution.raster_text_profile_revision != profile.raster_text_profile_revision
        or bundle.raster_text_profile_revision != profile.raster_text_profile_revision
        or execution.extractor_revision != profile.extractor_revision
        or bundle.extractor_revision != profile.extractor_revision
        or execution.candidate_count != 24
        or len(bundle.candidates) != 24
    ):
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_candidate_invalid")
    expected = {(item.asset_id, item.profile_id): item.candidate for item in bundle.candidates}
    assets = _scope_assets(receipt, "smoke")
    if set(expected) != {(item.asset_id, item.profile_id) for item in assets}:
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_candidate_invalid")
    provider = getattr(resolution.engine, "provider", None)
    input_adapter = getattr(resolution.engine, "input_adapter", None)
    start_service_session = getattr(provider, "start_service_session", None)
    verify_parent_materials = getattr(provider, "_verify_parent_materials", None)
    if (
        not callable(start_service_session)
        or not callable(verify_parent_materials)
        or input_adapter is None
    ):
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_runtime_unavailable")

    session: object | None = None
    try:
        # Candidate resolution can precede source/evidence loading by enough
        # time for a local bundle swap.  Re-run the complete parent check just
        # before launching the measured warm child, but keep that verification
        # outside its profile-bound model-load/ready measurement.
        verify_parent_materials()
    except Exception:
        raise SyntheticRasterTextActivationError(
            "raster_text_warm_smoke_runtime_unavailable"
        ) from None
    startup_started = time.monotonic()
    try:
        # ``verify_parent_materials`` immediately above bound this exact
        # launch.  Avoid counting that full tree check as native warm startup.
        session = start_service_session(parent_materials_verified=True)
    except Exception:
        raise SyntheticRasterTextActivationError(
            "raster_text_warm_smoke_runtime_unavailable"
        ) from None
    startup_elapsed = time.monotonic() - startup_started
    request_timeout = getattr(session, "request_timeout_seconds", None)
    startup_timeout = getattr(session, "startup_timeout_seconds", None)
    expected_startup_timeout = raster_text_service_startup_timeout_seconds(profile)
    expected_request_timeout = min(profile.sidecar_timeout_seconds, 30)
    if (
        startup_elapsed <= 0
        or startup_elapsed > expected_startup_timeout
        or not isinstance(request_timeout, int)
        or request_timeout != expected_request_timeout
        or not isinstance(startup_timeout, int)
        or startup_timeout != expected_startup_timeout
    ):
        close = getattr(session, "close", None)
        if callable(close):
            close()
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_runtime_unavailable")

    elapsed_values: list[float] = []
    observed_results: list[tuple[str, str, RasterTextExtractionResultV1]] = []
    limits = _InputLimits()
    try:
        service_engine = MediaAwareRasterTextEngine(
            input_adapter=input_adapter,
            provider=session,  # type: ignore[arg-type]
        )
        for asset in assets:
            raw = _verified_asset(root, asset, pdfium_version=receipt.pdfium_version)
            raster_input = _raster_text_input_for_asset(
                asset=asset,
                raw=raw,
                raster_text_profile_revision=profile.raster_text_profile_revision,
            )
            started = time.monotonic()
            result = extract_raster_text_preview(
                service_engine,
                raster_input=raster_input,
                structure=inspect_raster_document_structure(raw, asset.media_type, limits),  # type: ignore[arg-type]
                settings=limits,  # type: ignore[arg-type]
            )
            elapsed = time.monotonic() - started
            candidate = expected[(asset.asset_id, asset.profile_id)]
            if (
                elapsed <= 0
                or elapsed > 30
                or result.status == "unavailable"
                or result.canonical_text is None
                or result.extractor_revision != profile.extractor_revision
                or raster_text_result_json(result) != raster_text_result_json(candidate)
            ):
                raise SyntheticRasterTextActivationError("raster_text_warm_smoke_failed")
            elapsed_values.append(elapsed)
            observed_results.append((asset.asset_id, asset.profile_id, result))
    except SyntheticRasterTextActivationError:
        raise
    except Exception:
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_failed") from None
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # The session is already no longer eligible for use; do not
                # expose child details from its best-effort cleanup path.
                pass

    artifact = RasterTextWarmServiceSmokeReceiptV1(
        artifact_kind="synthetic_raster_text_warm_service_smoke",
        dataset_id="synthetic-raster-text-v1",
        scope="smoke",
        source_truth_kind="pdf_content_stream_text_v1",
        execution_mode="persistent_service_warm",
        service_protocol="customs-raster-text-sidecar-v1-service",
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_sha256=resolution.runtime_configuration_sha256,
        extractor_revision=profile.extractor_revision,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        candidate_bundle_sha256=execution.candidate_bundle_sha256,
        equality_mode="exact_canonical_result_v1",
        equality_set_sha256=raster_text_exact_result_set_sha256(observed_results),
        asset_count=24,
        startup_latency_seconds=startup_elapsed,
        startup_timeout_seconds=startup_timeout,
        warm_latency_p95_seconds=_p95(elapsed_values),
        request_timeout_seconds=request_timeout,
    )
    digest = _digest_model(artifact)
    relative = (
        "synthetic-raster-text-v1/warm-service-smoke/"
        f"{profile.raster_text_profile_revision.rsplit(':', 1)[1]}/{digest}.json"
    )
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_write_failed") from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_warm_smoke_write_failed")
    return relative, digest


def _load_source_gold_cases(
    *, root: Path, source_gold_relative_path: str, execution_relative_path: str
) -> tuple[
    RasterTextSourceGoldReceiptV1,
    RasterTextCandidateExecutionV1,
    RasterTextCandidateBundleV1,
    list[RasterTextEvaluationCaseV1],
]:
    source_raw = _read(
        root,
        source_gold_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_evaluation_invalid",
    )
    execution_raw = _read(
        root,
        execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_evaluation_invalid",
    )
    try:
        source = RasterTextSourceGoldReceiptV1.model_validate_json(source_raw)
        execution = RasterTextCandidateExecutionV1.model_validate_json(execution_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid") from None
    if source_raw != _canonical_json(
        source.model_dump(mode="json", warnings=False)
    ) or execution_raw != _canonical_json(execution.model_dump(mode="json", warnings=False)):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    bundle_raw = _read(
        root,
        execution.candidate_bundle_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_evaluation_invalid",
    )
    try:
        bundle = RasterTextCandidateBundleV1.model_validate_json(bundle_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid") from None
    if (
        hashlib.sha256(bundle_raw).hexdigest() != execution.candidate_bundle_sha256
        or bundle_raw != _canonical_json(bundle.model_dump(mode="json", warnings=False))
        or bundle.scope != source.scope
        or bundle.synthetic_raster_receipt_sha256 != source.synthetic_raster_receipt_sha256
        or bundle.raster_text_profile_revision != execution.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != execution.runtime_configuration_sha256
        or len(bundle.candidates) != execution.candidate_count
    ):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    candidates = {(item.asset_id, item.profile_id): item.candidate for item in bundle.candidates}
    if set(candidates) != {(item.asset_id, item.profile_id) for item in source.bindings}:
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    cases: list[RasterTextEvaluationCaseV1] = []
    for binding in source.bindings:
        raw_gold = _read(
            root,
            binding.source_raster_gold_relative_path,
            limit=_MAX_RECEIPT_BYTES,
            code="raster_text_evaluation_invalid",
        )
        try:
            raster_gold = RasterGoldArtifactV1.model_validate_json(raw_gold)
        except ValidationError:
            raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid") from None
        text_gold = source_only_text_gold_from_raster_gold(raster_gold)
        if (
            raw_gold != _canonical_json(raster_gold.model_dump(mode="json", warnings=False))
            or hashlib.sha256(raw_gold).hexdigest() != binding.source_raster_gold_sha256
            or _digest_model(text_gold) != binding.derived_text_gold_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
        cases.append(
            RasterTextEvaluationCaseV1(
                asset_id=binding.asset_id,
                document_type=binding.document_type,
                profile_id=binding.profile_id,
                image_sha256=binding.image_sha256,
                gold=text_gold,
                candidate=candidates[(binding.asset_id, binding.profile_id)],
            )
        )
    return source, execution, bundle, cases


def verify_source_attested_raster_text_evidence(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    scope: _Scope,
    expected_raster_text_profile_revision: str | None = None,
    expected_runtime_configuration_sha256: str | None = None,
    expected_extractor_revision: str | None = None,
    expected_runner_implementation_sha256: str | None = None,
) -> tuple[
    RasterTextSourceGoldReceiptV1,
    RasterTextCandidateExecutionV1,
    RasterTextCandidateBundleV1,
    list[RasterTextEvaluationCaseV1],
]:
    """Rebuild one candidate chain from the fixed source-attested dataset.

    Compact source-Gold and candidate receipts are not independently trusted:
    this re-reads the fixed raster receipt for the requested split, verifies
    every requested source/raster/Gold material, then proves each compact
    binding and public candidate result still belongs to that exact asset.
    It is deliberately an evaluation/activation boundary helper, never a
    customer OCR request path.
    """

    root = _root(document_preview_root)
    receipt, receipt_sha256 = _load_dataset_receipt(
        root, SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH, scope=scope
    )
    source, execution, bundle, cases = _load_source_gold_cases(
        root=root,
        source_gold_relative_path=source_gold_receipt_relative_path,
        execution_relative_path=candidate_execution_relative_path,
    )
    assets = _scope_assets(receipt, scope)
    expected_assets = {(asset.asset_id, asset.profile_id): asset for asset in assets}
    source_bindings = {
        (binding.asset_id, binding.profile_id): binding for binding in source.bindings
    }
    candidates = {
        (observation.asset_id, observation.profile_id): observation.candidate
        for observation in bundle.candidates
    }
    runner_implementation_sha256 = (
        _runner_implementation_sha256()
        if expected_runner_implementation_sha256 is None
        else expected_runner_implementation_sha256
    )
    if not re.fullmatch(_SHA256, runner_implementation_sha256):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    if (
        source.scope != scope
        or execution.scope != scope
        or bundle.scope != scope
        or source.synthetic_raster_receipt_sha256 != receipt_sha256
        or execution.synthetic_raster_receipt_sha256 != receipt_sha256
        or bundle.synthetic_raster_receipt_sha256 != receipt_sha256
        or execution.extractor_revision != bundle.extractor_revision
        or bundle.runner_implementation_sha256 != runner_implementation_sha256
        or execution.candidate_count != len(assets)
        or set(source_bindings) != set(expected_assets)
        or set(candidates) != set(expected_assets)
    ):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    if expected_raster_text_profile_revision is not None and (
        execution.raster_text_profile_revision != expected_raster_text_profile_revision
        or bundle.raster_text_profile_revision != expected_raster_text_profile_revision
    ):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    if expected_runtime_configuration_sha256 is not None and (
        execution.runtime_configuration_sha256 != expected_runtime_configuration_sha256
        or bundle.runtime_configuration_sha256 != expected_runtime_configuration_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    if expected_extractor_revision is not None and (
        execution.extractor_revision != expected_extractor_revision
        or bundle.extractor_revision != expected_extractor_revision
    ):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    for key, asset in expected_assets.items():
        binding = source_bindings[key]
        candidate = candidates[key]
        if (
            binding.family_id != asset.family_id
            or binding.document_type != asset.document_type
            or binding.source_pdf_sha256 != asset.source_pdf_sha256
            or binding.image_sha256 != asset.image_sha256
            or binding.source_raster_gold_relative_path != asset.gold_relative_path
            or binding.source_raster_gold_sha256 != asset.gold_sha256
            or candidate.document_asset_id != asset.asset_id
            or candidate.extraction_request_id
            != f"raster-text-{asset.profile_id}-{asset.image_sha256[:24]}"
            or candidate.raster_text_profile_revision != execution.raster_text_profile_revision
            or candidate.source_sha256 != asset.image_sha256
            or candidate.media_type != asset.media_type
            or candidate.byte_size != asset.byte_size
            or candidate.extractor_revision != execution.extractor_revision
        ):
            raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    if len(cases) != len(assets):
        raise SyntheticRasterTextActivationError("raster_text_evaluation_invalid")
    return source, execution, bundle, cases


def _semantic_v3_counts_by_stratum(
    cases: Sequence[RasterTextEvaluationCaseV1],
) -> dict[tuple[str, str], RasterTextMetricCountsV1]:
    counts: dict[tuple[str, str], RasterTextMetricCountsV1] = {}
    for case in cases:
        key = (case.document_type, case.profile_id)
        counts[key] = counts.get(key, RasterTextMetricCountsV1()) + (
            evaluate_synthetic_raster_semantic_text_case_v3(case)
        )
    expected = {
        (document_type, profile_id)
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in RASTER_PROFILES
    }
    if set(counts) != expected:
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    return counts


def _full_semantic_v3_reference_counts(
    *,
    root: Path,
    context: _FreshScopeContext,
    source: RasterTextFreshSourceGoldReceiptV1,
) -> dict[tuple[str, str], RasterTextMetricCountsV1]:
    expected_assets = {(asset.asset_id, asset.profile_id): asset for asset in context.assets}
    bindings = {(binding.asset_id, binding.profile_id): binding for binding in source.bindings}
    if set(bindings) != set(expected_assets):
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    counts: dict[tuple[str, str], RasterTextMetricCountsV1] = {}
    for key, asset in expected_assets.items():
        binding = bindings[key]
        gold = _gold_for_asset(root, asset)
        if (
            binding.family_id != asset.family_id
            or binding.document_type != asset.document_type
            or binding.source_pdf_sha256 != asset.source_pdf_sha256
            or binding.image_sha256 != asset.image_sha256
            or binding.source_raster_gold_relative_path != asset.gold_relative_path
            or binding.source_raster_gold_sha256 != asset.gold_sha256
            or binding.derived_text_gold_sha256 != _digest_model(gold)
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
        stratum = (asset.document_type, asset.profile_id)
        reference = canonicalize_synthetic_raster_semantic_text_v3(gold.canonical_text)
        counts[stratum] = counts.get(
            stratum, RasterTextMetricCountsV1()
        ) + RasterTextMetricCountsV1(
            character_reference=len(reference),
            critical_total=len(gold.critical_values),
        )
    return counts


def _one_sided_simultaneous_wilson_lower_bound(*, successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    alpha_per_claim = _FRESH_SEALED_SIMULTANEOUS_ALPHA / _FRESH_SEALED_SIMULTANEOUS_CLAIM_COUNT
    z = NormalDist().inv_cdf(1.0 - alpha_per_claim)
    proportion = successes / total
    squared = z * z
    denominator = 1.0 + squared / total
    centre = proportion + squared / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + squared / (4.0 * total * total)
    )
    return max(0.0, (centre - radius) / denominator)


def _fresh_sealed_stage1_strata(
    *,
    stage_counts: dict[tuple[str, str], RasterTextMetricCountsV1],
    full_counts: dict[tuple[str, str], RasterTextMetricCountsV1],
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> list[RasterTextFreshSealedStageStratumV1]:
    if quality_gate_revision not in (None, _V19_PREVIEW97_QUALITY_GATE_REVISION):
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    preview97 = quality_gate_revision == _V19_PREVIEW97_QUALITY_GATE_REVISION
    if set(stage_counts) != set(full_counts) or len(stage_counts) != 12:
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    result: list[RasterTextFreshSealedStageStratumV1] = []
    for (document_type, profile_id), current in sorted(stage_counts.items()):
        full = full_counts[(document_type, profile_id)]
        if (
            current.character_reference <= 0
            or current.critical_total <= 0
            or current.character_errors > current.character_reference
            or current.critical_matches > current.critical_total
            or full.character_reference < current.character_reference
            or full.critical_total < current.critical_total
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
        character_accuracy_lower = _one_sided_simultaneous_wilson_lower_bound(
            successes=current.character_reference - current.character_errors,
            total=current.character_reference,
        )
        critical_accuracy_lower = _one_sided_simultaneous_wilson_lower_bound(
            successes=current.critical_matches,
            total=current.critical_total,
        )
        character_gate_passed = current.character_errors / current.character_reference <= 0.02
        critical_gate_passed = current.critical_matches / current.critical_total >= (
            0.97 if preview97 else 0.98
        )
        character_error_limit = math.floor(full.character_reference * 0.02)
        critical_error_limit = (
            full.critical_total - ((97 * full.critical_total + 99) // 100)
            if preview97
            else math.floor(full.critical_total * 0.02)
        )
        result.append(
            RasterTextFreshSealedStageStratumV1(
                document_type=cast(
                    Literal["bill_of_lading", "commercial_invoice", "packing_list"],
                    document_type,
                ),
                profile_id=cast(
                    Literal[
                        "png-216",
                        "jpeg-216-q75",
                        "jpeg-150-q75",
                        "scan-pdf-216-q75",
                    ],
                    profile_id,
                ),
                character_errors=current.character_errors,
                character_reference=current.character_reference,
                critical_matches=current.critical_matches,
                critical_total=current.critical_total,
                full_character_reference=full.character_reference,
                full_critical_total=full.critical_total,
                character_accuracy_lower_bound=character_accuracy_lower,
                critical_accuracy_lower_bound=critical_accuracy_lower,
                point_gate_passed=(character_gate_passed and critical_gate_passed),
                full_gate_recoverable=(
                    current.character_errors <= character_error_limit
                    and current.critical_total - current.critical_matches <= critical_error_limit
                ),
            )
        )
    return result


def _fresh_stage1_decision_from_strata(
    strata: Sequence[RasterTextFreshSealedStageStratumV1],
    *,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> _FreshSealedStageDecision:
    if quality_gate_revision not in (None, _V19_PREVIEW97_QUALITY_GATE_REVISION):
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    critical_threshold = (
        0.97 if quality_gate_revision == _V19_PREVIEW97_QUALITY_GATE_REVISION else 0.98
    )
    if len(strata) != 12:
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    if all(
        item.point_gate_passed
        and item.character_accuracy_lower_bound >= 0.98
        and item.critical_accuracy_lower_bound >= critical_threshold
        for item in strata
    ):
        return "early_pass"
    if any(not item.full_gate_recoverable for item in strata):
        return "early_fail"
    return "continue"


def verify_source_attested_fresh_raster_text_evidence(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    scope: _FreshScope,
    allow_inconclusive_stage1: bool = False,
    required_selection_revision: _FreshSelectionRevision = (_FRESH_HOLDOUT_SELECTION_REVISION),
    expected_runner_implementation_sha256: str | None = None,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> tuple[
    RasterTextFreshSourceGoldReceiptV1,
    RasterTextFreshCandidateExecutionV1 | RasterTextFreshProjectedCandidateExecutionV1,
    RasterTextFreshCandidateBundleV1 | RasterTextFreshProjectedCandidateBundleV1,
    list[RasterTextEvaluationCaseV1],
]:
    """Rebuild a fresh holdout chain without accepting an observed old split."""

    root = _root(document_preview_root)
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_evaluation_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_evaluation_invalid",
    )
    try:
        source = RasterTextFreshSourceGoldReceiptV1.model_validate_json(source_raw)
        execution_value = json.loads(execution_raw)
        projected = (
            isinstance(execution_value, dict)
            and execution_value.get("artifact_kind")
            == "synthetic_raster_text_fresh_projected_candidate_execution"
        )
        execution: (
            RasterTextFreshCandidateExecutionV1 | RasterTextFreshProjectedCandidateExecutionV1
        )
        if projected:
            execution = RasterTextFreshProjectedCandidateExecutionV1.model_validate(execution_value)
        else:
            execution = RasterTextFreshCandidateExecutionV1.model_validate(execution_value)
    except (json.JSONDecodeError, ValidationError):
        raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid") from None
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    execution_sha256 = hashlib.sha256(execution_raw).hexdigest()
    expected_source_path = (
        f"synthetic-raster-text-v1/fresh-source-gold-receipts/{scope}/{source_sha256}.json"
    )
    execution_directory = (
        "fresh-projected-candidate-executions" if projected else "fresh-candidate-executions"
    )
    expected_execution_path = (
        f"synthetic-raster-text-v1/{execution_directory}/{scope}/{execution_sha256}.json"
    )
    runner_implementation_sha256 = (
        _runner_implementation_sha256()
        if expected_runner_implementation_sha256 is None
        else expected_runner_implementation_sha256
    )
    if not re.fullmatch(_SHA256, runner_implementation_sha256):
        raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid")
    if (
        source_raw != _canonical_json(source.model_dump(mode="json", warnings=False))
        or execution_raw != _canonical_json(execution.model_dump(mode="json", warnings=False))
        or source_gold_receipt_relative_path != expected_source_path
        or candidate_execution_relative_path != expected_execution_path
        or source.scope != scope
        or execution.scope != scope
        or execution.source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or execution.source_gold_receipt_sha256 != hashlib.sha256(source_raw).hexdigest()
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid")
    selection_revision = _fresh_scope_selection_revision_from_path(
        root=root,
        relative_path=source.selection_receipt_relative_path,
    )
    if selection_revision != required_selection_revision:
        raise SyntheticRasterTextActivationError("raster_text_fresh_selection_invalid")
    context = _fresh_scope_context(
        root=root,
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope=scope,
        selection_revision=selection_revision,
        existing_selection_relative_path=source.selection_receipt_relative_path,
    )
    projection: RasterTextFreshEvaluationProjectionReceiptV1 | None = None
    if isinstance(execution, RasterTextFreshProjectedCandidateExecutionV1):
        projection = _read_fresh_evaluation_projection(
            root=root,
            context=context,
            relative_path=execution.projection_receipt_relative_path,
            expected_sha256=execution.projection_receipt_sha256,
            source_gold_receipt_relative_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_sha256,
        )
        projected_assets = _fresh_projection_assets(context=context, projection=projection)
    else:
        projected_assets = list(context.assets)
    bundle_raw = _read(
        root,
        execution.candidate_bundle_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_evaluation_invalid",
    )
    try:
        bundle: RasterTextFreshCandidateBundleV1 | RasterTextFreshProjectedCandidateBundleV1
        if projected:
            bundle = RasterTextFreshProjectedCandidateBundleV1.model_validate_json(bundle_raw)
        else:
            bundle = RasterTextFreshCandidateBundleV1.model_validate_json(bundle_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid") from None
    bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
    bundle_directory = (
        "fresh-projected-candidate-bundles" if projected else "fresh-candidate-bundles"
    )
    expected_bundle_path = (
        f"synthetic-raster-text-v1/{bundle_directory}/{scope}/{bundle_sha256}.json"
    )
    full_expected_assets = {(asset.asset_id, asset.profile_id): asset for asset in context.assets}
    expected_assets = {(asset.asset_id, asset.profile_id): asset for asset in projected_assets}
    bindings = {(binding.asset_id, binding.profile_id): binding for binding in source.bindings}
    candidates = {
        (observation.asset_id, observation.profile_id): observation.candidate
        for observation in bundle.candidates
    }
    if (
        bundle_raw != _canonical_json(bundle.model_dump(mode="json", warnings=False))
        or execution.candidate_bundle_relative_path != expected_bundle_path
        or execution.candidate_bundle_sha256 != bundle_sha256
        or source.synthetic_raster_receipt_sha256 != context.receipt_sha256
        or execution.synthetic_raster_receipt_sha256 != context.receipt_sha256
        or bundle.synthetic_raster_receipt_sha256 != context.receipt_sha256
        or source.source_receipt_sha256 != context.source_receipt_sha256
        or source.selection_receipt_relative_path != context.selection_relative_path
        or execution.selection_receipt_relative_path != context.selection_relative_path
        or bundle.selection_receipt_relative_path != context.selection_relative_path
        or source.selection_receipt_sha256 != context.selection_sha256
        or execution.selection_receipt_sha256 != context.selection_sha256
        or bundle.selection_receipt_sha256 != context.selection_sha256
        or execution.source_gold_receipt_sha256 != source_sha256
        or bundle.source_gold_receipt_sha256 != source_sha256
        or execution.scope != bundle.scope
        or execution.runtime_configuration_sha256 != bundle.runtime_configuration_sha256
        or execution.raster_text_profile_revision != bundle.raster_text_profile_revision
        or execution.extractor_revision != bundle.extractor_revision
        or bundle.runner_implementation_sha256 != runner_implementation_sha256
        or execution.candidate_count != len(expected_assets)
        or set(bindings) != set(full_expected_assets)
        or set(candidates) != set(expected_assets)
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid")
    if projection is None:
        if isinstance(execution, RasterTextFreshProjectedCandidateExecutionV1) or isinstance(
            bundle, RasterTextFreshProjectedCandidateBundleV1
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid")
    elif (
        not isinstance(execution, RasterTextFreshProjectedCandidateExecutionV1)
        or not isinstance(bundle, RasterTextFreshProjectedCandidateBundleV1)
        or execution.projection_receipt_relative_path != bundle.projection_receipt_relative_path
        or execution.projection_receipt_sha256 != bundle.projection_receipt_sha256
        or execution.projection_receipt_relative_path
        != _fresh_projection_expected_path(projection)[0]
        or execution.projection_receipt_sha256 != _fresh_projection_expected_path(projection)[1]
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid")
    for key, asset in full_expected_assets.items():
        binding = bindings[key]
        if (
            binding.family_id != asset.family_id
            or binding.document_type != asset.document_type
            or binding.source_pdf_sha256 != asset.source_pdf_sha256
            or binding.image_sha256 != asset.image_sha256
            or binding.source_raster_gold_relative_path != asset.gold_relative_path
            or binding.source_raster_gold_sha256 != asset.gold_sha256
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid")
    cases: list[RasterTextEvaluationCaseV1] = []
    for key, asset in expected_assets.items():
        _verified_asset(root, asset, pdfium_version=context.receipt.pdfium_version)
        binding = bindings[key]
        candidate = candidates[key]
        text_gold = _gold_for_asset(root, asset)
        if (
            binding.family_id != asset.family_id
            or binding.document_type != asset.document_type
            or binding.source_pdf_sha256 != asset.source_pdf_sha256
            or binding.image_sha256 != asset.image_sha256
            or binding.source_raster_gold_relative_path != asset.gold_relative_path
            or binding.source_raster_gold_sha256 != asset.gold_sha256
            or binding.derived_text_gold_sha256 != _digest_model(text_gold)
        ):
            raise SyntheticRasterTextActivationError("raster_text_fresh_evaluation_invalid")
        _validate_candidate_provenance(
            candidate=candidate,
            asset=asset,
            raster_text_profile_revision=(execution.raster_text_profile_revision),
            extractor_revision=execution.extractor_revision,
            code="raster_text_fresh_evaluation_invalid",
        )
        cases.append(
            RasterTextEvaluationCaseV1(
                asset_id=asset.asset_id,
                document_type=asset.document_type,
                profile_id=asset.profile_id,
                image_sha256=asset.image_sha256,
                gold=text_gold,
                candidate=candidate,
            )
        )
    if projection is not None and projection.projection_kind.startswith("sealed-stage1-"):
        full_counts = _full_semantic_v3_reference_counts(root=root, context=context, source=source)
        strata = _fresh_sealed_stage1_strata(
            stage_counts=_semantic_v3_counts_by_stratum(cases),
            full_counts=full_counts,
            quality_gate_revision=quality_gate_revision,
        )
        if (
            not allow_inconclusive_stage1
            and _fresh_stage1_decision_from_strata(
                strata, quality_gate_revision=quality_gate_revision
            )
            != "early_pass"
        ):
            raise SyntheticRasterTextActivationError(
                "raster_text_fresh_stage1_not_approval_eligible"
            )
    elif projection is not None and projection.projection_kind == ("sealed-stage2-100-family-v1"):
        _verify_fresh_stage2_parent_decision(
            root=root,
            projection=projection,
            required_selection_revision=required_selection_revision,
            expected_runner_implementation_sha256=runner_implementation_sha256,
            quality_gate_revision=quality_gate_revision,
        )
    return source, execution, bundle, cases


def _verify_v6_candidate_checkpoints(
    *,
    root: Path,
    source: RasterTextFreshSourceGoldReceiptV1,
    execution: RasterTextFreshProjectedCandidateExecutionV1,
    bundle: RasterTextFreshProjectedCandidateBundleV1,
    expected_runner_implementation_sha256: str | None = None,
) -> None:
    """Reopen every V6 observation checkpoint from its immutable identity.

    Candidate bundles are compact evaluation inputs, not an independent
    execution receipt. The V6 lane therefore proves that every bundled
    observation still has the exact source/selection/profile/runner-bound
    checkpoint written by the evaluator. This is intentionally stricter
    than the historical V5 writer and never reinterprets V1--V5 evidence.
    """

    selection_revision = _fresh_scope_selection_revision_from_path(
        root=root,
        relative_path=source.selection_receipt_relative_path,
    )
    if selection_revision != _FRESH_HOLDOUT_SELECTION_REVISION_V2:
        raise SyntheticRasterTextActivationError("raster_text_v6_candidate_checkpoint_invalid")
    context = _fresh_scope_context(
        root=root,
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope=execution.scope,
        selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
        existing_selection_relative_path=source.selection_receipt_relative_path,
    )
    projection = _read_fresh_evaluation_projection(
        root=root,
        context=context,
        relative_path=execution.projection_receipt_relative_path,
        expected_sha256=execution.projection_receipt_sha256,
        source_gold_receipt_relative_path=(execution.source_gold_receipt_relative_path),
        source_gold_receipt_sha256=execution.source_gold_receipt_sha256,
    )
    assets = _fresh_projection_assets(context=context, projection=projection)
    observations = {(item.asset_id, item.profile_id): item for item in bundle.candidates}
    runner_implementation_sha256 = (
        _runner_implementation_sha256()
        if expected_runner_implementation_sha256 is None
        else expected_runner_implementation_sha256
    )
    if not re.fullmatch(_SHA256, runner_implementation_sha256):
        raise SyntheticRasterTextActivationError("raster_text_v6_candidate_checkpoint_invalid")
    if (
        bundle.runner_implementation_sha256 != runner_implementation_sha256
        or len(observations) != len(bundle.candidates)
        or len(assets) != execution.candidate_count
        or len(observations) != execution.candidate_count
    ):
        raise SyntheticRasterTextActivationError("raster_text_v6_candidate_checkpoint_invalid")
    for asset in assets:
        identity = _candidate_checkpoint_identity(
            scope=execution.scope,
            synthetic_raster_receipt_sha256=context.receipt_sha256,
            source_gold_receipt_sha256=execution.source_gold_receipt_sha256,
            selection_receipt_sha256=context.selection_sha256,
            runtime_configuration_sha256=execution.runtime_configuration_sha256,
            raster_text_profile_revision=execution.raster_text_profile_revision,
            extractor_revision=execution.extractor_revision,
            runner_implementation_sha256=runner_implementation_sha256,
            asset=asset,
        )
        relative_path, _identity_sha256 = _candidate_checkpoint_relative_path(identity)
        checkpoint_path = _candidate_checkpoint_path_if_present(root, relative_path)
        if checkpoint_path is None:
            raise SyntheticRasterTextActivationError("raster_text_v6_candidate_checkpoint_invalid")
        _verified_asset(root, asset, pdfium_version=context.receipt.pdfium_version)
        observed = _read_candidate_checkpoint(
            root=root,
            path=checkpoint_path,
            expected_identity=identity,
            asset=asset,
        )
        if observed != observations.get((asset.asset_id, asset.profile_id)):
            raise SyntheticRasterTextActivationError("raster_text_v6_candidate_checkpoint_invalid")


def verify_source_attested_fresh_raster_text_evidence_v6(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    scope: _FreshScope,
    allow_inconclusive_stage1: bool = False,
    expected_runner_implementation_sha256: str | None = None,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> tuple[
    RasterTextFreshSourceGoldReceiptV1,
    RasterTextFreshProjectedCandidateExecutionV1,
    RasterTextFreshProjectedCandidateBundleV1,
    list[RasterTextEvaluationCaseV1],
]:
    """Verify projected evidence from exactly the second train holdout.

    Besides source/Gold/projection/bundle replay, schema V6 reopens every
    immutable candidate checkpoint. A V1 selection or an unprojected
    candidate can therefore never enter the V18 activation lane.
    """

    source, execution, bundle, cases = verify_source_attested_fresh_raster_text_evidence(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope=scope,
        allow_inconclusive_stage1=allow_inconclusive_stage1,
        required_selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
        expected_runner_implementation_sha256=expected_runner_implementation_sha256,
        quality_gate_revision=quality_gate_revision,
    )
    if not isinstance(execution, RasterTextFreshProjectedCandidateExecutionV1) or not isinstance(
        bundle, RasterTextFreshProjectedCandidateBundleV1
    ):
        raise SyntheticRasterTextActivationError("raster_text_v6_candidate_checkpoint_invalid")
    _verify_v6_candidate_checkpoints(
        root=_root(document_preview_root),
        source=source,
        execution=execution,
        bundle=bundle,
        expected_runner_implementation_sha256=expected_runner_implementation_sha256,
    )
    return source, execution, bundle, cases


def verify_source_attested_fresh_raster_text_evidence_v8(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    scope: _FreshScope,
    historical_runner_implementation_sha256: str,
    current_runner_implementation_sha256: str,
    allow_inconclusive_stage1: bool = False,
) -> tuple[
    RasterTextFreshSourceGoldReceiptV1,
    RasterTextFreshProjectedCandidateExecutionV1,
    RasterTextFreshProjectedCandidateBundleV1,
    list[RasterTextEvaluationCaseV1],
]:
    """Replay the one attested V19 chain under the preview-97 policy.

    The historical runner identity is an exact allow-list, not an input knob.
    The current implementation identity is also bound so a later source
    change makes the promoted pointer unavailable until it is revalidated.
    Every source raster, Gold artifact and historical checkpoint is still
    reopened by the V6 verifier; this helper merely makes that immutable
    historical identity explicit instead of pretending it was produced by
    today's orchestration source.
    """

    if (
        historical_runner_implementation_sha256
        != _V19_PREVIEW97_HISTORICAL_RUNNER_IMPLEMENTATION_SHA256
        or current_runner_implementation_sha256 != _runner_implementation_sha256()
    ):
        raise SyntheticRasterTextActivationError("raster_text_v8_evidence_revalidation_invalid")
    try:
        return verify_source_attested_fresh_raster_text_evidence_v6(
            document_preview_root=document_preview_root,
            source_gold_receipt_relative_path=source_gold_receipt_relative_path,
            candidate_execution_relative_path=candidate_execution_relative_path,
            scope=scope,
            allow_inconclusive_stage1=allow_inconclusive_stage1,
            expected_runner_implementation_sha256=(historical_runner_implementation_sha256),
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_v8_evidence_revalidation_invalid"
        ) from None


def _build_fresh_stage1_decision_artifact(
    *,
    root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    source: RasterTextFreshSourceGoldReceiptV1,
    execution: RasterTextFreshCandidateExecutionV1 | RasterTextFreshProjectedCandidateExecutionV1,
    cases: Sequence[RasterTextEvaluationCaseV1],
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> RasterTextFreshSealedStageDecisionReceipt:
    if not isinstance(execution, RasterTextFreshProjectedCandidateExecutionV1):
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_stage_decision_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_stage_decision_invalid",
    )
    selection_revision = _fresh_scope_selection_revision_from_path(
        root=root,
        relative_path=source.selection_receipt_relative_path,
    )
    context = _fresh_scope_context(
        root=root,
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope="fresh-sealed",
        selection_revision=selection_revision,
        existing_selection_relative_path=source.selection_receipt_relative_path,
    )
    projection = _read_fresh_evaluation_projection(
        root=root,
        context=context,
        relative_path=execution.projection_receipt_relative_path,
        expected_sha256=execution.projection_receipt_sha256,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
    )
    if projection.projection_kind != "sealed-stage1-50-family-v1":
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    strata = _fresh_sealed_stage1_strata(
        stage_counts=_semantic_v3_counts_by_stratum(cases),
        full_counts=_full_semantic_v3_reference_counts(root=root, context=context, source=source),
        quality_gate_revision=quality_gate_revision,
    )
    common: dict[str, object] = {
        "metric_revision": RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        "source_gold_receipt_relative_path": source_gold_receipt_relative_path,
        "source_gold_receipt_sha256": hashlib.sha256(source_raw).hexdigest(),
        "candidate_execution_relative_path": candidate_execution_relative_path,
        "candidate_execution_sha256": hashlib.sha256(execution_raw).hexdigest(),
        "projection_receipt_relative_path": execution.projection_receipt_relative_path,
        "projection_receipt_sha256": execution.projection_receipt_sha256,
        "decision": _fresh_stage1_decision_from_strata(
            strata, quality_gate_revision=quality_gate_revision
        ),
        "strata": strata,
    }
    if quality_gate_revision is not None:
        return RasterTextFreshSealedStageDecisionReceiptV2.model_validate(
            common | {"quality_gate_revision": quality_gate_revision}
        )
    return RasterTextFreshSealedStageDecisionReceiptV1.model_validate(common)


def write_fresh_raster_text_sealed_stage1_decision(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    selection_revision: _FreshSelectionRevision = _FRESH_HOLDOUT_SELECTION_REVISION,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> tuple[str, str, _FreshSealedStageDecision]:
    """Recompute and immutably record the result-blind stage-1 decision."""

    root = _root(document_preview_root)
    source, execution, _bundle, cases = verify_source_attested_fresh_raster_text_evidence(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-sealed",
        allow_inconclusive_stage1=True,
        required_selection_revision=selection_revision,
        quality_gate_revision=quality_gate_revision,
    )
    artifact = _build_fresh_stage1_decision_artifact(
        root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        source=source,
        execution=execution,
        cases=cases,
        quality_gate_revision=quality_gate_revision,
    )
    digest = _digest_model(artifact)
    relative = _fresh_stage_decision_expected_path(digest)
    try:
        written, written_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_stage_decision_write_failed"
        ) from None
    if written != relative or written_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_write_failed")
    return relative, digest, artifact.decision


def verify_fresh_raster_text_sealed_stage1_decision(
    *,
    document_preview_root: Path,
    decision_relative_path: str,
    expected_sha256: str | None = None,
    required_selection_revision: _FreshSelectionRevision = (_FRESH_HOLDOUT_SELECTION_REVISION),
    expected_runner_implementation_sha256: str | None = None,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> RasterTextFreshSealedStageDecisionReceipt:
    """Recompute every count, bound, flag and decision from source/candidates."""

    root = _root(document_preview_root)
    observed, _observed_sha256 = _read_fresh_sealed_stage_decision(
        root=root,
        relative_path=decision_relative_path,
        expected_sha256=expected_sha256,
    )
    if quality_gate_revision is None:
        if type(observed) is not RasterTextFreshSealedStageDecisionReceiptV1:
            raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    elif (
        type(observed) is not RasterTextFreshSealedStageDecisionReceiptV2
        or observed.quality_gate_revision != quality_gate_revision
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    source, execution, _bundle, cases = verify_source_attested_fresh_raster_text_evidence(
        document_preview_root=root,
        source_gold_receipt_relative_path=(observed.source_gold_receipt_relative_path),
        candidate_execution_relative_path=(observed.candidate_execution_relative_path),
        scope="fresh-sealed",
        allow_inconclusive_stage1=True,
        required_selection_revision=required_selection_revision,
        expected_runner_implementation_sha256=expected_runner_implementation_sha256,
        quality_gate_revision=quality_gate_revision,
    )
    expected = _build_fresh_stage1_decision_artifact(
        root=root,
        source_gold_receipt_relative_path=(observed.source_gold_receipt_relative_path),
        candidate_execution_relative_path=(observed.candidate_execution_relative_path),
        source=source,
        execution=execution,
        cases=cases,
        quality_gate_revision=quality_gate_revision,
    )
    if expected != observed:
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    return observed


def verify_fresh_raster_text_sealed_stage1_decision_v6(
    *,
    document_preview_root: Path,
    decision_relative_path: str,
    expected_sha256: str | None = None,
    expected_runner_implementation_sha256: str | None = None,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> RasterTextFreshSealedStageDecisionReceipt:
    """Recompute a stage-1 decision from exactly selection generation V2."""

    return verify_fresh_raster_text_sealed_stage1_decision(
        document_preview_root=document_preview_root,
        decision_relative_path=decision_relative_path,
        expected_sha256=expected_sha256,
        required_selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
        expected_runner_implementation_sha256=expected_runner_implementation_sha256,
        quality_gate_revision=quality_gate_revision,
    )


def _verify_fresh_stage2_parent_decision(
    *,
    root: Path,
    projection: RasterTextFreshEvaluationProjectionReceiptV1,
    required_selection_revision: _FreshSelectionRevision = (_FRESH_HOLDOUT_SELECTION_REVISION),
    expected_runner_implementation_sha256: str | None = None,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> None:
    relative_path = projection.parent_stage1_decision_relative_path
    expected_sha256 = projection.parent_stage1_decision_sha256
    if relative_path is None or expected_sha256 is None:
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    observed = verify_fresh_raster_text_sealed_stage1_decision(
        document_preview_root=root,
        decision_relative_path=relative_path,
        expected_sha256=expected_sha256,
        required_selection_revision=required_selection_revision,
        expected_runner_implementation_sha256=expected_runner_implementation_sha256,
        quality_gate_revision=quality_gate_revision,
    )
    if observed.decision != "continue":
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")
    if (
        projection.full_source_gold_receipt_relative_path
        != observed.source_gold_receipt_relative_path
        or projection.full_source_gold_receipt_sha256 != observed.source_gold_receipt_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_stage_decision_invalid")


def write_raster_text_smoke_quality_screen(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    metric_revision: _SemanticMetricRevision | None = None,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
) -> tuple[str, str, bool]:
    """Record a conservative 24-asset quality screen before full splits.

    The source family contains two assets for every document-type/image-profile
    stratum.  Reusing the sealed thresholds here is safe because this function
    can only reject a candidate early; it cannot write an approval pointer or
    substitute for the sealed evaluation.
    """

    semantic_generation = _semantic_metric_generation(metric_revision)
    if quality_gate_revision is not None and (
        semantic_generation != "v3" or quality_gate_revision != _V19_PREVIEW97_QUALITY_GATE_REVISION
    ):
        raise SyntheticRasterTextActivationError("raster_text_smoke_quality_screen_invalid")
    root = _root(document_preview_root)
    source, execution, _bundle, cases = _load_source_gold_cases(
        root=root,
        source_gold_relative_path=source_gold_receipt_relative_path,
        execution_relative_path=candidate_execution_relative_path,
    )
    if source.scope != "smoke" or execution.scope != "smoke":
        raise SyntheticRasterTextActivationError("raster_text_smoke_quality_screen_invalid")
    try:
        if semantic_generation == "v3":
            scores = evaluate_synthetic_raster_semantic_text_gates_v3(cases)
        elif semantic_generation == "v2":
            scores = evaluate_synthetic_raster_semantic_text_gates_v2(cases)
        else:
            scores = evaluate_synthetic_raster_text_gates(cases)
    except ValueError:
        raise SyntheticRasterTextActivationError(
            "raster_text_smoke_quality_screen_invalid"
        ) from None
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_smoke_quality_screen_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_smoke_quality_screen_invalid",
    )
    if semantic_generation == "v3":
        gate_passed = (
            synthetic_raster_semantic_text_preview_draft_gate_passes_v1(scores)
            if quality_gate_revision is not None
            else synthetic_raster_semantic_text_gate_passes_v3(scores)
        )
    elif semantic_generation == "v2":
        gate_passed = synthetic_raster_semantic_text_gate_passes_v2(scores)
    else:
        gate_passed = synthetic_raster_text_gate_passes(scores)
    artifact: (
        RasterTextSmokeQualityScreenV1
        | RasterTextSmokeQualityScreenV2
        | RasterTextSmokeQualityScreenV3
        | RasterTextSmokeQualityScreenV4
    )
    if semantic_generation == "v3" and quality_gate_revision is not None:
        artifact = RasterTextSmokeQualityScreenV4(
            metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
            quality_gate_revision=quality_gate_revision,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
            raster_text_profile_revision=execution.raster_text_profile_revision,
            runtime_configuration_sha256=execution.runtime_configuration_sha256,
            extractor_revision=execution.extractor_revision,
            scores=scores,
            gate_passed=gate_passed,
        )
    elif semantic_generation == "v3":
        artifact = RasterTextSmokeQualityScreenV3(
            metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
            raster_text_profile_revision=execution.raster_text_profile_revision,
            runtime_configuration_sha256=execution.runtime_configuration_sha256,
            extractor_revision=execution.extractor_revision,
            scores=scores,
            gate_passed=gate_passed,
        )
    elif semantic_generation == "v2":
        artifact = RasterTextSmokeQualityScreenV2(
            metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
            raster_text_profile_revision=execution.raster_text_profile_revision,
            runtime_configuration_sha256=execution.runtime_configuration_sha256,
            extractor_revision=execution.extractor_revision,
            scores=scores,
            gate_passed=gate_passed,
        )
    else:
        artifact = RasterTextSmokeQualityScreenV1(
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
            raster_text_profile_revision=execution.raster_text_profile_revision,
            runtime_configuration_sha256=execution.runtime_configuration_sha256,
            extractor_revision=execution.extractor_revision,
            scores=scores,
            gate_passed=gate_passed,
        )
    digest = _digest_model(artifact)
    evidence_directory = {
        "v2": "semantic-smoke-quality-screens-v2",
        "v3": "semantic-smoke-quality-screens-v3",
        None: "smoke-quality-screens",
    }[semantic_generation]
    if quality_gate_revision is not None:
        evidence_directory = "semantic-smoke-quality-screens-v4"
    relative = (
        f"synthetic-raster-text-v1/{evidence_directory}/"
        f"{artifact.raster_text_profile_revision.rsplit(':', 1)[1]}/{digest}.json"
    )
    try:
        written, observed_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_smoke_quality_screen_write_failed"
        ) from None
    if written != relative or observed_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_smoke_quality_screen_write_failed")
    return relative, digest, gate_passed


def _calibrated_floor(
    cases: Sequence[RasterTextEvaluationCaseV1],
    *,
    metric_revision: _SemanticMetricRevision | None = None,
) -> float:
    semantic_generation = _semantic_metric_generation(metric_revision)
    eligible: list[float] = []
    for case in cases:
        if semantic_generation == "v3":
            counts = evaluate_synthetic_raster_semantic_text_case_v3(case)
        elif semantic_generation == "v2":
            counts = evaluate_synthetic_raster_semantic_text_case_v2(case)
        else:
            counts = evaluate_synthetic_raster_text_case(case)
        if (
            counts.character_reference > 0
            and counts.character_errors / counts.character_reference <= 0.02
            and counts.critical_matches == counts.critical_total
            and case.candidate.pages
            and all(page.confidence is not None for page in case.candidate.pages)
        ):
            eligible.append(
                min(
                    float(page.confidence)
                    for page in case.candidate.pages
                    if page.confidence is not None
                )
            )
    if not eligible:
        raise SyntheticRasterTextActivationError("raster_text_calibration_gate_failed")
    return min(eligible)


def write_raster_text_calibration_receipt(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    activated_profile: RasterTextOfflineProfile,
    activated_runtime_configuration_sha256: str,
    metric_revision: _SemanticMetricRevision | None = None,
) -> tuple[str, str]:
    semantic_generation = _semantic_metric_generation(metric_revision)
    root = _root(document_preview_root)
    source, execution, _bundle, cases = verify_source_attested_raster_text_evidence(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="calibration",
    )
    if source.scope != "calibration":
        raise SyntheticRasterTextActivationError("raster_text_calibration_invalid")
    floor = _calibrated_floor(cases, metric_revision=metric_revision)
    if activated_profile.minimum_page_confidence != floor:
        raise SyntheticRasterTextActivationError("raster_text_calibration_invalid")
    source_sha = hashlib.sha256(
        _read(
            root,
            source_gold_receipt_relative_path,
            limit=_MAX_RECEIPT_BYTES,
            code="raster_text_calibration_invalid",
        )
    ).hexdigest()
    execution_sha = hashlib.sha256(
        _read(
            root,
            candidate_execution_relative_path,
            limit=_MAX_RECEIPT_BYTES,
            code="raster_text_calibration_invalid",
        )
    ).hexdigest()
    artifact: (
        RasterTextCalibrationReceiptV1
        | RasterTextCalibrationReceiptV2
        | RasterTextCalibrationReceiptV3
    )
    if semantic_generation == "v3":
        artifact = RasterTextCalibrationReceiptV3(
            metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
            raster_text_profile_revision=activated_profile.raster_text_profile_revision,
            runtime_configuration_sha256=activated_runtime_configuration_sha256,
            calibration_raster_text_profile_revision=execution.raster_text_profile_revision,
            calibration_runtime_configuration_sha256=execution.runtime_configuration_sha256,
            extractor_revision=activated_profile.extractor_revision,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_sha,
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=execution_sha,
            minimum_page_confidence=floor,
            latency_p95_seconds=execution.latency_p95_seconds,
        )
    elif semantic_generation == "v2":
        artifact = RasterTextCalibrationReceiptV2(
            metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
            raster_text_profile_revision=activated_profile.raster_text_profile_revision,
            runtime_configuration_sha256=activated_runtime_configuration_sha256,
            calibration_raster_text_profile_revision=execution.raster_text_profile_revision,
            calibration_runtime_configuration_sha256=execution.runtime_configuration_sha256,
            extractor_revision=activated_profile.extractor_revision,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_sha,
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=execution_sha,
            minimum_page_confidence=floor,
            latency_p95_seconds=execution.latency_p95_seconds,
        )
    else:
        artifact = RasterTextCalibrationReceiptV1(
            raster_text_profile_revision=activated_profile.raster_text_profile_revision,
            runtime_configuration_sha256=activated_runtime_configuration_sha256,
            calibration_raster_text_profile_revision=execution.raster_text_profile_revision,
            calibration_runtime_configuration_sha256=execution.runtime_configuration_sha256,
            extractor_revision=activated_profile.extractor_revision,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_sha,
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=execution_sha,
            minimum_page_confidence=floor,
            latency_p95_seconds=execution.latency_p95_seconds,
        )
    digest = _digest_model(artifact)
    evidence_directory = {
        "v2": "semantic-calibration-receipts-v2",
        "v3": "semantic-calibration-receipts-v3",
        None: "calibration-receipts",
    }[semantic_generation]
    relative = f"synthetic-raster-text-v1/{evidence_directory}/{digest}.json"
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError("raster_text_calibration_write_failed") from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_calibration_write_failed")
    return relative, digest


def write_raster_text_sealed_evaluation(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfile,
    runtime_configuration_sha256: str,
    metric_revision: _SemanticMetricRevision | None = None,
) -> tuple[str, str]:
    semantic_generation = _semantic_metric_generation(metric_revision)
    root = _root(document_preview_root)
    try:
        source, execution, _bundle, cases = verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=source_gold_receipt_relative_path,
            candidate_execution_relative_path=candidate_execution_relative_path,
            scope="sealed",
            expected_raster_text_profile_revision=profile.raster_text_profile_revision,
            expected_runtime_configuration_sha256=runtime_configuration_sha256,
            expected_extractor_revision=profile.extractor_revision,
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError("raster_text_sealed_evaluation_invalid") from None
    calibration_raw = _read(
        root,
        calibration_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_sealed_evaluation_invalid",
    )
    warm_smoke_raw = _read(
        root,
        warm_service_smoke_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_sealed_evaluation_invalid",
    )
    try:
        if semantic_generation == "v3":
            calibration: (
                RasterTextCalibrationReceiptV1
                | RasterTextCalibrationReceiptV2
                | RasterTextCalibrationReceiptV3
            ) = RasterTextCalibrationReceiptV3.model_validate_json(calibration_raw)
        elif semantic_generation == "v2":
            calibration = RasterTextCalibrationReceiptV2.model_validate_json(calibration_raw)
        else:
            calibration = RasterTextCalibrationReceiptV1.model_validate_json(calibration_raw)
        warm_smoke = RasterTextWarmServiceSmokeReceiptV1.model_validate_json(warm_smoke_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError("raster_text_sealed_evaluation_invalid") from None
    if calibration_raw != _canonical_json(
        calibration.model_dump(mode="json", warnings=False)
    ) or warm_smoke_raw != _canonical_json(warm_smoke.model_dump(mode="json", warnings=False)):
        raise SyntheticRasterTextActivationError("raster_text_sealed_evaluation_invalid")
    calibration_source_raw = _read(
        root,
        calibration.source_gold_receipt_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_sealed_evaluation_invalid",
    )
    calibration_execution_raw = _read(
        root,
        calibration.candidate_execution_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_sealed_evaluation_invalid",
    )
    try:
        calibration_source, calibration_execution, calibration_bundle, _calibration_cases = (
            verify_source_attested_raster_text_evidence(
                document_preview_root=root,
                source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
                candidate_execution_relative_path=calibration.candidate_execution_path,
                scope="calibration",
                expected_raster_text_profile_revision=(
                    calibration.calibration_raster_text_profile_revision
                ),
                expected_runtime_configuration_sha256=(
                    calibration.calibration_runtime_configuration_sha256
                ),
                expected_extractor_revision=calibration.extractor_revision,
            )
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError("raster_text_sealed_evaluation_invalid") from None
    warm_source_raw = _read(
        root,
        warm_smoke.source_gold_receipt_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_sealed_evaluation_invalid",
    )
    warm_execution_raw = _read(
        root,
        warm_smoke.candidate_execution_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_sealed_evaluation_invalid",
    )
    try:
        warm_source, warm_execution, warm_bundle, _warm_cases = (
            verify_source_attested_raster_text_evidence(
                document_preview_root=root,
                source_gold_receipt_relative_path=warm_smoke.source_gold_receipt_path,
                candidate_execution_relative_path=warm_smoke.candidate_execution_path,
                scope="smoke",
                expected_raster_text_profile_revision=profile.raster_text_profile_revision,
                expected_runtime_configuration_sha256=runtime_configuration_sha256,
                expected_extractor_revision=profile.extractor_revision,
            )
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError("raster_text_sealed_evaluation_invalid") from None
    if (
        source.scope != "sealed"
        or execution.raster_text_profile_revision != profile.raster_text_profile_revision
        or execution.runtime_configuration_sha256 != runtime_configuration_sha256
        or execution.extractor_revision != profile.extractor_revision
        or calibration.raster_text_profile_revision != profile.raster_text_profile_revision
        or calibration.runtime_configuration_sha256 != runtime_configuration_sha256
        or calibration.minimum_page_confidence != profile.minimum_page_confidence
        or hashlib.sha256(calibration_source_raw).hexdigest()
        != calibration.source_gold_receipt_sha256
        or hashlib.sha256(calibration_execution_raw).hexdigest()
        != calibration.candidate_execution_sha256
        or calibration_source.scope != "calibration"
        or calibration_execution.scope != "calibration"
        or calibration_bundle.scope != "calibration"
        or warm_smoke.raster_text_profile_revision != profile.raster_text_profile_revision
        or warm_smoke.runtime_configuration_sha256 != runtime_configuration_sha256
        or warm_smoke.extractor_revision != profile.extractor_revision
        or hashlib.sha256(warm_source_raw).hexdigest() != warm_smoke.source_gold_receipt_sha256
        or hashlib.sha256(warm_execution_raw).hexdigest() != warm_smoke.candidate_execution_sha256
        or warm_source.scope != "smoke"
        or warm_execution.scope != "smoke"
        or warm_bundle.scope != "smoke"
        or warm_execution.candidate_bundle_sha256 != warm_smoke.candidate_bundle_sha256
        or warm_execution.raster_text_profile_revision != profile.raster_text_profile_revision
        or warm_execution.runtime_configuration_sha256 != runtime_configuration_sha256
        or warm_execution.extractor_revision != profile.extractor_revision
        or warm_bundle.raster_text_profile_revision != profile.raster_text_profile_revision
        or warm_bundle.runtime_configuration_sha256 != runtime_configuration_sha256
        or warm_bundle.extractor_revision != profile.extractor_revision
        or warm_smoke.startup_latency_seconds > raster_text_service_startup_timeout_seconds(profile)
        or warm_smoke.startup_timeout_seconds
        != raster_text_service_startup_timeout_seconds(profile)
        or warm_smoke.request_timeout_seconds != min(profile.sidecar_timeout_seconds, 30)
        or raster_text_exact_result_set_sha256(
            [(item.asset_id, item.profile_id, item.candidate) for item in warm_bundle.candidates]
        )
        != warm_smoke.equality_set_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_sealed_evaluation_invalid")
    if semantic_generation == "v3":
        scores = evaluate_synthetic_raster_semantic_text_gates_v3(cases)
        gate_passed = synthetic_raster_semantic_text_gate_passes_v3(scores)
    elif semantic_generation == "v2":
        scores = evaluate_synthetic_raster_semantic_text_gates_v2(cases)
        gate_passed = synthetic_raster_semantic_text_gate_passes_v2(scores)
    else:
        scores = evaluate_synthetic_raster_text_gates(cases)
        gate_passed = synthetic_raster_text_gate_passes(scores)
    if not gate_passed:
        raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
    source_sha = hashlib.sha256(
        _read(
            root,
            source_gold_receipt_relative_path,
            limit=_MAX_RECEIPT_BYTES,
            code="raster_text_sealed_evaluation_invalid",
        )
    ).hexdigest()
    execution_sha = hashlib.sha256(
        _read(
            root,
            candidate_execution_relative_path,
            limit=_MAX_RECEIPT_BYTES,
            code="raster_text_sealed_evaluation_invalid",
        )
    ).hexdigest()
    calibration_sha = hashlib.sha256(calibration_raw).hexdigest()
    artifact: (
        RasterTextSealedEvaluationV1 | RasterTextSealedEvaluationV2 | RasterTextSealedEvaluationV3
    )
    if semantic_generation == "v3":
        artifact = RasterTextSealedEvaluationV3(
            metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
            raster_text_profile_revision=profile.raster_text_profile_revision,
            runtime_configuration_sha256=runtime_configuration_sha256,
            extractor_revision=profile.extractor_revision,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_sha,
            calibration_receipt_path=calibration_receipt_relative_path,
            calibration_receipt_sha256=calibration_sha,
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=execution_sha,
            warm_service_smoke_receipt_path=warm_service_smoke_receipt_relative_path,
            warm_service_smoke_receipt_sha256=hashlib.sha256(warm_smoke_raw).hexdigest(),
            latency_p95_seconds=execution.latency_p95_seconds,
            service_warm_latency_p95_seconds=warm_smoke.warm_latency_p95_seconds,
            minimum_page_confidence=profile.minimum_page_confidence,
            scores=[RasterTextSealedProfileScoreV3(**item.model_dump()) for item in scores],
        )
    elif semantic_generation == "v2":
        artifact = RasterTextSealedEvaluationV2(
            metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
            raster_text_profile_revision=profile.raster_text_profile_revision,
            runtime_configuration_sha256=runtime_configuration_sha256,
            extractor_revision=profile.extractor_revision,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_sha,
            calibration_receipt_path=calibration_receipt_relative_path,
            calibration_receipt_sha256=calibration_sha,
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=execution_sha,
            warm_service_smoke_receipt_path=warm_service_smoke_receipt_relative_path,
            warm_service_smoke_receipt_sha256=hashlib.sha256(warm_smoke_raw).hexdigest(),
            latency_p95_seconds=execution.latency_p95_seconds,
            service_warm_latency_p95_seconds=warm_smoke.warm_latency_p95_seconds,
            minimum_page_confidence=profile.minimum_page_confidence,
            scores=[RasterTextSealedProfileScoreV2(**item.model_dump()) for item in scores],
        )
    else:
        artifact = RasterTextSealedEvaluationV1(
            raster_text_profile_revision=profile.raster_text_profile_revision,
            runtime_configuration_sha256=runtime_configuration_sha256,
            extractor_revision=profile.extractor_revision,
            source_gold_receipt_path=source_gold_receipt_relative_path,
            source_gold_receipt_sha256=source_sha,
            calibration_receipt_path=calibration_receipt_relative_path,
            calibration_receipt_sha256=calibration_sha,
            candidate_execution_path=candidate_execution_relative_path,
            candidate_execution_sha256=execution_sha,
            warm_service_smoke_receipt_path=warm_service_smoke_receipt_relative_path,
            warm_service_smoke_receipt_sha256=hashlib.sha256(warm_smoke_raw).hexdigest(),
            latency_p95_seconds=execution.latency_p95_seconds,
            service_warm_latency_p95_seconds=warm_smoke.warm_latency_p95_seconds,
            minimum_page_confidence=profile.minimum_page_confidence,
            scores=[RasterTextSealedProfileScoreV1(**item.model_dump()) for item in scores],
        )
    digest = _digest_model(artifact)
    evidence_directory = {
        "v2": "semantic-sealed-evaluations-v2",
        "v3": "semantic-sealed-evaluations-v3",
        None: "sealed-evaluations",
    }[semantic_generation]
    relative = (
        f"synthetic-raster-text-v1/{evidence_directory}/"
        f"{profile.raster_text_profile_revision.rsplit(':', 1)[1]}/{digest}.json"
    )
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_sealed_evaluation_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_sealed_evaluation_write_failed")
    return relative, digest


def write_raster_semantic_text_smoke_quality_screen_v2(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
) -> tuple[str, str, bool]:
    """Write only semantic-metric V2 smoke evidence."""

    return write_raster_text_smoke_quality_screen(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
    )


def write_raster_semantic_text_calibration_receipt_v2(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    activated_profile: RasterTextOfflineProfile,
    activated_runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write only semantic-metric V2 calibration evidence."""

    return write_raster_text_calibration_receipt(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        activated_profile=activated_profile,
        activated_runtime_configuration_sha256=activated_runtime_configuration_sha256,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
    )


def write_raster_semantic_text_sealed_evaluation_v2(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfile,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write only semantic-metric V2 sealed evidence."""

    return write_raster_text_sealed_evaluation(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        calibration_receipt_relative_path=calibration_receipt_relative_path,
        warm_service_smoke_receipt_relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V2,
    )


def write_raster_semantic_text_smoke_quality_screen_v3(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
) -> tuple[str, str, bool]:
    """Write only semantic-metric V3 smoke evidence."""

    return write_raster_text_smoke_quality_screen(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )


def write_raster_semantic_text_smoke_quality_screen_preview97_v1(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
) -> tuple[str, str, bool]:
    """Write a V3 smoke screen under the explicit preview-draft 97% Gate."""

    return write_raster_text_smoke_quality_screen(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        quality_gate_revision=_V19_PREVIEW97_QUALITY_GATE_REVISION,
    )


def write_raster_semantic_text_calibration_receipt_v3(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    activated_profile: RasterTextOfflineProfile,
    activated_runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write only semantic-metric V3 calibration evidence."""

    return write_raster_text_calibration_receipt(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        activated_profile=activated_profile,
        activated_runtime_configuration_sha256=activated_runtime_configuration_sha256,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )


def write_raster_semantic_text_sealed_evaluation_v3(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfile,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write only semantic-metric V3 sealed evidence."""

    return write_raster_text_sealed_evaluation(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        calibration_receipt_relative_path=calibration_receipt_relative_path,
        warm_service_smoke_receipt_relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )


def _v14_ocr_material(profile: RasterTextOfflineProfileV14) -> dict[str, object]:
    return profile.model_dump(
        mode="json",
        warnings=False,
        exclude={
            "minimum_page_confidence",
            "profile_fingerprint_sha256",
            "raster_text_profile_revision",
        },
    )


def _v15_ocr_material(profile: RasterTextOfflineProfileV15) -> dict[str, object]:
    return profile.model_dump(
        mode="json",
        warnings=False,
        exclude={
            "minimum_page_confidence",
            "profile_fingerprint_sha256",
            "raster_text_profile_revision",
        },
    )


def _v18_ocr_material(profile: RasterTextOfflineProfileV18) -> dict[str, object]:
    """Return the immutable V18 detector/recognizer/runtime material tuple."""

    return profile.model_dump(
        mode="json",
        warnings=False,
        exclude={
            "minimum_page_confidence",
            "profile_fingerprint_sha256",
            "raster_text_profile_revision",
        },
    )


def _v19_ocr_material(profile: RasterTextOfflineProfileV19) -> dict[str, object]:
    """Return the immutable fine-tuned recognizer serving material tuple."""

    return profile.model_dump(
        mode="json",
        warnings=False,
        exclude={
            "minimum_page_confidence",
            "profile_fingerprint_sha256",
            "raster_text_profile_revision",
        },
    )


def _resolved_v14_calibration_profile(
    *, root: Path, runtime_configuration_relative_path: str
) -> tuple[RasterTextOfflineProfileV14, str]:
    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or not isinstance(resolution.profile, RasterTextOfflineProfileV14)
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_invalid")
    return resolution.profile, resolution.runtime_configuration_sha256


def _resolved_v15_calibration_profile(
    *, root: Path, runtime_configuration_relative_path: str
) -> tuple[RasterTextOfflineProfileV15, str]:
    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or not isinstance(resolution.profile, RasterTextOfflineProfileV15)
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_invalid")
    return resolution.profile, resolution.runtime_configuration_sha256


def _resolved_v18_calibration_profile(
    *, root: Path, runtime_configuration_relative_path: str
) -> tuple[RasterTextOfflineProfileV18, str]:
    """Resolve only the pinned mobile-detector/server-recognizer profile."""

    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or not isinstance(resolution.profile, RasterTextOfflineProfileV18)
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_calibration_invalid")
    return resolution.profile, resolution.runtime_configuration_sha256


def _resolved_v19_calibration_profile(
    *, root: Path, runtime_configuration_relative_path: str
) -> tuple[RasterTextOfflineProfileV19, str]:
    """Resolve only the mobile detector plus leakage-safe V2 recognizer."""

    resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=runtime_configuration_relative_path,
    )
    if (
        resolution.status != "available"
        or not isinstance(resolution.profile, RasterTextOfflineProfileV19)
        or resolution.runtime_configuration_sha256 is None
    ):
        raise SyntheticRasterTextActivationError("raster_text_v7_fresh_calibration_invalid")
    return resolution.profile, resolution.runtime_configuration_sha256


class _V19FineTuneChildSuccessV1(BaseModel):
    """Strict, content-free handshake emitted by the fixed training tool."""

    model_config = ConfigDict(extra="forbid", strict=True)

    release_relative_path: str = Field(min_length=1, max_length=512)
    release_sha256: str = Field(pattern=_SHA256)
    stage: Literal["leakage-safe-v2"]
    status: Literal["published", "reused"]
    tuning_scope: Literal["head"]

    @model_validator(mode="after")
    def validate_relative_path(self) -> _V19FineTuneChildSuccessV1:
        path = PurePosixPath(self.release_relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != self.release_relative_path
        ):
            raise ValueError("public fine-tune child release path is invalid")
        return self


@dataclass
class _BoundedV19FineTuneChildStdout:
    """Drain child stdout while retaining at most one small status record."""

    data: bytearray = field(default_factory=bytearray)
    overflow: bool = False

    def consume(self, stream: BinaryIO) -> None:
        try:
            while True:
                block = stream.read(1024)
                if not block:
                    return
                remaining = _MAX_V19_FINE_TUNE_CHILD_STDOUT_BYTES + 1 - len(self.data)
                if remaining > 0:
                    self.data.extend(block[:remaining])
                if len(block) > remaining or len(self.data) > _MAX_V19_FINE_TUNE_CHILD_STDOUT_BYTES:
                    self.overflow = True
        except (OSError, ValueError):
            self.overflow = True


def _v19_fine_tune_repo_and_tool() -> tuple[Path, Path, str]:
    """Resolve the one tracked tool without accepting a caller path."""

    try:
        module = Path(__file__)
        if module.is_symlink():
            raise OSError
        repository = module.resolve(strict=True).parents[3]
        unresolved_tool = repository / _V19_FINE_TUNE_TOOL_RELATIVE_PATH
        if unresolved_tool.is_symlink():
            raise OSError
        tool = unresolved_tool.resolve(strict=True)
        tool.relative_to(repository)
        if not tool.is_file() or tool != unresolved_tool:
            raise OSError
        before_sha256 = hashlib.sha256(tool.read_bytes()).hexdigest()
        return repository, tool, before_sha256
    except (IndexError, OSError, ValueError):
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_unavailable") from None


def _v19_fine_tune_child_environment() -> dict[str, str]:
    """Drop Python/model/network overrides and set the fixed CPU contract."""

    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        normalized = key.upper()
        if (
            normalized.startswith("PYTHON")
            or normalized.startswith("CUSTOMS_")
            or normalized.startswith("PADDLE")
            or normalized.startswith("FLAGS_")
            or normalized.startswith("OMP_")
            or normalized.startswith("MKL_")
            or normalized.startswith("ONEDNN_")
            or normalized
            in {
                "ALL_PROXY",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "FTP_PROXY",
                "NO_PROXY",
            }
        ):
            continue
        environment[key] = value
    environment.update(
        {
            "FLAGS_json_format_model": "1",
            "OMP_NUM_THREADS": "6",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _v19_fine_tune_hidden_startupinfo() -> object:
    startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return startupinfo


def _terminate_v19_fine_tune_child(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=_V19_FINE_TUNE_SUBPROCESS_TERMINATE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_V19_FINE_TUNE_SUBPROCESS_TERMINATE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return


def _parse_v19_fine_tune_child_stdout(
    bounded: _BoundedV19FineTuneChildStdout,
    *,
    return_code: int,
) -> _V19FineTuneChildSuccessV1:
    raw = bytes(bounded.data)
    if (
        bounded.overflow
        or not raw
        or len(raw) > _MAX_V19_FINE_TUNE_CHILD_STDOUT_BYTES
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
    ):
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_output_invalid")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SyntheticRasterTextActivationError(
            "public_finetune_subprocess_output_invalid"
        ) from None
    if return_code == 2:
        if (
            isinstance(payload, dict)
            and set(payload) == {"code", "status"}
            and payload.get("status") == "failed"
            and isinstance(payload.get("code"), str)
            and _SAFE_V19_FINE_TUNE_CHILD_CODE.fullmatch(payload["code"]) is not None
        ):
            raise SyntheticRasterTextActivationError(payload["code"])
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_output_invalid")
    if return_code != 0:
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_failed")
    try:
        return _V19FineTuneChildSuccessV1.model_validate(payload)
    except ValidationError:
        raise SyntheticRasterTextActivationError(
            "public_finetune_subprocess_output_invalid"
        ) from None


def _run_fixed_v19_fine_tune_subprocess(
    root: Path,
) -> VerifiedPublicFineTunedRecognizerReleaseV1:
    """Run V2 training in a clean Python process and reopen its release.

    The only child input is the verified document-preview root.  No OCR text,
    Gold label, score, model name, hash or training setting crosses stdin or
    argv.  The child independently rebuilds/reopens its fixed source-attested
    corpus and denies all socket routes before importing Paddle.
    """

    verified_root = _root(root)
    repository, tool, tool_sha256 = _v19_fine_tune_repo_and_tool()
    try:
        interpreter = Path(sys.executable)
        if os.name == "nt" and interpreter.is_symlink():
            raise OSError
        interpreter = interpreter.resolve(strict=True)
        if not interpreter.is_file():
            raise OSError
        command = [
            str(interpreter),
            "-I",
            "-B",
            str(tool),
            "--document-preview-root",
            str(verified_root),
            "--stage",
            "leakage-safe-v2",
        ]
        common: dict[str, object] = {
            "close_fds": True,
            "cwd": str(repository),
            "env": _v19_fine_tune_child_environment(),
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            common.update(
                {
                    "creationflags": 0x08000000,  # CREATE_NO_WINDOW
                    "startupinfo": _v19_fine_tune_hidden_startupinfo(),
                }
            )
        else:
            common["start_new_session"] = True
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen
        process = popen(command, **common)
    except OSError:
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_unavailable") from None
    if process.stdout is None:
        _terminate_v19_fine_tune_child(process)
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_output_invalid")
    bounded = _BoundedV19FineTuneChildStdout()
    reader = threading.Thread(target=bounded.consume, args=(process.stdout,), daemon=True)
    reader.start()
    try:
        return_code = process.wait(timeout=_V19_FINE_TUNE_SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_v19_fine_tune_child(process)
        reader.join(timeout=_V19_FINE_TUNE_SUBPROCESS_TERMINATE_SECONDS)
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_timeout") from None
    reader.join(timeout=_V19_FINE_TUNE_SUBPROCESS_TERMINATE_SECONDS)
    if reader.is_alive():
        _terminate_v19_fine_tune_child(process)
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_output_invalid")
    try:
        if hashlib.sha256(tool.read_bytes()).hexdigest() != tool_sha256:
            raise OSError
    except OSError:
        raise SyntheticRasterTextActivationError("public_finetune_subprocess_tool_drift") from None
    summary = _parse_v19_fine_tune_child_stdout(bounded, return_code=return_code)
    try:
        release = load_verified_public_finetuned_recognizer_release_v1(
            verified_root, summary.release_sha256
        )
        observed_relative_path = (
            release.release_path.resolve(strict=True).relative_to(verified_root).as_posix()
        )
        if observed_relative_path != summary.release_relative_path:
            raise ValueError
        return release
    except (OSError, ValueError, SourceAttestedPublicRecognizerFineTuneError):
        raise SyntheticRasterTextActivationError(
            "public_finetune_subprocess_release_invalid"
        ) from None


def _verify_v19_public_finetune_contract(
    *,
    root: Path,
    recognizer_release: VerifiedPublicFineTunedRecognizerReleaseV1,
    model_bundle: VerifiedRasterTextPublicFineTunedPaddleBundleV1 | None = None,
    profile: RasterTextOfflineProfileV19 | None = None,
) -> VerifiedPublicFineTunedRecognizerReleaseV1:
    """Reopen and bind the exact result-blind V2 training material.

    Pydantic validates the detailed receipt shape.  These comparisons add the
    cross-artifact boundary required by activation: the corpus holdout must be
    the same immutable V2 selection used by calibration and sealed scoring,
    and neither a legacy overfit receipt nor a caller-selected recognizer can
    enter V19.
    """

    try:
        release = load_verified_public_finetuned_recognizer_release_v1(
            root, recognizer_release.release_sha256
        )
        receipt = release.receipt
        if not isinstance(receipt, PublicFineTuneReceiptV2):
            raise ValueError
        context = _fresh_scope_context(
            root=root,
            synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
            scope="fresh-calibration",
            selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
            existing_selection_relative_path=(receipt.selection_receipt_relative_path),
        )
        selection = context.selection
        if (
            release.release_sha256 != recognizer_release.release_sha256
            or release.manifest.receipt_sha256 != receipt.receipt_sha256
            or receipt.schema_version != "2.0"
            or receipt.trainer_revision != "public-paddle-jit-ctc-leakage-safe-head-v2"
            or receipt.model_revision != "en-ppocrv5-mobile-rec-source-attested-head-v2"
            or receipt.stage != "leakage-safe-v2"
            or receipt.tuning_scope != "head"
            or receipt.maximum_epochs != 4
            or receipt.overlap_verifier_revision != "selection-v2-all-holdouts-zero-overlap-v1"
            or receipt.train_holdout_overlap_count != 0
            or receipt.train_crop_manifest.family_count != 500
            or receipt.train_crop_manifest.usable_crop_count != 1_000
            or receipt.development_crop_manifest.family_count != 100
            or receipt.development_crop_manifest.usable_crop_count != 200
            or not isinstance(selection, RasterTextFreshScopeSelectionReceiptV2)
            or context.selection_relative_path != receipt.selection_receipt_relative_path
            or context.selection_sha256 != receipt.selection_receipt_sha256
            or selection.parent_selection_receipt_sha256 != receipt.parent_selection_receipt_sha256
        ):
            raise ValueError
        if model_bundle is not None:
            manifest = model_bundle.manifest
            if (
                manifest.source_recognizer_release_sha256 != release.release_sha256
                or manifest.source_recognizer_receipt_sha256 != receipt.receipt_sha256
                or manifest.source_recognizer_training_intent_sha256
                != receipt.training_intent_sha256
                or manifest.source_recognizer_selection_receipt_relative_path
                != receipt.selection_receipt_relative_path
                or manifest.source_recognizer_selection_receipt_sha256
                != receipt.selection_receipt_sha256
                or manifest.source_recognizer_parent_selection_receipt_sha256
                != receipt.parent_selection_receipt_sha256
            ):
                raise ValueError
        if profile is not None and (
            profile.source_recognizer_release_sha256 != release.release_sha256
            or profile.source_recognizer_receipt_sha256 != receipt.receipt_sha256
            or profile.source_recognizer_training_intent_sha256 != receipt.training_intent_sha256
            or profile.source_recognizer_selection_receipt_relative_path
            != receipt.selection_receipt_relative_path
            or profile.source_recognizer_selection_receipt_sha256
            != receipt.selection_receipt_sha256
            or profile.source_recognizer_parent_selection_receipt_sha256
            != receipt.parent_selection_receipt_sha256
        ):
            raise ValueError
        return release
    except SyntheticRasterTextActivationError:
        raise
    except (SourceAttestedPublicRecognizerFineTuneError, ValueError):
        raise SyntheticRasterTextActivationError("public_finetune_v2_contract_invalid") from None


class _VoyageReplayChildSuccessV1(BaseModel):
    """Strict, content-free handshake from the fixed V3 replay tool."""

    model_config = ConfigDict(extra="forbid", strict=True)

    release_relative_path: str = Field(min_length=1, max_length=512)
    release_sha256: str = Field(pattern=_SHA256)
    stage: Literal["voyage-replay-v3"]
    status: Literal["published", "reused"]
    tuning_scope: Literal["ctc_head_only"]

    @model_validator(mode="after")
    def validate_relative_path(self) -> _VoyageReplayChildSuccessV1:
        path = PurePosixPath(self.release_relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != self.release_relative_path
        ):
            raise ValueError("Voyage replay child release path is invalid")
        return self


@dataclass
class _BoundedVoyageReplayChildStdout:
    """Drain child stdout while retaining exactly one small status record."""

    data: bytearray = field(default_factory=bytearray)
    overflow: bool = False

    def consume(self, stream: BinaryIO) -> None:
        try:
            while True:
                block = stream.read(1024)
                if not block:
                    return
                remaining = _MAX_VOYAGE_REPLAY_CHILD_STDOUT_BYTES + 1 - len(self.data)
                if remaining > 0:
                    self.data.extend(block[:remaining])
                if len(block) > remaining or len(self.data) > _MAX_VOYAGE_REPLAY_CHILD_STDOUT_BYTES:
                    self.overflow = True
        except (OSError, ValueError):
            self.overflow = True


def _voyage_replay_repo_and_tool() -> tuple[Path, Path, str]:
    """Resolve the sole tracked replay tool without accepting a caller path."""

    try:
        module = Path(__file__)
        if module.is_symlink():
            raise OSError
        repository = module.resolve(strict=True).parents[3]
        unresolved_tool = repository / _VOYAGE_REPLAY_TOOL_RELATIVE_PATH
        if unresolved_tool.is_symlink():
            raise OSError
        tool = unresolved_tool.resolve(strict=True)
        tool.relative_to(repository)
        if not tool.is_file() or tool != unresolved_tool:
            raise OSError
        return repository, tool, hashlib.sha256(tool.read_bytes()).hexdigest()
    except (IndexError, OSError, ValueError):
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_unavailable") from None


def _voyage_replay_child_environment() -> dict[str, str]:
    """Use the same scrubbed fixed-CPU environment as the proven V19 child."""

    return _v19_fine_tune_child_environment()


def _terminate_voyage_replay_child(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=_VOYAGE_REPLAY_SUBPROCESS_TERMINATE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_VOYAGE_REPLAY_SUBPROCESS_TERMINATE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return


def _parse_voyage_replay_child_stdout(
    bounded: _BoundedVoyageReplayChildStdout,
    *,
    return_code: int,
) -> _VoyageReplayChildSuccessV1:
    raw = bytes(bounded.data)
    if (
        bounded.overflow
        or not raw
        or len(raw) > _MAX_VOYAGE_REPLAY_CHILD_STDOUT_BYTES
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
    ):
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_output_invalid")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SyntheticRasterTextActivationError(
            "voyage_replay_subprocess_output_invalid"
        ) from None
    if return_code == 2:
        if (
            isinstance(payload, dict)
            and set(payload) == {"code", "status"}
            and payload.get("status") == "failed"
            and isinstance(payload.get("code"), str)
            and _SAFE_VOYAGE_REPLAY_CHILD_CODE.fullmatch(payload["code"]) is not None
        ):
            raise SyntheticRasterTextActivationError(payload["code"])
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_output_invalid")
    if return_code != 0:
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_failed")
    try:
        return _VoyageReplayChildSuccessV1.model_validate(payload)
    except ValidationError:
        raise SyntheticRasterTextActivationError(
            "voyage_replay_subprocess_output_invalid"
        ) from None


def _run_fixed_voyage_replay_subprocess(
    root: Path,
) -> VerifiedVoyageReplayReleaseV1:
    """Create/reuse V3 in a fresh process, then reopen its immutable release."""

    verified_root = _root(root)
    repository, tool, tool_sha256 = _voyage_replay_repo_and_tool()
    try:
        interpreter = Path(sys.executable)
        if os.name == "nt" and interpreter.is_symlink():
            raise OSError
        interpreter = interpreter.resolve(strict=True)
        if not interpreter.is_file():
            raise OSError
        command = [
            str(interpreter),
            "-I",
            "-B",
            str(tool),
            "--document-preview-root",
            str(verified_root),
            "--stage",
            "voyage-replay-v3",
        ]
        common: dict[str, object] = {
            "close_fds": True,
            "cwd": str(repository),
            "env": _voyage_replay_child_environment(),
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            common.update(
                {
                    "creationflags": 0x08000000,  # CREATE_NO_WINDOW
                    "startupinfo": _v19_fine_tune_hidden_startupinfo(),
                }
            )
        else:
            common["start_new_session"] = True
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen
        process = popen(command, **common)
    except OSError:
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_unavailable") from None
    if process.stdout is None:
        _terminate_voyage_replay_child(process)
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_output_invalid")
    bounded = _BoundedVoyageReplayChildStdout()
    reader = threading.Thread(target=bounded.consume, args=(process.stdout,), daemon=True)
    reader.start()
    try:
        return_code = process.wait(timeout=_VOYAGE_REPLAY_SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_voyage_replay_child(process)
        reader.join(timeout=_VOYAGE_REPLAY_SUBPROCESS_TERMINATE_SECONDS)
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_timeout") from None
    reader.join(timeout=_VOYAGE_REPLAY_SUBPROCESS_TERMINATE_SECONDS)
    if reader.is_alive():
        _terminate_voyage_replay_child(process)
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_output_invalid")
    try:
        if hashlib.sha256(tool.read_bytes()).hexdigest() != tool_sha256:
            raise OSError
    except OSError:
        raise SyntheticRasterTextActivationError("voyage_replay_subprocess_tool_drift") from None
    summary = _parse_voyage_replay_child_stdout(bounded, return_code=return_code)
    try:
        release = load_verified_voyage_replay_release_v1(verified_root, summary.release_sha256)
        observed_relative_path = (
            release.release_path.resolve(strict=True).relative_to(verified_root).as_posix()
        )
        if observed_relative_path != summary.release_relative_path:
            raise ValueError
        return release
    except (OSError, ValueError, VoyageReplayError):
        raise SyntheticRasterTextActivationError(
            "voyage_replay_subprocess_release_invalid"
        ) from None


def _verify_v20_voyage_replay_contract(
    *,
    root: Path,
    recognizer_release: VerifiedVoyageReplayReleaseV1,
    model_bundle: VerifiedRasterTextPublicVoyageReplayPaddleBundleV1 | None = None,
    profile: RasterTextOfflineProfileV20 | None = None,
) -> VerifiedVoyageReplayReleaseV1:
    """Reopen and cross-bind the exact result-blind V3 replay release."""

    try:
        release = load_verified_voyage_replay_release_v1(root, recognizer_release.release_sha256)
        receipt = release.receipt
        if not isinstance(receipt, VoyageReplayReceiptV3):
            raise ValueError
        train = receipt.train_corpus_manifest
        development = receipt.development_corpus_manifest
        context = _fresh_scope_context(
            root=root,
            synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
            scope="fresh-calibration",
            selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
            existing_selection_relative_path=train.selection_receipt_relative_path,
        )
        selection = context.selection
        if (
            release.release_sha256 != recognizer_release.release_sha256
            or release.manifest.receipt_sha256 != receipt.receipt_sha256
            or receipt.schema_version != "3.0"
            or receipt.trainer_revision
            != "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3"
            or receipt.model_revision
            != "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3"
            or receipt.output_model_name
            != "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
            or receipt.stage != "voyage-replay-v3"
            or receipt.tuning_scope != "ctc_head_only"
            or receipt.maximum_epochs != 2
            or receipt.batch_size != 30
            or receipt.voyage_per_batch != 20
            or receipt.general_per_batch != 10
            or receipt.batches_per_epoch != 25
            or receipt.trainable_tensor_count != 2
            or receipt.trainable_parameter_count != 52_998
            or receipt.static_roundtrip_verified is not True
            or train.split != "train"
            or train.purpose != "optimizer"
            or train.selected_family_count != 500
            or train.voyage_crop_count != 500
            or train.general_crop_count != 250
            or train.total_crop_count != 750
            or development.split != "development"
            or development.purpose != "validation-only"
            or development.selected_family_count != 100
            or development.voyage_crop_count != 200
            or development.general_crop_count != 200
            or development.total_crop_count != 400
            or train.holdout_overlap_count != 0
            or development.holdout_overlap_count != 0
            or train.independent_audit_family_overlap_count != 0
            or development.independent_audit_family_overlap_count != 0
            or train.independent_audit_source_overlap_count != 0
            or development.independent_audit_source_overlap_count != 0
            or train.selection_receipt_relative_path != development.selection_receipt_relative_path
            or train.selection_receipt_sha256 != development.selection_receipt_sha256
            or train.parent_selection_receipt_sha256 != development.parent_selection_receipt_sha256
            or not isinstance(selection, RasterTextFreshScopeSelectionReceiptV2)
            or context.selection_relative_path != train.selection_receipt_relative_path
            or context.selection_sha256 != train.selection_receipt_sha256
            or selection.parent_selection_receipt_sha256 != train.parent_selection_receipt_sha256
        ):
            raise ValueError
        if model_bundle is not None:
            manifest = model_bundle.manifest
            if (
                manifest.source_recognizer_release_sha256 != release.release_sha256
                or manifest.source_recognizer_receipt_sha256 != receipt.receipt_sha256
                or manifest.source_recognizer_training_intent_sha256
                != receipt.training_intent_sha256
                or manifest.source_recognizer_parent_release_sha256 != receipt.parent_release_sha256
                or manifest.source_recognizer_parent_receipt_sha256 != receipt.parent_receipt_sha256
                or manifest.source_recognizer_train_manifest_sha256 != receipt.train_manifest_sha256
                or manifest.source_recognizer_development_manifest_sha256
                != receipt.development_manifest_sha256
                or manifest.source_recognizer_train_corpus_manifest != train
                or manifest.source_recognizer_development_corpus_manifest != development
            ):
                raise ValueError
        if profile is not None and (
            type(profile) is not RasterTextOfflineProfileV20
            or profile.schema_version != "20.0"
            or profile.engine_kind != _V20_ENGINE_KIND
            or profile.sidecar_timeout_seconds != 30
            or profile.source_recognizer_release_sha256 != release.release_sha256
            or profile.source_recognizer_receipt_sha256 != receipt.receipt_sha256
            or profile.source_recognizer_training_intent_sha256 != receipt.training_intent_sha256
            or profile.source_recognizer_parent_release_sha256 != receipt.parent_release_sha256
            or profile.source_recognizer_parent_receipt_sha256 != receipt.parent_receipt_sha256
            or profile.source_recognizer_train_manifest_sha256 != receipt.train_manifest_sha256
            or profile.source_recognizer_development_manifest_sha256
            != receipt.development_manifest_sha256
            or profile.source_recognizer_train_corpus_manifest != train
            or profile.source_recognizer_development_corpus_manifest != development
            or profile.source_recognizer_voyage_recipe_set_sha256 != train.voyage_recipe_set_sha256
            or profile.recognition_batch_size != 8
            or profile.recognition_crop_policy != "horizontal-height-0p32-max8-v1"
            or profile.onednn_model_policy != "recognizer-only-v1"
        ):
            raise ValueError
        return release
    except SyntheticRasterTextActivationError:
        raise
    except (VoyageReplayError, ValueError):
        raise SyntheticRasterTextActivationError("voyage_replay_v3_contract_invalid") from None


def write_raster_semantic_text_fresh_calibration_receipt_v4(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_runtime_configuration_relative_path: str,
    activated_profile: RasterTextOfflineProfileV14,
    activated_runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write confidence evidence from the source-bound fresh holdout only."""

    root = _root(document_preview_root)
    source, execution, bundle, cases = verify_source_attested_fresh_raster_text_evidence(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-calibration",
    )
    if (
        not isinstance(execution, RasterTextFreshCandidateExecutionV1)
        or not isinstance(bundle, RasterTextFreshCandidateBundleV1)
        or execution.candidate_count != 1_200
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_invalid")
    calibration_profile, calibration_configuration_sha256 = _resolved_v14_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration_runtime_configuration_relative_path),
    )
    floor = _calibrated_floor(
        cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_calibration_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_calibration_invalid",
    )
    if (
        _v14_ocr_material(calibration_profile) != _v14_ocr_material(activated_profile)
        or calibration_profile.minimum_page_confidence != 0.0
        or activated_profile.minimum_page_confidence != floor
        or execution.raster_text_profile_revision
        != calibration_profile.raster_text_profile_revision
        or execution.runtime_configuration_sha256 != calibration_configuration_sha256
        or execution.extractor_revision != calibration_profile.extractor_revision
        or bundle.raster_text_profile_revision != calibration_profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != calibration_configuration_sha256
        or bundle.extractor_revision != calibration_profile.extractor_revision
        or source.selection_receipt_relative_path != execution.selection_receipt_relative_path
        or source.selection_receipt_relative_path != bundle.selection_receipt_relative_path
        or source.selection_receipt_sha256 != execution.selection_receipt_sha256
        or source.selection_receipt_sha256 != bundle.selection_receipt_sha256
        or execution.source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or execution.source_gold_receipt_sha256 != hashlib.sha256(source_raw).hexdigest()
        or bundle.source_gold_receipt_sha256 != hashlib.sha256(source_raw).hexdigest()
        or activated_profile.extractor_revision != calibration_profile.extractor_revision
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_invalid")
    artifact = RasterTextCalibrationReceiptV4(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=activated_profile.raster_text_profile_revision,
        runtime_configuration_sha256=activated_runtime_configuration_sha256,
        calibration_raster_text_profile_revision=(calibration_profile.raster_text_profile_revision),
        calibration_runtime_configuration_path=(calibration_runtime_configuration_relative_path),
        calibration_runtime_configuration_sha256=calibration_configuration_sha256,
        extractor_revision=activated_profile.extractor_revision,
        selection_receipt_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        minimum_page_confidence=floor,
        latency_p95_seconds=execution.latency_p95_seconds,
    )
    digest = _digest_model(artifact)
    relative = f"synthetic-raster-text-v1/semantic-fresh-calibration-receipts-v4/{digest}.json"
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_calibration_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_write_failed")
    return relative, digest


def _load_bound_fresh_projection(
    *,
    root: Path,
    execution: RasterTextFreshProjectedCandidateExecutionV1,
    code: str,
) -> tuple[RasterTextFreshEvaluationProjectionReceiptV1, bytes]:
    raw = _read(
        root,
        execution.projection_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code=code,
    )
    try:
        projection = RasterTextFreshEvaluationProjectionReceiptV1.model_validate_json(raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError(code) from None
    if (
        raw != _canonical_json(projection.model_dump(mode="json", warnings=False))
        or hashlib.sha256(raw).hexdigest() != execution.projection_receipt_sha256
    ):
        raise SyntheticRasterTextActivationError(code)
    return projection, raw


def write_raster_semantic_text_fresh_calibration_receipt_v5(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_runtime_configuration_relative_path: str,
    activated_profile: RasterTextOfflineProfileV15,
    activated_runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write the fixed 10-family projected calibration evidence for V15."""

    root = _root(document_preview_root)
    source, execution, bundle, cases = verify_source_attested_fresh_raster_text_evidence(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-calibration",
    )
    if (
        not isinstance(activated_profile, RasterTextOfflineProfileV15)
        or not isinstance(execution, RasterTextFreshProjectedCandidateExecutionV1)
        or not isinstance(bundle, RasterTextFreshProjectedCandidateBundleV1)
        or execution.candidate_count != 240
        or len(bundle.candidates) != 240
        or execution.latency_p95_seconds > 10
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_invalid")
    projection, projection_raw = _load_bound_fresh_projection(
        root=root,
        execution=execution,
        code="raster_text_fresh_calibration_invalid",
    )
    calibration_profile, calibration_configuration_sha256 = _resolved_v15_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration_runtime_configuration_relative_path),
    )
    floor = _calibrated_floor(
        cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_calibration_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_calibration_invalid",
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        projection.projection_kind != "calibration-10-family-v1"
        or projection.family_count != 10
        or projection.asset_count != 240
        or projection.full_source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or projection.full_source_gold_receipt_sha256 != source_sha256
        or projection.selection_receipt_relative_path != source.selection_receipt_relative_path
        or projection.selection_receipt_sha256 != source.selection_receipt_sha256
        or execution.projection_receipt_relative_path != bundle.projection_receipt_relative_path
        or execution.projection_receipt_sha256 != bundle.projection_receipt_sha256
        or hashlib.sha256(projection_raw).hexdigest() != execution.projection_receipt_sha256
        or _v15_ocr_material(calibration_profile) != _v15_ocr_material(activated_profile)
        or calibration_profile.minimum_page_confidence != 0.0
        or activated_profile.minimum_page_confidence != floor
        or execution.raster_text_profile_revision
        != calibration_profile.raster_text_profile_revision
        or execution.runtime_configuration_sha256 != calibration_configuration_sha256
        or execution.extractor_revision != calibration_profile.extractor_revision
        or bundle.raster_text_profile_revision != calibration_profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != calibration_configuration_sha256
        or bundle.extractor_revision != calibration_profile.extractor_revision
        or source.selection_receipt_relative_path != execution.selection_receipt_relative_path
        or source.selection_receipt_relative_path != bundle.selection_receipt_relative_path
        or source.selection_receipt_sha256 != execution.selection_receipt_sha256
        or source.selection_receipt_sha256 != bundle.selection_receipt_sha256
        or execution.source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or execution.source_gold_receipt_sha256 != source_sha256
        or bundle.source_gold_receipt_sha256 != source_sha256
        or activated_profile.extractor_revision != calibration_profile.extractor_revision
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_invalid")
    artifact = RasterTextCalibrationReceiptV5(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        evidence_revision="source-attested-fresh-staged-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=activated_profile.raster_text_profile_revision,
        runtime_configuration_sha256=activated_runtime_configuration_sha256,
        calibration_raster_text_profile_revision=(calibration_profile.raster_text_profile_revision),
        calibration_runtime_configuration_path=(calibration_runtime_configuration_relative_path),
        calibration_runtime_configuration_sha256=calibration_configuration_sha256,
        extractor_revision=activated_profile.extractor_revision,
        selection_receipt_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        projection_receipt_path=execution.projection_receipt_relative_path,
        projection_receipt_sha256=execution.projection_receipt_sha256,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_sha256,
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        minimum_page_confidence=floor,
        latency_p95_seconds=execution.latency_p95_seconds,
    )
    digest = _digest_model(artifact)
    # Keep generated evidence usable below the already-deep Windows project
    # root.  The content digest binds the schema/profile, so another nested
    # profile directory would add path length without adding identity.
    relative = f"synthetic-raster-text-v1/calibration-v5/{digest}.json"
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_calibration_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_calibration_write_failed")
    return relative, digest


def write_raster_semantic_text_fresh_calibration_receipt_v6(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_runtime_configuration_relative_path: str,
    activated_profile: RasterTextOfflineProfileV18,
    activated_runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write selection-V2 calibration evidence for exactly V18."""

    root = _root(document_preview_root)
    source, execution, bundle, cases = verify_source_attested_fresh_raster_text_evidence_v6(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-calibration",
    )
    if (
        not isinstance(activated_profile, RasterTextOfflineProfileV18)
        or execution.candidate_count != 240
        or len(bundle.candidates) != 240
        or execution.latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
    ):
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_calibration_invalid")
    projection, projection_raw = _load_bound_fresh_projection(
        root=root,
        execution=execution,
        code="raster_text_v6_fresh_calibration_invalid",
    )
    calibration_profile, calibration_configuration_sha256 = _resolved_v18_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration_runtime_configuration_relative_path),
    )
    floor = _calibrated_floor(
        cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_calibration_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_calibration_invalid",
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        projection.projection_kind != "calibration-10-family-v1"
        or projection.family_count != 10
        or projection.asset_count != 240
        or projection.full_source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or projection.full_source_gold_receipt_sha256 != source_sha256
        or projection.selection_receipt_relative_path != source.selection_receipt_relative_path
        or projection.selection_receipt_sha256 != source.selection_receipt_sha256
        or execution.projection_receipt_relative_path != bundle.projection_receipt_relative_path
        or execution.projection_receipt_sha256 != bundle.projection_receipt_sha256
        or hashlib.sha256(projection_raw).hexdigest() != execution.projection_receipt_sha256
        or _v18_ocr_material(calibration_profile) != _v18_ocr_material(activated_profile)
        or calibration_profile.minimum_page_confidence != 0.0
        or activated_profile.minimum_page_confidence != floor
        or execution.raster_text_profile_revision
        != calibration_profile.raster_text_profile_revision
        or execution.runtime_configuration_sha256 != calibration_configuration_sha256
        or execution.extractor_revision != calibration_profile.extractor_revision
        or bundle.raster_text_profile_revision != calibration_profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != calibration_configuration_sha256
        or bundle.extractor_revision != calibration_profile.extractor_revision
        or source.selection_receipt_relative_path != execution.selection_receipt_relative_path
        or source.selection_receipt_relative_path != bundle.selection_receipt_relative_path
        or source.selection_receipt_sha256 != execution.selection_receipt_sha256
        or source.selection_receipt_sha256 != bundle.selection_receipt_sha256
        or execution.source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or execution.source_gold_receipt_sha256 != source_sha256
        or bundle.source_gold_receipt_sha256 != source_sha256
        or activated_profile.extractor_revision != calibration_profile.extractor_revision
    ):
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_calibration_invalid")
    artifact = RasterTextCalibrationReceiptV6(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        evidence_revision="source-attested-fresh-staged-train-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=activated_profile.raster_text_profile_revision,
        runtime_configuration_sha256=activated_runtime_configuration_sha256,
        calibration_raster_text_profile_revision=(calibration_profile.raster_text_profile_revision),
        calibration_runtime_configuration_path=(calibration_runtime_configuration_relative_path),
        calibration_runtime_configuration_sha256=calibration_configuration_sha256,
        extractor_revision=activated_profile.extractor_revision,
        selection_receipt_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        projection_receipt_path=execution.projection_receipt_relative_path,
        projection_receipt_sha256=execution.projection_receipt_sha256,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_sha256,
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        minimum_page_confidence=floor,
        latency_p95_seconds=execution.latency_p95_seconds,
    )
    digest = _digest_model(artifact)
    relative = f"synthetic-raster-text-v1/calibration-v6/{digest}.json"
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_v6_fresh_calibration_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_calibration_write_failed")
    return relative, digest


def write_raster_semantic_text_fresh_calibration_receipt_v7(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_runtime_configuration_relative_path: str,
    activated_profile: RasterTextOfflineProfileV19,
    activated_runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write V2-holdout calibration bound to the exact V19 fine-tune."""

    root = _root(document_preview_root)
    source, execution, bundle, cases = verify_source_attested_fresh_raster_text_evidence_v6(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-calibration",
    )
    if (
        not isinstance(activated_profile, RasterTextOfflineProfileV19)
        or execution.candidate_count != 240
        or len(bundle.candidates) != 240
        or execution.latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
    ):
        raise SyntheticRasterTextActivationError("raster_text_v7_fresh_calibration_invalid")
    projection, projection_raw = _load_bound_fresh_projection(
        root=root,
        execution=execution,
        code="raster_text_v7_fresh_calibration_invalid",
    )
    calibration_profile, calibration_configuration_sha256 = _resolved_v19_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration_runtime_configuration_relative_path),
    )
    context = _fresh_scope_context(
        root=root,
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope="fresh-calibration",
        selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
        existing_selection_relative_path=source.selection_receipt_relative_path,
    )
    selection = context.selection
    floor = _calibrated_floor(
        cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v7_fresh_calibration_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v7_fresh_calibration_invalid",
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        not isinstance(selection, RasterTextFreshScopeSelectionReceiptV2)
        or projection.projection_kind != "calibration-10-family-v1"
        or projection.family_count != 10
        or projection.asset_count != 240
        or projection.full_source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or projection.full_source_gold_receipt_sha256 != source_sha256
        or projection.selection_receipt_relative_path != source.selection_receipt_relative_path
        or projection.selection_receipt_sha256 != source.selection_receipt_sha256
        or execution.projection_receipt_relative_path != bundle.projection_receipt_relative_path
        or execution.projection_receipt_sha256 != bundle.projection_receipt_sha256
        or hashlib.sha256(projection_raw).hexdigest() != execution.projection_receipt_sha256
        or _v19_ocr_material(calibration_profile) != _v19_ocr_material(activated_profile)
        or calibration_profile.minimum_page_confidence != 0.0
        or activated_profile.minimum_page_confidence != floor
        or execution.raster_text_profile_revision
        != calibration_profile.raster_text_profile_revision
        or execution.runtime_configuration_sha256 != calibration_configuration_sha256
        or execution.extractor_revision != calibration_profile.extractor_revision
        or bundle.raster_text_profile_revision != calibration_profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != calibration_configuration_sha256
        or bundle.extractor_revision != calibration_profile.extractor_revision
        or source.selection_receipt_relative_path != execution.selection_receipt_relative_path
        or source.selection_receipt_relative_path != bundle.selection_receipt_relative_path
        or source.selection_receipt_sha256 != execution.selection_receipt_sha256
        or source.selection_receipt_sha256 != bundle.selection_receipt_sha256
        or execution.source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or execution.source_gold_receipt_sha256 != source_sha256
        or bundle.source_gold_receipt_sha256 != source_sha256
        or activated_profile.extractor_revision != calibration_profile.extractor_revision
        or source.selection_receipt_relative_path
        != activated_profile.source_recognizer_selection_receipt_relative_path
        or source.selection_receipt_sha256
        != activated_profile.source_recognizer_selection_receipt_sha256
        or selection.parent_selection_receipt_sha256
        != activated_profile.source_recognizer_parent_selection_receipt_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_v7_fresh_calibration_invalid")
    artifact = RasterTextCalibrationReceiptV7(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        evidence_revision=("source-attested-public-finetuned-staged-train-holdout-v2"),
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=(activated_profile.raster_text_profile_revision),
        runtime_configuration_sha256=(activated_runtime_configuration_sha256),
        calibration_raster_text_profile_revision=(calibration_profile.raster_text_profile_revision),
        calibration_runtime_configuration_path=(calibration_runtime_configuration_relative_path),
        calibration_runtime_configuration_sha256=(calibration_configuration_sha256),
        extractor_revision=activated_profile.extractor_revision,
        selection_receipt_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        projection_receipt_path=execution.projection_receipt_relative_path,
        projection_receipt_sha256=execution.projection_receipt_sha256,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_sha256,
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        minimum_page_confidence=floor,
        latency_p95_seconds=execution.latency_p95_seconds,
        source_recognizer_release_sha256=(activated_profile.source_recognizer_release_sha256),
        source_recognizer_receipt_sha256=(activated_profile.source_recognizer_receipt_sha256),
        source_recognizer_training_intent_sha256=(
            activated_profile.source_recognizer_training_intent_sha256
        ),
        source_recognizer_parent_selection_receipt_sha256=(
            activated_profile.source_recognizer_parent_selection_receipt_sha256
        ),
    )
    digest = _digest_model(artifact)
    relative = f"synthetic-raster-text-v1/calibration-v7/{digest}.json"
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_v7_fresh_calibration_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_v7_fresh_calibration_write_failed")
    return relative, digest


def _load_verified_final_warm_smoke(
    *,
    root: Path,
    relative_path: str,
    profile: (
        RasterTextOfflineProfileV14
        | RasterTextOfflineProfileV15
        | RasterTextOfflineProfileV18
        | RasterTextOfflineProfileV19
    ),
    runtime_configuration_sha256: str,
    expected_runner_implementation_sha256: str | None = None,
) -> tuple[RasterTextWarmServiceSmokeReceiptV1, bytes]:
    raw = _read(
        root,
        relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    try:
        warm = RasterTextWarmServiceSmokeReceiptV1.model_validate_json(raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_sealed_evaluation_invalid"
        ) from None
    source_raw = _read(
        root,
        warm.source_gold_receipt_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    execution_raw = _read(
        root,
        warm.candidate_execution_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    try:
        source, execution, bundle, _cases = verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=warm.source_gold_receipt_path,
            candidate_execution_relative_path=warm.candidate_execution_path,
            scope="smoke",
            expected_raster_text_profile_revision=profile.raster_text_profile_revision,
            expected_runtime_configuration_sha256=runtime_configuration_sha256,
            expected_extractor_revision=profile.extractor_revision,
            expected_runner_implementation_sha256=(expected_runner_implementation_sha256),
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_sealed_evaluation_invalid"
        ) from None
    if (
        raw != _canonical_json(warm.model_dump(mode="json", warnings=False))
        or hashlib.sha256(source_raw).hexdigest() != warm.source_gold_receipt_sha256
        or hashlib.sha256(execution_raw).hexdigest() != warm.candidate_execution_sha256
        or source.scope != "smoke"
        or execution.scope != "smoke"
        or bundle.scope != "smoke"
        or execution.candidate_bundle_sha256 != warm.candidate_bundle_sha256
        or warm.raster_text_profile_revision != profile.raster_text_profile_revision
        or warm.runtime_configuration_sha256 != runtime_configuration_sha256
        or warm.extractor_revision != profile.extractor_revision
        or warm.startup_latency_seconds > raster_text_service_startup_timeout_seconds(profile)
        or warm.startup_timeout_seconds != raster_text_service_startup_timeout_seconds(profile)
        or warm.request_timeout_seconds != min(profile.sidecar_timeout_seconds, 30)
        or raster_text_exact_result_set_sha256(
            [(item.asset_id, item.profile_id, item.candidate) for item in bundle.candidates]
        )
        != warm.equality_set_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    return warm, raw


def write_raster_semantic_text_fresh_sealed_evaluation_v4(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfileV14,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Gate and write only the independently selected fresh sealed split."""

    root = _root(document_preview_root)
    source, execution, bundle, sealed_cases = verify_source_attested_fresh_raster_text_evidence(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-sealed",
    )
    if (
        not isinstance(execution, RasterTextFreshCandidateExecutionV1)
        or not isinstance(bundle, RasterTextFreshCandidateBundleV1)
        or execution.candidate_count != 2_400
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    calibration_raw = _read(
        root,
        calibration_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    try:
        calibration = RasterTextCalibrationReceiptV4.model_validate_json(calibration_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_sealed_evaluation_invalid"
        ) from None
    if calibration_raw != _canonical_json(calibration.model_dump(mode="json", warnings=False)):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    calibration_source, calibration_execution, calibration_bundle, calibration_cases = (
        verify_source_attested_fresh_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
            candidate_execution_relative_path=calibration.candidate_execution_path,
            scope="fresh-calibration",
        )
    )
    calibration_profile, calibration_configuration_sha256 = _resolved_v14_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration.calibration_runtime_configuration_path),
    )
    warm_smoke, warm_smoke_raw = _load_verified_final_warm_smoke(
        root=root,
        relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
    )
    floor = _calibrated_floor(
        calibration_cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    scores = evaluate_synthetic_raster_semantic_text_gates_v3(sealed_cases)
    if not synthetic_raster_semantic_text_gate_passes_v3(scores):
        raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    calibration_source_raw = _read(
        root,
        calibration.source_gold_receipt_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    calibration_execution_raw = _read(
        root,
        calibration.candidate_execution_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    if (
        _v14_ocr_material(calibration_profile) != _v14_ocr_material(profile)
        or calibration_configuration_sha256 != calibration.calibration_runtime_configuration_sha256
        or calibration_profile.raster_text_profile_revision
        != calibration.calibration_raster_text_profile_revision
        or profile.raster_text_profile_revision != execution.raster_text_profile_revision
        or runtime_configuration_sha256 != execution.runtime_configuration_sha256
        or profile.extractor_revision != execution.extractor_revision
        or bundle.raster_text_profile_revision != profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != runtime_configuration_sha256
        or bundle.extractor_revision != profile.extractor_revision
        or calibration.raster_text_profile_revision != profile.raster_text_profile_revision
        or calibration.runtime_configuration_sha256 != runtime_configuration_sha256
        or calibration.extractor_revision != profile.extractor_revision
        or calibration.minimum_page_confidence != profile.minimum_page_confidence
        or calibration.minimum_page_confidence != floor
        or calibration_source.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_source.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration.selection_receipt_path != source.selection_receipt_relative_path
        or calibration.selection_receipt_sha256 != source.selection_receipt_sha256
        or execution.selection_receipt_relative_path != source.selection_receipt_relative_path
        or execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or bundle.selection_receipt_relative_path != source.selection_receipt_relative_path
        or bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_execution.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_bundle.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or hashlib.sha256(calibration_source_raw).hexdigest()
        != calibration.source_gold_receipt_sha256
        or hashlib.sha256(calibration_execution_raw).hexdigest()
        != calibration.candidate_execution_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    artifact = RasterTextSealedEvaluationV4(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_sha256=runtime_configuration_sha256,
        extractor_revision=profile.extractor_revision,
        selection_receipt_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
        calibration_receipt_path=calibration_receipt_relative_path,
        calibration_receipt_sha256=hashlib.sha256(calibration_raw).hexdigest(),
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        warm_service_smoke_receipt_path=warm_service_smoke_receipt_relative_path,
        warm_service_smoke_receipt_sha256=hashlib.sha256(warm_smoke_raw).hexdigest(),
        latency_p95_seconds=execution.latency_p95_seconds,
        service_warm_latency_p95_seconds=warm_smoke.warm_latency_p95_seconds,
        minimum_page_confidence=profile.minimum_page_confidence,
        scores=[RasterTextSealedProfileScoreV4(**item.model_dump()) for item in scores],
    )
    digest = _digest_model(artifact)
    relative = (
        "synthetic-raster-text-v1/semantic-fresh-sealed-evaluations-v4/"
        f"{profile.raster_text_profile_revision.rsplit(':', 1)[1]}/{digest}.json"
    )
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_sealed_evaluation_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_write_failed")
    return relative, digest


def write_raster_semantic_text_fresh_sealed_evaluation_v5(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    stage1_decision_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfileV15,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write a V15 stage-1 early pass or bound stage-2 sealed result."""

    root = _root(document_preview_root)
    source, execution, bundle, sealed_cases = verify_source_attested_fresh_raster_text_evidence(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-sealed",
    )
    if (
        not isinstance(profile, RasterTextOfflineProfileV15)
        or not isinstance(execution, RasterTextFreshProjectedCandidateExecutionV1)
        or not isinstance(bundle, RasterTextFreshProjectedCandidateBundleV1)
        or execution.candidate_count not in (1_200, 2_400)
        or len(bundle.candidates) != execution.candidate_count
        or execution.latency_p95_seconds > 10
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    projection, projection_raw = _load_bound_fresh_projection(
        root=root,
        execution=execution,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    decision_raw = _read(
        root,
        stage1_decision_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
    try:
        stage1_decision = verify_fresh_raster_text_sealed_stage1_decision(
            document_preview_root=root,
            decision_relative_path=stage1_decision_receipt_relative_path,
            expected_sha256=decision_sha256,
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_sealed_evaluation_invalid"
        ) from None
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    execution_sha256 = hashlib.sha256(execution_raw).hexdigest()
    sealed_projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"]
    sealed_stage1_decision: Literal["early_pass", "continue"]
    sealed_family_count: Literal[50, 100]
    sealed_asset_count: Literal[1_200, 2_400]
    if projection.projection_kind == "sealed-stage1-50-family-v1":
        sealed_projection_kind = "sealed-stage1-50-family-v1"
        sealed_stage1_decision = "early_pass"
        sealed_family_count = 50
        sealed_asset_count = 1_200
        stage_is_valid = (
            execution.candidate_count == 1_200
            and projection.family_count == 50
            and projection.asset_count == 1_200
            and stage1_decision.decision == "early_pass"
            and stage1_decision.source_gold_receipt_relative_path
            == source_gold_receipt_relative_path
            and stage1_decision.source_gold_receipt_sha256 == source_sha256
            and stage1_decision.candidate_execution_relative_path
            == candidate_execution_relative_path
            and stage1_decision.candidate_execution_sha256 == execution_sha256
            and stage1_decision.projection_receipt_relative_path
            == execution.projection_receipt_relative_path
            and stage1_decision.projection_receipt_sha256 == execution.projection_receipt_sha256
        )
    elif projection.projection_kind == "sealed-stage2-100-family-v1":
        sealed_projection_kind = "sealed-stage2-100-family-v1"
        sealed_stage1_decision = "continue"
        sealed_family_count = 100
        sealed_asset_count = 2_400
        stage_is_valid = (
            execution.candidate_count == 2_400
            and projection.family_count == 100
            and projection.asset_count == 2_400
            and stage1_decision.decision == "continue"
            and stage1_decision.source_gold_receipt_relative_path
            == source_gold_receipt_relative_path
            and stage1_decision.source_gold_receipt_sha256 == source_sha256
            and projection.parent_stage1_decision_relative_path
            == stage1_decision_receipt_relative_path
            and projection.parent_stage1_decision_sha256 == decision_sha256
        )
    else:
        stage_is_valid = False
    if not stage_is_valid:
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")

    calibration_raw = _read(
        root,
        calibration_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    try:
        calibration = RasterTextCalibrationReceiptV5.model_validate_json(calibration_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_sealed_evaluation_invalid"
        ) from None
    if calibration_raw != _canonical_json(calibration.model_dump(mode="json", warnings=False)):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    calibration_source, calibration_execution, calibration_bundle, calibration_cases = (
        verify_source_attested_fresh_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
            candidate_execution_relative_path=calibration.candidate_execution_path,
            scope="fresh-calibration",
        )
    )
    if (
        not isinstance(calibration_execution, RasterTextFreshProjectedCandidateExecutionV1)
        or not isinstance(calibration_bundle, RasterTextFreshProjectedCandidateBundleV1)
        or calibration_execution.candidate_count != 240
        or len(calibration_bundle.candidates) != 240
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    calibration_profile, calibration_configuration_sha256 = _resolved_v15_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration.calibration_runtime_configuration_path),
    )
    warm_smoke, warm_smoke_raw = _load_verified_final_warm_smoke(
        root=root,
        relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
    )
    floor = _calibrated_floor(
        calibration_cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    scores = evaluate_synthetic_raster_semantic_text_gates_v3(sealed_cases)
    if not synthetic_raster_semantic_text_gate_passes_v3(scores):
        raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
    calibration_source_raw = _read(
        root,
        calibration.source_gold_receipt_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    calibration_execution_raw = _read(
        root,
        calibration.candidate_execution_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_fresh_sealed_evaluation_invalid",
    )
    if (
        warm_smoke.warm_latency_p95_seconds > 10
        or calibration.latency_p95_seconds > 10
        or projection.full_source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or projection.full_source_gold_receipt_sha256 != source_sha256
        or projection.selection_receipt_relative_path != source.selection_receipt_relative_path
        or projection.selection_receipt_sha256 != source.selection_receipt_sha256
        or hashlib.sha256(projection_raw).hexdigest() != execution.projection_receipt_sha256
        or execution.projection_receipt_relative_path != bundle.projection_receipt_relative_path
        or execution.projection_receipt_sha256 != bundle.projection_receipt_sha256
        or _v15_ocr_material(calibration_profile) != _v15_ocr_material(profile)
        or calibration_configuration_sha256 != calibration.calibration_runtime_configuration_sha256
        or calibration_profile.raster_text_profile_revision
        != calibration.calibration_raster_text_profile_revision
        or profile.raster_text_profile_revision != execution.raster_text_profile_revision
        or runtime_configuration_sha256 != execution.runtime_configuration_sha256
        or profile.extractor_revision != execution.extractor_revision
        or bundle.raster_text_profile_revision != profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != runtime_configuration_sha256
        or bundle.extractor_revision != profile.extractor_revision
        or calibration.raster_text_profile_revision != profile.raster_text_profile_revision
        or calibration.runtime_configuration_sha256 != runtime_configuration_sha256
        or calibration.extractor_revision != profile.extractor_revision
        or calibration.minimum_page_confidence != profile.minimum_page_confidence
        or calibration.minimum_page_confidence != floor
        or calibration.projection_kind != "calibration-10-family-v1"
        or calibration.family_count != 10
        or calibration.asset_count != 240
        or calibration.projection_receipt_path
        != calibration_execution.projection_receipt_relative_path
        or calibration.projection_receipt_sha256 != calibration_execution.projection_receipt_sha256
        or calibration.projection_receipt_path
        != calibration_bundle.projection_receipt_relative_path
        or calibration.projection_receipt_sha256 != calibration_bundle.projection_receipt_sha256
        or calibration_source.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_source.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration.selection_receipt_path != source.selection_receipt_relative_path
        or calibration.selection_receipt_sha256 != source.selection_receipt_sha256
        or execution.selection_receipt_relative_path != source.selection_receipt_relative_path
        or execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or bundle.selection_receipt_relative_path != source.selection_receipt_relative_path
        or bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_execution.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_bundle.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or hashlib.sha256(calibration_source_raw).hexdigest()
        != calibration.source_gold_receipt_sha256
        or hashlib.sha256(calibration_execution_raw).hexdigest()
        != calibration.candidate_execution_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_invalid")
    artifact = RasterTextSealedEvaluationV5(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        evidence_revision="source-attested-fresh-staged-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind=sealed_projection_kind,
        stage1_decision=sealed_stage1_decision,
        family_count=sealed_family_count,
        asset_count=sealed_asset_count,
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_sha256=runtime_configuration_sha256,
        extractor_revision=profile.extractor_revision,
        selection_receipt_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        projection_receipt_path=execution.projection_receipt_relative_path,
        projection_receipt_sha256=execution.projection_receipt_sha256,
        stage1_decision_receipt_path=stage1_decision_receipt_relative_path,
        stage1_decision_receipt_sha256=decision_sha256,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_sha256,
        calibration_receipt_path=calibration_receipt_relative_path,
        calibration_receipt_sha256=hashlib.sha256(calibration_raw).hexdigest(),
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=execution_sha256,
        warm_service_smoke_receipt_path=warm_service_smoke_receipt_relative_path,
        warm_service_smoke_receipt_sha256=hashlib.sha256(warm_smoke_raw).hexdigest(),
        latency_p95_seconds=execution.latency_p95_seconds,
        service_warm_latency_p95_seconds=warm_smoke.warm_latency_p95_seconds,
        minimum_page_confidence=profile.minimum_page_confidence,
        scores=[RasterTextSealedProfileScoreV5(**item.model_dump()) for item in scores],
    )
    digest = _digest_model(artifact)
    relative = f"synthetic-raster-text-v1/sealed-v5/{digest}.json"
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_fresh_sealed_evaluation_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError("raster_text_fresh_sealed_evaluation_write_failed")
    return relative, digest


def write_raster_semantic_text_fresh_sealed_evaluation_v6(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    stage1_decision_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfileV18,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write a passing staged selection-V2 evaluation for exactly V18."""

    root = _root(document_preview_root)
    source, execution, bundle, sealed_cases = verify_source_attested_fresh_raster_text_evidence_v6(
        document_preview_root=root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        scope="fresh-sealed",
    )
    if (
        not isinstance(profile, RasterTextOfflineProfileV18)
        or execution.candidate_count not in (1_200, 2_400)
        or len(bundle.candidates) != execution.candidate_count
        or execution.latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
    ):
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_sealed_evaluation_invalid")
    projection, projection_raw = _load_bound_fresh_projection(
        root=root,
        execution=execution,
        code="raster_text_v6_fresh_sealed_evaluation_invalid",
    )
    decision_raw = _read(
        root,
        stage1_decision_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_sealed_evaluation_invalid",
    )
    decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
    try:
        stage1_decision = verify_fresh_raster_text_sealed_stage1_decision_v6(
            document_preview_root=root,
            decision_relative_path=stage1_decision_receipt_relative_path,
            expected_sha256=decision_sha256,
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_v6_fresh_sealed_evaluation_invalid"
        ) from None
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_sealed_evaluation_invalid",
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_sealed_evaluation_invalid",
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    execution_sha256 = hashlib.sha256(execution_raw).hexdigest()
    sealed_projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"]
    sealed_stage1_decision: Literal["early_pass", "continue"]
    sealed_family_count: Literal[50, 100]
    sealed_asset_count: Literal[1_200, 2_400]
    if projection.projection_kind == "sealed-stage1-50-family-v1":
        sealed_projection_kind = "sealed-stage1-50-family-v1"
        sealed_stage1_decision = "early_pass"
        sealed_family_count = 50
        sealed_asset_count = 1_200
        stage_is_valid = (
            execution.candidate_count == 1_200
            and projection.family_count == 50
            and projection.asset_count == 1_200
            and stage1_decision.decision == "early_pass"
            and stage1_decision.source_gold_receipt_relative_path
            == source_gold_receipt_relative_path
            and stage1_decision.source_gold_receipt_sha256 == source_sha256
            and stage1_decision.candidate_execution_relative_path
            == candidate_execution_relative_path
            and stage1_decision.candidate_execution_sha256 == execution_sha256
            and stage1_decision.projection_receipt_relative_path
            == execution.projection_receipt_relative_path
            and stage1_decision.projection_receipt_sha256 == execution.projection_receipt_sha256
        )
    elif projection.projection_kind == "sealed-stage2-100-family-v1":
        sealed_projection_kind = "sealed-stage2-100-family-v1"
        sealed_stage1_decision = "continue"
        sealed_family_count = 100
        sealed_asset_count = 2_400
        stage_is_valid = (
            execution.candidate_count == 2_400
            and projection.family_count == 100
            and projection.asset_count == 2_400
            and stage1_decision.decision == "continue"
            and stage1_decision.source_gold_receipt_relative_path
            == source_gold_receipt_relative_path
            and stage1_decision.source_gold_receipt_sha256 == source_sha256
            and projection.parent_stage1_decision_relative_path
            == stage1_decision_receipt_relative_path
            and projection.parent_stage1_decision_sha256 == decision_sha256
        )
    else:
        stage_is_valid = False
    if not stage_is_valid:
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_sealed_evaluation_invalid")

    calibration_raw = _read(
        root,
        calibration_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_sealed_evaluation_invalid",
    )
    try:
        calibration = RasterTextCalibrationReceiptV6.model_validate_json(calibration_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError(
            "raster_text_v6_fresh_sealed_evaluation_invalid"
        ) from None
    if calibration_raw != _canonical_json(calibration.model_dump(mode="json", warnings=False)):
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_sealed_evaluation_invalid")
    (
        calibration_source,
        calibration_execution,
        calibration_bundle,
        calibration_cases,
    ) = verify_source_attested_fresh_raster_text_evidence_v6(
        document_preview_root=root,
        source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
        candidate_execution_relative_path=calibration.candidate_execution_path,
        scope="fresh-calibration",
    )
    if calibration_execution.candidate_count != 240 or len(calibration_bundle.candidates) != 240:
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_sealed_evaluation_invalid")
    calibration_profile, calibration_configuration_sha256 = _resolved_v18_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration.calibration_runtime_configuration_path),
    )
    warm_smoke, warm_smoke_raw = _load_verified_final_warm_smoke(
        root=root,
        relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
    )
    floor = _calibrated_floor(
        calibration_cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    scores = evaluate_synthetic_raster_semantic_text_gates_v3(sealed_cases)
    if not synthetic_raster_semantic_text_gate_passes_v3(scores):
        raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
    calibration_source_raw = _read(
        root,
        calibration.source_gold_receipt_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_sealed_evaluation_invalid",
    )
    calibration_execution_raw = _read(
        root,
        calibration.candidate_execution_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_v6_fresh_sealed_evaluation_invalid",
    )
    if (
        warm_smoke.warm_latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
        or calibration.latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
        or projection.full_source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or projection.full_source_gold_receipt_sha256 != source_sha256
        or projection.selection_receipt_relative_path != source.selection_receipt_relative_path
        or projection.selection_receipt_sha256 != source.selection_receipt_sha256
        or hashlib.sha256(projection_raw).hexdigest() != execution.projection_receipt_sha256
        or execution.projection_receipt_relative_path != bundle.projection_receipt_relative_path
        or execution.projection_receipt_sha256 != bundle.projection_receipt_sha256
        or _v18_ocr_material(calibration_profile) != _v18_ocr_material(profile)
        or calibration_configuration_sha256 != calibration.calibration_runtime_configuration_sha256
        or calibration_profile.raster_text_profile_revision
        != calibration.calibration_raster_text_profile_revision
        or profile.raster_text_profile_revision != execution.raster_text_profile_revision
        or runtime_configuration_sha256 != execution.runtime_configuration_sha256
        or profile.extractor_revision != execution.extractor_revision
        or bundle.raster_text_profile_revision != profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != runtime_configuration_sha256
        or bundle.extractor_revision != profile.extractor_revision
        or calibration.raster_text_profile_revision != profile.raster_text_profile_revision
        or calibration.runtime_configuration_sha256 != runtime_configuration_sha256
        or calibration.extractor_revision != profile.extractor_revision
        or calibration.minimum_page_confidence != profile.minimum_page_confidence
        or calibration.minimum_page_confidence != floor
        or calibration.projection_kind != "calibration-10-family-v1"
        or calibration.family_count != 10
        or calibration.asset_count != 240
        or calibration.projection_receipt_path
        != calibration_execution.projection_receipt_relative_path
        or calibration.projection_receipt_sha256 != calibration_execution.projection_receipt_sha256
        or calibration.projection_receipt_path
        != calibration_bundle.projection_receipt_relative_path
        or calibration.projection_receipt_sha256 != calibration_bundle.projection_receipt_sha256
        or calibration_source.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_source.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration.selection_receipt_path != source.selection_receipt_relative_path
        or calibration.selection_receipt_sha256 != source.selection_receipt_sha256
        or execution.selection_receipt_relative_path != source.selection_receipt_relative_path
        or execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or bundle.selection_receipt_relative_path != source.selection_receipt_relative_path
        or bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_execution.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_bundle.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or hashlib.sha256(calibration_source_raw).hexdigest()
        != calibration.source_gold_receipt_sha256
        or hashlib.sha256(calibration_execution_raw).hexdigest()
        != calibration.candidate_execution_sha256
    ):
        raise SyntheticRasterTextActivationError("raster_text_v6_fresh_sealed_evaluation_invalid")
    artifact = RasterTextSealedEvaluationV6(
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        evidence_revision="source-attested-fresh-staged-train-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind=sealed_projection_kind,
        stage1_decision=sealed_stage1_decision,
        family_count=sealed_family_count,
        asset_count=sealed_asset_count,
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_sha256=runtime_configuration_sha256,
        extractor_revision=profile.extractor_revision,
        selection_receipt_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        projection_receipt_path=execution.projection_receipt_relative_path,
        projection_receipt_sha256=execution.projection_receipt_sha256,
        stage1_decision_receipt_path=stage1_decision_receipt_relative_path,
        stage1_decision_receipt_sha256=decision_sha256,
        source_gold_receipt_path=source_gold_receipt_relative_path,
        source_gold_receipt_sha256=source_sha256,
        calibration_receipt_path=calibration_receipt_relative_path,
        calibration_receipt_sha256=hashlib.sha256(calibration_raw).hexdigest(),
        candidate_execution_path=candidate_execution_relative_path,
        candidate_execution_sha256=execution_sha256,
        warm_service_smoke_receipt_path=warm_service_smoke_receipt_relative_path,
        warm_service_smoke_receipt_sha256=hashlib.sha256(warm_smoke_raw).hexdigest(),
        latency_p95_seconds=execution.latency_p95_seconds,
        service_warm_latency_p95_seconds=warm_smoke.warm_latency_p95_seconds,
        minimum_page_confidence=profile.minimum_page_confidence,
        scores=[RasterTextSealedProfileScoreV6(**item.model_dump()) for item in scores],
    )
    digest = _digest_model(artifact)
    relative = f"synthetic-raster-text-v1/sealed-v6/{digest}.json"
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        raise SyntheticRasterTextActivationError(
            "raster_text_v6_fresh_sealed_evaluation_write_failed"
        ) from None
    if written != relative or result_digest != digest:
        raise SyntheticRasterTextActivationError(
            "raster_text_v6_fresh_sealed_evaluation_write_failed"
        )
    return relative, digest


@dataclass(frozen=True, slots=True)
class _Preview97RevalidationV1:
    revalidation_smoke_receipt_relative_path: str
    revalidation_candidate_execution_relative_path: str


def _write_raster_semantic_text_fresh_sealed_evaluation_v7_or_v8(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    stage1_decision_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfileV19,
    runtime_configuration_sha256: str,
    preview97_revalidation: _Preview97RevalidationV1 | None = None,
    current_native_preview97: bool = False,
) -> tuple[str, str]:
    """Validate one V19 chain under its explicit immutable Gate policy."""

    historical_preview97 = preview97_revalidation is not None
    if historical_preview97 and current_native_preview97:
        raise SyntheticRasterTextActivationError("raster_text_v9_fresh_sealed_evaluation_invalid")
    preview97 = historical_preview97 or current_native_preview97
    invalid = {
        (False, False): "raster_text_v7_fresh_sealed_evaluation_invalid",
        (True, False): "raster_text_v8_fresh_sealed_evaluation_invalid",
        (False, True): "raster_text_v9_fresh_sealed_evaluation_invalid",
    }[(historical_preview97, current_native_preview97)]
    root = _root(document_preview_root)
    historical_runner_sha256 = (
        _V19_PREVIEW97_HISTORICAL_RUNNER_IMPLEMENTATION_SHA256 if historical_preview97 else None
    )
    current_runner_sha256 = _runner_implementation_sha256()
    if historical_preview97:
        source, execution, bundle, sealed_cases = (
            verify_source_attested_fresh_raster_text_evidence_v8(
                document_preview_root=root,
                source_gold_receipt_relative_path=source_gold_receipt_relative_path,
                candidate_execution_relative_path=candidate_execution_relative_path,
                scope="fresh-sealed",
                historical_runner_implementation_sha256=cast(str, historical_runner_sha256),
                current_runner_implementation_sha256=current_runner_sha256,
            )
        )
    else:
        source, execution, bundle, sealed_cases = (
            verify_source_attested_fresh_raster_text_evidence_v6(
                document_preview_root=root,
                source_gold_receipt_relative_path=source_gold_receipt_relative_path,
                candidate_execution_relative_path=candidate_execution_relative_path,
                scope="fresh-sealed",
                expected_runner_implementation_sha256=(
                    current_runner_sha256 if current_native_preview97 else None
                ),
                quality_gate_revision=(
                    _V19_PREVIEW97_QUALITY_GATE_REVISION if current_native_preview97 else None
                ),
            )
        )
    if (
        not isinstance(profile, RasterTextOfflineProfileV19)
        or execution.candidate_count not in (1_200, 2_400)
        or len(bundle.candidates) != execution.candidate_count
        or execution.latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
    ):
        raise SyntheticRasterTextActivationError(invalid)
    projection, projection_raw = _load_bound_fresh_projection(
        root=root,
        execution=execution,
        code=invalid,
    )
    decision_raw = _read(
        root,
        stage1_decision_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code=invalid,
    )
    decision_sha256 = hashlib.sha256(decision_raw).hexdigest()
    try:
        stage1_decision = verify_fresh_raster_text_sealed_stage1_decision_v6(
            document_preview_root=root,
            decision_relative_path=stage1_decision_receipt_relative_path,
            expected_sha256=decision_sha256,
            expected_runner_implementation_sha256=(
                historical_runner_sha256
                if historical_preview97
                else current_runner_sha256
                if current_native_preview97
                else None
            ),
            quality_gate_revision=(
                _V19_PREVIEW97_QUALITY_GATE_REVISION if current_native_preview97 else None
            ),
        )
    except SyntheticRasterTextActivationError:
        raise SyntheticRasterTextActivationError(invalid) from None
    source_raw = _read(
        root,
        source_gold_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code=invalid,
    )
    execution_raw = _read(
        root,
        candidate_execution_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code=invalid,
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    execution_sha256 = hashlib.sha256(execution_raw).hexdigest()
    sealed_projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"]
    sealed_stage1_decision: Literal["early_pass", "continue"]
    sealed_family_count: Literal[50, 100]
    sealed_asset_count: Literal[1_200, 2_400]
    if projection.projection_kind == "sealed-stage1-50-family-v1":
        sealed_projection_kind = "sealed-stage1-50-family-v1"
        sealed_stage1_decision = "early_pass"
        sealed_family_count = 50
        sealed_asset_count = 1_200
        stage_is_valid = (
            execution.candidate_count == 1_200
            and projection.family_count == 50
            and projection.asset_count == 1_200
            and stage1_decision.decision == "early_pass"
            and stage1_decision.source_gold_receipt_relative_path
            == source_gold_receipt_relative_path
            and stage1_decision.source_gold_receipt_sha256 == source_sha256
            and stage1_decision.candidate_execution_relative_path
            == candidate_execution_relative_path
            and stage1_decision.candidate_execution_sha256 == execution_sha256
            and stage1_decision.projection_receipt_relative_path
            == execution.projection_receipt_relative_path
            and stage1_decision.projection_receipt_sha256 == execution.projection_receipt_sha256
        )
    elif projection.projection_kind == "sealed-stage2-100-family-v1":
        sealed_projection_kind = "sealed-stage2-100-family-v1"
        sealed_stage1_decision = "continue"
        sealed_family_count = 100
        sealed_asset_count = 2_400
        stage_is_valid = (
            execution.candidate_count == 2_400
            and projection.family_count == 100
            and projection.asset_count == 2_400
            and stage1_decision.decision == "continue"
            and stage1_decision.source_gold_receipt_relative_path
            == source_gold_receipt_relative_path
            and stage1_decision.source_gold_receipt_sha256 == source_sha256
            and projection.parent_stage1_decision_relative_path
            == stage1_decision_receipt_relative_path
            and projection.parent_stage1_decision_sha256 == decision_sha256
        )
    else:
        stage_is_valid = False
    if not stage_is_valid:
        raise SyntheticRasterTextActivationError(invalid)

    calibration_raw = _read(
        root,
        calibration_receipt_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code=invalid,
    )
    try:
        calibration = RasterTextCalibrationReceiptV7.model_validate_json(calibration_raw)
    except ValidationError:
        raise SyntheticRasterTextActivationError(invalid) from None
    if calibration_raw != _canonical_json(calibration.model_dump(mode="json", warnings=False)):
        raise SyntheticRasterTextActivationError(invalid)
    if historical_preview97:
        (
            calibration_source,
            calibration_execution,
            calibration_bundle,
            calibration_cases,
        ) = verify_source_attested_fresh_raster_text_evidence_v8(
            document_preview_root=root,
            source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
            candidate_execution_relative_path=calibration.candidate_execution_path,
            scope="fresh-calibration",
            historical_runner_implementation_sha256=cast(str, historical_runner_sha256),
            current_runner_implementation_sha256=current_runner_sha256,
        )
    else:
        (
            calibration_source,
            calibration_execution,
            calibration_bundle,
            calibration_cases,
        ) = verify_source_attested_fresh_raster_text_evidence_v6(
            document_preview_root=root,
            source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
            candidate_execution_relative_path=calibration.candidate_execution_path,
            scope="fresh-calibration",
            expected_runner_implementation_sha256=(
                current_runner_sha256 if current_native_preview97 else None
            ),
        )
    if calibration_execution.candidate_count != 240 or len(calibration_bundle.candidates) != 240:
        raise SyntheticRasterTextActivationError(invalid)
    calibration_profile, calibration_configuration_sha256 = _resolved_v19_calibration_profile(
        root=root,
        runtime_configuration_relative_path=(calibration.calibration_runtime_configuration_path),
    )
    warm_smoke, warm_smoke_raw = _load_verified_final_warm_smoke(
        root=root,
        relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
        expected_runner_implementation_sha256=(
            historical_runner_sha256
            if historical_preview97
            else current_runner_sha256
            if current_native_preview97
            else None
        ),
    )
    selection_context = _fresh_scope_context(
        root=root,
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope="fresh-sealed",
        selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
        existing_selection_relative_path=source.selection_receipt_relative_path,
    )
    selection = selection_context.selection
    floor = _calibrated_floor(
        calibration_cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    scores = evaluate_synthetic_raster_semantic_text_gates_v3(sealed_cases)
    gate_passed = (
        synthetic_raster_semantic_text_preview_draft_gate_passes_v1(scores)
        if preview97
        else synthetic_raster_semantic_text_gate_passes_v3(scores)
    )
    if not gate_passed:
        raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
    revalidation_smoke_raw: bytes | None = None
    revalidation_execution_raw: bytes | None = None
    if historical_preview97:
        assert preview97_revalidation is not None
        revalidation_smoke, revalidation_smoke_raw = _load_verified_final_warm_smoke(
            root=root,
            relative_path=(preview97_revalidation.revalidation_smoke_receipt_relative_path),
            profile=profile,
            runtime_configuration_sha256=runtime_configuration_sha256,
            expected_runner_implementation_sha256=current_runner_sha256,
        )
        revalidation_execution_raw = _read(
            root,
            preview97_revalidation.revalidation_candidate_execution_relative_path,
            limit=_MAX_RECEIPT_BYTES,
            code=invalid,
        )
        if (
            revalidation_smoke.candidate_execution_path
            != preview97_revalidation.revalidation_candidate_execution_relative_path
            or revalidation_smoke.candidate_execution_sha256
            != hashlib.sha256(revalidation_execution_raw).hexdigest()
            or revalidation_smoke.equality_set_sha256 != warm_smoke.equality_set_sha256
            or revalidation_smoke.warm_latency_p95_seconds
            > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
        ):
            raise SyntheticRasterTextActivationError(invalid)
    calibration_source_raw = _read(
        root,
        calibration.source_gold_receipt_path,
        limit=_MAX_RECEIPT_BYTES,
        code=invalid,
    )
    calibration_execution_raw = _read(
        root,
        calibration.candidate_execution_path,
        limit=_MAX_RECEIPT_BYTES,
        code=invalid,
    )
    if (
        not isinstance(selection, RasterTextFreshScopeSelectionReceiptV2)
        or warm_smoke.warm_latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
        or calibration.latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
        or projection.full_source_gold_receipt_relative_path != source_gold_receipt_relative_path
        or projection.full_source_gold_receipt_sha256 != source_sha256
        or projection.selection_receipt_relative_path != source.selection_receipt_relative_path
        or projection.selection_receipt_sha256 != source.selection_receipt_sha256
        or hashlib.sha256(projection_raw).hexdigest() != execution.projection_receipt_sha256
        or execution.projection_receipt_relative_path != bundle.projection_receipt_relative_path
        or execution.projection_receipt_sha256 != bundle.projection_receipt_sha256
        or _v19_ocr_material(calibration_profile) != _v19_ocr_material(profile)
        or calibration_configuration_sha256 != calibration.calibration_runtime_configuration_sha256
        or calibration_profile.raster_text_profile_revision
        != calibration.calibration_raster_text_profile_revision
        or profile.raster_text_profile_revision != execution.raster_text_profile_revision
        or runtime_configuration_sha256 != execution.runtime_configuration_sha256
        or profile.extractor_revision != execution.extractor_revision
        or bundle.raster_text_profile_revision != profile.raster_text_profile_revision
        or bundle.runtime_configuration_sha256 != runtime_configuration_sha256
        or bundle.extractor_revision != profile.extractor_revision
        or calibration.raster_text_profile_revision != profile.raster_text_profile_revision
        or calibration.runtime_configuration_sha256 != runtime_configuration_sha256
        or calibration.extractor_revision != profile.extractor_revision
        or calibration.minimum_page_confidence != profile.minimum_page_confidence
        or calibration.minimum_page_confidence != floor
        or calibration.projection_kind != "calibration-10-family-v1"
        or calibration.family_count != 10
        or calibration.asset_count != 240
        or calibration.projection_receipt_path
        != calibration_execution.projection_receipt_relative_path
        or calibration.projection_receipt_sha256 != calibration_execution.projection_receipt_sha256
        or calibration.projection_receipt_path
        != calibration_bundle.projection_receipt_relative_path
        or calibration.projection_receipt_sha256 != calibration_bundle.projection_receipt_sha256
        or calibration_source.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_source.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration.selection_receipt_path != source.selection_receipt_relative_path
        or calibration.selection_receipt_sha256 != source.selection_receipt_sha256
        or execution.selection_receipt_relative_path != source.selection_receipt_relative_path
        or execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or bundle.selection_receipt_relative_path != source.selection_receipt_relative_path
        or bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_execution.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_execution.selection_receipt_sha256 != source.selection_receipt_sha256
        or calibration_bundle.selection_receipt_relative_path
        != source.selection_receipt_relative_path
        or calibration_bundle.selection_receipt_sha256 != source.selection_receipt_sha256
        or hashlib.sha256(calibration_source_raw).hexdigest()
        != calibration.source_gold_receipt_sha256
        or hashlib.sha256(calibration_execution_raw).hexdigest()
        != calibration.candidate_execution_sha256
        or source.selection_receipt_relative_path
        != profile.source_recognizer_selection_receipt_relative_path
        or source.selection_receipt_sha256 != profile.source_recognizer_selection_receipt_sha256
        or selection.parent_selection_receipt_sha256
        != profile.source_recognizer_parent_selection_receipt_sha256
        or calibration.source_recognizer_release_sha256 != profile.source_recognizer_release_sha256
        or calibration.source_recognizer_receipt_sha256 != profile.source_recognizer_receipt_sha256
        or calibration.source_recognizer_training_intent_sha256
        != profile.source_recognizer_training_intent_sha256
        or calibration.source_recognizer_parent_selection_receipt_sha256
        != profile.source_recognizer_parent_selection_receipt_sha256
    ):
        raise SyntheticRasterTextActivationError(invalid)
    artifact_kwargs: dict[str, object] = {
        "metric_revision": RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
        "evidence_revision": ("source-attested-public-finetuned-staged-train-holdout-v2"),
        "latency_gate_revision": "warm-p95-10s-v1",
        "projection_kind": sealed_projection_kind,
        "stage1_decision": sealed_stage1_decision,
        "family_count": sealed_family_count,
        "asset_count": sealed_asset_count,
        "raster_text_profile_revision": profile.raster_text_profile_revision,
        "runtime_configuration_sha256": runtime_configuration_sha256,
        "extractor_revision": profile.extractor_revision,
        "selection_receipt_path": source.selection_receipt_relative_path,
        "selection_receipt_sha256": source.selection_receipt_sha256,
        "projection_receipt_path": execution.projection_receipt_relative_path,
        "projection_receipt_sha256": execution.projection_receipt_sha256,
        "stage1_decision_receipt_path": stage1_decision_receipt_relative_path,
        "stage1_decision_receipt_sha256": decision_sha256,
        "source_gold_receipt_path": source_gold_receipt_relative_path,
        "source_gold_receipt_sha256": source_sha256,
        "calibration_receipt_path": calibration_receipt_relative_path,
        "calibration_receipt_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "candidate_execution_path": candidate_execution_relative_path,
        "candidate_execution_sha256": execution_sha256,
        "warm_service_smoke_receipt_path": warm_service_smoke_receipt_relative_path,
        "warm_service_smoke_receipt_sha256": hashlib.sha256(warm_smoke_raw).hexdigest(),
        "latency_p95_seconds": execution.latency_p95_seconds,
        "service_warm_latency_p95_seconds": warm_smoke.warm_latency_p95_seconds,
        "minimum_page_confidence": profile.minimum_page_confidence,
        "source_recognizer_release_sha256": profile.source_recognizer_release_sha256,
        "source_recognizer_receipt_sha256": profile.source_recognizer_receipt_sha256,
        "source_recognizer_training_intent_sha256": (
            profile.source_recognizer_training_intent_sha256
        ),
        "source_recognizer_parent_selection_receipt_sha256": (
            profile.source_recognizer_parent_selection_receipt_sha256
        ),
    }
    artifact: (
        RasterTextSealedEvaluationV7 | RasterTextSealedEvaluationV8 | RasterTextSealedEvaluationV9
    )
    if current_native_preview97:
        artifact = RasterTextSealedEvaluationV9.model_validate(
            artifact_kwargs
            | {
                "quality_gate_revision": _V19_PREVIEW97_QUALITY_GATE_REVISION,
                "evidence_generation_revision": (_V19_CURRENT_NATIVE_EVIDENCE_GENERATION_REVISION),
                "current_runner_implementation_sha256": current_runner_sha256,
                "scores": [RasterTextSealedProfileScoreV8(**item.model_dump()) for item in scores],
            }
        )
    elif historical_preview97:
        assert preview97_revalidation is not None
        assert revalidation_smoke_raw is not None
        assert revalidation_execution_raw is not None
        artifact = RasterTextSealedEvaluationV8.model_validate(
            artifact_kwargs
            | {
                "quality_gate_revision": _V19_PREVIEW97_QUALITY_GATE_REVISION,
                "evidence_revalidation_revision": (_V19_PREVIEW97_EVIDENCE_REVALIDATION_REVISION),
                "historical_runner_implementation_sha256": cast(str, historical_runner_sha256),
                "current_runner_implementation_sha256": current_runner_sha256,
                "revalidation_smoke_receipt_path": (
                    preview97_revalidation.revalidation_smoke_receipt_relative_path
                ),
                "revalidation_smoke_receipt_sha256": hashlib.sha256(
                    revalidation_smoke_raw
                ).hexdigest(),
                "revalidation_candidate_execution_path": (
                    preview97_revalidation.revalidation_candidate_execution_relative_path
                ),
                "revalidation_candidate_execution_sha256": hashlib.sha256(
                    revalidation_execution_raw
                ).hexdigest(),
                "scores": [RasterTextSealedProfileScoreV8(**item.model_dump()) for item in scores],
            }
        )
    else:
        artifact = RasterTextSealedEvaluationV7.model_validate(
            artifact_kwargs
            | {"scores": [RasterTextSealedProfileScoreV7(**item.model_dump()) for item in scores]}
        )
    digest = _digest_model(artifact)
    relative = {
        (False, False): f"synthetic-raster-text-v1/sealed-v7/{digest}.json",
        (True, False): f"synthetic-raster-text-v1/sealed-v8/{digest}.json",
        (False, True): f"synthetic-raster-text-v1/sealed-v9/{digest}.json",
    }[(historical_preview97, current_native_preview97)]
    try:
        written, result_digest = write_or_verify_canonical_local_json(
            document_preview_root=root,
            relative_path=relative,
            value=artifact.model_dump(mode="json", warnings=False),
        )
    except (OSError, ValueError, RuntimeError):
        code = {
            (False, False): "raster_text_v7_fresh_sealed_evaluation_write_failed",
            (True, False): "raster_text_v8_fresh_sealed_evaluation_write_failed",
            (False, True): "raster_text_v9_fresh_sealed_evaluation_write_failed",
        }[(historical_preview97, current_native_preview97)]
        raise SyntheticRasterTextActivationError(code) from None
    if written != relative or result_digest != digest:
        code = {
            (False, False): "raster_text_v7_fresh_sealed_evaluation_write_failed",
            (True, False): "raster_text_v8_fresh_sealed_evaluation_write_failed",
            (False, True): "raster_text_v9_fresh_sealed_evaluation_write_failed",
        }[(historical_preview97, current_native_preview97)]
        raise SyntheticRasterTextActivationError(code)
    return relative, digest


def write_raster_semantic_text_fresh_sealed_evaluation_v7(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    stage1_decision_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfileV19,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write the unchanged 98%-Gate V19 sealed evaluation."""

    return _write_raster_semantic_text_fresh_sealed_evaluation_v7_or_v8(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        calibration_receipt_relative_path=calibration_receipt_relative_path,
        stage1_decision_receipt_relative_path=stage1_decision_receipt_relative_path,
        warm_service_smoke_receipt_relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
    )


def write_raster_semantic_text_fresh_sealed_evaluation_v8(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    stage1_decision_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    revalidation_smoke_receipt_relative_path: str,
    revalidation_candidate_execution_relative_path: str,
    profile: RasterTextOfflineProfileV19,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Promote only the fixed V19 chain under the explicit preview-97 Gate."""

    return _write_raster_semantic_text_fresh_sealed_evaluation_v7_or_v8(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        calibration_receipt_relative_path=calibration_receipt_relative_path,
        stage1_decision_receipt_relative_path=stage1_decision_receipt_relative_path,
        warm_service_smoke_receipt_relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
        preview97_revalidation=_Preview97RevalidationV1(
            revalidation_smoke_receipt_relative_path=(revalidation_smoke_receipt_relative_path),
            revalidation_candidate_execution_relative_path=(
                revalidation_candidate_execution_relative_path
            ),
        ),
    )


def write_raster_semantic_text_fresh_sealed_evaluation_v9(
    *,
    document_preview_root: Path,
    source_gold_receipt_relative_path: str,
    candidate_execution_relative_path: str,
    calibration_receipt_relative_path: str,
    stage1_decision_receipt_relative_path: str,
    warm_service_smoke_receipt_relative_path: str,
    profile: RasterTextOfflineProfileV19,
    runtime_configuration_sha256: str,
) -> tuple[str, str]:
    """Write a fully current native V19 evaluation under the preview-97 Gate."""

    return _write_raster_semantic_text_fresh_sealed_evaluation_v7_or_v8(
        document_preview_root=document_preview_root,
        source_gold_receipt_relative_path=source_gold_receipt_relative_path,
        candidate_execution_relative_path=candidate_execution_relative_path,
        calibration_receipt_relative_path=calibration_receipt_relative_path,
        stage1_decision_receipt_relative_path=stage1_decision_receipt_relative_path,
        warm_service_smoke_receipt_relative_path=warm_service_smoke_receipt_relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
        current_native_preview97=True,
    )


@contextmanager
def _deny_network() -> Iterator[None]:
    """Reject all parent-process network routes during auto activation."""

    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def deny(*_args: object, **_kwargs: object) -> object:
        raise SyntheticRasterTextActivationError("raster_text_network_forbidden")

    # Keep ``socket.socket`` a type because package import code can use
    # ``isinstance`` checks.  The subclass preserves local construction but
    # blocks each route that could establish a peer or emit a packet.
    class _BlockedSocket(original_socket):  # type: ignore[misc,valid-type]
        def connect(self, *_args: object, **_kwargs: object) -> None:
            deny()

        def connect_ex(self, *_args: object, **_kwargs: object) -> int:
            deny()
            raise AssertionError("unreachable")

        def sendto(self, *_args: object, **_kwargs: object) -> int:
            deny()
            raise AssertionError("unreachable")

    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]
    socket.create_connection = deny  # type: ignore[assignment]
    socket.getaddrinfo = deny  # type: ignore[assignment]
    socket.gethostbyname = deny  # type: ignore[assignment]
    socket.gethostbyname_ex = deny  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment,misc]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
        socket.gethostbyname = original_gethostbyname  # type: ignore[assignment]
        socket.gethostbyname_ex = original_gethostbyname_ex  # type: ignore[assignment]


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    try:
        with exclusive_local_process_lock(root / _ACTIVATION_LOCK_NAME):
            yield
    except LocalProcessLockBusy:
        raise SyntheticRasterTextActivationError("raster_text_auto_activation_locked") from None
    except LocalProcessLockInvalid:
        raise SyntheticRasterTextActivationError(
            "raster_text_auto_activation_lock_invalid"
        ) from None


@dataclass(frozen=True, slots=True)
class SyntheticRasterTextAutoActivationResultV1:
    raster_text_profile_revision: str
    activation_pointer_sha256: str
    sealed_evaluation_relative_path: str
    sealed_evaluation_sha256: str


def _require_auto_activation_candidate_latency(
    run: RasterTextCandidateRunV1,
) -> None:
    """Reject a measured candidate before any later activation work."""

    if (
        not math.isfinite(run.latency_p95_seconds)
        or run.latency_p95_seconds <= 0
        or run.latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS
    ):
        raise SyntheticRasterTextActivationError("raster_text_latency_gate_failed")


def _require_auto_activation_warm_latency(
    *,
    root: Path,
    relative_path: str,
    profile: (
        RasterTextOfflineProfileV15 | RasterTextOfflineProfileV18 | RasterTextOfflineProfileV19
    ),
    runtime_configuration_sha256: str,
) -> None:
    """Reopen the bound warm-service receipt and enforce the fixed deadline."""

    warm, _raw = _load_verified_final_warm_smoke(
        root=root,
        relative_path=relative_path,
        profile=profile,
        runtime_configuration_sha256=runtime_configuration_sha256,
    )
    if warm.warm_latency_p95_seconds > _AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS:
        raise SyntheticRasterTextActivationError("raster_text_latency_gate_failed")


class _PreparedRasterTextRuntimeProfile(Protocol):
    @property
    def runtime_configuration_relative_path(self) -> str: ...

    @property
    def runtime_configuration_sha256(self) -> str: ...


def _run_v2_staged_candidate_activation(
    *,
    root: Path,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1,
    existing_dataset: bool,
    dataset_materials_preverified: bool,
    calibration_profile_info: _PreparedRasterTextRuntimeProfile,
    prepare_final_profile: Callable[[float], _PreparedRasterTextRuntimeProfile],
    expected_profile_type: (
        type[RasterTextOfflineProfileV18]
        | type[RasterTextOfflineProfileV19]
        | type[RasterTextOfflineProfileV20]
    ),
    expected_pointer_type: (
        type[RasterTextActivationPointerV6]
        | type[RasterTextActivationPointerV7]
        | type[RasterTextActivationPointerV8]
        | type[RasterTextActivationPointerV9]
        | type[RasterTextActivationPointerV12]
    ),
    write_calibration: Callable[..., tuple[str, str]],
    write_sealed: Callable[..., tuple[str, str]],
    recheck_materials: Callable[[], None],
    drift_code: str,
    quality_gate_revision: _Preview97QualityGateRevision | None = None,
    pre_sealed_activation_check: (
        Callable[
            [
                _PreparedRasterTextRuntimeProfile,
                RasterTextOfflineProfileV18
                | RasterTextOfflineProfileV19
                | RasterTextOfflineProfileV20,
            ],
            BlRepairAuditCandidateRunV1,
        ]
        | None
    ) = None,
) -> SyntheticRasterTextAutoActivationResultV1:
    """Run the shared V2-holdout smoke/calibration/staged-sealed transaction.

    Model acquisition and its model-specific attestation stay outside this
    helper.  V19's expensive 24/240/1,200(+1,200) sequence, latency checks,
    source-only evidence replay, final material rehash and atomic pointer swap
    stay in one fail-closed path while the historical V6/V18 path remains
    byte-compatible and independently callable.
    """

    selection_revision = _FRESH_HOLDOUT_SELECTION_REVISION_V2
    if quality_gate_revision is None:
        write_smoke = write_raster_semantic_text_smoke_quality_screen_v3
    elif quality_gate_revision == _V19_PREVIEW97_QUALITY_GATE_REVISION:
        write_smoke = write_raster_semantic_text_smoke_quality_screen_preview97_v1
    else:
        raise SyntheticRasterTextActivationError("raster_text_smoke_quality_screen_invalid")
    preflight_smoke = run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path=(
            calibration_profile_info.runtime_configuration_relative_path
        ),
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope="smoke",
        preflight_smoke_only=existing_dataset,
        build_inputs=build_inputs if existing_dataset else None,
    )
    _require_auto_activation_candidate_latency(preflight_smoke)
    _preflight_path, _preflight_digest, preflight_passed = write_smoke(
        document_preview_root=root,
        source_gold_receipt_relative_path=(preflight_smoke.source_gold_receipt_relative_path),
        candidate_execution_relative_path=(preflight_smoke.candidate_execution_relative_path),
    )
    if not preflight_passed:
        raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")
    if existing_dataset and not dataset_materials_preverified:
        compile_synthetic_raster_dataset(
            document_preview_root=root,
            build_inputs_relative_path=build_inputs.build_inputs_relative_path,
        )
    write_fresh_raster_text_scope_selection_receipt(
        document_preview_root=root,
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        selection_revision=selection_revision,
    )
    calibration = run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path=(
            calibration_profile_info.runtime_configuration_relative_path
        ),
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope="fresh-calibration",
        fresh_projection_kind="calibration-10-family-v1",
        fresh_selection_revision=selection_revision,
    )
    _require_auto_activation_candidate_latency(calibration)
    _source, _execution, _candidate_bundle, calibration_cases = (
        verify_source_attested_fresh_raster_text_evidence_v6(
            document_preview_root=root,
            source_gold_receipt_relative_path=(calibration.source_gold_receipt_relative_path),
            candidate_execution_relative_path=(calibration.candidate_execution_relative_path),
            scope="fresh-calibration",
        )
    )
    floor = _calibrated_floor(
        calibration_cases,
        metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
    )
    final_profile_info = prepare_final_profile(floor)
    final_profile_raw = _read(
        root,
        final_profile_info.runtime_configuration_relative_path,
        limit=_MAX_RECEIPT_BYTES,
        code="raster_text_profile_invalid",
    )
    final_profile = parse_raster_text_offline_profile(final_profile_raw)
    if type(final_profile) is not expected_profile_type:
        raise SyntheticRasterTextActivationError("raster_text_profile_invalid")
    final_smoke = run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path=(
            final_profile_info.runtime_configuration_relative_path
        ),
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope="smoke",
    )
    _require_auto_activation_candidate_latency(final_smoke)
    _final_path, _final_digest, final_smoke_passed = write_smoke(
        document_preview_root=root,
        source_gold_receipt_relative_path=(final_smoke.source_gold_receipt_relative_path),
        candidate_execution_relative_path=(final_smoke.candidate_execution_relative_path),
    )
    if not final_smoke_passed:
        raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")
    warm_smoke_path, _warm_smoke_digest = run_warm_service_raster_text_smoke(
        document_preview_root=root,
        runtime_configuration_relative_path=(
            final_profile_info.runtime_configuration_relative_path
        ),
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        source_gold_receipt_relative_path=(final_smoke.source_gold_receipt_relative_path),
        candidate_execution_relative_path=(final_smoke.candidate_execution_relative_path),
    )
    _require_auto_activation_warm_latency(
        root=root,
        relative_path=warm_smoke_path,
        profile=final_profile,
        runtime_configuration_sha256=(final_profile_info.runtime_configuration_sha256),
    )
    repair_audit: BlRepairAuditCandidateRunV1 | None = None
    if pre_sealed_activation_check is not None:
        repair_audit = pre_sealed_activation_check(final_profile_info, final_profile)
        if (
            repair_audit.raster_text_profile_revision != final_profile.raster_text_profile_revision
            or repair_audit.runtime_configuration_sha256
            != final_profile_info.runtime_configuration_sha256
        ):
            raise SyntheticRasterTextActivationError("bl_repair_audit_evidence_invalid")
    calibration_receipt_path, _calibration_digest = write_calibration(
        document_preview_root=root,
        source_gold_receipt_relative_path=(calibration.source_gold_receipt_relative_path),
        candidate_execution_relative_path=(calibration.candidate_execution_relative_path),
        calibration_runtime_configuration_relative_path=(
            calibration_profile_info.runtime_configuration_relative_path
        ),
        activated_profile=final_profile,
        activated_runtime_configuration_sha256=(final_profile_info.runtime_configuration_sha256),
    )
    sealed_stage1 = run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path=(
            final_profile_info.runtime_configuration_relative_path
        ),
        synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
        scope="fresh-sealed",
        fresh_projection_kind="sealed-stage1-50-family-v1",
        fresh_selection_revision=selection_revision,
    )
    _require_auto_activation_candidate_latency(sealed_stage1)
    stage1_decision_path, _stage1_digest, stage1_decision = (
        write_fresh_raster_text_sealed_stage1_decision(
            document_preview_root=root,
            source_gold_receipt_relative_path=(sealed_stage1.source_gold_receipt_relative_path),
            candidate_execution_relative_path=(sealed_stage1.candidate_execution_relative_path),
            selection_revision=selection_revision,
            quality_gate_revision=quality_gate_revision,
        )
    )
    if stage1_decision == "early_fail":
        raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
    if stage1_decision == "early_pass":
        sealed = sealed_stage1
    else:
        sealed = run_paddleocr3_synthetic_raster_text_candidates(
            document_preview_root=root,
            runtime_configuration_relative_path=(
                final_profile_info.runtime_configuration_relative_path
            ),
            synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
            scope="fresh-sealed",
            fresh_projection_kind="sealed-stage2-100-family-v1",
            parent_stage1_decision_relative_path=stage1_decision_path,
            fresh_selection_revision=selection_revision,
        )
        _require_auto_activation_candidate_latency(sealed)
    sealed_path, sealed_digest = write_sealed(
        document_preview_root=root,
        source_gold_receipt_relative_path=(sealed.source_gold_receipt_relative_path),
        candidate_execution_relative_path=(sealed.candidate_execution_relative_path),
        calibration_receipt_relative_path=calibration_receipt_path,
        stage1_decision_receipt_relative_path=stage1_decision_path,
        warm_service_smoke_receipt_relative_path=warm_smoke_path,
        profile=final_profile,
        runtime_configuration_sha256=(final_profile_info.runtime_configuration_sha256),
    )
    recheck_materials()
    if repair_audit is not None:
        _verify_bl_repair_audit_activation_evidence(
            root=root,
            runtime_configuration_relative_path=(
                final_profile_info.runtime_configuration_relative_path
            ),
            evidence=repair_audit,
        )
    final_resolution = resolve_paddleocr3_raster_text_candidate(
        document_preview_root=root,
        configuration_relative_path=(final_profile_info.runtime_configuration_relative_path),
    )
    if (
        final_resolution.status != "available"
        or type(final_resolution.profile) is not expected_profile_type
        or final_resolution.profile != final_profile
        or final_resolution.runtime_configuration_sha256
        != final_profile_info.runtime_configuration_sha256
    ):
        raise SyntheticRasterTextActivationError(drift_code)
    receipt, _receipt_sha, _source_receipt, _source_receipt_sha = _load_fresh_dataset_receipts(
        root,
        SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
    )
    if receipt.build_inputs_relative_path != build_inputs.build_inputs_relative_path:
        raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid")
    if repair_audit is None:
        pointer_sha, pointer = write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path=(
                final_profile_info.runtime_configuration_relative_path
            ),
            sealed_evaluation_relative_path=sealed_path,
        )
    else:
        pointer_sha, pointer = write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path=(
                final_profile_info.runtime_configuration_relative_path
            ),
            sealed_evaluation_relative_path=sealed_path,
            repair_audit_dataset_receipt_relative_path=(repair_audit.dataset_receipt_relative_path),
            repair_audit_evaluation_relative_path=(repair_audit.evaluation_relative_path),
        )
    if not isinstance(pointer, expected_pointer_type):
        raise SyntheticRasterTextActivationError("raster_text_pointer_generation_invalid")
    return SyntheticRasterTextAutoActivationResultV1(
        raster_text_profile_revision=pointer.raster_text_profile_revision,
        activation_pointer_sha256=pointer_sha,
        sealed_evaluation_relative_path=sealed_path,
        sealed_evaluation_sha256=sealed_digest,
    )


def auto_activate_synthetic_raster_text_preview97_v1(
    *, document_preview_root: Path
) -> SyntheticRasterTextAutoActivationResultV1:
    """Revalidate and atomically promote the retained best V19 preview lane.

    This intentionally does not retrain or rerun the 2,400 sealed OCR jobs.
    It reopens every immutable source/Gold/checkpoint, recomputes all 12
    quality cells under the explicit 97%-critical preview policy, and runs a
    fresh 24-asset candidate plus persistent-service smoke with the current
    implementation.  Only an exact historical/current smoke result match can
    reach the V11 pointer writer.
    """

    with _deny_network():
        root = _root(document_preview_root)
        with _lock(root):
            try:
                profile_raw = _read(
                    root,
                    _V19_PREVIEW97_PROFILE_RELATIVE_PATH,
                    limit=_MAX_RECEIPT_BYTES,
                    code="raster_text_preview97_profile_invalid",
                )
                profile = parse_raster_text_offline_profile(profile_raw)
                runtime_configuration_sha256 = hashlib.sha256(profile_raw).hexdigest()
                if (
                    type(profile) is not RasterTextOfflineProfileV19
                    or profile.engine_kind != _V19_ENGINE_KIND
                    or profile.raster_text_profile_revision
                    != "synthetic-raster-text-v1:9eb7f34b40da6ba6c7316e8b"
                ):
                    raise SyntheticRasterTextActivationError(
                        "raster_text_preview97_profile_invalid"
                    )

                recognizer_release = load_verified_public_finetuned_recognizer_release_v1(
                    root, profile.source_recognizer_release_sha256
                )
                model_bundle = load_verified_raster_text_public_finetuned_paddle_bundle_v1(
                    root, profile.mobile_model_bundle_manifest_sha256
                )
                _verify_v19_public_finetune_contract(
                    root=root,
                    recognizer_release=recognizer_release,
                    model_bundle=model_bundle,
                    profile=profile,
                )
                resolution = resolve_paddleocr3_raster_text_candidate(
                    document_preview_root=root,
                    configuration_relative_path=_V19_PREVIEW97_PROFILE_RELATIVE_PATH,
                )
                if (
                    resolution.status != "available"
                    or type(resolution.profile) is not RasterTextOfflineProfileV19
                    or resolution.profile != profile
                    or resolution.runtime_configuration_sha256 != runtime_configuration_sha256
                ):
                    raise SyntheticRasterTextActivationError(
                        "raster_text_preview97_profile_invalid"
                    )

                revalidation = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=root,
                    runtime_configuration_relative_path=(_V19_PREVIEW97_PROFILE_RELATIVE_PATH),
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    scope="smoke",
                )
                _require_auto_activation_candidate_latency(revalidation)
                _screen_path, _screen_digest, smoke_passed = (
                    write_raster_semantic_text_smoke_quality_screen_v3(
                        document_preview_root=root,
                        source_gold_receipt_relative_path=(
                            revalidation.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            revalidation.candidate_execution_relative_path
                        ),
                    )
                )
                if not smoke_passed:
                    raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")
                revalidation_warm_path, _revalidation_warm_sha256 = (
                    run_warm_service_raster_text_smoke(
                        document_preview_root=root,
                        runtime_configuration_relative_path=(_V19_PREVIEW97_PROFILE_RELATIVE_PATH),
                        synthetic_raster_receipt_relative_path=(
                            SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH
                        ),
                        source_gold_receipt_relative_path=(
                            revalidation.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            revalidation.candidate_execution_relative_path
                        ),
                    )
                )
                _require_auto_activation_warm_latency(
                    root=root,
                    relative_path=revalidation_warm_path,
                    profile=profile,
                    runtime_configuration_sha256=runtime_configuration_sha256,
                )

                sealed_path, sealed_sha256 = write_raster_semantic_text_fresh_sealed_evaluation_v8(
                    document_preview_root=root,
                    source_gold_receipt_relative_path=(
                        _V19_PREVIEW97_SEALED_SOURCE_GOLD_RELATIVE_PATH
                    ),
                    candidate_execution_relative_path=(
                        _V19_PREVIEW97_SEALED_EXECUTION_RELATIVE_PATH
                    ),
                    calibration_receipt_relative_path=(
                        _V19_PREVIEW97_CALIBRATION_RECEIPT_RELATIVE_PATH
                    ),
                    stage1_decision_receipt_relative_path=(
                        _V19_PREVIEW97_STAGE1_DECISION_RELATIVE_PATH
                    ),
                    warm_service_smoke_receipt_relative_path=(
                        _V19_PREVIEW97_HISTORICAL_WARM_SMOKE_RELATIVE_PATH
                    ),
                    revalidation_smoke_receipt_relative_path=(revalidation_warm_path),
                    revalidation_candidate_execution_relative_path=(
                        revalidation.candidate_execution_relative_path
                    ),
                    profile=profile,
                    runtime_configuration_sha256=runtime_configuration_sha256,
                )

                # Reopen every model and runtime boundary after the measured
                # run and before the only mutable operation: pointer replace.
                rechecked_release = load_verified_public_finetuned_recognizer_release_v1(
                    root, profile.source_recognizer_release_sha256
                )
                rechecked_model_bundle = (
                    load_verified_raster_text_public_finetuned_paddle_bundle_v1(
                        root, profile.mobile_model_bundle_manifest_sha256
                    )
                )
                _verify_v19_public_finetune_contract(
                    root=root,
                    recognizer_release=rechecked_release,
                    model_bundle=rechecked_model_bundle,
                    profile=profile,
                )
                final_resolution = resolve_paddleocr3_raster_text_candidate(
                    document_preview_root=root,
                    configuration_relative_path=_V19_PREVIEW97_PROFILE_RELATIVE_PATH,
                )
                if (
                    final_resolution.status != "available"
                    or type(final_resolution.profile) is not RasterTextOfflineProfileV19
                    or final_resolution.profile != profile
                    or final_resolution.runtime_configuration_sha256 != runtime_configuration_sha256
                ):
                    raise SyntheticRasterTextActivationError("raster_text_preview97_material_drift")
                pointer_sha256, pointer = write_source_attested_synthetic_raster_text_pointer(
                    document_preview_root=root,
                    runtime_configuration_relative_path=(_V19_PREVIEW97_PROFILE_RELATIVE_PATH),
                    sealed_evaluation_relative_path=sealed_path,
                )
                if type(pointer) is not RasterTextActivationPointerV11:
                    raise SyntheticRasterTextActivationError(
                        "raster_text_pointer_generation_invalid"
                    )
                return SyntheticRasterTextAutoActivationResultV1(
                    raster_text_profile_revision=pointer.raster_text_profile_revision,
                    activation_pointer_sha256=pointer_sha256,
                    sealed_evaluation_relative_path=sealed_path,
                    sealed_evaluation_sha256=sealed_sha256,
                )
            except SyntheticRasterTextActivationError:
                raise
            except (
                RasterTextPublicFineTunedPaddleBundleError,
                SourceAttestedPublicRecognizerFineTuneError,
                OSError,
                ValidationError,
                ValueError,
                RuntimeError,
            ):
                raise SyntheticRasterTextActivationError(
                    "raster_text_preview97_auto_activation_failed"
                ) from None


def auto_activate_synthetic_raster_text_v1(
    *, document_preview_root: Path
) -> SyntheticRasterTextAutoActivationResultV1:
    """Perform Gold → smoke → fresh calibration/sealed → activation.

    The only activation candidate is public-mobile V15 with its fixed V14
    OCR semantics and bounded oneDNN runtime controls.  It reuses the verified
    public detector and recognizer bundle and binds every quality decision to
    semantic metric V3, staged source-attested holdouts, and a 10-second p95
    deadline.  V1--V14, trained
    recognizers, and combined bundles remain comparison material only; this
    transaction neither prepares nor falls back to them.
    """

    with _deny_network():
        try:
            sidecar_bundle = materialize_verified_raster_text_direct_paddle_bundle_v1(
                document_preview_root=document_preview_root
            )
            mobile_bundle = materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=sidecar_bundle.document_preview_root
            )
        except SyntheticRasterTextActivationError:
            raise
        except (RasterTextDirectPaddleBundleError, RasterTextMobilePaddleBundleError) as error:
            # A completely absent local source is an unmet offline-bundle
            # prerequisite, not a candidate/Gate failure.  Other attestation
            # failures retain their precise content-free code.
            if error.code in {
                "direct_paddle_bundle_path_invalid",
                "mobile_paddle_bundle_path_invalid",
            }:
                raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
            raise SyntheticRasterTextActivationError(error.code) from None
        with _lock(sidecar_bundle.document_preview_root):
            try:
                build_inputs = ensure_fixed_synthetic_raster_build_inputs(
                    document_preview_root=sidecar_bundle.document_preview_root
                )
                # A new dataset must be built and fully verified before any
                # OCR can run.  A completed immutable dataset instead gets
                # the fixed 24-item smoke-family replay first; after it
                # passes, the unchanged compiler call below performs the
                # original full 24,000-asset replay before calibration.
                existing_dataset = _existing_synthetic_raster_dataset(
                    sidecar_bundle.document_preview_root
                )
                if not existing_dataset:
                    compile_synthetic_raster_dataset(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        build_inputs_relative_path=build_inputs.build_inputs_relative_path,
                    )
                # The zero-floor public V15 binding must pass the fixed V3
                # semantic and 10-second p95 smoke Gates before the projected
                # 240-asset calibration is allowed to run.
                # This screen is deliberately an early rejection only: its
                # evidence is never used as a substitute for the separately
                # rerun final-profile smoke below.
                calibration_profile = prepare_raster_text_direct_mobile_runtime_profile_v15(
                    sidecar_bundle,
                    mobile_bundle,
                    pdfium_version=build_inputs.renderer_configuration.pdfium_version,
                    minimum_page_confidence=0.0,
                    cpu_threads=_AUTO_ACTIVATION_V15_CPU_THREADS,
                    recognition_batch_size=(_AUTO_ACTIVATION_V15_RECOGNITION_BATCH_SIZE),
                    opencv_threads=_AUTO_ACTIVATION_V15_OPENCV_THREADS,
                )
                preflight_smoke = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        calibration_profile.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                    scope="smoke",
                    preflight_smoke_only=existing_dataset,
                    build_inputs=build_inputs if existing_dataset else None,
                )
                _require_auto_activation_candidate_latency(preflight_smoke)
                _preflight_screen_path, _preflight_screen_digest, preflight_passed = (
                    write_raster_semantic_text_smoke_quality_screen_v3(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        source_gold_receipt_relative_path=(
                            preflight_smoke.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            preflight_smoke.candidate_execution_relative_path
                        ),
                    )
                )
                if not preflight_passed:
                    raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")
                if existing_dataset:
                    compile_synthetic_raster_dataset(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        build_inputs_relative_path=build_inputs.build_inputs_relative_path,
                    )
                write_fresh_raster_text_scope_selection_receipt(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                )
                calibration = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        calibration_profile.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                    scope="fresh-calibration",
                    fresh_projection_kind="calibration-10-family-v1",
                )
                _require_auto_activation_candidate_latency(calibration)
                _source, _execution, _candidate_bundle, cases = (
                    verify_source_attested_fresh_raster_text_evidence(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        source_gold_receipt_relative_path=(
                            calibration.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            calibration.candidate_execution_relative_path
                        ),
                        scope="fresh-calibration",
                    )
                )
                floor = _calibrated_floor(
                    cases,
                    metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
                )
                final_profile_info = prepare_raster_text_direct_mobile_runtime_profile_v15(
                    sidecar_bundle,
                    mobile_bundle,
                    pdfium_version=build_inputs.renderer_configuration.pdfium_version,
                    minimum_page_confidence=floor,
                    cpu_threads=_AUTO_ACTIVATION_V15_CPU_THREADS,
                    recognition_batch_size=(_AUTO_ACTIVATION_V15_RECOGNITION_BATCH_SIZE),
                    opencv_threads=_AUTO_ACTIVATION_V15_OPENCV_THREADS,
                )
                final_profile_raw = _read(
                    sidecar_bundle.document_preview_root,
                    final_profile_info.runtime_configuration_relative_path,
                    limit=_MAX_RECEIPT_BYTES,
                    code="raster_text_profile_invalid",
                )
                final_profile = parse_raster_text_offline_profile(final_profile_raw)
                if not isinstance(final_profile, RasterTextOfflineProfileV15):
                    raise SyntheticRasterTextActivationError("raster_text_profile_invalid")
                # A confidence-bound profile has a distinct immutable
                # revision.  Prove both evaluation and customer-service
                # protocols against that exact final binding before sealing.
                final_smoke = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                    scope="smoke",
                )
                _require_auto_activation_candidate_latency(final_smoke)
                _final_screen_path, _final_screen_digest, final_smoke_passed = (
                    write_raster_semantic_text_smoke_quality_screen_v3(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        source_gold_receipt_relative_path=(
                            final_smoke.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            final_smoke.candidate_execution_relative_path
                        ),
                    )
                )
                if not final_smoke_passed:
                    raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")
                warm_smoke_path, _warm_smoke_digest = run_warm_service_raster_text_smoke(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                    source_gold_receipt_relative_path=(
                        final_smoke.source_gold_receipt_relative_path
                    ),
                    candidate_execution_relative_path=(
                        final_smoke.candidate_execution_relative_path
                    ),
                )
                _require_auto_activation_warm_latency(
                    root=sidecar_bundle.document_preview_root,
                    relative_path=warm_smoke_path,
                    profile=final_profile,
                    runtime_configuration_sha256=(final_profile_info.runtime_configuration_sha256),
                )
                calibration_receipt_path, _calibration_digest = (
                    write_raster_semantic_text_fresh_calibration_receipt_v5(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        source_gold_receipt_relative_path=(
                            calibration.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            calibration.candidate_execution_relative_path
                        ),
                        calibration_runtime_configuration_relative_path=(
                            calibration_profile.runtime_configuration_relative_path
                        ),
                        activated_profile=final_profile,
                        activated_runtime_configuration_sha256=(
                            final_profile_info.runtime_configuration_sha256
                        ),
                    )
                )
                sealed_stage1 = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                    scope="fresh-sealed",
                    fresh_projection_kind="sealed-stage1-50-family-v1",
                )
                _require_auto_activation_candidate_latency(sealed_stage1)
                stage1_decision_path, _stage1_decision_digest, stage1_decision = (
                    write_fresh_raster_text_sealed_stage1_decision(
                        document_preview_root=(sidecar_bundle.document_preview_root),
                        source_gold_receipt_relative_path=(
                            sealed_stage1.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            sealed_stage1.candidate_execution_relative_path
                        ),
                    )
                )
                if stage1_decision == "early_fail":
                    raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
                if stage1_decision == "early_pass":
                    sealed = sealed_stage1
                else:
                    sealed = run_paddleocr3_synthetic_raster_text_candidates(
                        document_preview_root=(sidecar_bundle.document_preview_root),
                        runtime_configuration_relative_path=(
                            final_profile_info.runtime_configuration_relative_path
                        ),
                        synthetic_raster_receipt_relative_path=(
                            SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH
                        ),
                        scope="fresh-sealed",
                        fresh_projection_kind="sealed-stage2-100-family-v1",
                        parent_stage1_decision_relative_path=(stage1_decision_path),
                    )
                    _require_auto_activation_candidate_latency(sealed)
                sealed_path, sealed_digest = write_raster_semantic_text_fresh_sealed_evaluation_v5(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    source_gold_receipt_relative_path=(sealed.source_gold_receipt_relative_path),
                    candidate_execution_relative_path=(sealed.candidate_execution_relative_path),
                    calibration_receipt_relative_path=calibration_receipt_path,
                    stage1_decision_receipt_relative_path=stage1_decision_path,
                    warm_service_smoke_receipt_relative_path=warm_smoke_path,
                    profile=final_profile,
                    runtime_configuration_sha256=(final_profile_info.runtime_configuration_sha256),
                )
                # Full sidecar and public mobile model-bundle rehashes plus
                # the source/build receipt replay are the last checks before
                # the single mutable schema-5 pointer write.
                rechecked_sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    manifest_sha256=sidecar_bundle.manifest_sha256,
                )
                rechecked_mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                    document_preview_root=mobile_bundle.document_preview_root,
                    manifest_sha256=mobile_bundle.manifest_sha256,
                )
                if (
                    rechecked_sidecar.manifest_sha256 != sidecar_bundle.manifest_sha256
                    or rechecked_sidecar.bundle_root_relative_path
                    != sidecar_bundle.bundle_root_relative_path
                    or rechecked_mobile.manifest_sha256 != mobile_bundle.manifest_sha256
                    or rechecked_mobile.bundle_root_relative_path
                    != mobile_bundle.bundle_root_relative_path
                ):
                    raise SyntheticRasterTextActivationError("mobile_paddle_bundle_drift")
                receipt, _receipt_sha, _source_receipt, _source_receipt_sha = (
                    _load_fresh_dataset_receipts(
                        sidecar_bundle.document_preview_root,
                        SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                    )
                )
                if receipt.build_inputs_relative_path != build_inputs.build_inputs_relative_path:
                    raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid")
                pointer_sha, pointer = write_source_attested_synthetic_raster_text_pointer(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    sealed_evaluation_relative_path=sealed_path,
                )
                return SyntheticRasterTextAutoActivationResultV1(
                    raster_text_profile_revision=pointer.raster_text_profile_revision,
                    activation_pointer_sha256=pointer_sha,
                    sealed_evaluation_relative_path=sealed_path,
                    sealed_evaluation_sha256=sealed_digest,
                )
            except SyntheticRasterTextActivationError:
                raise
            except (RasterTextDirectPaddleBundleError, RasterTextMobilePaddleBundleError) as error:
                raise SyntheticRasterTextActivationError(error.code) from None
            except (
                SyntheticRasterCompilerUnavailable,
                OSError,
                ValidationError,
                ValueError,
                RuntimeError,
            ):
                raise SyntheticRasterTextActivationError(
                    "raster_text_auto_activation_failed"
                ) from None


def auto_activate_synthetic_raster_text_v6(
    *, document_preview_root: Path
) -> SyntheticRasterTextAutoActivationResultV1:
    """Build, evaluate and activate only the pinned V18 server recognizer.

    This additive lane never falls back to V15 or the first fresh selection.
    Calibration and sealed candidates are projected exclusively from
    ``source-receipt-train-holdout-v2``; schema-6 evidence additionally
    reopens every immutable OCR checkpoint before the pointer can change.
    """

    selection_revision = _FRESH_HOLDOUT_SELECTION_REVISION_V2
    with _deny_network():
        try:
            sidecar_bundle = materialize_verified_raster_text_direct_paddle_bundle_v1(
                document_preview_root=document_preview_root
            )
            mobile_bundle = materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=sidecar_bundle.document_preview_root
            )
            offline_bundle = load_verified_raster_offline_bundle(
                sidecar_bundle.document_preview_root
            )
            server_bundle = materialize_verified_raster_text_direct_server_paddle_bundle_v1(
                mobile_bundle=mobile_bundle,
                offline_bundle=offline_bundle,
            )
        except SyntheticRasterTextActivationError:
            raise
        except (
            RasterTextDirectPaddleBundleError,
            RasterTextMobilePaddleBundleError,
            RasterOfflineBundleError,
            RasterTextServerPaddleBundleError,
        ) as error:
            if error.code in {
                "direct_paddle_bundle_path_invalid",
                "mobile_paddle_bundle_path_invalid",
                "offline_bundle_unavailable",
                "server_paddle_bundle_source_unavailable",
            }:
                raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
            raise SyntheticRasterTextActivationError(error.code) from None

        with _lock(sidecar_bundle.document_preview_root):
            try:
                build_inputs = ensure_fixed_synthetic_raster_build_inputs(
                    document_preview_root=sidecar_bundle.document_preview_root
                )
                existing_dataset = _existing_synthetic_raster_dataset(
                    sidecar_bundle.document_preview_root
                )
                if not existing_dataset:
                    compile_synthetic_raster_dataset(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        build_inputs_relative_path=(build_inputs.build_inputs_relative_path),
                    )
                calibration_profile_info = prepare_raster_text_direct_server_runtime_profile_v18(
                    sidecar_bundle,
                    server_bundle,
                    pdfium_version=(build_inputs.renderer_configuration.pdfium_version),
                    minimum_page_confidence=0.0,
                    cpu_threads=_AUTO_ACTIVATION_V15_CPU_THREADS,
                )
                preflight_smoke = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        calibration_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    scope="smoke",
                    preflight_smoke_only=existing_dataset,
                    build_inputs=build_inputs if existing_dataset else None,
                )
                _require_auto_activation_candidate_latency(preflight_smoke)
                _preflight_path, _preflight_digest, preflight_passed = (
                    write_raster_semantic_text_smoke_quality_screen_v3(
                        document_preview_root=(sidecar_bundle.document_preview_root),
                        source_gold_receipt_relative_path=(
                            preflight_smoke.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            preflight_smoke.candidate_execution_relative_path
                        ),
                    )
                )
                if not preflight_passed:
                    raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")
                if existing_dataset:
                    compile_synthetic_raster_dataset(
                        document_preview_root=sidecar_bundle.document_preview_root,
                        build_inputs_relative_path=(build_inputs.build_inputs_relative_path),
                    )

                write_fresh_raster_text_scope_selection_receipt(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    selection_revision=selection_revision,
                )
                calibration = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        calibration_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    scope="fresh-calibration",
                    fresh_projection_kind="calibration-10-family-v1",
                    fresh_selection_revision=selection_revision,
                )
                _require_auto_activation_candidate_latency(calibration)
                _source, _execution, _candidate_bundle, calibration_cases = (
                    verify_source_attested_fresh_raster_text_evidence_v6(
                        document_preview_root=(sidecar_bundle.document_preview_root),
                        source_gold_receipt_relative_path=(
                            calibration.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            calibration.candidate_execution_relative_path
                        ),
                        scope="fresh-calibration",
                    )
                )
                floor = _calibrated_floor(
                    calibration_cases,
                    metric_revision=RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3,
                )
                final_profile_info = prepare_raster_text_direct_server_runtime_profile_v18(
                    sidecar_bundle,
                    server_bundle,
                    pdfium_version=(build_inputs.renderer_configuration.pdfium_version),
                    minimum_page_confidence=floor,
                    cpu_threads=_AUTO_ACTIVATION_V15_CPU_THREADS,
                )
                final_profile_raw = _read(
                    sidecar_bundle.document_preview_root,
                    final_profile_info.runtime_configuration_relative_path,
                    limit=_MAX_RECEIPT_BYTES,
                    code="raster_text_profile_invalid",
                )
                final_profile = parse_raster_text_offline_profile(final_profile_raw)
                if not isinstance(final_profile, RasterTextOfflineProfileV18):
                    raise SyntheticRasterTextActivationError("raster_text_profile_invalid")

                final_smoke = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    scope="smoke",
                )
                _require_auto_activation_candidate_latency(final_smoke)
                _final_path, _final_digest, final_smoke_passed = (
                    write_raster_semantic_text_smoke_quality_screen_v3(
                        document_preview_root=(sidecar_bundle.document_preview_root),
                        source_gold_receipt_relative_path=(
                            final_smoke.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            final_smoke.candidate_execution_relative_path
                        ),
                    )
                )
                if not final_smoke_passed:
                    raise SyntheticRasterTextActivationError("raster_text_smoke_gate_failed")
                warm_smoke_path, _warm_smoke_digest = run_warm_service_raster_text_smoke(
                    document_preview_root=(sidecar_bundle.document_preview_root),
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    source_gold_receipt_relative_path=(
                        final_smoke.source_gold_receipt_relative_path
                    ),
                    candidate_execution_relative_path=(
                        final_smoke.candidate_execution_relative_path
                    ),
                )
                _require_auto_activation_warm_latency(
                    root=sidecar_bundle.document_preview_root,
                    relative_path=warm_smoke_path,
                    profile=final_profile,
                    runtime_configuration_sha256=(final_profile_info.runtime_configuration_sha256),
                )
                calibration_receipt_path, _calibration_digest = (
                    write_raster_semantic_text_fresh_calibration_receipt_v6(
                        document_preview_root=(sidecar_bundle.document_preview_root),
                        source_gold_receipt_relative_path=(
                            calibration.source_gold_receipt_relative_path
                        ),
                        candidate_execution_relative_path=(
                            calibration.candidate_execution_relative_path
                        ),
                        calibration_runtime_configuration_relative_path=(
                            calibration_profile_info.runtime_configuration_relative_path
                        ),
                        activated_profile=final_profile,
                        activated_runtime_configuration_sha256=(
                            final_profile_info.runtime_configuration_sha256
                        ),
                    )
                )
                sealed_stage1 = run_paddleocr3_synthetic_raster_text_candidates(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    scope="fresh-sealed",
                    fresh_projection_kind="sealed-stage1-50-family-v1",
                    fresh_selection_revision=selection_revision,
                )
                _require_auto_activation_candidate_latency(sealed_stage1)
                (
                    stage1_decision_path,
                    _stage1_decision_digest,
                    stage1_decision,
                ) = write_fresh_raster_text_sealed_stage1_decision(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    source_gold_receipt_relative_path=(
                        sealed_stage1.source_gold_receipt_relative_path
                    ),
                    candidate_execution_relative_path=(
                        sealed_stage1.candidate_execution_relative_path
                    ),
                    selection_revision=selection_revision,
                )
                if stage1_decision == "early_fail":
                    raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")
                if stage1_decision == "early_pass":
                    sealed = sealed_stage1
                else:
                    sealed = run_paddleocr3_synthetic_raster_text_candidates(
                        document_preview_root=(sidecar_bundle.document_preview_root),
                        runtime_configuration_relative_path=(
                            final_profile_info.runtime_configuration_relative_path
                        ),
                        synthetic_raster_receipt_relative_path=(
                            SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH
                        ),
                        scope="fresh-sealed",
                        fresh_projection_kind="sealed-stage2-100-family-v1",
                        parent_stage1_decision_relative_path=(stage1_decision_path),
                        fresh_selection_revision=selection_revision,
                    )
                    _require_auto_activation_candidate_latency(sealed)
                sealed_path, sealed_digest = write_raster_semantic_text_fresh_sealed_evaluation_v6(
                    document_preview_root=(sidecar_bundle.document_preview_root),
                    source_gold_receipt_relative_path=(sealed.source_gold_receipt_relative_path),
                    candidate_execution_relative_path=(sealed.candidate_execution_relative_path),
                    calibration_receipt_relative_path=(calibration_receipt_path),
                    stage1_decision_receipt_relative_path=(stage1_decision_path),
                    warm_service_smoke_receipt_relative_path=(warm_smoke_path),
                    profile=final_profile,
                    runtime_configuration_sha256=(final_profile_info.runtime_configuration_sha256),
                )

                rechecked_sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    manifest_sha256=sidecar_bundle.manifest_sha256,
                )
                rechecked_mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                    document_preview_root=mobile_bundle.document_preview_root,
                    manifest_sha256=mobile_bundle.manifest_sha256,
                )
                rechecked_offline = load_verified_raster_offline_bundle_release(
                    document_preview_root=offline_bundle.document_preview_root,
                    manifest_sha256=offline_bundle.manifest_sha256,
                )
                rechecked_server = load_verified_raster_text_direct_server_paddle_bundle_v1(
                    document_preview_root=server_bundle.document_preview_root,
                    manifest_sha256=server_bundle.manifest_sha256,
                )
                if (
                    rechecked_sidecar.manifest_sha256 != sidecar_bundle.manifest_sha256
                    or rechecked_sidecar.bundle_root_relative_path
                    != sidecar_bundle.bundle_root_relative_path
                    or rechecked_mobile.manifest_sha256 != mobile_bundle.manifest_sha256
                    or rechecked_mobile.bundle_root_relative_path
                    != mobile_bundle.bundle_root_relative_path
                    or rechecked_offline.manifest_sha256 != offline_bundle.manifest_sha256
                    or rechecked_offline.bundle_root_relative_path
                    != offline_bundle.bundle_root_relative_path
                    or rechecked_server.manifest_sha256 != server_bundle.manifest_sha256
                    or rechecked_server.bundle_root_relative_path
                    != server_bundle.bundle_root_relative_path
                    or rechecked_server.manifest.source_mobile_bundle_manifest_sha256
                    != mobile_bundle.manifest_sha256
                    or rechecked_server.manifest.source_offline_bundle_manifest_sha256
                    != offline_bundle.manifest_sha256
                ):
                    raise SyntheticRasterTextActivationError("server_paddle_bundle_drift")
                final_resolution = resolve_paddleocr3_raster_text_candidate(
                    document_preview_root=sidecar_bundle.document_preview_root,
                    configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                )
                if (
                    final_resolution.status != "available"
                    or not isinstance(final_resolution.profile, RasterTextOfflineProfileV18)
                    or final_resolution.profile != final_profile
                    or final_resolution.runtime_configuration_sha256
                    != final_profile_info.runtime_configuration_sha256
                ):
                    raise SyntheticRasterTextActivationError("server_paddle_bundle_drift")
                receipt, _receipt_sha, _source_receipt, _source_receipt_sha = (
                    _load_fresh_dataset_receipts(
                        sidecar_bundle.document_preview_root,
                        SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH,
                    )
                )
                if receipt.build_inputs_relative_path != build_inputs.build_inputs_relative_path:
                    raise SyntheticRasterTextActivationError("raster_text_dataset_material_invalid")
                pointer_sha, pointer = write_source_attested_synthetic_raster_text_pointer(
                    document_preview_root=(sidecar_bundle.document_preview_root),
                    runtime_configuration_relative_path=(
                        final_profile_info.runtime_configuration_relative_path
                    ),
                    sealed_evaluation_relative_path=sealed_path,
                )
                if not isinstance(pointer, RasterTextActivationPointerV6):
                    raise SyntheticRasterTextActivationError(
                        "raster_text_pointer_generation_invalid"
                    )
                return SyntheticRasterTextAutoActivationResultV1(
                    raster_text_profile_revision=(pointer.raster_text_profile_revision),
                    activation_pointer_sha256=pointer_sha,
                    sealed_evaluation_relative_path=sealed_path,
                    sealed_evaluation_sha256=sealed_digest,
                )
            except SyntheticRasterTextActivationError:
                raise
            except (
                RasterTextDirectPaddleBundleError,
                RasterTextMobilePaddleBundleError,
                RasterOfflineBundleError,
                RasterTextServerPaddleBundleError,
            ) as error:
                raise SyntheticRasterTextActivationError(error.code) from None
            except (
                SyntheticRasterCompilerUnavailable,
                OSError,
                ValidationError,
                ValueError,
                RuntimeError,
            ):
                raise SyntheticRasterTextActivationError(
                    "raster_text_v6_auto_activation_failed"
                ) from None


def auto_activate_synthetic_raster_text_linux_preview97_v1(
    *, document_preview_root: Path
) -> SyntheticRasterTextAutoActivationResultV1:
    """Fully evaluate and activate the fixed one-pass V19 lane on Linux.

    Unlike the schema-11 compatibility command, this path consumes no
    historical Windows OCR evidence.  It requires a native schema-2
    Linux/AMD64/CPython-3.11 direct bundle, reuses only the fixed public model
    bytes, and produces the complete current-runner 24/240/1,200(+1,200)
    evidence chain before the schema-12 pointer can be replaced.
    """

    with _deny_network():
        try:
            root = _root(document_preview_root)
            if current_raster_runtime_platform_binding() != ("linux", "amd64", "cp311"):
                raise SyntheticRasterTextActivationError(
                    "raster_text_linux_activation_platform_invalid"
                )
            offline_bundle = load_verified_raster_offline_bundle(root)
            if offline_bundle.manifest.schema_version != "2.0" or (
                offline_bundle.manifest.runtime_operating_system,
                offline_bundle.manifest.runtime_architecture,
                offline_bundle.manifest.runtime_python_abi,
            ) != ("linux", "amd64", "cp311"):
                raise SyntheticRasterTextActivationError("raster_text_linux_bundle_invalid")
            sidecar_bundle = materialize_verified_raster_text_direct_paddle_bundle_v1(
                document_preview_root=root,
                source_manifest_sha256=offline_bundle.manifest_sha256,
            )
            if (
                sidecar_bundle.manifest.schema_version != "2.0"
                or sidecar_bundle.manifest.source_bundle_manifest_sha256
                != offline_bundle.manifest_sha256
                or (
                    sidecar_bundle.manifest.runtime_operating_system,
                    sidecar_bundle.manifest.runtime_architecture,
                    sidecar_bundle.manifest.runtime_python_abi,
                )
                != ("linux", "amd64", "cp311")
            ):
                raise SyntheticRasterTextActivationError("raster_text_linux_bundle_invalid")
        except SyntheticRasterTextActivationError:
            raise
        except (RasterOfflineBundleError, RasterTextDirectPaddleBundleError) as error:
            if error.code in {
                "offline_bundle_unavailable",
                "offline_bundle_release_selector_invalid",
                "direct_paddle_bundle_path_invalid",
            }:
                raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
            raise SyntheticRasterTextActivationError(error.code) from None

        with _lock(root):
            try:
                build_inputs = ensure_fixed_synthetic_raster_build_inputs(
                    document_preview_root=root
                )
                existing_dataset = _existing_synthetic_raster_dataset(root)
                # The public model is fixed and already trained.  Linux still
                # performs a complete source/raster/Gold replay before any
                # current-runner score is accepted.
                compile_synthetic_raster_dataset(
                    document_preview_root=root,
                    build_inputs_relative_path=build_inputs.build_inputs_relative_path,
                )
                write_fresh_raster_text_scope_selection_receipt(
                    document_preview_root=root,
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
                )
                recognizer_release = load_verified_public_finetuned_recognizer_release_v1(
                    root, _V19_PUBLIC_FINETUNED_RECOGNIZER_RELEASE_SHA256
                )
                model_bundle = load_verified_raster_text_public_finetuned_paddle_bundle_v1(
                    root, _V19_PUBLIC_FINETUNED_MODEL_BUNDLE_SHA256
                )
                recognizer_release = _verify_v19_public_finetune_contract(
                    root=root,
                    recognizer_release=recognizer_release,
                    model_bundle=model_bundle,
                )

                def prepare_profile(
                    minimum_page_confidence: float,
                ) -> _PreparedRasterTextRuntimeProfile:
                    return prepare_raster_text_public_finetuned_one_pass_runtime_profile_v19(
                        sidecar_bundle,
                        model_bundle,
                        pdfium_version=build_inputs.renderer_configuration.pdfium_version,
                        minimum_page_confidence=minimum_page_confidence,
                        cpu_threads=_AUTO_ACTIVATION_V15_CPU_THREADS,
                    )

                calibration_profile_info = prepare_profile(0.0)
                calibration_profile_raw = _read(
                    root,
                    calibration_profile_info.runtime_configuration_relative_path,
                    limit=_MAX_RECEIPT_BYTES,
                    code="raster_text_profile_invalid",
                )
                calibration_profile = parse_raster_text_offline_profile(calibration_profile_raw)
                if (
                    type(calibration_profile) is not RasterTextOfflineProfileV19
                    or calibration_profile.engine_kind != _V19_ENGINE_KIND
                ):
                    raise SyntheticRasterTextActivationError("raster_text_profile_invalid")
                _verify_v19_public_finetune_contract(
                    root=root,
                    recognizer_release=recognizer_release,
                    model_bundle=model_bundle,
                    profile=calibration_profile,
                )

                def recheck_materials() -> None:
                    rechecked_offline = load_verified_raster_offline_bundle(root)
                    rechecked_sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
                        document_preview_root=root,
                        manifest_sha256=sidecar_bundle.manifest_sha256,
                    )
                    rechecked_release = load_verified_public_finetuned_recognizer_release_v1(
                        root, _V19_PUBLIC_FINETUNED_RECOGNIZER_RELEASE_SHA256
                    )
                    rechecked_model = load_verified_raster_text_public_finetuned_paddle_bundle_v1(
                        root, _V19_PUBLIC_FINETUNED_MODEL_BUNDLE_SHA256
                    )
                    _verify_v19_public_finetune_contract(
                        root=root,
                        recognizer_release=rechecked_release,
                        model_bundle=rechecked_model,
                    )
                    if (
                        rechecked_offline.manifest_sha256 != offline_bundle.manifest_sha256
                        or rechecked_offline.bundle_root_relative_path
                        != offline_bundle.bundle_root_relative_path
                        or rechecked_offline.manifest.schema_version != "2.0"
                        or rechecked_sidecar.manifest_sha256 != sidecar_bundle.manifest_sha256
                        or rechecked_sidecar.bundle_root_relative_path
                        != sidecar_bundle.bundle_root_relative_path
                        or rechecked_sidecar.manifest.schema_version != "2.0"
                        or rechecked_sidecar.manifest.source_bundle_manifest_sha256
                        != offline_bundle.manifest_sha256
                        or (
                            rechecked_sidecar.manifest.runtime_operating_system,
                            rechecked_sidecar.manifest.runtime_architecture,
                            rechecked_sidecar.manifest.runtime_python_abi,
                        )
                        != ("linux", "amd64", "cp311")
                        or rechecked_release.release_sha256
                        != _V19_PUBLIC_FINETUNED_RECOGNIZER_RELEASE_SHA256
                        or rechecked_model.manifest_sha256
                        != _V19_PUBLIC_FINETUNED_MODEL_BUNDLE_SHA256
                    ):
                        raise SyntheticRasterTextActivationError("raster_text_linux_bundle_drift")

                return _run_v2_staged_candidate_activation(
                    root=root,
                    build_inputs=build_inputs,
                    existing_dataset=existing_dataset,
                    dataset_materials_preverified=True,
                    calibration_profile_info=calibration_profile_info,
                    prepare_final_profile=prepare_profile,
                    expected_profile_type=RasterTextOfflineProfileV19,
                    expected_pointer_type=RasterTextActivationPointerV12,
                    write_calibration=(write_raster_semantic_text_fresh_calibration_receipt_v7),
                    write_sealed=write_raster_semantic_text_fresh_sealed_evaluation_v9,
                    recheck_materials=recheck_materials,
                    drift_code="raster_text_linux_bundle_drift",
                    quality_gate_revision=_V19_PREVIEW97_QUALITY_GATE_REVISION,
                )
            except SyntheticRasterTextActivationError:
                raise
            except (
                RasterOfflineBundleError,
                RasterTextDirectPaddleBundleError,
                RasterTextPublicFineTunedPaddleBundleError,
                SourceAttestedPublicRecognizerFineTuneError,
            ) as error:
                if error.code in {
                    "public_finetune_bundle_unavailable",
                    "public_finetuned_paddle_bundle_source_unavailable",
                }:
                    raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
                raise SyntheticRasterTextActivationError(error.code) from None
            except (
                SyntheticRasterCompilerUnavailable,
                OSError,
                ValidationError,
                ValueError,
                RuntimeError,
            ):
                raise SyntheticRasterTextActivationError(
                    "raster_text_linux_auto_activation_failed"
                ) from None


def auto_activate_synthetic_raster_text_v7(
    *, document_preview_root: Path
) -> SyntheticRasterTextAutoActivationResultV1:
    """Build, fine-tune, evaluate, and activate only the pinned V19b lane.

    The command has no model, epoch, score, hash, or approval input.  It
    creates or reuses the maximum-four-epoch head-only source-Gold fine-tune,
    proves zero overlap with the result-blind V2 holdout, then enters the
    shared 24/240/staged-sealed transaction.  The final V19b profile must also
    pass the independently generated 800-asset B/L repair audit before sealed
    evaluation continues.  Every failure occurs before the generic pointer
    writer's final atomic replace.
    """

    with _deny_network():
        try:
            sidecar_bundle = materialize_verified_raster_text_direct_paddle_bundle_v1(
                document_preview_root=document_preview_root
            )
            mobile_bundle = materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=sidecar_bundle.document_preview_root
            )
        except SyntheticRasterTextActivationError:
            raise
        except (
            RasterTextDirectPaddleBundleError,
            RasterTextMobilePaddleBundleError,
        ) as error:
            if error.code in {
                "direct_paddle_bundle_path_invalid",
                "mobile_paddle_bundle_path_invalid",
            }:
                raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
            raise SyntheticRasterTextActivationError(error.code) from None

        with _lock(sidecar_bundle.document_preview_root):
            try:
                root = sidecar_bundle.document_preview_root
                build_inputs = ensure_fixed_synthetic_raster_build_inputs(
                    document_preview_root=root
                )
                existing_dataset = _existing_synthetic_raster_dataset(root)
                # V19 training must never run against a dataset that was
                # accepted merely because its directory exists.  The
                # compiler is create-or-verify, so this builds a missing
                # dataset or fully replays every immutable receipt/material
                # check before selection and the expensive child process.
                compile_synthetic_raster_dataset(
                    document_preview_root=root,
                    build_inputs_relative_path=(build_inputs.build_inputs_relative_path),
                )
                # Training consumes only the 500-family remainder, so the
                # immutable V2 holdout must exist and verify before any
                # optimizer process can start.
                write_fresh_raster_text_scope_selection_receipt(
                    document_preview_root=root,
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
                )
                # Dataset compilation imports Pillow/NumPy in this parent.
                # Training must therefore enter a fresh interpreter so the
                # pinned native package provenance check sees no preloaded
                # modules.  The child accepts only this root and fixed V2
                # stage; its small handshake is reopened below.
                child_release = _run_fixed_v19_fine_tune_subprocess(root)
                recognizer_release = _verify_v19_public_finetune_contract(
                    root=root,
                    recognizer_release=child_release,
                )
                model_bundle = materialize_verified_raster_text_public_finetuned_paddle_bundle_v1(
                    mobile_bundle,
                    recognizer_release,
                )
                recognizer_release = _verify_v19_public_finetune_contract(
                    root=root,
                    recognizer_release=recognizer_release,
                    model_bundle=model_bundle,
                )

                def prepare_profile(
                    minimum_page_confidence: float,
                ) -> _PreparedRasterTextRuntimeProfile:
                    return prepare_raster_text_public_finetuned_runtime_profile_v19(
                        sidecar_bundle,
                        model_bundle,
                        pdfium_version=(build_inputs.renderer_configuration.pdfium_version),
                        minimum_page_confidence=minimum_page_confidence,
                        cpu_threads=_AUTO_ACTIVATION_V15_CPU_THREADS,
                    )

                calibration_profile_info = prepare_profile(0.0)
                calibration_profile_raw = _read(
                    root,
                    calibration_profile_info.runtime_configuration_relative_path,
                    limit=_MAX_RECEIPT_BYTES,
                    code="raster_text_profile_invalid",
                )
                calibration_profile = parse_raster_text_offline_profile(calibration_profile_raw)
                if not isinstance(calibration_profile, RasterTextOfflineProfileV19):
                    raise SyntheticRasterTextActivationError("raster_text_profile_invalid")
                _verify_v19_public_finetune_contract(
                    root=root,
                    recognizer_release=recognizer_release,
                    model_bundle=model_bundle,
                    profile=calibration_profile,
                )

                def recheck_materials() -> None:
                    rechecked_sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
                        document_preview_root=root,
                        manifest_sha256=sidecar_bundle.manifest_sha256,
                    )
                    rechecked_mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                        document_preview_root=root,
                        manifest_sha256=mobile_bundle.manifest_sha256,
                    )
                    rechecked_release = _verify_v19_public_finetune_contract(
                        root=root,
                        recognizer_release=recognizer_release,
                    )
                    rechecked_model = load_verified_raster_text_public_finetuned_paddle_bundle_v1(
                        root,
                        model_bundle.manifest_sha256,
                    )
                    _verify_v19_public_finetune_contract(
                        root=root,
                        recognizer_release=rechecked_release,
                        model_bundle=rechecked_model,
                    )
                    if (
                        rechecked_sidecar.manifest_sha256 != sidecar_bundle.manifest_sha256
                        or rechecked_sidecar.bundle_root_relative_path
                        != sidecar_bundle.bundle_root_relative_path
                        or rechecked_mobile.manifest_sha256 != mobile_bundle.manifest_sha256
                        or rechecked_mobile.bundle_root_relative_path
                        != mobile_bundle.bundle_root_relative_path
                        or rechecked_release.release_sha256 != recognizer_release.release_sha256
                        or rechecked_model.manifest_sha256 != model_bundle.manifest_sha256
                        or rechecked_model.bundle_root_relative_path
                        != model_bundle.bundle_root_relative_path
                        or rechecked_model.manifest.source_mobile_bundle_manifest_sha256
                        != mobile_bundle.manifest_sha256
                        or rechecked_model.manifest.source_recognizer_release_sha256
                        != recognizer_release.release_sha256
                    ):
                        raise SyntheticRasterTextActivationError(
                            "public_finetuned_paddle_bundle_drift"
                        )

                def run_repair_audit(
                    final_profile_info: _PreparedRasterTextRuntimeProfile,
                    final_profile: RasterTextOfflineProfileV18 | RasterTextOfflineProfileV19,
                ) -> BlRepairAuditCandidateRunV1:
                    if (
                        not isinstance(final_profile, RasterTextOfflineProfileV19)
                        or final_profile.engine_kind != _V19B_ENGINE_KIND
                    ):
                        raise SyntheticRasterTextActivationError(
                            "bl_repair_audit_candidate_runtime_unavailable"
                        )
                    return run_source_attested_bl_repair_audit_candidates(
                        document_preview_root=root,
                        runtime_configuration_relative_path=(
                            final_profile_info.runtime_configuration_relative_path
                        ),
                    )

                return _run_v2_staged_candidate_activation(
                    root=root,
                    build_inputs=build_inputs,
                    existing_dataset=existing_dataset,
                    dataset_materials_preverified=True,
                    calibration_profile_info=calibration_profile_info,
                    prepare_final_profile=prepare_profile,
                    expected_profile_type=RasterTextOfflineProfileV19,
                    expected_pointer_type=RasterTextActivationPointerV8,
                    write_calibration=(write_raster_semantic_text_fresh_calibration_receipt_v7),
                    write_sealed=(write_raster_semantic_text_fresh_sealed_evaluation_v7),
                    recheck_materials=recheck_materials,
                    drift_code="public_finetuned_paddle_bundle_drift",
                    pre_sealed_activation_check=run_repair_audit,
                )
            except SyntheticRasterTextActivationError:
                raise
            except (
                RasterTextDirectPaddleBundleError,
                RasterTextMobilePaddleBundleError,
                RasterTextPublicFineTunedPaddleBundleError,
                SourceAttestedPublicRecognizerFineTuneError,
            ) as error:
                if error.code in {
                    "public_finetune_bundle_unavailable",
                    "public_finetuned_paddle_bundle_source_unavailable",
                }:
                    raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
                raise SyntheticRasterTextActivationError(error.code) from None
            except (
                SyntheticRasterCompilerUnavailable,
                OSError,
                ValidationError,
                ValueError,
                RuntimeError,
            ):
                raise SyntheticRasterTextActivationError(
                    "raster_text_v7_auto_activation_failed"
                ) from None


def auto_activate_synthetic_raster_text_v8(
    *, document_preview_root: Path
) -> SyntheticRasterTextAutoActivationResultV1:
    """Build, replay-train, evaluate, and activate only exact V20/V9.

    This remains a no-knob transaction.  A fresh isolated child creates or
    reuses the fixed two-epoch Voyage replay head, activation reopens its V3
    receipt, materializes and rehashes the detector/recognizer bundle, then
    reuses the V7 24/240/staged-sealed evidence path.  The independent
    800-asset B/L audit is the final pre-sealed Gate.  The pointer writer is
    reached only after every quality, latency (p95 <= 10 seconds), runtime
    (30-second request ceiling), provenance and drift check succeeds.
    """

    with _deny_network():
        try:
            sidecar_bundle = materialize_verified_raster_text_direct_paddle_bundle_v1(
                document_preview_root=document_preview_root
            )
            mobile_bundle = materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=sidecar_bundle.document_preview_root
            )
        except SyntheticRasterTextActivationError:
            raise
        except (
            RasterTextDirectPaddleBundleError,
            RasterTextMobilePaddleBundleError,
        ) as error:
            if error.code in {
                "direct_paddle_bundle_path_invalid",
                "mobile_paddle_bundle_path_invalid",
            }:
                raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
            raise SyntheticRasterTextActivationError(error.code) from None

        with _lock(sidecar_bundle.document_preview_root):
            try:
                root = sidecar_bundle.document_preview_root
                build_inputs = ensure_fixed_synthetic_raster_build_inputs(
                    document_preview_root=root
                )
                existing_dataset = _existing_synthetic_raster_dataset(root)
                # Fully build/replay the immutable 24,000-asset receipt before
                # the optimizer can observe any source-derived crop.
                compile_synthetic_raster_dataset(
                    document_preview_root=root,
                    build_inputs_relative_path=build_inputs.build_inputs_relative_path,
                )
                write_fresh_raster_text_scope_selection_receipt(
                    document_preview_root=root,
                    synthetic_raster_receipt_relative_path=(SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH),
                    selection_revision=_FRESH_HOLDOUT_SELECTION_REVISION_V2,
                )
                child_release = _run_fixed_voyage_replay_subprocess(root)
                recognizer_release = _verify_v20_voyage_replay_contract(
                    root=root,
                    recognizer_release=child_release,
                )
                model_bundle = (
                    materialize_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
                        mobile_bundle,
                        recognizer_release,
                    )
                )
                recognizer_release = _verify_v20_voyage_replay_contract(
                    root=root,
                    recognizer_release=recognizer_release,
                    model_bundle=model_bundle,
                )

                def prepare_profile(
                    minimum_page_confidence: float,
                ) -> _PreparedRasterTextRuntimeProfile:
                    return prepare_raster_text_public_voyage_replay_runtime_profile_v20(
                        sidecar_bundle,
                        model_bundle,
                        pdfium_version=(build_inputs.renderer_configuration.pdfium_version),
                        minimum_page_confidence=minimum_page_confidence,
                        cpu_threads=_AUTO_ACTIVATION_V15_CPU_THREADS,
                    )

                calibration_profile_info = prepare_profile(0.0)
                calibration_profile_raw = _read(
                    root,
                    calibration_profile_info.runtime_configuration_relative_path,
                    limit=_MAX_RECEIPT_BYTES,
                    code="raster_text_profile_invalid",
                )
                calibration_profile = parse_raster_text_offline_profile(calibration_profile_raw)
                if type(calibration_profile) is not RasterTextOfflineProfileV20:
                    raise SyntheticRasterTextActivationError("raster_text_profile_invalid")
                _verify_v20_voyage_replay_contract(
                    root=root,
                    recognizer_release=recognizer_release,
                    model_bundle=model_bundle,
                    profile=calibration_profile,
                )

                def recheck_materials() -> None:
                    rechecked_sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
                        document_preview_root=root,
                        manifest_sha256=sidecar_bundle.manifest_sha256,
                    )
                    rechecked_mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                        document_preview_root=root,
                        manifest_sha256=mobile_bundle.manifest_sha256,
                    )
                    rechecked_release = _verify_v20_voyage_replay_contract(
                        root=root,
                        recognizer_release=recognizer_release,
                    )
                    rechecked_model = (
                        load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
                            root,
                            model_bundle.manifest_sha256,
                        )
                    )
                    _verify_v20_voyage_replay_contract(
                        root=root,
                        recognizer_release=rechecked_release,
                        model_bundle=rechecked_model,
                    )
                    if (
                        rechecked_sidecar.manifest_sha256 != sidecar_bundle.manifest_sha256
                        or rechecked_sidecar.bundle_root_relative_path
                        != sidecar_bundle.bundle_root_relative_path
                        or rechecked_mobile.manifest_sha256 != mobile_bundle.manifest_sha256
                        or rechecked_mobile.bundle_root_relative_path
                        != mobile_bundle.bundle_root_relative_path
                        or rechecked_release.release_sha256 != recognizer_release.release_sha256
                        or rechecked_model.manifest_sha256 != model_bundle.manifest_sha256
                        or rechecked_model.bundle_root_relative_path
                        != model_bundle.bundle_root_relative_path
                        or rechecked_model.manifest.source_mobile_bundle_manifest_sha256
                        != mobile_bundle.manifest_sha256
                        or rechecked_model.manifest.source_recognizer_release_sha256
                        != recognizer_release.release_sha256
                    ):
                        raise SyntheticRasterTextActivationError(
                            "public_voyage_replay_paddle_bundle_drift"
                        )

                def run_repair_audit(
                    final_profile_info: _PreparedRasterTextRuntimeProfile,
                    final_profile: (
                        RasterTextOfflineProfileV18
                        | RasterTextOfflineProfileV19
                        | RasterTextOfflineProfileV20
                    ),
                ) -> BlRepairAuditCandidateRunV1:
                    if type(
                        final_profile
                    ) is not RasterTextOfflineProfileV20 or not _is_exact_bl_repair_audit_profile(
                        final_profile
                    ):
                        raise SyntheticRasterTextActivationError(
                            "bl_repair_audit_candidate_runtime_unavailable"
                        )
                    return run_source_attested_bl_repair_audit_candidates(
                        document_preview_root=root,
                        runtime_configuration_relative_path=(
                            final_profile_info.runtime_configuration_relative_path
                        ),
                    )

                return _run_v2_staged_candidate_activation(
                    root=root,
                    build_inputs=build_inputs,
                    existing_dataset=existing_dataset,
                    dataset_materials_preverified=True,
                    calibration_profile_info=calibration_profile_info,
                    prepare_final_profile=prepare_profile,
                    expected_profile_type=RasterTextOfflineProfileV20,
                    expected_pointer_type=RasterTextActivationPointerV9,
                    write_calibration=(write_raster_semantic_text_fresh_calibration_receipt_v7),
                    write_sealed=(write_raster_semantic_text_fresh_sealed_evaluation_v7),
                    recheck_materials=recheck_materials,
                    drift_code="public_voyage_replay_paddle_bundle_drift",
                    pre_sealed_activation_check=run_repair_audit,
                )
            except SyntheticRasterTextActivationError:
                raise
            except (
                RasterTextDirectPaddleBundleError,
                RasterTextMobilePaddleBundleError,
                RasterTextPublicVoyageReplayPaddleBundleError,
                VoyageReplayError,
            ) as error:
                if error.code in {
                    "voyage_replay_bundle_unavailable",
                    "public_voyage_replay_paddle_bundle_source_unavailable",
                }:
                    raise SyntheticRasterTextActivationError("offline_bundle_unavailable") from None
                raise SyntheticRasterTextActivationError(error.code) from None
            except (
                SyntheticRasterCompilerUnavailable,
                OSError,
                ValidationError,
                ValueError,
                RuntimeError,
            ):
                raise SyntheticRasterTextActivationError(
                    "raster_text_v8_auto_activation_failed"
                ) from None


__all__ = [
    "BlRepairAuditCandidateRunV1",
    "RasterTextCalibrationReceiptV2",
    "RasterTextCalibrationReceiptV3",
    "RasterTextCalibrationReceiptV4",
    "RasterTextCalibrationReceiptV5",
    "RasterTextCalibrationReceiptV6",
    "RasterTextCalibrationReceiptV7",
    "RasterTextCandidateRunV1",
    "RasterTextFreshCandidateBundleV1",
    "RasterTextFreshCandidateExecutionV1",
    "RasterTextFreshEvaluationProjectionReceiptV1",
    "RasterTextFreshProjectedCandidateBundleV1",
    "RasterTextFreshProjectedCandidateExecutionV1",
    "RasterTextFreshSealedStageDecisionReceiptV1",
    "RasterTextFreshSealedStageDecisionReceiptV2",
    "RasterTextFreshSealedStageStratumV1",
    "RasterTextFreshScopeSelectionReceiptV1",
    "RasterTextFreshScopeSelectionReceiptV2",
    "RasterTextFreshSourceGoldReceiptV1",
    "RasterTextSealedEvaluationV2",
    "RasterTextSealedEvaluationV3",
    "RasterTextSealedEvaluationV4",
    "RasterTextSealedEvaluationV5",
    "RasterTextSealedEvaluationV6",
    "RasterTextSealedEvaluationV7",
    "RasterTextSealedEvaluationV8",
    "RasterTextSealedEvaluationV9",
    "RasterTextSealedProfileScoreV2",
    "RasterTextSealedProfileScoreV3",
    "RasterTextSealedProfileScoreV4",
    "RasterTextSealedProfileScoreV5",
    "RasterTextSealedProfileScoreV6",
    "RasterTextSealedProfileScoreV7",
    "RasterTextSealedProfileScoreV8",
    "RasterTextSmokeQualityScreenV1",
    "RasterTextSmokeQualityScreenV2",
    "RasterTextSmokeQualityScreenV3",
    "RasterTextSmokeQualityScreenV4",
    "RasterTextSourceGoldReceiptV1",
    "SyntheticRasterTextActivationError",
    "SyntheticRasterTextAutoActivationResultV1",
    "auto_activate_synthetic_raster_text_v1",
    "auto_activate_synthetic_raster_text_v6",
    "auto_activate_synthetic_raster_text_v7",
    "auto_activate_synthetic_raster_text_v8",
    "auto_activate_synthetic_raster_text_linux_preview97_v1",
    "auto_activate_synthetic_raster_text_preview97_v1",
    "run_warm_service_raster_text_smoke",
    "run_paddleocr3_synthetic_raster_text_candidates",
    "run_source_attested_bl_repair_audit_candidates",
    "verify_source_attested_fresh_raster_text_evidence",
    "verify_source_attested_raster_text_evidence",
    "verify_fresh_raster_text_sealed_stage1_decision",
    "verify_fresh_raster_text_sealed_stage1_decision_v6",
    "verify_source_attested_fresh_raster_text_evidence_v6",
    "verify_source_attested_fresh_raster_text_evidence_v8",
    "write_raster_text_calibration_receipt",
    "write_raster_text_sealed_evaluation",
    "write_raster_text_smoke_quality_screen",
    "write_raster_semantic_text_calibration_receipt_v2",
    "write_raster_semantic_text_calibration_receipt_v3",
    "write_raster_semantic_text_fresh_calibration_receipt_v4",
    "write_raster_semantic_text_fresh_calibration_receipt_v5",
    "write_raster_semantic_text_fresh_calibration_receipt_v6",
    "write_raster_semantic_text_fresh_calibration_receipt_v7",
    "write_raster_semantic_text_fresh_sealed_evaluation_v4",
    "write_raster_semantic_text_fresh_sealed_evaluation_v5",
    "write_raster_semantic_text_fresh_sealed_evaluation_v6",
    "write_raster_semantic_text_fresh_sealed_evaluation_v7",
    "write_raster_semantic_text_fresh_sealed_evaluation_v8",
    "write_raster_semantic_text_fresh_sealed_evaluation_v9",
    "write_raster_semantic_text_sealed_evaluation_v2",
    "write_raster_semantic_text_sealed_evaluation_v3",
    "write_raster_semantic_text_smoke_quality_screen_v2",
    "write_raster_semantic_text_smoke_quality_screen_v3",
    "write_raster_semantic_text_smoke_quality_screen_preview97_v1",
    "write_fresh_raster_text_scope_selection_receipt",
    "write_fresh_raster_text_sealed_stage1_decision",
    "write_source_attested_fresh_raster_text_gold_receipt",
    "write_source_attested_raster_text_gold_receipt",
]
