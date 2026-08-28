"""Local-only PDFium and PP-StructureV3 runtime seams for raster OCR v2.

Nothing in this module downloads a package/model or silently selects a
fallback.  It provides the concrete execution path that a future, sealed
``raster-preview-active.json`` approval can wire in; absent sidecars or native
dependencies resolve to the existing content-free unavailable result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
import os
import re
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Iterator,
    Literal,
    Protocol,
    Sequence,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from customs_ai_agent.schemas.raster_document_extraction import (
    RasterExtractionWarningV2,
    RasterPreviewCandidateV2,
    RasterPreviewInputV2,
)
from customs_ai_agent.service.document_input import DocumentStructure
from customs_ai_agent.service.document_preview_runtime import local_files_only
from customs_ai_agent.service.ppstructurev3_sidecar import (
    __file__ as _PPSTRUCTUREV3_SIDECAR_MODULE_FILE,
    PpStructureV3SidecarConfigurationV1,
    PpStructureV3SidecarError,
    PpStructureV3SidecarResultV1,
    PpStructureV3StructureProbeResultV1,
)
from customs_ai_agent.service.raster_document_extraction import (
    DisabledRasterDocumentPreviewEngine,
    LocalRasterDocumentPreviewEngine,
    RasterDocumentPreviewInputError,
    RasterDocumentPreviewUnsupported,
    RasterDocumentPreviewUnavailable,
    RasterInputAdapter,
    RasterLedgerCellV2,
    RasterLedgerFieldV2,
    RasterLedgerSpanV2,
    RasterLedgerTableV2,
    RasterOcrLayoutLedgerV2,
    RasterOcrLayoutProvider,
    RasterizedPageV2,
)

if TYPE_CHECKING:
    from customs_ai_agent.evaluation.synthetic_raster import SyntheticRasterReceiptV1


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_RASTER_PROFILE_REVISION_V2 = re.compile(r"^synthetic-raster-v1:[a-f0-9]{24}$")
_MAX_SIDECAR_BYTES = 1024 * 1024
# The exhaustive offline-bundle manifest includes the pinned sidecar
# interpreter's complete package tree, so it is intentionally allowed to be
# larger than a small sidecar configuration or runtime profile.  Keep this
# separate from ``_MAX_SIDECAR_BYTES``: raising the latter would unnecessarily
# broaden every untrusted sidecar/configuration input boundary.
_MAX_OFFLINE_BUNDLE_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_RASTER_RECEIPT_BYTES = 32 * 1024 * 1024
_RASTER_OFFLINE_BUNDLE_DIRECTORY = "raster-offline-bundle-v1"
_RASTER_OFFLINE_BUNDLE_RELEASE_COLLECTION_DIRECTORY = "raster-offline-bundles-v1"
_PPSTRUCTUREV3_BATCH_PROTOCOL = "customs-raster-ppstructurev3-batch-v1"
_PPSTRUCTUREV3_EVALUATION_BATCH_PROTOCOL = "customs-raster-ppstructurev3-evaluation-batch-v2"
_PPSTRUCTUREV3_SERVICE_PROTOCOL = "customs-raster-ppstructurev3-service-v1"
_MAX_PPSTRUCTUREV3_BATCH_REQUEST_BYTES = 128 * 1024 * 1024
_MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS = 4
_MAX_PPSTRUCTUREV3_EVALUATION_BATCH_RAW_BYTES = 64 * 1024 * 1024
_MAX_PPSTRUCTUREV3_EVALUATION_BATCH_RESPONSE_BYTES = (4 * _MAX_SIDECAR_BYTES) + (16 * 1024)
# A customer request may never inherit the old 420-second evaluator bound.
# Cold start is handled while a profile is adopted; this is the hard bound for
# one already-warm sidecar request and leaves a small margin for the Agent's
# 60-second HTTP boundary.
_MAX_PPSTRUCTUREV3_SERVICE_REQUEST_SECONDS = 55
# This protocol deliberately carries only bounded counters, never OCR text,
# geometry, source hashes, native result keys, or image bytes.  Keep its
# parent-side limit independent of ordinary OCR responses.
_MAX_PPSTRUCTUREV3_STRUCTURE_PROBE_BYTES = 32 * 1024

# The v2 child receives an exact interpreter and import root from the verified
# bundle. ``-I`` ignores PYTHONPATH, user site directories, and Python-related
# environment variables; ``-B`` prevents the child from creating an untracked
# ``__pycache__`` entry in that root. This bootstrap then removes interpreter
# site-packages as well and admits only the bundle root plus standard-library
# locations. Every non-stdlib import used by the sidecar must therefore live
# in the hash-bound module root.
_ISOLATED_SIDECAR_BOOTSTRAP = (
    "import os,runpy,sys,sysconfig\n"
    "interpreter_argument=os.path.abspath(sys.argv[1])\n"
    "module_argument=os.path.abspath(sys.argv[2])\n"
    "command_argument=os.path.abspath(sys.argv[3])\n"
    "interpreter_root=os.path.realpath(interpreter_argument)\n"
    "module_root=os.path.realpath(module_argument)\n"
    "command_path=os.path.realpath(command_argument)\n"
    "def inside(root,path):\n"
    " return os.path.commonpath((os.path.normcase(root),os.path.normcase(path))) == os.path.normcase(root)\n"
    "if (not os.path.isdir(interpreter_root) or os.path.islink(interpreter_root) or "
    "not inside(interpreter_root,os.path.realpath(sys.executable))): raise SystemExit(70)\n"
    "root_case=os.path.normcase(module_root)\n"
    "command_case=os.path.normcase(command_path)\n"
    "if (not os.path.isdir(module_root) or os.path.islink(module_root) or "
    "os.path.islink(command_path) or os.path.commonpath((root_case,command_case)) != root_case): "
    "raise SystemExit(70)\n"
    "stdlib_paths=[]\n"
    "for key in ('stdlib','platstdlib'):\n"
    " path=sysconfig.get_path(key)\n"
    " if path and path not in stdlib_paths: stdlib_paths.append(path)\n"
    "for path in (os.path.join(sys.base_prefix,'DLLs'), "
    "os.path.join(stdlib_paths[0],'lib-dynload') if stdlib_paths else ''):\n"
    " if path and os.path.isdir(path) and path not in stdlib_paths: stdlib_paths.append(path)\n"
    "if any(not inside(interpreter_root,os.path.realpath(path)) for path in stdlib_paths): "
    "raise SystemExit(70)\n"
    # ``realpath`` expands DOS 8.3 aliases on Windows.  Retain the original
    # validated module/command spellings for imports/runpy so deeply nested
    # Paddle packages beneath a Korean-named project path stay below the
    # legacy Win32 path limit.
    "sys.path[:]=[module_argument,*stdlib_paths]\n"
    "sys.argv=[command_argument,*sys.argv[4:]]\n"
    "runpy.run_path(command_argument,run_name='__main__')\n"
)


class RasterRuntimeError(RuntimeError):
    """Content-free runtime-sidecar failure."""


class _RasterWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _strict_profile_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
        or "\\" in value
        or ":" in value
    ):
        raise ValueError("raster runtime path is invalid")
    return value


def _is_explicit_raster_bundle_root(value: str) -> bool:
    """Return whether a profile names one allowed immutable bundle root."""

    try:
        path = PurePosixPath(value)
        if _strict_profile_relative_path(value) != value:
            return False
    except ValueError:
        return False
    if value == _RASTER_OFFLINE_BUNDLE_DIRECTORY:
        return True
    return (
        len(path.parts) == 3
        and path.parts[0] == _RASTER_OFFLINE_BUNDLE_RELEASE_COLLECTION_DIRECTORY
        and path.parts[1] == "releases"
        and _SHA256.fullmatch(path.parts[2]) is not None
    )


def _is_canonical_raster_bundle_release_binding(
    *, bundle_root_relative_path: str, bundle_manifest_sha256: str
) -> bool:
    """Return whether a new binding names its own content-addressed release.

    A release directory is not a generic location hint.  Its final component
    is the full manifest digest that identifies the exact immutable bundle.
    Old v2 profiles/pointers did not carry either binding field and remain
    readable as the fixed legacy-root form; once the pair is present, however,
    accepting the legacy path or a mismatched directory would make the
    release-selector boundary ambiguous.
    """

    if _SHA256.fullmatch(bundle_manifest_sha256) is None:
        return False
    return bundle_root_relative_path == (
        f"{_RASTER_OFFLINE_BUNDLE_RELEASE_COLLECTION_DIRECTORY}/releases/{bundle_manifest_sha256}"
    )


class PpStructureV3OfflineProfileV1(_RasterWireModel):
    """Pinned local profile; it contains no URL, credentials, or fallback."""

    schema_version: Literal["1.0"] = "1.0"
    engine_kind: Literal["ppstructurev3"] = "ppstructurev3"
    runtime_api: Literal["paddleocr3-ppstructurev3"] = "paddleocr3-ppstructurev3"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_profile_revision: str = Field(min_length=1, max_length=128)
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    required_packages: list[str] = Field(min_length=3, max_length=128)
    local_model_root: str = Field(min_length=1, max_length=240)
    local_sidecar_command: str = Field(min_length=1, max_length=240)
    sidecar_command_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_configuration_path: str = Field(min_length=1, max_length=240)
    sidecar_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_timeout_seconds: int = Field(default=420, ge=1, le=420)

    def model_root_relative_path(self) -> PurePosixPath:
        path = PurePosixPath(self.local_model_root)
        if path.is_absolute() or ".." in path.parts or any("//" in part for part in path.parts):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        if ":" in self.local_model_root or "\\" in self.local_model_root:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        return path

    def sidecar_relative_path(self, value: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or ":" in value or "\\" in value:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        return path


def make_synthetic_raster_profile_revision(fingerprint_sha256: str) -> str:
    """Return the opaque v2 revision bound to one immutable release fingerprint.

    The caller supplies a SHA-256 of its immutable release material (normally
    the offline-bundle manifest).  The profile stores only a short opaque
    derivative; all complete hashes remain in the local pointer/evidence
    chain and are never exposed by the status endpoint.
    """

    if _SHA256.fullmatch(fingerprint_sha256) is None:
        raise RasterRuntimeError("raster_profile_revision_invalid")
    return f"synthetic-raster-v1:{fingerprint_sha256[:24]}"


class PpStructureV3OfflineProfileV2(_RasterWireModel):
    """Split parent-renderer and child-sidecar inventories for automatic v2.

    The regular Agent interpreter verifies only PDFium/Pillow renderer
    packages.  PaddleOCR/PaddleX/PaddlePaddle remain entirely in the pinned
    sidecar interpreter, which verifies its own inventory before importing
    any native package.  This avoids an accidental heavy OCR dependency in
    the normal service environment.
    """

    schema_version: Literal["2.0"] = "2.0"
    engine_kind: Literal["ppstructurev3"] = "ppstructurev3"
    runtime_api: Literal["paddleocr3-ppstructurev3"] = "paddleocr3-ppstructurev3"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_profile_revision: str = Field(pattern=r"^synthetic-raster-v1:[a-f0-9]{24}$")
    # These two fields were added after the initial fixed-root v2 profile.
    # They are optional solely so an already-written/approved fixed-root
    # pointer remains verifiable.  Newly generated profiles always carry
    # both, which binds a release selector's path and full manifest hash into
    # runtime evidence rather than relying on an implicit filename prefix.
    bundle_root_relative_path: str | None = Field(default=None, min_length=1, max_length=240)
    bundle_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_required_packages: list[str] = Field(min_length=2, max_length=32)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_required_packages: list[str] = Field(min_length=3, max_length=128)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_inventory_path: str = Field(min_length=1, max_length=240)
    local_model_root: str = Field(min_length=1, max_length=240)
    # Unlike the preserved v1/static profile, v2 never launches a generic
    # wrapper.  It invokes this exact bundle-owned Python interpreter with
    # isolated mode and a fully controlled import root.
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_command_path: str = Field(min_length=1, max_length=240)
    sidecar_command_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_configuration_path: str = Field(min_length=1, max_length=240)
    sidecar_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_timeout_seconds: int = Field(default=420, ge=1, le=420)
    # Old immutable v2 profiles did not have an evaluation list-input
    # protocol. Treat their absent field as exact batch-v1/sequential
    # behavior; newly generated batch-4 releases always bind this value into
    # the runtime profile and sidecar configuration hashes.
    evaluation_batch_size: Literal[1, 4] = 1

    @field_validator(
        "bundle_root_relative_path",
        "parent_renderer_package_inventory_path",
        "sidecar_package_inventory_path",
        "model_inventory_path",
        "local_model_root",
        "sidecar_interpreter_root",
        "sidecar_interpreter_path",
        "sidecar_module_root",
        "sidecar_command_path",
        "sidecar_module_path",
        "sidecar_configuration_path",
    )
    @classmethod
    def _paths_are_strictly_relative(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _strict_profile_relative_path(value)

    @model_validator(mode="after")
    def _split_inventories_are_unambiguous(self) -> "PpStructureV3OfflineProfileV2":
        if (
            set(self.parent_renderer_required_packages) != {"pypdfium2", "Pillow"}
            or len(self.parent_renderer_required_packages) != 2
            # Keep the sidecar boundary narrow and auditable.  Adding another
            # package would change the native execution surface without a new
            # runtime/profile contract, so it is a new version rather than a
            # permissive inventory entry.
            or set(self.sidecar_required_packages) != {"paddleocr", "paddlex", "paddlepaddle"}
            or len(self.sidecar_required_packages) != 3
            or self.parent_renderer_package_inventory_path == self.sidecar_package_inventory_path
        ):
            raise ValueError("raster v2 package inventories are invalid")
        if (self.bundle_root_relative_path is None) != (self.bundle_manifest_sha256 is None):
            raise ValueError("raster v2 bundle binding is invalid")
        # Profiles created before release selection have neither field.  They
        # remain locked to the original fixed root.  A profile with the new
        # fields must name exactly that release for every bundle-owned path.
        bundle_root = self.bundle_root_relative_path or _RASTER_OFFLINE_BUNDLE_DIRECTORY
        if self.bundle_root_relative_path is None:
            # Backward compatibility only: previously written v2 profiles had
            # neither release-binding field and therefore remain fixed to the
            # original bundle root.
            if not _is_explicit_raster_bundle_root(bundle_root):
                raise ValueError("raster v2 bundle binding is invalid")
        elif self.bundle_manifest_sha256 is None or not _is_canonical_raster_bundle_release_binding(
            bundle_root_relative_path=self.bundle_root_relative_path,
            bundle_manifest_sha256=self.bundle_manifest_sha256,
        ):
            raise ValueError("raster v2 bundle binding is invalid")
        module_root = PurePosixPath(self.sidecar_module_root)
        module_path = PurePosixPath(self.sidecar_module_path)
        command_path = PurePosixPath(self.sidecar_command_path)
        interpreter_root = PurePosixPath(self.sidecar_interpreter_root)
        interpreter_path = PurePosixPath(self.sidecar_interpreter_path)
        try:
            module_relative = module_path.relative_to(module_root)
            command_relative = command_path.relative_to(module_root)
            interpreter_relative = interpreter_path.relative_to(interpreter_root)
        except ValueError:
            raise ValueError("raster v2 sidecar module root is invalid") from None
        if (
            module_relative == PurePosixPath(".")
            or command_relative == PurePosixPath(".")
            or interpreter_relative == PurePosixPath(".")
            # The v2 command is not a separately trusted wrapper. It is
            # precisely the hash-bound source module that the isolated child
            # runs, preventing an unmodelled entry-point substitution.
            or self.sidecar_command_path != self.sidecar_module_path
            or self.sidecar_command_sha256 != self.sidecar_module_sha256
            or (
                self.bundle_root_relative_path is None
                and any(
                    PurePosixPath(path).parts[0] != _RASTER_OFFLINE_BUNDLE_DIRECTORY
                    for path in (
                        self.sidecar_interpreter_root,
                        self.sidecar_interpreter_path,
                        self.sidecar_module_root,
                        self.sidecar_command_path,
                        self.sidecar_module_path,
                    )
                )
            )
            or (
                self.bundle_root_relative_path is not None
                and any(
                    not (
                        PurePosixPath(path) != PurePosixPath(bundle_root)
                        and PurePosixPath(path).is_relative_to(PurePosixPath(bundle_root))
                    )
                    for path in (
                        self.parent_renderer_package_inventory_path,
                        self.sidecar_package_inventory_path,
                        self.model_inventory_path,
                        self.local_model_root,
                        self.sidecar_interpreter_root,
                        self.sidecar_interpreter_path,
                        self.sidecar_module_root,
                        self.sidecar_command_path,
                        self.sidecar_module_path,
                    )
                )
            )
        ):
            raise ValueError("raster v2 sidecar module root is invalid")
        return self

    def model_root_relative_path(self) -> PurePosixPath:
        path = PurePosixPath(self.local_model_root)
        if path.is_absolute() or ".." in path.parts or any("//" in part for part in path.parts):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        if ":" in self.local_model_root or "\\" in self.local_model_root:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        return path

    def sidecar_relative_path(self, value: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or ":" in value or "\\" in value:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        return path


@dataclass(frozen=True, slots=True)
class PpStructureV3EvaluationBatchInputV2:
    """One parent-bound native list-input item used only by synthetic eval."""

    item_id: str
    pages: tuple[RasterizedPageV2, ...]
    source_sha256: str
    raster_profile_revision: str

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER.fullmatch(self.item_id) is None
            or _SHA256.fullmatch(self.source_sha256) is None
            or not self.pages
        ):
            raise ValueError("raster evaluation batch item is invalid")


class _PpStructureV3EvaluationBatchResponseItemV2(_RasterWireModel):
    item_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER.pattern)
    result: PpStructureV3SidecarResultV1


class _PpStructureV3EvaluationBatchResponseV2(_RasterWireModel):
    protocol: Literal["customs-raster-ppstructurev3-evaluation-batch-v2"]
    results: list[_PpStructureV3EvaluationBatchResponseItemV2] = Field(
        min_length=1, max_length=_MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS
    )

    @model_validator(mode="after")
    def _result_identifiers_are_unique(self) -> "_PpStructureV3EvaluationBatchResponseV2":
        identifiers = [item.item_id for item in self.results]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("raster evaluation batch result identifiers are invalid")
        return self


class PaddleOcr3DirectRasterBaselineProfileV1(_RasterWireModel):
    """Evaluation-only direct P3 OCR baseline; never an approval profile."""

    schema_version: Literal["1.0"] = "1.0"
    engine_kind: Literal["paddleocr3_direct_baseline"] = "paddleocr3_direct_baseline"
    runtime_api: Literal["paddleocr3-paddlex-3.2"] = "paddleocr3-paddlex-3.2"
    approval_eligible: Literal[False] = False
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_profile_revision: str = Field(min_length=1, max_length=128)
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    required_packages: list[str] = Field(min_length=3, max_length=128)
    local_model_root: str = Field(min_length=1, max_length=240)
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"]
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"]
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    def local_relative_path(self, value: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or ":" in value or "\\" in value:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        return path


class RasterApprovalPointerV1(_RasterWireModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["approved"] = "approved"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    raster_profile_revision: str = Field(min_length=1, max_length=128)
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RasterSourceAttestedActivationPointerV2(_RasterWireModel):
    """Canonical non-human automatic activation pointer for raster OCR.

    This shape intentionally has no approval ID, signer, operator flag, Gold
    input, or external URL.  Its only accepted evidence source is the local
    source-attested synthetic raster receipt chain.
    """

    schema_version: Literal["2.0"] = "2.0"
    activation_kind: Literal["source_attested_synthetic_raster"] = (
        "source_attested_synthetic_raster"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    raster_profile_revision: str = Field(pattern=r"^synthetic-raster-v1:[a-f0-9]{24}$")
    # Optional for backwards verification of an already-written v2 pointer.
    # All pointers produced after release selection include this pair.
    bundle_root_relative_path: str | None = Field(default=None, min_length=1, max_length=240)
    bundle_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("bundle_root_relative_path")
    @classmethod
    def _bundle_root_is_relative(cls, value: str | None) -> str | None:
        if value is not None and not _is_explicit_raster_bundle_root(value):
            raise ValueError("raster activation bundle binding is invalid")
        return value

    @model_validator(mode="after")
    def _bundle_binding_is_paired(self) -> "RasterSourceAttestedActivationPointerV2":
        if (self.bundle_root_relative_path is None) != (self.bundle_manifest_sha256 is None):
            raise ValueError("raster activation bundle binding is invalid")
        if self.bundle_root_relative_path is not None and (
            self.bundle_manifest_sha256 is None
            or not _is_canonical_raster_bundle_release_binding(
                bundle_root_relative_path=self.bundle_root_relative_path,
                bundle_manifest_sha256=self.bundle_manifest_sha256,
            )
        ):
            raise ValueError("raster activation bundle binding is invalid")
        return self


class RasterCalibrationThresholdsV1(_RasterWireModel):
    """Frozen calibration decisions applied before a draft becomes ready."""

    classification_confidence: float = Field(ge=0, le=1)
    span_confidence: float = Field(ge=0, le=1)
    field_confidence: float = Field(ge=0, le=1)
    table_cell_confidence: float = Field(ge=0, le=1)


class RasterCalibrationThresholdReceiptV1(_RasterWireModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    split: Literal["calibration"] = "calibration"
    raster_profile_revision: str = Field(min_length=1, max_length=128)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    synthetic_raster_receipt_path: str = Field(min_length=1, max_length=240)
    synthetic_raster_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_relative_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_case_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    thresholds: RasterCalibrationThresholdsV1


class RasterCalibrationThresholdReceiptV2(_RasterWireModel):
    """v2 calibration receipt that binds parent and sidecar environments."""

    schema_version: Literal["2.0"] = "2.0"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    split: Literal["calibration"] = "calibration"
    raster_profile_revision: str = Field(pattern=r"^synthetic-raster-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    synthetic_raster_receipt_path: str = Field(min_length=1, max_length=240)
    synthetic_raster_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_relative_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_case_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    thresholds: RasterCalibrationThresholdsV1


class RasterSealedGateScoreV1(_RasterWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    field_exact_match: float = Field(ge=0.98, le=1)
    character_error_rate: float = Field(ge=0, le=0.02)
    table_cell_f1: float = Field(ge=0.95, le=1)
    span_localization_f1: float = Field(ge=0.90, le=1)
    # Python's typing.Literal cannot represent floats; equal bounds enforce
    # the same required 0.5 IoU threshold in the sealed artifact.
    bbox_iou_threshold: float = Field(default=0.5, ge=0.5, le=0.5)


class RasterSealedEvaluationResultV1(_RasterWireModel):
    """Sealed result accepted by the raster activation resolver only."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    split: Literal["sealed"] = "sealed"
    status: Literal["passed"] = "passed"
    raster_profile_revision: str = Field(min_length=1, max_length=128)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    synthetic_raster_receipt_path: str = Field(min_length=1, max_length=240)
    synthetic_raster_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_relative_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_case_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_case_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_threshold_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_threshold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    gate_scores: list[RasterSealedGateScoreV1] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _has_one_passing_score_for_every_document_profile(self) -> "RasterSealedEvaluationResultV1":
        pairs = {(score.document_type, score.profile_id) for score in self.gate_scores}
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if pairs != expected:
            raise ValueError("sealed raster evaluation must contain every document/profile gate")
        return self


class RasterSealedEvaluationResultV2(_RasterWireModel):
    """Passing v2 sealed evaluation bound to both isolated runtimes."""

    schema_version: Literal["2.0"] = "2.0"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    split: Literal["sealed"] = "sealed"
    status: Literal["passed"] = "passed"
    raster_profile_revision: str = Field(pattern=r"^synthetic-raster-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    synthetic_raster_receipt_path: str = Field(min_length=1, max_length=240)
    synthetic_raster_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_relative_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_case_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_case_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_threshold_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_threshold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    gate_scores: list[RasterSealedGateScoreV1] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _has_one_passing_score_for_every_document_profile(
        self,
    ) -> "RasterSealedEvaluationResultV2":
        pairs = {(score.document_type, score.profile_id) for score in self.gate_scores}
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if pairs != expected:
            raise ValueError("sealed raster evaluation must contain every document/profile gate")
        return self


class PpStructureV3NativeExecutor(Protocol):
    """Native sidecar seam; caller controls an already-local implementation."""

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
        profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2,
    ) -> RasterOcrLayoutLedgerV2: ...


def _sha256_regular_file(path: Path, *, maximum_size: int = 512 * 1024 * 1024) -> str:
    """Hash one non-link local sidecar without returning its contents."""

    try:
        if path.is_symlink() or not path.is_file():
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        if path.stat().st_size > maximum_size:
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
    except RasterRuntimeError:
        raise
    except OSError:
        raise RasterRuntimeError("raster_runtime_sidecar_invalid") from None


def _controlled_directory_tree_sha256(directory: Path) -> str:
    """Hash a complete bundle-owned directory used by the isolated v2 child.

    The profile stores a digest over paths, byte hashes, and sizes.  Rebuilding
    that digest from the live tree rejects an extra ``paddleocr`` package just
    as strictly as it rejects a changed declared module.  No symlink or
    special-file traversal is admitted because the child imports directly from
    this root.
    """

    try:
        root = directory.resolve(strict=True)
        if directory.is_symlink() or not root.is_dir():
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        # Keep the canonical path for the link/containment decision, but do
        # the expensive traversal through the existing DOS alias.  Calling
        # ``resolve`` above intentionally returns the long Korean/deep path;
        # using that path for ``rglob`` can exceed Win32's legacy limit once a
        # 64-character content-addressed release directory is added.
        operational_root = _windows_operational_sidecar_path(root)
        files: list[dict[str, object]] = []
        for candidate in sorted(operational_root.rglob("*"), key=lambda path: path.as_posix()):
            if candidate.is_symlink():
                raise RasterRuntimeError("raster_runtime_sidecar_invalid")
            if candidate.is_file():
                size = candidate.stat().st_size
                if size > 1024 * 1024 * 1024:
                    raise RasterRuntimeError("raster_runtime_sidecar_invalid")
                files.append(
                    {
                        "path": candidate.relative_to(operational_root).as_posix(),
                        "sha256": _sha256_regular_file(candidate, maximum_size=1024 * 1024 * 1024),
                        "byte_size": size,
                    }
                )
            elif not candidate.is_dir():
                raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        if not files:
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        return hashlib.sha256(
            json.dumps(
                {"files": files},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    except RasterRuntimeError:
        raise
    except (OSError, ValueError):
        raise RasterRuntimeError("raster_runtime_sidecar_invalid") from None


def _controlled_module_root_tree_sha256(module_root: Path) -> str:
    """Return the import-root digest while keeping the public test seam clear."""

    return _controlled_directory_tree_sha256(module_root)


def _validate_controlled_child_file(*, root_path: Path, file_path: Path) -> None:
    """Require one bound executable/source file to stay inside its root."""

    try:
        root = root_path.resolve(strict=True)
        file = file_path.resolve(strict=True)
        if (
            root_path.is_symlink()
            or file_path.is_symlink()
            or not root.is_dir()
            or not file.is_file()
        ):
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        file.relative_to(root)
    except RasterRuntimeError:
        raise
    except (OSError, ValueError):
        raise RasterRuntimeError("raster_runtime_sidecar_invalid") from None


def _sidecar_environment(
    *,
    configuration_path: Path,
    configuration_sha256: str,
    model_root: Path,
    package_inventory_path: Path,
    package_inventory_sha256: str,
    model_inventory_path: Path,
    model_inventory_sha256: str,
    required_packages: Sequence[str],
    sidecar_module_sha256: str,
) -> dict[str, str]:
    """Pass only hash-bound local inputs and offline switches to the child.

    Environment flags alone are not considered a network control.  The
    first-party child also patches its own socket functions before importing
    PaddleOCR.  These values bind that child to the same configuration,
    package inventory, and model root the parent activation resolver checked.
    """

    allowed = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "COMSPEC",
            "PATH",
            "PYTHONPATH",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "WINDIR",
        }
    }
    allowed.update(
        {
            "CUSTOMS_AGENT_LOCAL_FILES_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            "PADDLEOCR_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "TRANSFORMERS_OFFLINE": "1",
            "CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_PATH": str(configuration_path),
            "CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_SHA256": configuration_sha256,
            "CUSTOMS_AGENT_PPSTRUCTURE_MODEL_ROOT": str(model_root),
            "CUSTOMS_AGENT_PPSTRUCTURE_PACKAGE_INVENTORY_PATH": str(package_inventory_path),
            "CUSTOMS_AGENT_PPSTRUCTURE_PACKAGE_INVENTORY_SHA256": package_inventory_sha256,
            "CUSTOMS_AGENT_PPSTRUCTURE_MODEL_INVENTORY_PATH": str(model_inventory_path),
            "CUSTOMS_AGENT_PPSTRUCTURE_MODEL_INVENTORY_SHA256": model_inventory_sha256,
            "CUSTOMS_AGENT_PPSTRUCTURE_REQUIRED_PACKAGES": json.dumps(
                list(required_packages), ensure_ascii=True, separators=(",", ":")
            ),
            "CUSTOMS_AGENT_PPSTRUCTURE_SIDECAR_MODULE_SHA256": sidecar_module_sha256,
        }
    )
    return allowed


def _isolated_v2_sidecar_environment(
    *,
    interpreter_path: Path,
    module_root: Path,
    runtime_home: Path,
    configuration_path: Path,
    configuration_sha256: str,
    model_root: Path,
    package_inventory_path: Path,
    package_inventory_sha256: str,
    model_inventory_path: Path,
    model_inventory_sha256: str,
    required_packages: Sequence[str],
    sidecar_module_sha256: str,
) -> dict[str, str]:
    """Return a v2 child environment with no inherited import/search paths.

    ``PATH`` is deliberately rebuilt from bundle-owned directories only, and
    ``PYTHONPATH`` is omitted.  The exact interpreter is passed by absolute
    path, so neither environment variable can choose a different Paddle
    installation.  ``-I`` and the bootstrap provide a second independent
    Python-import boundary inside the child.
    """

    try:
        if runtime_home.is_symlink() or not runtime_home.is_dir():
            raise RasterRuntimeError("raster_native_sidecar_home_invalid")
        resolved_home = runtime_home.resolve(strict=True)
        if resolved_home.is_symlink() or resolved_home != runtime_home.resolve():
            raise RasterRuntimeError("raster_native_sidecar_home_invalid")
        cache_directories = {
            "cache": resolved_home / "cache",
            "modelscope": resolved_home / "modelscope",
            "paddle": resolved_home / "paddle",
            "huggingface": resolved_home / "huggingface",
            "temp": resolved_home / "temp",
            "appdata": resolved_home / "appdata",
        }
        for directory in cache_directories.values():
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink() or directory.resolve(strict=True).parent != resolved_home:
                raise RasterRuntimeError("raster_native_sidecar_home_invalid")
    except RasterRuntimeError:
        raise
    except OSError:
        raise RasterRuntimeError("raster_native_sidecar_home_invalid") from None

    home_value = str(resolved_home)
    drive = resolved_home.drive
    home_path = home_value[len(drive) :] if drive and home_value.startswith(drive) else home_value
    if not home_value or not drive or not home_path.startswith("\\"):
        raise RasterRuntimeError("raster_native_sidecar_home_invalid")
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"COMSPEC", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"}
    }
    controlled_path_entries = (str(interpreter_path.parent), str(module_root))
    allowed["PATH"] = os.pathsep.join(dict.fromkeys(controlled_path_entries))
    allowed.update(
        {
            "CUSTOMS_AGENT_LOCAL_FILES_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            "PADDLEOCR_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "TRANSFORMERS_OFFLINE": "1",
            # ``modelscope`` eagerly evaluates ``Path.home()`` while it is
            # imported.  Do not inherit a user profile: give the isolated
            # child one ephemeral, parent-owned home/cache tree instead.  It
            # is outside the immutable bundle, contains no source document,
            # and is removed immediately after the child exits.
            "HOME": home_value,
            "USERPROFILE": home_value,
            "HOMEDRIVE": drive,
            "HOMEPATH": home_path,
            "APPDATA": str(cache_directories["appdata"]),
            "LOCALAPPDATA": str(cache_directories["appdata"]),
            "XDG_CACHE_HOME": str(cache_directories["cache"]),
            "MODELSCOPE_CACHE": str(cache_directories["modelscope"]),
            "PADDLE_HOME": str(cache_directories["paddle"]),
            "HF_HOME": str(cache_directories["huggingface"]),
            "TMP": str(cache_directories["temp"]),
            "TEMP": str(cache_directories["temp"]),
            "CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_PATH": str(configuration_path),
            "CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_SHA256": configuration_sha256,
            "CUSTOMS_AGENT_PPSTRUCTURE_MODEL_ROOT": str(model_root),
            "CUSTOMS_AGENT_PPSTRUCTURE_PACKAGE_INVENTORY_PATH": str(package_inventory_path),
            "CUSTOMS_AGENT_PPSTRUCTURE_PACKAGE_INVENTORY_SHA256": package_inventory_sha256,
            "CUSTOMS_AGENT_PPSTRUCTURE_MODEL_INVENTORY_PATH": str(model_inventory_path),
            "CUSTOMS_AGENT_PPSTRUCTURE_MODEL_INVENTORY_SHA256": model_inventory_sha256,
            "CUSTOMS_AGENT_PPSTRUCTURE_REQUIRED_PACKAGES": json.dumps(
                list(required_packages), ensure_ascii=True, separators=(",", ":")
            ),
            "CUSTOMS_AGENT_PPSTRUCTURE_SIDECAR_MODULE_SHA256": sidecar_module_sha256,
        }
    )
    return allowed


def _new_isolated_v2_sidecar_home() -> tuple[Path, Callable[[], None]]:
    """Create a writable, disposable home retained by one isolated child.

    Third-party PP-Structure dependencies assume a home/cache directory even
    when every model path is explicit and networking is denied.  The ordinary
    user profile is deliberately not part of this child boundary, and the
    bundle itself is immutable, so the only safe place for that state is a
    short-lived OS temporary directory owned by the parent process.  The
    returned cleanup closure retains the ``TemporaryDirectory`` object, which
    lets a persistent service child own the directory for its full lifetime.
    """

    try:
        temporary = tempfile.TemporaryDirectory(prefix="customs-raster-sidecar-")
        home = Path(temporary.name)
        home.mkdir(mode=0o700, exist_ok=True)
        if home.is_symlink() or not home.is_dir():
            temporary.cleanup()
            raise RasterRuntimeError("raster_native_sidecar_home_invalid")
        return home, temporary.cleanup
    except RasterRuntimeError:
        raise
    except OSError:
        raise RasterRuntimeError("raster_native_sidecar_home_invalid") from None


@contextmanager
def _isolated_v2_sidecar_home() -> Iterator[Path]:
    """Context-managed form used by finite one-shot/evaluation children."""

    home, cleanup = _new_isolated_v2_sidecar_home()
    try:
        yield home
    finally:
        cleanup()


@dataclass(frozen=True, slots=True)
class PpStructureV3CommandExecutor:
    """Run one already-local PP-StructureV3 sidecar with a fixed stdin protocol.

    It deliberately invokes a hash-bound executable directly (no shell, no
    interpolated arguments), supplies only RGB raster pages, discards stderr,
    and validates all returned geometry before it enters the Agent ledger.
    """

    command_path: Path
    command_sha256: str
    timeout_seconds: int
    configuration_path: Path
    configuration_sha256: str
    model_root: Path
    package_inventory_path: Path
    package_inventory_sha256: str
    model_inventory_path: Path
    model_inventory_sha256: str
    required_packages: tuple[str, ...]
    sidecar_module_sha256: str

    def _validate_command(self) -> None:
        if _SHA256.fullmatch(self.command_sha256) is None:
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        if _sha256_regular_file(self.command_path) != self.command_sha256:
            raise RasterRuntimeError("raster_native_sidecar_drift")

    def _validate_runtime_bindings(self) -> None:
        if (
            _SHA256.fullmatch(self.configuration_sha256) is None
            or _SHA256.fullmatch(self.package_inventory_sha256) is None
            or _SHA256.fullmatch(self.model_inventory_sha256) is None
            or _SHA256.fullmatch(self.sidecar_module_sha256) is None
            or not self.required_packages
        ):
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        if (
            hashlib.sha256(
                _read_local_file(self.configuration_path, limit=_MAX_SIDECAR_BYTES)
            ).hexdigest()
            != self.configuration_sha256
        ):
            raise RasterRuntimeError("raster_native_sidecar_drift")
        package_inventory_raw = _read_local_file(
            self.package_inventory_path, limit=_MAX_SIDECAR_BYTES
        )
        _verify_package_inventory(
            raw=package_inventory_raw,
            expected_sha256=self.package_inventory_sha256,
            required_packages=self.required_packages,
        )
        if self.model_root.is_symlink() or not self.model_root.is_dir():
            raise RasterRuntimeError("raster_model_inventory_invalid")
        model_inventory_raw = _read_local_file(self.model_inventory_path, limit=_MAX_SIDECAR_BYTES)
        _verify_model_inventory(
            raw=model_inventory_raw,
            expected_sha256=self.model_inventory_sha256,
            model_root=self.model_root,
        )
        if (
            _sha256_regular_file(Path(_PPSTRUCTUREV3_SIDECAR_MODULE_FILE))
            != self.sidecar_module_sha256
        ):
            raise RasterRuntimeError("raster_native_sidecar_drift")

    @staticmethod
    def _payload(
        pages: Sequence[RasterizedPageV2], *, source_sha256: str, raster_profile_revision: str
    ) -> bytes:
        header = {
            "protocol": "customs-raster-ppstructurev3-v1",
            "source_sha256": source_sha256,
            "raster_profile_revision": raster_profile_revision,
            "pages": [
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "pixel_length": len(page.pixels),
                    "pixel_sha256": hashlib.sha256(page.pixels).hexdigest(),
                }
                for page in pages
            ],
        }
        encoded_header = json.dumps(
            header, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        payload = encoded_header + b"\n" + b"".join(page.pixels for page in pages)
        if len(payload) > 256 * 1024 * 1024:
            raise RasterRuntimeError("raster_native_sidecar_input_too_large")
        return payload

    @staticmethod
    def _ledger_from_sidecar(
        result: PpStructureV3SidecarResultV1,
        *,
        pages: Sequence[RasterizedPageV2],
    ) -> RasterOcrLayoutLedgerV2:
        expected_dimensions = [(page.page_number, page.width, page.height) for page in pages]
        returned_dimensions = [(page.page_number, page.width, page.height) for page in result.pages]
        if returned_dimensions != expected_dimensions:
            raise RasterRuntimeError("raster_native_sidecar_page_binding_invalid")
        spans = tuple(
            RasterLedgerSpanV2(
                span_id=item.span_id,
                page_number=page.page_number,
                ordinal=item.ordinal,
                reading_order=item.reading_order,
                text=item.text,
                bbox=item.bbox,
                confidence=item.confidence,
                polygon=item.polygon,
            )
            for page in result.pages
            for item in page.spans
        )
        fields = tuple(
            RasterLedgerFieldV2(
                field_id=item.field_id,
                value=item.value,
                confidence=item.confidence,
                evidence_span_ids=tuple(item.evidence_span_ids),
                bbox=item.bbox,
            )
            for item in result.field_candidates
        )
        tables = tuple(
            RasterLedgerTableV2(
                table_id=table.table_id,
                page_number=table.page_number,
                bbox=table.bbox,
                cells=tuple(
                    RasterLedgerCellV2(
                        cell_id=cell.cell_id,
                        page_number=table.page_number,
                        row_index=cell.row_index,
                        column_index=cell.column_index,
                        text=cell.text,
                        bbox=cell.bbox,
                        confidence=cell.confidence,
                        evidence_span_ids=tuple(cell.evidence_span_ids),
                    )
                    for row in table.rows
                    for cell in row.cells
                ),
            )
            for table in result.tables
        )
        return RasterOcrLayoutLedgerV2(
            source_sha256=result.source_sha256,
            raster_profile_revision=result.raster_profile_revision,
            provider_revision=result.provider_revision,
            pages=tuple(pages),
            spans=spans,
            document_type=result.document_type,
            classification_confidence=result.classification_confidence,
            fields=fields,
            tables=tables,
            warnings=tuple(result.warnings),
            needs_review_reasons=tuple(result.needs_review_reasons),
        )

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
        profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2,
    ) -> RasterOcrLayoutLedgerV2:
        self._validate_command()
        self._validate_runtime_bindings()
        try:
            completed = subprocess.run(
                [str(self.command_path), "--customs-raster-ppstructurev3-v1"],
                input=self._payload(
                    pages,
                    source_sha256=source_sha256,
                    raster_profile_revision=raster_profile_revision,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                cwd=str(self.command_path.parent),
                env=_sidecar_environment(
                    configuration_path=self.configuration_path,
                    configuration_sha256=self.configuration_sha256,
                    model_root=self.model_root,
                    package_inventory_path=self.package_inventory_path,
                    package_inventory_sha256=self.package_inventory_sha256,
                    model_inventory_path=self.model_inventory_path,
                    model_inventory_sha256=self.model_inventory_sha256,
                    required_packages=self.required_packages,
                    sidecar_module_sha256=self.sidecar_module_sha256,
                ),
                shell=False,
                timeout=min(self.timeout_seconds, profile.sidecar_timeout_seconds, 420),
            )
        except (OSError, subprocess.TimeoutExpired):
            raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
        if completed.returncode != 0 or len(completed.stdout) > _MAX_SIDECAR_BYTES:
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        try:
            result = PpStructureV3SidecarResultV1.model_validate_json(completed.stdout)
        except (ValidationError, UnicodeDecodeError, ValueError):
            raise RasterRuntimeError("raster_native_sidecar_output_invalid") from None
        if (
            result.source_sha256 != source_sha256
            or result.raster_profile_revision != raster_profile_revision
            or result.provider_revision != profile.extractor_revision
        ):
            raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
        return self._ledger_from_sidecar(result, pages=pages)


@dataclass(frozen=True, slots=True)
class PpStructureV3CommandExecutorV2:
    """v2 command executor with parent and sidecar inventories kept separate."""

    interpreter_path: Path
    interpreter_sha256: str
    interpreter_root: Path
    interpreter_root_tree_sha256: str
    timeout_seconds: int
    configuration_path: Path
    configuration_sha256: str
    model_root: Path
    parent_renderer_package_inventory_path: Path
    parent_renderer_package_inventory_sha256: str
    parent_renderer_required_packages: tuple[str, ...]
    sidecar_package_inventory_path: Path
    sidecar_package_inventory_sha256: str
    sidecar_required_packages: tuple[str, ...]
    model_inventory_path: Path
    model_inventory_sha256: str
    sidecar_module_root: Path
    sidecar_module_root_tree_sha256: str
    sidecar_command_path: Path
    sidecar_command_sha256: str
    sidecar_module_path: Path
    sidecar_module_sha256: str

    def _validate_interpreter(self) -> None:
        if (
            _SHA256.fullmatch(self.interpreter_sha256) is None
            or _SHA256.fullmatch(self.interpreter_root_tree_sha256) is None
        ):
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        _validate_controlled_child_file(
            root_path=self.interpreter_root,
            file_path=self.interpreter_path,
        )
        if _controlled_directory_tree_sha256(self.interpreter_root) != (
            self.interpreter_root_tree_sha256
        ):
            raise RasterRuntimeError("raster_native_sidecar_drift")
        if _sha256_regular_file(self.interpreter_path) != self.interpreter_sha256:
            raise RasterRuntimeError("raster_native_sidecar_drift")

    def verify_runtime_state(self) -> None:
        """Recheck all parent-observable v2 materials before every request."""

        if (
            _SHA256.fullmatch(self.configuration_sha256) is None
            or _SHA256.fullmatch(self.parent_renderer_package_inventory_sha256) is None
            or _SHA256.fullmatch(self.sidecar_package_inventory_sha256) is None
            or _SHA256.fullmatch(self.model_inventory_sha256) is None
            or _SHA256.fullmatch(self.sidecar_module_root_tree_sha256) is None
            or _SHA256.fullmatch(self.sidecar_command_sha256) is None
            or _SHA256.fullmatch(self.sidecar_module_sha256) is None
            or not self.parent_renderer_required_packages
            or not self.sidecar_required_packages
            or self.sidecar_command_path != self.sidecar_module_path
            or self.sidecar_command_sha256 != self.sidecar_module_sha256
        ):
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        if (
            hashlib.sha256(
                _read_local_file(self.configuration_path, limit=_MAX_SIDECAR_BYTES)
            ).hexdigest()
            != self.configuration_sha256
        ):
            raise RasterRuntimeError("raster_native_sidecar_drift")
        parent_inventory_raw = _read_local_file(
            self.parent_renderer_package_inventory_path, limit=_MAX_SIDECAR_BYTES
        )
        _verify_package_inventory(
            raw=parent_inventory_raw,
            expected_sha256=self.parent_renderer_package_inventory_sha256,
            required_packages=self.parent_renderer_required_packages,
        )
        # The regular Agent must never import or version-check Paddle packages.
        # It only verifies that the immutable inventory exists and has the
        # expected declared package names.  The sidecar verifies actual
        # installed versions before its native imports on every OCR request.
        sidecar_inventory_raw = _read_local_file(
            self.sidecar_package_inventory_path, limit=_MAX_SIDECAR_BYTES
        )
        _verify_package_inventory(
            raw=sidecar_inventory_raw,
            expected_sha256=self.sidecar_package_inventory_sha256,
            required_packages=self.sidecar_required_packages,
            verify_installed=False,
        )
        if self.model_root.is_symlink() or not self.model_root.is_dir():
            raise RasterRuntimeError("raster_model_inventory_invalid")
        model_inventory_raw = _read_local_file(self.model_inventory_path, limit=_MAX_SIDECAR_BYTES)
        _verify_model_inventory(
            raw=model_inventory_raw,
            expected_sha256=self.model_inventory_sha256,
            model_root=self.model_root,
        )
        _validate_controlled_child_file(
            root_path=self.sidecar_module_root,
            file_path=self.sidecar_module_path,
        )
        _validate_controlled_child_file(
            root_path=self.sidecar_module_root,
            file_path=self.sidecar_command_path,
        )
        if _controlled_module_root_tree_sha256(self.sidecar_module_root) != (
            self.sidecar_module_root_tree_sha256
        ):
            raise RasterRuntimeError("raster_native_sidecar_drift")
        if _sha256_regular_file(self.sidecar_module_path) != self.sidecar_module_sha256:
            raise RasterRuntimeError("raster_native_sidecar_drift")
        if _sha256_regular_file(self.sidecar_command_path) != self.sidecar_command_sha256:
            raise RasterRuntimeError("raster_native_sidecar_drift")
        self._validate_interpreter()

    # Keep the same private name as v1 so test seams and callers can ask for
    # an explicit drift check without invoking a native OCR process.
    def _validate_runtime_bindings(self) -> None:
        self.verify_runtime_state()

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
        profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2,
    ) -> RasterOcrLayoutLedgerV2:
        self.verify_runtime_state()
        try:
            with _isolated_v2_sidecar_home() as runtime_home:
                completed = subprocess.run(
                    [
                        str(self.interpreter_path),
                        "-I",
                        "-B",
                        "-c",
                        _ISOLATED_SIDECAR_BOOTSTRAP,
                        str(self.interpreter_root),
                        str(self.sidecar_module_root),
                        str(self.sidecar_command_path),
                        "--customs-raster-ppstructurev3-v1",
                    ],
                    input=PpStructureV3CommandExecutor._payload(
                        pages,
                        source_sha256=source_sha256,
                        raster_profile_revision=raster_profile_revision,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    cwd=str(self.sidecar_module_root),
                    env=_isolated_v2_sidecar_environment(
                        interpreter_path=self.interpreter_path,
                        module_root=self.sidecar_module_root,
                        runtime_home=runtime_home,
                        configuration_path=self.configuration_path,
                        configuration_sha256=self.configuration_sha256,
                        model_root=self.model_root,
                        package_inventory_path=self.sidecar_package_inventory_path,
                        package_inventory_sha256=self.sidecar_package_inventory_sha256,
                        model_inventory_path=self.model_inventory_path,
                        model_inventory_sha256=self.model_inventory_sha256,
                        required_packages=self.sidecar_required_packages,
                        sidecar_module_sha256=self.sidecar_module_sha256,
                    ),
                    shell=False,
                    timeout=min(self.timeout_seconds, profile.sidecar_timeout_seconds, 420),
                )
        except (OSError, subprocess.TimeoutExpired):
            raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
        if completed.returncode != 0 or len(completed.stdout) > _MAX_SIDECAR_BYTES:
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        try:
            result = PpStructureV3SidecarResultV1.model_validate_json(completed.stdout)
        except (ValidationError, UnicodeDecodeError, ValueError):
            raise RasterRuntimeError("raster_native_sidecar_output_invalid") from None
        if (
            result.source_sha256 != source_sha256
            or result.raster_profile_revision != raster_profile_revision
            or result.provider_revision != profile.extractor_revision
        ):
            raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
        return PpStructureV3CommandExecutor._ledger_from_sidecar(result, pages=pages)

    def probe_native_structure(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
        profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2,
    ) -> PpStructureV3StructureProbeResultV1:
        """Return bounded native table/OCR counters for local diagnosis only.

        This follows the exact same hash-bound interpreter, sidecar module,
        configuration, model inventory, RGB-to-BGR conversion, network denial
        and input protocol as ``extract_layout``.  It is deliberately absent
        from the provider protocol and cannot create candidate/evaluation or
        approval evidence.  Its result contains no document or OCR content.
        """

        if (
            not pages
            or raster_profile_revision != profile.raster_profile_revision
            or not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
        ):
            raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
        self.verify_runtime_state()
        try:
            with _isolated_v2_sidecar_home() as runtime_home:
                completed = subprocess.run(
                    [
                        str(self.interpreter_path),
                        "-I",
                        "-B",
                        "-c",
                        _ISOLATED_SIDECAR_BOOTSTRAP,
                        str(self.interpreter_root),
                        str(self.sidecar_module_root),
                        str(self.sidecar_command_path),
                        "--customs-raster-ppstructurev3-structure-probe-v1",
                    ],
                    input=PpStructureV3CommandExecutor._payload(
                        pages,
                        source_sha256=source_sha256,
                        raster_profile_revision=raster_profile_revision,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    cwd=str(self.sidecar_module_root),
                    env=_isolated_v2_sidecar_environment(
                        interpreter_path=self.interpreter_path,
                        module_root=self.sidecar_module_root,
                        runtime_home=runtime_home,
                        configuration_path=self.configuration_path,
                        configuration_sha256=self.configuration_sha256,
                        model_root=self.model_root,
                        package_inventory_path=self.sidecar_package_inventory_path,
                        package_inventory_sha256=self.sidecar_package_inventory_sha256,
                        model_inventory_path=self.model_inventory_path,
                        model_inventory_sha256=self.model_inventory_sha256,
                        required_packages=self.sidecar_required_packages,
                        sidecar_module_sha256=self.sidecar_module_sha256,
                    ),
                    shell=False,
                    timeout=min(self.timeout_seconds, profile.sidecar_timeout_seconds, 420),
                )
        except (OSError, subprocess.TimeoutExpired):
            raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
        if (
            completed.returncode != 0
            or not completed.stdout
            or len(completed.stdout) > _MAX_PPSTRUCTUREV3_STRUCTURE_PROBE_BYTES
        ):
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        try:
            result = PpStructureV3StructureProbeResultV1.model_validate_json(completed.stdout)
        except (ValidationError, UnicodeDecodeError, ValueError):
            raise RasterRuntimeError("raster_native_sidecar_output_invalid") from None
        if len(result.pages) != len(pages) or [item.page_number for item in result.pages] != [
            page.page_number for page in pages
        ]:
            raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
        return result

    @contextmanager
    def open_evaluation_session(
        self, *, profile: PpStructureV3OfflineProfileV2
    ) -> Iterator["PpStructureV3CandidateBatchSessionV2"]:
        """Open the hash-bound batch protocol used only by synthetic evaluation.

        Normal OCR requests stay on ``extract_layout`` above, which verifies
        the whole bundle for every request.  The evaluator instead validates
        the same complete material immediately before process start and after
        its clean exit, then keeps the one verified native pipeline alive for
        a finite split.  It has no approval-pointer capability or fallback.
        """

        if (
            not isinstance(profile, PpStructureV3OfflineProfileV2)
            or profile.sidecar_timeout_seconds != self.timeout_seconds
            or profile.raster_profile_revision.startswith("synthetic-raster-v1:") is False
            or profile.evaluation_batch_size not in {1, _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS}
        ):
            raise RasterRuntimeError("raster_evaluation_session_invalid")
        self.verify_runtime_state()
        with _isolated_v2_sidecar_home() as runtime_home:
            try:
                process = subprocess.Popen(
                    [
                        str(self.interpreter_path),
                        "-I",
                        "-B",
                        "-c",
                        _ISOLATED_SIDECAR_BOOTSTRAP,
                        str(self.interpreter_root),
                        str(self.sidecar_module_root),
                        str(self.sidecar_command_path),
                        (
                            "--customs-raster-ppstructurev3-evaluation-batch-v2"
                            if profile.evaluation_batch_size
                            == _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS
                            else "--customs-raster-ppstructurev3-batch-v1"
                        ),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    cwd=str(self.sidecar_module_root),
                    env=_isolated_v2_sidecar_environment(
                        interpreter_path=self.interpreter_path,
                        module_root=self.sidecar_module_root,
                        runtime_home=runtime_home,
                        configuration_path=self.configuration_path,
                        configuration_sha256=self.configuration_sha256,
                        model_root=self.model_root,
                        package_inventory_path=self.sidecar_package_inventory_path,
                        package_inventory_sha256=self.sidecar_package_inventory_sha256,
                        model_inventory_path=self.model_inventory_path,
                        model_inventory_sha256=self.model_inventory_sha256,
                        required_packages=self.sidecar_required_packages,
                        sidecar_module_sha256=self.sidecar_module_sha256,
                    ),
                    shell=False,
                    bufsize=0,
                )
            except OSError:
                raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
            session = PpStructureV3CandidateBatchSessionV2(
                process=process,
                profile=profile,
                # Cold import/model loading is an activation/evaluation setup
                # cost.  It may use the profile's existing 420-second bound,
                # while every warm candidate response must fit the same
                # 55-second native budget used by the actual service.
                timeout_seconds=min(
                    self.timeout_seconds,
                    profile.sidecar_timeout_seconds,
                    _MAX_PPSTRUCTUREV3_SERVICE_REQUEST_SECONDS,
                ),
                startup_timeout_seconds=min(
                    self.timeout_seconds, profile.sidecar_timeout_seconds, 420
                ),
            )
            try:
                session.verify_ready()
                yield session
                session.close()
            except BaseException:
                session.abort()
                raise
            finally:
                # Even a failed/aborted evaluation may not leave evidence if a
                # model, package, interpreter, or sidecar changed while it ran.
                # This recheck intentionally has no one-shot fallback.
                self.verify_runtime_state()

    def start_service_session(
        self, *, profile: PpStructureV3OfflineProfileV2
    ) -> "PpStructureV3ServiceSessionV1":
        """Start one warm sidecar for one already-approved v2 profile.

        This is intentionally separate from :meth:`open_evaluation_session`:
        it launches a distinct child protocol, produces no evaluation
        artifacts, and gives every customer request a short independent
        deadline.  Full parent material verification occurs before process
        start; the sidecar independently verifies its package/model inventory
        before importing PaddleOCR.  Later requests use only the process that
        was created from that immutable snapshot.
        """

        if (
            not isinstance(profile, PpStructureV3OfflineProfileV2)
            or profile.raster_profile_revision.startswith("synthetic-raster-v1:") is False
            # The candidate evaluator may use four-image batches, but a
            # serving sidecar still receives one customer document at a time.
            # Batch size is therefore evaluation-only provenance; rejecting a
            # verified batch-4 profile here would make a successfully
            # activated profile unavailable for an unrelated reason.
            or profile.evaluation_batch_size not in {1, _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS}
        ):
            raise RasterRuntimeError("raster_service_session_invalid")
        self.verify_runtime_state()
        runtime_home, cleanup_home = _new_isolated_v2_sidecar_home()
        process: subprocess.Popen[bytes] | None = None
        session: PpStructureV3ServiceSessionV1 | None = None
        try:
            process = subprocess.Popen(
                [
                    str(self.interpreter_path),
                    "-I",
                    "-B",
                    "-c",
                    _ISOLATED_SIDECAR_BOOTSTRAP,
                    str(self.interpreter_root),
                    str(self.sidecar_module_root),
                    str(self.sidecar_command_path),
                    f"--{_PPSTRUCTUREV3_SERVICE_PROTOCOL}",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self.sidecar_module_root),
                env=_isolated_v2_sidecar_environment(
                    interpreter_path=self.interpreter_path,
                    module_root=self.sidecar_module_root,
                    runtime_home=runtime_home,
                    configuration_path=self.configuration_path,
                    configuration_sha256=self.configuration_sha256,
                    model_root=self.model_root,
                    package_inventory_path=self.sidecar_package_inventory_path,
                    package_inventory_sha256=self.sidecar_package_inventory_sha256,
                    model_inventory_path=self.model_inventory_path,
                    model_inventory_sha256=self.model_inventory_sha256,
                    required_packages=self.sidecar_required_packages,
                    sidecar_module_sha256=self.sidecar_module_sha256,
                ),
                shell=False,
                bufsize=0,
            )
            session = PpStructureV3ServiceSessionV1(
                process=process,
                profile=profile,
                request_timeout_seconds=min(
                    self.timeout_seconds,
                    profile.sidecar_timeout_seconds,
                    _MAX_PPSTRUCTUREV3_SERVICE_REQUEST_SECONDS,
                ),
                startup_timeout_seconds=min(
                    self.timeout_seconds, profile.sidecar_timeout_seconds, 420
                ),
                cleanup_home=cleanup_home,
            )
            session.verify_ready()
            return session
        except (OSError, RasterRuntimeError):
            if session is not None:
                session.abort()
            elif process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if session is None:
                cleanup_home()
            raise RasterRuntimeError("raster_native_sidecar_unavailable") from None


def _read_exact_batch_pipe(stream: BinaryIO, size: int) -> bytes:
    if size < 1:
        raise RasterRuntimeError("raster_native_sidecar_output_invalid")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_batch_response_from_pipe(stream: BinaryIO, *, maximum_size: int) -> bytes:
    raw_length = _read_exact_batch_pipe(stream, 8)
    length = struct.unpack(">Q", raw_length)[0]
    if not 1 <= length <= maximum_size:
        raise RasterRuntimeError("raster_native_sidecar_output_invalid")
    return _read_exact_batch_pipe(stream, length)


def _evaluation_batch_payload(items: Sequence[PpStructureV3EvaluationBatchInputV2]) -> bytes:
    """Encode 1--4 independently bound requests for sidecar batch-v2."""

    if not 1 <= len(items) <= _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS:
        raise RasterRuntimeError("raster_evaluation_session_invalid")
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise RasterRuntimeError("raster_evaluation_session_invalid")
    nested_payloads: list[bytes] = []
    for item in items:
        if len(item.pages) != 1:
            raise RasterRuntimeError("raster_evaluation_session_invalid")
        nested_payloads.append(
            PpStructureV3CommandExecutor._payload(
                item.pages,
                source_sha256=item.source_sha256,
                raster_profile_revision=item.raster_profile_revision,
            )
        )
    raw_total = sum(len(payload) for payload in nested_payloads)
    if raw_total > _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_RAW_BYTES:
        raise RasterRuntimeError("raster_native_sidecar_input_too_large")
    header = {
        "protocol": _PPSTRUCTUREV3_EVALUATION_BATCH_PROTOCOL,
        "items": [
            {
                "item_id": item.item_id,
                "input_length": len(payload),
                "input_sha256": hashlib.sha256(payload).hexdigest(),
            }
            for item, payload in zip(items, nested_payloads, strict=True)
        ],
    }
    encoded_header = json.dumps(
        header, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    payload = encoded_header + b"\n" + b"".join(nested_payloads)
    if not 1 <= len(payload) <= _MAX_PPSTRUCTUREV3_BATCH_REQUEST_BYTES:
        raise RasterRuntimeError("raster_native_sidecar_input_too_large")
    return payload


@dataclass(slots=True)
class PpStructureV3ServiceSessionV1:
    """One sequential, warm sidecar session for a single approved profile.

    Evaluation sessions intentionally have a different class and protocol.
    This session owns its temporary home until it has reaped the child, so
    Windows never retains a cache directory or a descendant process after a
    timeout, pointer replacement, or service shutdown.
    """

    process: subprocess.Popen[bytes]
    profile: PpStructureV3OfflineProfileV2
    request_timeout_seconds: int
    startup_timeout_seconds: int
    cleanup_home: Callable[[], None]
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def _cleanup_home(self) -> None:
        try:
            self.cleanup_home()
        except OSError:
            # The process has already been terminated/reaped first.  A failed
            # best-effort temporary-directory cleanup cannot make a stale
            # service binding usable again.
            pass

    def _terminate_process(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                    self.process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        for stream in (self.process.stdin, self.process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def _discard(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminate_process()
        self._cleanup_home()

    def abort(self) -> None:
        """Immediately terminate an unhealthy or superseded service child."""

        with self._lock:
            if self._closed:
                return
            self._discard()

    def _read_response_with_timeout(self, *, timeout_seconds: int) -> bytes:
        stdout = self.process.stdout
        if stdout is None:
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        result: list[bytes] = []
        failure: list[BaseException] = []

        def _reader() -> None:
            try:
                result.append(
                    _read_batch_response_from_pipe(
                        cast(BinaryIO, stdout), maximum_size=_MAX_SIDECAR_BYTES
                    )
                )
            except BaseException as error:  # Process boundary only.
                failure.append(error)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            self._discard()
            thread.join(timeout=10)
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        if failure:
            if isinstance(failure[0], RasterRuntimeError):
                raise failure[0]
            raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
        if len(result) != 1:
            raise RasterRuntimeError("raster_native_sidecar_output_invalid")
        return result[0]

    def verify_ready(self) -> None:
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            raw = self._read_response_with_timeout(timeout_seconds=self.startup_timeout_seconds)
            expected = json.dumps(
                {"protocol": _PPSTRUCTUREV3_SERVICE_PROTOCOL, "status": "ready"},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if raw != expected:
                self._discard()
                raise RasterRuntimeError("raster_native_sidecar_output_invalid")

    def verify_runtime_state(self) -> None:
        """Cheap liveness check for the already full-verified process.

        The full package/model/module-tree verification is performed before
        this process starts.  Rehashing those trees per request defeats the
        purpose of the immutable warm session; a changed pointer instead
        retires this session and forces a fresh full verification.
        """

        with self._lock:
            if self._closed or self.process.poll() is not None:
                if not self._closed:
                    self._discard()
                raise RasterRuntimeError("raster_native_sidecar_unavailable")

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
        profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2,
    ) -> RasterOcrLayoutLedgerV2:
        if (
            profile != self.profile
            or raster_profile_revision != self.profile.raster_profile_revision
        ):
            raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
        payload = PpStructureV3CommandExecutor._payload(
            pages,
            source_sha256=source_sha256,
            raster_profile_revision=raster_profile_revision,
        )
        if not 1 <= len(payload) <= _MAX_PPSTRUCTUREV3_BATCH_REQUEST_BYTES:
            raise RasterRuntimeError("raster_native_sidecar_input_too_large")
        with self._lock:
            if self._closed or self.process.poll() is not None:
                if not self._closed:
                    self._discard()
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            stdin = self.process.stdin
            if stdin is None:
                self._discard()
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            try:
                stdin.write(struct.pack(">Q", len(payload)))
                stdin.write(payload)
                stdin.flush()
                raw = self._read_response_with_timeout(timeout_seconds=self.request_timeout_seconds)
            except RasterRuntimeError:
                self._discard()
                raise
            except OSError:
                self._discard()
                raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
            try:
                result = PpStructureV3SidecarResultV1.model_validate_json(raw)
            except (ValidationError, UnicodeDecodeError, ValueError):
                self._discard()
                raise RasterRuntimeError("raster_native_sidecar_output_invalid") from None
            if (
                result.source_sha256 != source_sha256
                or result.raster_profile_revision != raster_profile_revision
                or result.provider_revision != self.profile.extractor_revision
            ):
                self._discard()
                raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
            return PpStructureV3CommandExecutor._ledger_from_sidecar(result, pages=pages)

    def close(self) -> None:
        """Close and reap the service child without writing evaluation evidence."""

        self.abort()


@dataclass(slots=True)
class PpStructureV3CandidateBatchSessionV2:
    """One sequential, evaluation-only PP-StructureV3 sidecar session."""

    process: subprocess.Popen[bytes]
    profile: PpStructureV3OfflineProfileV2
    timeout_seconds: int
    startup_timeout_seconds: int | None = None
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        # Retain the compact test/evaluation construction form while making
        # the production caller state its longer setup bound explicitly.
        if self.startup_timeout_seconds is None:
            self.startup_timeout_seconds = self.timeout_seconds
        if self.timeout_seconds < 1 or self.startup_timeout_seconds < 1:
            raise RasterRuntimeError("raster_evaluation_session_invalid")

    def _abort_process(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                    self.process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        for stream in (self.process.stdin, self.process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def abort(self) -> None:
        """End a failed session without accepting a partial candidate set."""

        if self._closed:
            return
        self._closed = True
        self._abort_process()

    def _read_response_with_timeout(
        self,
        *,
        maximum_size: int = _MAX_SIDECAR_BYTES,
        timeout_seconds: int | None = None,
    ) -> bytes:
        stdout = self.process.stdout
        if stdout is None:
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        result: list[bytes] = []
        failure: list[BaseException] = []

        def _reader() -> None:
            try:
                result.append(
                    _read_batch_response_from_pipe(
                        cast(BinaryIO, stdout), maximum_size=maximum_size
                    )
                )
            except BaseException as error:  # Kept private at the process boundary.
                failure.append(error)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds)
        if thread.is_alive():
            self._abort_process()
            thread.join(timeout=10)
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        if failure:
            if isinstance(failure[0], RasterRuntimeError):
                raise failure[0]
            raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
        if len(result) != 1:
            raise RasterRuntimeError("raster_native_sidecar_output_invalid")
        return result[0]

    def verify_ready(self) -> None:
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            assert self.startup_timeout_seconds is not None
            raw = self._read_response_with_timeout(timeout_seconds=self.startup_timeout_seconds)
            expected_protocol = (
                _PPSTRUCTUREV3_EVALUATION_BATCH_PROTOCOL
                if self.profile.evaluation_batch_size == _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS
                else _PPSTRUCTUREV3_BATCH_PROTOCOL
            )
            expected = json.dumps(
                {"protocol": expected_protocol, "status": "ready"},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if raw != expected:
                raise RasterRuntimeError("raster_native_sidecar_output_invalid")

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
        profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2,
    ) -> RasterOcrLayoutLedgerV2:
        if (
            profile != self.profile
            or raster_profile_revision != self.profile.raster_profile_revision
            or self.profile.evaluation_batch_size != 1
        ):
            raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
        payload = PpStructureV3CommandExecutor._payload(
            pages,
            source_sha256=source_sha256,
            raster_profile_revision=raster_profile_revision,
        )
        if not 1 <= len(payload) <= _MAX_PPSTRUCTUREV3_BATCH_REQUEST_BYTES:
            raise RasterRuntimeError("raster_native_sidecar_input_too_large")
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            stdin = self.process.stdin
            if stdin is None:
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            try:
                stdin.write(struct.pack(">Q", len(payload)))
                stdin.write(payload)
                stdin.flush()
            except OSError:
                self._abort_process()
                raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
            raw = self._read_response_with_timeout()
            try:
                result = PpStructureV3SidecarResultV1.model_validate_json(raw)
            except (ValidationError, UnicodeDecodeError, ValueError):
                raise RasterRuntimeError("raster_native_sidecar_output_invalid") from None
            if (
                result.source_sha256 != source_sha256
                or result.raster_profile_revision != raster_profile_revision
                or result.provider_revision != self.profile.extractor_revision
            ):
                raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
            return PpStructureV3CommandExecutor._ledger_from_sidecar(result, pages=pages)

    def extract_layout_batch(
        self,
        items: Sequence[PpStructureV3EvaluationBatchInputV2],
        *,
        profile: PpStructureV3OfflineProfileV2,
    ) -> tuple[RasterOcrLayoutLedgerV2, ...]:
        """Execute one hash-bound native list-input batch for synthetic eval.

        This method is deliberately absent from the service provider protocol.
        It is callable only by the finite, local source-attested evaluator and
        rejects a legacy sequential profile rather than silently changing its
        sidecar protocol.
        """

        if (
            profile != self.profile
            or profile.evaluation_batch_size != _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS
            or not 1 <= len(items) <= _MAX_PPSTRUCTUREV3_EVALUATION_BATCH_ITEMS
            or any(
                item.raster_profile_revision != profile.raster_profile_revision
                or len(item.pages) != 1
                for item in items
            )
        ):
            raise RasterRuntimeError("raster_evaluation_session_invalid")
        payload = _evaluation_batch_payload(items)
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            stdin = self.process.stdin
            if stdin is None:
                raise RasterRuntimeError("raster_native_sidecar_unavailable")
            try:
                stdin.write(struct.pack(">Q", len(payload)))
                stdin.write(payload)
                stdin.flush()
            except OSError:
                self._abort_process()
                raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
            raw = self._read_response_with_timeout(
                maximum_size=_MAX_PPSTRUCTUREV3_EVALUATION_BATCH_RESPONSE_BYTES
            )
            try:
                response = _PpStructureV3EvaluationBatchResponseV2.model_validate_json(raw)
            except (ValidationError, UnicodeDecodeError, ValueError):
                raise RasterRuntimeError("raster_native_sidecar_output_invalid") from None
            expected_ids = {item.item_id for item in items}
            observed = {item.item_id: item.result for item in response.results}
            if len(observed) != len(response.results) or set(observed) != expected_ids:
                raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
            ledgers: list[RasterOcrLayoutLedgerV2] = []
            for item in items:
                result = observed[item.item_id]
                if (
                    result.source_sha256 != item.source_sha256
                    or result.raster_profile_revision != item.raster_profile_revision
                    or result.provider_revision != profile.extractor_revision
                ):
                    raise RasterRuntimeError("raster_native_sidecar_binding_invalid")
                ledgers.append(
                    PpStructureV3CommandExecutor._ledger_from_sidecar(result, pages=item.pages)
                )
            return tuple(ledgers)

    def close(self) -> None:
        """Require clean EOF and a zero exit before evidence can be written."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                returncode = self.process.wait(timeout=self.timeout_seconds)
                if returncode != 0:
                    raise RasterRuntimeError("raster_native_sidecar_unavailable")
                if self.process.stdout is None or self.process.stdout.read(1):
                    raise RasterRuntimeError("raster_native_sidecar_output_invalid")
            except RasterRuntimeError:
                self._abort_process()
                raise
            except (OSError, subprocess.TimeoutExpired):
                self._abort_process()
                raise RasterRuntimeError("raster_native_sidecar_unavailable") from None
            finally:
                try:
                    if self.process.stdout is not None:
                        self.process.stdout.close()
                except OSError:
                    pass


@dataclass(frozen=True, slots=True)
class PpStructureV3OfflineProvider(RasterOcrLayoutProvider):
    """Adapter that refuses a sidecar output not bound to its pinned profile."""

    profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2
    executor: PpStructureV3NativeExecutor
    thresholds: RasterCalibrationThresholdsV1 | None = None

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
    ) -> RasterOcrLayoutLedgerV2:
        if raster_profile_revision != self.profile.raster_profile_revision:
            raise RasterDocumentPreviewUnavailable("raster profile revision is not approved")
        try:
            with local_files_only():
                ledger = self.executor.extract_layout(
                    pages,
                    source_sha256=source_sha256,
                    raster_profile_revision=raster_profile_revision,
                    profile=self.profile,
                )
        except RasterDocumentPreviewUnavailable:
            raise
        except Exception:
            raise RasterDocumentPreviewUnavailable(
                "PP-StructureV3 native sidecar is unavailable"
            ) from None
        if (
            ledger.provider_revision != self.profile.extractor_revision
            or ledger.raster_profile_revision != self.profile.raster_profile_revision
            or ledger.source_sha256 != source_sha256
        ):
            raise RasterDocumentPreviewUnavailable("PP-StructureV3 sidecar binding is invalid")
        if self.thresholds is None:
            return ledger
        reasons = list(ledger.needs_review_reasons)
        if (
            ledger.classification_confidence is not None
            and ledger.classification_confidence < self.thresholds.classification_confidence
        ):
            reasons.append(RasterExtractionWarningV2(code="confidence_below_calibration"))
        if any(span.confidence < self.thresholds.span_confidence for span in ledger.spans):
            reasons.append(RasterExtractionWarningV2(code="confidence_below_calibration"))
        if any(field.confidence < self.thresholds.field_confidence for field in ledger.fields):
            reasons.append(RasterExtractionWarningV2(code="confidence_below_calibration"))
        if any(
            cell.confidence < self.thresholds.table_cell_confidence
            for table in ledger.tables
            for cell in table.cells
        ):
            reasons.append(RasterExtractionWarningV2(code="confidence_below_calibration"))
        if tuple(reasons) == ledger.needs_review_reasons:
            return ledger
        deduplicated = tuple({reason.code: reason for reason in reasons}.values())
        return replace(ledger, needs_review_reasons=deduplicated)


@dataclass(slots=True)
class PaddleOcr3DirectRasterBaselineProvider(RasterOcrLayoutProvider):
    """Pinned direct P3 OCR comparison candidate, never a layout approval.

    It reuses the existing local P3 det/rec adapter only for direct text
    spans.  PP-StructureV3 remains the sole layout/table approval candidate:
    this baseline emits no document classification, fields, or table cells and
    always carries a review reason.  The resolver below has no approval
    pointer input, so it cannot accidentally become the active runtime.
    """

    profile: PaddleOcr3DirectRasterBaselineProfileV1
    model_root: Path
    _adapter: object | None = field(default=None, init=False, repr=False)

    def _native_adapter(self) -> object:
        if self._adapter is None:
            from customs_ai_agent.service.document_preview_runtime import (
                PaddleOcr3PdfPreviewAdapter,
            )

            self._adapter = PaddleOcr3PdfPreviewAdapter(
                extractor_revision=self.profile.extractor_revision,
                model_root=self.model_root,
                options={
                    "text_detection_model_name": self.profile.text_detection_model_name,
                    "text_detection_model_directory": self.profile.text_detection_model_directory,
                    "text_recognition_model_name": self.profile.text_recognition_model_name,
                    "text_recognition_model_directory": self.profile.text_recognition_model_directory,
                },
            )
        return self._adapter

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
    ) -> RasterOcrLayoutLedgerV2:
        if raster_profile_revision != self.profile.raster_profile_revision:
            raise RasterDocumentPreviewUnavailable(
                "raster baseline profile revision is not provisioned"
            )
        try:
            import numpy as np  # type: ignore[import-not-found]

            from customs_ai_agent.service.document_preview_runtime import _paddleocr3_page_spans

            adapter = self._native_adapter()
            # The existing adapter verifies that the installed P3 runtime can
            # execute offline before any customer raster is passed to it.
            adapter.verify_native_runtime()  # type: ignore[attr-defined]
            engine = adapter._paddle_engine()  # type: ignore[attr-defined]
            spans: list[RasterLedgerSpanV2] = []
            for page in pages:
                if len(page.pixels) != page.width * page.height * 3:
                    raise RasterRuntimeError("raster_baseline_page_pixels_invalid")
                image = np.frombuffer(page.pixels, dtype=np.uint8).reshape(
                    (page.height, page.width, 3)
                )
                with local_files_only():
                    prediction = engine.predict(image, **adapter._prediction_kwargs())  # type: ignore[attr-defined]
                direct_spans = _paddleocr3_page_spans(
                    prediction,
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                )
                spans.extend(
                    RasterLedgerSpanV2(
                        span_id=span.span_id,
                        page_number=page.page_number,
                        ordinal=span.ordinal,
                        reading_order=span.reading_order,
                        text=span.text,
                        bbox=span.bbox,
                        # The legacy direct adapter has no schema-bound P3
                        # confidence mapping.  Zero is conservative and makes
                        # calibration reject it from automatic readiness.
                        confidence=0.0,
                    )
                    for span in direct_spans
                )
        except Exception:
            raise RasterDocumentPreviewUnavailable(
                "PaddleOCR3 direct baseline is unavailable"
            ) from None
        return RasterOcrLayoutLedgerV2(
            source_sha256=source_sha256,
            raster_profile_revision=raster_profile_revision,
            provider_revision=self.profile.extractor_revision,
            pages=tuple(pages),
            spans=tuple(spans),
            document_type="unknown",
            classification_confidence=None,
            warnings=(RasterExtractionWarningV2(code="paddleocr3_direct_ocr_baseline"),),
            needs_review_reasons=(RasterExtractionWarningV2(code="baseline_layout_not_available"),),
        )


@dataclass(frozen=True, slots=True)
class PdfiumRasterInputAdapter(RasterInputAdapter):
    """Concrete strict decoder/PDFium rasterizer used only by an approved profile."""

    pdfium_version: str
    raster_dpi: int
    # Raster v1 is sealed only against single-page synthetic documents.  Do
    # not let an approved single-page profile silently cover multi-page PDFs.
    max_pages: int = 1

    @staticmethod
    def _close_raster_resources(*resources: object | None) -> None:
        """Close short-lived PDFium/Pillow objects before their document.

        ``pypdfium2`` objects keep native handles whose finalizers can run
        after the document/library is already gone.  That is harmless for the
        returned RGB bytes, but it emits native warnings and can retain memory
        across a long-lived OCR worker.  Explicit best-effort close in reverse
        ownership order keeps cleanup local to rasterization without making a
        cleanup failure visible at the document-input boundary.
        """

        closed: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in closed:
                continue
            closed.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _pdfium(self) -> object:
        try:
            installed = importlib.metadata.version("pypdfium2")
            if installed != self.pdfium_version:
                raise RasterDocumentPreviewUnavailable("PDFium package version is not approved")
            import pypdfium2  # type: ignore[import-not-found]
        except RasterDocumentPreviewUnavailable:
            raise
        except Exception:
            raise RasterDocumentPreviewUnavailable("PDFium runtime is unavailable") from None
        return pypdfium2

    def rasterize_pdf(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> Sequence[RasterizedPageV2]:
        if raster_input.metadata.media_type != "application/pdf":
            raise RasterDocumentPreviewInputError("PDFium input media type is invalid")
        pdfium = self._pdfium()
        document: Any = None
        try:
            with local_files_only():
                document = pdfium.PdfDocument(raster_input.content)  # type: ignore[attr-defined]
                page_count = len(document)
                if page_count < 1:
                    raise RasterDocumentPreviewInputError("PDF raster page limit is invalid")
                if page_count > self.max_pages:
                    raise RasterDocumentPreviewUnsupported("multi_page_pdf_unsupported")
                if structure.pdf_page_hint is not None and structure.pdf_page_hint > page_count:
                    raise RasterDocumentPreviewInputError("PDF raster omitted admitted pages")
                pages: list[RasterizedPageV2] = []
                for page_number in range(1, page_count + 1):
                    page: Any = None
                    bitmap: Any = None
                    source_image: Any = None
                    image: Any = None
                    try:
                        page = document[page_number - 1]
                        bitmap = page.render(scale=self.raster_dpi / 72)
                        source_image = bitmap.to_pil()
                        image = source_image.convert("RGB")
                        width, height = image.size
                        if width < 1 or height < 1 or width * height > 40_000_000:
                            raise RasterDocumentPreviewInputError(
                                "PDF raster dimensions are invalid"
                            )
                        pages.append(
                            RasterizedPageV2(
                                page_number=page_number,
                                width=width,
                                height=height,
                                pixels=image.tobytes(),
                            )
                        )
                    finally:
                        self._close_raster_resources(image, source_image, bitmap, page)
        except RasterDocumentPreviewInputError:
            raise
        except Exception:
            raise RasterDocumentPreviewInputError("PDF raster decoding failed") from None
        finally:
            self._close_raster_resources(document)
        return tuple(pages)

    def rasterize_pdf_page(
        self,
        raster_input: RasterPreviewInputV2,
        structure: DocumentStructure,
        *,
        page_number: int,
    ) -> RasterizedPageV2:
        """Rasterize one approved PDF page without materializing the others."""

        if raster_input.metadata.media_type != "application/pdf" or page_number < 1:
            raise RasterDocumentPreviewInputError("PDF raster page request is invalid")
        pdfium = self._pdfium()
        document: Any = None
        page: Any = None
        bitmap: Any = None
        source_image: Any = None
        image: Any = None
        try:
            with local_files_only():
                document = pdfium.PdfDocument(raster_input.content)  # type: ignore[attr-defined]
                page_count = len(document)
                if page_count < 1 or page_number > page_count:
                    raise RasterDocumentPreviewInputError("PDF raster page request is invalid")
                if page_count > self.max_pages:
                    raise RasterDocumentPreviewUnsupported("multi_page_pdf_unsupported")
                if structure.pdf_page_hint is not None and structure.pdf_page_hint > page_count:
                    raise RasterDocumentPreviewInputError("PDF raster omitted admitted pages")
                page = document[page_number - 1]
                bitmap = page.render(scale=self.raster_dpi / 72)
                source_image = bitmap.to_pil()
                image = source_image.convert("RGB")
                width, height = image.size
                if width < 1 or height < 1 or width * height > 40_000_000:
                    raise RasterDocumentPreviewInputError("PDF raster dimensions are invalid")
                return RasterizedPageV2(
                    page_number=page_number,
                    width=width,
                    height=height,
                    pixels=image.tobytes(),
                )
        except RasterDocumentPreviewInputError:
            raise
        except Exception:
            raise RasterDocumentPreviewInputError("PDF raster decoding failed") from None
        finally:
            self._close_raster_resources(image, source_image, bitmap, page, document)

    def decode_image(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> RasterizedPageV2:
        media_type = raster_input.metadata.media_type
        expected_format = {"image/jpeg": "JPEG", "image/png": "PNG"}.get(media_type)
        if (
            expected_format is None
            or structure.image_width is None
            or structure.image_height is None
        ):
            raise RasterDocumentPreviewInputError("raster image media structure is invalid")
        try:
            from PIL import Image

            # Verify first, then reopen and fully load.  This catches damaged
            # compressed payloads that a magic/dimension-only admission cannot.
            with Image.open(io.BytesIO(raster_input.content)) as probe:
                if probe.format != expected_format:
                    raise RasterDocumentPreviewInputError("raster image format is invalid")
                probe.verify()
            with Image.open(io.BytesIO(raster_input.content)) as image:
                if image.format != expected_format:
                    raise RasterDocumentPreviewInputError("raster image format is invalid")
                image.load()
                if image.size != (structure.image_width, structure.image_height):
                    raise RasterDocumentPreviewInputError("raster image dimensions drifted")
                decoded = image.convert("RGB")
                return RasterizedPageV2(
                    page_number=1,
                    width=decoded.width,
                    height=decoded.height,
                    pixels=decoded.tobytes(),
                )
        except RasterDocumentPreviewInputError:
            raise
        except Exception:
            raise RasterDocumentPreviewInputError("raster image decoding failed") from None


@dataclass(frozen=True, slots=True)
class RasterViewerPageV1:
    """Ephemeral PNG for a reviewer-only backend proxy, never a public ledger.

    The bytes are encoded from the exact RGB page pixels presented to the OCR
    provider, so the frontend can use the same pixel coordinate system as the
    page/span/table overlay.  Callers must keep this object in memory and send
    it only through an authenticated ``Cache-Control: no-store`` backend
    viewer; this Agent module deliberately exposes no viewer HTTP route.
    """

    page_number: int
    width: int
    height: int
    media_type: Literal["image/png"]
    content: bytes
    content_sha256: str
    source_sha256: str
    raster_profile_revision: str
    renderer_revision: str


@dataclass(frozen=True, slots=True)
class ApprovedPdfiumRasterPageRenderer:
    """Approved-only PDF page renderer for the Backend reviewer endpoint."""

    input_adapter: PdfiumRasterInputAdapter
    raster_profile_revision: str
    renderer_revision: str
    runtime_verifier: Callable[[], None] | None = None

    def _render_page(
        self,
        page: RasterizedPageV2,
        raster_input: RasterPreviewInputV2,
    ) -> RasterViewerPageV1:
        from PIL import Image

        if len(page.pixels) != page.width * page.height * 3:
            raise RasterDocumentPreviewUnavailable("raster page viewer pixels are invalid")
        image = Image.frombytes("RGB", (page.width, page.height), page.pixels)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG", optimize=False)
        content = encoded.getvalue()
        if not content or len(content) > 32 * 1024 * 1024:
            raise RasterDocumentPreviewUnavailable("raster page viewer output is invalid")
        return RasterViewerPageV1(
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            media_type="image/png",
            content=content,
            content_sha256=hashlib.sha256(content).hexdigest(),
            source_sha256=raster_input.metadata.source_sha256,
            raster_profile_revision=self.raster_profile_revision,
            renderer_revision=self.renderer_revision,
        )

    def _require_approved_pdf_input(self, raster_input: RasterPreviewInputV2) -> None:
        if (
            raster_input.metadata.media_type != "application/pdf"
            or raster_input.metadata.raster_profile_revision != self.raster_profile_revision
        ):
            raise RasterDocumentPreviewUnavailable("raster page viewer profile is not approved")
        if self.runtime_verifier is not None:
            try:
                self.runtime_verifier()
            except RasterRuntimeError as error:
                raise RasterDocumentPreviewUnavailable(
                    "raster page viewer is unavailable"
                ) from error

    def render_pdf_page(
        self,
        raster_input: RasterPreviewInputV2,
        structure: DocumentStructure,
        *,
        page_number: int,
    ) -> RasterViewerPageV1:
        """Render exactly one PDF page for the reviewer-only proxy path."""

        self._require_approved_pdf_input(raster_input)
        try:
            page = self.input_adapter.rasterize_pdf_page(
                raster_input, structure, page_number=page_number
            )
            return self._render_page(page, raster_input)
        except RasterDocumentPreviewUnavailable:
            raise
        except RasterDocumentPreviewInputError:
            raise
        except Exception:
            raise RasterDocumentPreviewUnavailable("raster page viewer is unavailable") from None

    def render_pdf_pages(
        self,
        raster_input: RasterPreviewInputV2,
        structure: DocumentStructure,
    ) -> tuple[RasterViewerPageV1, ...]:
        self._require_approved_pdf_input(raster_input)
        try:
            pages = self.input_adapter.rasterize_pdf(raster_input, structure)
            return tuple(self._render_page(page, raster_input) for page in pages)
        except RasterDocumentPreviewUnavailable:
            raise
        except RasterDocumentPreviewInputError:
            raise
        except Exception:
            raise RasterDocumentPreviewUnavailable("raster page viewer is unavailable") from None


@dataclass(frozen=True, slots=True)
class RasterPreviewEngineResolution:
    engine: LocalRasterDocumentPreviewEngine
    status: Literal["disabled", "approved", "unavailable"]
    reason_code: str | None = None
    page_renderer: ApprovedPdfiumRasterPageRenderer | None = None
    raster_profile_revision: str | None = None


@dataclass(frozen=True, slots=True)
class RasterBaselineEngineResolutionV1:
    """Evaluation-only direct-P3 engine resolution; it has no approval path."""

    engine: LocalRasterDocumentPreviewEngine | None
    status: Literal["disabled", "available", "unavailable"]
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class RasterCandidateEngineResolutionV1:
    """A locally pinned PP-StructureV3 engine for offline evaluation only.

    This deliberately has no approval-pointer input and cannot be used by the
    service startup path.  It exists so the synthetic-raster evaluator runs
    the same strict decoder/provider/sidecar boundary *before* a sealed
    result exists.  The caller receives the actual configuration digest rather
    than an operator-supplied label, which is later bound into the candidate
    execution receipt.
    """

    engine: LocalRasterDocumentPreviewEngine | None
    status: Literal["available", "unavailable"]
    runtime_configuration_sha256: str | None = None
    profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2 | None = None
    reason_code: str | None = None


def _read_local_file(path: Path, *, limit: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise RasterRuntimeError("raster_runtime_sidecar_invalid")
        value = path.read_bytes()
    except RasterRuntimeError:
        raise
    except OSError:
        raise RasterRuntimeError("raster_runtime_sidecar_invalid") from None
    if not value or len(value) > limit:
        raise RasterRuntimeError("raster_runtime_sidecar_invalid")
    return value


def _sidecar_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value or ":" in value:
        raise RasterRuntimeError("raster_runtime_sidecar_invalid")
    resolved_root = root.resolve(strict=False)
    candidate = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise RasterRuntimeError("raster_runtime_sidecar_invalid") from None
    return _windows_operational_sidecar_path(candidate)


def _document_preview_root(path: Path) -> Path:
    try:
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        return root
    except RasterRuntimeError:
        raise
    except OSError:
        raise RasterRuntimeError("raster_runtime_configuration_invalid") from None


def _windows_operational_sidecar_path(path: Path) -> Path:
    """Return a validated short Windows alias for deep sidecar package paths."""

    if os.name != "nt":
        return path
    try:
        import ctypes

        value = str(path)
        if value.isascii() and len(value) <= 220:
            return path
        kernel32 = ctypes.windll.kernel32
        required = kernel32.GetShortPathNameW(value, None, 0)
        if required <= 0:
            raise OSError("short path unavailable")
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = kernel32.GetShortPathNameW(value, buffer, len(buffer))
        short_value = buffer.value if 0 < written < len(buffer) else ""
    except Exception:
        raise RasterRuntimeError("raster_runtime_windows_path_unsupported") from None
    if not short_value or not short_value.isascii():
        raise RasterRuntimeError("raster_runtime_windows_path_unsupported")
    return Path(short_value)


def _canonical_raster_json_bytes(value: object) -> bytes:
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


def _strict_raster_json_object(raw: bytes, *, code: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RasterRuntimeError(code) from None
    if not isinstance(value, dict):
        raise RasterRuntimeError(code)
    return value


def validate_raster_approval_pointer_path(path: Path) -> tuple[Path, Path]:
    """Return the fixed local raster-pointer path without following links.

    The settings parser supplies the lexical constraint, but the automatic
    loader repeats it before every read because an activation may arrive after
    process startup.  A missing pointer remains a normal unavailable state;
    a link or path escape is not accepted as an alternative local source.
    """

    if (
        not path.is_absolute()
        or path.name != "raster-preview-active.json"
        or path.parent.name != "approvals"
        or tuple(part.casefold() for part in path.parent.parent.parts[-3:])
        != ("_local", "data", "document-preview")
    ):
        raise RasterRuntimeError("raster_auto_pointer_invalid")
    root = _document_preview_root(path.parent.parent)
    expected = root / "approvals" / "raster-preview-active.json"
    try:
        if path.is_symlink() or path.parent.is_symlink() or path.resolve(strict=False) != expected:
            raise RasterRuntimeError("raster_auto_pointer_invalid")
    except OSError:
        raise RasterRuntimeError("raster_auto_pointer_invalid") from None
    return root, expected


def parse_source_attested_synthetic_raster_activation(
    raw: bytes,
) -> RasterSourceAttestedActivationPointerV2:
    """Parse only the canonical non-human v2 raster activation pointer."""

    code = "raster_source_attested_activation_invalid"
    value = _strict_raster_json_object(raw, code=code)
    if raw != _canonical_raster_json_bytes(value):
        raise RasterRuntimeError(code)
    try:
        return RasterSourceAttestedActivationPointerV2.model_validate(value)
    except ValidationError:
        raise RasterRuntimeError(code) from None


def _verify_package_inventory(
    *,
    raw: bytes,
    expected_sha256: str,
    required_packages: Sequence[str],
    verify_installed: bool = True,
    package_version_reader: Callable[[str], str] | None = None,
) -> None:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RasterRuntimeError("raster_package_inventory_invalid")
    try:
        payload = json.loads(raw)
        packages = payload["packages"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise RasterRuntimeError("raster_package_inventory_invalid") from None
    if not isinstance(packages, list):
        raise RasterRuntimeError("raster_package_inventory_invalid")
    observed: dict[str, str] = {}
    for item in packages:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or not item["name"]
            or not item["version"]
            or item["name"] in observed
        ):
            raise RasterRuntimeError("raster_package_inventory_invalid")
        observed[item["name"]] = item["version"]
    if not set(required_packages) <= set(observed):
        raise RasterRuntimeError("raster_package_inventory_invalid")
    if verify_installed:
        version_reader = package_version_reader or importlib.metadata.version
        for package_name in required_packages:
            try:
                if version_reader(package_name) != observed[package_name]:
                    raise RasterRuntimeError("raster_package_inventory_drift")
            except importlib.metadata.PackageNotFoundError:
                raise RasterRuntimeError("raster_package_inventory_drift") from None


def _verify_model_inventory(*, raw: bytes, expected_sha256: str, model_root: Path) -> None:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RasterRuntimeError("raster_model_inventory_invalid")
    try:
        payload = json.loads(raw)
        artifacts = payload["artifacts"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise RasterRuntimeError("raster_model_inventory_invalid") from None
    if not isinstance(artifacts, list) or not artifacts:
        raise RasterRuntimeError("raster_model_inventory_invalid")
    seen: set[str] = set()
    try:
        root = model_root.resolve(strict=True)
        if model_root.is_symlink() or not root.is_dir():
            raise RasterRuntimeError("raster_model_inventory_invalid")
        operational_root = _windows_operational_sidecar_path(root)
    except RasterRuntimeError:
        raise
    except OSError:
        raise RasterRuntimeError("raster_model_inventory_invalid") from None
    for item in artifacts:
        if not isinstance(item, dict):
            raise RasterRuntimeError("raster_model_inventory_invalid")
        relative = item.get("relative_path")
        expected = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
            or relative in seen
        ):
            raise RasterRuntimeError("raster_model_inventory_invalid")
        seen.add(relative)
        path = _sidecar_path(root, relative)
        try:
            if path.is_symlink() or not path.is_file():
                raise RasterRuntimeError("raster_model_inventory_invalid")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RasterRuntimeError("raster_model_inventory_drift")
        except RasterRuntimeError:
            raise
        except OSError:
            raise RasterRuntimeError("raster_model_inventory_invalid") from None
    # PP-StructureV3 receives a directory, not individual model file handles.
    # An undeclared file could therefore change the native runtime even when
    # every listed artifact still hashes correctly.  Bind the complete regular
    # file set and reject link/special-file traversal rather than accepting a
    # partial manifest.
    try:
        actual: set[str] = set()
        for path in operational_root.rglob("*"):
            if path.is_symlink():
                raise RasterRuntimeError("raster_model_inventory_invalid")
            if path.is_file():
                actual.add(path.relative_to(operational_root).as_posix())
            elif not path.is_dir():
                raise RasterRuntimeError("raster_model_inventory_invalid")
        if actual != seen:
            raise RasterRuntimeError("raster_model_inventory_drift")
    except RasterRuntimeError:
        raise
    except OSError:
        raise RasterRuntimeError("raster_model_inventory_invalid") from None


@dataclass(frozen=True, slots=True)
class _PpStructureV3V2RuntimeMaterials:
    """Verified local v2 material set shared by evaluation and auto serving."""

    profile: PpStructureV3OfflineProfileV2
    input_adapter: PdfiumRasterInputAdapter
    executor: PpStructureV3CommandExecutorV2


def _load_ppstructurev3_v2_runtime_materials(
    *, root: Path, configuration_path: Path, configuration_raw: bytes, expected_sha256: str
) -> _PpStructureV3V2RuntimeMaterials:
    """Validate split inventories without importing sidecar packages in parent."""

    if hashlib.sha256(configuration_raw).hexdigest() != expected_sha256:
        raise RasterRuntimeError("raster_runtime_configuration_invalid")
    try:
        profile = PpStructureV3OfflineProfileV2.model_validate_json(configuration_raw)
    except ValidationError:
        raise RasterRuntimeError("raster_runtime_configuration_invalid") from None
    if profile.bundle_root_relative_path is not None:
        # Pydantic already applies the same invariant, but repeat it at the
        # material-loading boundary so a future parser/schema change cannot
        # turn this hash-bound path into a generic local directory selector.
        if (
            profile.bundle_manifest_sha256 is None
            or not _is_canonical_raster_bundle_release_binding(
                bundle_root_relative_path=profile.bundle_root_relative_path,
                bundle_manifest_sha256=profile.bundle_manifest_sha256,
            )
        ):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        bundle_manifest_path = _sidecar_path(
            root, f"{profile.bundle_root_relative_path}/manifest.json"
        )
        bundle_manifest_raw = _read_local_file(
            bundle_manifest_path,
            limit=_MAX_OFFLINE_BUNDLE_MANIFEST_BYTES,
        )
        if hashlib.sha256(bundle_manifest_raw).hexdigest() != profile.bundle_manifest_sha256:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
    profile.model_root_relative_path()
    for relative in (
        profile.parent_renderer_package_inventory_path,
        profile.sidecar_package_inventory_path,
        profile.model_inventory_path,
        profile.sidecar_interpreter_root,
        profile.sidecar_interpreter_path,
        profile.sidecar_module_root,
        profile.sidecar_command_path,
        profile.sidecar_module_path,
        profile.sidecar_configuration_path,
    ):
        profile.sidecar_relative_path(relative)
    parent_inventory_path = _sidecar_path(root, profile.parent_renderer_package_inventory_path)
    parent_inventory_raw = _read_local_file(parent_inventory_path, limit=_MAX_SIDECAR_BYTES)
    _verify_package_inventory(
        raw=parent_inventory_raw,
        expected_sha256=profile.parent_renderer_package_inventory_sha256,
        required_packages=profile.parent_renderer_required_packages,
    )
    sidecar_inventory_path = _sidecar_path(root, profile.sidecar_package_inventory_path)
    sidecar_inventory_raw = _read_local_file(sidecar_inventory_path, limit=_MAX_SIDECAR_BYTES)
    _verify_package_inventory(
        raw=sidecar_inventory_raw,
        expected_sha256=profile.sidecar_package_inventory_sha256,
        required_packages=profile.sidecar_required_packages,
        verify_installed=False,
    )
    model_root = _sidecar_path(root, profile.local_model_root)
    if model_root.is_symlink() or not model_root.is_dir():
        raise RasterRuntimeError("raster_model_inventory_invalid")
    model_inventory_path = _sidecar_path(root, profile.model_inventory_path)
    model_inventory_raw = _read_local_file(model_inventory_path, limit=_MAX_SIDECAR_BYTES)
    _verify_model_inventory(
        raw=model_inventory_raw,
        expected_sha256=profile.model_inventory_sha256,
        model_root=model_root,
    )
    sidecar_configuration_path = _sidecar_path(root, profile.sidecar_configuration_path)
    sidecar_configuration_raw = _read_local_file(
        sidecar_configuration_path, limit=_MAX_SIDECAR_BYTES
    )
    if (
        hashlib.sha256(sidecar_configuration_raw).hexdigest()
        != profile.sidecar_configuration_sha256
    ):
        raise RasterRuntimeError("raster_runtime_configuration_invalid")
    try:
        sidecar_configuration = PpStructureV3SidecarConfigurationV1.model_validate_json(
            sidecar_configuration_raw
        )
    except ValidationError:
        raise RasterRuntimeError("raster_runtime_configuration_invalid") from None
    if (
        sidecar_configuration.extractor_revision != profile.extractor_revision
        or sidecar_configuration.raster_profile_revision != profile.raster_profile_revision
        or sidecar_configuration.evaluation_batch_size != profile.evaluation_batch_size
    ):
        raise RasterRuntimeError("raster_runtime_configuration_invalid")
    try:
        sidecar_configuration.validate_model_directories(model_root)
    except PpStructureV3SidecarError:
        raise RasterRuntimeError("raster_model_inventory_invalid") from None
    interpreter_root = _sidecar_path(root, profile.sidecar_interpreter_root)
    interpreter_path = _sidecar_path(root, profile.sidecar_interpreter_path)
    sidecar_module_root = _sidecar_path(root, profile.sidecar_module_root)
    sidecar_command_path = _sidecar_path(root, profile.sidecar_command_path)
    sidecar_module_path = _sidecar_path(root, profile.sidecar_module_path)
    _validate_controlled_child_file(
        root_path=interpreter_root,
        file_path=interpreter_path,
    )
    _validate_controlled_child_file(
        root_path=sidecar_module_root,
        file_path=sidecar_command_path,
    )
    _validate_controlled_child_file(
        root_path=sidecar_module_root,
        file_path=sidecar_module_path,
    )
    if _controlled_directory_tree_sha256(interpreter_root) != (
        profile.sidecar_interpreter_root_tree_sha256
    ):
        raise RasterRuntimeError("raster_native_sidecar_drift")
    if _controlled_module_root_tree_sha256(sidecar_module_root) != (
        profile.sidecar_module_root_tree_sha256
    ):
        raise RasterRuntimeError("raster_native_sidecar_drift")
    if _sha256_regular_file(sidecar_command_path) != profile.sidecar_command_sha256:
        raise RasterRuntimeError("raster_native_sidecar_drift")
    if _sha256_regular_file(sidecar_module_path) != profile.sidecar_module_sha256:
        raise RasterRuntimeError("raster_native_sidecar_drift")
    if _sha256_regular_file(interpreter_path) != profile.sidecar_interpreter_sha256:
        raise RasterRuntimeError("raster_native_sidecar_drift")
    executor = PpStructureV3CommandExecutorV2(
        interpreter_path=interpreter_path,
        interpreter_sha256=profile.sidecar_interpreter_sha256,
        interpreter_root=interpreter_root,
        interpreter_root_tree_sha256=profile.sidecar_interpreter_root_tree_sha256,
        timeout_seconds=profile.sidecar_timeout_seconds,
        configuration_path=sidecar_configuration_path,
        configuration_sha256=profile.sidecar_configuration_sha256,
        model_root=model_root,
        parent_renderer_package_inventory_path=parent_inventory_path,
        parent_renderer_package_inventory_sha256=(profile.parent_renderer_package_inventory_sha256),
        parent_renderer_required_packages=tuple(profile.parent_renderer_required_packages),
        sidecar_package_inventory_path=sidecar_inventory_path,
        sidecar_package_inventory_sha256=profile.sidecar_package_inventory_sha256,
        sidecar_required_packages=tuple(profile.sidecar_required_packages),
        model_inventory_path=model_inventory_path,
        model_inventory_sha256=profile.model_inventory_sha256,
        sidecar_module_root=sidecar_module_root,
        sidecar_module_root_tree_sha256=profile.sidecar_module_root_tree_sha256,
        sidecar_command_path=sidecar_command_path,
        sidecar_command_sha256=profile.sidecar_command_sha256,
        sidecar_module_path=sidecar_module_path,
        sidecar_module_sha256=profile.sidecar_module_sha256,
    )
    # The explicit recheck catches a model/package/interpreter/import-root
    # replacement made between the individual reads above and construction.
    executor.verify_runtime_state()
    return _PpStructureV3V2RuntimeMaterials(
        profile=profile,
        input_adapter=PdfiumRasterInputAdapter(profile.pdfium_version, profile.raster_dpi),
        executor=executor,
    )


def _verify_synthetic_raster_receipt(
    *, raw: bytes, expected_sha256: str
) -> "SyntheticRasterReceiptV1":
    """Bind activation to the complete fixed 1,000-family raster receipt."""

    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RasterRuntimeError("synthetic_raster_receipt_invalid")
    try:
        from customs_ai_agent.evaluation.synthetic_raster import (
            RasterAssetReceiptV1,
            SyntheticRasterReceiptV1,
        )

        receipt = SyntheticRasterReceiptV1.model_validate_json(raw)
    except (ImportError, ValidationError, ValueError):
        raise RasterRuntimeError("synthetic_raster_receipt_invalid") from None
    if len(receipt.assets) != 24_000:
        raise RasterRuntimeError("synthetic_raster_receipt_invalid")
    if (
        receipt.source_receipt_relative_path is None
        or receipt.build_inputs_relative_path is None
        or receipt.build_inputs_sha256 is None
    ):
        raise RasterRuntimeError("synthetic_raster_receipt_invalid")
    expected_families = {"train": 750, "development": 100, "calibration": 50, "sealed": 100}
    families_by_split: dict[str, set[str]] = defaultdict(set)
    assets_by_family: dict[str, list[RasterAssetReceiptV1]] = defaultdict(list)
    for asset in receipt.assets:
        families_by_split[asset.split].add(asset.family_id)
        assets_by_family[asset.family_id].append(asset)
    if {split: len(families_by_split[split]) for split in expected_families} != expected_families:
        raise RasterRuntimeError("synthetic_raster_receipt_invalid")
    if sum(len(families) for families in families_by_split.values()) != 1_000:
        raise RasterRuntimeError("synthetic_raster_receipt_invalid")
    for family_assets in assets_by_family.values():
        if len(family_assets) != 24:
            raise RasterRuntimeError("synthetic_raster_receipt_invalid")
        profile_counts = Counter(asset.profile_id for asset in family_assets)
        document_counts = Counter(asset.document_type for asset in family_assets)
        if set(profile_counts) != {"png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"}:
            raise RasterRuntimeError("synthetic_raster_receipt_invalid")
        if any(count != 6 for count in profile_counts.values()) or document_counts != {
            "bill_of_lading": 8,
            "commercial_invoice": 8,
            "packing_list": 8,
        }:
            raise RasterRuntimeError("synthetic_raster_receipt_invalid")
    return receipt


def _verify_synthetic_raster_activation_materials(
    *, root: Path, receipt: "SyntheticRasterReceiptV1"
) -> None:
    """Rebind the sealed/calibration result to its local source and Gold bytes.

    This runs only while an approval pointer is resolved.  Customer OCR calls
    do not reopen the synthetic dataset.  It closes the otherwise unavoidable
    gap between an earlier evaluator run and a later local file replacement.
    """

    try:
        from customs_ai_agent.evaluation.synthetic_raster import (
            SyntheticRasterCompilerUnavailable,
            verify_synthetic_raster_receipt_materials,
        )

        verify_synthetic_raster_receipt_materials(
            document_preview_root=root,
            receipt=receipt,
            required_splits=("calibration", "sealed"),
        )
    except (ImportError, OSError, SyntheticRasterCompilerUnavailable, ValueError):
        raise RasterRuntimeError("synthetic_raster_material_provenance_invalid") from None


def _verify_raster_candidate_provenance(
    *,
    root: Path,
    receipt: "SyntheticRasterReceiptV1",
    receipt_sha256: str,
    split: Literal["calibration", "sealed"],
    profile: PpStructureV3OfflineProfileV1 | PpStructureV3OfflineProfileV2,
    runtime_configuration_sha256: str,
    extractor_revision: str,
    candidate_bundle_relative_path: str,
    candidate_bundle_sha256: str,
    candidate_execution_relative_path: str,
    candidate_execution_sha256: str,
) -> None:
    """Replay the candidate-to-profile binding without rerunning OCR.

    Activation already independently checks package/model/command/sidecar
    drift.  This narrower check proves that the sealed or calibration result
    points at the exact canonical OCR bundle produced for that same profile,
    receipt and split.  It intentionally does not open raster/Gold document
    bytes or emit any OCR text at startup.
    """

    try:
        from customs_ai_agent.evaluation.synthetic_raster_candidates import (
            SyntheticRasterCandidateError,
            read_canonical_raster_candidate_bundle,
            read_canonical_raster_candidate_execution,
            validate_raster_candidate_bundle_against_receipt,
            validate_raster_candidate_execution,
        )
    except ImportError:
        raise RasterRuntimeError("raster_candidate_provenance_invalid") from None
    try:
        bundle, bundle_raw, bundle_sha256 = read_canonical_raster_candidate_bundle(
            document_preview_root=root,
            relative_path=candidate_bundle_relative_path,
            expected_sha256=candidate_bundle_sha256,
        )
        execution, _execution_raw, execution_sha256 = read_canonical_raster_candidate_execution(
            document_preview_root=root,
            relative_path=candidate_execution_relative_path,
            expected_sha256=candidate_execution_sha256,
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
            runtime_configuration_sha256=runtime_configuration_sha256,
        )
    except (SyntheticRasterCandidateError, ValueError):
        raise RasterRuntimeError("raster_candidate_provenance_invalid") from None
    if (
        execution_sha256 != candidate_execution_sha256
        or execution.engine_kind != "ppstructurev3"
        or execution.split != split
        or execution.synthetic_raster_receipt_sha256 != receipt_sha256
        or execution.raster_profile_revision != profile.raster_profile_revision
        or execution.extractor_revision != extractor_revision
        or bundle.engine_kind != "ppstructurev3"
        or bundle.raster_profile_revision != profile.raster_profile_revision
        or bundle.extractor_revision != extractor_revision
        or receipt.pdfium_version != profile.pdfium_version
    ):
        raise RasterRuntimeError("raster_candidate_provenance_invalid")


def resolve_paddleocr3_direct_raster_baseline(
    *,
    document_preview_root: Path,
    configuration_path: Path | None,
    configuration_sha256: str | None,
) -> RasterBaselineEngineResolutionV1:
    """Resolve the pinned direct P3 comparison engine without an approval pointer.

    This deliberately has a different entry point from
    :func:`resolve_raster_document_preview_engine`: callers can run baseline
    evaluations with a local, hash-pinned config, but no valid baseline config
    can flip the service's raster approval state.  It checks the same package,
    model, PDFium and local-path invariants as the PP-StructureV3 candidate.
    """

    if configuration_path is None and configuration_sha256 is None:
        return RasterBaselineEngineResolutionV1(None, "disabled")
    if (
        configuration_path is None
        or configuration_sha256 is None
        or _SHA256.fullmatch(configuration_sha256) is None
    ):
        return RasterBaselineEngineResolutionV1(None, "unavailable", "raster_baseline_invalid")
    try:
        root = _document_preview_root(document_preview_root)
        try:
            if configuration_path.is_symlink():
                raise OSError("baseline configuration link is not allowed")
            resolved_configuration = configuration_path.resolve(strict=True)
            resolved_configuration.relative_to(root)
        except (OSError, ValueError):
            raise RasterRuntimeError("raster_runtime_configuration_invalid") from None
        configuration_raw = _read_local_file(resolved_configuration, limit=_MAX_SIDECAR_BYTES)
        if hashlib.sha256(configuration_raw).hexdigest() != configuration_sha256:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        profile = PaddleOcr3DirectRasterBaselineProfileV1.model_validate_json(configuration_raw)
        if profile.approval_eligible is not False:
            raise RasterRuntimeError("raster_baseline_invalid")
        for relative in (
            profile.package_inventory_path,
            profile.model_inventory_path,
            profile.local_model_root,
            profile.text_detection_model_directory,
            profile.text_recognition_model_directory,
        ):
            profile.local_relative_path(relative)
        package_inventory_raw = _read_local_file(
            _sidecar_path(root, profile.package_inventory_path), limit=_MAX_SIDECAR_BYTES
        )
        _verify_package_inventory(
            raw=package_inventory_raw,
            expected_sha256=profile.package_inventory_sha256,
            required_packages=profile.required_packages,
        )
        model_root = _sidecar_path(root, profile.local_model_root)
        if model_root.is_symlink() or not model_root.is_dir():
            raise RasterRuntimeError("raster_model_inventory_invalid")
        model_inventory_raw = _read_local_file(
            _sidecar_path(root, profile.model_inventory_path), limit=_MAX_SIDECAR_BYTES
        )
        _verify_model_inventory(
            raw=model_inventory_raw,
            expected_sha256=profile.model_inventory_sha256,
            model_root=model_root,
        )
        for relative in (
            profile.text_detection_model_directory,
            profile.text_recognition_model_directory,
        ):
            model_directory = _sidecar_path(model_root, relative)
            if model_directory.is_symlink() or not model_directory.is_dir():
                raise RasterRuntimeError("raster_model_inventory_invalid")
        from customs_ai_agent.service.raster_document_extraction import (
            MediaAwareRasterPreviewEngine,
        )

        return RasterBaselineEngineResolutionV1(
            MediaAwareRasterPreviewEngine(
                PdfiumRasterInputAdapter(profile.pdfium_version, profile.raster_dpi),
                PaddleOcr3DirectRasterBaselineProvider(profile=profile, model_root=model_root),
            ),
            "available",
        )
    except (RasterRuntimeError, ValidationError, UnicodeDecodeError, json.JSONDecodeError):
        return RasterBaselineEngineResolutionV1(None, "unavailable", "raster_baseline_invalid")


def resolve_ppstructurev3_raster_candidate(
    *,
    document_preview_root: Path,
    configuration_relative_path: str,
    executor_factory: Callable[[PpStructureV3OfflineProfileV1], PpStructureV3NativeExecutor]
    | None = None,
) -> RasterCandidateEngineResolutionV1:
    """Resolve one offline PP-StructureV3 evaluation engine without approval.

    The candidate runner invokes this function before it creates a
    content-addressed result bundle.  It validates the same local package,
    model, executable, sidecar-module and sidecar-configuration bindings as
    the serving resolver, but intentionally does not read an approval pointer
    or a calibration/sealed artifact.  Consequently this path can produce
    evidence for a future approval but cannot activate a provider itself.
    """

    try:
        root = _document_preview_root(document_preview_root)
        configuration_path = _sidecar_path(root, configuration_relative_path)
        configuration_raw = _read_local_file(configuration_path, limit=_MAX_SIDECAR_BYTES)
        configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
        configuration_value = _strict_raster_json_object(
            configuration_raw, code="raster_runtime_configuration_invalid"
        )
        if configuration_value.get("schema_version") == "2.0":
            materials = _load_ppstructurev3_v2_runtime_materials(
                root=root,
                configuration_path=configuration_path,
                configuration_raw=configuration_raw,
                expected_sha256=configuration_sha256,
            )
            from customs_ai_agent.service.raster_document_extraction import (
                MediaAwareRasterPreviewEngine,
            )

            return RasterCandidateEngineResolutionV1(
                MediaAwareRasterPreviewEngine(
                    materials.input_adapter,
                    PpStructureV3OfflineProvider(
                        materials.profile, materials.executor, thresholds=None
                    ),
                ),
                "available",
                runtime_configuration_sha256=configuration_sha256,
                profile=materials.profile,
            )
        profile = PpStructureV3OfflineProfileV1.model_validate_json(configuration_raw)
        profile.model_root_relative_path()
        profile.sidecar_relative_path(profile.package_inventory_path)
        profile.sidecar_relative_path(profile.model_inventory_path)
        profile.sidecar_relative_path(profile.local_sidecar_command)
        profile.sidecar_relative_path(profile.sidecar_configuration_path)

        package_inventory_path = _sidecar_path(root, profile.package_inventory_path)
        package_inventory_raw = _read_local_file(package_inventory_path, limit=_MAX_SIDECAR_BYTES)
        _verify_package_inventory(
            raw=package_inventory_raw,
            expected_sha256=profile.package_inventory_sha256,
            required_packages=profile.required_packages,
        )
        model_root = _sidecar_path(root, profile.local_model_root)
        if model_root.is_symlink() or not model_root.is_dir():
            raise RasterRuntimeError("raster_model_inventory_invalid")
        model_inventory_path = _sidecar_path(root, profile.model_inventory_path)
        model_inventory_raw = _read_local_file(model_inventory_path, limit=_MAX_SIDECAR_BYTES)
        _verify_model_inventory(
            raw=model_inventory_raw,
            expected_sha256=profile.model_inventory_sha256,
            model_root=model_root,
        )
        sidecar_configuration_path = _sidecar_path(root, profile.sidecar_configuration_path)
        sidecar_configuration_raw = _read_local_file(
            sidecar_configuration_path, limit=_MAX_SIDECAR_BYTES
        )
        if (
            hashlib.sha256(sidecar_configuration_raw).hexdigest()
            != profile.sidecar_configuration_sha256
        ):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        sidecar_configuration = PpStructureV3SidecarConfigurationV1.model_validate_json(
            sidecar_configuration_raw
        )
        if (
            sidecar_configuration.extractor_revision != profile.extractor_revision
            or sidecar_configuration.raster_profile_revision != profile.raster_profile_revision
        ):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        try:
            sidecar_configuration.validate_model_directories(model_root)
        except PpStructureV3SidecarError:
            raise RasterRuntimeError("raster_model_inventory_invalid") from None
        command_path = _sidecar_path(root, profile.local_sidecar_command)
        if _sha256_regular_file(command_path) != profile.sidecar_command_sha256:
            raise RasterRuntimeError("raster_native_sidecar_drift")
        executor = (
            executor_factory(profile)
            if executor_factory is not None
            else PpStructureV3CommandExecutor(
                command_path=command_path,
                command_sha256=profile.sidecar_command_sha256,
                timeout_seconds=profile.sidecar_timeout_seconds,
                configuration_path=sidecar_configuration_path,
                configuration_sha256=profile.sidecar_configuration_sha256,
                model_root=model_root,
                package_inventory_path=package_inventory_path,
                package_inventory_sha256=profile.package_inventory_sha256,
                model_inventory_path=model_inventory_path,
                model_inventory_sha256=profile.model_inventory_sha256,
                required_packages=tuple(profile.required_packages),
                sidecar_module_sha256=profile.sidecar_module_sha256,
            )
        )
        from customs_ai_agent.service.raster_document_extraction import (
            MediaAwareRasterPreviewEngine,
        )

        return RasterCandidateEngineResolutionV1(
            MediaAwareRasterPreviewEngine(
                PdfiumRasterInputAdapter(profile.pdfium_version, profile.raster_dpi),
                PpStructureV3OfflineProvider(profile, executor, thresholds=None),
            ),
            "available",
            runtime_configuration_sha256=configuration_sha256,
            profile=profile,
        )
    except (
        RasterRuntimeError,
        PpStructureV3SidecarError,
        ValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return RasterCandidateEngineResolutionV1(
            None, "unavailable", reason_code="raster_candidate_profile_invalid"
        )


def resolve_raster_document_preview_engine(
    *,
    approval_path: Path | None,
    approval_sha256: str | None,
    executor_factory: Callable[[PpStructureV3OfflineProfileV1], PpStructureV3NativeExecutor]
    | None = None,
    receipt_material_verifier: Callable[..., None] | None = None,
) -> RasterPreviewEngineResolution:
    """Resolve an approved local sidecar only; defaults stay disabled.

    The normal service creates the hash-pinned command executor itself after
    all approval bindings pass.  ``executor_factory`` remains a narrowly
    scoped test/provisioning seam; it is not required for normal activation.
    ``receipt_material_verifier`` is likewise a test-only seam.  The default
    always re-hashes the local source/build/raster/Gold materials that back the
    calibration and sealed result before a profile becomes approved.
    """

    disabled = RasterPreviewEngineResolution(DisabledRasterDocumentPreviewEngine(), "disabled")
    if approval_path is None and approval_sha256 is None:
        return disabled
    if (
        approval_path is None
        or approval_sha256 is None
        or _SHA256.fullmatch(approval_sha256) is None
    ):
        return RasterPreviewEngineResolution(
            DisabledRasterDocumentPreviewEngine(), "unavailable", "raster_approval_invalid"
        )
    return _resolve_legacy_raster_document_preview_engine_body(
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        executor_factory=executor_factory,
        receipt_material_verifier=receipt_material_verifier,
    )


@dataclass(frozen=True, slots=True)
class SourceAttestedRasterPreviewBindingV2:
    """One immutable engine/renderer/profile snapshot selected from a v2 pointer."""

    root: Path
    approval_path: Path
    approval_sha256: str
    pointer: RasterSourceAttestedActivationPointerV2
    profile: PpStructureV3OfflineProfileV2
    calibration: RasterCalibrationThresholdReceiptV2
    evaluation: RasterSealedEvaluationResultV2
    engine: LocalRasterDocumentPreviewEngine
    page_renderer: ApprovedPdfiumRasterPageRenderer
    executor: PpStructureV3NativeExecutor
    runtime_verifier: Callable[[], None] | None = None
    runtime_closer: Callable[[], None] | None = None

    @property
    def raster_profile_revision(self) -> str:
        return self.profile.raster_profile_revision

    def verify_runtime_state(self) -> None:
        """Fail closed when the binding's approved runtime is no longer healthy."""

        if self.runtime_verifier is not None:
            self.runtime_verifier()
            return
        verifier = getattr(self.executor, "verify_runtime_state", None)
        if not callable(verifier):
            raise RasterRuntimeError("raster_native_sidecar_unavailable")
        verifier()

    def close(self) -> None:
        """Retire a warm service child when its pointer stops being current."""

        if self.runtime_closer is not None:
            self.runtime_closer()


def _read_source_attested_synthetic_raster_pointer_snapshot(
    approval_path: Path,
) -> tuple[Path, Path, bytes, str, RasterSourceAttestedActivationPointerV2]:
    root, canonical_path = validate_raster_approval_pointer_path(approval_path)
    raw = _read_local_file(canonical_path, limit=_MAX_SIDECAR_BYTES)
    pointer = parse_source_attested_synthetic_raster_activation(raw)
    return root, canonical_path, raw, hashlib.sha256(raw).hexdigest(), pointer


def _load_source_attested_synthetic_raster_binding_from_snapshot(
    *,
    root: Path,
    approval_path: Path,
    approval_raw: bytes,
    approval_sha256: str,
    receipt_material_verifier: Callable[..., None] | None = None,
    start_service_session: bool = True,
) -> SourceAttestedRasterPreviewBindingV2:
    """Load one fully verified v2 automatic pointer snapshot.

    This is deliberately separate from the preserved static v1 resolver.  A
    source-attested v2 pointer cannot be mistaken for a static pointer, and
    no human approval shape can enter this automatic path.
    """

    if (
        _SHA256.fullmatch(approval_sha256) is None
        or hashlib.sha256(approval_raw).hexdigest() != approval_sha256
    ):
        raise RasterRuntimeError("raster_source_attested_activation_invalid")
    pointer = parse_source_attested_synthetic_raster_activation(approval_raw)
    configuration_path = _sidecar_path(root, pointer.runtime_configuration_path)
    configuration_raw = _read_local_file(configuration_path, limit=_MAX_SIDECAR_BYTES)
    materials = _load_ppstructurev3_v2_runtime_materials(
        root=root,
        configuration_path=configuration_path,
        configuration_raw=configuration_raw,
        expected_sha256=pointer.runtime_configuration_sha256,
    )
    profile = materials.profile
    if (
        profile.raster_profile_revision != pointer.raster_profile_revision
        or profile.bundle_root_relative_path != pointer.bundle_root_relative_path
        or profile.bundle_manifest_sha256 != pointer.bundle_manifest_sha256
    ):
        raise RasterRuntimeError("raster_approval_profile_mismatch")

    evaluation_path = _sidecar_path(root, pointer.sealed_evaluation_path)
    evaluation_raw = _read_local_file(evaluation_path, limit=_MAX_SIDECAR_BYTES)
    if hashlib.sha256(evaluation_raw).hexdigest() != pointer.sealed_evaluation_sha256:
        raise RasterRuntimeError("raster_sealed_evaluation_invalid")
    try:
        evaluation = RasterSealedEvaluationResultV2.model_validate_json(evaluation_raw)
    except ValidationError:
        raise RasterRuntimeError("raster_sealed_evaluation_invalid") from None
    if (
        evaluation.raster_profile_revision != profile.raster_profile_revision
        or evaluation.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
        or evaluation.parent_renderer_package_inventory_sha256
        != profile.parent_renderer_package_inventory_sha256
        or evaluation.sidecar_package_inventory_sha256 != profile.sidecar_package_inventory_sha256
        or evaluation.model_inventory_sha256 != profile.model_inventory_sha256
        or evaluation.extractor_revision != profile.extractor_revision
    ):
        raise RasterRuntimeError("raster_sealed_evaluation_not_passing")

    calibration_path = _sidecar_path(root, evaluation.calibration_threshold_receipt_path)
    calibration_raw = _read_local_file(calibration_path, limit=_MAX_SIDECAR_BYTES)
    if (
        hashlib.sha256(calibration_raw).hexdigest()
        != evaluation.calibration_threshold_receipt_sha256
    ):
        raise RasterRuntimeError("raster_calibration_threshold_invalid")
    try:
        calibration = RasterCalibrationThresholdReceiptV2.model_validate_json(calibration_raw)
    except ValidationError:
        raise RasterRuntimeError("raster_calibration_threshold_invalid") from None
    if (
        calibration.raster_profile_revision != profile.raster_profile_revision
        or calibration.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
        or calibration.parent_renderer_package_inventory_sha256
        != profile.parent_renderer_package_inventory_sha256
        or calibration.sidecar_package_inventory_sha256 != profile.sidecar_package_inventory_sha256
        or calibration.model_inventory_sha256 != profile.model_inventory_sha256
        or calibration.extractor_revision != profile.extractor_revision
        or calibration.calibration_case_manifest_sha256
        != evaluation.calibration_case_manifest_sha256
        or calibration.synthetic_raster_receipt_path != evaluation.synthetic_raster_receipt_path
        or calibration.synthetic_raster_receipt_sha256 != evaluation.synthetic_raster_receipt_sha256
    ):
        raise RasterRuntimeError("raster_calibration_threshold_invalid")

    synthetic_receipt_path = _sidecar_path(root, calibration.synthetic_raster_receipt_path)
    synthetic_receipt_raw = _read_local_file(
        synthetic_receipt_path, limit=_MAX_RASTER_RECEIPT_BYTES
    )
    synthetic_receipt = _verify_synthetic_raster_receipt(
        raw=synthetic_receipt_raw,
        expected_sha256=calibration.synthetic_raster_receipt_sha256,
    )
    material_verifier = receipt_material_verifier or _verify_synthetic_raster_activation_materials
    try:
        material_verifier(root=root, receipt=synthetic_receipt)
    except RasterRuntimeError:
        raise
    except Exception:
        raise RasterRuntimeError("synthetic_raster_material_provenance_invalid") from None
    _verify_raster_candidate_provenance(
        root=root,
        receipt=synthetic_receipt,
        receipt_sha256=calibration.synthetic_raster_receipt_sha256,
        split="calibration",
        profile=profile,
        runtime_configuration_sha256=pointer.runtime_configuration_sha256,
        extractor_revision=profile.extractor_revision,
        candidate_bundle_relative_path=calibration.candidate_bundle_relative_path,
        candidate_bundle_sha256=calibration.candidate_bundle_sha256,
        candidate_execution_relative_path=calibration.candidate_execution_relative_path,
        candidate_execution_sha256=calibration.candidate_execution_sha256,
    )
    _verify_raster_candidate_provenance(
        root=root,
        receipt=synthetic_receipt,
        receipt_sha256=calibration.synthetic_raster_receipt_sha256,
        split="sealed",
        profile=profile,
        runtime_configuration_sha256=pointer.runtime_configuration_sha256,
        extractor_revision=profile.extractor_revision,
        candidate_bundle_relative_path=evaluation.candidate_bundle_relative_path,
        candidate_bundle_sha256=evaluation.candidate_bundle_sha256,
        candidate_execution_relative_path=evaluation.candidate_execution_relative_path,
        candidate_execution_sha256=evaluation.candidate_execution_sha256,
    )
    from customs_ai_agent.service.raster_document_extraction import MediaAwareRasterPreviewEngine

    native_executor: PpStructureV3NativeExecutor = materials.executor
    runtime_verifier: Callable[[], None] = materials.executor.verify_runtime_state
    runtime_closer: Callable[[], None] | None = None
    if start_service_session:
        service_session = materials.executor.start_service_session(profile=profile)
        native_executor = service_session
        runtime_verifier = service_session.verify_runtime_state
        runtime_closer = service_session.close
    provider = PpStructureV3OfflineProvider(profile, native_executor, calibration.thresholds)
    engine = MediaAwareRasterPreviewEngine(materials.input_adapter, provider)
    renderer = ApprovedPdfiumRasterPageRenderer(
        input_adapter=materials.input_adapter,
        raster_profile_revision=profile.raster_profile_revision,
        renderer_revision=f"pdfium-{profile.pdfium_version}-dpi-{profile.raster_dpi}",
        runtime_verifier=runtime_verifier,
    )
    return SourceAttestedRasterPreviewBindingV2(
        root=root,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        pointer=pointer,
        profile=profile,
        calibration=calibration,
        evaluation=evaluation,
        engine=engine,
        page_renderer=renderer,
        executor=native_executor,
        runtime_verifier=runtime_verifier,
        runtime_closer=runtime_closer,
    )


def load_source_attested_synthetic_raster_binding(
    *,
    approval_path: Path,
    receipt_material_verifier: Callable[..., None] | None = None,
) -> SourceAttestedRasterPreviewBindingV2:
    """Load the current fixed-path automatic pointer and all its evidence."""

    root, canonical_path, raw, digest, _pointer = (
        _read_source_attested_synthetic_raster_pointer_snapshot(approval_path)
    )
    return _load_source_attested_synthetic_raster_binding_from_snapshot(
        root=root,
        approval_path=canonical_path,
        approval_raw=raw,
        approval_sha256=digest,
        receipt_material_verifier=receipt_material_verifier,
    )


def _atomic_replace_raster_activation_pointer(*, destination: Path, payload: bytes) -> None:
    """Replace the one local pointer only after its full chain was validated."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink():
            raise OSError("approval directory link is not allowed")
        temporary = destination.parent / (
            f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError:
        raise RasterRuntimeError("raster_source_attested_activation_write_failed") from None
    finally:
        try:
            if "temporary" in locals() and temporary.exists():
                temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_source_attested_synthetic_raster_preview_pointer(
    *,
    document_preview_root: Path,
    runtime_configuration_relative_path: str,
    sealed_evaluation_relative_path: str,
    receipt_material_verifier: Callable[..., None] | None = None,
) -> tuple[str, RasterSourceAttestedActivationPointerV2]:
    """Validate v2 local evidence and atomically activate it without a human flag.

    The release command supplies only the two already-created immutable local
    artifact paths.  This helper derives every hash and profile identity from
    their bytes, validates the complete source/Gold/candidate chain, and does
    not alter an existing pointer on any failure.
    """

    root = _document_preview_root(document_preview_root)
    configuration_path = _sidecar_path(root, runtime_configuration_relative_path)
    evaluation_path = _sidecar_path(root, sealed_evaluation_relative_path)
    configuration_raw = _read_local_file(configuration_path, limit=_MAX_SIDECAR_BYTES)
    evaluation_raw = _read_local_file(evaluation_path, limit=_MAX_SIDECAR_BYTES)
    try:
        profile = PpStructureV3OfflineProfileV2.model_validate_json(configuration_raw)
        evaluation = RasterSealedEvaluationResultV2.model_validate_json(evaluation_raw)
    except ValidationError:
        raise RasterRuntimeError("raster_source_attested_activation_arguments_invalid") from None
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    evaluation_sha256 = hashlib.sha256(evaluation_raw).hexdigest()
    if (
        profile.raster_profile_revision != evaluation.raster_profile_revision
        or configuration_sha256 != evaluation.runtime_configuration_sha256
    ):
        raise RasterRuntimeError("raster_source_attested_activation_arguments_invalid")
    pointer_value: dict[str, object] = {
        "schema_version": "2.0",
        "activation_kind": "source_attested_synthetic_raster",
        "status": "activated",
        "dataset_id": "synthetic-raster-v1",
        "raster_profile_revision": profile.raster_profile_revision,
        "runtime_configuration_path": runtime_configuration_relative_path,
        "runtime_configuration_sha256": configuration_sha256,
        "sealed_evaluation_path": sealed_evaluation_relative_path,
        "sealed_evaluation_sha256": evaluation_sha256,
    }
    if profile.bundle_root_relative_path is not None:
        if profile.bundle_manifest_sha256 is None:
            raise RasterRuntimeError("raster_source_attested_activation_arguments_invalid")
        pointer_value["bundle_root_relative_path"] = profile.bundle_root_relative_path
        pointer_value["bundle_manifest_sha256"] = profile.bundle_manifest_sha256
    pointer_raw = _canonical_raster_json_bytes(pointer_value)
    pointer = parse_source_attested_synthetic_raster_activation(pointer_raw)
    destination = root / "approvals" / "raster-preview-active.json"
    # Validate from the exact bytes that will be written.  The destination is
    # still untouched at this point, which preserves a prior valid activation
    # when any runtime/evidence gate fails.
    _load_source_attested_synthetic_raster_binding_from_snapshot(
        root=root,
        approval_path=destination,
        approval_raw=pointer_raw,
        approval_sha256=hashlib.sha256(pointer_raw).hexdigest(),
        receipt_material_verifier=receipt_material_verifier,
        start_service_session=False,
    )
    _atomic_replace_raster_activation_pointer(destination=destination, payload=pointer_raw)
    return hashlib.sha256(pointer_raw).hexdigest(), pointer


@dataclass(frozen=True, slots=True)
class RasterDocumentPreviewStatusV1:
    availability: Literal["available", "unavailable"]
    raster_profile_revision: str | None


class RasterDocumentPreviewRuntime(Protocol):
    """Common app-facing OCR/page/status facade for static and dynamic modes."""

    def extract(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> RasterPreviewCandidateV2: ...

    def render_pdf_page(
        self,
        raster_input: RasterPreviewInputV2,
        structure: DocumentStructure,
        *,
        page_number: int,
    ) -> RasterViewerPageV1: ...

    def status(self) -> RasterDocumentPreviewStatusV1: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticRasterDocumentPreviewRuntime:
    """App facade for the preserved static v1 approval path or test injection."""

    engine: LocalRasterDocumentPreviewEngine
    page_renderer: ApprovedPdfiumRasterPageRenderer | None = None
    raster_profile_revision: str | None = None
    available: bool = False

    def extract(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> RasterPreviewCandidateV2:
        return self.engine.extract(raster_input, structure)

    def render_pdf_page(
        self,
        raster_input: RasterPreviewInputV2,
        structure: DocumentStructure,
        *,
        page_number: int,
    ) -> RasterViewerPageV1:
        if self.page_renderer is None:
            raise RasterDocumentPreviewUnavailable("raster page renderer is unavailable")
        if (
            self.raster_profile_revision is not None
            and raster_input.metadata.raster_profile_revision != self.raster_profile_revision
        ):
            raise RasterDocumentPreviewUnavailable("raster profile revision is superseded")
        return self.page_renderer.render_pdf_page(raster_input, structure, page_number=page_number)

    def status(self) -> RasterDocumentPreviewStatusV1:
        return RasterDocumentPreviewStatusV1(
            availability="available"
            if self.available and self.raster_profile_revision
            else "unavailable",
            raster_profile_revision=self.raster_profile_revision if self.available else None,
        )

    def close(self) -> None:
        """Match the automatic runtime lifecycle without changing v1 behavior."""

        return None


@dataclass(slots=True)
class AutoActivatingSourceAttestedRasterPreviewRuntime:
    """Atomically adopt only a complete current v2 pointer snapshot.

    The OCR engine and the reviewer page renderer are retained inside the same
    immutable binding.  A changed, missing, malformed, or drifting pointer
    clears both at once; no request can combine an old OCR engine with a new
    renderer profile.
    """

    approval_path: Path
    binding_loader: Callable[..., SourceAttestedRasterPreviewBindingV2] = (
        _load_source_attested_synthetic_raster_binding_from_snapshot
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _active_pointer_sha256: str | None = field(default=None, init=False, repr=False)
    _active_binding: SourceAttestedRasterPreviewBindingV2 | None = field(
        default=None, init=False, repr=False
    )

    def _clear_active(self) -> None:
        binding = self._active_binding
        self._active_pointer_sha256 = None
        self._active_binding = None
        if binding is not None:
            try:
                binding.close()
            except Exception:
                # Clearing a stale binding is fail-closed even if the child
                # cannot be reaped cleanly.  The session itself has already
                # been marked closed before its best-effort cleanup runs.
                pass

    def _confirm_snapshot_unchanged(
        self,
        *,
        root: Path,
        approval_path: Path,
        approval_raw: bytes,
        approval_sha256: str,
    ) -> None:
        """Reject a pointer replacement that raced a binding verification.

        A digest read before verification is not enough: the pointer can be
        atomically replaced while the model/package/sidecar material is being
        checked.  Re-reading the complete canonical pointer makes that race
        fail closed instead of allowing one request to retain an old engine
        after a new profile became current.

        The caller holds ``_lock`` so clearing the cached engine and renderer
        is one state transition.
        """

        confirmed_root, confirmed_path, confirmed_raw, confirmed_sha256, _confirmed = (
            _read_source_attested_synthetic_raster_pointer_snapshot(self.approval_path)
        )
        if (
            confirmed_root != root
            or confirmed_path != approval_path
            or confirmed_sha256 != approval_sha256
            or confirmed_raw != approval_raw
        ):
            self._clear_active()
            raise RasterRuntimeError("raster_source_attested_auto_pointer_changed")

    def _confirm_active_binding_current(
        self, binding: SourceAttestedRasterPreviewBindingV2
    ) -> None:
        """Confirm a completed OCR/page operation still belongs to the profile.

        OCR runs outside the facade lock, so a writer can replace the pointer
        while native work is in progress.  Do not return a result from the
        old binding after that replacement; the Backend can then make a fresh
        request with the newly published profile revision.
        """

        with self._lock:
            root, approval_path, approval_raw, approval_sha256, _pointer = (
                _read_source_attested_synthetic_raster_pointer_snapshot(self.approval_path)
            )
            if (
                self._active_binding is not binding
                or self._active_pointer_sha256 != binding.approval_sha256
                or approval_sha256 != binding.approval_sha256
            ):
                self._clear_active()
                raise RasterRuntimeError("raster_source_attested_auto_pointer_changed")
            binding.verify_runtime_state()
            self._confirm_snapshot_unchanged(
                root=root,
                approval_path=approval_path,
                approval_raw=approval_raw,
                approval_sha256=approval_sha256,
            )

    def _current_binding(self) -> SourceAttestedRasterPreviewBindingV2:
        with self._lock:
            root, canonical_path, initial_raw, pointer_sha256, _pointer = (
                _read_source_attested_synthetic_raster_pointer_snapshot(self.approval_path)
            )
            if self._active_binding is not None and self._active_pointer_sha256 == pointer_sha256:
                self._active_binding.verify_runtime_state()
                self._confirm_snapshot_unchanged(
                    root=root,
                    approval_path=canonical_path,
                    approval_raw=initial_raw,
                    approval_sha256=pointer_sha256,
                )
                return self._active_binding
            self._clear_active()
            binding = self.binding_loader(
                root=root,
                approval_path=canonical_path,
                approval_raw=initial_raw,
                approval_sha256=pointer_sha256,
            )
            self._confirm_snapshot_unchanged(
                root=root,
                approval_path=canonical_path,
                approval_raw=initial_raw,
                approval_sha256=pointer_sha256,
            )
            binding.verify_runtime_state()
            self._confirm_snapshot_unchanged(
                root=root,
                approval_path=canonical_path,
                approval_raw=initial_raw,
                approval_sha256=pointer_sha256,
            )
            self._active_pointer_sha256 = pointer_sha256
            self._active_binding = binding
            return binding

    def _binding_for_input(
        self, raster_input: RasterPreviewInputV2
    ) -> SourceAttestedRasterPreviewBindingV2:
        try:
            binding = self._current_binding()
            if raster_input.metadata.raster_profile_revision != binding.raster_profile_revision:
                raise RasterDocumentPreviewUnavailable("raster profile revision is superseded")
            return binding
        except RasterDocumentPreviewUnavailable:
            with self._lock:
                self._clear_active()
            raise
        except RasterRuntimeError as error:
            with self._lock:
                self._clear_active()
            raise RasterDocumentPreviewUnavailable(
                "raster preview engine is unavailable"
            ) from error

    def extract(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> RasterPreviewCandidateV2:
        binding = self._binding_for_input(raster_input)
        try:
            candidate = binding.engine.extract(raster_input, structure)
            self._confirm_active_binding_current(binding)
            return candidate
        except RasterDocumentPreviewUnavailable:
            with self._lock:
                self._clear_active()
            raise
        except RasterRuntimeError as error:
            with self._lock:
                self._clear_active()
            raise RasterDocumentPreviewUnavailable(
                "raster preview engine is unavailable"
            ) from error

    def render_pdf_page(
        self,
        raster_input: RasterPreviewInputV2,
        structure: DocumentStructure,
        *,
        page_number: int,
    ) -> RasterViewerPageV1:
        binding = self._binding_for_input(raster_input)
        try:
            rendered = binding.page_renderer.render_pdf_page(
                raster_input, structure, page_number=page_number
            )
            self._confirm_active_binding_current(binding)
            return rendered
        except RasterDocumentPreviewUnavailable:
            with self._lock:
                self._clear_active()
            raise
        except RasterRuntimeError as error:
            with self._lock:
                self._clear_active()
            raise RasterDocumentPreviewUnavailable("raster page viewer is unavailable") from error

    def status(self) -> RasterDocumentPreviewStatusV1:
        try:
            binding = self._current_binding()
            return RasterDocumentPreviewStatusV1(
                availability="available",
                raster_profile_revision=binding.raster_profile_revision,
            )
        except (RasterRuntimeError, RasterDocumentPreviewUnavailable):
            with self._lock:
                self._clear_active()
            return RasterDocumentPreviewStatusV1("unavailable", None)

    def close(self) -> None:
        """Retire the active immutable binding during timeout or app shutdown."""

        with self._lock:
            self._clear_active()


def _resolve_legacy_raster_document_preview_engine_body(
    *,
    approval_path: Path,
    approval_sha256: str,
    executor_factory: Callable[[PpStructureV3OfflineProfileV1], PpStructureV3NativeExecutor] | None,
    receipt_material_verifier: Callable[..., None] | None,
) -> RasterPreviewEngineResolution:
    try:
        approval_raw = _read_local_file(approval_path, limit=_MAX_SIDECAR_BYTES)
        if hashlib.sha256(approval_raw).hexdigest() != approval_sha256:
            raise RasterRuntimeError("raster_approval_invalid")
        pointer = RasterApprovalPointerV1.model_validate_json(approval_raw)
        root = approval_path.parent.parent
        configuration_path = _sidecar_path(root, pointer.runtime_configuration_path)
        configuration_raw = _read_local_file(configuration_path, limit=_MAX_SIDECAR_BYTES)
        if hashlib.sha256(configuration_raw).hexdigest() != pointer.runtime_configuration_sha256:
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        profile = PpStructureV3OfflineProfileV1.model_validate_json(configuration_raw)
        profile.model_root_relative_path()
        profile.sidecar_relative_path(profile.package_inventory_path)
        profile.sidecar_relative_path(profile.model_inventory_path)
        profile.sidecar_relative_path(profile.local_sidecar_command)
        profile.sidecar_relative_path(profile.sidecar_configuration_path)
        if profile.raster_profile_revision != pointer.raster_profile_revision:
            raise RasterRuntimeError("raster_approval_profile_mismatch")
        package_inventory_path = _sidecar_path(root, profile.package_inventory_path)
        package_inventory_raw = _read_local_file(package_inventory_path, limit=_MAX_SIDECAR_BYTES)
        _verify_package_inventory(
            raw=package_inventory_raw,
            expected_sha256=profile.package_inventory_sha256,
            required_packages=profile.required_packages,
        )
        model_root = _sidecar_path(root, profile.local_model_root)
        if model_root.is_symlink() or not model_root.is_dir():
            raise RasterRuntimeError("raster_model_inventory_invalid")
        model_inventory_path = _sidecar_path(root, profile.model_inventory_path)
        model_inventory_raw = _read_local_file(model_inventory_path, limit=_MAX_SIDECAR_BYTES)
        _verify_model_inventory(
            raw=model_inventory_raw,
            expected_sha256=profile.model_inventory_sha256,
            model_root=model_root,
        )
        sidecar_configuration_path = _sidecar_path(root, profile.sidecar_configuration_path)
        sidecar_configuration_raw = _read_local_file(
            sidecar_configuration_path, limit=_MAX_SIDECAR_BYTES
        )
        if (
            hashlib.sha256(sidecar_configuration_raw).hexdigest()
            != profile.sidecar_configuration_sha256
        ):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        sidecar_configuration = PpStructureV3SidecarConfigurationV1.model_validate_json(
            sidecar_configuration_raw
        )
        if (
            sidecar_configuration.extractor_revision != profile.extractor_revision
            or sidecar_configuration.raster_profile_revision != profile.raster_profile_revision
        ):
            raise RasterRuntimeError("raster_runtime_configuration_invalid")
        try:
            sidecar_configuration.validate_model_directories(model_root)
        except PpStructureV3SidecarError:
            raise RasterRuntimeError("raster_model_inventory_invalid") from None
        evaluation_path = _sidecar_path(root, pointer.sealed_evaluation_path)
        evaluation_raw = _read_local_file(evaluation_path, limit=_MAX_SIDECAR_BYTES)
        if hashlib.sha256(evaluation_raw).hexdigest() != pointer.sealed_evaluation_sha256:
            raise RasterRuntimeError("raster_sealed_evaluation_invalid")
        evaluation = RasterSealedEvaluationResultV1.model_validate_json(evaluation_raw)
        if (
            evaluation.raster_profile_revision != profile.raster_profile_revision
            or evaluation.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or evaluation.package_inventory_sha256 != profile.package_inventory_sha256
            or evaluation.model_inventory_sha256 != profile.model_inventory_sha256
            or evaluation.extractor_revision != profile.extractor_revision
        ):
            raise RasterRuntimeError("raster_sealed_evaluation_not_passing")
        calibration_path = _sidecar_path(root, evaluation.calibration_threshold_receipt_path)
        calibration_raw = _read_local_file(calibration_path, limit=_MAX_SIDECAR_BYTES)
        if (
            hashlib.sha256(calibration_raw).hexdigest()
            != evaluation.calibration_threshold_receipt_sha256
        ):
            raise RasterRuntimeError("raster_calibration_threshold_invalid")
        calibration = RasterCalibrationThresholdReceiptV1.model_validate_json(calibration_raw)
        if (
            calibration.raster_profile_revision != profile.raster_profile_revision
            or calibration.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or calibration.package_inventory_sha256 != profile.package_inventory_sha256
            or calibration.model_inventory_sha256 != profile.model_inventory_sha256
            or calibration.extractor_revision != profile.extractor_revision
            or calibration.calibration_case_manifest_sha256
            != evaluation.calibration_case_manifest_sha256
            or calibration.synthetic_raster_receipt_path != evaluation.synthetic_raster_receipt_path
            or calibration.synthetic_raster_receipt_sha256
            != evaluation.synthetic_raster_receipt_sha256
        ):
            raise RasterRuntimeError("raster_calibration_threshold_invalid")
        synthetic_receipt_path = _sidecar_path(root, calibration.synthetic_raster_receipt_path)
        synthetic_receipt_raw = _read_local_file(
            synthetic_receipt_path, limit=_MAX_RASTER_RECEIPT_BYTES
        )
        synthetic_receipt = _verify_synthetic_raster_receipt(
            raw=synthetic_receipt_raw,
            expected_sha256=calibration.synthetic_raster_receipt_sha256,
        )
        material_verifier = (
            receipt_material_verifier or _verify_synthetic_raster_activation_materials
        )
        try:
            material_verifier(root=root, receipt=synthetic_receipt)
        except RasterRuntimeError:
            raise
        except Exception:
            raise RasterRuntimeError("synthetic_raster_material_provenance_invalid") from None
        _verify_raster_candidate_provenance(
            root=root,
            receipt=synthetic_receipt,
            receipt_sha256=calibration.synthetic_raster_receipt_sha256,
            split="calibration",
            profile=profile,
            runtime_configuration_sha256=pointer.runtime_configuration_sha256,
            extractor_revision=profile.extractor_revision,
            candidate_bundle_relative_path=calibration.candidate_bundle_relative_path,
            candidate_bundle_sha256=calibration.candidate_bundle_sha256,
            candidate_execution_relative_path=calibration.candidate_execution_relative_path,
            candidate_execution_sha256=calibration.candidate_execution_sha256,
        )
        _verify_raster_candidate_provenance(
            root=root,
            receipt=synthetic_receipt,
            receipt_sha256=calibration.synthetic_raster_receipt_sha256,
            split="sealed",
            profile=profile,
            runtime_configuration_sha256=pointer.runtime_configuration_sha256,
            extractor_revision=profile.extractor_revision,
            candidate_bundle_relative_path=evaluation.candidate_bundle_relative_path,
            candidate_bundle_sha256=evaluation.candidate_bundle_sha256,
            candidate_execution_relative_path=evaluation.candidate_execution_relative_path,
            candidate_execution_sha256=evaluation.candidate_execution_sha256,
        )
        command_path = _sidecar_path(root, profile.local_sidecar_command)
        if _sha256_regular_file(command_path) != profile.sidecar_command_sha256:
            raise RasterRuntimeError("raster_native_sidecar_drift")
        executor = (
            executor_factory(profile)
            if executor_factory is not None
            else PpStructureV3CommandExecutor(
                command_path=command_path,
                command_sha256=profile.sidecar_command_sha256,
                timeout_seconds=profile.sidecar_timeout_seconds,
                configuration_path=sidecar_configuration_path,
                configuration_sha256=profile.sidecar_configuration_sha256,
                model_root=model_root,
                package_inventory_path=package_inventory_path,
                package_inventory_sha256=profile.package_inventory_sha256,
                model_inventory_path=model_inventory_path,
                model_inventory_sha256=profile.model_inventory_sha256,
                required_packages=tuple(profile.required_packages),
                sidecar_module_sha256=profile.sidecar_module_sha256,
            )
        )
        provider = PpStructureV3OfflineProvider(profile, executor, calibration.thresholds)
        from customs_ai_agent.service.raster_document_extraction import (
            MediaAwareRasterPreviewEngine,
        )

        input_adapter = PdfiumRasterInputAdapter(profile.pdfium_version, profile.raster_dpi)
        return RasterPreviewEngineResolution(
            MediaAwareRasterPreviewEngine(input_adapter, provider),
            "approved",
            page_renderer=ApprovedPdfiumRasterPageRenderer(
                input_adapter=input_adapter,
                raster_profile_revision=profile.raster_profile_revision,
                renderer_revision=f"pdfium-{profile.pdfium_version}-dpi-{profile.raster_dpi}",
            ),
            raster_profile_revision=profile.raster_profile_revision,
        )
    except (
        RasterRuntimeError,
        PpStructureV3SidecarError,
        ValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return RasterPreviewEngineResolution(
            DisabledRasterDocumentPreviewEngine(), "unavailable", "raster_approval_invalid"
        )


__all__ = [
    "AutoActivatingSourceAttestedRasterPreviewRuntime",
    "ApprovedPdfiumRasterPageRenderer",
    "PdfiumRasterInputAdapter",
    "PaddleOcr3DirectRasterBaselineProfileV1",
    "PaddleOcr3DirectRasterBaselineProvider",
    "RasterCalibrationThresholdReceiptV1",
    "RasterCalibrationThresholdsV1",
    "RasterBaselineEngineResolutionV1",
    "RasterCandidateEngineResolutionV1",
    "PpStructureV3CommandExecutor",
    "PpStructureV3CommandExecutorV2",
    "PpStructureV3NativeExecutor",
    "PpStructureV3OfflineProfileV1",
    "PpStructureV3OfflineProfileV2",
    "PpStructureV3OfflineProvider",
    "PpStructureV3ServiceSessionV1",
    "PpStructureV3SidecarConfigurationV1",
    "PpStructureV3SidecarResultV1",
    "RasterApprovalPointerV1",
    "RasterCalibrationThresholdReceiptV2",
    "RasterDocumentPreviewRuntime",
    "RasterDocumentPreviewStatusV1",
    "RasterPreviewEngineResolution",
    "RasterSealedEvaluationResultV1",
    "RasterSealedEvaluationResultV2",
    "RasterSealedGateScoreV1",
    "RasterRuntimeError",
    "RasterSourceAttestedActivationPointerV2",
    "SourceAttestedRasterPreviewBindingV2",
    "StaticRasterDocumentPreviewRuntime",
    "RasterViewerPageV1",
    "load_source_attested_synthetic_raster_binding",
    "make_synthetic_raster_profile_revision",
    "parse_source_attested_synthetic_raster_activation",
    "resolve_paddleocr3_direct_raster_baseline",
    "resolve_ppstructurev3_raster_candidate",
    "resolve_raster_document_preview_engine",
    "validate_raster_approval_pointer_path",
    "write_source_attested_synthetic_raster_preview_pointer",
]
