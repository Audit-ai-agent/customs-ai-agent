"""Immutable local runtime selection for text-only raster OCR.

This is deliberately independent from ``raster_document_preview_runtime``:
the structured v2 pointer and its table/geometry Gate can neither enable nor
disable the lighter editable-text path.  Both the parent and the sidecar
verify fixed local material; neither accepts a URL, model fallback, or
download selector.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import threading
import time
from typing import Any, BinaryIO, Callable, Iterator, Literal, Mapping, Protocol, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.raster_offline_bundle import (
    current_raster_runtime_platform_binding,
)
from customs_ai_agent.evaluation.raster_text_trained_paddle_bundle import (
    RasterTextTrainedPaddleBundleError,
    load_verified_raster_text_trained_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_server_paddle_bundle import (
    RasterTextServerPaddleBundleError,
    load_verified_raster_text_direct_server_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_public_finetuned_paddle_bundle import (
    RasterTextPublicFineTuneCorpusProvenanceV1,
    RasterTextPublicFineTunedPaddleBundleError,
    load_verified_raster_text_public_finetuned_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_voyage_replay_corpus import (
    VoyageReplayCorpusManifestV1,
)
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionResultV1,
    RasterTextPageV1,
    RasterTextWarningV1,
    raster_text_result_json,
)
from customs_ai_agent.service.document_input import DocumentStructure
from customs_ai_agent.service.raster_document_extraction import RasterizedPageV2
from customs_ai_agent.service.raster_document_preview_runtime import (
    PdfiumRasterInputAdapter,
    _ISOLATED_SIDECAR_BOOTSTRAP,
    _controlled_directory_tree_sha256,
    _controlled_module_root_tree_sha256,
    _document_preview_root,
    _isolated_v2_sidecar_environment,
    _isolated_v2_sidecar_home,
    _new_isolated_v2_sidecar_home,
    _read_local_file,
    _sidecar_path,
    _validate_controlled_child_file,
    _verify_model_inventory,
    _verify_package_inventory,
)
from customs_ai_agent.service.raster_text_extraction import (
    DisabledRasterTextPreviewEngine,
    LocalRasterTextPreviewEngine,
    MediaAwareRasterTextEngine,
    RasterTextLedgerV1,
    RasterTextPreviewUnavailable,
    RasterTextProvider,
)
from customs_ai_agent.service.raster_text_extraction import RasterTextInputAdapter
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextInputV1,
    RasterTextPreviewCandidateV1,
)


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_CONFIGURATION_BYTES = 1024 * 1024
# Candidate bundles retain local editable OCR drafts.  They are never exposed
# through the pointer, but a 24-asset smoke bundle can legitimately exceed a
# configuration file.  Keep the loader aligned with activation's compact
# evidence cap rather than rejecting a valid passing profile by size alone.
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
# An independent repair audit contains 800 compact OCR result records.  The
# activation writer replays those records against source-only Gold once; the
# runtime loader then re-hashes and parses the immutable bundle without
# reopening every raster and Gold asset.  Keep this separate from the smaller
# ordinary evidence cap so schema-1 through schema-7 behavior is unchanged.
_MAX_REPAIR_AUDIT_BUNDLE_BYTES = 64 * 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_INPUT_BYTES = 128 * 1024 * 1024
_PROTOCOL = "customs-raster-text-sidecar-v1"
_SERVICE_PROTOCOL = f"{_PROTOCOL}-service"
_EVALUATION_SESSION_PROTOCOL = f"{_PROTOCOL}-session"
_TEXT_BUNDLE_RELEASE_PREFIX = "raster-offline-bundles-v1/releases"
_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX = "raster-text-direct-bundles-v1/releases"
_TEXT_DIRECT_PADDLE_MOBILE_BUNDLE_RELEASE_PREFIX = "raster-text-direct-mobile-bundles-v1/releases"
_TEXT_DIRECT_PADDLE_SERVER_REC_BUNDLE_RELEASE_PREFIX = (
    "raster-text-direct-server-rec-bundles-v1/releases"
)
_TEXT_PUBLIC_FINETUNED_PADDLE_BUNDLE_RELEASE_PREFIX = (
    "raster-text-public-finetuned-paddle-bundles-v1/releases"
)
_TEXT_PUBLIC_VOYAGE_REPLAY_PADDLE_BUNDLE_RELEASE_PREFIX = (
    "raster-text-public-voyage-replay-paddle-bundles-v1/releases"
)
_TEXT_TRAINED_PADDLE_BUNDLE_RELEASE_PREFIX = "raster-text-trained-paddle-bundles-v1/releases"
_V19_ENGINE_KIND = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_V19B_ENGINE_KIND = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
)
_V20_ENGINE_KIND = (
    "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_MAX_OWNED_TEXT_SIDECAR_SECONDS = 30
# V1 keeps its established 55-second cold-ready bound.  The independently
# bundled V2 static-inference child needs one extra five-second setup margin
# on this hardware, but that margin belongs only to readiness: document OCR
# remains in the 30-second owned request lane below.
_V1_TEXT_SERVICE_STARTUP_SECONDS = 55
_V2_TEXT_SERVICE_STARTUP_SECONDS = 60
_V3_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
# V4 keeps the same bounded direct-Paddle cold-ready lane as V3.  Its only
# semantic change is sidecar line ordering, never a longer user request.
_V4_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
# V5 changes only the fixed detector box threshold; it never expands either
# the 30-second request lane or the 60-second cold-ready lane.
_V5_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
# V6 isolates only a modest detector input increase; its user-facing request
# and cold-ready contracts remain the same bounded 30/60-second lane.
_V6_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
# V7 keeps V6's bounded 1024-side lane and lowers only its fixed detector box
# threshold.  It is still limited to 30 seconds per ready request and 60
# seconds before the native child announces readiness.
_V7_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
# V8 returns to V5's native 960-side detector limit and lowers only the box
# threshold.  It remains within the same bounded 30-second request and
# 60-second cold-ready lane as the other direct mobile candidates.
_V8_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
# V9 retains V8's 30/60-second bounded direct-mobile lane.  Its only new
# engine semantic is a fixed two-strip detector pass; the recognizer remains
# single-pass on the original page.
_V9_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
# V10/V11 change only exact detector/NMS semantics; both retain the direct
# mobile lane's fixed 30-second request and 60-second cold-ready bounds.
_V10_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_V11_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_V12_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_V13_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_V14_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_V15_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_V18_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_V19_TEXT_SERVICE_STARTUP_SECONDS = _V15_TEXT_SERVICE_STARTUP_SECONDS
_V20_TEXT_SERVICE_STARTUP_SECONDS = _V19_TEXT_SERVICE_STARTUP_SECONDS
_MAX_TEXT_SERVICE_STARTUP_SECONDS = _V2_TEXT_SERVICE_STARTUP_SECONDS
_PIPE_OPERATION_REAP_SECONDS = 2.0
_DEFERRED_HOME_CLEANUP_ATTEMPTS = 3
_DEFERRED_HOME_CLEANUP_RETRY_SECONDS = 0.05

# The generic PP-Structure bootstrap intentionally resolves every supplied
# path before comparing it.  That is correct for its fixed bundle layout, but
# Windows can expand the direct-Paddle microbundle's 8.3 operational root
# into a long Korean workspace spelling while resolving its deeply nested
# first-party command leaf.  ``commonpath`` then receives two different
# spellings for the same location and can fail before the sidecar has a
# chance to publish its content-free ready frame.
#
# V2 keeps the original operational spellings for the executable/module/
# command containment checks.  It still resolves the interpreter root only
# when validating the standard-library boundary, where both operands use the
# same canonical spelling.  Do not share this with V1: V1 retains the
# established generic high-level PaddleOCR bootstrap unchanged.
_DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP = (
    "import os,runpy,sys,sysconfig\n"
    "if len(sys.argv)<4: raise SystemExit(70)\n"
    "interpreter_argument=os.path.abspath(sys.argv[1])\n"
    "module_argument=os.path.abspath(sys.argv[2])\n"
    "command_argument=os.path.abspath(sys.argv[3])\n"
    "def lexical_inside(root,path):\n"
    " try:\n"
    "  root_case=os.path.normcase(root)\n"
    "  return os.path.commonpath((root_case,os.path.normcase(path)))==root_case\n"
    " except ValueError:\n"
    "  return False\n"
    "def no_links(path):\n"
    " if not os.path.isabs(path): return False\n"
    " drive,tail=os.path.splitdrive(path)\n"
    " current=(drive+os.sep) if drive else os.sep\n"
    " for part in tail.split(os.sep):\n"
    "  if not part: continue\n"
    "  current=os.path.join(current,part)\n"
    "  if os.path.islink(current): return False\n"
    " return True\n"
    "if (not os.path.isdir(interpreter_argument) or not no_links(interpreter_argument) or "
    "not os.path.isdir(module_argument) or not no_links(module_argument) or "
    "not os.path.isfile(command_argument) or not no_links(command_argument) or "
    "not lexical_inside(module_argument,command_argument)): raise SystemExit(70)\n"
    "try:\n"
    " interpreter_root=os.path.realpath(interpreter_argument)\n"
    " executable_path=os.path.realpath(sys.executable)\n"
    "except OSError:\n"
    " raise SystemExit(70)\n"
    "if not lexical_inside(interpreter_root,executable_path): raise SystemExit(70)\n"
    "stdlib_paths=[]\n"
    "for key in ('stdlib','platstdlib'):\n"
    " path=sysconfig.get_path(key)\n"
    " if path and path not in stdlib_paths: stdlib_paths.append(path)\n"
    "for path in (os.path.join(sys.base_prefix,'DLLs'), "
    "os.path.join(stdlib_paths[0],'lib-dynload') if stdlib_paths else ''):\n"
    " if path and os.path.isdir(path) and path not in stdlib_paths: stdlib_paths.append(path)\n"
    "try:\n"
    " resolved_stdlib_paths=[os.path.realpath(path) for path in stdlib_paths]\n"
    "except OSError:\n"
    " raise SystemExit(70)\n"
    "if any(not lexical_inside(interpreter_root,path) for path in resolved_stdlib_paths): raise SystemExit(70)\n"
    "sys.path[:]=[module_argument,*stdlib_paths]\n"
    "sys.argv=[command_argument,*sys.argv[4:]]\n"
    "runpy.run_path(command_argument,run_name='__main__')\n"
)


@dataclass(slots=True)
class _DeferredHomeCleanupState:
    """Detached, serial cleanup ownership for one isolated sidecar home.

    The state intentionally owns only the cleanup callable, never a service
    session.  A module-level exit registry can therefore keep a failed
    deadline cleanup recoverable without pinning a retired provider/session.
    """

    cleanup_home: Callable[[], None]
    _cleaned: bool = field(default=False, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def cleaned(self) -> bool:
        with self._lock:
            return self._cleaned

    def attempt(self) -> bool:
        """Run one cleanup attempt without exposing filesystem details."""

        with self._lock:
            if self._cleaned:
                return True
            if self._running:
                return False
            self._running = True
        succeeded = False
        try:
            self.cleanup_home()
            succeeded = True
        except Exception:
            # A just-reaped native child can temporarily retain a Windows
            # handle.  The caller owns bounded retries and stays content-free.
            pass
        finally:
            with self._lock:
                self._running = False
                if succeeded:
                    self._cleaned = True
        return succeeded


_DEFERRED_HOME_CLEANUP_REGISTRY: dict[object, _DeferredHomeCleanupState] = {}
_DEFERRED_HOME_CLEANUP_REGISTRY_LOCK = threading.Lock()
_DEFERRED_HOME_CLEANUP_HANDLER_REGISTERED = False


def _cleanup_detached_home_with_retries(state: _DeferredHomeCleanupState) -> bool:
    """Return whether bounded, detached cleanup succeeded."""

    for attempt in range(_DEFERRED_HOME_CLEANUP_ATTEMPTS):
        if state.attempt():
            return True
        if attempt + 1 < _DEFERRED_HOME_CLEANUP_ATTEMPTS:
            time.sleep(_DEFERRED_HOME_CLEANUP_RETRY_SECONDS)
    return state.cleaned


def _remove_deferred_home_cleanup(token: object) -> None:
    """Release one successful cleanup from the module exit registry."""

    with _DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:
        _DEFERRED_HOME_CLEANUP_REGISTRY.pop(token, None)


def _run_registered_deferred_home_cleanups() -> None:
    """Best-effort process-exit fallback for homes whose workers exhausted.

    This function is registered once while this module imports, not once per
    service session.  It only retains detached cleanup state and removes a
    registry entry after the callback succeeds.
    """

    with _DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:
        entries = tuple(_DEFERRED_HOME_CLEANUP_REGISTRY.items())
    for token, state in entries:
        if _cleanup_detached_home_with_retries(state):
            _remove_deferred_home_cleanup(token)


def _ensure_deferred_home_cleanup_handler() -> bool:
    """Install the single registry handler before any request deadline path.

    A normal service imports this module during startup, before Python begins
    interpreter shutdown.  If an embedding process imports only after atexit
    registration is unavailable, no in-process exit callback can be created;
    callers still retain deferred ownership and never turn a 30-second error
    path into synchronous filesystem cleanup.  Retrying here lets an unusual
    transient registration failure recover before a later session schedules.
    """

    global _DEFERRED_HOME_CLEANUP_HANDLER_REGISTERED
    with _DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:
        if _DEFERRED_HOME_CLEANUP_HANDLER_REGISTERED:
            return True
        try:
            atexit.register(_run_registered_deferred_home_cleanups)
        except Exception:
            return False
        _DEFERRED_HOME_CLEANUP_HANDLER_REGISTERED = True
        return True


def _register_deferred_home_cleanup(state: _DeferredHomeCleanupState) -> object:
    """Keep detached cleanup recoverable even if its daemon cannot start."""

    # Do not use a per-session atexit bound method here: that would retain the
    # entire retired session after a successful cleanup.  The import-time
    # handler is already installed in ordinary operation; this call only
    # retries registration for an unusual transient initialization failure.
    _ensure_deferred_home_cleanup_handler()
    token = object()
    with _DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:
        _DEFERRED_HOME_CLEANUP_REGISTRY[token] = state
    return token


# This runs at module import, while ordinary application startup can still
# register a process-exit callback.  Later deadline paths never depend on a
# fresh per-session ``atexit.register`` call.
_ensure_deferred_home_cleanup_handler()


class RasterTextRuntimeError(RuntimeError):
    """Content-free runtime failure; never include OCR text or child stderr."""


class _RuntimeWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or any(not part or part == "." for part in path.parts)
    ):
        raise ValueError("raster text path is invalid")
    return value


def raster_text_profile_fingerprint_sha256(payload: object) -> str:
    """Return the immutable identity of a direct-text runtime profile.

    The calibrated confidence floor is part of the profile material.  It must
    therefore produce a new profile revision instead of sharing the bundle
    fingerprint used by an earlier calibration pass.
    """

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


class RasterTextOfflineProfileV1(_RuntimeWireModel):
    """All local direct OCR material required by a text-only profile."""

    schema_version: Literal["1.0"] = "1.0"
    engine_kind: Literal["paddleocr3_direct_text"] = "paddleocr3_direct_text"
    runtime_api: Literal["paddleocr3-paddlex-3.2"] = "paddleocr3-paddlex-3.2"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: int = Field(ge=1, le=55)
    cpu_threads: Literal[1, 2, 4, 6]
    text_detection_limit_side_len: Literal[1536, 2048, 2526]
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(ge=0.4, le=0.6)
    text_detection_unclip_ratio: float = Field(ge=1.5, le=2.0)
    # This value is selected from calibration evidence and then folded into a
    # distinct immutable runtime profile before sealed evaluation.
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_server_det"]
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"]
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV1":
        for value in (
            self.bundle_root_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if len(set(self.parent_renderer_required_packages)) != len(
            self.parent_renderer_required_packages
        ) or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages):
            raise ValueError("raster text required packages are invalid")
        if (
            tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages) != ("paddleocr", "paddlex", "paddlepaddle")
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold not in {0.4, 0.6}
            or self.text_detection_unclip_ratio not in {1.5, 2.0}
        ):
            raise ValueError("raster text required packages are invalid")
        expected_bundle_root = f"{_TEXT_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        if self.bundle_root_relative_path != expected_bundle_root:
            raise ValueError("raster text bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text bundle binding is invalid")
        bundle_root = PurePosixPath(self.bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
        ):
            path = PurePosixPath(value)
            try:
                relative = path.relative_to(bundle_root)
            except ValueError:
                raise ValueError("raster text bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV2(_RuntimeWireModel):
    """Immutable profile for the narrowly bundled direct-Paddle runtime.

    V1 deliberately remains untouched because it binds the earlier
    PaddleOCR/PaddleX child.  V2 has a different schema discriminator,
    package set, engine pair, and release-root namespace, so a V1 artifact
    can never silently select the new static-inference sidecar (or vice
    versa).
    """

    schema_version: Literal["2.0"] = "2.0"
    engine_kind: Literal["direct_paddle_text"] = "direct_paddle_text"
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    # V2 is the customer-facing direct engine: its immutable profile must
    # carry the same 30-second bound that the provider enforces at runtime.
    # Keeping this as a literal prevents a valid-looking edited profile from
    # advertising a longer request budget and relying on a hidden clamp.
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[1, 2, 4, 6]
    # The direct static detector is intentionally bounded to 960 pixels on
    # the long side.  Measured CPU inference at 1280 or above exceeds the
    # owned 30-second customer lane on the supported local hardware; V2 must
    # not reintroduce that unsafe configuration through a profile edit.
    text_detection_limit_side_len: Literal[960]
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(ge=0.4, le=0.6)
    text_detection_unclip_ratio: float = Field(ge=1.5, le=2.0)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_server_det"]
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"]
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV2":
        for value in (
            self.bundle_root_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold not in {0.4, 0.6}
            or self.text_detection_unclip_ratio not in {1.5, 2.0}
        ):
            raise ValueError("raster text required packages are invalid")
        expected_bundle_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        if self.bundle_root_relative_path != expected_bundle_root:
            raise ValueError("raster text bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text bundle binding is invalid")
        bundle_root = PurePosixPath(self.bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(bundle_root)
            except ValueError:
                raise ValueError("raster text bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV3(_RuntimeWireModel):
    """Immutable mobile-detector profile bound to two distinct local bundles.

    The direct-Paddle sidecar/interpreter closure remains the previously
    attested V2 microbundle.  The mobile detector and recognizer live in a
    separately content-addressed model bundle sourced from the fixed local
    PaddleOCR3 inventory.  Keeping the two roots explicit prevents a V2
    server model, a V3 mobile model, or an arbitrary cache directory from
    being silently substituted for another.
    """

    schema_version: Literal["3.0"] = "3.0"
    engine_kind: Literal["direct_paddle_mobile_text"] = "direct_paddle_mobile_text"
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    # The independently runnable direct-Paddle closure (interpreter,
    # packages and first-party sidecar), not the mobile model bytes.
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    # The copied, immutable PP-OCRv5 mobile detector/recognizer material.
    mobile_model_bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    mobile_model_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_model_inventory_relative_path: str = Field(min_length=1, max_length=240)
    source_model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[1, 2, 4, 6]
    text_detection_limit_side_len: Literal[960]
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(ge=0.4, le=0.6)
    text_detection_unclip_ratio: float = Field(ge=1.5, le=2.0)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"]
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"]
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV3":
        for value in (
            self.bundle_root_relative_path,
            self.mobile_model_bundle_root_relative_path,
            self.source_model_inventory_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold not in {0.4, 0.6}
            or self.text_detection_unclip_ratio not in {1.5, 2.0}
        ):
            raise ValueError("raster text required packages are invalid")
        expected_sidecar_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        expected_mobile_model_root = (
            f"{_TEXT_DIRECT_PADDLE_MOBILE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
        )
        if (
            self.bundle_root_relative_path != expected_sidecar_root
            or self.mobile_model_bundle_root_relative_path != expected_mobile_model_root
            or self.source_model_inventory_relative_path
            != f"model-inventories/{self.source_model_inventory_sha256}.json"
        ):
            raise ValueError("raster text mobile bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text mobile bundle binding is invalid")
        sidecar_root = PurePosixPath(self.bundle_root_relative_path)
        mobile_model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(sidecar_root)
            except ValueError:
                raise ValueError("raster text mobile bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text mobile bundle binding is invalid")
        for value in (self.local_model_root, self.model_inventory_path):
            try:
                relative = PurePosixPath(value).relative_to(mobile_model_root)
            except ValueError:
                raise ValueError("raster text mobile bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text mobile bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV4(_RuntimeWireModel):
    """Immutable V4 mobile candidate with harmonic row-band ordering.

    V4 deliberately reuses V3's two independently attested roots: the
    direct-Paddle sidecar closure and the separately copied mobile-model
    bundle.  The schema/engine discriminator is the only way a child can
    select harmonic ordering, so a V3 profile can never acquire it through a
    tuning field or a fallback.
    """

    schema_version: Literal["4.0"] = "4.0"
    engine_kind: Literal["direct_paddle_mobile_text_harmonic"] = (
        "direct_paddle_mobile_text_harmonic"
    )
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mobile_model_bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    mobile_model_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_model_inventory_relative_path: str = Field(min_length=1, max_length=240)
    source_model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[6] = 6
    text_detection_limit_side_len: Literal[960] = 960
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(default=0.6, ge=0.6, le=0.6)
    text_detection_unclip_ratio: float = Field(default=1.5, ge=1.5, le=1.5)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV4":
        for value in (
            self.bundle_root_relative_path,
            self.mobile_model_bundle_root_relative_path,
            self.source_model_inventory_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.sidecar_timeout_seconds != 30
            or self.cpu_threads != 6
            or self.text_detection_limit_side_len != 960
            or self.text_detection_limit_type != "max"
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold != 0.6
            or self.text_detection_unclip_ratio != 1.5
            or self.text_detection_model_name != "PP-OCRv5_mobile_det"
            or self.text_recognition_model_name != "en_PP-OCRv5_mobile_rec"
        ):
            raise ValueError("raster text harmonic candidate is invalid")
        expected_sidecar_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        expected_mobile_model_root = (
            f"{_TEXT_DIRECT_PADDLE_MOBILE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
        )
        if (
            self.bundle_root_relative_path != expected_sidecar_root
            or self.mobile_model_bundle_root_relative_path != expected_mobile_model_root
            or self.source_model_inventory_relative_path
            != f"model-inventories/{self.source_model_inventory_sha256}.json"
        ):
            raise ValueError("raster text harmonic bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text harmonic bundle binding is invalid")
        sidecar_root = PurePosixPath(self.bundle_root_relative_path)
        mobile_model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(sidecar_root)
            except ValueError:
                raise ValueError("raster text harmonic bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text harmonic bundle binding is invalid")
        for value in (self.local_model_root, self.model_inventory_path):
            try:
                relative = PurePosixPath(value).relative_to(mobile_model_root)
            except ValueError:
                raise ValueError("raster text harmonic bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text harmonic bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV5(_RuntimeWireModel):
    """Immutable harmonic-recall candidate with the lower fixed box threshold.

    V5 retains V4's independent direct-sidecar and mobile-model releases.
    Its separate schema/engine identity is required because the .4 detector
    box threshold is a candidate semantic, never a mutable runtime knob.
    """

    schema_version: Literal["5.0"] = "5.0"
    engine_kind: Literal["direct_paddle_mobile_text_harmonic_recall"] = (
        "direct_paddle_mobile_text_harmonic_recall"
    )
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mobile_model_bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    mobile_model_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_model_inventory_relative_path: str = Field(min_length=1, max_length=240)
    source_model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[6] = 6
    text_detection_limit_side_len: Literal[960] = 960
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(default=0.4, ge=0.4, le=0.4)
    text_detection_unclip_ratio: float = Field(default=1.5, ge=1.5, le=1.5)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV5":
        for value in (
            self.bundle_root_relative_path,
            self.mobile_model_bundle_root_relative_path,
            self.source_model_inventory_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.sidecar_timeout_seconds != 30
            or self.cpu_threads != 6
            or self.text_detection_limit_side_len != 960
            or self.text_detection_limit_type != "max"
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold != 0.4
            or self.text_detection_unclip_ratio != 1.5
            or self.text_detection_model_name != "PP-OCRv5_mobile_det"
            or self.text_recognition_model_name != "en_PP-OCRv5_mobile_rec"
        ):
            raise ValueError("raster text harmonic recall candidate is invalid")
        expected_sidecar_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        expected_mobile_model_root = (
            f"{_TEXT_DIRECT_PADDLE_MOBILE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
        )
        if (
            self.bundle_root_relative_path != expected_sidecar_root
            or self.mobile_model_bundle_root_relative_path != expected_mobile_model_root
            or self.source_model_inventory_relative_path
            != f"model-inventories/{self.source_model_inventory_sha256}.json"
        ):
            raise ValueError("raster text harmonic recall bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text harmonic recall bundle binding is invalid")
        sidecar_root = PurePosixPath(self.bundle_root_relative_path)
        mobile_model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(sidecar_root)
            except ValueError:
                raise ValueError("raster text harmonic recall bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text harmonic recall bundle binding is invalid")
        for value in (self.local_model_root, self.model_inventory_path):
            try:
                relative = PurePosixPath(value).relative_to(mobile_model_root)
            except ValueError:
                raise ValueError("raster text harmonic recall bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text harmonic recall bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV6(_RuntimeWireModel):
    """Immutable V5-compatible candidate that changes only detector scale.

    V6 retains the V5 mobile detector/recognizer pair, harmonic reading order,
    and fixed .4 box threshold.  The only inference setting it changes is the
    detector-side limit from 960 to 1024, so its evaluation can attribute any
    difference to that modest resolution increase.
    """

    schema_version: Literal["6.0"] = "6.0"
    engine_kind: Literal["direct_paddle_mobile_text_1024"] = "direct_paddle_mobile_text_1024"
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mobile_model_bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    mobile_model_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_model_inventory_relative_path: str = Field(min_length=1, max_length=240)
    source_model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[6] = 6
    text_detection_limit_side_len: Literal[1024] = 1024
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(default=0.4, ge=0.4, le=0.4)
    text_detection_unclip_ratio: float = Field(default=1.5, ge=1.5, le=1.5)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV6":
        for value in (
            self.bundle_root_relative_path,
            self.mobile_model_bundle_root_relative_path,
            self.source_model_inventory_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.sidecar_timeout_seconds != 30
            or self.cpu_threads != 6
            or self.text_detection_limit_side_len != 1024
            or self.text_detection_limit_type != "max"
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold != 0.4
            or self.text_detection_unclip_ratio != 1.5
            or self.text_detection_model_name != "PP-OCRv5_mobile_det"
            or self.text_recognition_model_name != "en_PP-OCRv5_mobile_rec"
        ):
            raise ValueError("raster text 1024 candidate is invalid")
        expected_sidecar_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        expected_mobile_model_root = (
            f"{_TEXT_DIRECT_PADDLE_MOBILE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
        )
        if (
            self.bundle_root_relative_path != expected_sidecar_root
            or self.mobile_model_bundle_root_relative_path != expected_mobile_model_root
            or self.source_model_inventory_relative_path
            != f"model-inventories/{self.source_model_inventory_sha256}.json"
        ):
            raise ValueError("raster text 1024 bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text 1024 bundle binding is invalid")
        sidecar_root = PurePosixPath(self.bundle_root_relative_path)
        mobile_model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(sidecar_root)
            except ValueError:
                raise ValueError("raster text 1024 bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text 1024 bundle binding is invalid")
        for value in (self.local_model_root, self.model_inventory_path):
            try:
                relative = PurePosixPath(value).relative_to(mobile_model_root)
            except ValueError:
                raise ValueError("raster text 1024 bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text 1024 bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV7(_RuntimeWireModel):
    """Immutable V6-compatible candidate with only the .3 box threshold.

    V7 retains V6's PP-OCRv5 mobile detector/recognizer, 1024-side detector
    limit, six CPU threads, harmonic reading order, and both independently
    attested release roots.  Its separate schema/engine identity fixes the
    detector box threshold at .3; it is never a global runtime option.
    """

    schema_version: Literal["7.0"] = "7.0"
    engine_kind: Literal["direct_paddle_mobile_text_harmonic_recall_1024_lowbox"] = (
        "direct_paddle_mobile_text_harmonic_recall_1024_lowbox"
    )
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mobile_model_bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    mobile_model_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_model_inventory_relative_path: str = Field(min_length=1, max_length=240)
    source_model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[6] = 6
    text_detection_limit_side_len: Literal[1024] = 1024
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_unclip_ratio: float = Field(default=1.5, ge=1.5, le=1.5)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV7":
        for value in (
            self.bundle_root_relative_path,
            self.mobile_model_bundle_root_relative_path,
            self.source_model_inventory_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.sidecar_timeout_seconds != 30
            or self.cpu_threads != 6
            or self.text_detection_limit_side_len != 1024
            or self.text_detection_limit_type != "max"
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold != 0.3
            or self.text_detection_unclip_ratio != 1.5
            or self.text_detection_model_name != "PP-OCRv5_mobile_det"
            or self.text_recognition_model_name != "en_PP-OCRv5_mobile_rec"
        ):
            raise ValueError("raster text 1024 low-box candidate is invalid")
        expected_sidecar_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        expected_mobile_model_root = (
            f"{_TEXT_DIRECT_PADDLE_MOBILE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
        )
        if (
            self.bundle_root_relative_path != expected_sidecar_root
            or self.mobile_model_bundle_root_relative_path != expected_mobile_model_root
            or self.source_model_inventory_relative_path
            != f"model-inventories/{self.source_model_inventory_sha256}.json"
        ):
            raise ValueError("raster text 1024 low-box bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text 1024 low-box bundle binding is invalid")
        sidecar_root = PurePosixPath(self.bundle_root_relative_path)
        mobile_model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(sidecar_root)
            except ValueError:
                raise ValueError("raster text 1024 low-box bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text 1024 low-box bundle binding is invalid")
        for value in (self.local_model_root, self.model_inventory_path):
            try:
                relative = PurePosixPath(value).relative_to(mobile_model_root)
            except ValueError:
                raise ValueError("raster text 1024 low-box bundle binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text 1024 low-box bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV8(_RuntimeWireModel):
    """Immutable V5-compatible candidate with only the .3 box threshold.

    V8 retains V5's PP-OCRv5 mobile detector/recognizer, native 960-side
    detector limit, six CPU threads, harmonic reading order, and both
    independently attested release roots.  Its separate schema/engine fixes
    the detector box threshold at .3; it is never a global runtime option.
    """

    schema_version: Literal["8.0"] = "8.0"
    engine_kind: Literal["direct_paddle_mobile_text_harmonic_recall_native960_lowbox"] = (
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox"
    )
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mobile_model_bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    mobile_model_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_model_inventory_relative_path: str = Field(min_length=1, max_length=240)
    source_model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[6] = 6
    text_detection_limit_side_len: Literal[960] = 960
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_unclip_ratio: float = Field(default=1.5, ge=1.5, le=1.5)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)

    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)
    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: Literal["en_PP-OCRv5_mobile_rec"] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _local_paths_are_explicit(self) -> "RasterTextOfflineProfileV8":
        server_recognizer_profile = self.schema_version == "18.0"
        public_finetuned_profile = self.schema_version == "19.0"
        public_voyage_replay_profile = self.schema_version == "20.0"
        for value in (
            self.bundle_root_relative_path,
            self.mobile_model_bundle_root_relative_path,
            self.source_model_inventory_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.sidecar_timeout_seconds != 30
            or (
                self.cpu_threads != 6
                and not (
                    self.schema_version in {"15.0", "16.0", "17.0", "18.0", "19.0", "20.0"}
                    and self.cpu_threads == 4
                )
            )
            or self.text_detection_limit_side_len != 960
            or self.text_detection_limit_type != "max"
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold != 0.3
            or self.text_detection_unclip_ratio != 1.5
            or self.text_detection_model_name != "PP-OCRv5_mobile_det"
            or self.text_recognition_model_name
            != (
                "PP-OCRv5_server_rec"
                if server_recognizer_profile
                else "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
                if public_voyage_replay_profile
                else "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
                if public_finetuned_profile
                else "en_PP-OCRv5_mobile_rec"
            )
        ):
            raise ValueError("raster text native-960 low-box candidate is invalid")
        expected_sidecar_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        expected_mobile_model_root = (
            f"{_TEXT_DIRECT_PADDLE_SERVER_REC_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
            if server_recognizer_profile
            else f"{_TEXT_PUBLIC_VOYAGE_REPLAY_PADDLE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
            if public_voyage_replay_profile
            else f"{_TEXT_PUBLIC_FINETUNED_PADDLE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
            if public_finetuned_profile
            else f"{_TEXT_DIRECT_PADDLE_MOBILE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.mobile_model_bundle_manifest_sha256}"
        )
        if (
            self.bundle_root_relative_path != expected_sidecar_root
            or self.mobile_model_bundle_root_relative_path != expected_mobile_model_root
            or (
                not server_recognizer_profile
                and not public_finetuned_profile
                and not public_voyage_replay_profile
                and self.source_model_inventory_relative_path
                != f"model-inventories/{self.source_model_inventory_sha256}.json"
            )
        ):
            raise ValueError("raster text native-960 low-box bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text native-960 low-box bundle binding is invalid")
        sidecar_root = PurePosixPath(self.bundle_root_relative_path)
        mobile_model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(sidecar_root)
            except ValueError:
                raise ValueError(
                    "raster text native-960 low-box bundle binding is invalid"
                ) from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text native-960 low-box bundle binding is invalid")
        for value in (self.local_model_root, self.model_inventory_path):
            try:
                relative = PurePosixPath(value).relative_to(mobile_model_root)
            except ValueError:
                raise ValueError(
                    "raster text native-960 low-box bundle binding is invalid"
                ) from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text native-960 low-box bundle binding is invalid")
        try:
            if PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            ) == PurePosixPath(".") or PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            ) == PurePosixPath("."):
                raise ValueError("raster text sidecar root is invalid")
        except ValueError:
            raise ValueError("raster text sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV9(RasterTextOfflineProfileV8):
    """Immutable V8-compatible candidate with fixed two-strip detection.

    V9 has the same independently attested mobile detector/recognizer,
    native 960-side input, fixed .3 thresholds, six CPU threads, harmonic
    ordering, and two release roots as V8.  The distinct schema/engine tuple
    alone instructs the direct sidecar to detect the original RGB page in two
    overlapping horizontal strips.  There is deliberately no user-configured
    tile, overlap, or NMS setting in this profile.
    """

    # Pydantic supports narrowing inherited Literal fields for this distinct
    # wire schema; mypy cannot express that runtime field replacement.
    schema_version: Literal["9.0"] = "9.0"  # type: ignore[assignment]
    engine_kind: Literal["direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2"] = (
        cast(  # type: ignore[assignment]
            Any, "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2"
        )
    )


class RasterTextOfflineProfileV10(RasterTextOfflineProfileV8):
    """Immutable V8-compatible candidate with fixed 2x2 detector tiles.

    V10 retains the mobile model pair, native 960-side limit, fixed .3
    thresholds, six CPU threads, harmonic order, and both attested release
    roots.  Its distinct schema/engine tuple alone selects four overlapping
    detector tiles.  Tile geometry and cross-tile NMS are sidecar semantics,
    never profile inputs or environment-configured knobs.
    """

    # Pydantic supports narrowing inherited Literal fields for this distinct
    # wire schema; mypy cannot express that runtime field replacement.
    schema_version: Literal["10.0"] = "10.0"  # type: ignore[assignment]
    engine_kind: Literal["direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2"] = cast(  # type: ignore[assignment]
        Any, "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2"
    )


class RasterTextOfflineProfileV11(RasterTextOfflineProfileV8):
    """Immutable V10-compatible candidate with guarded fixed-grid NMS.

    V11 retains the exact V10 model pair, native 960 detector limit, fixed
    thresholds, six CPU threads, harmonic ordering, and 15%/10% grid.  Its
    distinct schema/engine tuple alone selects the sidecar's fixed internal
    seam marker and guarded cross-tile suppression; no profile field or
    environment variable can relax those semantics.
    """

    # Pydantic supports narrowing inherited Literal fields for this distinct
    # wire schema; mypy cannot express that runtime field replacement.
    schema_version: Literal["11.0"] = "11.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2"
    ] = cast(  # type: ignore[assignment]
        Any, "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2"
    )


class RasterTextOfflineProfileV12(_RuntimeWireModel):
    """Source-attested recognizer with V9's exact bounded strip2 semantics."""

    schema_version: Literal["12.0"] = "12.0"
    engine_kind: Literal[
        "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2"
    ] = "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2"
    runtime_api: Literal["paddle-inference-static-v1"] = "paddle-inference-static-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    profile_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    combined_model_bundle_root_relative_path: str = Field(min_length=1, max_length=240)
    combined_model_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_detector_bundle_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_detector_model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_detector_model_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_training_intent_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_character_dictionary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    pdfium_version: str = Field(min_length=1, max_length=128)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: Literal[30] = 30
    cpu_threads: Literal[6] = 6
    text_detection_limit_side_len: Literal[960] = 960
    text_detection_limit_type: Literal["max"] = "max"
    text_detection_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_box_threshold: float = Field(default=0.3, ge=0.3, le=0.3)
    text_detection_unclip_ratio: float = Field(default=1.5, ge=1.5, le=1.5)
    minimum_page_confidence: float = Field(ge=0, le=1)

    parent_renderer_package_inventory_path: str = Field(min_length=1, max_length=240)
    parent_renderer_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_renderer_required_packages: list[str] = Field(min_length=1, max_length=32)
    sidecar_interpreter_path: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_interpreter_root: str = Field(min_length=1, max_length=240)
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_path: str = Field(min_length=1, max_length=240)
    sidecar_module_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_module_root: str = Field(min_length=1, max_length=240)
    sidecar_module_root_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_package_inventory_path: str = Field(min_length=1, max_length=240)
    sidecar_package_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sidecar_required_packages: list[str] = Field(min_length=1, max_length=64)

    local_model_root: str = Field(min_length=1, max_length=240)
    model_inventory_path: str = Field(min_length=1, max_length=240)
    model_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_detection_model_directory: Literal["text-det"] = "text-det"
    text_recognition_model_name: Literal["source_attested_tiny_crnn_bigru_ctc_v1"] = (
        "source_attested_tiny_crnn_bigru_ctc_v1"
    )
    text_recognition_model_directory: Literal["text-rec"] = "text-rec"

    @model_validator(mode="after")
    def _exact_source_attested_binding(self) -> "RasterTextOfflineProfileV12":
        for value in (
            self.bundle_root_relative_path,
            self.combined_model_bundle_root_relative_path,
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
            self.local_model_root,
            self.model_inventory_path,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
        ):
            _relative_path(value)
        if (
            len(set(self.parent_renderer_required_packages))
            != len(self.parent_renderer_required_packages)
            or len(set(self.sidecar_required_packages)) != len(self.sidecar_required_packages)
            or tuple(self.parent_renderer_required_packages) != ("pypdfium2", "Pillow")
            or tuple(self.sidecar_required_packages)
            != ("numpy", "opencv-contrib-python", "paddlepaddle", "pyclipper")
            or self.sidecar_timeout_seconds != 30
            or self.cpu_threads != 6
            or self.text_detection_limit_side_len != 960
            or self.text_detection_limit_type != "max"
            or self.text_detection_threshold != 0.3
            or self.text_detection_box_threshold != 0.3
            or self.text_detection_unclip_ratio != 1.5
            or self.text_detection_model_name != "PP-OCRv5_mobile_det"
            or self.text_detection_model_directory != "text-det"
            or self.text_recognition_model_name != "source_attested_tiny_crnn_bigru_ctc_v1"
            or self.text_recognition_model_directory != "text-rec"
            or self.source_recognizer_release_sha256 != self.source_recognizer_receipt_sha256
        ):
            raise ValueError("raster text V12 source-attested candidate is invalid")
        expected_sidecar_root = (
            f"{_TEXT_DIRECT_PADDLE_BUNDLE_RELEASE_PREFIX}/{self.bundle_manifest_sha256}"
        )
        expected_model_bundle_root = (
            f"{_TEXT_TRAINED_PADDLE_BUNDLE_RELEASE_PREFIX}/"
            f"{self.combined_model_bundle_manifest_sha256}"
        )
        if (
            self.bundle_root_relative_path != expected_sidecar_root
            or self.combined_model_bundle_root_relative_path != expected_model_bundle_root
            or self.local_model_root != f"{expected_model_bundle_root}/models"
            or self.model_inventory_path
            != f"{expected_model_bundle_root}/inventories/direct-paddle-trained-models.json"
        ):
            raise ValueError("raster text V12 bundle binding is invalid")
        fingerprint_payload = self.model_dump(
            mode="json",
            warnings=False,
            exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
        )
        fingerprint = raster_text_profile_fingerprint_sha256(fingerprint_payload)
        if self.profile_fingerprint_sha256 != fingerprint or self.raster_text_profile_revision != (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        ):
            raise ValueError("raster text V12 fingerprint binding is invalid")
        sidecar_root = PurePosixPath(self.bundle_root_relative_path)
        model_bundle_root = PurePosixPath(self.combined_model_bundle_root_relative_path)
        for value in (
            self.parent_renderer_package_inventory_path,
            self.sidecar_interpreter_path,
            self.sidecar_interpreter_root,
            self.sidecar_module_path,
            self.sidecar_module_root,
            self.sidecar_package_inventory_path,
        ):
            try:
                relative = PurePosixPath(value).relative_to(sidecar_root)
            except ValueError:
                raise ValueError("raster text V12 sidecar binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text V12 sidecar binding is invalid")
        for value in (self.local_model_root, self.model_inventory_path):
            try:
                relative = PurePosixPath(value).relative_to(model_bundle_root)
            except ValueError:
                raise ValueError("raster text V12 model binding is invalid") from None
            if relative == PurePosixPath("."):
                raise ValueError("raster text V12 model binding is invalid")
        try:
            interpreter_relative = PurePosixPath(self.sidecar_interpreter_path).relative_to(
                PurePosixPath(self.sidecar_interpreter_root)
            )
            module_relative = PurePosixPath(self.sidecar_module_path).relative_to(
                PurePosixPath(self.sidecar_module_root)
            )
            if interpreter_relative == PurePosixPath(".") or module_relative == PurePosixPath("."):
                raise ValueError
        except ValueError:
            raise ValueError("raster text V12 sidecar root is invalid") from None
        return self


class RasterTextOfflineProfileV13(RasterTextOfflineProfileV9):
    """Public V9 OCR tuple with one fixed conservative semantic repair.

    The public mobile detector and recognizer, strip2 pass, harmonic order,
    native-960 input, thresholds, CPU threads, timeout, and both attested
    roots remain byte-for-byte V9-compatible.  Only this distinct immutable
    discriminator allows the sidecar's bounded supported-document repair;
    no profile field can enable or tune it.
    """

    schema_version: Literal["13.0"] = "13.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair"
    ] = cast(  # type: ignore[assignment]
        Any,
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
    )


class RasterTextOfflineProfileV14(RasterTextOfflineProfileV9):
    """V13-compatible OCR with one additional closed-shape repair.

    V14 retains the exact public V9 detector, recognizer, strip2 pass,
    harmonic order, native-960 input, thresholds, CPU threads, timeout and
    attested roots used by V13.  Its distinct immutable discriminator alone
    enables the second bounded semantic-repair revision in the sidecar; no
    profile or request field can tune that repair.
    """

    schema_version: Literal["14.0"] = "14.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2"
    ] = cast(  # type: ignore[assignment]
        Any,
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2",
    )


class RasterTextOfflineProfileV15(RasterTextOfflineProfileV9):
    """V14 semantics with bounded, fingerprinted oneDNN CPU controls.

    V15 does not inherit V14 intentionally.  Activation pointer schema 4 is
    restricted to an actual V14 instance, while staged schema 5 requires an
    actual V15.  This profile can still reuse V9's verified public model roots
    and strip2 geometry.  Every native control is part of the profile
    fingerprint; older profiles reject these fields in the sidecar rather
    than silently opting into oneDNN.
    """

    schema_version: Literal["15.0"] = "15.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
        "semantic_repair_v2_onednn"
    ] = cast(  # type: ignore[assignment]
        Any,
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
        "semantic_repair_v2_onednn",
    )
    cpu_threads: Literal[4, 6] = 6  # type: ignore[assignment]
    enable_mkldnn: Literal[True] = True
    mkldnn_cache_capacity: Literal[10] = 10
    recognition_batch_size: Literal[8, 16] = 8
    opencv_threads: Literal[1] = 1
    warmup_shape_policy: Literal["detector-736x960-recognizer-48x320-v1"] = (
        "detector-736x960-recognizer-48x320-v1"
    )
    onednn_model_policy: Literal["recognizer-only-v1"] = "recognizer-only-v1"

    @model_validator(mode="after")
    def _bounded_onednn_controls(self) -> "RasterTextOfflineProfileV15":
        if (
            self.cpu_threads not in {4, 6}
            or self.enable_mkldnn is not True
            or self.mkldnn_cache_capacity != 10
            or self.recognition_batch_size not in {8, 16}
            or self.opencv_threads != 1
            or self.warmup_shape_policy != "detector-736x960-recognizer-48x320-v1"
            or self.onednn_model_policy != "recognizer-only-v1"
        ):
            raise ValueError("raster text V15 oneDNN controls are invalid")
        return self


class RasterTextOfflineProfileV16(RasterTextOfflineProfileV9):
    """V15 native controls with one immutable cubic recognition resize.

    V16 is independent of V15 so the existing staged-activation type boundary
    remains unchanged until this candidate has its own evaluation evidence.
    The crop policy preserves the detector quad exactly; only the recognizer
    resize interpolation is selected by this new immutable profile tuple.
    """

    schema_version: Literal["16.0"] = "16.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
        "semantic_repair_v2_onednn_cubic_recognition"
    ] = cast(  # type: ignore[assignment]
        Any,
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
        "semantic_repair_v2_onednn_cubic_recognition",
    )
    cpu_threads: Literal[4, 6] = 6  # type: ignore[assignment]
    enable_mkldnn: Literal[True] = True
    mkldnn_cache_capacity: Literal[10] = 10
    recognition_batch_size: Literal[8, 16] = 8
    opencv_threads: Literal[1] = 1
    warmup_shape_policy: Literal["detector-736x960-recognizer-48x320-v1"] = (
        "detector-736x960-recognizer-48x320-v1"
    )
    onednn_model_policy: Literal["recognizer-only-v1"] = "recognizer-only-v1"
    recognition_crop_policy: Literal["detector-quad-v1"] = "detector-quad-v1"
    recognition_resize_interpolation: Literal["cubic-v1"] = "cubic-v1"

    @model_validator(mode="after")
    def _bounded_v16_controls(self) -> "RasterTextOfflineProfileV16":
        if (
            self.cpu_threads not in {4, 6}
            or self.enable_mkldnn is not True
            or self.mkldnn_cache_capacity != 10
            or self.recognition_batch_size not in {8, 16}
            or self.opencv_threads != 1
            or self.warmup_shape_policy != "detector-736x960-recognizer-48x320-v1"
            or self.onednn_model_policy != "recognizer-only-v1"
            or self.recognition_crop_policy != "detector-quad-v1"
            or self.recognition_resize_interpolation != "cubic-v1"
        ):
            raise ValueError("raster text V16 controls are invalid")
        return self


class RasterTextOfflineProfileV17(RasterTextOfflineProfileV9):
    """V15 native controls with one immutable horizontal crop expansion."""

    schema_version: Literal["17.0"] = "17.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
        "semantic_repair_v2_onednn_padded_recognition"
    ] = cast(  # type: ignore[assignment]
        Any,
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
        "semantic_repair_v2_onednn_padded_recognition",
    )
    cpu_threads: Literal[4, 6] = 6  # type: ignore[assignment]
    enable_mkldnn: Literal[True] = True
    mkldnn_cache_capacity: Literal[10] = 10
    recognition_batch_size: Literal[8, 16] = 8
    opencv_threads: Literal[1] = 1
    warmup_shape_policy: Literal["detector-736x960-recognizer-48x320-v1"] = (
        "detector-736x960-recognizer-48x320-v1"
    )
    onednn_model_policy: Literal["recognizer-only-v1"] = "recognizer-only-v1"
    recognition_crop_policy: Literal["horizontal-height-0p08-max4-v1"] = (
        "horizontal-height-0p08-max4-v1"
    )
    recognition_resize_interpolation: Literal["linear-v1"] = "linear-v1"

    @model_validator(mode="after")
    def _bounded_v17_controls(self) -> "RasterTextOfflineProfileV17":
        if (
            self.cpu_threads not in {4, 6}
            or self.enable_mkldnn is not True
            or self.mkldnn_cache_capacity != 10
            or self.recognition_batch_size not in {8, 16}
            or self.opencv_threads != 1
            or self.warmup_shape_policy != "detector-736x960-recognizer-48x320-v1"
            or self.onednn_model_policy != "recognizer-only-v1"
            or self.recognition_crop_policy != "horizontal-height-0p08-max4-v1"
            or self.recognition_resize_interpolation != "linear-v1"
        ):
            raise ValueError("raster text V17 controls are invalid")
        return self


class RasterTextOfflineProfileV18(RasterTextOfflineProfileV9):
    """V15 detector/runtime semantics with the pinned server recognizer.

    The recognizer is deliberately bound through a separate combined-model
    release.  No cache path or caller field can swap it into a historical
    mobile profile, and the detector remains the measured mobile strip2
    engine rather than the substantially heavier server detector.
    """

    schema_version: Literal["18.0"] = "18.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_det_server_rec_harmonic_recall_native960_lowbox_"
        "strip2_semantic_repair_v2_onednn"
    ] = cast(  # type: ignore[assignment]
        Any,
        "direct_paddle_mobile_det_server_rec_harmonic_recall_native960_lowbox_"
        "strip2_semantic_repair_v2_onednn",
    )
    text_recognition_model_name: Literal["PP-OCRv5_server_rec"] = cast(  # type: ignore[assignment]
        Any, "PP-OCRv5_server_rec"
    )
    cpu_threads: Literal[4, 6] = 6  # type: ignore[assignment]
    enable_mkldnn: Literal[True] = True
    mkldnn_cache_capacity: Literal[10] = 10
    recognition_batch_size: Literal[8] = 8
    opencv_threads: Literal[1] = 1
    warmup_shape_policy: Literal["detector-736x960-recognizer-48x320-v1"] = (
        "detector-736x960-recognizer-48x320-v1"
    )
    onednn_model_policy: Literal["recognizer-only-v1"] = "recognizer-only-v1"
    recognition_crop_policy: Literal["detector-quad-v1"] = "detector-quad-v1"
    recognition_resize_interpolation: Literal["linear-v1"] = "linear-v1"
    ctc_label_policy: Literal["append-space-v1"] = "append-space-v1"

    @model_validator(mode="after")
    def _bounded_v18_controls(self) -> "RasterTextOfflineProfileV18":
        model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        try:
            source_provenance = PurePosixPath(
                self.source_model_inventory_relative_path
            ).relative_to(model_root)
        except ValueError:
            raise ValueError("raster text V18 model provenance is invalid") from None
        if (
            self.cpu_threads not in {4, 6}
            or self.enable_mkldnn is not True
            or self.mkldnn_cache_capacity != 10
            or self.recognition_batch_size != 8
            or self.opencv_threads != 1
            or self.warmup_shape_policy != "detector-736x960-recognizer-48x320-v1"
            or self.onednn_model_policy != "recognizer-only-v1"
            or self.recognition_crop_policy != "detector-quad-v1"
            or self.recognition_resize_interpolation != "linear-v1"
            or self.ctc_label_policy != "append-space-v1"
            or self.text_recognition_model_name != "PP-OCRv5_server_rec"
            or source_provenance == PurePosixPath(".")
        ):
            raise ValueError("raster text V18 controls are invalid")
        return self


class RasterTextOfflineProfileV19(RasterTextOfflineProfileV9):
    """V15's fast lane with one source-attested public fine-tuned recognizer.

    V19 keeps the mobile detector, strip2 geometry, semantic repair, oneDNN
    recognizer-only policy and fixed request bounds.  The combined release
    independently binds the public warm-start training receipt and exact
    438-class English CTC contract; calibration or sealed Gold can never be
    used as serving material.
    """

    schema_version: Literal["19.0"] = "19.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_onednn",
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn",
    ] = cast(  # type: ignore[assignment]
        Any,
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_onednn",
    )
    text_recognition_model_name: Literal["source_attested_en_PP-OCRv5_mobile_rec_head_v2"] = cast(  # type: ignore[assignment]
        Any, "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
    )
    cpu_threads: Literal[4, 6] = 6  # type: ignore[assignment]
    enable_mkldnn: Literal[True] = True
    mkldnn_cache_capacity: Literal[10] = 10
    recognition_batch_size: Literal[8] = 8
    opencv_threads: Literal[1] = 1
    warmup_shape_policy: Literal["detector-736x960-recognizer-48x320-v1"] = (
        "detector-736x960-recognizer-48x320-v1"
    )
    onednn_model_policy: Literal["recognizer-only-v1"] = "recognizer-only-v1"
    ctc_label_policy: Literal["optional-space-v1"] = "optional-space-v1"
    source_recognizer_release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_training_intent_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    source_recognizer_selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_parent_selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_train_corpus_provenance: RasterTextPublicFineTuneCorpusProvenanceV1
    source_recognizer_development_corpus_provenance: RasterTextPublicFineTuneCorpusProvenanceV1
    source_recognizer_base_dictionary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_output_class_count: Literal[438] = 438

    @model_validator(mode="after")
    def _bounded_v19_controls(self) -> "RasterTextOfflineProfileV19":
        # V20 subclasses this model solely to reuse V19's fixed CPU/runtime
        # controls.  Its recognizer and provenance contract are deliberately
        # different and are checked by V20's own validator below.
        if self.schema_version != "19.0":
            return self
        model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        try:
            receipt_path = PurePosixPath(self.source_model_inventory_relative_path).relative_to(
                model_root
            )
        except ValueError:
            raise ValueError("raster text V19 provenance is invalid") from None
        if (
            receipt_path == PurePosixPath(".")
            or self.cpu_threads not in {4, 6}
            or self.enable_mkldnn is not True
            or self.mkldnn_cache_capacity != 10
            or self.recognition_batch_size != 8
            or self.opencv_threads != 1
            or self.warmup_shape_policy != "detector-736x960-recognizer-48x320-v1"
            or self.onednn_model_policy != "recognizer-only-v1"
            or self.ctc_label_policy != "optional-space-v1"
            or self.text_recognition_model_name != "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
            or self.source_recognizer_output_class_count != 438
            or self.source_recognizer_train_corpus_provenance.split != "train"
            or self.source_recognizer_development_corpus_provenance.split != "development"
            or self.source_recognizer_selection_receipt_relative_path
            != (
                "synthetic-raster-text-v1/fresh-scope-selections/"
                f"{self.source_recognizer_selection_receipt_sha256}.json"
            )
        ):
            raise ValueError("raster text V19 controls are invalid")
        return self


class RasterTextOfflineProfileV20(RasterTextOfflineProfileV19):
    """V19 CPU lane bound to the leakage-safe voyage-replay head V3.

    The detector, static runtime, strip2 geometry, oneDNN policy and 30-second
    request bound remain unchanged.  Only the source-attested recognizer and
    its deterministic horizontal crop policy change.  Full content-free
    train/development manifests are fingerprinted into this profile so a
    serving bundle cannot substitute a different replay corpus.
    """

    schema_version: Literal["20.0"] = "20.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_onednn"
    ] = cast(Any, _V20_ENGINE_KIND)  # type: ignore[assignment]
    text_recognition_model_name: Literal[
        "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
    ] = cast(  # type: ignore[assignment]
        Any, "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
    )
    recognition_crop_policy: Literal["horizontal-height-0p32-max8-v1"] = (
        "horizontal-height-0p32-max8-v1"
    )
    recognition_resize_interpolation: Literal["linear-v1"] = "linear-v1"

    # These V19-only fields are excluded from V20's canonical wire form.  The
    # replacement manifests below carry the stronger family/lane/audit proof.
    source_recognizer_train_corpus_provenance: None = Field(  # type: ignore[assignment]
        default=None, exclude=True
    )
    source_recognizer_development_corpus_provenance: None = Field(  # type: ignore[assignment]
        default=None, exclude=True
    )
    source_recognizer_train_corpus_manifest: VoyageReplayCorpusManifestV1
    source_recognizer_development_corpus_manifest: VoyageReplayCorpusManifestV1
    source_recognizer_train_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_development_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_parent_release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_parent_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_voyage_recipe_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _bounded_v20_controls(self) -> "RasterTextOfflineProfileV20":
        model_root = PurePosixPath(self.mobile_model_bundle_root_relative_path)
        try:
            receipt_path = PurePosixPath(self.source_model_inventory_relative_path).relative_to(
                model_root
            )
        except ValueError:
            raise ValueError("raster text V20 provenance is invalid") from None
        train = self.source_recognizer_train_corpus_manifest
        development = self.source_recognizer_development_corpus_manifest
        if (
            receipt_path == PurePosixPath(".")
            or self.schema_version != "20.0"
            or self.engine_kind != _V20_ENGINE_KIND
            or self.cpu_threads not in {4, 6}
            or self.enable_mkldnn is not True
            or self.mkldnn_cache_capacity != 10
            or self.recognition_batch_size != 8
            or self.opencv_threads != 1
            or self.warmup_shape_policy != "detector-736x960-recognizer-48x320-v1"
            or self.onednn_model_policy != "recognizer-only-v1"
            or self.ctc_label_policy != "optional-space-v1"
            or self.recognition_crop_policy != "horizontal-height-0p32-max8-v1"
            or self.recognition_resize_interpolation != "linear-v1"
            or self.text_recognition_model_name
            != "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
            or self.source_recognizer_output_class_count != 438
            or train.split != "train"
            or train.purpose != "optimizer"
            or development.split != "development"
            or development.purpose != "validation-only"
            or self.source_recognizer_train_manifest_sha256
            != hashlib.sha256(
                _canonical_evaluation_json_bytes(train.model_dump(mode="json", warnings=False))
            ).hexdigest()
            or self.source_recognizer_development_manifest_sha256
            != hashlib.sha256(
                _canonical_evaluation_json_bytes(
                    development.model_dump(mode="json", warnings=False)
                )
            ).hexdigest()
            or train.source_receipt_sha256 != development.source_receipt_sha256
            or train.raster_receipt_sha256 != development.raster_receipt_sha256
            or train.selection_receipt_relative_path != development.selection_receipt_relative_path
            or train.selection_receipt_sha256 != development.selection_receipt_sha256
            or train.parent_selection_receipt_sha256 != development.parent_selection_receipt_sha256
            or self.source_recognizer_selection_receipt_relative_path
            != train.selection_receipt_relative_path
            or self.source_recognizer_selection_receipt_sha256 != train.selection_receipt_sha256
            or self.source_recognizer_parent_selection_receipt_sha256
            != train.parent_selection_receipt_sha256
            or self.source_recognizer_voyage_recipe_set_sha256 != train.voyage_recipe_set_sha256
            or train.voyage_recipe_set_sha256 != development.voyage_recipe_set_sha256
        ):
            raise ValueError("raster text V20 controls are invalid")
        return self


RasterTextOfflineProfile = (
    RasterTextOfflineProfileV1
    | RasterTextOfflineProfileV2
    | RasterTextOfflineProfileV3
    | RasterTextOfflineProfileV4
    | RasterTextOfflineProfileV5
    | RasterTextOfflineProfileV6
    | RasterTextOfflineProfileV7
    | RasterTextOfflineProfileV8
    | RasterTextOfflineProfileV9
    | RasterTextOfflineProfileV10
    | RasterTextOfflineProfileV11
    | RasterTextOfflineProfileV12
    | RasterTextOfflineProfileV13
    | RasterTextOfflineProfileV14
    | RasterTextOfflineProfileV15
    | RasterTextOfflineProfileV16
    | RasterTextOfflineProfileV17
    | RasterTextOfflineProfileV18
    | RasterTextOfflineProfileV19
    | RasterTextOfflineProfileV20
)


def _sidecar_bootstrap_for_profile(profile: RasterTextOfflineProfile) -> str:
    """Return the isolated bootstrap bound to this immutable profile kind.

    The direct static-inference V2 launch is the sole user of the
    short-spelling-safe bootstrap.  Keeping this dispatch next to the schema
    union makes an accidental V1 or future profile fallback impossible at the
    three native-process launch sites below.
    """

    if isinstance(profile, RasterTextOfflineProfileV20):
        return _DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP
    if isinstance(
        profile,
        (
            RasterTextOfflineProfileV2,
            RasterTextOfflineProfileV3,
            RasterTextOfflineProfileV4,
            RasterTextOfflineProfileV5,
            RasterTextOfflineProfileV6,
            RasterTextOfflineProfileV7,
            RasterTextOfflineProfileV8,
            RasterTextOfflineProfileV9,
            RasterTextOfflineProfileV10,
            RasterTextOfflineProfileV11,
            RasterTextOfflineProfileV12,
            RasterTextOfflineProfileV13,
            RasterTextOfflineProfileV14,
            RasterTextOfflineProfileV15,
            RasterTextOfflineProfileV16,
            RasterTextOfflineProfileV17,
            RasterTextOfflineProfileV18,
            RasterTextOfflineProfileV19,
        ),
    ):
        return _DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP
    return _ISOLATED_SIDECAR_BOOTSTRAP


def raster_text_service_startup_timeout_seconds(profile: RasterTextOfflineProfile) -> int:
    """Return the fixed cold-ready bound for one immutable provider profile.

    This is deliberately not derived from ``sidecar_timeout_seconds``.  That
    value is the post-ready document request bound, while native imports and
    model construction happen once before the child publishes ``ready``.
    V1 retains its existing 55-second setup contract; only the V2 direct
    static-inference profile receives the separately attested 60-second
    setup budget.
    """

    if isinstance(profile, RasterTextOfflineProfileV20):
        return _V20_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV2):
        return _V2_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV3):
        return _V3_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV4):
        return _V4_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV5):
        return _V5_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV6):
        return _V6_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV7):
        return _V7_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(
        profile,
        (
            RasterTextOfflineProfileV15,
            RasterTextOfflineProfileV16,
            RasterTextOfflineProfileV17,
            RasterTextOfflineProfileV18,
            RasterTextOfflineProfileV19,
        ),
    ):
        return _V15_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV14):
        return _V14_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV13):
        return _V13_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV8):
        return _V8_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV9):
        return _V9_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV10):
        return _V10_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV11):
        return _V11_TEXT_SERVICE_STARTUP_SECONDS
    if isinstance(profile, RasterTextOfflineProfileV12):
        return _V12_TEXT_SERVICE_STARTUP_SECONDS
    return _V1_TEXT_SERVICE_STARTUP_SECONDS


def parse_raster_text_offline_profile(raw: bytes) -> RasterTextOfflineProfile:
    """Strictly dispatch the immutable text profile by its schema version."""

    try:
        discriminator = json.loads(raw.decode("utf-8")).get("schema_version")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("raster text profile schema is invalid") from None
    if discriminator == "1.0":
        return RasterTextOfflineProfileV1.model_validate_json(raw)
    if discriminator == "2.0":
        return RasterTextOfflineProfileV2.model_validate_json(raw)
    if discriminator == "3.0":
        return RasterTextOfflineProfileV3.model_validate_json(raw)
    if discriminator == "4.0":
        return RasterTextOfflineProfileV4.model_validate_json(raw)
    if discriminator == "5.0":
        return RasterTextOfflineProfileV5.model_validate_json(raw)
    if discriminator == "6.0":
        return RasterTextOfflineProfileV6.model_validate_json(raw)
    if discriminator == "7.0":
        return RasterTextOfflineProfileV7.model_validate_json(raw)
    if discriminator == "8.0":
        return RasterTextOfflineProfileV8.model_validate_json(raw)
    if discriminator == "9.0":
        return RasterTextOfflineProfileV9.model_validate_json(raw)
    if discriminator == "10.0":
        return RasterTextOfflineProfileV10.model_validate_json(raw)
    if discriminator == "11.0":
        return RasterTextOfflineProfileV11.model_validate_json(raw)
    if discriminator == "12.0":
        return RasterTextOfflineProfileV12.model_validate_json(raw)
    if discriminator == "13.0":
        return RasterTextOfflineProfileV13.model_validate_json(raw)
    if discriminator == "14.0":
        return RasterTextOfflineProfileV14.model_validate_json(raw)
    if discriminator == "15.0":
        return RasterTextOfflineProfileV15.model_validate_json(raw)
    if discriminator == "16.0":
        return RasterTextOfflineProfileV16.model_validate_json(raw)
    if discriminator == "17.0":
        return RasterTextOfflineProfileV17.model_validate_json(raw)
    if discriminator == "18.0":
        return RasterTextOfflineProfileV18.model_validate_json(raw)
    if discriminator == "19.0":
        return RasterTextOfflineProfileV19.model_validate_json(raw)
    if discriminator == "20.0":
        return RasterTextOfflineProfileV20.model_validate_json(raw)
    raise ValueError("raster text profile schema is invalid")


class RasterTextActivationPointerV1(_RuntimeWireModel):
    """Automatic source-attested pointer; no human approval field exists."""

    schema_version: Literal["1.0"] = "1.0"
    activation_kind: Literal["source_attested_synthetic_raster_text"] = (
        "source_attested_synthetic_raster_text"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextActivationPointerV1":
        _relative_path(self.runtime_configuration_path)
        _relative_path(self.sealed_evaluation_path)
        return self


class RasterTextActivationPointerV2(_RuntimeWireModel):
    """Semantic-metric activation pointer with an explicit evidence binding."""

    schema_version: Literal["2.0"] = "2.0"
    activation_kind: Literal["source_attested_synthetic_raster_text"] = (
        "source_attested_synthetic_raster_text"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v2"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextActivationPointerV2":
        _relative_path(self.runtime_configuration_path)
        _relative_path(self.sealed_evaluation_path)
        return self


class RasterTextActivationPointerV3(_RuntimeWireModel):
    """Semantic-metric V3 activation pointer with an explicit binding."""

    schema_version: Literal["3.0"] = "3.0"
    activation_kind: Literal["source_attested_synthetic_raster_text"] = (
        "source_attested_synthetic_raster_text"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextActivationPointerV3":
        _relative_path(self.runtime_configuration_path)
        _relative_path(self.sealed_evaluation_path)
        return self


class RasterTextActivationPointerV4(_RuntimeWireModel):
    """Fresh-holdout semantic-V3 pointer with an explicit generation."""

    schema_version: Literal["4.0"] = "4.0"
    activation_kind: Literal["source_attested_synthetic_raster_text"] = (
        "source_attested_synthetic_raster_text"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-holdout-v1"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextActivationPointerV4":
        _relative_path(self.runtime_configuration_path)
        _relative_path(self.sealed_evaluation_path)
        return self


class RasterTextActivationPointerV5(_RuntimeWireModel):
    """Projected/staged fresh-holdout pointer for the bounded V15 lane."""

    schema_version: Literal["5.0"] = "5.0"
    activation_kind: Literal["source_attested_synthetic_raster_text"] = (
        "source_attested_synthetic_raster_text"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-staged-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextActivationPointerV5":
        _relative_path(self.runtime_configuration_path)
        _relative_path(self.sealed_evaluation_path)
        return self


class RasterTextActivationPointerV6(_RuntimeWireModel):
    """Leakage-safe staged pointer for the bounded V18 server-rec lane."""

    schema_version: Literal["6.0"] = "6.0"
    activation_kind: Literal["source_attested_synthetic_raster_text"] = (
        "source_attested_synthetic_raster_text"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-staged-train-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextActivationPointerV6":
        _relative_path(self.runtime_configuration_path)
        _relative_path(self.sealed_evaluation_path)
        return self


class RasterTextActivationPointerV7(_RuntimeWireModel):
    """Leakage-safe V19 pointer bound to one public fine-tuned recognizer."""

    schema_version: Literal["7.0"] = "7.0"
    activation_kind: Literal["source_attested_synthetic_raster_text"] = (
        "source_attested_synthetic_raster_text"
    )
    status: Literal["activated"] = "activated"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-public-finetuned-staged-train-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_path: str = Field(min_length=1, max_length=240)
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sealed_evaluation_path: str = Field(min_length=1, max_length=240)
    sealed_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_training_intent_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_selection_receipt_path: str = Field(min_length=1, max_length=240)
    source_recognizer_selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_parent_selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextActivationPointerV7":
        _relative_path(self.runtime_configuration_path)
        _relative_path(self.sealed_evaluation_path)
        _relative_path(self.source_recognizer_selection_receipt_path)
        return self


class RasterTextActivationPointerV8(RasterTextActivationPointerV7):
    """V19 pointer additionally bound to an independent B/L repair audit."""

    schema_version: Literal["8.0"] = "8.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
    ]
    repair_audit_revision: Literal["source-attested-independent-bl-repair-audit-v1"] = (
        "source-attested-independent-bl-repair-audit-v1"
    )
    repair_audit_dataset_receipt_path: str = Field(min_length=1, max_length=240)
    repair_audit_dataset_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    repair_audit_evaluation_path: str = Field(min_length=1, max_length=240)
    repair_audit_evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _repair_audit_relative_paths(self) -> "RasterTextActivationPointerV8":
        _relative_path(self.repair_audit_dataset_receipt_path)
        _relative_path(self.repair_audit_evaluation_path)
        if self.repair_audit_dataset_receipt_path != "synthetic-bl-repair-audit-v1/receipt.json":
            raise ValueError("raster text repair audit receipt path is invalid")
        expected_evaluation_path = (
            f"synthetic-bl-repair-audit-v1/evaluations/{self.repair_audit_evaluation_sha256}.json"
        )
        if self.repair_audit_evaluation_path != expected_evaluation_path:
            raise ValueError("raster text repair audit evaluation path is invalid")
        return self


class RasterTextActivationPointerV9(RasterTextActivationPointerV8):
    """V20 pointer bound to replay/parent provenance and the independent audit."""

    schema_version: Literal["9.0"] = "9.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_onednn"
    ] = cast(Any, _V20_ENGINE_KIND)  # type: ignore[assignment]
    source_recognizer_parent_release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_parent_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_voyage_recipe_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_train_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_development_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _v20_engine_is_exact(self) -> "RasterTextActivationPointerV9":
        if self.engine_kind != _V20_ENGINE_KIND:
            raise ValueError("raster text V9 activation engine is invalid")
        return self


class RasterTextActivationPointerV10(RasterTextActivationPointerV7):
    """Preview-draft V19 pointer with an explicit 97% revalidation chain."""

    schema_version: Literal["10.0"] = "10.0"  # type: ignore[assignment]
    engine_kind: Literal[
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_onednn"
    ] = cast(Any, _V19_ENGINE_KIND)
    quality_gate_revision: Literal["preview-draft-critical97-cer2-v1"] = (
        "preview-draft-critical97-cer2-v1"
    )
    evidence_revalidation_revision: Literal["historical-runner-current-runtime-smoke-v1"] = (
        "historical-runner-current-runtime-smoke-v1"
    )
    historical_runner_implementation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    current_runner_implementation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revalidation_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    revalidation_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revalidation_candidate_execution_path: str = Field(min_length=1, max_length=240)
    revalidation_candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _v19_revalidation_is_exact(self) -> "RasterTextActivationPointerV10":
        if self.engine_kind != _V19_ENGINE_KIND:
            raise ValueError("raster text V10 activation engine is invalid")
        _relative_path(self.revalidation_smoke_receipt_path)
        _relative_path(self.revalidation_candidate_execution_path)
        return self


class RasterTextActivationPointerV11(RasterTextActivationPointerV10):
    """V10 quality evidence with activation-full/runtime-compact validation."""

    schema_version: Literal["11.0"] = "11.0"  # type: ignore[assignment]
    runtime_validation_revision: Literal["activation-full-runtime-compact-v1"] = (
        "activation-full-runtime-compact-v1"
    )


RasterTextActivationPointer = (
    RasterTextActivationPointerV1
    | RasterTextActivationPointerV2
    | RasterTextActivationPointerV3
    | RasterTextActivationPointerV4
    | RasterTextActivationPointerV5
    | RasterTextActivationPointerV6
    | RasterTextActivationPointerV7
    | RasterTextActivationPointerV8
    | RasterTextActivationPointerV9
    | RasterTextActivationPointerV10
    | RasterTextActivationPointerV11
)


class RasterTextSealedProfileScoreV1(_RuntimeWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextCalibrationReceiptV1(_RuntimeWireModel):
    """Immutable confidence selection evidence consumed by the runtime."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["calibration"] = "calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV1":
        _relative_path(self.source_gold_receipt_path)
        _relative_path(self.candidate_execution_path)
        return self


class RasterTextCalibrationReceiptV2(_RuntimeWireModel):
    """Confidence evidence explicitly measured with semantic text metric V2."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_calibration"] = (
        "synthetic_raster_semantic_text_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["calibration"] = "calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v2"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV2":
        _relative_path(self.source_gold_receipt_path)
        _relative_path(self.candidate_execution_path)
        return self


class RasterTextCalibrationReceiptV3(_RuntimeWireModel):
    """Confidence evidence explicitly measured with semantic text metric V3."""

    schema_version: Literal["3.0"] = "3.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_calibration"] = (
        "synthetic_raster_semantic_text_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["calibration"] = "calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV3":
        _relative_path(self.source_gold_receipt_path)
        _relative_path(self.candidate_execution_path)
        return self


class RasterTextCalibrationReceiptV4(_RuntimeWireModel):
    """Fresh calibration evidence for semantic metric V3."""

    schema_version: Literal["4.0"] = "4.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_calibration"] = (
        "synthetic_raster_semantic_text_fresh_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-calibration"] = "fresh-calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-holdout-v1"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_path: str = Field(min_length=1, max_length=240)
    calibration_runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV4":
        for value in (
            self.calibration_runtime_configuration_path,
            self.selection_receipt_path,
            self.source_gold_receipt_path,
            self.candidate_execution_path,
        ):
            _relative_path(value)
        return self


class RasterTextCalibrationReceiptV5(_RuntimeWireModel):
    """Ten-family projected calibration evidence for the V15 staged lane."""

    schema_version: Literal["5.0"] = "5.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_projected_calibration"] = (
        "synthetic_raster_semantic_text_fresh_projected_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-calibration"] = "fresh-calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-staged-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    projection_kind: Literal["calibration-10-family-v1"]
    family_count: Literal[10]
    asset_count: Literal[240]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_path: str = Field(min_length=1, max_length=240)
    calibration_runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    projection_receipt_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=10)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV5":
        for value in (
            self.calibration_runtime_configuration_path,
            self.selection_receipt_path,
            self.projection_receipt_path,
            self.source_gold_receipt_path,
            self.candidate_execution_path,
        ):
            _relative_path(value)
        return self


class RasterTextCalibrationReceiptV6(_RuntimeWireModel):
    """Leakage-safe ten-family calibration evidence for V18."""

    schema_version: Literal["6.0"] = "6.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_projected_calibration"] = (
        "synthetic_raster_semantic_text_fresh_projected_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-calibration"] = "fresh-calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-staged-train-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    projection_kind: Literal["calibration-10-family-v1"]
    family_count: Literal[10]
    asset_count: Literal[240]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_path: str = Field(min_length=1, max_length=240)
    calibration_runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    projection_receipt_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=10)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV6":
        for value in (
            self.calibration_runtime_configuration_path,
            self.selection_receipt_path,
            self.projection_receipt_path,
            self.source_gold_receipt_path,
            self.candidate_execution_path,
        ):
            _relative_path(value)
        return self


class RasterTextCalibrationReceiptV7(_RuntimeWireModel):
    """V19 calibration evidence bound to the leakage-safe fine-tune release."""

    schema_version: Literal["7.0"] = "7.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_projected_calibration"] = (
        "synthetic_raster_semantic_text_fresh_projected_calibration"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-calibration"] = "fresh-calibration"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-public-finetuned-staged-train-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    projection_kind: Literal["calibration-10-family-v1"]
    family_count: Literal[10]
    asset_count: Literal[240]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_raster_text_profile_revision: str = Field(
        pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$"
    )
    calibration_runtime_configuration_path: str = Field(min_length=1, max_length=240)
    calibration_runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    projection_receipt_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_training_intent_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_parent_selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    minimum_page_confidence: float = Field(ge=0, le=1)
    latency_p95_seconds: float = Field(gt=0, le=10)
    hard_timeout_seconds: Literal[30] = 30

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextCalibrationReceiptV7":
        for value in (
            self.calibration_runtime_configuration_path,
            self.selection_receipt_path,
            self.projection_receipt_path,
            self.source_gold_receipt_path,
            self.candidate_execution_path,
        ):
            _relative_path(value)
        return self


class RasterTextWarmServiceSmokeReceiptV1(_RuntimeWireModel):
    """24-asset persistent-service evidence required before activation.

    The candidate evaluator has its own bounded persistent protocol for
    throughput.  This receipt proves that the *customer* service protocol
    returns exactly the same results after its independently verified warm
    startup.  It intentionally contains only hashes, never OCR source text.
    """

    schema_version: Literal["1.0"] = "1.0"
    # These discriminators intentionally have no defaults.  A legacy cold
    # smoke receipt with overlapping fields must not become a valid warm
    # receipt merely because Pydantic fills a new default during parsing.
    artifact_kind: Literal["synthetic_raster_text_warm_service_smoke"]
    dataset_id: Literal["synthetic-raster-text-v1"]
    scope: Literal["smoke"]
    source_truth_kind: Literal["pdf_content_stream_text_v1"]
    execution_mode: Literal["persistent_service_warm"]
    service_protocol: Literal["customs-raster-text-sidecar-v1-service"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    equality_mode: Literal["exact_canonical_result_v1"]
    equality_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    asset_count: Literal[24]
    startup_latency_seconds: float = Field(gt=0, le=_MAX_TEXT_SERVICE_STARTUP_SECONDS)
    startup_timeout_seconds: int = Field(ge=1, le=_MAX_TEXT_SERVICE_STARTUP_SECONDS)
    warm_latency_p95_seconds: float = Field(gt=0, le=30)
    request_timeout_seconds: int = Field(ge=1, le=30)

    @model_validator(mode="after")
    def _relative_paths(self) -> "RasterTextWarmServiceSmokeReceiptV1":
        _relative_path(self.source_gold_receipt_path)
        _relative_path(self.candidate_execution_path)
        return self


class RasterTextSealedEvaluationV1(_RuntimeWireModel):
    """Compact passing source-only text evidence required for activation."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["sealed"] = "sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    latency_p95_seconds: float = Field(gt=0, le=30)
    service_warm_latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30
    minimum_page_confidence: float = Field(ge=0, le=1)
    scores: list[RasterTextSealedProfileScoreV1] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _all_profiles_are_passed(self) -> "RasterTextSealedEvaluationV1":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        observed = {(score.document_type, score.profile_id) for score in self.scores}
        if observed != expected:
            raise ValueError("raster text sealed profiles are incomplete")
        for value in (
            self.source_gold_receipt_path,
            self.calibration_receipt_path,
            self.candidate_execution_path,
            self.warm_service_smoke_receipt_path,
        ):
            _relative_path(value)
        return self


class RasterTextSealedProfileScoreV2(_RuntimeWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedEvaluationV2(_RuntimeWireModel):
    """Passing sealed evidence explicitly measured with semantic metric V2."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_sealed_evaluation"] = (
        "synthetic_raster_semantic_text_sealed_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["sealed"] = "sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v2"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
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


class RasterTextSealedProfileScoreV3(_RuntimeWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedEvaluationV3(_RuntimeWireModel):
    """Passing sealed evidence explicitly measured with semantic metric V3."""

    schema_version: Literal["3.0"] = "3.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_sealed_evaluation"] = (
        "synthetic_raster_semantic_text_sealed_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["sealed"] = "sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
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


class RasterTextSealedProfileScoreV4(_RuntimeWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedEvaluationV4(_RuntimeWireModel):
    """Passing fresh-sealed evidence bound to semantic metric V3."""

    schema_version: Literal["4.0"] = "4.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_sealed_evaluation"] = (
        "synthetic_raster_semantic_text_fresh_sealed_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-sealed"] = "fresh-sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-holdout-v1"]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    latency_p95_seconds: float = Field(gt=0, le=30)
    service_warm_latency_p95_seconds: float = Field(gt=0, le=30)
    hard_timeout_seconds: Literal[30] = 30
    minimum_page_confidence: float = Field(ge=0, le=1)
    scores: list[RasterTextSealedProfileScoreV4] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _all_profiles_are_passed(self) -> "RasterTextSealedEvaluationV4":
        expected = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        if {(score.document_type, score.profile_id) for score in self.scores} != expected:
            raise ValueError("fresh raster semantic text sealed profiles are incomplete")
        for value in (
            self.selection_receipt_path,
            self.source_gold_receipt_path,
            self.calibration_receipt_path,
            self.candidate_execution_path,
            self.warm_service_smoke_receipt_path,
        ):
            _relative_path(value)
        return self


class RasterTextSealedProfileScoreV5(_RuntimeWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedEvaluationV5(_RuntimeWireModel):
    """Passing staged fresh-sealed evidence for one immutable V15 profile."""

    schema_version: Literal["5.0"] = "5.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_staged_evaluation"] = (
        "synthetic_raster_semantic_text_fresh_staged_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-sealed"] = "fresh-sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-staged-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"]
    stage1_decision: Literal["early_pass", "continue"]
    family_count: Literal[50, 100]
    asset_count: Literal[1_200, 2_400]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    projection_receipt_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    stage1_decision_receipt_path: str = Field(min_length=1, max_length=240)
    stage1_decision_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    latency_p95_seconds: float = Field(gt=0, le=10)
    service_warm_latency_p95_seconds: float = Field(gt=0, le=10)
    hard_timeout_seconds: Literal[30] = 30
    minimum_page_confidence: float = Field(ge=0, le=1)
    scores: list[RasterTextSealedProfileScoreV5] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _staged_gate_and_paths(self) -> "RasterTextSealedEvaluationV5":
        expected_scores = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        expected_stage = {
            "sealed-stage1-50-family-v1": ("early_pass", 50, 1_200),
            "sealed-stage2-100-family-v1": ("continue", 100, 2_400),
        }[self.projection_kind]
        if {
            (score.document_type, score.profile_id) for score in self.scores
        } != expected_scores or (
            self.stage1_decision,
            self.family_count,
            self.asset_count,
        ) != expected_stage:
            raise ValueError("staged raster semantic text sealed evidence is incomplete")
        for value in (
            self.selection_receipt_path,
            self.projection_receipt_path,
            self.stage1_decision_receipt_path,
            self.source_gold_receipt_path,
            self.calibration_receipt_path,
            self.candidate_execution_path,
            self.warm_service_smoke_receipt_path,
        ):
            _relative_path(value)
        return self


class RasterTextSealedProfileScoreV6(_RuntimeWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedEvaluationV6(_RuntimeWireModel):
    """Passing staged V18 evidence using the leakage-safe V2 selection."""

    schema_version: Literal["6.0"] = "6.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_staged_evaluation"] = (
        "synthetic_raster_semantic_text_fresh_staged_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-sealed"] = "fresh-sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-fresh-staged-train-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"]
    stage1_decision: Literal["early_pass", "continue"]
    family_count: Literal[50, 100]
    asset_count: Literal[1_200, 2_400]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    projection_receipt_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    stage1_decision_receipt_path: str = Field(min_length=1, max_length=240)
    stage1_decision_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    latency_p95_seconds: float = Field(gt=0, le=10)
    service_warm_latency_p95_seconds: float = Field(gt=0, le=10)
    hard_timeout_seconds: Literal[30] = 30
    minimum_page_confidence: float = Field(ge=0, le=1)
    scores: list[RasterTextSealedProfileScoreV6] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _staged_gate_and_paths(self) -> "RasterTextSealedEvaluationV6":
        expected_scores = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        expected_stage = {
            "sealed-stage1-50-family-v1": ("early_pass", 50, 1_200),
            "sealed-stage2-100-family-v1": ("continue", 100, 2_400),
        }[self.projection_kind]
        if {
            (score.document_type, score.profile_id) for score in self.scores
        } != expected_scores or (
            self.stage1_decision,
            self.family_count,
            self.asset_count,
        ) != expected_stage:
            raise ValueError("staged V18 raster semantic text evidence is incomplete")
        for value in (
            self.selection_receipt_path,
            self.projection_receipt_path,
            self.stage1_decision_receipt_path,
            self.source_gold_receipt_path,
            self.calibration_receipt_path,
            self.candidate_execution_path,
            self.warm_service_smoke_receipt_path,
        ):
            _relative_path(value)
        return self


class RasterTextSealedProfileScoreV7(_RuntimeWireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.98, le=1)


class RasterTextSealedEvaluationV7(_RuntimeWireModel):
    """Passing staged V19 evidence bound to one leakage-safe fine-tune."""

    schema_version: Literal["7.0"] = "7.0"
    artifact_kind: Literal["synthetic_raster_semantic_text_fresh_staged_evaluation"] = (
        "synthetic_raster_semantic_text_fresh_staged_evaluation"
    )
    dataset_id: Literal["synthetic-raster-text-v1"] = "synthetic-raster-text-v1"
    split: Literal["fresh-sealed"] = "fresh-sealed"
    source_truth_kind: Literal["pdf_content_stream_text_v1"] = "pdf_content_stream_text_v1"
    metric_revision: Literal["source-attested-semantic-text-v3"]
    evidence_revision: Literal["source-attested-public-finetuned-staged-train-holdout-v2"]
    latency_gate_revision: Literal["warm-p95-10s-v1"]
    projection_kind: Literal["sealed-stage1-50-family-v1", "sealed-stage2-100-family-v1"]
    stage1_decision: Literal["early_pass", "continue"]
    family_count: Literal[50, 100]
    asset_count: Literal[1_200, 2_400]
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    runtime_configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str = Field(min_length=1, max_length=256)
    selection_receipt_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    projection_receipt_path: str = Field(min_length=1, max_length=240)
    projection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    stage1_decision_receipt_path: str = Field(min_length=1, max_length=240)
    stage1_decision_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_gold_receipt_path: str = Field(min_length=1, max_length=240)
    source_gold_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_receipt_path: str = Field(min_length=1, max_length=240)
    calibration_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_execution_path: str = Field(min_length=1, max_length=240)
    candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    warm_service_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    warm_service_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_training_intent_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_recognizer_parent_selection_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    latency_p95_seconds: float = Field(gt=0, le=10)
    service_warm_latency_p95_seconds: float = Field(gt=0, le=10)
    hard_timeout_seconds: Literal[30] = 30
    minimum_page_confidence: float = Field(ge=0, le=1)
    scores: list[RasterTextSealedProfileScoreV7] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def _staged_gate_and_paths(self) -> "RasterTextSealedEvaluationV7":
        expected_scores = {
            (document_type, profile_id)
            for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
            for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
        }
        expected_stage = {
            "sealed-stage1-50-family-v1": ("early_pass", 50, 1_200),
            "sealed-stage2-100-family-v1": ("continue", 100, 2_400),
        }[self.projection_kind]
        if {
            (score.document_type, score.profile_id) for score in self.scores
        } != expected_scores or (
            self.stage1_decision,
            self.family_count,
            self.asset_count,
        ) != expected_stage:
            raise ValueError("staged V19 raster semantic text evidence is incomplete")
        for value in (
            self.selection_receipt_path,
            self.projection_receipt_path,
            self.stage1_decision_receipt_path,
            self.source_gold_receipt_path,
            self.calibration_receipt_path,
            self.candidate_execution_path,
            self.warm_service_smoke_receipt_path,
        ):
            _relative_path(value)
        return self


class RasterTextSealedProfileScoreV8(_RuntimeWireModel):
    """Preview-draft score: keep CER at 2% and lower only critical exact to 97%."""

    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_error_rate: float = Field(ge=0, le=0.02)
    critical_value_exact_match: float = Field(ge=0.97, le=1)


class RasterTextSealedEvaluationV8(RasterTextSealedEvaluationV7):
    """V19 sealed evidence revalidated against the current runtime implementation."""

    schema_version: Literal["8.0"] = "8.0"  # type: ignore[assignment]
    quality_gate_revision: Literal["preview-draft-critical97-cer2-v1"] = (
        "preview-draft-critical97-cer2-v1"
    )
    evidence_revalidation_revision: Literal["historical-runner-current-runtime-smoke-v1"] = (
        "historical-runner-current-runtime-smoke-v1"
    )
    historical_runner_implementation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    current_runner_implementation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revalidation_smoke_receipt_path: str = Field(min_length=1, max_length=240)
    revalidation_smoke_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revalidation_candidate_execution_path: str = Field(min_length=1, max_length=240)
    revalidation_candidate_execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scores: list[RasterTextSealedProfileScoreV8] = Field(min_length=12, max_length=12)  # type: ignore[assignment]

    @model_validator(mode="after")
    def _revalidation_paths_are_relative(self) -> "RasterTextSealedEvaluationV8":
        _relative_path(self.revalidation_smoke_receipt_path)
        _relative_path(self.revalidation_candidate_execution_path)
        return self


RasterTextCalibrationReceipt = (
    RasterTextCalibrationReceiptV1
    | RasterTextCalibrationReceiptV2
    | RasterTextCalibrationReceiptV3
    | RasterTextCalibrationReceiptV4
    | RasterTextCalibrationReceiptV5
    | RasterTextCalibrationReceiptV6
    | RasterTextCalibrationReceiptV7
)
RasterTextSealedEvaluation = (
    RasterTextSealedEvaluationV1
    | RasterTextSealedEvaluationV2
    | RasterTextSealedEvaluationV3
    | RasterTextSealedEvaluationV4
    | RasterTextSealedEvaluationV5
    | RasterTextSealedEvaluationV6
    | RasterTextSealedEvaluationV7
    | RasterTextSealedEvaluationV8
)


class RasterTextSidecarResultV1(_RuntimeWireModel):
    protocol: Literal["customs-raster-text-sidecar-v1"] = "customs-raster-text-sidecar-v1"
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raster_text_profile_revision: str = Field(pattern=r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
    provider_revision: str = Field(min_length=1, max_length=256)
    pages: list[RasterTextPageV1] = Field(max_length=20)
    warnings: list[RasterTextWarningV1] = Field(default_factory=list, max_length=32)
    needs_review_reasons: list[RasterTextWarningV1] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _page_order(self) -> "RasterTextSidecarResultV1":
        if [page.page_number for page in self.pages] != list(range(1, len(self.pages) + 1)):
            raise ValueError("raster text sidecar page order is invalid")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_evaluation_json_bytes(value: object) -> bytes:
    """Match immutable evaluation receipts written by the activation runner.

    Runtime configuration and pointer records intentionally use the legacy
    newline-terminated ASCII encoding above.  Gold/candidate/evaluation
    evidence is created by the activation module with this no-newline UTF-8
    form, so never apply the configuration serializer to those receipts.
    """

    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def raster_text_exact_result_set_sha256(
    values: Sequence[tuple[str, str, RasterTextExtractionResultV1]],
) -> str:
    """Hash an exact, content-free comparison set of bound OCR results.

    The individual canonical results contain editable text and therefore stay
    in the local candidate bundle only.  Evidence outside that bundle stores
    only their deterministic hashes, while preserving every result field in
    the equality relation (including request binding, text, confidence,
    status, warnings, and provenance).
    """

    entries: list[dict[str, str]] = []
    for asset_id, profile_id, result in values:
        if (
            not isinstance(asset_id, str)
            or not asset_id
            or profile_id not in {"png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"}
            or result.document_asset_id != asset_id
        ):
            raise RasterTextRuntimeError("raster_text_result_equality_invalid")
        entries.append(
            {
                "asset_id": asset_id,
                "profile_id": profile_id,
                "canonical_result_sha256": hashlib.sha256(
                    raster_text_result_json(result)
                ).hexdigest(),
            }
        )
    entries.sort(key=lambda value: (value["asset_id"], value["profile_id"]))
    keys = [(value["asset_id"], value["profile_id"]) for value in entries]
    if not entries or len(keys) != len(set(keys)):
        raise RasterTextRuntimeError("raster_text_result_equality_invalid")
    return hashlib.sha256(_canonical_json_bytes(entries)).hexdigest()


def _read_json_object(raw: bytes, *, code: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RasterTextRuntimeError(code) from None
    if not isinstance(value, dict):
        raise RasterTextRuntimeError(code)
    return value


def validate_raster_text_approval_pointer_path(path: Path) -> tuple[Path, Path]:
    """Accept exactly the Git-excluded text-only active pointer location."""

    try:
        if (
            not path.is_absolute()
            or path.name != "raster-text-active.json"
            or path.parent.name != "approvals"
            or tuple(part.casefold() for part in path.parent.parent.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise RasterTextRuntimeError("raster_text_auto_pointer_invalid")
        root = _document_preview_root(path.parent.parent)
        expected = root / "approvals" / "raster-text-active.json"
        if path.is_symlink() or path.parent.is_symlink() or path.resolve(strict=False) != expected:
            raise RasterTextRuntimeError("raster_text_auto_pointer_invalid")
        return root, expected
    except RasterTextRuntimeError:
        raise
    except OSError:
        raise RasterTextRuntimeError("raster_text_auto_pointer_invalid") from None


def parse_raster_text_activation_pointer(raw: bytes) -> RasterTextActivationPointer:
    value = _read_json_object(raw, code="raster_text_activation_invalid")
    if raw != _canonical_json_bytes(value):
        raise RasterTextRuntimeError("raster_text_activation_invalid")
    try:
        discriminator = value.get("schema_version")
        if discriminator == "1.0":
            pointer: RasterTextActivationPointer = RasterTextActivationPointerV1.model_validate(
                value
            )
        elif discriminator == "2.0":
            pointer = RasterTextActivationPointerV2.model_validate(value)
        elif discriminator == "3.0":
            pointer = RasterTextActivationPointerV3.model_validate(value)
        elif discriminator == "4.0":
            pointer = RasterTextActivationPointerV4.model_validate(value)
        elif discriminator == "5.0":
            pointer = RasterTextActivationPointerV5.model_validate(value)
        elif discriminator == "6.0":
            pointer = RasterTextActivationPointerV6.model_validate(value)
        elif discriminator == "7.0":
            pointer = RasterTextActivationPointerV7.model_validate(value)
        elif discriminator == "8.0":
            pointer = RasterTextActivationPointerV8.model_validate(value)
        elif discriminator == "9.0":
            pointer = RasterTextActivationPointerV9.model_validate(value)
        elif discriminator == "10.0":
            pointer = RasterTextActivationPointerV10.model_validate(value)
        elif discriminator == "11.0":
            pointer = RasterTextActivationPointerV11.model_validate(value)
        else:
            raise ValueError("raster text activation schema is invalid")
    except ValidationError:
        raise RasterTextRuntimeError("raster_text_activation_invalid") from None
    except ValueError:
        raise RasterTextRuntimeError("raster_text_activation_invalid") from None
    # Comparing only the parsed input mapping lets Pydantic silently restore
    # a discriminator default omitted by an old pointer.  The active pointer
    # writer always emits the complete canonical model, so anything shorter
    # is legacy/ambiguous evidence and must fail closed.
    if raw != _canonical_json_bytes(pointer.model_dump(mode="json", warnings=False)):
        raise RasterTextRuntimeError("raster_text_activation_invalid")
    return pointer


def _parse_raster_text_calibration_receipt(raw: bytes) -> RasterTextCalibrationReceipt:
    value = _read_json_object(raw, code="raster_text_evaluation_invalid")
    try:
        discriminator = value.get("schema_version")
        if discriminator == "1.0":
            receipt: RasterTextCalibrationReceipt = RasterTextCalibrationReceiptV1.model_validate(
                value
            )
        elif discriminator == "2.0":
            receipt = RasterTextCalibrationReceiptV2.model_validate(value)
        elif discriminator == "3.0":
            receipt = RasterTextCalibrationReceiptV3.model_validate(value)
        elif discriminator == "4.0":
            receipt = RasterTextCalibrationReceiptV4.model_validate(value)
        elif discriminator == "5.0":
            receipt = RasterTextCalibrationReceiptV5.model_validate(value)
        elif discriminator == "6.0":
            receipt = RasterTextCalibrationReceiptV6.model_validate(value)
        elif discriminator == "7.0":
            receipt = RasterTextCalibrationReceiptV7.model_validate(value)
        else:
            raise ValueError("raster text calibration schema is invalid")
    except (ValidationError, ValueError):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None
    if raw != _canonical_evaluation_json_bytes(receipt.model_dump(mode="json", warnings=False)):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    return receipt


def _parse_raster_text_sealed_evaluation(raw: bytes) -> RasterTextSealedEvaluation:
    value = _read_json_object(raw, code="raster_text_evaluation_invalid")
    try:
        discriminator = value.get("schema_version")
        if discriminator == "1.0":
            evaluation: RasterTextSealedEvaluation = RasterTextSealedEvaluationV1.model_validate(
                value
            )
        elif discriminator == "2.0":
            evaluation = RasterTextSealedEvaluationV2.model_validate(value)
        elif discriminator == "3.0":
            evaluation = RasterTextSealedEvaluationV3.model_validate(value)
        elif discriminator == "4.0":
            evaluation = RasterTextSealedEvaluationV4.model_validate(value)
        elif discriminator == "5.0":
            evaluation = RasterTextSealedEvaluationV5.model_validate(value)
        elif discriminator == "6.0":
            evaluation = RasterTextSealedEvaluationV6.model_validate(value)
        elif discriminator == "7.0":
            evaluation = RasterTextSealedEvaluationV7.model_validate(value)
        elif discriminator == "8.0":
            evaluation = RasterTextSealedEvaluationV8.model_validate(value)
        else:
            raise ValueError("raster text sealed schema is invalid")
    except (ValidationError, ValueError):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None
    if raw != _canonical_evaluation_json_bytes(evaluation.model_dump(mode="json", warnings=False)):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    return evaluation


def _bound_raster_text_metric_revision(
    *,
    pointer: RasterTextActivationPointer,
    evaluation: RasterTextSealedEvaluation,
    calibration: RasterTextCalibrationReceipt,
) -> Literal["source-attested-semantic-text-v2", "source-attested-semantic-text-v3"] | None:
    """Reject any pointer/calibration/sealed evidence generation mixture."""

    if (
        isinstance(pointer, RasterTextActivationPointerV1)
        and isinstance(evaluation, RasterTextSealedEvaluationV1)
        and isinstance(calibration, RasterTextCalibrationReceiptV1)
    ):
        return None
    if (
        isinstance(pointer, RasterTextActivationPointerV2)
        and isinstance(evaluation, RasterTextSealedEvaluationV2)
        and isinstance(calibration, RasterTextCalibrationReceiptV2)
        and pointer.metric_revision
        == evaluation.metric_revision
        == calibration.metric_revision
        == "source-attested-semantic-text-v2"
    ):
        return pointer.metric_revision
    if (
        isinstance(pointer, RasterTextActivationPointerV3)
        and isinstance(evaluation, RasterTextSealedEvaluationV3)
        and isinstance(calibration, RasterTextCalibrationReceiptV3)
        and pointer.metric_revision
        == evaluation.metric_revision
        == calibration.metric_revision
        == "source-attested-semantic-text-v3"
    ):
        return pointer.metric_revision
    if (
        isinstance(pointer, RasterTextActivationPointerV4)
        and isinstance(evaluation, RasterTextSealedEvaluationV4)
        and isinstance(calibration, RasterTextCalibrationReceiptV4)
        and pointer.metric_revision
        == evaluation.metric_revision
        == calibration.metric_revision
        == "source-attested-semantic-text-v3"
        and pointer.evidence_revision
        == evaluation.evidence_revision
        == calibration.evidence_revision
        == "source-attested-fresh-holdout-v1"
    ):
        return pointer.metric_revision
    if (
        isinstance(pointer, RasterTextActivationPointerV5)
        and isinstance(evaluation, RasterTextSealedEvaluationV5)
        and isinstance(calibration, RasterTextCalibrationReceiptV5)
        and pointer.metric_revision
        == evaluation.metric_revision
        == calibration.metric_revision
        == "source-attested-semantic-text-v3"
        and pointer.evidence_revision
        == evaluation.evidence_revision
        == calibration.evidence_revision
        == "source-attested-fresh-staged-holdout-v2"
        and pointer.latency_gate_revision
        == evaluation.latency_gate_revision
        == calibration.latency_gate_revision
        == "warm-p95-10s-v1"
    ):
        return pointer.metric_revision
    if (
        isinstance(pointer, RasterTextActivationPointerV6)
        and isinstance(evaluation, RasterTextSealedEvaluationV6)
        and isinstance(calibration, RasterTextCalibrationReceiptV6)
        and pointer.metric_revision
        == evaluation.metric_revision
        == calibration.metric_revision
        == "source-attested-semantic-text-v3"
        and pointer.evidence_revision
        == evaluation.evidence_revision
        == calibration.evidence_revision
        == "source-attested-fresh-staged-train-holdout-v2"
        and pointer.latency_gate_revision
        == evaluation.latency_gate_revision
        == calibration.latency_gate_revision
        == "warm-p95-10s-v1"
    ):
        return pointer.metric_revision
    if (
        type(pointer) in (RasterTextActivationPointerV10, RasterTextActivationPointerV11)
        and type(evaluation) is RasterTextSealedEvaluationV8
        and type(calibration) is RasterTextCalibrationReceiptV7
    ):
        pointer_v10 = cast(RasterTextActivationPointerV10, pointer)
        evaluation_v8 = cast(RasterTextSealedEvaluationV8, evaluation)
        calibration_v7 = cast(RasterTextCalibrationReceiptV7, calibration)
        if (
            pointer_v10.metric_revision
            == evaluation_v8.metric_revision
            == calibration_v7.metric_revision
            == "source-attested-semantic-text-v3"
            and pointer_v10.evidence_revision
            == evaluation_v8.evidence_revision
            == calibration_v7.evidence_revision
            == "source-attested-public-finetuned-staged-train-holdout-v2"
            and pointer_v10.latency_gate_revision
            == evaluation_v8.latency_gate_revision
            == calibration_v7.latency_gate_revision
            == "warm-p95-10s-v1"
            and pointer_v10.quality_gate_revision
            == evaluation_v8.quality_gate_revision
            == "preview-draft-critical97-cer2-v1"
            and pointer_v10.evidence_revalidation_revision
            == evaluation_v8.evidence_revalidation_revision
            == "historical-runner-current-runtime-smoke-v1"
            and pointer_v10.historical_runner_implementation_sha256
            == evaluation_v8.historical_runner_implementation_sha256
            and pointer_v10.current_runner_implementation_sha256
            == evaluation_v8.current_runner_implementation_sha256
            and pointer_v10.revalidation_smoke_receipt_path
            == evaluation_v8.revalidation_smoke_receipt_path
            and pointer_v10.revalidation_smoke_receipt_sha256
            == evaluation_v8.revalidation_smoke_receipt_sha256
            and pointer_v10.revalidation_candidate_execution_path
            == evaluation_v8.revalidation_candidate_execution_path
            and pointer_v10.revalidation_candidate_execution_sha256
            == evaluation_v8.revalidation_candidate_execution_sha256
        ):
            return pointer_v10.metric_revision
    if (
        type(pointer)
        in (
            RasterTextActivationPointerV7,
            RasterTextActivationPointerV8,
            RasterTextActivationPointerV9,
        )
        and type(evaluation) is RasterTextSealedEvaluationV7
        and type(calibration) is RasterTextCalibrationReceiptV7
    ):
        pointer_v7 = cast(RasterTextActivationPointerV7, pointer)
        evaluation_v7 = cast(RasterTextSealedEvaluationV7, evaluation)
        calibration_v7 = cast(RasterTextCalibrationReceiptV7, calibration)
        if (
            pointer_v7.metric_revision
            == evaluation_v7.metric_revision
            == calibration_v7.metric_revision
            == "source-attested-semantic-text-v3"
            and pointer_v7.evidence_revision
            == evaluation_v7.evidence_revision
            == calibration_v7.evidence_revision
            == "source-attested-public-finetuned-staged-train-holdout-v2"
            and pointer_v7.latency_gate_revision
            == evaluation_v7.latency_gate_revision
            == calibration_v7.latency_gate_revision
            == "warm-p95-10s-v1"
        ):
            return pointer_v7.metric_revision
    raise RasterTextRuntimeError("raster_text_evaluation_invalid")


def _recalculate_source_attested_raster_text_quality(
    *,
    sealed_cases: Sequence[Any],
    calibration_cases: Sequence[Any],
    metric_revision: Literal["source-attested-semantic-text-v2", "source-attested-semantic-text-v3"]
    | None,
    quality_gate_revision: Literal["preview-draft-critical97-cer2-v1"] | None = None,
) -> tuple[list[Any], float, bool]:
    """Recompute the exact evaluator and confidence floor bound by evidence."""

    from customs_ai_agent.evaluation.synthetic_raster_text_activation import _calibrated_floor
    from customs_ai_agent.evaluation.synthetic_raster_text import (
        evaluate_synthetic_raster_semantic_text_gates_v2,
        evaluate_synthetic_raster_semantic_text_gates_v3,
        evaluate_synthetic_raster_text_gates,
        synthetic_raster_semantic_text_gate_passes_v2,
        synthetic_raster_semantic_text_gate_passes_v3,
        synthetic_raster_text_gate_passes,
    )

    if quality_gate_revision is not None and metric_revision != (
        "source-attested-semantic-text-v3"
    ):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    if metric_revision == "source-attested-semantic-text-v3":
        scores = evaluate_synthetic_raster_semantic_text_gates_v3(sealed_cases)
        floor = _calibrated_floor(
            calibration_cases,
            metric_revision=metric_revision,
        )
        gate_passed = (
            all(
                score.character_error_rate <= 0.02 and score.critical_value_exact_match >= 0.97
                for score in scores
            )
            if quality_gate_revision == "preview-draft-critical97-cer2-v1"
            else synthetic_raster_semantic_text_gate_passes_v3(scores)
        )
        return scores, floor, gate_passed
    if metric_revision == "source-attested-semantic-text-v2":
        scores = evaluate_synthetic_raster_semantic_text_gates_v2(sealed_cases)
        floor = _calibrated_floor(
            calibration_cases,
            metric_revision=metric_revision,
        )
        return scores, floor, synthetic_raster_semantic_text_gate_passes_v2(scores)
    scores = evaluate_synthetic_raster_text_gates(sealed_cases)
    floor = _calibrated_floor(calibration_cases)
    return scores, floor, synthetic_raster_text_gate_passes(scores)


def _sha256_regular_file(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise RasterTextRuntimeError("raster_text_runtime_material_invalid")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except RasterTextRuntimeError:
        raise
    except OSError:
        raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None


def _non_symlink_sidecar_path(root: Path, value: str) -> Path:
    """Resolve one declared bundle path without following an inserted link.

    ``_sidecar_path`` deliberately resolves its candidate before it returns a
    Windows-safe operational path.  That is useful for containment, but it
    would otherwise hide a link inserted at one of the declared path
    components from the lightweight cached-binding check.  Walking the short
    declared path first is bounded by the profile's path-length limit and
    avoids a directory-tree traversal.
    """

    try:
        relative = PurePosixPath(value)
        _relative_path(value)
        if root.is_symlink():
            raise RasterTextRuntimeError("raster_text_runtime_material_invalid")
        declared = root
        for part in relative.parts:
            declared = declared / part
            if declared.is_symlink():
                raise RasterTextRuntimeError("raster_text_runtime_material_invalid")
        return _sidecar_path(root, value)
    except RasterTextRuntimeError:
        raise
    except (OSError, ValueError):
        raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None


def _v2_bundle_sidecar_path(
    *,
    root: Path,
    profile: (
        RasterTextOfflineProfileV2
        | RasterTextOfflineProfileV3
        | RasterTextOfflineProfileV4
        | RasterTextOfflineProfileV5
        | RasterTextOfflineProfileV6
        | RasterTextOfflineProfileV7
        | RasterTextOfflineProfileV8
        | RasterTextOfflineProfileV9
        | RasterTextOfflineProfileV10
        | RasterTextOfflineProfileV11
        | RasterTextOfflineProfileV12
        | RasterTextOfflineProfileV13
        | RasterTextOfflineProfileV14
        | RasterTextOfflineProfileV15
        | RasterTextOfflineProfileV16
        | RasterTextOfflineProfileV17
        | RasterTextOfflineProfileV18
        | RasterTextOfflineProfileV19
        | RasterTextOfflineProfileV20
    ),
    value: str,
) -> Path:
    """Return one direct-profile material path through a verified short root.

    The direct-Paddle microbundle lives beneath a content-addressed release
    directory.  On Windows, resolving the *leaf* of its first-party sidecar
    module can expand the Korean workspace spelling past the legacy Win32
    path limit before the generic resolver gets a chance to obtain its 8.3
    alias.  V3 additionally has a distinct short model-bundle root.  Resolve
    and validate the owning declared release root once, then append only a
    validated lexical suffix to that operational spelling.

    This does not weaken the original path boundary: the release root still
    goes through ``_non_symlink_sidecar_path`` (canonical containment and
    component-link checks), the profile requires every material path to be a
    child of that root, and every suffix component is checked again through
    the operational path before it can be used.
    """

    try:
        declared_relative = PurePosixPath(_relative_path(value))
        roots = [PurePosixPath(_relative_path(profile.bundle_root_relative_path))]
        if isinstance(
            profile,
            (
                RasterTextOfflineProfileV3,
                RasterTextOfflineProfileV4,
                RasterTextOfflineProfileV5,
                RasterTextOfflineProfileV6,
                RasterTextOfflineProfileV7,
                RasterTextOfflineProfileV8,
                RasterTextOfflineProfileV9,
                RasterTextOfflineProfileV10,
                RasterTextOfflineProfileV11,
                RasterTextOfflineProfileV13,
                RasterTextOfflineProfileV14,
                RasterTextOfflineProfileV15,
                RasterTextOfflineProfileV16,
                RasterTextOfflineProfileV17,
                RasterTextOfflineProfileV18,
                RasterTextOfflineProfileV19,
                RasterTextOfflineProfileV20,
            ),
        ):
            roots.append(
                PurePosixPath(_relative_path(profile.mobile_model_bundle_root_relative_path))
            )
        elif isinstance(profile, RasterTextOfflineProfileV12):
            roots.append(
                PurePosixPath(_relative_path(profile.combined_model_bundle_root_relative_path))
            )
        bundle_relative: PurePosixPath | None = None
        suffix: PurePosixPath | None = None
        for candidate_root in roots:
            try:
                candidate_suffix = declared_relative.relative_to(candidate_root)
            except ValueError:
                continue
            if candidate_suffix != PurePosixPath(".") and candidate_suffix.parts:
                bundle_relative = candidate_root
                suffix = candidate_suffix
                break
        if bundle_relative is None or suffix is None:
            raise ValueError

        # This is intentionally the only generic resolver used for V2
        # bundle material.  The release root remains short enough for its
        # canonical containment check, and it returns the Windows-safe
        # spelling from which deep children can be opened safely.
        operational_bundle_root = _non_symlink_sidecar_path(root, bundle_relative.as_posix())
        candidate = operational_bundle_root
        for part in suffix.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError
        return candidate
    except RasterTextRuntimeError:
        raise
    except (OSError, ValueError):
        raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None


def _profile_sidecar_path(*, root: Path, profile: RasterTextOfflineProfile, value: str) -> Path:
    """Resolve one profile-owned material without changing the V1 boundary."""

    if isinstance(
        profile,
        (
            RasterTextOfflineProfileV2,
            RasterTextOfflineProfileV3,
            RasterTextOfflineProfileV4,
            RasterTextOfflineProfileV5,
            RasterTextOfflineProfileV6,
            RasterTextOfflineProfileV7,
            RasterTextOfflineProfileV8,
            RasterTextOfflineProfileV9,
            RasterTextOfflineProfileV10,
            RasterTextOfflineProfileV11,
            RasterTextOfflineProfileV12,
            RasterTextOfflineProfileV13,
            RasterTextOfflineProfileV14,
            RasterTextOfflineProfileV15,
            RasterTextOfflineProfileV16,
            RasterTextOfflineProfileV17,
            RasterTextOfflineProfileV18,
            RasterTextOfflineProfileV19,
            RasterTextOfflineProfileV20,
        ),
    ):
        return _v2_bundle_sidecar_path(root=root, profile=profile, value=value)
    return _sidecar_path(root, value)


_V3_MOBILE_MODEL_INVENTORY_RELATIVE_PATH = "inventories/direct-paddle-mobile-models.json"
_V3_MOBILE_SOURCE_TO_TARGET_ARTIFACTS = (
    ("models/paddleocr3/det/config.json", "text-det/config.json"),
    ("models/paddleocr3/det/inference.json", "text-det/inference.json"),
    ("models/paddleocr3/det/inference.pdiparams", "text-det/inference.pdiparams"),
    ("models/paddleocr3/det/inference.yml", "text-det/inference.yml"),
    ("models/paddleocr3/rec/config.json", "text-rec/config.json"),
    ("models/paddleocr3/rec/inference.json", "text-rec/inference.json"),
    ("models/paddleocr3/rec/inference.pdiparams", "text-rec/inference.pdiparams"),
    ("models/paddleocr3/rec/inference.yml", "text-rec/inference.yml"),
)
_V3_MOBILE_MODEL_ARTIFACTS = frozenset(
    target for _source, target in _V3_MOBILE_SOURCE_TO_TARGET_ARTIFACTS
)


def _v3_sha256_and_size(path: Path) -> tuple[str, int]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        return _sha256_regular_file(path), path.stat().st_size
    except (OSError, RasterTextRuntimeError):
        raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid") from None


def _v3_source_snapshot_sha256(records: Sequence[tuple[str, str, int]]) -> str:
    """Match the mobile builder's source snapshot byte-for-byte."""

    return hashlib.sha256(
        _canonical_evaluation_json_bytes(
            {
                "artifacts": [
                    {"relative_path": path, "sha256": digest, "byte_size": size}
                    for path, digest, size in records
                ]
            }
        )
    ).hexdigest()


def _v3_model_tree_sha256(records: Sequence[tuple[str, str, int]]) -> str:
    """Match the direct bundle's sorted model-root tree fingerprint."""

    return hashlib.sha256(
        _canonical_evaluation_json_bytes(
            {
                "files": [
                    {"path": path, "sha256": digest, "byte_size": size}
                    for path, digest, size in sorted(records)
                ]
            }
        )
    ).hexdigest()


def _v3_source_inventory_records(
    *,
    root: Path,
    profile: (
        RasterTextOfflineProfileV3
        | RasterTextOfflineProfileV4
        | RasterTextOfflineProfileV5
        | RasterTextOfflineProfileV6
        | RasterTextOfflineProfileV7
        | RasterTextOfflineProfileV8
        | RasterTextOfflineProfileV9
        | RasterTextOfflineProfileV10
        | RasterTextOfflineProfileV11
        | RasterTextOfflineProfileV13
        | RasterTextOfflineProfileV14
        | RasterTextOfflineProfileV15
        | RasterTextOfflineProfileV16
        | RasterTextOfflineProfileV17
    ),
) -> tuple[tuple[str, str, int], ...]:
    """Load the fixed source inventory and normalize it to copied paths."""

    source_path = _non_symlink_sidecar_path(root, profile.source_model_inventory_relative_path)
    source_raw = _read_local_file(source_path, limit=_MAX_CONFIGURATION_BYTES)
    if hashlib.sha256(source_raw).hexdigest() != profile.source_model_inventory_sha256:
        raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
    source = json.loads(source_raw.decode("utf-8"))
    required = {"schema_version", "engine_kind", "runtime_api", "packages", "artifacts"}
    if (
        not isinstance(source, dict)
        # The fixed PaddleOCR3 source inventory was published before the
        # no-newline bundle receipts.  Its immutable SHA includes this one
        # final LF, so retain that exact source format here.
        or source_raw != _canonical_evaluation_json_bytes(source) + b"\n"
        or set(source) != required
        or source.get("schema_version") != "1.0"
        or source.get("engine_kind") != "paddleocr"
        or source.get("runtime_api") != "paddleocr3-paddlex-3.2"
        or not isinstance(source.get("packages"), list)
        or not isinstance(source.get("artifacts"), list)
        or len(source["artifacts"]) != len(_V3_MOBILE_SOURCE_TO_TARGET_ARTIFACTS)
    ):
        raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
    records: list[tuple[str, str, int]] = []
    for (source_path_value, target_path), item in zip(
        _V3_MOBILE_SOURCE_TO_TARGET_ARTIFACTS, source["artifacts"], strict=True
    ):
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "artifact_id",
                "byte_size",
                "license_id",
                "relative_path",
                "sha256",
                "source_uri",
            }
            or item.get("relative_path") != source_path_value
            or not isinstance(item.get("artifact_id"), str)
            or not item["artifact_id"]
            or not isinstance(item.get("license_id"), str)
            or not item["license_id"]
            or not isinstance(item.get("source_uri"), str)
            or not item["source_uri"]
            or not isinstance(item.get("sha256"), str)
            or _SHA256.fullmatch(item["sha256"]) is None
            or isinstance(item.get("byte_size"), bool)
            or not isinstance(item.get("byte_size"), int)
            or item["byte_size"] < 0
        ):
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        records.append((target_path, item["sha256"], item["byte_size"]))
    return tuple(records)


def _verify_v3_mobile_model_bundle(
    *,
    root: Path,
    profile: (
        RasterTextOfflineProfileV3
        | RasterTextOfflineProfileV4
        | RasterTextOfflineProfileV5
        | RasterTextOfflineProfileV6
        | RasterTextOfflineProfileV7
        | RasterTextOfflineProfileV8
        | RasterTextOfflineProfileV9
        | RasterTextOfflineProfileV10
        | RasterTextOfflineProfileV11
        | RasterTextOfflineProfileV13
        | RasterTextOfflineProfileV14
        | RasterTextOfflineProfileV15
        | RasterTextOfflineProfileV16
        | RasterTextOfflineProfileV17
    ),
) -> None:
    """Verify the shared V3--V11/V13/V14 mobile bundle before exposing a model root.

    The generic model-inventory verifier below checks every model byte.  This
    additional boundary binds that inventory to the separately attested
    mobile-bundle manifest and the fixed source inventory identifier, so a
    profile cannot point at a valid-looking but unrelated local cache.
    """

    try:
        source_records = _v3_source_inventory_records(root=root, profile=profile)
        source_snapshot_sha256 = _v3_source_snapshot_sha256(source_records)
        expected_model_root = f"{profile.mobile_model_bundle_root_relative_path}/models"
        expected_model_inventory = (
            f"{profile.mobile_model_bundle_root_relative_path}/"
            f"{_V3_MOBILE_MODEL_INVENTORY_RELATIVE_PATH}"
        )
        if (
            profile.local_model_root != expected_model_root
            or profile.model_inventory_path != expected_model_inventory
        ):
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        manifest_path = _v2_bundle_sidecar_path(
            root=root,
            profile=profile,
            value=(
                PurePosixPath(profile.mobile_model_bundle_root_relative_path) / "manifest.json"
            ).as_posix(),
        )
        raw = _read_local_file(manifest_path, limit=_MAX_CONFIGURATION_BYTES)
        if hashlib.sha256(raw).hexdigest() != profile.mobile_model_bundle_manifest_sha256:
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        value = json.loads(raw.decode("utf-8"))
        if raw != _canonical_evaluation_json_bytes(value):
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        required = {
            "schema_version",
            "bundle_id",
            "source_model_inventory_relative_path",
            "source_model_inventory_sha256",
            "source_model_snapshot_sha256",
            "text_detection_model_name",
            "text_recognition_model_name",
            "local_model_root_relative_path",
            "model_inventory_relative_path",
            "model_root_tree_sha256",
            "files",
        }
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value.get("schema_version") != "1.0"
            or value.get("bundle_id") != "raster_text_direct_paddle_mobile_model_bundle_v1"
            or value.get("source_model_inventory_relative_path")
            != profile.source_model_inventory_relative_path
            or value.get("source_model_inventory_sha256") != profile.source_model_inventory_sha256
            or value.get("source_model_snapshot_sha256") != source_snapshot_sha256
            or value.get("text_detection_model_name") != "PP-OCRv5_mobile_det"
            or value.get("text_recognition_model_name") != "en_PP-OCRv5_mobile_rec"
            or value.get("local_model_root_relative_path") != "models"
            or value.get("model_inventory_relative_path")
            != _V3_MOBILE_MODEL_INVENTORY_RELATIVE_PATH
            or not isinstance(value.get("source_model_snapshot_sha256"), str)
            or _SHA256.fullmatch(value["source_model_snapshot_sha256"]) is None
            or not isinstance(value.get("model_root_tree_sha256"), str)
            or _SHA256.fullmatch(value["model_root_tree_sha256"]) is None
            or not isinstance(value.get("files"), list)
        ):
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        records: dict[str, tuple[str, int]] = {}
        record_paths: list[str] = []
        for item in value["files"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "sha256", "byte_size"}
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("sha256"), str)
                or _SHA256.fullmatch(item["sha256"]) is None
                or isinstance(item.get("byte_size"), bool)
                or not isinstance(item.get("byte_size"), int)
                or item["byte_size"] < 0
                or item["path"] in records
            ):
                raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
            _relative_path(item["path"])
            record_paths.append(item["path"])
            records[item["path"]] = (item["sha256"], item["byte_size"])
        expected_paths = {
            _V3_MOBILE_MODEL_INVENTORY_RELATIVE_PATH,
            *(f"models/{relative}" for relative in _V3_MOBILE_MODEL_ARTIFACTS),
        }
        if record_paths != sorted(record_paths) or set(records) != expected_paths:
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")

        # Bind every copied byte (including the mobile config) to the source
        # inventory, then independently hash the complete model tree.  This
        # closes both substitution and undeclared-extra-file paths before a
        # child gets a model directory.
        manifest_model_records = tuple(
            (target, *records[f"models/{target}"])
            for _source, target in _V3_MOBILE_SOURCE_TO_TARGET_ARTIFACTS
        )
        if manifest_model_records != source_records:
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        for relative, expected in records.items():
            artifact_path = _v2_bundle_sidecar_path(
                root=root,
                profile=profile,
                value=(
                    PurePosixPath(profile.mobile_model_bundle_root_relative_path) / relative
                ).as_posix(),
            )
            observed_sha256, observed_size = _v3_sha256_and_size(artifact_path)
            if (observed_sha256, observed_size) != expected:
                raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        model_root = _v2_bundle_sidecar_path(
            root=root, profile=profile, value=profile.local_model_root
        )
        if model_root.is_symlink() or not model_root.is_dir():
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        actual_model_records: list[tuple[str, str, int]] = []
        for artifact_path in model_root.rglob("*"):
            if artifact_path.is_symlink() or not (
                artifact_path.is_dir() or artifact_path.is_file()
            ):
                raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
            if not artifact_path.is_file():
                continue
            relative = artifact_path.relative_to(model_root).as_posix()
            expected_record = records.get(f"models/{relative}")
            if expected_record is None:
                raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
            observed_sha256, observed_size = _v3_sha256_and_size(artifact_path)
            if (observed_sha256, observed_size) != expected_record:
                raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
            actual_model_records.append((relative, observed_sha256, observed_size))
        if {
            path for path, _digest, _size in actual_model_records
        } != _V3_MOBILE_MODEL_ARTIFACTS or _v3_model_tree_sha256(actual_model_records) != value[
            "model_root_tree_sha256"
        ]:
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")

        model_inventory_path = _v2_bundle_sidecar_path(
            root=root, profile=profile, value=profile.model_inventory_path
        )
        model_inventory_raw = _read_local_file(model_inventory_path, limit=_MAX_CONFIGURATION_BYTES)
        inventory_sha256, inventory_size = records[_V3_MOBILE_MODEL_INVENTORY_RELATIVE_PATH]
        if (
            hashlib.sha256(model_inventory_raw).hexdigest() != inventory_sha256
            or len(model_inventory_raw) != inventory_size
            or hashlib.sha256(model_inventory_raw).hexdigest() != profile.model_inventory_sha256
        ):
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        inventory = json.loads(model_inventory_raw.decode("utf-8"))
        inventory_required = {
            "schema_version",
            "inventory_kind",
            "source_model_inventory_relative_path",
            "source_model_inventory_sha256",
            "source_model_snapshot_sha256",
            "text_detection_model_name",
            "text_recognition_model_name",
            "artifacts",
        }
        if (
            model_inventory_raw != _canonical_evaluation_json_bytes(inventory)
            or not isinstance(inventory, dict)
            or set(inventory) != inventory_required
            or inventory.get("schema_version") != "1.0"
            or inventory.get("inventory_kind") != "raster_text_direct_paddle_mobile_models"
            or inventory.get("source_model_inventory_relative_path")
            != profile.source_model_inventory_relative_path
            or inventory.get("source_model_inventory_sha256")
            != profile.source_model_inventory_sha256
            or inventory.get("source_model_snapshot_sha256") != source_snapshot_sha256
            or inventory.get("text_detection_model_name") != "PP-OCRv5_mobile_det"
            or inventory.get("text_recognition_model_name") != "en_PP-OCRv5_mobile_rec"
            or not isinstance(inventory.get("artifacts"), list)
            or len(inventory["artifacts"]) != len(source_records)
        ):
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        copied_records: list[tuple[str, str, int]] = []
        for source_record, item in zip(source_records, inventory["artifacts"], strict=True):
            if (
                not isinstance(item, dict)
                or set(item) != {"relative_path", "sha256", "byte_size"}
                or not isinstance(item.get("relative_path"), str)
                or not isinstance(item.get("sha256"), str)
                or _SHA256.fullmatch(item["sha256"]) is None
                or isinstance(item.get("byte_size"), bool)
                or not isinstance(item.get("byte_size"), int)
                or item["byte_size"] < 0
            ):
                raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
            copied_records.append((item["relative_path"], item["sha256"], item["byte_size"]))
            if copied_records[-1] != source_record:
                raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
        if tuple(copied_records) != manifest_model_records:
            raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid")
    except RasterTextRuntimeError as error:
        if str(error) == "raster_text_mobile_model_bundle_invalid":
            raise
        raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid") from None
    except (OSError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RasterTextRuntimeError("raster_text_mobile_model_bundle_invalid") from None


def _verify_v12_trained_model_bundle(*, root: Path, profile: RasterTextOfflineProfileV12) -> None:
    """Reverify and bind V12 to one exact combined detector/recognizer release."""

    try:
        bundle = load_verified_raster_text_trained_paddle_bundle_v1(
            document_preview_root=root,
            manifest_sha256=profile.combined_model_bundle_manifest_sha256,
        )
        manifest = bundle.manifest
        if (
            bundle.bundle_root_relative_path != profile.combined_model_bundle_root_relative_path
            or manifest.source_detector_bundle_manifest_sha256
            != profile.source_detector_bundle_manifest_sha256
            or manifest.source_detector_model_inventory_sha256
            != profile.source_detector_model_inventory_sha256
            or manifest.source_detector_model_snapshot_sha256
            != profile.source_detector_model_snapshot_sha256
            or manifest.source_recognizer_release_sha256 != profile.source_recognizer_release_sha256
            or manifest.source_recognizer_receipt_sha256 != profile.source_recognizer_receipt_sha256
            or manifest.source_recognizer_training_intent_sha256
            != profile.source_recognizer_training_intent_sha256
            or manifest.source_recognizer_character_dictionary_sha256
            != profile.source_recognizer_character_dictionary_sha256
            or manifest.source_recognizer_package_inventory_sha256
            != profile.source_recognizer_package_inventory_sha256
            or profile.local_model_root
            != bundle.bundle_relative_path(manifest.local_model_root_relative_path)
            or profile.model_inventory_path
            != bundle.bundle_relative_path(manifest.model_inventory_relative_path)
            or profile.model_inventory_sha256
            != bundle.file_sha256(manifest.model_inventory_relative_path)
            or manifest.text_detection_model_name != profile.text_detection_model_name
            or manifest.text_recognition_model_name != profile.text_recognition_model_name
        ):
            raise RasterTextRuntimeError("raster_text_trained_model_bundle_invalid")
    except RasterTextRuntimeError:
        raise
    except (RasterTextTrainedPaddleBundleError, OSError, TypeError, ValueError):
        raise RasterTextRuntimeError("raster_text_trained_model_bundle_invalid") from None


def _verify_v18_server_model_bundle(*, root: Path, profile: RasterTextOfflineProfileV18) -> None:
    """Reverify every byte and both upstream sources for the V18 model tree."""

    try:
        bundle = load_verified_raster_text_direct_server_paddle_bundle_v1(
            document_preview_root=root,
            manifest_sha256=profile.mobile_model_bundle_manifest_sha256,
        )
        manifest = bundle.manifest
        if (
            bundle.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
            or profile.local_model_root
            != bundle.bundle_relative_path(manifest.local_model_root_relative_path)
            or profile.model_inventory_path
            != bundle.bundle_relative_path(manifest.model_inventory_relative_path)
            or profile.model_inventory_sha256
            != bundle.file_sha256(manifest.model_inventory_relative_path)
            or profile.source_model_inventory_relative_path
            != bundle.bundle_relative_path(manifest.provenance_receipt_relative_path)
            or profile.source_model_inventory_sha256
            != bundle.file_sha256(manifest.provenance_receipt_relative_path)
            or manifest.text_detection_model_name != profile.text_detection_model_name
            or manifest.text_recognition_model_name != profile.text_recognition_model_name
        ):
            raise RasterTextRuntimeError("raster_text_server_model_bundle_invalid")
    except RasterTextRuntimeError:
        raise
    except (RasterTextServerPaddleBundleError, OSError, TypeError, ValueError):
        raise RasterTextRuntimeError("raster_text_server_model_bundle_invalid") from None


def _verify_v19_public_finetuned_model_bundle(
    *, root: Path, profile: RasterTextOfflineProfileV19
) -> None:
    """Bind V19 to one exact warm-start release and its 438-class receipt."""

    try:
        bundle = load_verified_raster_text_public_finetuned_paddle_bundle_v1(
            document_preview_root=root,
            manifest_sha256=profile.mobile_model_bundle_manifest_sha256,
        )
        manifest = bundle.manifest
        if (
            bundle.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
            or profile.local_model_root
            != bundle.bundle_relative_path(manifest.local_model_root_relative_path)
            or profile.model_inventory_path
            != bundle.bundle_relative_path(manifest.model_inventory_relative_path)
            or profile.model_inventory_sha256
            != bundle.file_sha256(manifest.model_inventory_relative_path)
            or profile.source_model_inventory_relative_path
            != bundle.bundle_relative_path(manifest.recognizer_receipt_provenance_relative_path)
            or profile.source_model_inventory_sha256
            != bundle.file_sha256(manifest.recognizer_receipt_provenance_relative_path)
            or profile.source_recognizer_release_sha256 != manifest.source_recognizer_release_sha256
            or profile.source_recognizer_receipt_sha256 != manifest.source_recognizer_receipt_sha256
            or profile.source_recognizer_training_intent_sha256
            != manifest.source_recognizer_training_intent_sha256
            or profile.source_recognizer_selection_receipt_relative_path
            != manifest.source_recognizer_selection_receipt_relative_path
            or profile.source_recognizer_selection_receipt_sha256
            != manifest.source_recognizer_selection_receipt_sha256
            or profile.source_recognizer_parent_selection_receipt_sha256
            != manifest.source_recognizer_parent_selection_receipt_sha256
            or profile.source_recognizer_train_corpus_provenance
            != manifest.source_recognizer_train_corpus_provenance
            or profile.source_recognizer_development_corpus_provenance
            != manifest.source_recognizer_development_corpus_provenance
            or profile.source_recognizer_base_dictionary_sha256
            != manifest.source_recognizer_base_dictionary_sha256
            or profile.source_recognizer_output_class_count
            != manifest.source_recognizer_output_class_count
            or manifest.source_recognizer_output_class_count != 438
            or manifest.text_detection_model_name != profile.text_detection_model_name
            or manifest.text_recognition_model_name != profile.text_recognition_model_name
        ):
            raise RasterTextRuntimeError("raster_text_public_finetuned_model_bundle_invalid")
    except RasterTextRuntimeError:
        raise
    except (
        RasterTextPublicFineTunedPaddleBundleError,
        OSError,
        TypeError,
        ValueError,
    ):
        raise RasterTextRuntimeError("raster_text_public_finetuned_model_bundle_invalid") from None


def _verify_v20_public_voyage_replay_model_bundle(
    *, root: Path, profile: RasterTextOfflineProfileV20
) -> None:
    """Bind V20 to the exact replay release, parent V2 and corpus receipts."""

    try:
        from customs_ai_agent.evaluation.raster_text_public_voyage_replay_paddle_bundle import (
            RasterTextPublicVoyageReplayPaddleBundleError,
            load_verified_raster_text_public_voyage_replay_paddle_bundle_v1,
        )
    except (ImportError, AttributeError):
        raise RasterTextRuntimeError(
            "raster_text_public_voyage_replay_model_bundle_invalid"
        ) from None
    try:
        bundle = load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
            document_preview_root=root,
            manifest_sha256=profile.mobile_model_bundle_manifest_sha256,
        )
        manifest = bundle.manifest
        if (
            bundle.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
            or profile.local_model_root
            != bundle.bundle_relative_path(manifest.local_model_root_relative_path)
            or profile.model_inventory_path
            != bundle.bundle_relative_path(manifest.model_inventory_relative_path)
            or profile.model_inventory_sha256
            != bundle.file_sha256(manifest.model_inventory_relative_path)
            or profile.source_model_inventory_relative_path
            != bundle.bundle_relative_path(manifest.recognizer_receipt_provenance_relative_path)
            or profile.source_model_inventory_sha256
            != bundle.file_sha256(manifest.recognizer_receipt_provenance_relative_path)
            or profile.source_recognizer_release_sha256 != manifest.source_recognizer_release_sha256
            or profile.source_recognizer_receipt_sha256 != manifest.source_recognizer_receipt_sha256
            or profile.source_recognizer_training_intent_sha256
            != manifest.source_recognizer_training_intent_sha256
            or profile.source_recognizer_parent_release_sha256
            != manifest.source_recognizer_parent_release_sha256
            or profile.source_recognizer_parent_receipt_sha256
            != manifest.source_recognizer_parent_receipt_sha256
            or profile.source_recognizer_train_manifest_sha256
            != manifest.source_recognizer_train_manifest_sha256
            or profile.source_recognizer_development_manifest_sha256
            != manifest.source_recognizer_development_manifest_sha256
            or profile.source_recognizer_train_corpus_manifest
            != manifest.source_recognizer_train_corpus_manifest
            or profile.source_recognizer_development_corpus_manifest
            != manifest.source_recognizer_development_corpus_manifest
            or profile.source_recognizer_selection_receipt_relative_path
            != manifest.source_recognizer_selection_receipt_relative_path
            or profile.source_recognizer_selection_receipt_sha256
            != manifest.source_recognizer_selection_receipt_sha256
            or profile.source_recognizer_parent_selection_receipt_sha256
            != manifest.source_recognizer_parent_selection_receipt_sha256
            or profile.source_recognizer_voyage_recipe_set_sha256
            != manifest.source_recognizer_voyage_recipe_set_sha256
            or profile.source_recognizer_base_dictionary_sha256
            != manifest.source_recognizer_base_dictionary_sha256
            or profile.source_recognizer_output_class_count
            != manifest.source_recognizer_output_class_count
            or manifest.source_recognizer_output_class_count != 438
            or manifest.text_detection_model_name != profile.text_detection_model_name
            or manifest.text_recognition_model_name != profile.text_recognition_model_name
        ):
            raise RasterTextRuntimeError("raster_text_public_voyage_replay_model_bundle_invalid")
    except RasterTextRuntimeError:
        raise
    except (
        RasterTextPublicVoyageReplayPaddleBundleError,
        OSError,
        TypeError,
        ValueError,
    ):
        raise RasterTextRuntimeError(
            "raster_text_public_voyage_replay_model_bundle_invalid"
        ) from None


def _verify_profile_model_bundle(*, root: Path, profile: RasterTextOfflineProfile) -> None:
    """Dispatch one profile to exactly one owning model-bundle verifier.

    V18 and V19 inherit V9 only to share its immutable detector semantics.
    Their combined model releases are independent boundaries, so this ordered
    dispatch must not also enter the historical V3 mobile verifier.
    """

    if isinstance(profile, RasterTextOfflineProfileV20):
        _verify_v20_public_voyage_replay_model_bundle(root=root, profile=profile)
    elif isinstance(profile, RasterTextOfflineProfileV12):
        _verify_v12_trained_model_bundle(root=root, profile=profile)
    elif isinstance(profile, RasterTextOfflineProfileV18):
        _verify_v18_server_model_bundle(root=root, profile=profile)
    elif isinstance(profile, RasterTextOfflineProfileV19):
        _verify_v19_public_finetuned_model_bundle(root=root, profile=profile)
    elif isinstance(
        profile,
        (
            RasterTextOfflineProfileV3,
            RasterTextOfflineProfileV4,
            RasterTextOfflineProfileV5,
            RasterTextOfflineProfileV6,
            RasterTextOfflineProfileV7,
            RasterTextOfflineProfileV8,
            RasterTextOfflineProfileV9,
            RasterTextOfflineProfileV10,
            RasterTextOfflineProfileV11,
            RasterTextOfflineProfileV13,
            RasterTextOfflineProfileV14,
            RasterTextOfflineProfileV15,
            RasterTextOfflineProfileV16,
            RasterTextOfflineProfileV17,
        ),
    ):
        _verify_v3_mobile_model_bundle(root=root, profile=profile)


def _validate_v2_controlled_child_file(*, root_path: Path, file_path: Path) -> None:
    """Validate a V2 child without re-expanding a deep Windows leaf path.

    Both paths originate from :func:`_v2_bundle_sidecar_path`, which already
    checks their complete paths from the verified bundle root for links.  The
    direct lexical relationship below keeps the same controlled-child
    invariant while avoiding ``Path.resolve()`` on a leaf that can acquire a
    ``\\\\?\\`` spelling and no longer compare equal to its root spelling.
    """

    try:
        if (
            root_path.is_symlink()
            or file_path.is_symlink()
            or not root_path.is_dir()
            or not file_path.is_file()
        ):
            raise ValueError
        relative = file_path.relative_to(root_path)
        if relative == Path(".") or not relative.parts:
            raise ValueError
        candidate = root_path
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError
    except (OSError, ValueError):
        raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None


def _validate_profile_controlled_child_file(
    *, profile: RasterTextOfflineProfile, root_path: Path, file_path: Path
) -> None:
    """Keep V1's imported validator exact while using the V2-safe variant."""

    if isinstance(
        profile,
        (
            RasterTextOfflineProfileV2,
            RasterTextOfflineProfileV3,
            RasterTextOfflineProfileV4,
            RasterTextOfflineProfileV5,
            RasterTextOfflineProfileV6,
            RasterTextOfflineProfileV7,
            RasterTextOfflineProfileV8,
            RasterTextOfflineProfileV9,
            RasterTextOfflineProfileV10,
            RasterTextOfflineProfileV11,
            RasterTextOfflineProfileV12,
            RasterTextOfflineProfileV13,
            RasterTextOfflineProfileV14,
            RasterTextOfflineProfileV15,
            RasterTextOfflineProfileV16,
            RasterTextOfflineProfileV17,
            RasterTextOfflineProfileV18,
            RasterTextOfflineProfileV19,
            RasterTextOfflineProfileV20,
        ),
    ):
        _validate_v2_controlled_child_file(root_path=root_path, file_path=file_path)
        return
    _validate_controlled_child_file(root_path=root_path, file_path=file_path)


def _verify_direct_profile_runtime_platform(
    *, root: Path, profile: RasterTextOfflineProfile
) -> None:
    """Bind a schema-2 direct profile to its native bundle target.

    The profile fingerprint already includes the complete manifest SHA and
    content-addressed release path. Reading that one small manifest here makes
    the OS/architecture/CPython ABI binding effective before any executable
    is opened. Historical schema-1 Windows bundles remain readable so the
    existing V19/schema-11 pointer contract is not rewritten.
    """

    if not isinstance(profile, RasterTextOfflineProfileV2):
        return
    try:
        manifest_path = _v2_bundle_sidecar_path(
            root=root,
            profile=profile,
            value=f"{profile.bundle_root_relative_path}/manifest.json",
        )
        raw = _read_local_file(manifest_path, limit=_MAX_CONFIGURATION_BYTES)
        if hashlib.sha256(raw).hexdigest() != profile.bundle_manifest_sha256:
            raise ValueError
        value = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(value, dict)
            or raw != _canonical_evaluation_json_bytes(value)
            or value.get("bundle_id") != "raster-text-direct-paddle-bundle-v1"
        ):
            raise ValueError
        schema_version = value.get("schema_version")
        if schema_version == "1.0":
            return
        if (
            schema_version != "2.0"
            or (
                value.get("runtime_operating_system"),
                value.get("runtime_architecture"),
                value.get("runtime_python_abi"),
            )
            != current_raster_runtime_platform_binding()
        ):
            raise ValueError
        if (
            value.get("sidecar_interpreter_relative_path")
            != PurePosixPath(profile.sidecar_interpreter_path)
            .relative_to(PurePosixPath(profile.bundle_root_relative_path))
            .as_posix()
        ):
            raise ValueError
    except Exception:
        raise RasterTextRuntimeError("raster_text_runtime_platform_mismatch") from None


def _load_profile_materials(
    *,
    root: Path,
    configuration_path: Path,
    configuration_raw: bytes,
    configuration_sha256: str,
    verify_complete_materials: bool = True,
) -> tuple[RasterTextOfflineProfile, Path, Path, Path, Path, Path, Path, Path]:
    if hashlib.sha256(configuration_raw).hexdigest() != configuration_sha256:
        raise RasterTextRuntimeError("raster_text_configuration_invalid")
    try:
        profile = parse_raster_text_offline_profile(configuration_raw)
    except (ValidationError, ValueError):
        raise RasterTextRuntimeError("raster_text_configuration_invalid") from None
    # Runtime profiles are immutable artifacts written with the same compact
    # canonical form as activation evidence.  Reject a legacy payload whose
    # omitted Literal/default values Pydantic would otherwise reconstruct.
    if configuration_raw != _canonical_evaluation_json_bytes(
        profile.model_dump(mode="json", warnings=False)
    ):
        raise RasterTextRuntimeError("raster_text_configuration_invalid")
    try:
        _verify_direct_profile_runtime_platform(root=root, profile=profile)
        # Activation owns the expensive byte-for-byte verification of the
        # complete model release and import roots.  A V11 service restart can
        # trust that content-addressed activation evidence and perform only
        # the directly executed-file checks below; the isolated child still
        # verifies its complete package/model inventories before publishing
        # ``ready`` and around every OCR request.  Older pointer generations
        # keep the exhaustive startup behavior.
        if verify_complete_materials:
            _verify_profile_model_bundle(root=root, profile=profile)
        parent_inventory = _profile_sidecar_path(
            root=root,
            profile=profile,
            value=profile.parent_renderer_package_inventory_path,
        )
        _verify_package_inventory(
            raw=_read_local_file(parent_inventory, limit=_MAX_CONFIGURATION_BYTES),
            expected_sha256=profile.parent_renderer_package_inventory_sha256,
            required_packages=profile.parent_renderer_required_packages,
        )
        interpreter_root = _profile_sidecar_path(
            root=root, profile=profile, value=profile.sidecar_interpreter_root
        )
        interpreter = _profile_sidecar_path(
            root=root, profile=profile, value=profile.sidecar_interpreter_path
        )
        module_root = _profile_sidecar_path(
            root=root, profile=profile, value=profile.sidecar_module_root
        )
        module = _profile_sidecar_path(
            root=root, profile=profile, value=profile.sidecar_module_path
        )
        _validate_profile_controlled_child_file(
            profile=profile, root_path=interpreter_root, file_path=interpreter
        )
        _validate_profile_controlled_child_file(
            profile=profile, root_path=module_root, file_path=module
        )
        if (
            _sha256_regular_file(interpreter) != profile.sidecar_interpreter_sha256
            or _sha256_regular_file(module) != profile.sidecar_module_sha256
            or (
                verify_complete_materials
                and (
                    _controlled_directory_tree_sha256(interpreter_root)
                    != profile.sidecar_interpreter_root_tree_sha256
                    or _controlled_module_root_tree_sha256(module_root)
                    != profile.sidecar_module_root_tree_sha256
                )
            )
        ):
            raise RasterTextRuntimeError("raster_text_sidecar_drift")
        sidecar_inventory = _profile_sidecar_path(
            root=root,
            profile=profile,
            value=profile.sidecar_package_inventory_path,
        )
        _verify_package_inventory(
            raw=_read_local_file(sidecar_inventory, limit=_MAX_CONFIGURATION_BYTES),
            expected_sha256=profile.sidecar_package_inventory_sha256,
            required_packages=profile.sidecar_required_packages,
            verify_installed=False,
        )
        model_root = _profile_sidecar_path(
            root=root, profile=profile, value=profile.local_model_root
        )
        model_inventory = _profile_sidecar_path(
            root=root, profile=profile, value=profile.model_inventory_path
        )
        model_inventory_raw = _read_local_file(model_inventory, limit=_MAX_CONFIGURATION_BYTES)
        if not verify_complete_materials:
            if hashlib.sha256(model_inventory_raw).hexdigest() != profile.model_inventory_sha256:
                raise RasterTextRuntimeError("raster_text_model_inventory_invalid")
        elif isinstance(
            profile,
            (
                RasterTextOfflineProfileV12,
                RasterTextOfflineProfileV18,
                RasterTextOfflineProfileV19,
                RasterTextOfflineProfileV20,
            ),
        ):
            # The V12 public combined-bundle loader above already verifies
            # every model byte, the exact eight-file tree, this inventory,
            # both upstream releases, and their provenance.  Re-entering the
            # generic PP-Structure leaf resolver here would expand the
            # already validated short Windows bundle root before opening its
            # deepest files and would duplicate that stronger verification.
            if hashlib.sha256(model_inventory_raw).hexdigest() != profile.model_inventory_sha256:
                code = (
                    "raster_text_trained_model_bundle_invalid"
                    if isinstance(profile, RasterTextOfflineProfileV12)
                    else "raster_text_server_model_bundle_invalid"
                    if isinstance(profile, RasterTextOfflineProfileV18)
                    else "raster_text_public_voyage_replay_model_bundle_invalid"
                    if isinstance(profile, RasterTextOfflineProfileV20)
                    else "raster_text_public_finetuned_model_bundle_invalid"
                )
                raise RasterTextRuntimeError(code)
        else:
            _verify_model_inventory(
                raw=model_inventory_raw,
                expected_sha256=profile.model_inventory_sha256,
                model_root=model_root,
            )
        for relative in (
            profile.text_detection_model_directory,
            profile.text_recognition_model_directory,
        ):
            if isinstance(
                profile,
                (
                    RasterTextOfflineProfileV2,
                    RasterTextOfflineProfileV3,
                    RasterTextOfflineProfileV4,
                    RasterTextOfflineProfileV5,
                    RasterTextOfflineProfileV6,
                    RasterTextOfflineProfileV7,
                    RasterTextOfflineProfileV8,
                    RasterTextOfflineProfileV9,
                    RasterTextOfflineProfileV10,
                    RasterTextOfflineProfileV11,
                    RasterTextOfflineProfileV12,
                    RasterTextOfflineProfileV13,
                    RasterTextOfflineProfileV14,
                    RasterTextOfflineProfileV15,
                    RasterTextOfflineProfileV16,
                    RasterTextOfflineProfileV17,
                    RasterTextOfflineProfileV18,
                    RasterTextOfflineProfileV19,
                    RasterTextOfflineProfileV20,
                ),
            ):
                directory = _v2_bundle_sidecar_path(
                    root=root,
                    profile=profile,
                    value=(PurePosixPath(profile.local_model_root) / relative).as_posix(),
                )
            else:
                directory = _sidecar_path(model_root, relative)
            if directory.is_symlink() or not directory.is_dir():
                raise RasterTextRuntimeError("raster_text_model_inventory_invalid")
    except RasterTextRuntimeError:
        raise
    except Exception:
        raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None
    return (
        profile,
        interpreter,
        interpreter_root,
        module,
        module_root,
        sidecar_inventory,
        model_root,
        model_inventory,
    )


def _validate_cached_raster_text_binding_materials(
    *, root: Path, approval_path: Path, approval_raw: bytes, approval_sha256: str
) -> str:
    """Recheck cheap parent-visible material before cached-engine reuse.

    The sidecar rechecks its package and complete model inventories before it
    imports PaddleOCR for every execution.  The provider also rechecks the
    complete interpreter/module import roots immediately before that child
    launch.  Repeating those full-tree hashes here would make the cached
    runtime path pay for the same expensive work twice.  This validator is
    deliberately limited to pointer/configuration identity, declared path
    shape, and the two directly executed/imported files; it never builds a
    replacement engine or executes a native process.
    """

    pointer = parse_raster_text_activation_pointer(approval_raw)
    if hashlib.sha256(approval_raw).hexdigest() != approval_sha256:
        raise RasterTextRuntimeError("raster_text_activation_invalid")
    configuration_path = _non_symlink_sidecar_path(root, pointer.runtime_configuration_path)
    configuration_raw = _read_local_file(configuration_path, limit=_MAX_CONFIGURATION_BYTES)
    if hashlib.sha256(configuration_raw).hexdigest() != pointer.runtime_configuration_sha256:
        raise RasterTextRuntimeError("raster_text_configuration_invalid")
    try:
        profile = parse_raster_text_offline_profile(configuration_raw)
    except (ValidationError, ValueError):
        raise RasterTextRuntimeError("raster_text_configuration_invalid") from None
    if profile.raster_text_profile_revision != pointer.raster_text_profile_revision:
        raise RasterTextRuntimeError("raster_text_profile_mismatch")
    try:
        # V11 activation has already byte-verified the complete content-
        # addressed bundle, and its persistent child independently verifies
        # package/model material.  Repeating the ~900 MiB bundle walk on the
        # one-second pointer poll is neither a useful runtime guard nor a
        # lightweight operation.  Legacy generations retain their original
        # exhaustive cached-binding check.
        if type(pointer) is not RasterTextActivationPointerV11:
            _verify_profile_model_bundle(root=root, profile=profile)
        if isinstance(
            profile,
            (
                RasterTextOfflineProfileV2,
                RasterTextOfflineProfileV3,
                RasterTextOfflineProfileV4,
                RasterTextOfflineProfileV5,
                RasterTextOfflineProfileV6,
                RasterTextOfflineProfileV7,
                RasterTextOfflineProfileV8,
                RasterTextOfflineProfileV9,
                RasterTextOfflineProfileV10,
                RasterTextOfflineProfileV11,
                RasterTextOfflineProfileV12,
                RasterTextOfflineProfileV13,
                RasterTextOfflineProfileV14,
                RasterTextOfflineProfileV15,
                RasterTextOfflineProfileV16,
                RasterTextOfflineProfileV17,
                RasterTextOfflineProfileV18,
                RasterTextOfflineProfileV19,
                RasterTextOfflineProfileV20,
            ),
        ):
            interpreter_root = _v2_bundle_sidecar_path(
                root=root, profile=profile, value=profile.sidecar_interpreter_root
            )
            interpreter = _v2_bundle_sidecar_path(
                root=root, profile=profile, value=profile.sidecar_interpreter_path
            )
            module_root = _v2_bundle_sidecar_path(
                root=root, profile=profile, value=profile.sidecar_module_root
            )
            module = _v2_bundle_sidecar_path(
                root=root, profile=profile, value=profile.sidecar_module_path
            )
        else:
            interpreter_root = _non_symlink_sidecar_path(root, profile.sidecar_interpreter_root)
            interpreter = _non_symlink_sidecar_path(root, profile.sidecar_interpreter_path)
            module_root = _non_symlink_sidecar_path(root, profile.sidecar_module_root)
            module = _non_symlink_sidecar_path(root, profile.sidecar_module_path)
        _validate_profile_controlled_child_file(
            profile=profile, root_path=interpreter_root, file_path=interpreter
        )
        _validate_profile_controlled_child_file(
            profile=profile, root_path=module_root, file_path=module
        )
        if (
            _sha256_regular_file(interpreter) != profile.sidecar_interpreter_sha256
            or _sha256_regular_file(module) != profile.sidecar_module_sha256
        ):
            raise RasterTextRuntimeError("raster_text_sidecar_drift")
    except RasterTextRuntimeError:
        raise
    except Exception:
        raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None
    return profile.raster_text_profile_revision


def _sidecar_environment(
    *,
    interpreter: Path,
    module_root: Path,
    runtime_home: Path,
    configuration_path: Path,
    configuration_sha256: str,
    profile: RasterTextOfflineProfile,
    model_root: Path,
    model_inventory: Path,
    sidecar_inventory: Path,
) -> dict[str, str]:
    # Reuse the audited isolated-child environment, then replace its
    # PP-Structure-only variables with the direct-text protocol names.  This
    # keeps the exact bundled interpreter/module root as the only import
    # surface and never inherits a user PYTHONPATH or home directory.
    try:
        allowed = _isolated_v2_sidecar_environment(
            interpreter_path=interpreter,
            module_root=module_root,
            runtime_home=runtime_home,
            configuration_path=configuration_path,
            configuration_sha256=configuration_sha256,
            model_root=model_root,
            package_inventory_path=sidecar_inventory,
            package_inventory_sha256=profile.sidecar_package_inventory_sha256,
            model_inventory_path=model_inventory,
            model_inventory_sha256=profile.model_inventory_sha256,
            required_packages=profile.sidecar_required_packages,
            sidecar_module_sha256=profile.sidecar_module_sha256,
        )
    except Exception:
        raise RasterTextRuntimeError("raster_text_sidecar_environment_invalid") from None
    for key in tuple(allowed):
        if key.startswith("CUSTOMS_AGENT_PPSTRUCTURE_"):
            allowed.pop(key)
    common = {
        "CUSTOMS_AGENT_LOCAL_FILES_ONLY": "1",
        "HF_HUB_OFFLINE": "1",
        "PADDLEOCR_DISABLE_MODEL_SOURCE_CHECK": "True",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        # The primary child is started with ``-B``.  Paddle may still launch
        # helper interpreters while importing native packages, so bind their
        # bytecode policy and cache root to the disposable runtime home too.
        # An evaluated content-addressed bundle must never create undeclared
        # ``__pycache__`` files inside its own module tree.
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(runtime_home / "pycache"),
        "TRANSFORMERS_OFFLINE": "1",
    }
    if isinstance(
        profile,
        (
            RasterTextOfflineProfileV2,
            RasterTextOfflineProfileV3,
            RasterTextOfflineProfileV4,
            RasterTextOfflineProfileV5,
            RasterTextOfflineProfileV6,
            RasterTextOfflineProfileV7,
            RasterTextOfflineProfileV8,
            RasterTextOfflineProfileV9,
            RasterTextOfflineProfileV10,
            RasterTextOfflineProfileV11,
            RasterTextOfflineProfileV12,
            RasterTextOfflineProfileV13,
            RasterTextOfflineProfileV14,
            RasterTextOfflineProfileV15,
            RasterTextOfflineProfileV16,
            RasterTextOfflineProfileV17,
            RasterTextOfflineProfileV18,
            RasterTextOfflineProfileV19,
            RasterTextOfflineProfileV20,
        ),
    ):
        # The static direct-Paddle sidecar has a separate environment
        # contract.  Do not leave legacy high-level variables available: a
        # mixed V1/V2/V3 child configuration must fail rather than quietly
        # use a package or model root from another engine.
        direct_prefix = "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_"
        allowed.update(
            {
                **common,
                direct_prefix + "CONFIG_PATH": str(configuration_path),
                direct_prefix + "CONFIG_SHA256": configuration_sha256,
                direct_prefix + "MODEL_ROOT": str(model_root),
                direct_prefix + "MODEL_INVENTORY_PATH": str(model_inventory),
                direct_prefix + "MODEL_INVENTORY_SHA256": profile.model_inventory_sha256,
                direct_prefix + "PACKAGE_INVENTORY_PATH": str(sidecar_inventory),
                direct_prefix
                + "PACKAGE_INVENTORY_SHA256": profile.sidecar_package_inventory_sha256,
                direct_prefix + "REQUIRED_PACKAGES": json.dumps(
                    profile.sidecar_required_packages,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                direct_prefix + "MODULE_SHA256": profile.sidecar_module_sha256,
            }
        )
    else:
        allowed.update(
            {
                **common,
                "CUSTOMS_AGENT_RASTER_TEXT_CONFIG_PATH": str(configuration_path),
                "CUSTOMS_AGENT_RASTER_TEXT_CONFIG_SHA256": configuration_sha256,
                "CUSTOMS_AGENT_RASTER_TEXT_MODEL_ROOT": str(model_root),
                "CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_PATH": str(model_inventory),
                "CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_SHA256": profile.model_inventory_sha256,
                "CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_PATH": str(sidecar_inventory),
                "CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_SHA256": profile.sidecar_package_inventory_sha256,
                "CUSTOMS_AGENT_RASTER_TEXT_REQUIRED_PACKAGES": json.dumps(
                    profile.sidecar_required_packages,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                "CUSTOMS_AGENT_RASTER_TEXT_SIDECAR_MODULE_SHA256": profile.sidecar_module_sha256,
            }
        )
    return allowed


def _sidecar_payload(
    pages: Sequence[RasterizedPageV2],
    *,
    source_sha256: str,
    raster_text_profile_revision: str,
) -> bytes:
    if not pages or _SHA256.fullmatch(source_sha256) is None:
        raise RasterTextRuntimeError("raster_text_sidecar_input_invalid")
    header = {
        "protocol": _PROTOCOL,
        "source_sha256": source_sha256,
        "raster_text_profile_revision": raster_text_profile_revision,
        "pages": [
            {"page_number": page.page_number, "width": page.width, "height": page.height}
            for page in pages
        ],
    }
    raw_pixels = b"".join(page.pixels for page in pages)
    if any(len(page.pixels) != page.width * page.height * 3 for page in pages):
        raise RasterTextRuntimeError("raster_text_sidecar_input_invalid")
    payload = _canonical_json_bytes(header) + raw_pixels
    if not 1 <= len(payload) <= _MAX_INPUT_BYTES:
        raise RasterTextRuntimeError("raster_text_sidecar_input_invalid")
    return payload


@dataclass(frozen=True, slots=True)
class PaddleOcrTextCommandProvider(RasterTextProvider):
    """One-shot, offline-only direct detector/recognizer sidecar provider."""

    profile: RasterTextOfflineProfile
    interpreter: Path
    interpreter_root: Path
    module: Path
    module_root: Path
    configuration_path: Path
    configuration_sha256: str
    sidecar_inventory: Path
    model_root: Path
    model_inventory: Path

    def _verify_parent_materials(self) -> None:
        """Reject a changed executable import boundary before child launch."""

        try:
            if (
                hashlib.sha256(
                    _read_local_file(self.configuration_path, limit=_MAX_CONFIGURATION_BYTES)
                ).hexdigest()
                != self.configuration_sha256
            ):
                raise RasterTextRuntimeError("raster_text_configuration_invalid")
            _validate_profile_controlled_child_file(
                profile=self.profile,
                root_path=self.interpreter_root,
                file_path=self.interpreter,
            )
            _validate_profile_controlled_child_file(
                profile=self.profile,
                root_path=self.module_root,
                file_path=self.module,
            )
            if (
                _sha256_regular_file(self.interpreter) != self.profile.sidecar_interpreter_sha256
                or _sha256_regular_file(self.module) != self.profile.sidecar_module_sha256
                or _controlled_directory_tree_sha256(self.interpreter_root)
                != self.profile.sidecar_interpreter_root_tree_sha256
                or _controlled_module_root_tree_sha256(self.module_root)
                != self.profile.sidecar_module_root_tree_sha256
            ):
                raise RasterTextRuntimeError("raster_text_sidecar_drift")
        except RasterTextRuntimeError:
            raise
        except Exception:
            raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None

    def _verify_parent_binding_materials(self) -> None:
        """Cheaply recheck the already-started child's parent-visible binding.

        Evaluation opens only after :meth:`_verify_parent_materials` has
        completed a full tree verification.  Repeating that expensive walk
        for each sealed asset makes the first candidate deadline meaningless.
        This narrow check still binds the exact configuration, interpreter,
        and sidecar module files; the persistent child independently performs
        config/package/model checks before and after every OCR frame.
        """

        try:
            if (
                hashlib.sha256(
                    _read_local_file(self.configuration_path, limit=_MAX_CONFIGURATION_BYTES)
                ).hexdigest()
                != self.configuration_sha256
            ):
                raise RasterTextRuntimeError("raster_text_configuration_invalid")
            _validate_profile_controlled_child_file(
                profile=self.profile,
                root_path=self.interpreter_root,
                file_path=self.interpreter,
            )
            _validate_profile_controlled_child_file(
                profile=self.profile,
                root_path=self.module_root,
                file_path=self.module,
            )
            if (
                _sha256_regular_file(self.interpreter) != self.profile.sidecar_interpreter_sha256
                or _sha256_regular_file(self.module) != self.profile.sidecar_module_sha256
            ):
                raise RasterTextRuntimeError("raster_text_sidecar_drift")
        except RasterTextRuntimeError:
            raise
        except Exception:
            raise RasterTextRuntimeError("raster_text_runtime_material_invalid") from None

    def extract_text(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_text_profile_revision: str,
    ) -> RasterTextLedgerV1:
        if raster_text_profile_revision != self.profile.raster_text_profile_revision:
            raise RasterTextPreviewUnavailable("raster text profile revision is not active")
        try:
            self._verify_parent_materials()
            with _isolated_v2_sidecar_home() as runtime_home:
                completed = subprocess.run(
                    [
                        str(self.interpreter),
                        "-I",
                        "-B",
                        "-c",
                        _sidecar_bootstrap_for_profile(self.profile),
                        str(self.interpreter_root),
                        str(self.module_root),
                        str(self.module),
                        f"--{_PROTOCOL}",
                    ],
                    input=_sidecar_payload(
                        pages,
                        source_sha256=source_sha256,
                        raster_text_profile_revision=raster_text_profile_revision,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    cwd=str(self.module_root),
                    env=_sidecar_environment(
                        interpreter=self.interpreter,
                        module_root=self.module_root,
                        runtime_home=runtime_home,
                        configuration_path=self.configuration_path,
                        configuration_sha256=self.configuration_sha256,
                        profile=self.profile,
                        model_root=self.model_root,
                        model_inventory=self.model_inventory,
                        sidecar_inventory=self.sidecar_inventory,
                    ),
                    shell=False,
                    # Keep a cleanup margin inside the 55-second HTTP
                    # envelope. ``subprocess.run`` kills and reaps the child
                    # before this call returns on timeout.
                    timeout=min(
                        self.profile.sidecar_timeout_seconds,
                        _MAX_OWNED_TEXT_SIDECAR_SECONDS,
                    ),
                )
        except (OSError, subprocess.TimeoutExpired, RasterTextRuntimeError):
            raise RasterTextPreviewUnavailable("raster text sidecar is unavailable") from None
        if (
            completed.returncode != 0
            or not completed.stdout
            or len(completed.stdout) > _MAX_RESPONSE_BYTES
        ):
            raise RasterTextPreviewUnavailable("raster text sidecar is unavailable")
        try:
            result = RasterTextSidecarResultV1.model_validate_json(completed.stdout)
        except (ValidationError, UnicodeDecodeError, ValueError):
            raise RasterTextPreviewUnavailable("raster text sidecar response is invalid") from None
        if (
            result.source_sha256 != source_sha256
            or result.raster_text_profile_revision != raster_text_profile_revision
            or result.provider_revision != self.profile.extractor_revision
            # Empty output is a valid OCR observation.  The caller maps it to
            # ``needs_review`` rather than confusing it with an unavailable
            # sidecar; non-empty output remains one-to-one with input pages.
            or (result.pages and len(result.pages) != len(pages))
            or any(page.confidence is None for page in result.pages)
        ):
            raise RasterTextPreviewUnavailable("raster text sidecar binding is invalid")
        return RasterTextLedgerV1(
            source_sha256=result.source_sha256,
            raster_text_profile_revision=result.raster_text_profile_revision,
            provider_revision=result.provider_revision,
            pages=tuple(result.pages),
            minimum_page_confidence=self.profile.minimum_page_confidence,
            warnings=tuple(result.warnings),
            needs_review_reasons=tuple(result.needs_review_reasons),
        )

    def start_service_session(
        self, *, parent_materials_verified: bool = False
    ) -> "PaddleOcrTextServiceSession":
        """Start one warm, single-lane child for this verified profile.

        A customer request must never pay the native Paddle import/model-load
        cost.  The full parent-side executable/module-tree check happens once
        here (or immediately before this call in the immutable binding
        loader), and the child independently checks its package/model inventory
        before it emits ``ready``, and later request frames stay within the
        short owned OCR deadline.
        """

        cleanup_home: Callable[[], None] | None = None
        process: subprocess.Popen[bytes] | None = None
        session: PaddleOcrTextServiceSession | None = None
        try:
            if not parent_materials_verified:
                self._verify_parent_materials()
            runtime_home, cleanup_home = _new_isolated_v2_sidecar_home()
            process = subprocess.Popen(
                [
                    str(self.interpreter),
                    "-I",
                    "-B",
                    "-c",
                    _sidecar_bootstrap_for_profile(self.profile),
                    str(self.interpreter_root),
                    str(self.module_root),
                    str(self.module),
                    f"--{_SERVICE_PROTOCOL}",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self.module_root),
                env=_sidecar_environment(
                    interpreter=self.interpreter,
                    module_root=self.module_root,
                    runtime_home=runtime_home,
                    configuration_path=self.configuration_path,
                    configuration_sha256=self.configuration_sha256,
                    profile=self.profile,
                    model_root=self.model_root,
                    model_inventory=self.model_inventory,
                    sidecar_inventory=self.sidecar_inventory,
                ),
                shell=False,
                bufsize=0,
            )
            if process.stdin is None or process.stdout is None:
                raise OSError
            session = PaddleOcrTextServiceSession(
                provider=self,
                process=process,
                cleanup_home=cleanup_home,
                request_timeout_seconds=min(
                    self.profile.sidecar_timeout_seconds,
                    _MAX_OWNED_TEXT_SIDECAR_SECONDS,
                ),
                # The cold native import/model load is a separate setup
                # budget.  ``sidecar_timeout_seconds`` caps only an
                # already-ready document request and is normally 30 seconds.
                startup_timeout_seconds=raster_text_service_startup_timeout_seconds(self.profile),
            )
            session.verify_ready()
            return session
        except Exception:
            # This is a private local process boundary.  Deliberately discard
            # all native diagnostics rather than allowing child/model details
            # into a transcript or public error.
            if session is not None:
                session.abort()
            elif process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if session is None and cleanup_home is not None:
                try:
                    cleanup_home()
                except OSError:
                    pass
            raise RasterTextPreviewUnavailable("raster text service is unavailable") from None

    def start_activation_attested_service_session(self) -> "PaddleOcrTextServiceSession":
        """Start V11 after compact parent checks and child-owned full checks.

        This named entrypoint prevents evaluation or legacy callers from
        accidentally treating compact runtime attestation as a fresh full
        parent verification.  Activation already verified the complete tree;
        launch still rechecks the exact executable/module/configuration, and
        the child verifies package/model inventories before ``ready``.
        """

        self._verify_parent_binding_materials()
        return self.start_service_session(parent_materials_verified=True)

    @contextmanager
    def open_evaluation_session(
        self, *, parent_materials_verified: bool = False
    ) -> Iterator["PaddleOcrTextEvaluationSession"]:
        """Load direct OCR once for the fixed local synthetic evaluation.

        Customer OCR uses a separately prewarmed persistent service session.
        This private evaluation-only session likewise uses one ready-bound
        isolated interpreter/config/sidecar and protocol, but
        waits for its own explicit ready acknowledgement before the evaluator
        starts timing any calibration or sealed asset.  It is never an HTTP
        fallback or activation pointer resolver.
        """

        process: subprocess.Popen[bytes] | None = None
        try:
            # Candidate resolution normally performs the complete parent
            # interpreter/module-tree verification immediately before this
            # call.  Re-reading that same ~900 MiB import boundary here adds
            # minutes of local setup time without strengthening the binding.
            # Keep the default exhaustive for every independent caller, but
            # let the fixed activation runner reuse its just-completed check.
            # The lightweight direct-file check still closes the small launch
            # race; the child independently validates configuration, package,
            # and direct-model material before ``ready`` and around every
            # evaluated frame.
            if parent_materials_verified:
                self._verify_parent_binding_materials()
            else:
                self._verify_parent_materials()
            with _isolated_v2_sidecar_home() as runtime_home:
                process = subprocess.Popen(
                    [
                        str(self.interpreter),
                        "-I",
                        "-B",
                        "-c",
                        _sidecar_bootstrap_for_profile(self.profile),
                        str(self.interpreter_root),
                        str(self.module_root),
                        str(self.module),
                        f"--{_PROTOCOL}-session",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    cwd=str(self.module_root),
                    env=_sidecar_environment(
                        interpreter=self.interpreter,
                        module_root=self.module_root,
                        runtime_home=runtime_home,
                        configuration_path=self.configuration_path,
                        configuration_sha256=self.configuration_sha256,
                        profile=self.profile,
                        model_root=self.model_root,
                        model_inventory=self.model_inventory,
                        sidecar_inventory=self.sidecar_inventory,
                    ),
                    shell=False,
                    bufsize=0,
                )
                if process.stdin is None or process.stdout is None:
                    raise OSError
                session = PaddleOcrTextEvaluationSession(
                    provider=self,
                    process=process,
                    request_timeout_seconds=min(
                        self.profile.sidecar_timeout_seconds,
                        _MAX_OWNED_TEXT_SIDECAR_SECONDS,
                    ),
                    # Model import/load belongs to setup, not the first
                    # candidate.  Its bound is deliberately independent of
                    # the 30-second post-ready OCR request deadline.
                    startup_timeout_seconds=raster_text_service_startup_timeout_seconds(
                        self.profile
                    ),
                )
                try:
                    session.verify_ready()
                    yield session
                finally:
                    session.close()
                    process = None
        except (OSError, RasterTextRuntimeError, RasterTextPreviewUnavailable):
            raise RasterTextPreviewUnavailable(
                "raster text evaluation sidecar is unavailable"
            ) from None
        finally:
            if process is not None:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except OSError:
                    pass


def _read_exact_text_pipe(stream: BinaryIO, *, size: int) -> bytes:
    """Read one exact service-pipe segment without accepting truncation."""

    if size < 1:
        raise RasterTextRuntimeError("raster_text_service_output_invalid")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_text_service_response_from_pipe(stream: BinaryIO) -> bytes:
    raw_length = _read_exact_text_pipe(stream, size=8)
    length = struct.unpack(">Q", raw_length)[0]
    if not 1 <= length <= _MAX_RESPONSE_BYTES:
        raise RasterTextRuntimeError("raster_text_service_output_invalid")
    return _read_exact_text_pipe(stream, size=length)


def _write_exact_text_pipe(stream: BinaryIO, payload: bytes) -> None:
    """Write one pipe segment completely or fail the local child boundary."""

    remaining = memoryview(payload)
    while remaining:
        written = stream.write(remaining)
        if (
            isinstance(written, bool)
            or not isinstance(written, int)
            or not 1 <= written <= len(remaining)
        ):
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        remaining = remaining[written:]


def _write_framed_request_and_read_response(
    *, stdin: BinaryIO, stdout: BinaryIO, payload: bytes
) -> bytes:
    """Perform the full request exchange on one dedicated pipe worker."""

    _write_exact_text_pipe(stdin, struct.pack(">Q", len(payload)))
    _write_exact_text_pipe(stdin, payload)
    stdin.flush()
    return _read_text_service_response_from_pipe(stdout)


def _run_pipe_operation_with_timeout(
    operation: Callable[[], bytes],
    *,
    timeout_seconds: int,
    discard: Callable[[float], None],
) -> bytes:
    """Run a pipe operation under one deadline and reap its worker on timeout.

    A request is not considered started only after its bytes have reached the
    child: a blocked ``write``/``flush`` is just as harmful as a stalled
    response.  The timeout therefore wraps write, flush, frame header and
    frame body as a single operation.  On expiry the owning session closes
    both pipes and reaps its child before this helper gives control back to a
    caller, allowing the worker to finish rather than surviving in a pool.
    """

    result: list[bytes] = []
    failure: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(operation())
        except BaseException as error:  # Never propagate child/native details.
            failure.append(error)

    started = time.monotonic()
    deadline = started + timeout_seconds
    # Preserve bounded time to close pipes, reap the child and join this
    # worker before returning.  At the production 30-second request cap this
    # gives OCR 28 seconds and reserves two seconds for deterministic cleanup;
    # small test-only timeouts retain a proportional, non-zero reserve.
    cleanup_reserve = min(
        _PIPE_OPERATION_REAP_SECONDS,
        max(0.001, float(timeout_seconds) / 10),
    )
    operation_deadline = deadline - cleanup_reserve
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=max(0.0, operation_deadline - time.monotonic()))
    if worker.is_alive():
        cleanup_deadline = min(
            deadline,
            time.monotonic() + cleanup_reserve,
        )
        try:
            discard(cleanup_deadline)
        except Exception:
            # The caller still maps this private boundary to its public,
            # content-free unavailable status.
            pass
        worker.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        raise RasterTextRuntimeError("raster_text_service_unavailable")
    if failure:
        # A failed pipe exchange cannot safely keep its single-lane child.
        # Reap it within the same operation deadline rather than waiting for
        # a later caller path that might exceed the public request budget.
        cleanup_deadline = min(
            deadline,
            time.monotonic() + cleanup_reserve,
        )
        try:
            discard(cleanup_deadline)
        except Exception:
            pass
        if isinstance(failure[0], RasterTextRuntimeError):
            raise failure[0]
        raise RasterTextRuntimeError("raster_text_service_unavailable") from None
    if len(result) != 1:
        raise RasterTextRuntimeError("raster_text_service_output_invalid")
    return result[0]


@dataclass(slots=True)
class PaddleOcrTextServiceSession(RasterTextProvider):
    """One persistent, serial direct-OCR child for one approved profile.

    This deliberately has a different protocol and lifecycle from
    :class:`PaddleOcrTextEvaluationSession`.  It owns the isolated temporary
    home for the child lifetime and is destroyed as one unit when the pointer
    changes, a material check fails, or a request cannot finish on time.
    """

    provider: PaddleOcrTextCommandProvider
    process: subprocess.Popen[bytes]
    cleanup_home: Callable[[], None]
    request_timeout_seconds: int
    startup_timeout_seconds: int
    # Process retirement and isolated-home cleanup have independent timing
    # requirements: a request timeout must retire the child immediately but
    # may not synchronously spend public latency on filesystem deletion.
    _process_closed: bool = field(default=False, init=False, repr=False)
    _home_cleanup_scheduled: bool = field(default=False, init=False, repr=False)
    _home_cleanup_state: _DeferredHomeCleanupState = field(init=False, repr=False)
    _home_cleanup_token: object | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _home_cleanup_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not (
            1 <= self.request_timeout_seconds <= _MAX_OWNED_TEXT_SIDECAR_SECONDS
            and 1 <= self.startup_timeout_seconds <= _MAX_TEXT_SERVICE_STARTUP_SECONDS
        ):
            raise RasterTextRuntimeError("raster_text_service_session_invalid")
        self._home_cleanup_state = _DeferredHomeCleanupState(self.cleanup_home)

    @property
    def profile(self) -> RasterTextOfflineProfile:
        return self.provider.profile

    def _attempt_cleanup_home(self, *, deferred: bool) -> bool:
        """Attempt one isolated-home cleanup without exposing local details."""

        with self._home_cleanup_lock:
            if self._home_cleanup_scheduled and not deferred:
                return self._home_cleanup_state.cleaned
        succeeded = self._home_cleanup_state.attempt()
        if succeeded:
            self._release_deferred_home_cleanup()
        return succeeded

    def _release_deferred_home_cleanup(self) -> None:
        """Drop the exact detached callback once cleanup has succeeded."""

        with self._home_cleanup_lock:
            token = self._home_cleanup_token
            self._home_cleanup_token = None
        if token is not None:
            _remove_deferred_home_cleanup(token)

    def _run_deferred_cleanup_home(self) -> None:
        for attempt in range(_DEFERRED_HOME_CLEANUP_ATTEMPTS):
            if self._attempt_cleanup_home(deferred=True):
                return
            if attempt + 1 < _DEFERRED_HOME_CLEANUP_ATTEMPTS:
                time.sleep(_DEFERRED_HOME_CLEANUP_RETRY_SECONDS)

    def _schedule_cleanup_home(self) -> None:
        """Defer isolated-home deletion after a deadline-owned failure."""

        with self._home_cleanup_lock:
            if self._home_cleanup_state.cleaned or self._home_cleanup_scheduled:
                return
            self._home_cleanup_scheduled = True
            # The registry stores only detached cleanup state, not a bound
            # session callback.  It is populated before worker creation, so
            # both a daemon-start failure and exhausted retries retain a real
            # process-exit fallback without synchronously deleting this home.
            self._home_cleanup_token = _register_deferred_home_cleanup(self._home_cleanup_state)
        worker = threading.Thread(target=self._run_deferred_cleanup_home, daemon=True)
        try:
            worker.start()
        except RuntimeError:
            # Thread creation failure is deliberately content-free.  The
            # process remains retired and the import-time registry handler
            # owns eventual cleanup.  Keep ``scheduled`` true so a later
            # error-path abort never performs deletion synchronously.
            pass

    def _terminate_process(self, *, deadline: float | None = None) -> None:
        # Close both pipe ends first: this wakes a worker blocked in either a
        # request write/flush or a response read before process reap starts.
        reap_deadline = (
            time.monotonic() + _PIPE_OPERATION_REAP_SECONDS if deadline is None else deadline
        )
        for stream in (self.process.stdin, self.process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        if self.process.poll() is None:
            try:
                self.process.terminate()
                # Reserve part of the bounded reap window for a forced kill
                # and a real reap if graceful termination does not return.
                self.process.wait(timeout=max(0.0, (reap_deadline - time.monotonic()) / 2))
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                    self.process.wait(timeout=max(0.0, reap_deadline - time.monotonic()))
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def _discard(self, *, deadline: float | None = None) -> None:
        if not self._process_closed:
            self._process_closed = True
            self._terminate_process(deadline=deadline)
        if deadline is None:
            # If a timeout already scheduled deletion, do not turn a later
            # error-path ``abort`` into a synchronous filesystem walk.
            if not self._home_cleanup_scheduled:
                if not self._attempt_cleanup_home(deferred=False):
                    self._schedule_cleanup_home()
        else:
            # A deadline-owned path releases its public caller first, then
            # deletes its private isolated home in a daemon cleanup worker.
            self._schedule_cleanup_home()

    def abort(self) -> None:
        """Immediately terminate and reap a failed or superseded child."""

        with self._lock:
            self._discard()

    def _read_response_with_timeout(self, *, timeout_seconds: int) -> bytes:
        stdout = self.process.stdout
        if stdout is None:
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        return _run_pipe_operation_with_timeout(
            lambda: _read_text_service_response_from_pipe(cast(BinaryIO, stdout)),
            timeout_seconds=timeout_seconds,
            discard=lambda deadline: self._discard(deadline=deadline),
        )

    def _exchange_request_with_timeout(self, *, payload: bytes) -> bytes:
        stdin = self.process.stdin
        stdout = self.process.stdout
        if stdin is None or stdout is None:
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        return _run_pipe_operation_with_timeout(
            lambda: _write_framed_request_and_read_response(
                stdin=cast(BinaryIO, stdin),
                stdout=cast(BinaryIO, stdout),
                payload=payload,
            ),
            timeout_seconds=self.request_timeout_seconds,
            discard=lambda deadline: self._discard(deadline=deadline),
        )

    def verify_ready(self) -> None:
        """Accept only the bounded, content-free profile-bound ready frame."""

        with self._lock:
            if self._process_closed or self.process.poll() is not None:
                self._discard()
                raise RasterTextRuntimeError("raster_text_service_unavailable")
            try:
                raw = self._read_response_with_timeout(timeout_seconds=self.startup_timeout_seconds)
            except RasterTextRuntimeError:
                self._discard()
                raise
            expected = json.dumps(
                {
                    "protocol": _SERVICE_PROTOCOL,
                    "status": "ready",
                    "raster_text_profile_revision": self.profile.raster_text_profile_revision,
                    "provider_revision": self.profile.extractor_revision,
                },
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if raw != expected or self.process.poll() is not None:
                self._discard()
                raise RasterTextRuntimeError("raster_text_service_output_invalid")

    def verify_runtime_state(self) -> None:
        """Perform only the cheap liveness check of the warm child.

        Parent material identity is checked by the cached pointer validator;
        config/package/model identities are checked before and after every
        child request.  Rehashing entire model/import trees here would turn a
        warm request back into the slow one-shot path.
        """

        with self._lock:
            if self._process_closed or self.process.poll() is not None:
                self._discard()
                raise RasterTextRuntimeError("raster_text_service_unavailable")

    def verify_runtime_state_nonblocking(self) -> None:
        """Probe an idle child without waiting behind an active OCR request.

        Holding the session lock means the single native child is currently
        exchanging one bounded request.  That is a healthy busy state, not a
        reason for the lifecycle poll to block Agent status behind OCR.  When
        the lane is idle, retain the same fail-closed process check as the
        strict request-bound verifier.
        """

        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._process_closed or self.process.poll() is not None:
                self._discard()
                raise RasterTextRuntimeError("raster_text_service_unavailable")
        finally:
            self._lock.release()

    def extract_text(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_text_profile_revision: str,
    ) -> RasterTextLedgerV1:
        if raster_text_profile_revision != self.profile.raster_text_profile_revision:
            raise RasterTextPreviewUnavailable("raster text profile revision is not active")
        try:
            payload = _sidecar_payload(
                pages,
                source_sha256=source_sha256,
                raster_text_profile_revision=raster_text_profile_revision,
            )
            with self._lock:
                if self._process_closed or self.process.poll() is not None:
                    self._discard()
                    raise RasterTextRuntimeError("raster_text_service_unavailable")
                raw = self._exchange_request_with_timeout(payload=payload)
                if self.process.poll() is not None:
                    self._discard()
                    raise RasterTextRuntimeError("raster_text_service_unavailable")
            result = RasterTextSidecarResultV1.model_validate_json(raw)
        except RasterTextPreviewUnavailable:
            self.abort()
            raise
        except (OSError, RasterTextRuntimeError, ValidationError, UnicodeDecodeError, ValueError):
            self.abort()
            raise RasterTextPreviewUnavailable("raster text service is unavailable") from None
        if (
            result.source_sha256 != source_sha256
            or result.raster_text_profile_revision != raster_text_profile_revision
            or result.provider_revision != self.profile.extractor_revision
            or (result.pages and len(result.pages) != len(pages))
            or any(page.confidence is None for page in result.pages)
        ):
            self.abort()
            raise RasterTextPreviewUnavailable("raster text service is unavailable")
        return RasterTextLedgerV1(
            source_sha256=result.source_sha256,
            raster_text_profile_revision=result.raster_text_profile_revision,
            provider_revision=result.provider_revision,
            pages=tuple(result.pages),
            minimum_page_confidence=self.profile.minimum_page_confidence,
            warnings=tuple(result.warnings),
            needs_review_reasons=tuple(result.needs_review_reasons),
        )

    def close(self) -> None:
        """End the child and its private home on shutdown or pointer drift."""

        self.abort()


@dataclass(slots=True)
class PaddleOcrTextEvaluationSession(RasterTextProvider):
    """Ready-bound persistent child used only for immutable evaluation assets."""

    provider: PaddleOcrTextCommandProvider
    process: subprocess.Popen[bytes]
    request_timeout_seconds: int | None = None
    startup_timeout_seconds: int | None = None
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.request_timeout_seconds is None:
            self.request_timeout_seconds = min(
                self.provider.profile.sidecar_timeout_seconds,
                _MAX_OWNED_TEXT_SIDECAR_SECONDS,
            )
        if self.startup_timeout_seconds is None:
            self.startup_timeout_seconds = raster_text_service_startup_timeout_seconds(
                self.provider.profile
            )
        if not (
            1 <= self.request_timeout_seconds <= _MAX_OWNED_TEXT_SIDECAR_SECONDS
            and 1 <= self.startup_timeout_seconds <= _MAX_TEXT_SERVICE_STARTUP_SECONDS
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_session_invalid")

    def _close_unlocked(self, *, deadline: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        # Match service cleanup: both sides of the pipe must be closed before
        # reap so a timed-out read/write worker cannot remain blocked.
        reap_deadline = (
            time.monotonic() + _PIPE_OPERATION_REAP_SECONDS if deadline is None else deadline
        )
        for stream in (self.process.stdin, self.process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=max(0.0, (reap_deadline - time.monotonic()) / 2))
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                    self.process.wait(timeout=max(0.0, reap_deadline - time.monotonic()))
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def _read_ready_with_timeout(self) -> bytes:
        stdout = self.process.stdout
        if stdout is None:
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        assert self.startup_timeout_seconds is not None
        return _run_pipe_operation_with_timeout(
            lambda: _read_text_service_response_from_pipe(cast(BinaryIO, stdout)),
            timeout_seconds=self.startup_timeout_seconds,
            discard=lambda deadline: self._close_unlocked(deadline=deadline),
        )

    def _exchange_request_with_timeout(self, *, payload: bytes) -> bytes:
        stdin = self.process.stdin
        stdout = self.process.stdout
        if stdin is None or stdout is None:
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        assert self.request_timeout_seconds is not None
        return _run_pipe_operation_with_timeout(
            lambda: _write_framed_request_and_read_response(
                stdin=cast(BinaryIO, stdin),
                stdout=cast(BinaryIO, stdout),
                payload=payload,
            ),
            timeout_seconds=self.request_timeout_seconds,
            discard=lambda deadline: self._close_unlocked(deadline=deadline),
        )

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def verify_ready(self) -> None:
        """Wait for cold model loading before timing the first candidate."""

        with self._lock:
            if self._closed or self.process.poll() is not None:
                self._close_unlocked()
                raise RasterTextPreviewUnavailable("raster text evaluation sidecar is unavailable")
            try:
                raw = self._read_ready_with_timeout()
            except RasterTextRuntimeError:
                self._close_unlocked()
                raise RasterTextPreviewUnavailable(
                    "raster text evaluation sidecar is unavailable"
                ) from None
            expected = json.dumps(
                {
                    "protocol": _EVALUATION_SESSION_PROTOCOL,
                    "status": "ready",
                    "raster_text_profile_revision": self.provider.profile.raster_text_profile_revision,
                    "provider_revision": self.provider.profile.extractor_revision,
                },
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if raw != expected or self.process.poll() is not None:
                self._close_unlocked()
                raise RasterTextPreviewUnavailable("raster text evaluation sidecar is unavailable")

    def extract_text(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_text_profile_revision: str,
    ) -> RasterTextLedgerV1:
        if raster_text_profile_revision != self.provider.profile.raster_text_profile_revision:
            raise RasterTextPreviewUnavailable("raster text profile revision is not active")
        try:
            payload = _sidecar_payload(
                pages,
                source_sha256=source_sha256,
                raster_text_profile_revision=raster_text_profile_revision,
            )
            with self._lock:
                if self._closed or self.process.poll() is not None:
                    self._close_unlocked()
                    raise RasterTextPreviewUnavailable(
                        "raster text evaluation sidecar is unavailable"
                    )
                # Full tree verification happened before session startup.  Do
                # not rehash it for every asset; the child handles complete
                # config/package/model verification before and after the
                # framed OCR call.
                self.provider._verify_parent_binding_materials()
                raw = self._exchange_request_with_timeout(payload=payload)
                if self.process.poll() is not None:
                    self._close_unlocked()
                    raise RasterTextPreviewUnavailable(
                        "raster text evaluation sidecar is unavailable"
                    )
            result = RasterTextSidecarResultV1.model_validate_json(raw)
        except RasterTextPreviewUnavailable:
            self.close()
            raise
        except (OSError, ValidationError, ValueError, RasterTextRuntimeError):
            self.close()
            raise RasterTextPreviewUnavailable(
                "raster text evaluation sidecar is unavailable"
            ) from None
        if (
            result.source_sha256 != source_sha256
            or result.raster_text_profile_revision != raster_text_profile_revision
            or result.provider_revision != self.provider.profile.extractor_revision
            or (result.pages and len(result.pages) != len(pages))
            or any(page.confidence is None for page in result.pages)
        ):
            self.close()
            raise RasterTextPreviewUnavailable("raster text evaluation sidecar binding is invalid")
        return RasterTextLedgerV1(
            source_sha256=result.source_sha256,
            raster_text_profile_revision=result.raster_text_profile_revision,
            provider_revision=result.provider_revision,
            pages=tuple(result.pages),
            minimum_page_confidence=self.provider.profile.minimum_page_confidence,
            warnings=tuple(result.warnings),
            needs_review_reasons=tuple(result.needs_review_reasons),
        )


@dataclass(frozen=True, slots=True)
class RasterTextPreviewEngineResolution:
    engine: LocalRasterTextPreviewEngine
    status: Literal["disabled", "available", "unavailable"]
    raster_text_profile_revision: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class RasterTextCandidateEngineResolutionV1:
    """Evaluation-only direct-text engine resolved without an approval pointer."""

    engine: LocalRasterTextPreviewEngine | None
    profile: RasterTextOfflineProfile | None
    runtime_configuration_sha256: str | None
    status: Literal["available", "unavailable"]


@dataclass(frozen=True, slots=True)
class SourceAttestedRasterTextPreviewBindingV1:
    """One immutable active direct-text engine/session pointer snapshot."""

    root: Path
    approval_path: Path
    approval_sha256: str
    pointer: RasterTextActivationPointer
    profile: RasterTextOfflineProfile
    engine: LocalRasterTextPreviewEngine
    runtime_verifier: Callable[[], None] | None = None
    runtime_nonblocking_verifier: Callable[[], None] | None = None
    runtime_closer: Callable[[], None] | None = None

    @property
    def raster_text_profile_revision(self) -> str:
        return self.profile.raster_text_profile_revision

    def verify_runtime_state(self) -> None:
        if self.runtime_verifier is None:
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        self.runtime_verifier()

    def verify_runtime_state_nonblocking(self) -> None:
        if self.runtime_nonblocking_verifier is None:
            raise RasterTextRuntimeError("raster_text_service_unavailable")
        self.runtime_nonblocking_verifier()

    def close(self) -> None:
        if self.runtime_closer is not None:
            self.runtime_closer()


def _current_raster_text_runner_implementation_sha256() -> str:
    """Hash the exact reviewed OCR/evaluation implementation manifest.

    V10/V11 bind current-runtime smoke evidence to this digest.  Recomputing
    thirteen small source files at process start is cheap, and ensures a code
    change cannot reuse a pointer activated for another implementation.
    """

    try:
        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            _runner_implementation_sha256,
        )

        value = _runner_implementation_sha256()
        if _SHA256.fullmatch(value) is None:
            raise ValueError
        return value
    except Exception:
        raise RasterTextRuntimeError("raster_text_runner_implementation_invalid") from None


def _verify_runtime_attested_warm_service_smoke(
    *,
    root: Path,
    raw: bytes,
    profile: RasterTextOfflineProfile,
    runtime_configuration_sha256: str,
    expected_runner_implementation_sha256: str,
    expected_candidate_execution_path: str | None = None,
    expected_candidate_execution_sha256: str | None = None,
    latency_p95_limit_seconds: float = 10.0,
) -> RasterTextWarmServiceSmokeReceiptV1:
    """Verify compact, content-addressed warm evidence without replaying OCR Gold.

    Full source/Gold/candidate equality is proved before the pointer is
    atomically written.  Service startup reopens the canonical receipt,
    execution and 24-result bundle, verifies every direct hash and runtime
    binding, and leaves the expensive 2,400-asset replay to activation.
    """

    try:
        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            RasterTextCandidateBundleV1,
            RasterTextCandidateExecutionV1,
        )

        if (
            (expected_candidate_execution_path is None)
            != (expected_candidate_execution_sha256 is None)
            or latency_p95_limit_seconds <= 0
            or latency_p95_limit_seconds > 30
        ):
            raise ValueError
        warm = RasterTextWarmServiceSmokeReceiptV1.model_validate_json(raw)
        if raw != _canonical_evaluation_json_bytes(warm.model_dump(mode="json", warnings=False)):
            raise ValueError
        if expected_candidate_execution_path is not None and (
            warm.candidate_execution_path != expected_candidate_execution_path
            or warm.candidate_execution_sha256 != expected_candidate_execution_sha256
        ):
            raise ValueError

        source_raw = _read_local_file(
            _sidecar_path(root, warm.source_gold_receipt_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        execution_raw = _read_local_file(
            _sidecar_path(root, warm.candidate_execution_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        if (
            hashlib.sha256(source_raw).hexdigest() != warm.source_gold_receipt_sha256
            or hashlib.sha256(execution_raw).hexdigest() != warm.candidate_execution_sha256
        ):
            raise ValueError
        source = _read_json_object(source_raw, code="raster_text_evaluation_invalid")
        execution = RasterTextCandidateExecutionV1.model_validate_json(execution_raw)
        if source_raw != _canonical_evaluation_json_bytes(
            source
        ) or execution_raw != _canonical_evaluation_json_bytes(
            execution.model_dump(mode="json", warnings=False)
        ):
            raise ValueError

        bundle_raw = _read_local_file(
            _sidecar_path(root, execution.candidate_bundle_relative_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        if hashlib.sha256(bundle_raw).hexdigest() != execution.candidate_bundle_sha256:
            raise ValueError
        bundle = RasterTextCandidateBundleV1.model_validate_json(bundle_raw)
        if bundle_raw != _canonical_evaluation_json_bytes(
            bundle.model_dump(mode="json", warnings=False)
        ):
            raise ValueError

        if (
            source.get("artifact_kind") != "synthetic_raster_text_source_gold_receipt"
            or source.get("dataset_id") != "synthetic-raster-text-v1"
            or source.get("scope") != "smoke"
            or source.get("source_truth_kind") != "pdf_content_stream_text_v1"
            or not isinstance(source.get("bindings"), list)
            or len(cast(list[object], source["bindings"])) != warm.asset_count
            or execution.scope != "smoke"
            or bundle.scope != "smoke"
            or execution.candidate_count != warm.asset_count
            or len(bundle.candidates) != warm.asset_count
            or execution.candidate_bundle_sha256 != warm.candidate_bundle_sha256
            or execution.candidate_bundle_sha256 != hashlib.sha256(bundle_raw).hexdigest()
            or bundle.runner_implementation_sha256 != expected_runner_implementation_sha256
            or source.get("synthetic_raster_receipt_sha256")
            != execution.synthetic_raster_receipt_sha256
            or bundle.synthetic_raster_receipt_sha256 != execution.synthetic_raster_receipt_sha256
            or execution.runtime_configuration_sha256 != runtime_configuration_sha256
            or bundle.runtime_configuration_sha256 != runtime_configuration_sha256
            or execution.raster_text_profile_revision != profile.raster_text_profile_revision
            or bundle.raster_text_profile_revision != profile.raster_text_profile_revision
            or execution.extractor_revision != profile.extractor_revision
            or bundle.extractor_revision != profile.extractor_revision
            or warm.raster_text_profile_revision != profile.raster_text_profile_revision
            or warm.runtime_configuration_sha256 != runtime_configuration_sha256
            or warm.extractor_revision != profile.extractor_revision
            or warm.startup_latency_seconds > raster_text_service_startup_timeout_seconds(profile)
            or warm.startup_timeout_seconds != raster_text_service_startup_timeout_seconds(profile)
            or warm.request_timeout_seconds
            != min(profile.sidecar_timeout_seconds, _MAX_OWNED_TEXT_SIDECAR_SECONDS)
            or warm.warm_latency_p95_seconds > latency_p95_limit_seconds
            or execution.latency_p95_seconds > latency_p95_limit_seconds
            or raster_text_exact_result_set_sha256(
                [(item.asset_id, item.profile_id, item.candidate) for item in bundle.candidates]
            )
            != warm.equality_set_sha256
        ):
            raise ValueError
        return warm
    except RasterTextRuntimeError:
        raise
    except Exception:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None


def _verify_runtime_attested_v11_evidence(
    *,
    root: Path,
    pointer: RasterTextActivationPointerV11,
    evaluation: RasterTextSealedEvaluationV8,
    profile: RasterTextOfflineProfileV19,
    evidence_by_path: Mapping[str, bytes],
) -> None:
    """Check V11 activation attestations without rebuilding sealed metrics."""

    current_runner_sha256 = _current_raster_text_runner_implementation_sha256()
    if (
        pointer.current_runner_implementation_sha256 != current_runner_sha256
        or evaluation.current_runner_implementation_sha256 != current_runner_sha256
    ):
        raise RasterTextRuntimeError("raster_text_runner_implementation_invalid")
    historical = _verify_runtime_attested_warm_service_smoke(
        root=root,
        raw=evidence_by_path[evaluation.warm_service_smoke_receipt_path],
        profile=profile,
        runtime_configuration_sha256=pointer.runtime_configuration_sha256,
        expected_runner_implementation_sha256=(evaluation.historical_runner_implementation_sha256),
        latency_p95_limit_seconds=10.0,
    )
    _verify_runtime_attested_warm_service_smoke(
        root=root,
        raw=evidence_by_path[evaluation.revalidation_smoke_receipt_path],
        profile=profile,
        runtime_configuration_sha256=pointer.runtime_configuration_sha256,
        expected_runner_implementation_sha256=current_runner_sha256,
        expected_candidate_execution_path=(evaluation.revalidation_candidate_execution_path),
        expected_candidate_execution_sha256=(evaluation.revalidation_candidate_execution_sha256),
        latency_p95_limit_seconds=10.0,
    )
    if evaluation.service_warm_latency_p95_seconds != historical.warm_latency_p95_seconds:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")


def _verify_warm_service_smoke_evidence(
    *,
    root: Path,
    raw: bytes,
    profile: RasterTextOfflineProfile,
    runtime_configuration_sha256: str,
    expected_runner_implementation_sha256: str | None = None,
    expected_candidate_execution_path: str | None = None,
    expected_candidate_execution_sha256: str | None = None,
    latency_p95_limit_seconds: float = 30.0,
) -> None:
    """Rebuild the warm-service/candidate equality chain before activation.

    This is intentionally local to the pointer loader rather than trusting a
    precomputed smoke receipt.  A matching receipt hash alone is insufficient
    if it references a different final profile or a changed candidate bundle.
    """

    try:
        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            verify_source_attested_raster_text_evidence,
        )

        if (
            (expected_candidate_execution_path is None)
            != (expected_candidate_execution_sha256 is None)
            or latency_p95_limit_seconds <= 0
            or latency_p95_limit_seconds > 30
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        warm = RasterTextWarmServiceSmokeReceiptV1.model_validate_json(raw)
        if raw != _canonical_evaluation_json_bytes(warm.model_dump(mode="json", warnings=False)):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        source_raw = _read_local_file(
            _sidecar_path(root, warm.source_gold_receipt_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        execution_raw = _read_local_file(
            _sidecar_path(root, warm.candidate_execution_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        if (
            hashlib.sha256(source_raw).hexdigest() != warm.source_gold_receipt_sha256
            or hashlib.sha256(execution_raw).hexdigest() != warm.candidate_execution_sha256
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        source, execution, bundle, _cases = verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=warm.source_gold_receipt_path,
            candidate_execution_relative_path=warm.candidate_execution_path,
            scope="smoke",
            expected_raster_text_profile_revision=profile.raster_text_profile_revision,
            expected_runtime_configuration_sha256=runtime_configuration_sha256,
            expected_extractor_revision=profile.extractor_revision,
            expected_runner_implementation_sha256=expected_runner_implementation_sha256,
        )
        source_keys = {(item.asset_id, item.profile_id) for item in source.bindings}
        candidate_keys = {(item.asset_id, item.profile_id) for item in bundle.candidates}
        if (
            source.scope != "smoke"
            or execution.scope != "smoke"
            or bundle.scope != "smoke"
            or len(source.bindings) != warm.asset_count
            or execution.candidate_count != warm.asset_count
            or len(bundle.candidates) != warm.asset_count
            or source_keys != candidate_keys
            or execution.candidate_bundle_sha256 != warm.candidate_bundle_sha256
            or source.synthetic_raster_receipt_sha256 != execution.synthetic_raster_receipt_sha256
            or bundle.synthetic_raster_receipt_sha256 != execution.synthetic_raster_receipt_sha256
            or execution.runtime_configuration_sha256 != runtime_configuration_sha256
            or bundle.runtime_configuration_sha256 != runtime_configuration_sha256
            or execution.raster_text_profile_revision != profile.raster_text_profile_revision
            or bundle.raster_text_profile_revision != profile.raster_text_profile_revision
            or execution.extractor_revision != profile.extractor_revision
            or bundle.extractor_revision != profile.extractor_revision
            or (
                expected_runner_implementation_sha256 is not None
                and bundle.runner_implementation_sha256 != expected_runner_implementation_sha256
            )
            or (
                expected_candidate_execution_path is not None
                and (
                    warm.candidate_execution_path != expected_candidate_execution_path
                    or warm.candidate_execution_sha256 != expected_candidate_execution_sha256
                )
            )
            or warm.raster_text_profile_revision != profile.raster_text_profile_revision
            or warm.runtime_configuration_sha256 != runtime_configuration_sha256
            or warm.extractor_revision != profile.extractor_revision
            or warm.startup_latency_seconds > raster_text_service_startup_timeout_seconds(profile)
            or warm.startup_timeout_seconds != raster_text_service_startup_timeout_seconds(profile)
            or warm.request_timeout_seconds
            != min(profile.sidecar_timeout_seconds, _MAX_OWNED_TEXT_SIDECAR_SECONDS)
            or warm.warm_latency_p95_seconds > latency_p95_limit_seconds
            or execution.latency_p95_seconds > latency_p95_limit_seconds
            or raster_text_exact_result_set_sha256(
                [(item.asset_id, item.profile_id, item.candidate) for item in bundle.candidates]
            )
            != warm.equality_set_sha256
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    except RasterTextRuntimeError:
        raise
    except Exception:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None


def _verify_fresh_v4_evidence_chain(
    *,
    root: Path,
    pointer: RasterTextActivationPointerV4,
    evaluation: RasterTextSealedEvaluationV4,
    calibration: RasterTextCalibrationReceiptV4,
    profile: RasterTextOfflineProfile,
    warm_service_smoke_raw: bytes,
) -> None:
    """Rebuild the complete fresh calibration/sealed chain for schema 4."""

    try:
        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            verify_source_attested_fresh_raster_text_evidence,
        )

        if not isinstance(profile, RasterTextOfflineProfileV14):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        selection_raw = _read_local_file(
            _sidecar_path(root, evaluation.selection_receipt_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        calibration_configuration_raw = _read_local_file(
            _sidecar_path(root, calibration.calibration_runtime_configuration_path),
            limit=_MAX_CONFIGURATION_BYTES,
        )
        if (
            hashlib.sha256(selection_raw).hexdigest() != evaluation.selection_receipt_sha256
            or evaluation.selection_receipt_path != calibration.selection_receipt_path
            or evaluation.selection_receipt_sha256 != calibration.selection_receipt_sha256
            or hashlib.sha256(calibration_configuration_raw).hexdigest()
            != calibration.calibration_runtime_configuration_sha256
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        calibration_profile = parse_raster_text_offline_profile(calibration_configuration_raw)
        if not isinstance(calibration_profile, RasterTextOfflineProfileV14):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        warm = RasterTextWarmServiceSmokeReceiptV1.model_validate_json(warm_service_smoke_raw)
        if warm_service_smoke_raw != _canonical_evaluation_json_bytes(
            warm.model_dump(mode="json", warnings=False)
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        activated_material = profile.model_dump(
            mode="json",
            warnings=False,
            exclude={
                "minimum_page_confidence",
                "profile_fingerprint_sha256",
                "raster_text_profile_revision",
            },
        )
        calibration_material = calibration_profile.model_dump(
            mode="json",
            warnings=False,
            exclude={
                "minimum_page_confidence",
                "profile_fingerprint_sha256",
                "raster_text_profile_revision",
            },
        )
        if (
            activated_material != calibration_material
            or calibration_profile.raster_text_profile_revision
            != calibration.calibration_raster_text_profile_revision
            or calibration_profile.extractor_revision != calibration.extractor_revision
            or calibration_profile.minimum_page_confidence != 0.0
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")

        sealed_source, sealed_execution, sealed_bundle, sealed_cases = (
            verify_source_attested_fresh_raster_text_evidence(
                document_preview_root=root,
                source_gold_receipt_relative_path=evaluation.source_gold_receipt_path,
                candidate_execution_relative_path=evaluation.candidate_execution_path,
                scope="fresh-sealed",
            )
        )
        calibration_source, calibration_execution, calibration_bundle, calibration_cases = (
            verify_source_attested_fresh_raster_text_evidence(
                document_preview_root=root,
                source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
                candidate_execution_relative_path=calibration.candidate_execution_path,
                scope="fresh-calibration",
            )
        )
        # Schema 4 predates result-blind projections and staged decisions.
        # The shared verifier now understands both generations, so keep the
        # old runtime contract explicit: V4 may consume only the complete
        # 50-family calibration and 100-family sealed executions it originally
        # attested.  Projected evidence belongs exclusively to schema 5.
        if any(
            hasattr(value, "projection_receipt_relative_path")
            for value in (
                sealed_execution,
                sealed_bundle,
                calibration_execution,
                calibration_bundle,
            )
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        sealed_scores, calibrated_floor, sealed_gate_passed = (
            _recalculate_source_attested_raster_text_quality(
                sealed_cases=sealed_cases,
                calibration_cases=calibration_cases,
                metric_revision="source-attested-semantic-text-v3",
            )
        )
        if (
            pointer.evidence_revision != evaluation.evidence_revision
            or evaluation.evidence_revision != calibration.evidence_revision
            or sealed_source.scope != "fresh-sealed"
            or sealed_execution.scope != "fresh-sealed"
            or sealed_bundle.scope != "fresh-sealed"
            or calibration_source.scope != "fresh-calibration"
            or calibration_execution.scope != "fresh-calibration"
            or calibration_bundle.scope != "fresh-calibration"
            or sealed_source.selection_receipt_relative_path != evaluation.selection_receipt_path
            or sealed_source.selection_receipt_sha256 != evaluation.selection_receipt_sha256
            or sealed_execution.selection_receipt_relative_path != evaluation.selection_receipt_path
            or sealed_execution.selection_receipt_sha256 != evaluation.selection_receipt_sha256
            or sealed_bundle.selection_receipt_relative_path != evaluation.selection_receipt_path
            or sealed_bundle.selection_receipt_sha256 != evaluation.selection_receipt_sha256
            or calibration_source.selection_receipt_relative_path
            != calibration.selection_receipt_path
            or calibration_source.selection_receipt_sha256 != calibration.selection_receipt_sha256
            or calibration_execution.selection_receipt_relative_path
            != calibration.selection_receipt_path
            or calibration_execution.selection_receipt_sha256
            != calibration.selection_receipt_sha256
            or calibration_bundle.selection_receipt_relative_path
            != calibration.selection_receipt_path
            or calibration_bundle.selection_receipt_sha256 != calibration.selection_receipt_sha256
            or sealed_execution.source_gold_receipt_relative_path
            != evaluation.source_gold_receipt_path
            or sealed_execution.source_gold_receipt_sha256 != evaluation.source_gold_receipt_sha256
            or sealed_bundle.source_gold_receipt_sha256 != evaluation.source_gold_receipt_sha256
            or calibration_execution.source_gold_receipt_relative_path
            != calibration.source_gold_receipt_path
            or calibration_execution.source_gold_receipt_sha256
            != calibration.source_gold_receipt_sha256
            or calibration_bundle.source_gold_receipt_sha256
            != calibration.source_gold_receipt_sha256
            or sealed_execution.raster_text_profile_revision != profile.raster_text_profile_revision
            or sealed_execution.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or sealed_execution.extractor_revision != profile.extractor_revision
            or sealed_bundle.raster_text_profile_revision != profile.raster_text_profile_revision
            or sealed_bundle.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or sealed_bundle.extractor_revision != profile.extractor_revision
            or calibration_execution.raster_text_profile_revision
            != calibration_profile.raster_text_profile_revision
            or calibration_execution.runtime_configuration_sha256
            != calibration.calibration_runtime_configuration_sha256
            or calibration_execution.extractor_revision != calibration.extractor_revision
            or calibration_bundle.raster_text_profile_revision
            != calibration_profile.raster_text_profile_revision
            or calibration_bundle.runtime_configuration_sha256
            != calibration.calibration_runtime_configuration_sha256
            or calibration_bundle.extractor_revision != calibration.extractor_revision
            or calibration.latency_p95_seconds != calibration_execution.latency_p95_seconds
            or evaluation.latency_p95_seconds != sealed_execution.latency_p95_seconds
            or calibrated_floor != calibration.minimum_page_confidence
            or calibrated_floor != evaluation.minimum_page_confidence
            or calibrated_floor != profile.minimum_page_confidence
            or evaluation.service_warm_latency_p95_seconds != warm.warm_latency_p95_seconds
            or not sealed_gate_passed
            or [score.model_dump(mode="json", warnings=False) for score in sealed_scores]
            != [score.model_dump(mode="json", warnings=False) for score in evaluation.scores]
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    except RasterTextRuntimeError:
        raise
    except Exception:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None


def _profile_ocr_material(profile: RasterTextOfflineProfile) -> dict[str, object]:
    """Return the immutable OCR tuple, excluding calibration-derived identity."""

    return profile.model_dump(
        mode="json",
        warnings=False,
        exclude={
            "minimum_page_confidence",
            "profile_fingerprint_sha256",
            "raster_text_profile_revision",
        },
    )


def _verify_v19_finetune_evidence_provenance(
    *,
    root: Path,
    pointer: (
        RasterTextActivationPointerV7
        | RasterTextActivationPointerV10
        | RasterTextActivationPointerV11
    ),
    evaluation: RasterTextSealedEvaluationV7 | RasterTextSealedEvaluationV8,
    calibration: RasterTextCalibrationReceiptV7,
    profile: RasterTextOfflineProfileV19,
    calibration_profile: RasterTextOfflineProfileV19,
    selection_raw: bytes,
) -> None:
    """Bind V19 quality evidence to the exact leakage-safe training split."""

    try:
        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            RasterTextFreshScopeSelectionReceiptV1,
            RasterTextFreshScopeSelectionReceiptV2,
        )

        selection = RasterTextFreshScopeSelectionReceiptV2.model_validate_json(selection_raw)
        parent_raw = _read_local_file(
            _sidecar_path(root, selection.parent_selection_receipt_relative_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        parent = RasterTextFreshScopeSelectionReceiptV1.model_validate_json(parent_raw)
        if (
            selection_raw
            != _canonical_evaluation_json_bytes(selection.model_dump(mode="json", warnings=False))
            or hashlib.sha256(selection_raw).hexdigest() != evaluation.selection_receipt_sha256
            or parent_raw
            != _canonical_evaluation_json_bytes(parent.model_dump(mode="json", warnings=False))
            or hashlib.sha256(parent_raw).hexdigest() != selection.parent_selection_receipt_sha256
            or not (
                pointer.source_recognizer_selection_receipt_path
                == profile.source_recognizer_selection_receipt_relative_path
                == calibration_profile.source_recognizer_selection_receipt_relative_path
                == evaluation.selection_receipt_path
                == calibration.selection_receipt_path
            )
            or not (
                pointer.source_recognizer_selection_receipt_sha256
                == profile.source_recognizer_selection_receipt_sha256
                == calibration_profile.source_recognizer_selection_receipt_sha256
                == evaluation.selection_receipt_sha256
                == calibration.selection_receipt_sha256
            )
            or not (
                pointer.source_recognizer_parent_selection_receipt_sha256
                == profile.source_recognizer_parent_selection_receipt_sha256
                == calibration_profile.source_recognizer_parent_selection_receipt_sha256
                == evaluation.source_recognizer_parent_selection_receipt_sha256
                == calibration.source_recognizer_parent_selection_receipt_sha256
                == selection.parent_selection_receipt_sha256
            )
            or not (
                pointer.source_recognizer_release_sha256
                == profile.source_recognizer_release_sha256
                == calibration_profile.source_recognizer_release_sha256
                == evaluation.source_recognizer_release_sha256
                == calibration.source_recognizer_release_sha256
            )
            or not (
                pointer.source_recognizer_receipt_sha256
                == profile.source_recognizer_receipt_sha256
                == calibration_profile.source_recognizer_receipt_sha256
                == evaluation.source_recognizer_receipt_sha256
                == calibration.source_recognizer_receipt_sha256
            )
            or not (
                pointer.source_recognizer_training_intent_sha256
                == profile.source_recognizer_training_intent_sha256
                == calibration_profile.source_recognizer_training_intent_sha256
                == evaluation.source_recognizer_training_intent_sha256
                == calibration.source_recognizer_training_intent_sha256
            )
        ):
            raise ValueError
    except RasterTextRuntimeError:
        raise
    except Exception:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None


def _verify_v20_voyage_replay_evidence_provenance(
    *,
    root: Path,
    pointer: RasterTextActivationPointerV9,
    evaluation: RasterTextSealedEvaluationV7,
    calibration: RasterTextCalibrationReceiptV7,
    profile: RasterTextOfflineProfileV20,
    calibration_profile: RasterTextOfflineProfileV20,
    selection_raw: bytes,
) -> None:
    """Extend V7 evidence with V20 parent, recipe and corpus-manifest identity."""

    _verify_v19_finetune_evidence_provenance(
        root=root,
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
        profile=profile,
        calibration_profile=calibration_profile,
        selection_raw=selection_raw,
    )
    if (
        not (
            pointer.source_recognizer_parent_release_sha256
            == profile.source_recognizer_parent_release_sha256
            == calibration_profile.source_recognizer_parent_release_sha256
        )
        or not (
            pointer.source_recognizer_parent_receipt_sha256
            == profile.source_recognizer_parent_receipt_sha256
            == calibration_profile.source_recognizer_parent_receipt_sha256
        )
        or not (
            pointer.source_recognizer_voyage_recipe_set_sha256
            == profile.source_recognizer_voyage_recipe_set_sha256
            == calibration_profile.source_recognizer_voyage_recipe_set_sha256
        )
        or not (
            pointer.source_recognizer_train_manifest_sha256
            == profile.source_recognizer_train_manifest_sha256
            == calibration_profile.source_recognizer_train_manifest_sha256
        )
        or not (
            pointer.source_recognizer_development_manifest_sha256
            == profile.source_recognizer_development_manifest_sha256
            == calibration_profile.source_recognizer_development_manifest_sha256
        )
        or profile.source_recognizer_train_corpus_manifest
        != calibration_profile.source_recognizer_train_corpus_manifest
        or profile.source_recognizer_development_corpus_manifest
        != calibration_profile.source_recognizer_development_corpus_manifest
    ):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")


def _verify_fresh_staged_evidence_chain(
    *,
    root: Path,
    pointer: (
        RasterTextActivationPointerV5
        | RasterTextActivationPointerV6
        | RasterTextActivationPointerV7
        | RasterTextActivationPointerV9
        | RasterTextActivationPointerV10
        | RasterTextActivationPointerV11
    ),
    evaluation: (
        RasterTextSealedEvaluationV5
        | RasterTextSealedEvaluationV6
        | RasterTextSealedEvaluationV7
        | RasterTextSealedEvaluationV8
    ),
    calibration: (
        RasterTextCalibrationReceiptV5
        | RasterTextCalibrationReceiptV6
        | RasterTextCalibrationReceiptV7
    ),
    profile: RasterTextOfflineProfile,
    warm_service_smoke_raw: bytes,
) -> None:
    """Rebuild projected calibration and staged sealed evidence for V5/V6/V7.

    Projection and stage-decision summaries are never treated as Gold.  The
    evaluation module reopens the source-bound cases, reconstructs complete
    family membership, and recomputes the stage-1 raw counts and simultaneous
    Wilson decision before this verifier accepts their compact bindings.
    """

    try:
        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            RasterTextFreshEvaluationProjectionReceiptV1,
            RasterTextFreshProjectedCandidateBundleV1,
            RasterTextFreshProjectedCandidateExecutionV1,
            verify_fresh_raster_text_sealed_stage1_decision,
            verify_source_attested_fresh_raster_text_evidence,
            verify_source_attested_fresh_raster_text_evidence_v6,
            verify_source_attested_fresh_raster_text_evidence_v8,
        )

        is_v10 = type(pointer) in (
            RasterTextActivationPointerV10,
            RasterTextActivationPointerV11,
        )
        is_v9 = type(pointer) is RasterTextActivationPointerV9
        is_v6 = type(pointer) is RasterTextActivationPointerV6
        is_v7 = isinstance(pointer, RasterTextActivationPointerV7)
        if is_v10:
            if (
                type(evaluation) is not RasterTextSealedEvaluationV8
                or type(calibration) is not RasterTextCalibrationReceiptV7
            ):
                raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        elif (
            is_v6 != (type(evaluation) is RasterTextSealedEvaluationV6)
            or is_v6 != (type(calibration) is RasterTextCalibrationReceiptV6)
            or is_v7 != (type(evaluation) is RasterTextSealedEvaluationV7)
            or is_v7 != (type(calibration) is RasterTextCalibrationReceiptV7)
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        if (
            (
                is_v10
                and (
                    type(profile) is not RasterTextOfflineProfileV19
                    or profile.engine_kind != _V19_ENGINE_KIND
                )
            )
            or (is_v9 and type(profile) is not RasterTextOfflineProfileV20)
            or (
                is_v7
                and not is_v9
                and not is_v10
                and not isinstance(profile, RasterTextOfflineProfileV19)
            )
            or (is_v6 and not isinstance(profile, RasterTextOfflineProfileV18))
            or (not is_v6 and not is_v7 and not isinstance(profile, RasterTextOfflineProfileV15))
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        selection_raw = _read_local_file(
            _sidecar_path(root, evaluation.selection_receipt_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        calibration_configuration_raw = _read_local_file(
            _sidecar_path(root, calibration.calibration_runtime_configuration_path),
            limit=_MAX_CONFIGURATION_BYTES,
        )
        calibration_projection_raw = _read_local_file(
            _sidecar_path(root, calibration.projection_receipt_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        sealed_projection_raw = _read_local_file(
            _sidecar_path(root, evaluation.projection_receipt_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        if (
            hashlib.sha256(selection_raw).hexdigest() != evaluation.selection_receipt_sha256
            or evaluation.selection_receipt_path != calibration.selection_receipt_path
            or evaluation.selection_receipt_sha256 != calibration.selection_receipt_sha256
            or hashlib.sha256(calibration_configuration_raw).hexdigest()
            != calibration.calibration_runtime_configuration_sha256
            or hashlib.sha256(calibration_projection_raw).hexdigest()
            != calibration.projection_receipt_sha256
            or hashlib.sha256(sealed_projection_raw).hexdigest()
            != evaluation.projection_receipt_sha256
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        calibration_projection = RasterTextFreshEvaluationProjectionReceiptV1.model_validate_json(
            calibration_projection_raw
        )
        sealed_projection = RasterTextFreshEvaluationProjectionReceiptV1.model_validate_json(
            sealed_projection_raw
        )
        if calibration_projection_raw != _canonical_evaluation_json_bytes(
            calibration_projection.model_dump(mode="json", warnings=False)
        ) or sealed_projection_raw != _canonical_evaluation_json_bytes(
            sealed_projection.model_dump(mode="json", warnings=False)
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")

        calibration_profile = parse_raster_text_offline_profile(calibration_configuration_raw)
        if (
            (
                is_v10
                and (
                    type(calibration_profile) is not RasterTextOfflineProfileV19
                    or calibration_profile.engine_kind != _V19_ENGINE_KIND
                )
            )
            or (is_v9 and type(calibration_profile) is not RasterTextOfflineProfileV20)
            or (
                is_v7
                and not is_v9
                and not is_v10
                and not isinstance(calibration_profile, RasterTextOfflineProfileV19)
            )
            or (is_v6 and not isinstance(calibration_profile, RasterTextOfflineProfileV18))
            or (
                not is_v6
                and not is_v7
                and not isinstance(calibration_profile, RasterTextOfflineProfileV15)
            )
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        warm = RasterTextWarmServiceSmokeReceiptV1.model_validate_json(warm_service_smoke_raw)
        if warm_service_smoke_raw != _canonical_evaluation_json_bytes(
            warm.model_dump(mode="json", warnings=False)
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        if (
            _profile_ocr_material(profile) != _profile_ocr_material(calibration_profile)
            or calibration_profile.raster_text_profile_revision
            != calibration.calibration_raster_text_profile_revision
            or calibration_profile.extractor_revision != calibration.extractor_revision
            or calibration_profile.minimum_page_confidence != 0.0
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")

        sealed_source: Any
        sealed_execution: Any
        sealed_bundle: Any
        sealed_cases: Any
        calibration_source: Any
        calibration_execution: Any
        calibration_bundle: Any
        calibration_cases: Any
        if is_v7:
            assert isinstance(pointer, RasterTextActivationPointerV7)
            assert isinstance(evaluation, RasterTextSealedEvaluationV7)
            assert isinstance(calibration, RasterTextCalibrationReceiptV7)
            if is_v10:
                assert type(pointer) in (
                    RasterTextActivationPointerV10,
                    RasterTextActivationPointerV11,
                )
                assert type(evaluation) is RasterTextSealedEvaluationV8
                assert type(profile) is RasterTextOfflineProfileV19
                assert type(calibration_profile) is RasterTextOfflineProfileV19
                _verify_v19_finetune_evidence_provenance(
                    root=root,
                    pointer=pointer,
                    evaluation=evaluation,
                    calibration=calibration,
                    profile=profile,
                    calibration_profile=calibration_profile,
                    selection_raw=selection_raw,
                )
            elif is_v9:
                assert isinstance(pointer, RasterTextActivationPointerV9)
                assert isinstance(profile, RasterTextOfflineProfileV20)
                assert isinstance(calibration_profile, RasterTextOfflineProfileV20)
                _verify_v20_voyage_replay_evidence_provenance(
                    root=root,
                    pointer=pointer,
                    evaluation=evaluation,
                    calibration=calibration,
                    profile=profile,
                    calibration_profile=calibration_profile,
                    selection_raw=selection_raw,
                )
            else:
                assert isinstance(profile, RasterTextOfflineProfileV19)
                assert isinstance(calibration_profile, RasterTextOfflineProfileV19)
                _verify_v19_finetune_evidence_provenance(
                    root=root,
                    pointer=pointer,
                    evaluation=evaluation,
                    calibration=calibration,
                    profile=profile,
                    calibration_profile=calibration_profile,
                    selection_raw=selection_raw,
                )

        if is_v10:
            assert type(evaluation) is RasterTextSealedEvaluationV8
            sealed_source, sealed_execution, sealed_bundle, sealed_cases = (
                verify_source_attested_fresh_raster_text_evidence_v8(
                    document_preview_root=root,
                    source_gold_receipt_relative_path=evaluation.source_gold_receipt_path,
                    candidate_execution_relative_path=evaluation.candidate_execution_path,
                    scope="fresh-sealed",
                    historical_runner_implementation_sha256=(
                        evaluation.historical_runner_implementation_sha256
                    ),
                    current_runner_implementation_sha256=(
                        evaluation.current_runner_implementation_sha256
                    ),
                )
            )
            calibration_source, calibration_execution, calibration_bundle, calibration_cases = (
                verify_source_attested_fresh_raster_text_evidence_v8(
                    document_preview_root=root,
                    source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
                    candidate_execution_relative_path=calibration.candidate_execution_path,
                    scope="fresh-calibration",
                    historical_runner_implementation_sha256=(
                        evaluation.historical_runner_implementation_sha256
                    ),
                    current_runner_implementation_sha256=(
                        evaluation.current_runner_implementation_sha256
                    ),
                )
            )
        elif is_v6 or is_v7:
            sealed_source, sealed_execution, sealed_bundle, sealed_cases = (
                verify_source_attested_fresh_raster_text_evidence_v6(
                    document_preview_root=root,
                    source_gold_receipt_relative_path=evaluation.source_gold_receipt_path,
                    candidate_execution_relative_path=evaluation.candidate_execution_path,
                    scope="fresh-sealed",
                )
            )
            calibration_source, calibration_execution, calibration_bundle, calibration_cases = (
                verify_source_attested_fresh_raster_text_evidence_v6(
                    document_preview_root=root,
                    source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
                    candidate_execution_relative_path=calibration.candidate_execution_path,
                    scope="fresh-calibration",
                )
            )
        else:
            sealed_source, sealed_execution, sealed_bundle, sealed_cases = (
                verify_source_attested_fresh_raster_text_evidence(
                    document_preview_root=root,
                    source_gold_receipt_relative_path=evaluation.source_gold_receipt_path,
                    candidate_execution_relative_path=evaluation.candidate_execution_path,
                    scope="fresh-sealed",
                )
            )
            calibration_source, calibration_execution, calibration_bundle, calibration_cases = (
                verify_source_attested_fresh_raster_text_evidence(
                    document_preview_root=root,
                    source_gold_receipt_relative_path=calibration.source_gold_receipt_path,
                    candidate_execution_relative_path=calibration.candidate_execution_path,
                    scope="fresh-calibration",
                )
            )
        if (
            not isinstance(sealed_execution, RasterTextFreshProjectedCandidateExecutionV1)
            or not isinstance(sealed_bundle, RasterTextFreshProjectedCandidateBundleV1)
            or not isinstance(calibration_execution, RasterTextFreshProjectedCandidateExecutionV1)
            or not isinstance(calibration_bundle, RasterTextFreshProjectedCandidateBundleV1)
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        historical_runner_sha256: str | None = None
        if is_v6 or is_v7:
            historical_runner_sha256 = (
                evaluation.historical_runner_implementation_sha256
                if type(evaluation) is RasterTextSealedEvaluationV8
                else None
            )
            stage1_decision = verify_fresh_raster_text_sealed_stage1_decision(
                document_preview_root=root,
                decision_relative_path=evaluation.stage1_decision_receipt_path,
                expected_sha256=evaluation.stage1_decision_receipt_sha256,
                required_selection_revision="source-receipt-train-holdout-v2",
                expected_runner_implementation_sha256=historical_runner_sha256,
            )
        else:
            stage1_decision = verify_fresh_raster_text_sealed_stage1_decision(
                document_preview_root=root,
                decision_relative_path=evaluation.stage1_decision_receipt_path,
                expected_sha256=evaluation.stage1_decision_receipt_sha256,
            )
        sealed_scores, calibrated_floor, sealed_gate_passed = (
            _recalculate_source_attested_raster_text_quality(
                sealed_cases=sealed_cases,
                calibration_cases=calibration_cases,
                metric_revision="source-attested-semantic-text-v3",
                quality_gate_revision=(
                    evaluation.quality_gate_revision
                    if type(evaluation) is RasterTextSealedEvaluationV8
                    else None
                ),
            )
        )
        sealed_stage_is_valid = (
            evaluation.projection_kind == "sealed-stage1-50-family-v1"
            and evaluation.stage1_decision == "early_pass"
            and stage1_decision.decision == "early_pass"
            and stage1_decision.source_gold_receipt_relative_path
            == evaluation.source_gold_receipt_path
            and stage1_decision.source_gold_receipt_sha256 == evaluation.source_gold_receipt_sha256
            and stage1_decision.candidate_execution_relative_path
            == evaluation.candidate_execution_path
            and stage1_decision.candidate_execution_sha256 == evaluation.candidate_execution_sha256
            and stage1_decision.projection_receipt_relative_path
            == evaluation.projection_receipt_path
            and stage1_decision.projection_receipt_sha256 == evaluation.projection_receipt_sha256
        ) or (
            evaluation.projection_kind == "sealed-stage2-100-family-v1"
            and evaluation.stage1_decision == "continue"
            and stage1_decision.decision == "continue"
            and stage1_decision.source_gold_receipt_relative_path
            == evaluation.source_gold_receipt_path
            and stage1_decision.source_gold_receipt_sha256 == evaluation.source_gold_receipt_sha256
            and sealed_projection.parent_stage1_decision_relative_path
            == evaluation.stage1_decision_receipt_path
            and sealed_projection.parent_stage1_decision_sha256
            == evaluation.stage1_decision_receipt_sha256
        )
        if (
            pointer.evidence_revision != evaluation.evidence_revision
            or evaluation.evidence_revision != calibration.evidence_revision
            or pointer.latency_gate_revision != evaluation.latency_gate_revision
            or evaluation.latency_gate_revision != calibration.latency_gate_revision
            or calibration.projection_kind != "calibration-10-family-v1"
            or calibration_projection.projection_kind != calibration.projection_kind
            or calibration_projection.family_count != calibration.family_count
            or calibration_projection.asset_count != calibration.asset_count
            or sealed_projection.projection_kind != evaluation.projection_kind
            or sealed_projection.family_count != evaluation.family_count
            or sealed_projection.asset_count != evaluation.asset_count
            or sealed_source.scope != "fresh-sealed"
            or sealed_execution.scope != "fresh-sealed"
            or sealed_bundle.scope != "fresh-sealed"
            or calibration_source.scope != "fresh-calibration"
            or calibration_execution.scope != "fresh-calibration"
            or calibration_bundle.scope != "fresh-calibration"
            or sealed_source.selection_receipt_relative_path != evaluation.selection_receipt_path
            or sealed_source.selection_receipt_sha256 != evaluation.selection_receipt_sha256
            or sealed_execution.selection_receipt_relative_path != evaluation.selection_receipt_path
            or sealed_execution.selection_receipt_sha256 != evaluation.selection_receipt_sha256
            or sealed_bundle.selection_receipt_relative_path != evaluation.selection_receipt_path
            or sealed_bundle.selection_receipt_sha256 != evaluation.selection_receipt_sha256
            or calibration_source.selection_receipt_relative_path
            != calibration.selection_receipt_path
            or calibration_source.selection_receipt_sha256 != calibration.selection_receipt_sha256
            or calibration_execution.selection_receipt_relative_path
            != calibration.selection_receipt_path
            or calibration_execution.selection_receipt_sha256
            != calibration.selection_receipt_sha256
            or calibration_bundle.selection_receipt_relative_path
            != calibration.selection_receipt_path
            or calibration_bundle.selection_receipt_sha256 != calibration.selection_receipt_sha256
            or sealed_execution.source_gold_receipt_relative_path
            != evaluation.source_gold_receipt_path
            or sealed_execution.source_gold_receipt_sha256 != evaluation.source_gold_receipt_sha256
            or sealed_bundle.source_gold_receipt_sha256 != evaluation.source_gold_receipt_sha256
            or calibration_execution.source_gold_receipt_relative_path
            != calibration.source_gold_receipt_path
            or calibration_execution.source_gold_receipt_sha256
            != calibration.source_gold_receipt_sha256
            or calibration_bundle.source_gold_receipt_sha256
            != calibration.source_gold_receipt_sha256
            or sealed_execution.projection_receipt_relative_path
            != evaluation.projection_receipt_path
            or sealed_execution.projection_receipt_sha256 != evaluation.projection_receipt_sha256
            or sealed_bundle.projection_receipt_relative_path != evaluation.projection_receipt_path
            or sealed_bundle.projection_receipt_sha256 != evaluation.projection_receipt_sha256
            or calibration_execution.projection_receipt_relative_path
            != calibration.projection_receipt_path
            or calibration_execution.projection_receipt_sha256
            != calibration.projection_receipt_sha256
            or calibration_bundle.projection_receipt_relative_path
            != calibration.projection_receipt_path
            or calibration_bundle.projection_receipt_sha256 != calibration.projection_receipt_sha256
            or sealed_execution.candidate_count != evaluation.asset_count
            or len(sealed_bundle.candidates) != evaluation.asset_count
            or calibration_execution.candidate_count != calibration.asset_count
            or len(calibration_bundle.candidates) != calibration.asset_count
            or sealed_execution.raster_text_profile_revision != profile.raster_text_profile_revision
            or sealed_execution.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or sealed_execution.extractor_revision != profile.extractor_revision
            or sealed_bundle.raster_text_profile_revision != profile.raster_text_profile_revision
            or sealed_bundle.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or sealed_bundle.extractor_revision != profile.extractor_revision
            or calibration_execution.raster_text_profile_revision
            != calibration_profile.raster_text_profile_revision
            or calibration_execution.runtime_configuration_sha256
            != calibration.calibration_runtime_configuration_sha256
            or calibration_execution.extractor_revision != calibration.extractor_revision
            or calibration_bundle.raster_text_profile_revision
            != calibration_profile.raster_text_profile_revision
            or calibration_bundle.runtime_configuration_sha256
            != calibration.calibration_runtime_configuration_sha256
            or calibration_bundle.extractor_revision != calibration.extractor_revision
            or calibration.latency_p95_seconds != calibration_execution.latency_p95_seconds
            or evaluation.latency_p95_seconds != sealed_execution.latency_p95_seconds
            or calibrated_floor != calibration.minimum_page_confidence
            or calibrated_floor != evaluation.minimum_page_confidence
            or calibrated_floor != profile.minimum_page_confidence
            or evaluation.service_warm_latency_p95_seconds != warm.warm_latency_p95_seconds
            or (
                is_v10
                and (
                    historical_runner_sha256 is None
                    or sealed_bundle.runner_implementation_sha256 != historical_runner_sha256
                    or calibration_bundle.runner_implementation_sha256 != historical_runner_sha256
                )
            )
            or not sealed_stage_is_valid
            or not sealed_gate_passed
            or [score.model_dump(mode="json", warnings=False) for score in sealed_scores]
            != [score.model_dump(mode="json", warnings=False) for score in evaluation.scores]
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    except RasterTextRuntimeError:
        raise
    except Exception:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None


def _verify_fresh_v5_evidence_chain(
    *,
    root: Path,
    pointer: RasterTextActivationPointerV5,
    evaluation: RasterTextSealedEvaluationV5,
    calibration: RasterTextCalibrationReceiptV5,
    profile: RasterTextOfflineProfile,
    warm_service_smoke_raw: bytes,
) -> None:
    """Compatibility entry point for the already-published V5 verifier."""

    _verify_fresh_staged_evidence_chain(
        root=root,
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
        profile=profile,
        warm_service_smoke_raw=warm_service_smoke_raw,
    )


def _verify_legacy_raster_text_quality_evidence(
    *,
    root: Path,
    pointer: RasterTextActivationPointer,
    evaluation: RasterTextSealedEvaluation,
    calibration: RasterTextCalibrationReceipt,
    profile: RasterTextOfflineProfile,
    metric_revision: Literal["source-attested-semantic-text-v2", "source-attested-semantic-text-v3"]
    | None,
) -> None:
    """Preserve the complete schema-1/2/3 verification path unchanged."""

    try:
        from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
            verify_source_attested_raster_text_evidence,
        )

        sealed_source, sealed_execution, sealed_bundle, sealed_cases = (
            verify_source_attested_raster_text_evidence(
                document_preview_root=root,
                source_gold_receipt_relative_path=evaluation.source_gold_receipt_path,
                candidate_execution_relative_path=evaluation.candidate_execution_path,
                scope="sealed",
                expected_raster_text_profile_revision=profile.raster_text_profile_revision,
                expected_runtime_configuration_sha256=(pointer.runtime_configuration_sha256),
                expected_extractor_revision=profile.extractor_revision,
            )
        )
        (
            calibration_source,
            calibration_execution,
            calibration_bundle,
            calibration_cases,
        ) = verify_source_attested_raster_text_evidence(
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
        sealed_scores, calibrated_floor, sealed_gate_passed = (
            _recalculate_source_attested_raster_text_quality(
                sealed_cases=sealed_cases,
                calibration_cases=calibration_cases,
                metric_revision=metric_revision,
            )
        )
    except Exception:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None
    if (
        sealed_source.scope != "sealed"
        or sealed_execution.scope != "sealed"
        or sealed_bundle.scope != "sealed"
        or calibration_source.scope != "calibration"
        or calibration_execution.scope != "calibration"
        or calibration_bundle.scope != "calibration"
        or calibration_execution.raster_text_profile_revision
        != calibration.calibration_raster_text_profile_revision
        or calibration_execution.runtime_configuration_sha256
        != calibration.calibration_runtime_configuration_sha256
        or calibrated_floor != calibration.minimum_page_confidence
        or not sealed_gate_passed
        or [score.model_dump(mode="json", warnings=False) for score in sealed_scores]
        != [score.model_dump(mode="json", warnings=False) for score in evaluation.scores]
    ):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")


def _verify_v8_bl_repair_audit_binding(
    *,
    root: Path,
    pointer: RasterTextActivationPointerV8,
    profile: RasterTextOfflineProfile,
) -> None:
    """Verify the compact immutable B/L audit chain during runtime loading.

    Pointer creation performs the expensive source-PDF/Gold replay.  A
    service reload instead verifies the exact canonical receipt, evaluation,
    and 800-result candidate bundle bound by that replay.  This keeps the
    independent repair proof fail-closed without putting raster decoding in
    the ordinary runtime binding path.
    """

    try:
        from customs_ai_agent.evaluation.source_attested_bl_repair_audit import (
            BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH,
            BlRepairAuditCandidateBundleV1,
            BlRepairAuditEvaluationV1,
            BlRepairAuditReceiptV1,
            bl_repair_audit_candidate_bundle_sha256,
            bl_repair_audit_evaluation_sha256,
            bl_repair_audit_receipt_sha256,
        )

        v20_pointer = isinstance(pointer, RasterTextActivationPointerV9)
        if (
            v20_pointer
            and (
                not isinstance(profile, RasterTextOfflineProfileV20)
                or profile.engine_kind != _V20_ENGINE_KIND
                or pointer.engine_kind != _V20_ENGINE_KIND
            )
        ) or (
            not v20_pointer
            and (
                not isinstance(profile, RasterTextOfflineProfileV19)
                or profile.engine_kind != _V19B_ENGINE_KIND
                or pointer.engine_kind != _V19B_ENGINE_KIND
            )
        ):
            raise ValueError
        receipt_raw = _read_local_file(
            _sidecar_path(root, pointer.repair_audit_dataset_receipt_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        evaluation_raw = _read_local_file(
            _sidecar_path(root, pointer.repair_audit_evaluation_path),
            limit=_MAX_EVIDENCE_BYTES,
        )
        receipt = BlRepairAuditReceiptV1.model_validate_json(receipt_raw)
        evaluation = BlRepairAuditEvaluationV1.model_validate_json(evaluation_raw)
        receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
        evaluation_sha256 = hashlib.sha256(evaluation_raw).hexdigest()
        if (
            pointer.repair_audit_revision != "source-attested-independent-bl-repair-audit-v1"
            or pointer.repair_audit_dataset_receipt_path != BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH
            or receipt_sha256 != pointer.repair_audit_dataset_receipt_sha256
            or receipt_sha256 != bl_repair_audit_receipt_sha256(receipt)
            or evaluation_sha256 != pointer.repair_audit_evaluation_sha256
            or evaluation_sha256 != bl_repair_audit_evaluation_sha256(evaluation)
            or evaluation.dataset_receipt_sha256 != receipt_sha256
            or evaluation.candidate_profile_revision != profile.raster_text_profile_revision
            or evaluation.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or evaluation.extractor_revision != profile.extractor_revision
        ):
            raise ValueError

        bundle_raw = _read_local_file(
            _sidecar_path(root, evaluation.candidate_bundle_relative_path),
            limit=_MAX_REPAIR_AUDIT_BUNDLE_BYTES,
        )
        bundle = BlRepairAuditCandidateBundleV1.model_validate_json(bundle_raw)
        bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
        score_keys = {(score.fixture_kind, score.profile_id) for score in evaluation.scores}
        expected_score_keys = {
            (fixture_kind, profile_id)
            for fixture_kind in ("clean", "error")
            for profile_id in (
                "png-216",
                "jpeg-216-q75",
                "jpeg-150-q75",
                "scan-pdf-216-q75",
            )
        }
        if (
            bundle_sha256 != evaluation.candidate_bundle_sha256
            or bundle_sha256 != bl_repair_audit_candidate_bundle_sha256(bundle)
            or bundle.dataset_receipt_sha256 != receipt_sha256
            or bundle.candidate_profile_revision != profile.raster_text_profile_revision
            or bundle.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
            or bundle.extractor_revision != profile.extractor_revision
            or score_keys != expected_score_keys
            or len(evaluation.scores) != len(expected_score_keys)
            or not evaluation.gate_passed
            or any(
                not score.gate_passed
                or score.character_error_rate > 0.02
                or score.critical_value_exact_match < 0.98
                or score.voyage_exact_match < 0.98
                for score in evaluation.scores
            )
        ):
            raise ValueError
    except Exception:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid") from None


def _verify_activation_pointer_engine_binding(
    *,
    pointer: RasterTextActivationPointer,
    profile: RasterTextOfflineProfile,
) -> None:
    """Keep V19, audited V19b and voyage-replay V20 lanes disjoint."""

    if type(pointer) in (RasterTextActivationPointerV10, RasterTextActivationPointerV11):
        preview_pointer = cast(RasterTextActivationPointerV10, pointer)
        if (
            type(profile) is not RasterTextOfflineProfileV19
            or preview_pointer.engine_kind != _V19_ENGINE_KIND
            or profile.engine_kind != _V19_ENGINE_KIND
        ):
            raise RasterTextRuntimeError("raster_text_profile_mismatch")
    elif isinstance(pointer, RasterTextActivationPointerV9):
        if (
            not isinstance(profile, RasterTextOfflineProfileV20)
            or pointer.engine_kind != _V20_ENGINE_KIND
            or profile.engine_kind != _V20_ENGINE_KIND
        ):
            raise RasterTextRuntimeError("raster_text_profile_mismatch")
    elif isinstance(pointer, RasterTextActivationPointerV8):
        if (
            not isinstance(profile, RasterTextOfflineProfileV19)
            or pointer.engine_kind != _V19B_ENGINE_KIND
            or profile.engine_kind != _V19B_ENGINE_KIND
        ):
            raise RasterTextRuntimeError("raster_text_profile_mismatch")
    elif isinstance(pointer, RasterTextActivationPointerV7) and (
        not isinstance(profile, RasterTextOfflineProfileV19)
        or profile.engine_kind != _V19_ENGINE_KIND
    ):
        raise RasterTextRuntimeError("raster_text_profile_mismatch")


def _load_binding_from_pointer(
    *,
    root: Path,
    approval_path: Path,
    approval_raw: bytes,
    approval_sha256: str,
    start_service_session: bool = True,
    verification_mode: Literal["runtime", "activation"] = "runtime",
) -> SourceAttestedRasterTextPreviewBindingV1:
    if verification_mode not in ("runtime", "activation"):
        raise RasterTextRuntimeError("raster_text_runtime_validation_mode_invalid")
    pointer = parse_raster_text_activation_pointer(approval_raw)
    if hashlib.sha256(approval_raw).hexdigest() != approval_sha256:
        raise RasterTextRuntimeError("raster_text_activation_invalid")
    runtime_compact_v11 = (
        verification_mode == "runtime" and type(pointer) is RasterTextActivationPointerV11
    )
    configuration_path = _sidecar_path(root, pointer.runtime_configuration_path)
    configuration_raw = _read_local_file(configuration_path, limit=_MAX_CONFIGURATION_BYTES)
    (
        profile,
        interpreter,
        interpreter_root,
        module,
        module_root,
        sidecar_inventory,
        model_root,
        model_inventory,
    ) = _load_profile_materials(
        root=root,
        configuration_path=configuration_path,
        configuration_raw=configuration_raw,
        configuration_sha256=pointer.runtime_configuration_sha256,
        verify_complete_materials=not runtime_compact_v11,
    )
    if profile.raster_text_profile_revision != pointer.raster_text_profile_revision:
        raise RasterTextRuntimeError("raster_text_profile_mismatch")
    _verify_activation_pointer_engine_binding(pointer=pointer, profile=profile)
    evaluation_path = _sidecar_path(root, pointer.sealed_evaluation_path)
    evaluation_raw = _read_local_file(evaluation_path, limit=_MAX_CONFIGURATION_BYTES)
    if hashlib.sha256(evaluation_raw).hexdigest() != pointer.sealed_evaluation_sha256:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    evaluation = _parse_raster_text_sealed_evaluation(evaluation_raw)
    if (
        evaluation.raster_text_profile_revision != profile.raster_text_profile_revision
        or evaluation.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
        or evaluation.extractor_revision != profile.extractor_revision
    ):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    if type(pointer) in (RasterTextActivationPointerV10, RasterTextActivationPointerV11):
        preview_pointer = cast(RasterTextActivationPointerV10, pointer)
        if (
            type(evaluation) is not RasterTextSealedEvaluationV8
            or preview_pointer.current_runner_implementation_sha256
            != evaluation.current_runner_implementation_sha256
            or preview_pointer.current_runner_implementation_sha256
            != _current_raster_text_runner_implementation_sha256()
        ):
            raise RasterTextRuntimeError("raster_text_runner_implementation_invalid")
    evidence_by_path: dict[str, bytes] = {}
    evaluation_evidence: list[tuple[str, str]] = [
        (evaluation.source_gold_receipt_path, evaluation.source_gold_receipt_sha256),
        (evaluation.calibration_receipt_path, evaluation.calibration_receipt_sha256),
        (evaluation.candidate_execution_path, evaluation.candidate_execution_sha256),
        (evaluation.warm_service_smoke_receipt_path, evaluation.warm_service_smoke_receipt_sha256),
    ]
    if type(evaluation) is RasterTextSealedEvaluationV8:
        evaluation_evidence.extend(
            (
                (
                    evaluation.revalidation_smoke_receipt_path,
                    evaluation.revalidation_smoke_receipt_sha256,
                ),
                (
                    evaluation.revalidation_candidate_execution_path,
                    evaluation.revalidation_candidate_execution_sha256,
                ),
            )
        )
    if isinstance(
        evaluation,
        (
            RasterTextSealedEvaluationV4,
            RasterTextSealedEvaluationV5,
            RasterTextSealedEvaluationV6,
            RasterTextSealedEvaluationV7,
        ),
    ):
        evaluation_evidence.append(
            (evaluation.selection_receipt_path, evaluation.selection_receipt_sha256)
        )
    if isinstance(
        evaluation,
        (RasterTextSealedEvaluationV5, RasterTextSealedEvaluationV6, RasterTextSealedEvaluationV7),
    ):
        evaluation_evidence.extend(
            (
                (evaluation.projection_receipt_path, evaluation.projection_receipt_sha256),
                (
                    evaluation.stage1_decision_receipt_path,
                    evaluation.stage1_decision_receipt_sha256,
                ),
            )
        )
    for relative_path, expected_sha256 in evaluation_evidence:
        evidence = _read_local_file(_sidecar_path(root, relative_path), limit=_MAX_EVIDENCE_BYTES)
        if hashlib.sha256(evidence).hexdigest() != expected_sha256:
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        evidence_by_path[relative_path] = evidence
    calibration = _parse_raster_text_calibration_receipt(
        evidence_by_path[evaluation.calibration_receipt_path]
    )
    metric_revision = _bound_raster_text_metric_revision(
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
    )
    if (
        calibration.raster_text_profile_revision != profile.raster_text_profile_revision
        or calibration.runtime_configuration_sha256 != pointer.runtime_configuration_sha256
        or calibration.extractor_revision != profile.extractor_revision
        or calibration.minimum_page_confidence != profile.minimum_page_confidence
    ):
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    calibration_evidence: list[tuple[str, str]] = [
        (calibration.source_gold_receipt_path, calibration.source_gold_receipt_sha256),
        (calibration.candidate_execution_path, calibration.candidate_execution_sha256),
    ]
    if isinstance(
        calibration,
        (
            RasterTextCalibrationReceiptV4,
            RasterTextCalibrationReceiptV5,
            RasterTextCalibrationReceiptV6,
            RasterTextCalibrationReceiptV7,
        ),
    ):
        calibration_evidence.extend(
            (
                (calibration.selection_receipt_path, calibration.selection_receipt_sha256),
                (
                    calibration.calibration_runtime_configuration_path,
                    calibration.calibration_runtime_configuration_sha256,
                ),
            )
        )
    if isinstance(
        calibration,
        (
            RasterTextCalibrationReceiptV5,
            RasterTextCalibrationReceiptV6,
            RasterTextCalibrationReceiptV7,
        ),
    ):
        calibration_evidence.append(
            (calibration.projection_receipt_path, calibration.projection_receipt_sha256)
        )
    for relative_path, expected_sha256 in calibration_evidence:
        evidence = _read_local_file(_sidecar_path(root, relative_path), limit=_MAX_EVIDENCE_BYTES)
        if hashlib.sha256(evidence).hexdigest() != expected_sha256:
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
    if runtime_compact_v11:
        if (
            type(pointer) is not RasterTextActivationPointerV11
            or type(evaluation) is not RasterTextSealedEvaluationV8
            or type(calibration) is not RasterTextCalibrationReceiptV7
            or type(profile) is not RasterTextOfflineProfileV19
        ):
            raise RasterTextRuntimeError("raster_text_evaluation_invalid")
        _verify_runtime_attested_v11_evidence(
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            profile=profile,
            evidence_by_path=evidence_by_path,
        )
    elif (
        isinstance(pointer, RasterTextActivationPointerV4)
        and isinstance(evaluation, RasterTextSealedEvaluationV4)
        and isinstance(calibration, RasterTextCalibrationReceiptV4)
    ):
        _verify_fresh_v4_evidence_chain(
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
            profile=profile,
            warm_service_smoke_raw=evidence_by_path[evaluation.warm_service_smoke_receipt_path],
        )
    elif (
        isinstance(pointer, RasterTextActivationPointerV5)
        and isinstance(evaluation, RasterTextSealedEvaluationV5)
        and isinstance(calibration, RasterTextCalibrationReceiptV5)
    ):
        _verify_fresh_staged_evidence_chain(
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
            profile=profile,
            warm_service_smoke_raw=evidence_by_path[evaluation.warm_service_smoke_receipt_path],
        )
    elif (
        isinstance(pointer, RasterTextActivationPointerV6)
        and isinstance(evaluation, RasterTextSealedEvaluationV6)
        and isinstance(calibration, RasterTextCalibrationReceiptV6)
    ):
        _verify_fresh_staged_evidence_chain(
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
            profile=profile,
            warm_service_smoke_raw=evidence_by_path[evaluation.warm_service_smoke_receipt_path],
        )
    elif (
        type(pointer) in (RasterTextActivationPointerV10, RasterTextActivationPointerV11)
        and type(evaluation) is RasterTextSealedEvaluationV8
        and type(calibration) is RasterTextCalibrationReceiptV7
    ):
        _verify_fresh_staged_evidence_chain(
            root=root,
            pointer=cast(RasterTextActivationPointerV10, pointer),
            evaluation=evaluation,
            calibration=calibration,
            profile=profile,
            warm_service_smoke_raw=evidence_by_path[evaluation.warm_service_smoke_receipt_path],
        )
    elif (
        type(pointer)
        in (
            RasterTextActivationPointerV7,
            RasterTextActivationPointerV8,
            RasterTextActivationPointerV9,
        )
        and type(evaluation) is RasterTextSealedEvaluationV7
        and type(calibration) is RasterTextCalibrationReceiptV7
    ):
        _verify_fresh_staged_evidence_chain(
            root=root,
            pointer=cast(RasterTextActivationPointerV7, pointer),
            evaluation=cast(RasterTextSealedEvaluationV7, evaluation),
            calibration=cast(RasterTextCalibrationReceiptV7, calibration),
            profile=profile,
            warm_service_smoke_raw=evidence_by_path[evaluation.warm_service_smoke_receipt_path],
        )
    else:
        _verify_legacy_raster_text_quality_evidence(
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
            profile=profile,
            metric_revision=metric_revision,
        )
    if not runtime_compact_v11:
        historical_runner_sha256 = (
            evaluation.historical_runner_implementation_sha256
            if type(evaluation) is RasterTextSealedEvaluationV8
            else None
        )
        _verify_warm_service_smoke_evidence(
            root=root,
            raw=evidence_by_path[evaluation.warm_service_smoke_receipt_path],
            profile=profile,
            runtime_configuration_sha256=pointer.runtime_configuration_sha256,
            expected_runner_implementation_sha256=historical_runner_sha256,
        )
        if type(pointer) in (RasterTextActivationPointerV10, RasterTextActivationPointerV11):
            if type(evaluation) is not RasterTextSealedEvaluationV8:
                raise RasterTextRuntimeError("raster_text_evaluation_invalid")
            _verify_warm_service_smoke_evidence(
                root=root,
                raw=evidence_by_path[evaluation.revalidation_smoke_receipt_path],
                profile=profile,
                runtime_configuration_sha256=pointer.runtime_configuration_sha256,
                expected_runner_implementation_sha256=(
                    evaluation.current_runner_implementation_sha256
                ),
                expected_candidate_execution_path=(
                    evaluation.revalidation_candidate_execution_path
                ),
                expected_candidate_execution_sha256=(
                    evaluation.revalidation_candidate_execution_sha256
                ),
                latency_p95_limit_seconds=10.0,
            )
    if isinstance(pointer, RasterTextActivationPointerV8):
        _verify_v8_bl_repair_audit_binding(
            root=root,
            pointer=pointer,
            profile=profile,
        )
    provider = PaddleOcrTextCommandProvider(
        profile=profile,
        interpreter=interpreter,
        interpreter_root=interpreter_root,
        module=module,
        module_root=module_root,
        configuration_path=configuration_path,
        configuration_sha256=pointer.runtime_configuration_sha256,
        sidecar_inventory=sidecar_inventory,
        model_root=model_root,
        model_inventory=model_inventory,
    )
    # Pdfium/Pillow stay in the parent inventory; all Paddle imports occur in
    # the separately validated child interpreter.  The adapter accepts the
    # same structural input shape and enforces the single-page sealed scope.
    active_provider: RasterTextProvider = provider
    runtime_verifier: Callable[[], None] | None = None
    runtime_nonblocking_verifier: Callable[[], None] | None = None
    runtime_closer: Callable[[], None] | None = None
    if start_service_session:
        if runtime_compact_v11:
            session = provider.start_activation_attested_service_session()
        else:
            # Full activation/legacy loading just verified the parent import
            # and model trees.  Do not hash them a second time before warmup.
            session = provider.start_service_session(parent_materials_verified=True)
        active_provider = session
        runtime_verifier = session.verify_runtime_state
        runtime_nonblocking_verifier = session.verify_runtime_state_nonblocking
        runtime_closer = session.close
    input_adapter = cast(
        RasterTextInputAdapter,
        PdfiumRasterInputAdapter(profile.pdfium_version, profile.raster_dpi),
    )
    return SourceAttestedRasterTextPreviewBindingV1(
        root=root,
        approval_path=approval_path,
        approval_sha256=approval_sha256,
        pointer=pointer,
        profile=profile,
        engine=MediaAwareRasterTextEngine(input_adapter=input_adapter, provider=active_provider),
        runtime_verifier=runtime_verifier,
        runtime_nonblocking_verifier=runtime_nonblocking_verifier,
        runtime_closer=runtime_closer,
    )


def resolve_paddleocr3_raster_text_candidate(
    *, document_preview_root: Path, configuration_relative_path: str
) -> RasterTextCandidateEngineResolutionV1:
    """Resolve one manifest-pinned candidate engine without activation state.

    Candidate evaluation is allowed to exercise a local profile, but it may
    never read/write an approval pointer or select an arbitrary executable.
    Any malformed local material is intentionally reported only as
    unavailable to the caller's content-free runner boundary.
    """

    try:
        root = _document_preview_root(document_preview_root)
        configuration_path = _sidecar_path(root, configuration_relative_path)
        configuration_raw = _read_local_file(configuration_path, limit=_MAX_CONFIGURATION_BYTES)
        configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
        (
            profile,
            interpreter,
            interpreter_root,
            module,
            module_root,
            sidecar_inventory,
            model_root,
            model_inventory,
        ) = _load_profile_materials(
            root=root,
            configuration_path=configuration_path,
            configuration_raw=configuration_raw,
            configuration_sha256=configuration_sha256,
        )
        provider = PaddleOcrTextCommandProvider(
            profile=profile,
            interpreter=interpreter,
            interpreter_root=interpreter_root,
            module=module,
            module_root=module_root,
            configuration_path=configuration_path,
            configuration_sha256=configuration_sha256,
            sidecar_inventory=sidecar_inventory,
            model_root=model_root,
            model_inventory=model_inventory,
        )
        return RasterTextCandidateEngineResolutionV1(
            engine=MediaAwareRasterTextEngine(
                input_adapter=cast(
                    RasterTextInputAdapter,
                    PdfiumRasterInputAdapter(profile.pdfium_version, profile.raster_dpi),
                ),
                provider=provider,
            ),
            profile=profile,
            runtime_configuration_sha256=configuration_sha256,
            status="available",
        )
    except Exception:
        return RasterTextCandidateEngineResolutionV1(
            engine=None,
            profile=None,
            runtime_configuration_sha256=None,
            status="unavailable",
        )


class RasterTextPreviewRuntime(Protocol):
    def extract(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> RasterTextPreviewCandidateV1: ...

    def prewarm(self) -> "RasterTextPreviewStatusV1": ...

    def status(self) -> "RasterTextPreviewStatusV1": ...

    def close(self) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RasterTextPreviewStatusV1:
    availability: Literal["available", "unavailable"]
    raster_text_profile_revision: str | None


@dataclass(frozen=True, slots=True)
class StaticRasterTextPreviewRuntime:
    engine: LocalRasterTextPreviewEngine
    raster_text_profile_revision: str | None = None
    available: bool = False

    def extract(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> RasterTextPreviewCandidateV1:
        if (
            self.raster_text_profile_revision is not None
            and raster_input.metadata.raster_text_profile_revision
            != self.raster_text_profile_revision
        ):
            raise RasterTextPreviewUnavailable("raster text profile is superseded")
        return self.engine.extract(raster_input, structure)

    def status(self) -> RasterTextPreviewStatusV1:
        return RasterTextPreviewStatusV1(
            availability="available"
            if self.available and self.raster_text_profile_revision
            else "unavailable",
            raster_text_profile_revision=self.raster_text_profile_revision
            if self.available
            else None,
        )

    def prewarm(self) -> RasterTextPreviewStatusV1:
        """Keep the static/disabled facade lifecycle-compatible with auto mode."""

        return self.status()

    def close(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _InjectedRasterTextPreviewBinding:
    """Compatibility adapter for older in-process test engine injectors.

    Production loading always returns a source-attested binding with a warm
    service session.  Tuple injection stays private to focused tests that
    assert pointer-cache behavior without booting a native sidecar.
    """

    engine: LocalRasterTextPreviewEngine
    raster_text_profile_revision: str
    approval_sha256: str

    def verify_runtime_state(self) -> None:
        return None

    def close(self) -> None:
        return None


_RasterTextRuntimeBinding = (
    SourceAttestedRasterTextPreviewBindingV1 | _InjectedRasterTextPreviewBinding
)


@dataclass(slots=True)
class AutoActivatingSourceAttestedRasterTextPreviewRuntime:
    """Keep one warm service binding only while its pointer stays current.

    ``prewarm`` is the sole cold-start entrypoint.  OCR requests and status
    reads never launch a native child, which keeps the customer request lane
    bounded even when Paddle import/model initialization is slow.
    """

    approval_path: Path
    binding_loader: Callable[
        ..., SourceAttestedRasterTextPreviewBindingV1 | tuple[LocalRasterTextPreviewEngine, str]
    ] = _load_binding_from_pointer
    binding_material_validator: Callable[..., str] = _validate_cached_raster_text_binding_materials
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _active_pointer_sha256: str | None = field(default=None, init=False, repr=False)
    _active_binding: _RasterTextRuntimeBinding | None = field(default=None, init=False, repr=False)
    _shutdown: bool = field(default=False, init=False, repr=False)

    def _clear(self) -> None:
        binding = self._active_binding
        self._active_pointer_sha256 = None
        self._active_binding = None
        if binding is not None:
            try:
                binding.close()
            except Exception:
                # A discarded process is never reused even if best-effort
                # native cleanup reports an operating-system failure.
                pass

    def _snapshot(self) -> tuple[Path, Path, bytes, str]:
        root, expected = validate_raster_text_approval_pointer_path(self.approval_path)
        raw = _read_local_file(expected, limit=_MAX_CONFIGURATION_BYTES)
        return root, expected, raw, hashlib.sha256(raw).hexdigest()

    def _confirm_snapshot_unchanged(
        self,
        *,
        root: Path,
        approval_path: Path,
        approval_raw: bytes,
        approval_sha256: str,
    ) -> None:
        """Reject a pointer replacement that raced binding validation.

        The cached validator intentionally avoids full directory hashes, but
        it still reads configuration and direct executable/module files.  A
        pointer replacement during that work must not let this runtime return
        the prior profile from either ``status`` or ``extract``.
        """

        confirmed_root, confirmed_path, confirmed_raw, confirmed_sha256 = self._snapshot()
        if (
            confirmed_root != root
            or confirmed_path != approval_path
            or confirmed_raw != approval_raw
            or confirmed_sha256 != approval_sha256
        ):
            self._clear()
            raise RasterTextPreviewUnavailable("raster text profile changed during validation")

    @staticmethod
    def _coerce_loaded_binding(
        value: SourceAttestedRasterTextPreviewBindingV1 | tuple[LocalRasterTextPreviewEngine, str],
        *,
        approval_sha256: str,
    ) -> _RasterTextRuntimeBinding:
        if isinstance(value, SourceAttestedRasterTextPreviewBindingV1):
            return value
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[1], str)
            or not value[1]
        ):
            raise RasterTextRuntimeError("raster_text_binding_invalid")
        return _InjectedRasterTextPreviewBinding(
            engine=cast(LocalRasterTextPreviewEngine, value[0]),
            raster_text_profile_revision=value[1],
            approval_sha256=approval_sha256,
        )

    def _validate_active_binding(
        self,
        *,
        binding: _RasterTextRuntimeBinding,
        root: Path,
        approval_path: Path,
        approval_raw: bytes,
        approval_sha256: str,
        require_runtime_liveness: bool,
    ) -> None:
        validated_revision = self.binding_material_validator(
            root=root,
            approval_path=approval_path,
            approval_raw=approval_raw,
            approval_sha256=approval_sha256,
        )
        if validated_revision != binding.raster_text_profile_revision:
            raise RasterTextRuntimeError("raster_text_cached_binding_changed")
        self._confirm_snapshot_unchanged(
            root=root,
            approval_path=approval_path,
            approval_raw=approval_raw,
            approval_sha256=approval_sha256,
        )
        if require_runtime_liveness:
            binding.verify_runtime_state()
            self._confirm_snapshot_unchanged(
                root=root,
                approval_path=approval_path,
                approval_raw=approval_raw,
                approval_sha256=approval_sha256,
            )

    def _current_binding(self, *, allow_cold_start: bool) -> _RasterTextRuntimeBinding:
        with self._lock:
            if self._shutdown:
                raise RasterTextPreviewUnavailable("raster text service is shut down")
            root, path, raw, digest = self._snapshot()
            binding = self._active_binding
            if binding is not None and self._active_pointer_sha256 == digest:
                try:
                    self._validate_active_binding(
                        binding=binding,
                        root=root,
                        approval_path=path,
                        approval_raw=raw,
                        approval_sha256=digest,
                        require_runtime_liveness=True,
                    )
                except Exception:
                    self._clear()
                    raise RasterTextPreviewUnavailable(
                        "raster text preview is unavailable"
                    ) from None
                return binding
            if binding is not None:
                self._clear()
            if not allow_cold_start:
                raise RasterTextPreviewUnavailable("raster text service is warming")
            self._clear()
            loaded: _RasterTextRuntimeBinding | None = None
            try:
                loaded = self._coerce_loaded_binding(
                    self.binding_loader(
                        root=root,
                        approval_path=path,
                        approval_raw=raw,
                        approval_sha256=digest,
                    ),
                    approval_sha256=digest,
                )
                pointer = parse_raster_text_activation_pointer(raw)
                if (
                    loaded.approval_sha256 != digest
                    or loaded.raster_text_profile_revision != pointer.raster_text_profile_revision
                ):
                    raise RasterTextRuntimeError("raster_text_binding_invalid")
                # Reject a replacement racing full profile/material validation
                # and service model startup before the child becomes active.
                self._confirm_snapshot_unchanged(
                    root=root,
                    approval_path=path,
                    approval_raw=raw,
                    approval_sha256=digest,
                )
                loaded.verify_runtime_state()
                self._confirm_snapshot_unchanged(
                    root=root,
                    approval_path=path,
                    approval_raw=raw,
                    approval_sha256=digest,
                )
            except Exception:
                if loaded is not None:
                    try:
                        loaded.close()
                    except Exception:
                        pass
                raise RasterTextPreviewUnavailable("raster text preview is unavailable") from None
            self._active_pointer_sha256 = digest
            self._active_binding = loaded
            return loaded

    def _try_cached_v11_prewarm_binding(self) -> _RasterTextRuntimeBinding | None:
        """Return one unchanged V11 binding using a nonblocking child probe.

        The one-second lifecycle poll must notice a dead idle child, but it
        must not wait on the session lock owned by a healthy in-flight OCR.
        Snapshot and child checks run outside the facade lock so the public
        status endpoint does not observe a false unavailable state.  Exact
        V11 is the only pointer whose activation contract permits this compact
        cached path; all legacy pointers retain the strict validator below.
        """

        with self._lock:
            if self._shutdown:
                raise RasterTextPreviewUnavailable("raster text service is shut down")
        try:
            root, path, raw, digest = self._snapshot()
        except Exception:
            return None
        with self._lock:
            if self._shutdown:
                raise RasterTextPreviewUnavailable("raster text service is shut down")
            binding = self._active_binding
            if not (
                isinstance(binding, SourceAttestedRasterTextPreviewBindingV1)
                and type(binding.pointer) is RasterTextActivationPointerV11
                and self._active_pointer_sha256 == digest
                and binding.approval_sha256 == digest
            ):
                return None
        try:
            binding.verify_runtime_state_nonblocking()
            confirmed_root, confirmed_path, confirmed_raw, confirmed_digest = self._snapshot()
            if (
                confirmed_root != root
                or confirmed_path != path
                or confirmed_raw != raw
                or confirmed_digest != digest
            ):
                raise RasterTextPreviewUnavailable("raster text profile changed during validation")
        except Exception:
            with self._lock:
                if self._active_binding is binding:
                    self._clear()
            return None
        with self._lock:
            if self._shutdown:
                return None
            if self._active_binding is binding and self._active_pointer_sha256 == digest:
                return binding
        return None

    def _confirm_active_binding_current(self, binding: _RasterTextRuntimeBinding) -> None:
        """Suppress an OCR result if pointer/material state changed mid-call."""

        with self._lock:
            root, path, raw, digest = self._snapshot()
            if (
                self._active_binding is not binding
                or self._active_pointer_sha256 != binding.approval_sha256
                or digest != binding.approval_sha256
            ):
                if self._active_binding is binding:
                    self._clear()
                raise RasterTextPreviewUnavailable("raster text profile changed during extraction")
            try:
                self._validate_active_binding(
                    binding=binding,
                    root=root,
                    approval_path=path,
                    approval_raw=raw,
                    approval_sha256=digest,
                    require_runtime_liveness=True,
                )
            except Exception:
                self._clear()
                raise RasterTextPreviewUnavailable("raster text preview is unavailable") from None

    def _binding_for_input(self, raster_input: RasterTextInputV1) -> _RasterTextRuntimeBinding:
        binding = self._current_binding(allow_cold_start=False)
        if (
            raster_input.metadata.raster_text_profile_revision
            != binding.raster_text_profile_revision
        ):
            # The request is stale; the currently published binding is not.
            # Reject the old job without retiring the replacement child.
            raise RasterTextPreviewUnavailable("raster text profile is superseded")
        return binding

    def extract(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> RasterTextPreviewCandidateV1:
        binding: _RasterTextRuntimeBinding | None = None
        try:
            binding = self._binding_for_input(raster_input)
            candidate = binding.engine.extract(raster_input, structure)
            self._confirm_active_binding_current(binding)
            return candidate
        except Exception:
            with self._lock:
                # A stale in-flight request must never retire a replacement
                # binding that a pointer reload published meanwhile.
                if binding is not None and self._active_binding is binding:
                    self._clear()
            raise RasterTextPreviewUnavailable("raster text preview is unavailable") from None

    def prewarm(self) -> RasterTextPreviewStatusV1:
        """Synchronously start and verify the current pointer's warm child.

        The application should invoke this from a background lifecycle loop;
        it performs the only full material/model startup path and never runs
        in a customer OCR request or a status read.
        """

        try:
            binding = self._try_cached_v11_prewarm_binding()
            if binding is None:
                binding = self._current_binding(allow_cold_start=True)
            return RasterTextPreviewStatusV1("available", binding.raster_text_profile_revision)
        except Exception:
            with self._lock:
                self._clear()
            return RasterTextPreviewStatusV1("unavailable", None)

    def status(self) -> RasterTextPreviewStatusV1:
        # A status endpoint must neither wait behind a cold native start nor
        # launch a child.  If prewarming or teardown owns the facade lock,
        # report unavailable and let the next cheap poll observe readiness.
        if not self._lock.acquire(blocking=False):
            # Status is advisory and must not turn a healthy in-flight job
            # into a Backend terminal failure merely because a compact
            # validator briefly owns the facade lock.  These references are
            # published together under the lock and the binding is immutable.
            # A concurrent pointer/material transition may expose the prior
            # revision for one poll, but extraction and confirmation enforce
            # exact revision again; Backend treats unavailable as transient.
            binding = self._active_binding
            digest = self._active_pointer_sha256
            if (
                not self._shutdown
                and isinstance(binding, SourceAttestedRasterTextPreviewBindingV1)
                and type(binding.pointer) is RasterTextActivationPointerV11
                and digest is not None
                and binding.approval_sha256 == digest
            ):
                return RasterTextPreviewStatusV1("available", binding.raster_text_profile_revision)
            return RasterTextPreviewStatusV1("unavailable", None)
        try:
            if self._shutdown:
                return RasterTextPreviewStatusV1("unavailable", None)
            binding = self._active_binding
            if binding is None or self._active_pointer_sha256 is None:
                return RasterTextPreviewStatusV1("unavailable", None)
            root, path, raw, digest = self._snapshot()
            if digest != binding.approval_sha256:
                self._clear()
                return RasterTextPreviewStatusV1("unavailable", None)
            self._validate_active_binding(
                binding=binding,
                root=root,
                approval_path=path,
                approval_raw=raw,
                approval_sha256=digest,
                # Do not wait on the serial OCR lane from a status poll.  The
                # request boundary itself verifies child liveness before use.
                require_runtime_liveness=False,
            )
            return RasterTextPreviewStatusV1("available", binding.raster_text_profile_revision)
        except Exception:
            self._clear()
            return RasterTextPreviewStatusV1("unavailable", None)
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            self._clear()

    def shutdown(self) -> None:
        """Permanently stop this facade; unlike timeout reset, never rewarm."""

        with self._lock:
            self._shutdown = True
            self._clear()


def resolve_raster_text_preview_engine(
    *, approval_path: Path | None, approval_sha256: str | None
) -> RasterTextPreviewEngineResolution:
    """Resolve only a hash-pinned local pointer; no fallback exists."""

    disabled = RasterTextPreviewEngineResolution(DisabledRasterTextPreviewEngine(), "disabled")
    if approval_path is None and approval_sha256 is None:
        return disabled
    if (
        approval_path is None
        or approval_sha256 is None
        or _SHA256.fullmatch(approval_sha256) is None
    ):
        return RasterTextPreviewEngineResolution(
            DisabledRasterTextPreviewEngine(),
            "unavailable",
            reason_code="raster_text_approval_invalid",
        )
    try:
        root, expected = validate_raster_text_approval_pointer_path(approval_path)
        raw = _read_local_file(expected, limit=_MAX_CONFIGURATION_BYTES)
        if hashlib.sha256(raw).hexdigest() != approval_sha256:
            raise RasterTextRuntimeError("raster_text_approval_invalid")
        binding = _load_binding_from_pointer(
            root=root,
            approval_path=expected,
            approval_raw=raw,
            approval_sha256=approval_sha256,
            start_service_session=False,
        )
        return RasterTextPreviewEngineResolution(
            binding.engine,
            "available",
            binding.raster_text_profile_revision,
        )
    except (RasterTextRuntimeError, RasterTextPreviewUnavailable):
        return RasterTextPreviewEngineResolution(
            DisabledRasterTextPreviewEngine(),
            "unavailable",
            reason_code="raster_text_approval_invalid",
        )


def _atomic_replace_raster_text_pointer(*, destination: Path, payload: bytes) -> None:
    """Write an already-validated pointer with fsync then atomic replace."""

    temporary: Path | None = None
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
        raise RasterTextRuntimeError("raster_text_activation_write_failed") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def write_source_attested_synthetic_raster_text_pointer(
    *,
    document_preview_root: Path,
    runtime_configuration_relative_path: str,
    sealed_evaluation_relative_path: str,
    repair_audit_dataset_receipt_relative_path: str | None = None,
    repair_audit_evaluation_relative_path: str | None = None,
) -> tuple[str, RasterTextActivationPointer]:
    """Activate only a complete passing source-only text evidence chain.

    The caller supplies local artifact locations, never hashes, scores,
    provider JSON, a model selector, or a human-approval flag.  Hashes and
    the profile revision are derived from the exact files; an old pointer is
    left unchanged when any verification step fails.
    """

    if (repair_audit_dataset_receipt_relative_path is None) != (
        repair_audit_evaluation_relative_path is None
    ):
        raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
    root = _document_preview_root(document_preview_root)
    configuration_path = _sidecar_path(root, runtime_configuration_relative_path)
    evaluation_path = _sidecar_path(root, sealed_evaluation_relative_path)
    configuration_raw = _read_local_file(configuration_path, limit=_MAX_CONFIGURATION_BYTES)
    evaluation_raw = _read_local_file(evaluation_path, limit=_MAX_CONFIGURATION_BYTES)
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    evaluation_sha256 = hashlib.sha256(evaluation_raw).hexdigest()
    try:
        profile = parse_raster_text_offline_profile(configuration_raw)
        evaluation = _parse_raster_text_sealed_evaluation(evaluation_raw)
    except (RasterTextRuntimeError, ValueError):
        raise RasterTextRuntimeError("raster_text_activation_arguments_invalid") from None
    if (
        profile.raster_text_profile_revision != evaluation.raster_text_profile_revision
        or profile.extractor_revision != evaluation.extractor_revision
        or configuration_sha256 != evaluation.runtime_configuration_sha256
    ):
        raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
    repair_audit_requested = repair_audit_dataset_receipt_relative_path is not None
    if type(evaluation) is RasterTextSealedEvaluationV8:
        if (
            type(profile) is not RasterTextOfflineProfileV19
            or profile.engine_kind != _V19_ENGINE_KIND
            or repair_audit_requested
        ):
            raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
    elif isinstance(evaluation, RasterTextSealedEvaluationV7):
        if not isinstance(profile, RasterTextOfflineProfileV19):
            raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
        if isinstance(profile, RasterTextOfflineProfileV20):
            if profile.engine_kind != _V20_ENGINE_KIND or not repair_audit_requested:
                raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
        elif profile.engine_kind == _V19B_ENGINE_KIND:
            if not repair_audit_requested:
                raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
        elif profile.engine_kind == _V19_ENGINE_KIND:
            if repair_audit_requested:
                raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
        else:
            raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
    elif repair_audit_requested:
        raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
    repair_audit_pointer_material: dict[str, object] | None = None
    if repair_audit_dataset_receipt_relative_path is not None:
        assert repair_audit_evaluation_relative_path is not None
        if (
            not isinstance(evaluation, RasterTextSealedEvaluationV7)
            or not isinstance(profile, RasterTextOfflineProfileV19)
            or profile.engine_kind not in {_V19B_ENGINE_KIND, _V20_ENGINE_KIND}
        ):
            raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
        try:
            from customs_ai_agent.evaluation.source_attested_bl_repair_audit import (
                BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH,
                BlRepairAuditReceiptV1,
                bl_repair_audit_evaluation_sha256,
                bl_repair_audit_receipt_sha256,
                verify_bl_repair_audit_evaluation,
            )

            if repair_audit_dataset_receipt_relative_path != BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH:
                raise ValueError
            repair_receipt_raw = _read_local_file(
                _sidecar_path(root, repair_audit_dataset_receipt_relative_path),
                limit=_MAX_EVIDENCE_BYTES,
            )
            repair_evaluation_raw = _read_local_file(
                _sidecar_path(root, repair_audit_evaluation_relative_path),
                limit=_MAX_EVIDENCE_BYTES,
            )
            repair_receipt_sha256 = hashlib.sha256(repair_receipt_raw).hexdigest()
            repair_evaluation_sha256 = hashlib.sha256(repair_evaluation_raw).hexdigest()
            repair_receipt = BlRepairAuditReceiptV1.model_validate_json(repair_receipt_raw)
            repair_evaluation = verify_bl_repair_audit_evaluation(
                document_preview_root=root,
                evaluation_relative_path=repair_audit_evaluation_relative_path,
                evaluation_sha256=repair_evaluation_sha256,
                expected_candidate_profile_revision=profile.raster_text_profile_revision,
                expected_runtime_configuration_sha256=configuration_sha256,
                require_gate_passed=True,
            )
            if (
                repair_receipt_sha256 != bl_repair_audit_receipt_sha256(repair_receipt)
                or repair_evaluation_sha256 != bl_repair_audit_evaluation_sha256(repair_evaluation)
                or repair_evaluation.dataset_receipt_sha256 != repair_receipt_sha256
                or repair_evaluation.extractor_revision != profile.extractor_revision
                or not repair_evaluation.gate_passed
                or any(not score.gate_passed for score in repair_evaluation.scores)
            ):
                raise ValueError
        except Exception:
            raise RasterTextRuntimeError("raster_text_activation_arguments_invalid") from None
        repair_audit_pointer_material = {
            "repair_audit_revision": "source-attested-independent-bl-repair-audit-v1",
            "repair_audit_dataset_receipt_path": (repair_audit_dataset_receipt_relative_path),
            "repair_audit_dataset_receipt_sha256": repair_receipt_sha256,
            "repair_audit_evaluation_path": repair_audit_evaluation_relative_path,
            "repair_audit_evaluation_sha256": repair_evaluation_sha256,
        }
    pointer_payload: dict[str, object] = {
        "activation_kind": "source_attested_synthetic_raster_text",
        "status": "activated",
        "dataset_id": "synthetic-raster-text-v1",
        "raster_text_profile_revision": profile.raster_text_profile_revision,
        "runtime_configuration_path": runtime_configuration_relative_path,
        "runtime_configuration_sha256": configuration_sha256,
        "sealed_evaluation_path": sealed_evaluation_relative_path,
        "sealed_evaluation_sha256": evaluation_sha256,
    }
    if type(evaluation) is RasterTextSealedEvaluationV8:
        if (
            type(profile) is not RasterTextOfflineProfileV19
            or profile.engine_kind != _V19_ENGINE_KIND
            or repair_audit_pointer_material is not None
        ):
            raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
        pointer_payload.update(
            schema_version="11.0",
            runtime_validation_revision="activation-full-runtime-compact-v1",
            engine_kind=profile.engine_kind,
            metric_revision=evaluation.metric_revision,
            evidence_revision=evaluation.evidence_revision,
            latency_gate_revision=evaluation.latency_gate_revision,
            quality_gate_revision=evaluation.quality_gate_revision,
            evidence_revalidation_revision=(evaluation.evidence_revalidation_revision),
            historical_runner_implementation_sha256=(
                evaluation.historical_runner_implementation_sha256
            ),
            current_runner_implementation_sha256=(evaluation.current_runner_implementation_sha256),
            revalidation_smoke_receipt_path=(evaluation.revalidation_smoke_receipt_path),
            revalidation_smoke_receipt_sha256=(evaluation.revalidation_smoke_receipt_sha256),
            revalidation_candidate_execution_path=(
                evaluation.revalidation_candidate_execution_path
            ),
            revalidation_candidate_execution_sha256=(
                evaluation.revalidation_candidate_execution_sha256
            ),
            source_recognizer_release_sha256=profile.source_recognizer_release_sha256,
            source_recognizer_receipt_sha256=profile.source_recognizer_receipt_sha256,
            source_recognizer_training_intent_sha256=(
                profile.source_recognizer_training_intent_sha256
            ),
            source_recognizer_selection_receipt_path=(
                profile.source_recognizer_selection_receipt_relative_path
            ),
            source_recognizer_selection_receipt_sha256=(
                profile.source_recognizer_selection_receipt_sha256
            ),
            source_recognizer_parent_selection_receipt_sha256=(
                profile.source_recognizer_parent_selection_receipt_sha256
            ),
        )
    elif isinstance(evaluation, RasterTextSealedEvaluationV7):
        if not isinstance(profile, RasterTextOfflineProfileV19):
            raise RasterTextRuntimeError("raster_text_activation_arguments_invalid")
        pointer_schema = (
            "9.0"
            if isinstance(profile, RasterTextOfflineProfileV20)
            else ("8.0" if repair_audit_pointer_material is not None else "7.0")
        )
        pointer_payload.update(
            schema_version=pointer_schema,
            metric_revision=evaluation.metric_revision,
            evidence_revision=evaluation.evidence_revision,
            latency_gate_revision=evaluation.latency_gate_revision,
            source_recognizer_release_sha256=profile.source_recognizer_release_sha256,
            source_recognizer_receipt_sha256=profile.source_recognizer_receipt_sha256,
            source_recognizer_training_intent_sha256=(
                profile.source_recognizer_training_intent_sha256
            ),
            source_recognizer_selection_receipt_path=(
                profile.source_recognizer_selection_receipt_relative_path
            ),
            source_recognizer_selection_receipt_sha256=(
                profile.source_recognizer_selection_receipt_sha256
            ),
            source_recognizer_parent_selection_receipt_sha256=(
                profile.source_recognizer_parent_selection_receipt_sha256
            ),
        )
        if repair_audit_pointer_material is not None:
            pointer_payload["engine_kind"] = profile.engine_kind
            pointer_payload.update(repair_audit_pointer_material)
        if isinstance(profile, RasterTextOfflineProfileV20):
            pointer_payload.update(
                source_recognizer_parent_release_sha256=(
                    profile.source_recognizer_parent_release_sha256
                ),
                source_recognizer_parent_receipt_sha256=(
                    profile.source_recognizer_parent_receipt_sha256
                ),
                source_recognizer_voyage_recipe_set_sha256=(
                    profile.source_recognizer_voyage_recipe_set_sha256
                ),
                source_recognizer_train_manifest_sha256=(
                    profile.source_recognizer_train_manifest_sha256
                ),
                source_recognizer_development_manifest_sha256=(
                    profile.source_recognizer_development_manifest_sha256
                ),
            )
    elif isinstance(evaluation, RasterTextSealedEvaluationV6):
        pointer_payload.update(
            schema_version="6.0",
            metric_revision=evaluation.metric_revision,
            evidence_revision=evaluation.evidence_revision,
            latency_gate_revision=evaluation.latency_gate_revision,
        )
    elif isinstance(evaluation, RasterTextSealedEvaluationV5):
        pointer_payload.update(
            schema_version="5.0",
            metric_revision=evaluation.metric_revision,
            evidence_revision=evaluation.evidence_revision,
            latency_gate_revision=evaluation.latency_gate_revision,
        )
    elif isinstance(evaluation, RasterTextSealedEvaluationV4):
        pointer_payload.update(
            schema_version="4.0",
            metric_revision=evaluation.metric_revision,
            evidence_revision=evaluation.evidence_revision,
        )
    elif isinstance(evaluation, RasterTextSealedEvaluationV3):
        pointer_payload.update(
            schema_version="3.0",
            metric_revision=evaluation.metric_revision,
        )
    elif isinstance(evaluation, RasterTextSealedEvaluationV2):
        pointer_payload.update(
            schema_version="2.0",
            metric_revision=evaluation.metric_revision,
        )
    else:
        pointer_payload["schema_version"] = "1.0"
    pointer_raw = _canonical_json_bytes(pointer_payload)
    pointer = parse_raster_text_activation_pointer(pointer_raw)
    destination = root / "approvals" / "raster-text-active.json"
    # Re-load the exact bytes that will be written, including both package
    # inventories, full model tree, sidecar module/interpreter, and source
    # Gold/calibration/candidate evidence bindings.  No pointer is touched
    # before this succeeds.
    _load_binding_from_pointer(
        root=root,
        approval_path=destination,
        approval_raw=pointer_raw,
        approval_sha256=hashlib.sha256(pointer_raw).hexdigest(),
        start_service_session=False,
        verification_mode="activation",
    )
    _atomic_replace_raster_text_pointer(destination=destination, payload=pointer_raw)
    return hashlib.sha256(pointer_raw).hexdigest(), pointer


__all__ = [
    "AutoActivatingSourceAttestedRasterTextPreviewRuntime",
    "PaddleOcrTextCommandProvider",
    "PaddleOcrTextServiceSession",
    "RasterTextActivationPointer",
    "RasterTextActivationPointerV1",
    "RasterTextActivationPointerV2",
    "RasterTextActivationPointerV3",
    "RasterTextActivationPointerV4",
    "RasterTextActivationPointerV5",
    "RasterTextActivationPointerV6",
    "RasterTextActivationPointerV7",
    "RasterTextActivationPointerV8",
    "RasterTextActivationPointerV9",
    "RasterTextActivationPointerV10",
    "RasterTextActivationPointerV11",
    "RasterTextOfflineProfile",
    "RasterTextOfflineProfileV1",
    "RasterTextOfflineProfileV2",
    "RasterTextOfflineProfileV3",
    "RasterTextOfflineProfileV4",
    "RasterTextOfflineProfileV5",
    "RasterTextOfflineProfileV6",
    "RasterTextOfflineProfileV7",
    "RasterTextOfflineProfileV8",
    "RasterTextOfflineProfileV9",
    "RasterTextOfflineProfileV10",
    "RasterTextOfflineProfileV11",
    "RasterTextOfflineProfileV12",
    "RasterTextOfflineProfileV13",
    "RasterTextOfflineProfileV14",
    "RasterTextOfflineProfileV15",
    "RasterTextOfflineProfileV16",
    "RasterTextOfflineProfileV17",
    "RasterTextOfflineProfileV18",
    "RasterTextOfflineProfileV19",
    "RasterTextOfflineProfileV20",
    "RasterTextCalibrationReceiptV1",
    "RasterTextCalibrationReceiptV2",
    "RasterTextCalibrationReceiptV3",
    "RasterTextCalibrationReceiptV4",
    "RasterTextCalibrationReceiptV5",
    "RasterTextCalibrationReceiptV6",
    "RasterTextCalibrationReceiptV7",
    "RasterTextWarmServiceSmokeReceiptV1",
    "RasterTextCandidateEngineResolutionV1",
    "RasterTextPreviewEngineResolution",
    "RasterTextPreviewRuntime",
    "RasterTextPreviewStatusV1",
    "RasterTextRuntimeError",
    "RasterTextSealedEvaluationV1",
    "RasterTextSealedEvaluationV2",
    "RasterTextSealedEvaluationV3",
    "RasterTextSealedEvaluationV4",
    "RasterTextSealedEvaluationV5",
    "RasterTextSealedEvaluationV6",
    "RasterTextSealedEvaluationV7",
    "RasterTextSealedEvaluationV8",
    "RasterTextSealedProfileScoreV1",
    "RasterTextSealedProfileScoreV2",
    "RasterTextSealedProfileScoreV3",
    "RasterTextSealedProfileScoreV4",
    "RasterTextSealedProfileScoreV5",
    "RasterTextSealedProfileScoreV6",
    "RasterTextSealedProfileScoreV7",
    "RasterTextSealedProfileScoreV8",
    "RasterTextSidecarResultV1",
    "SourceAttestedRasterTextPreviewBindingV1",
    "StaticRasterTextPreviewRuntime",
    "parse_raster_text_activation_pointer",
    "parse_raster_text_offline_profile",
    "raster_text_service_startup_timeout_seconds",
    "resolve_raster_text_preview_engine",
    "resolve_paddleocr3_raster_text_candidate",
    "raster_text_profile_fingerprint_sha256",
    "raster_text_exact_result_set_sha256",
    "validate_raster_text_approval_pointer_path",
    "write_source_attested_synthetic_raster_text_pointer",
]
