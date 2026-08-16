"""Approved local PDF-preview runtime and immutable local sidecars.

This module deliberately keeps the OCR activation boundary separate from the
public document-extraction wire contract.  A configured service can use an
OCR-capable adapter only after a local approval pointer binds all of these
immutable, Git-excluded inputs:

* the exact approval pointer bytes;
* a pinned runtime configuration and model inventory;
* a sealed, passing evaluation result; and
* every declared local model artifact.

The service never fetches a package, a model, a document, or an approval
artifact from the network.  Missing or mismatched local state becomes the
existing content-free ``unavailable`` preview result instead of a fallback.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import re
import socket
import stat
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    cast,
)
from urllib.parse import urlparse

from pydantic import ValidationError

from customs_ai_agent.schemas.document_extraction import (
    BoundingBoxV1,
    DocumentExtractionWarningV1,
    DocumentPreviewPageV1,
    DocumentSpanV1,
    PreviewCandidateV1,
    PreviewInputV1,
)
from customs_ai_agent.service.document_extraction import (
    DisabledDocumentPreviewEngine,
    DocumentPreviewExecutionError,
    DocumentPreviewInputError,
    DocumentPreviewUnavailable,
    LayeredLocalPreviewEngine,
    LocalDocumentPreviewEngine,
)

if TYPE_CHECKING:
    from customs_ai_agent.evaluation.native_pdf_vector_grid import NativePdfVectorGridExtraction
    from customs_ai_agent.evaluation.supervised_template_provenance import (
        R1SupervisedTemplateProvenanceBinding,
    )


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ENGINE_KINDS = frozenset({"pdf_text_layer", "paddleocr", "docling"})
_MAX_POINTER_BYTES = 64 * 1024
_MAX_SIDECAR_BYTES = 1 * 1024 * 1024
_MAX_EVALUATION_BYTES = 1 * 1024 * 1024
_MAX_LAYOUT_TRUTH_RECEIPT_BYTES = 8 * 1024 * 1024
_PREVIEW_ROOT_PARTS = ("_local", "data", "document-preview")
_APPROVAL_FILENAME = "pdf-preview-active.json"
_SOURCE_LAYOUT_TRUTH_DIRECTORY = "source-layout-truths"
_SOURCE_DATASET_RECEIPT_DIRECTORY = "source-attested-dataset-receipts"
_SOURCE_PACKET_DIRECTORY = "source-attested-packets"
_SOURCE_CANDIDATE_BUNDLE_DIRECTORY = "source-attested-candidate-bundles"
_SOURCE_CANDIDATE_EXECUTION_DIRECTORY = "source-attested-candidate-executions"
_SOURCE_EVALUATION_RESULT_DIRECTORY = "source-attested-evaluation-results"
_SOURCE_DEVELOPMENT_SELECTION_DIRECTORY = "source-attested-development-selections"
_SOURCE_SEALED_RUN_DIRECTORY = "source-attested-sealed-runs"
_SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION: Final = (
    "synthetic-pdf-supervised-template-normalizer-v11"
)
# The source-attested automatic pointer has exactly one production corpus and
# two canonical packet receipts.  Keep the profile pin local to this
# import-cycle-sensitive module: the evaluation package imports this runtime
# through the candidate runner, so a top-level contract import would cycle.
_SOURCE_ATTESTED_SUPERVISED_DATASET_ID: Final = "synthetic-customs-pdf-2000"
_SOURCE_ATTESTED_SUPERVISED_DATASET_REVISION: Final = "20260811-r1"
_SOURCE_ATTESTED_SUPERVISED_DEVELOPMENT_PACKET_SHA256: Final = (
    "5b8a804b022e97df0e58a60c4fadfc9d3637c900f368273b8c1b6ff347c3f7e2"
)
_SOURCE_ATTESTED_SUPERVISED_SEALED_PACKET_SHA256: Final = (
    "174d7fad0040e89ee93d367f31694183049ea649a28515cde7a59458261aa271"
)
_SEALED_RESERVATION_DIRECTORY = "candidate-run-reservations"
_SOURCE_LAYOUT_RECORD_COUNTS: Mapping[str, int] = {"development": 60, "sealed": 20}
_MAX_SOURCE_DATASET_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_PACKET_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_CANDIDATE_BUNDLE_BYTES = 32 * 1024 * 1024
_FIELD_EXACT_MATCH_MIN = 0.99
_OCR_CER_MAX = 0.01
_TABLE_CELL_F1_MIN = 0.98
# The text-layer adapter has two materially different extraction algorithms.
# These names are part of the approved-engine evidence, not cosmetic labels:
# a pointer approved against the historic aggregate ``extract_text`` result
# must never silently begin serving visitor-callback fragments after a code
# update.  Keep the legacy revision loadable for existing human approvals;
# candidate-only visitor geometry is available only from the explicit v2.
_PDF_TEXT_LAYER_LEGACY_EXTRACTOR_REVISION: Final = "pypdf-5.3.1-text-layer-v1"
_PDF_TEXT_LAYER_VISITOR_EXTRACTOR_REVISION: Final = "pypdf-5.3.1-text-layer-visitor-v2"
_PDF_TEXT_LAYER_NATIVE_VECTOR_GRID_EXTRACTOR_REVISION: Final = "pypdf-5.3.1-native-vector-grid-v1"
_PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION: Final = (
    "pypdf-5.3.1-supervised-template-r1-v1"
)
# These literals are repeated at this import-cycle-sensitive service boundary.
# The provenance loader below imports the public evaluation contract lazily and
# verifies the model's exact identity before any PDF bytes are supplied.
SUPERVISED_TEMPLATE_EXTRACTOR_REVISION: Final = (
    _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION
)
SUPERVISED_TEMPLATE_MODEL_KIND: Final = "native_pdf_semantic_model"
_PDF_TEXT_LAYER_EXTRACTOR_REVISIONS: Final[frozenset[str]] = frozenset(
    {
        _PDF_TEXT_LAYER_LEGACY_EXTRACTOR_REVISION,
        _PDF_TEXT_LAYER_VISITOR_EXTRACTOR_REVISION,
        _PDF_TEXT_LAYER_NATIVE_VECTOR_GRID_EXTRACTOR_REVISION,
        _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION,
    }
)
_NETWORK_GUARD_LOCK = threading.Lock()
_BYTECODE_GUARD_LOCK = threading.Lock()
_NETWORK_GUARD_STATE = threading.local()
_NETWORK_GUARD_INSTALLED = False
_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_SOCKET_CONNECT_EX = socket.socket.connect_ex

_DOCLING_NATIVE_PACKAGE_NAMES: Final = ("docling_parse", "deepsearch_glm")
_DOCLING_PDF_BACKEND: Final = "pypdfium2"
_DOCLING_TABLE_STRUCTURE_MODES: Final[frozenset[str]] = frozenset({"fast", "accurate"})
_PADDLEOCR2_RUNTIME_API: Final = "paddleocr2-2.9"
_PADDLEOCR3_RUNTIME_API: Final = "paddleocr3-paddlex-3.2"
# This revision names the complete, fixed P3 extraction contract: the
# PaddleOCR/PaddleX API family, the two PP-OCRv5 mobile models, and PDFium's
# 3x (216 dpi) raster input.  It is intentionally not interchangeable with
# the legacy PaddleOCR 2 sidecars, which retain their existing revisions.
_PADDLEOCR3_EXTRACTOR_REVISION: Final = "paddleocr-3.2.0-ppocrv5-mobile-en-raster216-v1"

ApprovedPdfEngineKind = Literal["pdf_text_layer", "paddleocr", "docling"]
PaddleOcrRuntimeApi = Literal["paddleocr2-2.9", "paddleocr3-paddlex-3.2"]


class DocumentPreviewRuntimeError(RuntimeError):
    """A deliberately content-free local activation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _network_connect_guard(socket_instance: socket.socket, address: Any) -> None:
    if getattr(_NETWORK_GUARD_STATE, "local_files_only", False):
        raise OSError("local preview runtime forbids network access")
    _ORIGINAL_SOCKET_CONNECT(socket_instance, address)


def _network_connect_ex_guard(socket_instance: socket.socket, address: Any) -> int:
    if getattr(_NETWORK_GUARD_STATE, "local_files_only", False):
        return getattr(socket, "EAI_FAIL", 1)
    return _ORIGINAL_SOCKET_CONNECT_EX(socket_instance, address)


def _install_network_guard_once() -> None:
    global _NETWORK_GUARD_INSTALLED
    with _NETWORK_GUARD_LOCK:
        if _NETWORK_GUARD_INSTALLED:
            return
        # ``setattr`` avoids making the global patch appear as an arbitrary
        # protocol implementation to static type checkers.  The wrapper is
        # thread-local: normal Evidence HTTP traffic in another service thread
        # is unaffected while an OCR adapter is executing.
        setattr(socket.socket, "connect", _network_connect_guard)
        setattr(socket.socket, "connect_ex", _network_connect_ex_guard)
        _NETWORK_GUARD_INSTALLED = True


@contextmanager
def local_files_only() -> Iterator[None]:
    """Forbid socket-based network access in this adapter's execution thread."""

    _install_network_guard_once()
    previous = getattr(_NETWORK_GUARD_STATE, "local_files_only", False)
    _NETWORK_GUARD_STATE.local_files_only = True
    try:
        yield
    finally:
        _NETWORK_GUARD_STATE.local_files_only = previous


def _require_docling_native_ascii_package_paths() -> None:
    """Reject a Windows path that the pinned native Docling stack cannot read.

    ``docling-parse`` and ``deepsearch-glm`` in the pinned Docling 2.15.1
    profile resolve their package-owned native resources with a narrow path
    encoding API on Windows.  A non-ASCII site-packages path therefore fails
    only after a PDF is supplied and can produce native diagnostics.  Detect
    that known unsupported runtime before any source PDF is read instead of
    silently trying a fallback or mapping a drive from product code.
    """

    if sys.platform != "win32":
        return
    for package_name in _DOCLING_NATIVE_PACKAGE_NAMES:
        specification = importlib.util.find_spec(package_name)
        origin = None if specification is None else specification.origin
        if (
            not isinstance(origin, str)
            or not origin
            or origin in {"built-in", "frozen"}
            or any(ord(character) > 0x7F for character in origin)
        ):
            raise DocumentPreviewUnavailable("document_preview_docling_native_runtime_unavailable")


def _canonical_json_bytes(value: object, *, trailing_newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_newline else b"")


def _canonical_local_artifact_file_bytes(value: object) -> bytes:
    """Return the source evaluator's immutable local artifact representation.

    Runtime sidecars use compact JSON.  The source-attested evaluator instead
    writes its receipts, packets, candidate bundles, and results as one exact
    pretty JSON representation.  Keep that distinction explicit: accepting a
    compact reserialization here would make a hash-named source artifact less
    auditable and would diverge from the evaluator's create-only writer.
    """

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_object(raw: bytes, *, code: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DocumentPreviewRuntimeError(code) from error
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError(code)
    return cast(dict[str, object], value)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, code: str) -> None:
    if set(value) != expected:
        raise DocumentPreviewRuntimeError(code)


def _identifier(value: object, *, code: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise DocumentPreviewRuntimeError(code)
    return value


def _sha256(value: object, *, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DocumentPreviewRuntimeError(code)
    return value


def _engine_kind(value: object, *, code: str) -> ApprovedPdfEngineKind:
    if not isinstance(value, str) or value not in _ENGINE_KINDS:
        raise DocumentPreviewRuntimeError(code)
    return cast(ApprovedPdfEngineKind, value)


def _finite_number(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DocumentPreviewRuntimeError(code)
    number = float(value)
    if not math.isfinite(number):
        raise DocumentPreviewRuntimeError(code)
    return number


def _validate_https_source(value: object, *, code: str) -> str:
    if not isinstance(value, str) or len(value) > 512:
        raise DocumentPreviewRuntimeError(code)
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise DocumentPreviewRuntimeError(code)
    return value


def _relative_model_path(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise DocumentPreviewRuntimeError(code)
    candidate = Path(value)
    if candidate.is_absolute() or "\\" in value or "\x00" in value:
        raise DocumentPreviewRuntimeError(code)
    parts = candidate.parts
    if not parts or parts[0] != "models" or any(part in {"", ".", ".."} for part in parts):
        raise DocumentPreviewRuntimeError(code)
    return candidate.as_posix()


def _is_link_or_reparse_point(metadata: os.stat_result) -> bool:
    """Return whether an ``lstat`` result can redirect a local model path.

    ``Path.is_symlink`` does not identify Windows directory junctions.  Those
    are reparse points too, so treating them as ordinary directories would let
    a model inventory point outside the approved immutable tree.
    """

    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse_point and attributes & reparse_point)


def _assert_no_symlink(path: Path, *, code: str) -> None:
    """Reject a link or reparse point in every existing path component."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise DocumentPreviewRuntimeError(code) from error
        if _is_link_or_reparse_point(metadata):
            raise DocumentPreviewRuntimeError(code)


def _assert_document_preview_root(root: Path, *, must_exist: bool) -> Path:
    if (
        not root.is_absolute()
        or tuple(part.casefold() for part in root.parts[-3:]) != _PREVIEW_ROOT_PARTS
    ):
        raise DocumentPreviewRuntimeError("document_preview_local_root_invalid")
    _assert_no_symlink(root, code="document_preview_local_root_invalid")
    try:
        if must_exist and (not root.is_dir() or root.is_symlink()):
            raise DocumentPreviewRuntimeError("document_preview_local_root_invalid")
        resolved = root.resolve(strict=must_exist)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_local_root_invalid") from error
    if tuple(part.casefold() for part in resolved.parts[-3:]) != _PREVIEW_ROOT_PARTS:
        raise DocumentPreviewRuntimeError("document_preview_local_root_invalid")
    return resolved


def validate_document_preview_root(root: Path, *, create: bool = False) -> Path:
    """Return the only local root allowed to hold runtime OCR artifacts."""

    checked = _assert_document_preview_root(root, must_exist=False)
    if create:
        try:
            checked.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise DocumentPreviewRuntimeError("document_preview_local_root_invalid") from error
    return _assert_document_preview_root(checked, must_exist=True)


def validate_approval_pointer_path(path: Path) -> tuple[Path, Path]:
    """Validate the exact, direct active-pointer location and return its root."""

    if not path.is_absolute() or path.name != _APPROVAL_FILENAME:
        raise DocumentPreviewRuntimeError("document_preview_approval_path_invalid")
    parent = path.parent
    if parent.name != "approvals":
        raise DocumentPreviewRuntimeError("document_preview_approval_path_invalid")
    root = validate_document_preview_root(parent.parent, create=False)
    _assert_no_symlink(path, code="document_preview_approval_path_invalid")
    try:
        if not path.is_file() or path.is_symlink():
            raise DocumentPreviewRuntimeError("document_preview_approval_path_invalid")
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_approval_path_invalid") from error
    if resolved.parent != root / "approvals":
        raise DocumentPreviewRuntimeError("document_preview_approval_path_invalid")
    return root, resolved


def _sidecar_path(root: Path, directory: str, digest: str) -> Path:
    destination = root / directory / f"{digest}.json"
    _assert_no_symlink(destination, code="document_preview_sidecar_path_invalid")
    try:
        resolved_parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_sidecar_path_invalid") from error
    if resolved_parent != root / directory:
        raise DocumentPreviewRuntimeError("document_preview_sidecar_path_invalid")
    return destination


def _read_regular_file(path: Path, *, limit: int, code: str) -> bytes:
    _assert_no_symlink(path, code=code)
    try:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 1 or metadata.st_size > limit:
            raise DocumentPreviewRuntimeError(code)
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if len(raw) != metadata.st_size or len(raw) > limit:
        raise DocumentPreviewRuntimeError(code)
    return raw


def _read_canonical_hash_named_receipt(
    *,
    root: Path,
    directory: str,
    digest: str,
    limit: int,
    code: str,
) -> tuple[Path, bytes, dict[str, object]]:
    """Read a local receipt only from its hash-named, non-link sidecar path."""

    path = _sidecar_path(root, directory, digest)
    raw = _read_regular_file(path, limit=limit, code=code)
    value = _strict_json_object(raw, code=code)
    if raw != _canonical_json_bytes(value, trailing_newline=True):
        raise DocumentPreviewRuntimeError(code)
    return path, raw, value


def _read_canonical_source_attested_artifact(
    *,
    root: Path,
    directory: str,
    digest: str,
    limit: int,
    code: str,
) -> tuple[Path, bytes, dict[str, object]]:
    """Read only one source evaluator artifact from its fixed local path."""

    path = _sidecar_path(root, directory, digest)
    raw = _read_regular_file(path, limit=limit, code=code)
    value = _strict_json_object(raw, code=code)
    if raw != _canonical_local_artifact_file_bytes(value):
        raise DocumentPreviewRuntimeError(code)
    return path, raw, value


def _verify_self_excluding_receipt_digest(
    *,
    value: Mapping[str, object],
    field_name: str,
    expected_digest: str,
    code: str,
) -> None:
    """Verify a canonical receipt's semantic self digest and filename digest."""

    embedded = _sha256(value.get(field_name), code=code)
    if embedded != expected_digest:
        raise DocumentPreviewRuntimeError(code)
    unsigned = dict(value)
    unsigned.pop(field_name)
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != expected_digest:
        raise DocumentPreviewRuntimeError(code)


def _write_immutable_json(path: Path, value: Mapping[str, object], *, code: str) -> str:
    """Write canonical bytes once, never replacing a hash-named sidecar."""

    payload = _canonical_json_bytes(value, trailing_newline=True)
    digest = _sha256_bytes(payload)
    if path.name != f"{digest}.json":
        raise DocumentPreviewRuntimeError(code)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink(path.parent, code=code)
        if path.exists():
            if _read_regular_file(path, limit=_MAX_SIDECAR_BYTES, code=code) != payload:
                raise DocumentPreviewRuntimeError(code)
            return digest
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        except FileExistsError:
            # Windows normally replaces atomically.  This branch preserves the
            # immutability rule for providers that choose create-only semantics.
            if _read_regular_file(path, limit=_MAX_SIDECAR_BYTES, code=code) != payload:
                raise DocumentPreviewRuntimeError(code)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    return digest


def _write_exact_immutable_bytes(path: Path, payload: bytes, *, limit: int, code: str) -> None:
    """Write an immutable byte sequence whose filename uses a semantic digest.

    Sealed evaluator-result filenames use their self-excluding canonical
    result digest rather than a digest of the full file (which includes that
    self field).  The exact canonical bytes are still immutable once written.
    """

    if not payload or len(payload) > limit:
        raise DocumentPreviewRuntimeError(code)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink(path.parent, code=code)
        if path.exists():
            if _read_regular_file(path, limit=limit, code=code) != payload:
                raise DocumentPreviewRuntimeError(code)
            return
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error


@dataclass(frozen=True, slots=True)
class RuntimeConfigurationV1:
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    runtime_api: PaddleOcrRuntimeApi | None
    options: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ModelPackageV1:
    name: str
    version: str
    license_id: str
    source_uri: str


@dataclass(frozen=True, slots=True)
class ModelArtifactV1:
    artifact_id: str
    relative_path: str
    sha256: str
    byte_size: int
    provenance_kind: Literal["external", "locally_trained_source_attested"]
    license_id: str | None
    source_uri: str | None
    model_training_receipt_sha256: str | None


@dataclass(frozen=True, slots=True)
class ModelInventoryV1:
    engine_kind: ApprovedPdfEngineKind
    runtime_api: PaddleOcrRuntimeApi | None
    packages: tuple[ModelPackageV1, ...]
    artifacts: tuple[ModelArtifactV1, ...]


@dataclass(frozen=True, slots=True)
class PdfPreviewApprovalV1:
    approval_id: str
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    dataset_revision: str
    gold_receipt_sha256: str
    evaluation_result_sha256: str


@dataclass(frozen=True, slots=True)
class SealedEvaluationResultV1:
    dataset_id: str
    dataset_revision: str
    packet_receipt_sha256: str
    gold_receipt_sha256: str
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    evaluation_result_sha256: str
    field_exact_match: float
    ocr_cer: float
    table_cell_f1: float


@dataclass(frozen=True, slots=True)
class SourceAttestedSyntheticPdfPreviewActivationV1:
    """One automatic activation bound only to synthetic layout truth.

    This is intentionally a separate wire shape from ``PdfPreviewApprovalV1``:
    it has no reviewer identity, approval ID, or human Gold receipt.  The
    exact-key parser below makes a mixed human/source-attested pointer
    invalid, rather than allowing either provenance claim to be silently
    substituted for the other.
    """

    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    dataset_id: str
    dataset_revision: str
    packet_receipt_sha256: str
    evaluation_scope: Literal["synthetic_r1_supervised_native_template_compatibility"]
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"]
    layout_truth_receipt_sha256: str
    layout_truth_parser_spec_sha256: str
    development_selection_receipt_sha256: str
    sealed_run_receipt_sha256: str
    evaluation_result_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAttestedSyntheticSealedEvaluationResultV1:
    """Canonical passing result for the synthetic source-attested path only."""

    dataset_id: str
    dataset_revision: str
    packet_receipt_sha256: str
    evaluation_scope: Literal["synthetic_r1_supervised_native_template_compatibility"]
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"]
    layout_truth_receipt_sha256: str
    layout_truth_parser_spec_sha256: str
    development_selection_receipt_sha256: str
    sealed_run_receipt_sha256: str
    candidate_parser_revision: str
    candidate_bundle_sha256: str
    candidate_execution_receipt_sha256: str
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    evaluation_result_sha256: str
    field_exact_match: float
    field_exact_match_count: int
    field_exact_match_total: int
    field_exact_match_denominators: tuple["SourceAttestedFieldMetricDenominatorV1", ...]
    ocr_cer: float
    table_cell_f1: float


@dataclass(frozen=True, slots=True)
class SourceAttestedSyntheticDevelopmentEvaluationResultV1:
    """One exact development candidate result, retained without OCR content."""

    dataset_id: str
    dataset_revision: str
    packet_receipt_sha256: str
    layout_truth_receipt_sha256: str
    layout_truth_parser_spec_sha256: str
    candidate_parser_revision: str
    candidate_bundle_sha256: str
    candidate_execution_receipt_sha256: str
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    evaluation_result_sha256: str
    passed: bool
    field_exact_match: float
    field_exact_match_count: int
    field_exact_match_total: int
    field_exact_match_denominators: tuple["SourceAttestedFieldMetricDenominatorV1", ...]
    ocr_cer: float
    table_cell_f1: float


@dataclass(frozen=True, slots=True)
class SourceAttestedFieldMetricDenominatorV1:
    """One content-free rendered-field denominator used by source replay."""

    document_type: Literal["bill_of_lading", "invoice", "packing_list"]
    document_count: int
    fields_per_document: int
    total_fields: int


@dataclass(frozen=True, slots=True)
class SourceAttestedDatasetAssetV1:
    asset_id: str
    source_sha256: str
    media_type: str
    source_role: str


@dataclass(frozen=True, slots=True)
class SourceAttestedDatasetReceiptV1:
    dataset_id: str
    dataset_revision: str
    dataset_receipt_sha256: str
    assets: tuple[SourceAttestedDatasetAssetV1, ...]


@dataclass(frozen=True, slots=True)
class SourceAttestedPacketItemV1:
    asset_id: str
    source_sha256: str
    fixture_kind: Literal["clean", "error"]
    document_type: Literal["bill_of_lading", "invoice", "packing_list"]
    set_id: str


@dataclass(frozen=True, slots=True)
class SourceAttestedPacketV1:
    dataset_id: str
    dataset_revision: str
    dataset_receipt_sha256: str
    packet_split: Literal["development", "sealed"]
    packet_receipt_sha256: str
    error_labels_sha256: str
    items: tuple[SourceAttestedPacketItemV1, ...]


@dataclass(frozen=True, slots=True)
class SourceAttestedCandidateBundleV1:
    dataset_id: str
    dataset_revision: str
    packet_receipt_sha256: str
    parser_revision: str
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    candidate_bundle_sha256: str
    source_assets: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SourceAttestedLayoutRecordV1:
    asset_id: str
    source_sha256: str
    fixture_kind: Literal["clean", "error"]
    document_type: Literal["bill_of_lading", "invoice", "packing_list"]
    set_id: str
    content_stream_sha256: str
    source_truth_record_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAttestedLayoutTruthReceiptV1:
    dataset_id: str
    dataset_revision: str
    dataset_receipt_sha256: str
    packet_split: Literal["development", "sealed"]
    packet_receipt_sha256: str
    error_labels_sha256: str
    parser_spec_sha256: str
    records: tuple[SourceAttestedLayoutRecordV1, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAttestedDevelopmentCandidateResultV1:
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    candidate_parser_revision: str
    evaluation_result_sha256: str
    candidate_bundle_sha256: str
    candidate_execution_receipt_sha256: str
    passed: bool


@dataclass(frozen=True, slots=True)
class SourceAttestedDevelopmentSelectionReceiptV1:
    dataset_id: str
    dataset_revision: str
    development_packet_receipt_sha256: str
    layout_truth_receipt_sha256: str
    layout_truth_parser_spec_sha256: str
    candidate_parser_revision: str
    candidate_results: tuple[SourceAttestedDevelopmentCandidateResultV1, ...]
    selected: SourceAttestedDevelopmentCandidateResultV1
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAttestedSealedRunReceiptV1:
    dataset_id: str
    dataset_revision: str
    sealed_packet_receipt_sha256: str
    layout_truth_receipt_sha256: str
    layout_truth_parser_spec_sha256: str
    development_selection_receipt_sha256: str
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    selected_development_evaluation_result_sha256: str
    candidate_bundle_sha256: str
    candidate_parser_revision: str
    candidate_execution_receipt_sha256: str
    sealed_candidate_reservation_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAttestedSyntheticReceiptBindingV1:
    dataset_receipt_path: Path
    dataset_receipt: SourceAttestedDatasetReceiptV1
    development_packet_path: Path
    development_packet: SourceAttestedPacketV1
    sealed_packet_path: Path
    sealed_packet: SourceAttestedPacketV1
    development_layout_truth_path: Path
    development_layout_truth: SourceAttestedLayoutTruthReceiptV1
    layout_truth_path: Path
    layout_truth: SourceAttestedLayoutTruthReceiptV1
    development_selection_path: Path
    development_selection: SourceAttestedDevelopmentSelectionReceiptV1
    development_evaluation_paths: tuple[Path, ...]
    development_evaluations: tuple[SourceAttestedSyntheticDevelopmentEvaluationResultV1, ...]
    development_candidate_paths: tuple[Path, ...]
    development_candidates: tuple[SourceAttestedCandidateBundleV1, ...]
    development_execution_paths: tuple[Path, ...]
    sealed_run_path: Path
    sealed_run: SourceAttestedSealedRunReceiptV1
    sealed_run_state_path: Path
    sealed_candidate_path: Path
    sealed_candidate: SourceAttestedCandidateBundleV1
    sealed_execution_path: Path
    sealed_evaluation_path: Path
    sealed_evaluation_state_path: Path
    sealed_reservation_path: Path


PdfPreviewActivationV1 = PdfPreviewApprovalV1 | SourceAttestedSyntheticPdfPreviewActivationV1
PdfPreviewEvaluationResultV1 = (
    SealedEvaluationResultV1 | SourceAttestedSyntheticSealedEvaluationResultV1
)


def _parse_paddleocr_runtime_api(
    value: object,
    *,
    kind: ApprovedPdfEngineKind,
    present: bool,
    code: str,
) -> PaddleOcrRuntimeApi | None:
    """Return one explicit Paddle family without rewriting legacy sidecars.

    Existing PaddleOCR 2.9 sidecars predate ``runtime_api`` and remain valid
    as the legacy family.  The newer PaddleOCR 3/PaddleX profile must name
    itself explicitly, so a P3 option set can never select the 2.9 adapter.
    """

    if kind != "paddleocr":
        if present:
            raise DocumentPreviewRuntimeError(code)
        return None
    if not present:
        return _PADDLEOCR2_RUNTIME_API
    if value != _PADDLEOCR3_RUNTIME_API:
        raise DocumentPreviewRuntimeError(code)
    return _PADDLEOCR3_RUNTIME_API


def _parse_options(
    value: object,
    *,
    kind: ApprovedPdfEngineKind,
    runtime_api: PaddleOcrRuntimeApi | None,
    extractor_revision: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
    options = cast(dict[str, object], value)
    if kind == "pdf_text_layer":
        if extractor_revision == _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION:
            required = {
                "semantic_model_directory",
                "semantic_model_artifact",
                "model_training_receipt_sha256",
            }
            if set(options) != required:
                raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
            model_directory = _relative_model_path(
                options["semantic_model_directory"],
                code="document_preview_configuration_invalid",
            )
            model_artifact = _relative_model_path(
                options["semantic_model_artifact"],
                code="document_preview_configuration_invalid",
            )
            directory_parts = Path(model_directory).parts
            if (
                len(directory_parts) != 3
                or directory_parts[:2] != ("models", "native-pdf-semantic")
                or _SHA256.fullmatch(directory_parts[2]) is None
                or model_artifact
                != f"{model_directory}/{directory_parts[2]}.json"
            ):
                raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
            return {
                "semantic_model_directory": model_directory,
                "semantic_model_artifact": model_artifact,
                "model_training_receipt_sha256": _sha256(
                    options["model_training_receipt_sha256"],
                    code="document_preview_configuration_invalid",
                ),
            }
        if options:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        return {}
    if kind == "paddleocr" and runtime_api == _PADDLEOCR2_RUNTIME_API:
        allowed = {
            "det_model_directory",
            "rec_model_directory",
            "cls_model_directory",
            "ocr_version",
            "use_gpu",
            "cpu_threads",
            "language",
            "use_angle_cls",
        }
        if (
            not {
                "det_model_directory",
                "rec_model_directory",
                "cls_model_directory",
                "ocr_version",
                "use_gpu",
                "cpu_threads",
            }.issubset(options)
            or set(options) - allowed
        ):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        if options["ocr_version"] != "PP-OCRv4":
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        if options["use_gpu"] is not False:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        if options["cpu_threads"] != 1:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        normalized: dict[str, object] = {
            "det_model_directory": _relative_model_path(
                options["det_model_directory"], code="document_preview_configuration_invalid"
            ),
            "rec_model_directory": _relative_model_path(
                options["rec_model_directory"], code="document_preview_configuration_invalid"
            ),
            # PaddleOCR 2.9.1 initializes its classifier model even when
            # angle classification is disabled.  Require an explicit local
            # directory so it cannot choose and download a default one.
            "cls_model_directory": _relative_model_path(
                options["cls_model_directory"], code="document_preview_configuration_invalid"
            ),
            "ocr_version": "PP-OCRv4",
            "use_gpu": False,
            "cpu_threads": 1,
            "language": "en",
            "use_angle_cls": False,
        }
        if "language" in options:
            language = options["language"]
            if not isinstance(language, str) or not re.fullmatch(r"[a-z]{2,16}", language):
                raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
            normalized["language"] = language
        if "use_angle_cls" in options:
            if not isinstance(options["use_angle_cls"], bool):
                raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
            normalized["use_angle_cls"] = options["use_angle_cls"]
        return normalized
    if kind == "paddleocr" and runtime_api == _PADDLEOCR3_RUNTIME_API:
        # PaddleOCR 3 delegates this pipeline to PaddleX.  Keep every model
        # directory and every optional preprocessor explicit: leaving one at
        # its upstream default can select a non-inventoried model and download
        # it during constructor or first inference.
        required = {
            "text_detection_model_name",
            "text_detection_model_directory",
            "text_recognition_model_name",
            "text_recognition_model_directory",
            "raster_scale",
            "device",
            "cpu_threads",
            "enable_hpi",
            "enable_mkldnn",
            "use_tensorrt",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
        }
        if set(options) != required:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        if (
            options["text_detection_model_name"] != "PP-OCRv5_mobile_det"
            or options["text_recognition_model_name"] != "en_PP-OCRv5_mobile_rec"
            # The extractor revision names its input resolution.  Keep the
            # raster scale in the configuration hash rather than relying on
            # pdfium's default (72 dpi).
            or options["raster_scale"] != 3
            or options["device"] != "cpu"
            or options["cpu_threads"] != 1
            or options["enable_hpi"] is not False
            or options["enable_mkldnn"] is not False
            or options["use_tensorrt"] is not False
            or options["use_doc_orientation_classify"] is not False
            or options["use_doc_unwarping"] is not False
            or options["use_textline_orientation"] is not False
        ):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        return {
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_model_directory": _relative_model_path(
                options["text_detection_model_directory"],
                code="document_preview_configuration_invalid",
            ),
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_recognition_model_directory": _relative_model_path(
                options["text_recognition_model_directory"],
                code="document_preview_configuration_invalid",
            ),
            "raster_scale": 3,
            "device": "cpu",
            "cpu_threads": 1,
            "enable_hpi": False,
            "enable_mkldnn": False,
            "use_tensorrt": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
    if kind == "docling":
        allowed = {
            "artifacts_directory",
            "pdf_backend",
            "ocr_enabled",
            "device",
            "num_threads",
            "do_table_structure",
            "table_structure_mode",
        }
        if (
            not {
                "artifacts_directory",
                "pdf_backend",
                "ocr_enabled",
                "device",
                "num_threads",
                "do_table_structure",
                "table_structure_mode",
            }.issubset(options)
            or set(options) - allowed
        ):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        # Docling 2.15.1's OCR-enabled pipeline constructs EasyOCR with a
        # downloader/cache outside our inventory boundary.  This profile is a
        # pinned local PDF structure/table parser, not an optical OCR engine,
        # so only the explicitly offline mode is admissible.
        if (
            options["pdf_backend"] != _DOCLING_PDF_BACKEND
            or options["ocr_enabled"] is not False
            or options["device"] != "cpu"
            or options["num_threads"] != 1
            or options["do_table_structure"] is not True
            or not isinstance(options["table_structure_mode"], str)
            or options["table_structure_mode"] not in _DOCLING_TABLE_STRUCTURE_MODES
        ):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        return {
            "artifacts_directory": _relative_model_path(
                options["artifacts_directory"], code="document_preview_configuration_invalid"
            ),
            "pdf_backend": _DOCLING_PDF_BACKEND,
            "ocr_enabled": False,
            "device": "cpu",
            "num_threads": 1,
            "do_table_structure": True,
            "table_structure_mode": options["table_structure_mode"],
        }
    raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")


def parse_runtime_configuration(raw: bytes) -> RuntimeConfigurationV1:
    value = _strict_json_object(raw, code="document_preview_configuration_invalid")
    base_keys = {"schema_version", "engine_kind", "extractor_revision", "options"}
    runtime_api_present = "runtime_api" in value
    if set(value) != base_keys | ({"runtime_api"} if runtime_api_present else set()):
        raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
    if value["schema_version"] != "1.0":
        raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
    kind = _engine_kind(value["engine_kind"], code="document_preview_configuration_invalid")
    runtime_api = _parse_paddleocr_runtime_api(
        value.get("runtime_api"),
        kind=kind,
        present=runtime_api_present,
        code="document_preview_configuration_invalid",
    )
    extractor_revision = _identifier(
        value["extractor_revision"], code="document_preview_configuration_invalid"
    )
    if kind == "pdf_text_layer" and extractor_revision not in _PDF_TEXT_LAYER_EXTRACTOR_REVISIONS:
        raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
    if (
        kind == "paddleocr"
        and runtime_api == _PADDLEOCR3_RUNTIME_API
        and extractor_revision != _PADDLEOCR3_EXTRACTOR_REVISION
    ):
        raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
    return RuntimeConfigurationV1(
        engine_kind=kind,
        extractor_revision=extractor_revision,
        runtime_api=runtime_api,
        options=_parse_options(
            value["options"],
            kind=kind,
            runtime_api=runtime_api,
            extractor_revision=extractor_revision,
        ),
    )


def _parse_package(value: object) -> ModelPackageV1:
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    item = cast(dict[str, object], value)
    _require_exact_keys(
        item,
        {"name", "version", "license_id", "source_uri"},
        code="document_preview_inventory_invalid",
    )
    name = item["name"]
    version = item["version"]
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name)
        or not isinstance(version, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}", version)
    ):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    return ModelPackageV1(
        name=name,
        version=version,
        license_id=_identifier(item["license_id"], code="document_preview_inventory_invalid"),
        source_uri=_validate_https_source(
            item["source_uri"], code="document_preview_inventory_invalid"
        ),
    )


def _parse_artifact(value: object) -> ModelArtifactV1:
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    item = cast(dict[str, object], value)
    external_keys = {
        "artifact_id",
        "relative_path",
        "sha256",
        "byte_size",
        "license_id",
        "source_uri",
    }
    locally_trained_keys = {
        "artifact_id",
        "relative_path",
        "sha256",
        "byte_size",
        "provenance_kind",
        "model_training_receipt_sha256",
    }
    keys = frozenset(item)
    if keys not in {frozenset(external_keys), frozenset(locally_trained_keys)}:
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    byte_size = item["byte_size"]
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or not 1 <= byte_size <= 8 * 1024**3
    ):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    artifact_id = _identifier(item["artifact_id"], code="document_preview_inventory_invalid")
    relative_path = _relative_model_path(
        item["relative_path"], code="document_preview_inventory_invalid"
    )
    sha256 = _sha256(item["sha256"], code="document_preview_inventory_invalid")
    if keys == external_keys:
        return ModelArtifactV1(
            artifact_id=artifact_id,
            relative_path=relative_path,
            sha256=sha256,
            byte_size=byte_size,
            provenance_kind="external",
            license_id=_identifier(item["license_id"], code="document_preview_inventory_invalid"),
            source_uri=_validate_https_source(
                item["source_uri"], code="document_preview_inventory_invalid"
            ),
            model_training_receipt_sha256=None,
        )
    if item["provenance_kind"] != "locally_trained_source_attested":
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    return ModelArtifactV1(
        artifact_id=artifact_id,
        relative_path=relative_path,
        sha256=sha256,
        byte_size=byte_size,
        provenance_kind="locally_trained_source_attested",
        license_id=None,
        source_uri=None,
        model_training_receipt_sha256=_sha256(
            item["model_training_receipt_sha256"],
            code="document_preview_inventory_invalid",
        ),
    )


def parse_model_inventory(raw: bytes) -> ModelInventoryV1:
    value = _strict_json_object(raw, code="document_preview_inventory_invalid")
    base_keys = {"schema_version", "engine_kind", "packages", "artifacts"}
    runtime_api_present = "runtime_api" in value
    if set(value) != base_keys | ({"runtime_api"} if runtime_api_present else set()):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    if (
        value["schema_version"] != "1.0"
        or not isinstance(value["packages"], list)
        or not isinstance(value["artifacts"], list)
    ):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    kind = _engine_kind(value["engine_kind"], code="document_preview_inventory_invalid")
    runtime_api = _parse_paddleocr_runtime_api(
        value.get("runtime_api"),
        kind=kind,
        present=runtime_api_present,
        code="document_preview_inventory_invalid",
    )
    packages = tuple(_parse_package(item) for item in value["packages"])
    artifacts = tuple(_parse_artifact(item) for item in value["artifacts"])
    if not packages:
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    names = [item.name.casefold().replace("_", "-") for item in packages]
    artifact_ids = [item.artifact_id for item in artifacts]
    artifact_paths = [item.relative_path for item in artifacts]
    if (
        len(names) != len(set(names))
        or len(artifact_ids) != len(set(artifact_ids))
        or len(artifact_paths) != len(set(artifact_paths))
    ):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    return ModelInventoryV1(
        engine_kind=kind,
        runtime_api=runtime_api,
        packages=packages,
        artifacts=artifacts,
    )


def parse_pdf_preview_approval(raw: bytes) -> PdfPreviewApprovalV1:
    value = _strict_json_object(raw, code="document_preview_approval_invalid")
    _require_exact_keys(
        value,
        {"schema_version", "approval_id", "status", "media_type", "engine", "evaluation"},
        code="document_preview_approval_invalid",
    )
    if (
        value["schema_version"] != "1.0"
        or value["status"] != "approved"
        or value["media_type"] != "application/pdf"
    ):
        raise DocumentPreviewRuntimeError("document_preview_approval_invalid")
    engine_value = value["engine"]
    evaluation_value = value["evaluation"]
    if not isinstance(engine_value, dict) or not isinstance(evaluation_value, dict):
        raise DocumentPreviewRuntimeError("document_preview_approval_invalid")
    engine = cast(dict[str, object], engine_value)
    evaluation = cast(dict[str, object], evaluation_value)
    _require_exact_keys(
        engine,
        {"kind", "extractor_revision", "configuration_sha256", "model_inventory_sha256"},
        code="document_preview_approval_invalid",
    )
    _require_exact_keys(
        evaluation,
        {"dataset_revision", "gold_receipt_sha256", "evaluation_result_sha256", "packet_kind"},
        code="document_preview_approval_invalid",
    )
    if evaluation["packet_kind"] != "sealed":
        raise DocumentPreviewRuntimeError("document_preview_approval_invalid")
    return PdfPreviewApprovalV1(
        approval_id=_identifier(value["approval_id"], code="document_preview_approval_invalid"),
        engine_kind=_engine_kind(engine["kind"], code="document_preview_approval_invalid"),
        extractor_revision=_identifier(
            engine["extractor_revision"], code="document_preview_approval_invalid"
        ),
        configuration_sha256=_sha256(
            engine["configuration_sha256"], code="document_preview_approval_invalid"
        ),
        model_inventory_sha256=_sha256(
            engine["model_inventory_sha256"], code="document_preview_approval_invalid"
        ),
        dataset_revision=_identifier(
            evaluation["dataset_revision"], code="document_preview_approval_invalid"
        ),
        gold_receipt_sha256=_sha256(
            evaluation["gold_receipt_sha256"], code="document_preview_approval_invalid"
        ),
        evaluation_result_sha256=_sha256(
            evaluation["evaluation_result_sha256"], code="document_preview_approval_invalid"
        ),
    )


def parse_source_attested_synthetic_pdf_preview_activation(
    raw: bytes,
) -> SourceAttestedSyntheticPdfPreviewActivationV1:
    """Parse the non-human, synthetic-layout activation pointer exactly.

    The automatic path is deliberately unable to carry a manual approval ID
    or a human Gold receipt.  Conversely, its source-layout receipt and
    explicit evaluation basis are mandatory.  This makes provenance mixing a
    parse error before an adapter can be constructed.
    """

    code = "document_preview_source_attested_activation_invalid"
    value = _strict_json_object(raw, code=code)
    _require_exact_keys(
        value,
        {"schema_version", "activation_kind", "status", "media_type", "engine", "evaluation"},
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["activation_kind"] != "source_attested_synthetic"
        or value["status"] != "activated"
        or value["media_type"] != "application/pdf"
    ):
        raise DocumentPreviewRuntimeError(code)
    engine_value = value["engine"]
    evaluation_value = value["evaluation"]
    if not isinstance(engine_value, dict) or not isinstance(evaluation_value, dict):
        raise DocumentPreviewRuntimeError(code)
    engine = cast(dict[str, object], engine_value)
    evaluation = cast(dict[str, object], evaluation_value)
    _require_exact_keys(
        engine,
        {"kind", "extractor_revision", "configuration_sha256", "model_inventory_sha256"},
        code=code,
    )
    _require_exact_keys(
        evaluation,
        {
            "dataset_id",
            "dataset_revision",
            "packet_kind",
            "packet_receipt_sha256",
            "evaluation_scope",
            "layout_truth_kind",
            "layout_truth_receipt_sha256",
            "layout_truth_parser_spec_sha256",
            "development_selection_receipt_sha256",
            "sealed_run_receipt_sha256",
            "evaluation_basis",
            "evaluation_result_sha256",
        },
        code=code,
    )
    if (
        evaluation["packet_kind"] != "sealed"
        or evaluation["evaluation_basis"] != "source_attested_synthetic"
        or evaluation["evaluation_scope"]
        != "synthetic_r1_supervised_native_template_compatibility"
        or evaluation["layout_truth_kind"] != "pdf_content_stream_layout_v1"
    ):
        raise DocumentPreviewRuntimeError(code)
    engine_kind = _engine_kind(engine["kind"], code=code)
    extractor_revision = _identifier(engine["extractor_revision"], code=code)
    dataset_id = _identifier(evaluation["dataset_id"], code=code)
    dataset_revision = _identifier(evaluation["dataset_revision"], code=code)
    packet_receipt_sha256 = _sha256(evaluation["packet_receipt_sha256"], code=code)
    _require_canonical_supervised_source_attested_profile(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=packet_receipt_sha256,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        code=code,
    )
    return SourceAttestedSyntheticPdfPreviewActivationV1(
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=_sha256(engine["configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(engine["model_inventory_sha256"], code=code),
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=packet_receipt_sha256,
        evaluation_scope=cast(
            Literal["synthetic_r1_supervised_native_template_compatibility"],
            evaluation["evaluation_scope"],
        ),
        layout_truth_kind=cast(
            Literal["pdf_content_stream_layout_v1"], evaluation["layout_truth_kind"]
        ),
        layout_truth_receipt_sha256=_sha256(evaluation["layout_truth_receipt_sha256"], code=code),
        layout_truth_parser_spec_sha256=_sha256(
            evaluation["layout_truth_parser_spec_sha256"], code=code
        ),
        development_selection_receipt_sha256=_sha256(
            evaluation["development_selection_receipt_sha256"], code=code
        ),
        sealed_run_receipt_sha256=_sha256(evaluation["sealed_run_receipt_sha256"], code=code),
        evaluation_result_sha256=_sha256(evaluation["evaluation_result_sha256"], code=code),
    )


def parse_pdf_preview_activation(raw: bytes) -> PdfPreviewActivationV1:
    """Parse one of the two non-interchangeable local activation pointers."""

    value = _strict_json_object(raw, code="document_preview_approval_invalid")
    if "activation_kind" in value:
        return parse_source_attested_synthetic_pdf_preview_activation(raw)
    return parse_pdf_preview_approval(raw)


def parse_sealed_evaluation_result(raw: bytes) -> SealedEvaluationResultV1:
    """Validate Gold evaluator's exact canonical sealed-result contract."""

    value = _strict_json_object(raw, code="document_preview_evaluation_invalid")
    if raw != _canonical_json_bytes(value, trailing_newline=True):
        raise DocumentPreviewRuntimeError("document_preview_evaluation_invalid")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "dataset_id",
            "dataset_revision",
            "packet_kind",
            "packet_receipt_sha256",
            "gold_receipt_sha256",
            "engine_kind",
            "extractor_revision",
            "configuration_sha256",
            "model_inventory_sha256",
            "evaluation_result_sha256",
            "passed",
            "metrics",
        },
        code="document_preview_evaluation_invalid",
    )
    if (
        value["schema_version"] != "1.0"
        or value["packet_kind"] != "sealed"
        or value["passed"] is not True
    ):
        raise DocumentPreviewRuntimeError("document_preview_evaluation_invalid")
    metrics_value = value["metrics"]
    if not isinstance(metrics_value, dict):
        raise DocumentPreviewRuntimeError("document_preview_evaluation_invalid")
    metrics = cast(dict[str, object], metrics_value)
    _require_exact_keys(
        metrics,
        {"field_exact_match", "ocr_cer", "table_cell_f1"},
        code="document_preview_evaluation_invalid",
    )
    embedded_sha = _sha256(
        value["evaluation_result_sha256"], code="document_preview_evaluation_invalid"
    )
    unsigned = dict(value)
    unsigned.pop("evaluation_result_sha256")
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != embedded_sha:
        raise DocumentPreviewRuntimeError("document_preview_evaluation_invalid")
    field_exact_match = _finite_number(
        metrics["field_exact_match"], code="document_preview_evaluation_invalid"
    )
    ocr_cer = _finite_number(metrics["ocr_cer"], code="document_preview_evaluation_invalid")
    table_cell_f1 = _finite_number(
        metrics["table_cell_f1"], code="document_preview_evaluation_invalid"
    )
    if (
        field_exact_match < _FIELD_EXACT_MATCH_MIN
        or ocr_cer > _OCR_CER_MAX
        or table_cell_f1 < _TABLE_CELL_F1_MIN
    ):
        raise DocumentPreviewRuntimeError("document_preview_evaluation_gate_failed")
    return SealedEvaluationResultV1(
        dataset_id=_identifier(value["dataset_id"], code="document_preview_evaluation_invalid"),
        dataset_revision=_identifier(
            value["dataset_revision"], code="document_preview_evaluation_invalid"
        ),
        packet_receipt_sha256=_sha256(
            value["packet_receipt_sha256"], code="document_preview_evaluation_invalid"
        ),
        gold_receipt_sha256=_sha256(
            value["gold_receipt_sha256"], code="document_preview_evaluation_invalid"
        ),
        engine_kind=_engine_kind(value["engine_kind"], code="document_preview_evaluation_invalid"),
        extractor_revision=_identifier(
            value["extractor_revision"], code="document_preview_evaluation_invalid"
        ),
        configuration_sha256=_sha256(
            value["configuration_sha256"], code="document_preview_evaluation_invalid"
        ),
        model_inventory_sha256=_sha256(
            value["model_inventory_sha256"], code="document_preview_evaluation_invalid"
        ),
        evaluation_result_sha256=embedded_sha,
        field_exact_match=field_exact_match,
        ocr_cer=ocr_cer,
        table_cell_f1=table_cell_f1,
    )


def _source_attested_nonnegative_int(value: object, *, code: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise DocumentPreviewRuntimeError(code)
    return value


def _source_attested_candidate_parser_revision(value: object, *, code: str) -> str:
    """Accept only the one reviewed visible-layout candidate parser revision.

    Equality across receipts is necessary but not sufficient: a fully
    self-consistent older parser could otherwise mint a new automatic pointer.
    The v11 parser is the only candidate normalizer whose supervised native
    template 13/15/12 field contract has been accepted for this narrow
    synthetic scope.  It supersedes v10, so a coherent v10 chain remains
    ineligible.
    """

    revision = _identifier(value, code=code)
    if revision != _SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION:
        raise DocumentPreviewRuntimeError(code)
    return revision


def _is_canonical_supervised_source_attested_packet(
    *, dataset_id: str, dataset_revision: str, packet_receipt_sha256: str
) -> bool:
    """Identify the one production r1 packet without widening test fixtures.

    Hermetic tests use injected fixture receipts and do not reach the
    production replay boundary.  The actual source-attested loader later
    independently pins this exact corpus and packets.  This narrow predicate
    therefore keeps those fixtures useful while requiring the supervised
    profile everywhere a real r1 receipt can be selected or activated.
    """

    return (
        dataset_id == _SOURCE_ATTESTED_SUPERVISED_DATASET_ID
        and dataset_revision == _SOURCE_ATTESTED_SUPERVISED_DATASET_REVISION
        and packet_receipt_sha256
        in {
            _SOURCE_ATTESTED_SUPERVISED_DEVELOPMENT_PACKET_SHA256,
            _SOURCE_ATTESTED_SUPERVISED_SEALED_PACKET_SHA256,
        }
    )


def _require_canonical_supervised_source_attested_profile(
    *,
    dataset_id: str,
    dataset_revision: str,
    packet_receipt_sha256: str,
    engine_kind: ApprovedPdfEngineKind,
    extractor_revision: str,
    code: str,
) -> None:
    """Reject a coherent but non-supervised r1 automatic-chain member."""

    if not _is_canonical_supervised_source_attested_packet(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=packet_receipt_sha256,
    ):
        return
    if (
        engine_kind != "pdf_text_layer"
        or extractor_revision != _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION
    ):
        raise DocumentPreviewRuntimeError(code)


def _parse_source_attested_metrics(
    value: object,
    *,
    code: str,
) -> tuple[
    float,
    int,
    int,
    tuple[SourceAttestedFieldMetricDenominatorV1, ...],
    float,
    float,
]:
    """Parse the content-free metric shape that source replay recomputes.

    The numerator and denominator are retained in the runtime representation
    even though pointer selection uses the ratio.  That makes it impossible
    for a forged rounded ratio to conceal a changed rendered-field universe
    before the source-truth verifier receives the immutable raw result.
    """

    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError(code)
    metrics = cast(dict[str, object], value)
    _require_exact_keys(
        metrics,
        {
            "field_exact_match",
            "field_exact_match_count",
            "field_exact_match_total",
            "field_exact_match_denominators",
            "ocr_cer",
            "table_cell_f1",
        },
        code=code,
    )
    field_exact_match = _finite_number(metrics["field_exact_match"], code=code)
    ocr_cer = _finite_number(metrics["ocr_cer"], code=code)
    table_cell_f1 = _finite_number(metrics["table_cell_f1"], code=code)
    if not (
        0.0 <= field_exact_match <= 1.0 and 0.0 <= ocr_cer <= 1.0 and 0.0 <= table_cell_f1 <= 1.0
    ):
        raise DocumentPreviewRuntimeError(code)
    field_exact_match_count = _source_attested_nonnegative_int(
        metrics["field_exact_match_count"], code=code, maximum=100_000_000
    )
    field_exact_match_total = _source_attested_nonnegative_int(
        metrics["field_exact_match_total"], code=code, maximum=100_000_000
    )
    if field_exact_match_total < 1 or field_exact_match_count > field_exact_match_total:
        raise DocumentPreviewRuntimeError(code)
    raw_denominators = metrics["field_exact_match_denominators"]
    if not isinstance(raw_denominators, list) or not 1 <= len(raw_denominators) <= 3:
        raise DocumentPreviewRuntimeError(code)
    denominators: list[SourceAttestedFieldMetricDenominatorV1] = []
    for raw_denominator in raw_denominators:
        if not isinstance(raw_denominator, dict):
            raise DocumentPreviewRuntimeError(code)
        denominator = cast(dict[str, object], raw_denominator)
        _require_exact_keys(
            denominator,
            {"document_type", "document_count", "fields_per_document", "total_fields"},
            code=code,
        )
        document_type = denominator["document_type"]
        if document_type not in {"bill_of_lading", "invoice", "packing_list"}:
            raise DocumentPreviewRuntimeError(code)
        document_count = _source_attested_nonnegative_int(
            denominator["document_count"], code=code, maximum=100_000
        )
        fields_per_document = _source_attested_nonnegative_int(
            denominator["fields_per_document"], code=code, maximum=1_000
        )
        total_fields = _source_attested_nonnegative_int(
            denominator["total_fields"], code=code, maximum=100_000_000
        )
        if (
            document_count < 1
            or fields_per_document < 1
            or total_fields < 1
            or total_fields != document_count * fields_per_document
        ):
            raise DocumentPreviewRuntimeError(code)
        denominators.append(
            SourceAttestedFieldMetricDenominatorV1(
                document_type=cast(
                    Literal["bill_of_lading", "invoice", "packing_list"], document_type
                ),
                document_count=document_count,
                fields_per_document=fields_per_document,
                total_fields=total_fields,
            )
        )
    if (
        len({item.document_type for item in denominators}) != len(denominators)
        or sum(item.total_fields for item in denominators) != field_exact_match_total
        or abs(field_exact_match - (field_exact_match_count / field_exact_match_total)) > 1e-12
    ):
        raise DocumentPreviewRuntimeError(code)
    return (
        field_exact_match,
        field_exact_match_count,
        field_exact_match_total,
        tuple(denominators),
        ocr_cer,
        table_cell_f1,
    )


def parse_source_attested_synthetic_sealed_evaluation_result(
    raw: bytes,
) -> SourceAttestedSyntheticSealedEvaluationResultV1:
    """Validate a passing result that uses PDF content-stream layout truth, not Gold.

    The evaluator must emit this distinct canonical shape when its reference
    text and table cells were derived from the synthetic PDF's immutable
    authored content stream.  A human Gold receipt is intentionally not an
    accepted substitute on this path.
    """

    code = "document_preview_source_attested_evaluation_invalid"
    value = _strict_json_object(raw, code=code)
    if raw not in {
        _canonical_json_bytes(value, trailing_newline=True),
        _canonical_local_artifact_file_bytes(value),
    }:
        raise DocumentPreviewRuntimeError(code)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "evaluation_basis",
            "dataset_id",
            "dataset_revision",
            "packet_kind",
            "packet_receipt_sha256",
            "evaluation_scope",
            "layout_truth_kind",
            "layout_truth_receipt_sha256",
            "layout_truth_parser_spec_sha256",
            "development_selection_receipt_sha256",
            "sealed_run_receipt_sha256",
            "engine_kind",
            "extractor_revision",
            "configuration_sha256",
            "model_inventory_sha256",
            "candidate_parser_revision",
            "evaluation_result_sha256",
            "candidate_bundle_sha256",
            "candidate_execution_receipt_sha256",
            "passed",
            "metrics",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["evaluation_basis"] != "source_attested_synthetic"
        or value["packet_kind"] != "sealed"
        or value["evaluation_scope"]
        != "synthetic_r1_supervised_native_template_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
        or value["passed"] is not True
    ):
        raise DocumentPreviewRuntimeError(code)
    embedded_sha = _sha256(value["evaluation_result_sha256"], code=code)
    unsigned = dict(value)
    unsigned.pop("evaluation_result_sha256")
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != embedded_sha:
        raise DocumentPreviewRuntimeError(code)
    (
        field_exact_match,
        field_exact_match_count,
        field_exact_match_total,
        field_exact_match_denominators,
        ocr_cer,
        table_cell_f1,
    ) = _parse_source_attested_metrics(value["metrics"], code=code)
    if (
        field_exact_match < _FIELD_EXACT_MATCH_MIN
        or ocr_cer > _OCR_CER_MAX
        or table_cell_f1 < _TABLE_CELL_F1_MIN
    ):
        raise DocumentPreviewRuntimeError("document_preview_source_attested_evaluation_gate_failed")
    dataset_id = _identifier(value["dataset_id"], code=code)
    dataset_revision = _identifier(value["dataset_revision"], code=code)
    packet_receipt_sha256 = _sha256(value["packet_receipt_sha256"], code=code)
    engine_kind = _engine_kind(value["engine_kind"], code=code)
    extractor_revision = _identifier(value["extractor_revision"], code=code)
    _require_canonical_supervised_source_attested_profile(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=packet_receipt_sha256,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        code=code,
    )
    return SourceAttestedSyntheticSealedEvaluationResultV1(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=packet_receipt_sha256,
        evaluation_scope=cast(
            Literal["synthetic_r1_supervised_native_template_compatibility"],
            value["evaluation_scope"],
        ),
        layout_truth_kind=cast(Literal["pdf_content_stream_layout_v1"], value["layout_truth_kind"]),
        layout_truth_receipt_sha256=_sha256(value["layout_truth_receipt_sha256"], code=code),
        layout_truth_parser_spec_sha256=_sha256(
            value["layout_truth_parser_spec_sha256"], code=code
        ),
        development_selection_receipt_sha256=_sha256(
            value["development_selection_receipt_sha256"], code=code
        ),
        sealed_run_receipt_sha256=_sha256(value["sealed_run_receipt_sha256"], code=code),
        candidate_parser_revision=_source_attested_candidate_parser_revision(
            value["candidate_parser_revision"], code=code
        ),
        candidate_bundle_sha256=_sha256(value["candidate_bundle_sha256"], code=code),
        candidate_execution_receipt_sha256=_sha256(
            value["candidate_execution_receipt_sha256"], code=code
        ),
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=_sha256(value["configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(value["model_inventory_sha256"], code=code),
        evaluation_result_sha256=embedded_sha,
        field_exact_match=field_exact_match,
        field_exact_match_count=field_exact_match_count,
        field_exact_match_total=field_exact_match_total,
        field_exact_match_denominators=field_exact_match_denominators,
        ocr_cer=ocr_cer,
        table_cell_f1=table_cell_f1,
    )


def parse_source_attested_synthetic_development_evaluation_result(
    raw: bytes,
) -> SourceAttestedSyntheticDevelopmentEvaluationResultV1:
    """Parse one source-attested development result without its OCR payload.

    Development artifacts deliberately retain the same strict result shape as
    a sealed result, but their two sealed-only receipt fields are *exactly*
    ``null``.  This prevents a sealed passing result being repackaged as one
    of the three candidates used for deterministic selection.
    """

    code = "document_preview_source_attested_development_evaluation_invalid"
    value = _strict_json_object(raw, code=code)
    if raw not in {
        _canonical_json_bytes(value, trailing_newline=True),
        _canonical_local_artifact_file_bytes(value),
    }:
        raise DocumentPreviewRuntimeError(code)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "evaluation_basis",
            "dataset_id",
            "dataset_revision",
            "packet_kind",
            "packet_receipt_sha256",
            "evaluation_scope",
            "layout_truth_kind",
            "layout_truth_receipt_sha256",
            "layout_truth_parser_spec_sha256",
            "development_selection_receipt_sha256",
            "sealed_run_receipt_sha256",
            "engine_kind",
            "extractor_revision",
            "configuration_sha256",
            "model_inventory_sha256",
            "candidate_parser_revision",
            "evaluation_result_sha256",
            "candidate_bundle_sha256",
            "candidate_execution_receipt_sha256",
            "passed",
            "metrics",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["evaluation_basis"] != "source_attested_synthetic"
        or value["packet_kind"] != "development"
        or value["evaluation_scope"]
        != "synthetic_r1_supervised_native_template_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
        or value["development_selection_receipt_sha256"] is not None
        or value["sealed_run_receipt_sha256"] is not None
        or not isinstance(value["passed"], bool)
    ):
        raise DocumentPreviewRuntimeError(code)
    embedded_sha = _sha256(value["evaluation_result_sha256"], code=code)
    unsigned = dict(value)
    unsigned.pop("evaluation_result_sha256")
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != embedded_sha:
        raise DocumentPreviewRuntimeError(code)
    (
        field_exact_match,
        field_exact_match_count,
        field_exact_match_total,
        field_exact_match_denominators,
        ocr_cer,
        table_cell_f1,
    ) = _parse_source_attested_metrics(value["metrics"], code=code)
    passed = (
        field_exact_match >= _FIELD_EXACT_MATCH_MIN
        and ocr_cer <= _OCR_CER_MAX
        and table_cell_f1 >= _TABLE_CELL_F1_MIN
    )
    if value["passed"] is not passed:
        raise DocumentPreviewRuntimeError(code)
    return SourceAttestedSyntheticDevelopmentEvaluationResultV1(
        dataset_id=_identifier(value["dataset_id"], code=code),
        dataset_revision=_identifier(value["dataset_revision"], code=code),
        packet_receipt_sha256=_sha256(value["packet_receipt_sha256"], code=code),
        layout_truth_receipt_sha256=_sha256(value["layout_truth_receipt_sha256"], code=code),
        layout_truth_parser_spec_sha256=_sha256(
            value["layout_truth_parser_spec_sha256"], code=code
        ),
        candidate_parser_revision=_source_attested_candidate_parser_revision(
            value["candidate_parser_revision"], code=code
        ),
        candidate_bundle_sha256=_sha256(value["candidate_bundle_sha256"], code=code),
        candidate_execution_receipt_sha256=_sha256(
            value["candidate_execution_receipt_sha256"], code=code
        ),
        engine_kind=_engine_kind(value["engine_kind"], code=code),
        extractor_revision=_identifier(value["extractor_revision"], code=code),
        configuration_sha256=_sha256(value["configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(value["model_inventory_sha256"], code=code),
        evaluation_result_sha256=embedded_sha,
        passed=passed,
        field_exact_match=field_exact_match,
        field_exact_match_count=field_exact_match_count,
        field_exact_match_total=field_exact_match_total,
        field_exact_match_denominators=field_exact_match_denominators,
        ocr_cer=ocr_cer,
        table_cell_f1=table_cell_f1,
    )


def _source_attested_layout_record(
    value: object,
) -> SourceAttestedLayoutRecordV1:
    code = "document_preview_source_attested_layout_receipt_invalid"
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError(code)
    record = cast(dict[str, object], value)
    _require_exact_keys(
        record,
        {
            "asset_id",
            "source_sha256",
            "fixture_kind",
            "document_type",
            "set_id",
            "content_stream_sha256",
            "fields",
            "spans",
            "table_cells",
            "source_truth_record_sha256",
        },
        code=code,
    )
    if (
        not isinstance(record["fixture_kind"], str)
        or record["fixture_kind"] not in {"clean", "error"}
        or not isinstance(record["document_type"], str)
        or record["document_type"] not in {"bill_of_lading", "invoice", "packing_list"}
        or not isinstance(record["fields"], list)
        or not isinstance(record["spans"], list)
        or not isinstance(record["table_cells"], list)
    ):
        raise DocumentPreviewRuntimeError(code)
    record_digest = _sha256(record["source_truth_record_sha256"], code=code)
    unsigned = dict(record)
    unsigned.pop("source_truth_record_sha256")
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != record_digest:
        raise DocumentPreviewRuntimeError(code)
    return SourceAttestedLayoutRecordV1(
        asset_id=_identifier(record["asset_id"], code=code),
        source_sha256=_sha256(record["source_sha256"], code=code),
        fixture_kind=cast(Literal["clean", "error"], record["fixture_kind"]),
        document_type=cast(
            Literal["bill_of_lading", "invoice", "packing_list"], record["document_type"]
        ),
        set_id=_identifier(record["set_id"], code=code),
        content_stream_sha256=_sha256(record["content_stream_sha256"], code=code),
        source_truth_record_sha256=record_digest,
    )


def _validate_source_attested_layout_coverage(
    *, records: tuple[SourceAttestedLayoutRecordV1, ...], packet_split: str, code: str
) -> None:
    """Require the fixed source-attested packet shape before activation.

    The source-layout compiler owns deterministic selection.  Runtime does
    not recreate that algorithm, but it does reject an incomplete or
    cross-split packet: development must retain all 60 stratified examples;
    sealed must retain 20 with at least three in every clean/error × document
    type stratum.
    """

    expected_count = _SOURCE_LAYOUT_RECORD_COUNTS[packet_split]
    if len(records) != expected_count:
        raise DocumentPreviewRuntimeError(code)
    asset_ids = [record.asset_id for record in records]
    source_hashes = [record.source_sha256 for record in records]
    if len(asset_ids) != len(set(asset_ids)) or len(source_hashes) != len(set(source_hashes)):
        raise DocumentPreviewRuntimeError(code)
    strata: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.fixture_kind, record.document_type)
        strata[key] = strata.get(key, 0) + 1
    expected_strata = {
        (fixture_kind, document_type)
        for fixture_kind in ("clean", "error")
        for document_type in ("bill_of_lading", "invoice", "packing_list")
    }
    if set(strata) != expected_strata:
        raise DocumentPreviewRuntimeError(code)
    if packet_split == "development":
        if any(count != 10 for count in strata.values()):
            raise DocumentPreviewRuntimeError(code)
    elif any(count < 3 for count in strata.values()):
        raise DocumentPreviewRuntimeError(code)


def _parse_source_attested_layout_truth_receipt(
    *,
    value: Mapping[str, object],
    receipt_sha256: str,
) -> SourceAttestedLayoutTruthReceiptV1:
    code = "document_preview_source_attested_layout_receipt_invalid"
    _require_exact_keys(
        value,
        {
            "schema_version",
            "artifact_kind",
            "provenance_kind",
            "evaluation_scope",
            "layout_truth_kind",
            "compiler_revision",
            "parser_spec_sha256",
            "dataset_id",
            "dataset_revision",
            "dataset_receipt_sha256",
            "packet_kind",
            "packet_split",
            "packet_receipt_sha256",
            "clean_manifest_sha256",
            "error_labels_sha256",
            "source_manifest_mismatch_count",
            "source_manifest_mismatch_receipt_sha256",
            "records",
            "layout_truth_receipt_sha256",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["artifact_kind"] != "source_attested_synthetic_pdf_layout_truth"
        or value["provenance_kind"] != "source_attested_synthetic"
        or value["evaluation_scope"]
        != "synthetic_r1_supervised_native_template_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
        or value["packet_kind"] != "stratified-v2"
        or not isinstance(value["packet_split"], str)
        or value["packet_split"] not in _SOURCE_LAYOUT_RECORD_COUNTS
        or not isinstance(value["source_manifest_mismatch_count"], int)
        or isinstance(value["source_manifest_mismatch_count"], bool)
        or value["source_manifest_mismatch_count"] < 0
        or not isinstance(value["records"], list)
    ):
        raise DocumentPreviewRuntimeError(code)
    # These fields are only needed for the compiler's own provenance record,
    # but accepting untyped values would let a malformed receipt masquerade
    # as one produced by that compiler.
    _identifier(value["compiler_revision"], code=code)
    _sha256(value["dataset_receipt_sha256"], code=code)
    _sha256(value["clean_manifest_sha256"], code=code)
    _sha256(value["error_labels_sha256"], code=code)
    _sha256(value["source_manifest_mismatch_receipt_sha256"], code=code)
    _verify_self_excluding_receipt_digest(
        value=value,
        field_name="layout_truth_receipt_sha256",
        expected_digest=receipt_sha256,
        code=code,
    )
    records = tuple(_source_attested_layout_record(item) for item in value["records"])
    packet_split = cast(Literal["development", "sealed"], value["packet_split"])
    _validate_source_attested_layout_coverage(records=records, packet_split=packet_split, code=code)
    return SourceAttestedLayoutTruthReceiptV1(
        dataset_id=_identifier(value["dataset_id"], code=code),
        dataset_revision=_identifier(value["dataset_revision"], code=code),
        dataset_receipt_sha256=_sha256(value["dataset_receipt_sha256"], code=code),
        packet_split=packet_split,
        packet_receipt_sha256=_sha256(value["packet_receipt_sha256"], code=code),
        error_labels_sha256=_sha256(value["error_labels_sha256"], code=code),
        parser_spec_sha256=_sha256(value["parser_spec_sha256"], code=code),
        records=records,
        receipt_sha256=receipt_sha256,
    )


def _source_attested_development_candidate(
    value: object,
) -> SourceAttestedDevelopmentCandidateResultV1:
    code = "document_preview_source_attested_development_selection_invalid"
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError(code)
    candidate = cast(dict[str, object], value)
    _require_exact_keys(
        candidate,
        {
            "engine_kind",
            "extractor_revision",
            "configuration_sha256",
            "model_inventory_sha256",
            "candidate_parser_revision",
            "evaluation_result_sha256",
            "candidate_bundle_sha256",
            "candidate_execution_receipt_sha256",
            "passed",
        },
        code=code,
    )
    if not isinstance(candidate["passed"], bool):
        raise DocumentPreviewRuntimeError(code)
    return SourceAttestedDevelopmentCandidateResultV1(
        engine_kind=_engine_kind(candidate["engine_kind"], code=code),
        extractor_revision=_identifier(candidate["extractor_revision"], code=code),
        configuration_sha256=_sha256(candidate["configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(candidate["model_inventory_sha256"], code=code),
        candidate_parser_revision=_source_attested_candidate_parser_revision(
            candidate["candidate_parser_revision"], code=code
        ),
        evaluation_result_sha256=_sha256(candidate["evaluation_result_sha256"], code=code),
        candidate_bundle_sha256=_sha256(candidate["candidate_bundle_sha256"], code=code),
        candidate_execution_receipt_sha256=_sha256(
            candidate["candidate_execution_receipt_sha256"], code=code
        ),
        passed=candidate["passed"],
    )


def _parse_source_attested_development_selection_receipt(
    *,
    value: Mapping[str, object],
    receipt_sha256: str,
) -> SourceAttestedDevelopmentSelectionReceiptV1:
    code = "document_preview_source_attested_development_selection_invalid"
    _require_exact_keys(
        value,
        {
            "schema_version",
            "artifact_kind",
            "evaluation_basis",
            "evaluation_scope",
            "layout_truth_kind",
            "dataset_id",
            "dataset_revision",
            "development_packet_receipt_sha256",
            "layout_truth_receipt_sha256",
            "layout_truth_parser_spec_sha256",
            "candidate_parser_revision",
            "candidate_results",
            "selected_engine_kind",
            "selected_extractor_revision",
            "selected_configuration_sha256",
            "selected_model_inventory_sha256",
            "selected_evaluation_result_sha256",
            "development_selection_receipt_sha256",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["artifact_kind"] != "source_attested_synthetic_pdf_development_selection"
        or value["evaluation_basis"] != "source_attested_synthetic"
        or value["evaluation_scope"]
        != "synthetic_r1_supervised_native_template_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
        or not isinstance(value["candidate_results"], list)
    ):
        raise DocumentPreviewRuntimeError(code)
    _verify_self_excluding_receipt_digest(
        value=value,
        field_name="development_selection_receipt_sha256",
        expected_digest=receipt_sha256,
        code=code,
    )
    # The development packet and its distinct layout receipt are loaded and
    # cross-checked by the caller.  Validate their lexical form here so a
    # selection cannot carry an arbitrary non-receipt value until then.
    dataset_id = _identifier(value["dataset_id"], code=code)
    dataset_revision = _identifier(value["dataset_revision"], code=code)
    development_packet_receipt_sha256 = _sha256(
        value["development_packet_receipt_sha256"], code=code
    )
    layout_truth_receipt_sha256 = _sha256(value["layout_truth_receipt_sha256"], code=code)
    candidates = tuple(
        _source_attested_development_candidate(item) for item in value["candidate_results"]
    )
    candidate_parser_revision = _source_attested_candidate_parser_revision(
        value["candidate_parser_revision"], code=code
    )
    selected_engine_kind = _engine_kind(value["selected_engine_kind"], code=code)
    selected_extractor_revision = _identifier(value["selected_extractor_revision"], code=code)
    is_canonical_supervised = _is_canonical_supervised_source_attested_packet(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=development_packet_receipt_sha256,
    )
    if is_canonical_supervised:
        candidates_are_valid = (
            len(candidates) == 1
            and candidates[0].engine_kind == "pdf_text_layer"
            and candidates[0].extractor_revision
            == _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION
            and {item.candidate_parser_revision for item in candidates}
            == {candidate_parser_revision}
        )
    else:
        candidates_are_valid = (
            len(candidates) == len(_ENGINE_KINDS)
            and {item.engine_kind for item in candidates} == _ENGINE_KINDS
            and {item.candidate_parser_revision for item in candidates}
            == {candidate_parser_revision}
        )
    if not candidates_are_valid:
        raise DocumentPreviewRuntimeError(code)
    selected_matches = [
        candidate
        for candidate in candidates
        if (
            candidate.engine_kind == selected_engine_kind
            and candidate.extractor_revision
            == selected_extractor_revision
            and candidate.configuration_sha256
            == _sha256(value["selected_configuration_sha256"], code=code)
            and candidate.model_inventory_sha256
            == _sha256(value["selected_model_inventory_sha256"], code=code)
            and candidate.candidate_parser_revision == candidate_parser_revision
            and candidate.evaluation_result_sha256
            == _sha256(value["selected_evaluation_result_sha256"], code=code)
        )
    ]
    if len(selected_matches) != 1 or not selected_matches[0].passed:
        raise DocumentPreviewRuntimeError(code)
    selected = selected_matches[0]
    _require_canonical_supervised_source_attested_profile(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=development_packet_receipt_sha256,
        engine_kind=selected.engine_kind,
        extractor_revision=selected.extractor_revision,
        code=code,
    )
    return SourceAttestedDevelopmentSelectionReceiptV1(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        development_packet_receipt_sha256=development_packet_receipt_sha256,
        layout_truth_receipt_sha256=layout_truth_receipt_sha256,
        layout_truth_parser_spec_sha256=_sha256(
            value["layout_truth_parser_spec_sha256"], code=code
        ),
        candidate_parser_revision=candidate_parser_revision,
        candidate_results=candidates,
        selected=selected,
        receipt_sha256=receipt_sha256,
    )


def _parse_source_attested_sealed_run_receipt(
    *,
    value: Mapping[str, object],
    receipt_sha256: str,
) -> SourceAttestedSealedRunReceiptV1:
    code = "document_preview_source_attested_sealed_run_invalid"
    _require_exact_keys(
        value,
        {
            "schema_version",
            "artifact_kind",
            "evaluation_basis",
            "evaluation_scope",
            "layout_truth_kind",
            "dataset_id",
            "dataset_revision",
            "sealed_packet_receipt_sha256",
            "layout_truth_receipt_sha256",
            "layout_truth_parser_spec_sha256",
            "development_selection_receipt_sha256",
            "selected_engine_kind",
            "selected_extractor_revision",
            "selected_configuration_sha256",
            "selected_model_inventory_sha256",
            "selected_development_evaluation_result_sha256",
            "candidate_bundle_sha256",
            "candidate_parser_revision",
            "candidate_execution_receipt_sha256",
            "sealed_candidate_reservation_sha256",
            "sealed_run_receipt_sha256",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["artifact_kind"] != "source_attested_synthetic_pdf_sealed_run"
        or value["evaluation_basis"] != "source_attested_synthetic"
        or value["evaluation_scope"]
        != "synthetic_r1_supervised_native_template_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
    ):
        raise DocumentPreviewRuntimeError(code)
    _verify_self_excluding_receipt_digest(
        value=value,
        field_name="sealed_run_receipt_sha256",
        expected_digest=receipt_sha256,
        code=code,
    )
    dataset_id = _identifier(value["dataset_id"], code=code)
    dataset_revision = _identifier(value["dataset_revision"], code=code)
    sealed_packet_receipt_sha256 = _sha256(value["sealed_packet_receipt_sha256"], code=code)
    engine_kind = _engine_kind(value["selected_engine_kind"], code=code)
    extractor_revision = _identifier(value["selected_extractor_revision"], code=code)
    _require_canonical_supervised_source_attested_profile(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        packet_receipt_sha256=sealed_packet_receipt_sha256,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        code=code,
    )
    return SourceAttestedSealedRunReceiptV1(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        sealed_packet_receipt_sha256=sealed_packet_receipt_sha256,
        layout_truth_receipt_sha256=_sha256(value["layout_truth_receipt_sha256"], code=code),
        layout_truth_parser_spec_sha256=_sha256(
            value["layout_truth_parser_spec_sha256"], code=code
        ),
        development_selection_receipt_sha256=_sha256(
            value["development_selection_receipt_sha256"], code=code
        ),
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=_sha256(value["selected_configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(value["selected_model_inventory_sha256"], code=code),
        selected_development_evaluation_result_sha256=_sha256(
            value["selected_development_evaluation_result_sha256"], code=code
        ),
        candidate_bundle_sha256=_sha256(value["candidate_bundle_sha256"], code=code),
        candidate_parser_revision=_source_attested_candidate_parser_revision(
            value["candidate_parser_revision"], code=code
        ),
        candidate_execution_receipt_sha256=_sha256(
            value["candidate_execution_receipt_sha256"], code=code
        ),
        sealed_candidate_reservation_sha256=_sha256(
            value["sealed_candidate_reservation_sha256"], code=code
        ),
        receipt_sha256=receipt_sha256,
    )


def _load_source_attested_dataset_receipt(
    *, root: Path, dataset_receipt_sha256: str
) -> tuple[Path, bytes, SourceAttestedDatasetReceiptV1]:
    code = "document_preview_source_attested_dataset_receipt_invalid"
    # Import only after this runtime module has initialized.  The evaluation
    # package also exposes local runner utilities which import this module, so
    # a top-level import would create an import cycle during service startup.
    from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1, receipt_sha256

    path, raw, _value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_DATASET_RECEIPT_DIRECTORY,
        digest=dataset_receipt_sha256,
        limit=_MAX_SOURCE_DATASET_RECEIPT_BYTES,
        code=code,
    )
    try:
        receipt = DatasetReceiptV1.model_validate_json(raw)
    except ValidationError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if (
        raw != _canonical_local_artifact_file_bytes(receipt.model_dump(mode="json", warnings=False))
        or receipt_sha256(receipt) != dataset_receipt_sha256
    ):
        raise DocumentPreviewRuntimeError(code)
    assets = tuple(
        SourceAttestedDatasetAssetV1(
            asset_id=_identifier(asset.asset_id, code=code),
            source_sha256=_sha256(asset.sha256, code=code),
            media_type=asset.media_type,
            source_role=asset.source_role,
        )
        for asset in receipt.assets
    )
    return (
        path,
        raw,
        SourceAttestedDatasetReceiptV1(
            dataset_id=_identifier(receipt.dataset_id, code=code),
            dataset_revision=_identifier(receipt.dataset_revision, code=code),
            dataset_receipt_sha256=dataset_receipt_sha256,
            assets=assets,
        ),
    )


def _load_source_attested_packet(
    *, root: Path, expected_packet_receipt_sha256: str
) -> tuple[Path, bytes, SourceAttestedPacketV1]:
    code = "document_preview_source_attested_packet_invalid"
    from customs_ai_agent.evaluation.document_ocr_quality import (
        EvaluationPacketV2,
        packet_receipt_sha256,
    )

    path, raw, _value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_PACKET_DIRECTORY,
        digest=expected_packet_receipt_sha256,
        limit=_MAX_SOURCE_PACKET_BYTES,
        code=code,
    )
    try:
        packet = EvaluationPacketV2.model_validate_json(raw)
    except ValidationError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if (
        raw != _canonical_local_artifact_file_bytes(packet.model_dump(mode="json", warnings=False))
        or packet_receipt_sha256(packet) != expected_packet_receipt_sha256
    ):
        raise DocumentPreviewRuntimeError(code)
    items = tuple(
        SourceAttestedPacketItemV1(
            asset_id=_identifier(item.asset_id, code=code),
            source_sha256=_sha256(item.source_sha256, code=code),
            fixture_kind=cast(Literal["clean", "error"], item.fixture_kind),
            document_type=cast(
                Literal["bill_of_lading", "invoice", "packing_list"], item.document_type
            ),
            set_id=_identifier(item.set_id, code=code),
        )
        for item in packet.items
    )
    return (
        path,
        raw,
        SourceAttestedPacketV1(
            dataset_id=_identifier(packet.dataset_id, code=code),
            dataset_revision=_identifier(packet.dataset_revision, code=code),
            dataset_receipt_sha256=_sha256(packet.receipt_sha256, code=code),
            packet_split=cast(Literal["development", "sealed"], packet.split),
            packet_receipt_sha256=expected_packet_receipt_sha256,
            error_labels_sha256=_sha256(packet.error_labels_sha256, code=code),
            items=items,
        ),
    )


def _load_source_attested_candidate_bundle(
    *, root: Path, candidate_bundle_sha256_value: str
) -> tuple[Path, bytes, SourceAttestedCandidateBundleV1]:
    code = "document_preview_source_attested_candidate_bundle_invalid"
    from customs_ai_agent.evaluation.document_ocr_quality import (
        SyntheticPdfCandidateOutputBundleV1,
        candidate_bundle_sha256,
    )

    path, raw, _value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_CANDIDATE_BUNDLE_DIRECTORY,
        digest=candidate_bundle_sha256_value,
        limit=_MAX_SOURCE_CANDIDATE_BUNDLE_BYTES,
        code=code,
    )
    try:
        candidate = SyntheticPdfCandidateOutputBundleV1.model_validate_json(raw)
    except ValidationError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if (
        raw
        != _canonical_local_artifact_file_bytes(candidate.model_dump(mode="json", warnings=False))
        or candidate_bundle_sha256(candidate) != candidate_bundle_sha256_value
    ):
        raise DocumentPreviewRuntimeError(code)
    source_assets = tuple(
        (
            _identifier(document.asset_id, code=code),
            _sha256(document.source_sha256, code=code),
        )
        for document in candidate.documents
    )
    if len(source_assets) != len(set(source_assets)):
        raise DocumentPreviewRuntimeError(code)
    return (
        path,
        raw,
        SourceAttestedCandidateBundleV1(
            dataset_id=_identifier(candidate.dataset_id, code=code),
            dataset_revision=_identifier(candidate.dataset_revision, code=code),
            packet_receipt_sha256=_sha256(candidate.packet_receipt_sha256, code=code),
            parser_revision=_source_attested_candidate_parser_revision(
                candidate.parser_revision, code=code
            ),
            engine_kind=_engine_kind(candidate.engine_kind, code=code),
            extractor_revision=_identifier(candidate.extractor_revision, code=code),
            configuration_sha256=_sha256(candidate.configuration_sha256, code=code),
            model_inventory_sha256=_sha256(candidate.model_inventory_sha256, code=code),
            candidate_bundle_sha256=candidate_bundle_sha256_value,
            source_assets=source_assets,
        ),
    )


def _load_source_attested_development_evaluation(
    *, root: Path, evaluation_result_sha256: str
) -> tuple[Path, bytes, SourceAttestedSyntheticDevelopmentEvaluationResultV1]:
    code = "document_preview_source_attested_development_evaluation_invalid"
    path, raw, _value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_EVALUATION_RESULT_DIRECTORY,
        digest=evaluation_result_sha256,
        limit=_MAX_EVALUATION_BYTES,
        code=code,
    )
    return path, raw, parse_source_attested_synthetic_development_evaluation_result(raw)


def _load_source_attested_sealed_evaluation(
    *, root: Path, evaluation_result_sha256: str
) -> tuple[Path, bytes, SourceAttestedSyntheticSealedEvaluationResultV1]:
    code = "document_preview_source_attested_evaluation_invalid"
    path, raw, _value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_EVALUATION_RESULT_DIRECTORY,
        digest=evaluation_result_sha256,
        limit=_MAX_EVALUATION_BYTES,
        code=code,
    )
    return path, raw, parse_source_attested_synthetic_sealed_evaluation_result(raw)


def _load_source_attested_candidate_execution(
    *, root: Path, candidate_execution_receipt_sha256: str
) -> tuple[Path, bytes]:
    """Read one immutable, content-free candidate execution receipt.

    Its semantic model and raw candidate binding are validated by the source
    replay verifier below.  This reader intentionally establishes the local
    path, ordinary-file, canonical-byte, and hash-named boundary first, so a
    verifier never follows a caller-controlled path or reopens a file after
    it has been checked.
    """

    path, raw, _value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_CANDIDATE_EXECUTION_DIRECTORY,
        digest=candidate_execution_receipt_sha256,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_source_attested_candidate_execution_invalid",
    )
    return path, raw


def _validate_source_attested_packet_coverage(
    *,
    dataset_receipt: SourceAttestedDatasetReceiptV1,
    packet: SourceAttestedPacketV1,
    layout_truth: SourceAttestedLayoutTruthReceiptV1,
    expected_split: Literal["development", "sealed"],
) -> None:
    """Bind a packet's complete PDF inventory to its receipt and layout truth."""

    code = "document_preview_source_attested_packet_binding_mismatch"
    if (
        packet.packet_split != expected_split
        or layout_truth.packet_split != expected_split
        or packet.dataset_id != dataset_receipt.dataset_id
        or packet.dataset_revision != dataset_receipt.dataset_revision
        or packet.dataset_receipt_sha256 != dataset_receipt.dataset_receipt_sha256
        or layout_truth.dataset_id != dataset_receipt.dataset_id
        or layout_truth.dataset_revision != dataset_receipt.dataset_revision
        or layout_truth.dataset_receipt_sha256 != dataset_receipt.dataset_receipt_sha256
        or layout_truth.packet_receipt_sha256 != packet.packet_receipt_sha256
        or layout_truth.error_labels_sha256 != packet.error_labels_sha256
    ):
        raise DocumentPreviewRuntimeError(code)
    assets_by_id = {asset.asset_id: asset for asset in dataset_receipt.assets}
    packet_by_id = {item.asset_id: item for item in packet.items}
    layout_by_id = {record.asset_id: record for record in layout_truth.records}
    if len(packet_by_id) != len(packet.items) or set(packet_by_id) != set(layout_by_id):
        raise DocumentPreviewRuntimeError(code)
    for asset_id, item in packet_by_id.items():
        asset = assets_by_id.get(asset_id)
        layout = layout_by_id.get(asset_id)
        if (
            asset is None
            or asset.source_sha256 != item.source_sha256
            or asset.media_type != "application/pdf"
            or asset.source_role != "preview_candidate"
            or layout is None
            or layout.source_sha256 != item.source_sha256
            or layout.fixture_kind != item.fixture_kind
            or layout.document_type != item.document_type
            or layout.set_id != item.set_id
        ):
            raise DocumentPreviewRuntimeError(code)


def _validate_source_attested_candidate_coverage(
    *,
    candidate: SourceAttestedCandidateBundleV1,
    packet: SourceAttestedPacketV1,
    engine_kind: ApprovedPdfEngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    expected_parser_revision: str | None = None,
) -> None:
    """Prove an OCR bundle covers exactly the already-bound packet, no more."""

    code = "document_preview_source_attested_candidate_bundle_binding_mismatch"
    packet_assets = {(item.asset_id, item.source_sha256) for item in packet.items}
    if (
        candidate.dataset_id != packet.dataset_id
        or candidate.dataset_revision != packet.dataset_revision
        or candidate.packet_receipt_sha256 != packet.packet_receipt_sha256
        or candidate.engine_kind != engine_kind
        or candidate.extractor_revision != extractor_revision
        or candidate.configuration_sha256 != configuration_sha256
        or candidate.model_inventory_sha256 != model_inventory_sha256
        or (
            expected_parser_revision is not None
            and candidate.parser_revision != expected_parser_revision
        )
        or len(candidate.source_assets) != len(packet_assets)
        or set(candidate.source_assets) != packet_assets
    ):
        raise DocumentPreviewRuntimeError(code)


def _validate_source_attested_development_result(
    *,
    result: SourceAttestedSyntheticDevelopmentEvaluationResultV1,
    selection_member: SourceAttestedDevelopmentCandidateResultV1,
    candidate: SourceAttestedCandidateBundleV1,
    development_packet: SourceAttestedPacketV1,
    development_layout_truth: SourceAttestedLayoutTruthReceiptV1,
    selection: SourceAttestedDevelopmentSelectionReceiptV1,
) -> None:
    code = "document_preview_source_attested_development_evaluation_binding_mismatch"
    if (
        result.dataset_id != development_packet.dataset_id
        or result.dataset_revision != development_packet.dataset_revision
        or result.packet_receipt_sha256 != development_packet.packet_receipt_sha256
        or result.layout_truth_receipt_sha256 != development_layout_truth.receipt_sha256
        or result.layout_truth_parser_spec_sha256 != development_layout_truth.parser_spec_sha256
        or result.layout_truth_parser_spec_sha256 != selection.layout_truth_parser_spec_sha256
        or result.candidate_parser_revision != selection_member.candidate_parser_revision
        or result.candidate_parser_revision != selection.candidate_parser_revision
        or result.engine_kind != selection_member.engine_kind
        or result.extractor_revision != selection_member.extractor_revision
        or result.configuration_sha256 != selection_member.configuration_sha256
        or result.model_inventory_sha256 != selection_member.model_inventory_sha256
        or result.evaluation_result_sha256 != selection_member.evaluation_result_sha256
        or result.candidate_bundle_sha256 != selection_member.candidate_bundle_sha256
        or result.candidate_execution_receipt_sha256
        != selection_member.candidate_execution_receipt_sha256
        or result.passed != selection_member.passed
    ):
        raise DocumentPreviewRuntimeError(code)
    _validate_source_attested_candidate_coverage(
        candidate=candidate,
        packet=development_packet,
        engine_kind=result.engine_kind,
        extractor_revision=result.extractor_revision,
        configuration_sha256=result.configuration_sha256,
        model_inventory_sha256=result.model_inventory_sha256,
        expected_parser_revision=result.candidate_parser_revision,
    )
    if candidate.candidate_bundle_sha256 != result.candidate_bundle_sha256:
        raise DocumentPreviewRuntimeError(code)


def _recompute_source_attested_development_selection(
    *,
    results: tuple[SourceAttestedSyntheticDevelopmentEvaluationResultV1, ...],
    selection: SourceAttestedDevelopmentSelectionReceiptV1,
) -> SourceAttestedSyntheticDevelopmentEvaluationResultV1:
    """Replay a verified source-attested development selection.

    The historic hermetic fixtures retain their three-engine metric ordering.
    The only production r1/v11 scope is instead a fixed, hash-bound
    supervised-template profile: it validates one candidate and never
    silently switches to a different engine because of a metric ranking.
    """

    code = "document_preview_source_attested_development_selection_mismatch"
    if _is_canonical_supervised_source_attested_packet(
        dataset_id=selection.dataset_id,
        dataset_revision=selection.dataset_revision,
        packet_receipt_sha256=selection.development_packet_receipt_sha256,
    ):
        if (
            len(results) != 1
            or not results[0].passed
            or results[0].engine_kind != "pdf_text_layer"
            or results[0].extractor_revision
            != _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION
        ):
            raise DocumentPreviewRuntimeError(code)
        expected = results[0]
    else:
        if (
            len(results) != len(_ENGINE_KINDS)
            or {result.engine_kind for result in results} != _ENGINE_KINDS
        ):
            raise DocumentPreviewRuntimeError(code)
        passing = [result for result in results if result.passed]
        if not passing:
            raise DocumentPreviewRuntimeError(code)
        expected = sorted(
            passing,
            key=lambda result: (
                -result.field_exact_match,
                result.ocr_cer,
                -result.table_cell_f1,
                result.engine_kind,
            ),
        )[0]
    if (
        selection.selected.engine_kind != expected.engine_kind
        or selection.selected.extractor_revision != expected.extractor_revision
        or selection.selected.configuration_sha256 != expected.configuration_sha256
        or selection.selected.model_inventory_sha256 != expected.model_inventory_sha256
        or selection.selected.evaluation_result_sha256 != expected.evaluation_result_sha256
        or selection.selected.candidate_bundle_sha256 != expected.candidate_bundle_sha256
        or selection.selected.candidate_parser_revision != expected.candidate_parser_revision
        or selection.selected.candidate_execution_receipt_sha256
        != expected.candidate_execution_receipt_sha256
    ):
        raise DocumentPreviewRuntimeError(code)
    return expected


def _sealed_candidate_reservation_path(root: Path, packet_receipt_sha256: str) -> Path:
    destination = root / _SEALED_RESERVATION_DIRECTORY / f"sealed-{packet_receipt_sha256}.json"
    _assert_no_symlink(destination, code="document_preview_source_attested_reservation_invalid")
    try:
        resolved_parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_source_attested_reservation_invalid"
        ) from error
    if resolved_parent != root / _SEALED_RESERVATION_DIRECTORY:
        raise DocumentPreviewRuntimeError("document_preview_source_attested_reservation_invalid")
    return destination


def _source_attested_state_path(*, root: Path, directory: str, filename: str, code: str) -> Path:
    """Resolve one deterministic create-only evaluator ledger path safely."""

    destination = root / directory / filename
    _assert_no_symlink(destination, code=code)
    try:
        resolved_parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if resolved_parent != root / directory:
        raise DocumentPreviewRuntimeError(code)
    return destination


def _sealed_run_state_path(root: Path, packet_receipt_sha256: str) -> Path:
    return _source_attested_state_path(
        root=root,
        directory=_SOURCE_SEALED_RUN_DIRECTORY,
        filename=f"sealed-run-state-{packet_receipt_sha256}.json",
        code="document_preview_source_attested_sealed_run_state_invalid",
    )


def _sealed_evaluation_state_path(root: Path, packet_receipt_sha256: str) -> Path:
    return _source_attested_state_path(
        root=root,
        directory=_SOURCE_EVALUATION_RESULT_DIRECTORY,
        filename=f"sealed-evaluation-state-{packet_receipt_sha256}.json",
        code="document_preview_source_attested_sealed_evaluation_state_invalid",
    )


def _verify_source_attested_sealed_run_state(
    *, root: Path, sealed_run: SourceAttestedSealedRunReceiptV1
) -> Path:
    """Bind the pre-score, create-only sealed-run ledger to its receipt."""

    code = "document_preview_source_attested_sealed_run_state_invalid"
    path = _sealed_run_state_path(root, sealed_run.sealed_packet_receipt_sha256)
    raw = _read_regular_file(path, limit=_MAX_SIDECAR_BYTES, code=code)
    value = _strict_json_object(raw, code=code)
    if raw != _canonical_local_artifact_file_bytes(value):
        raise DocumentPreviewRuntimeError(code)
    _require_exact_keys(
        value,
        {"schema_version", "sealed_packet_receipt_sha256", "sealed_run_receipt_sha256"},
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or _sha256(value["sealed_packet_receipt_sha256"], code=code)
        != sealed_run.sealed_packet_receipt_sha256
        or _sha256(value["sealed_run_receipt_sha256"], code=code) != sealed_run.receipt_sha256
    ):
        raise DocumentPreviewRuntimeError(code)
    return path


def _verify_source_attested_sealed_evaluation_state(
    *, root: Path, evaluation: SourceAttestedSyntheticSealedEvaluationResultV1
) -> Path:
    """Bind the post-score, create-only evaluation ledger to the sealed result."""

    code = "document_preview_source_attested_sealed_evaluation_state_invalid"
    path = _sealed_evaluation_state_path(root, evaluation.packet_receipt_sha256)
    raw = _read_regular_file(path, limit=_MAX_SIDECAR_BYTES, code=code)
    value = _strict_json_object(raw, code=code)
    if raw != _canonical_local_artifact_file_bytes(value):
        raise DocumentPreviewRuntimeError(code)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "packet_kind",
            "packet_receipt_sha256",
            "layout_truth_receipt_sha256",
            "evaluation_result_sha256",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["packet_kind"] != "sealed"
        or _sha256(value["packet_receipt_sha256"], code=code) != evaluation.packet_receipt_sha256
        or _sha256(value["layout_truth_receipt_sha256"], code=code)
        != evaluation.layout_truth_receipt_sha256
        or _sha256(value["evaluation_result_sha256"], code=code)
        != evaluation.evaluation_result_sha256
    ):
        raise DocumentPreviewRuntimeError(code)
    return path


def _verify_source_attested_sealed_reservation(
    *,
    root: Path,
    sealed_run: SourceAttestedSealedRunReceiptV1,
) -> Path:
    code = "document_preview_source_attested_reservation_invalid"
    path = _sealed_candidate_reservation_path(root, sealed_run.sealed_packet_receipt_sha256)
    raw = _read_regular_file(path, limit=_MAX_SIDECAR_BYTES, code=code)
    if _sha256_bytes(raw) != sealed_run.sealed_candidate_reservation_sha256:
        raise DocumentPreviewRuntimeError(code)
    value = _strict_json_object(raw, code=code)
    if raw != _canonical_local_artifact_file_bytes(value):
        raise DocumentPreviewRuntimeError(code)
    _require_exact_keys(
        value,
        {
            "schema_version",
            "artifact_kind",
            "profile_mode",
            "parser_revision",
            "dataset_id",
            "dataset_revision",
            "packet_kind",
            "packet_receipt_sha256",
            "engine_kind",
            "extractor_revision",
            "configuration_sha256",
            "model_inventory_sha256",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["artifact_kind"] != "sealed_synthetic_pdf_candidate_reservation"
        or value["profile_mode"] != "offline_synthetic_pdf_evaluation"
        or value["packet_kind"] != "sealed"
        or _source_attested_candidate_parser_revision(value["parser_revision"], code=code)
        != sealed_run.candidate_parser_revision
        or _identifier(value["dataset_id"], code=code) != sealed_run.dataset_id
        or _identifier(value["dataset_revision"], code=code) != sealed_run.dataset_revision
        or _sha256(value["packet_receipt_sha256"], code=code)
        != sealed_run.sealed_packet_receipt_sha256
        or _engine_kind(value["engine_kind"], code=code) != sealed_run.engine_kind
        or _identifier(value["extractor_revision"], code=code) != sealed_run.extractor_revision
        or _sha256(value["configuration_sha256"], code=code) != sealed_run.configuration_sha256
        or _sha256(value["model_inventory_sha256"], code=code) != sealed_run.model_inventory_sha256
    ):
        raise DocumentPreviewRuntimeError(code)
    return path


def _verify_source_attested_synthetic_replay(
    *,
    dataset_receipt_raw: bytes,
    development_packet_raw: bytes,
    sealed_packet_raw: bytes,
    development_layout_truth_raw: bytes,
    sealed_layout_truth_raw: bytes,
    development_selection_raw: bytes,
    sealed_run_raw: bytes,
    development_evaluation_raws: tuple[bytes, ...],
    development_candidate_raws: tuple[bytes, ...],
    development_execution_raws: tuple[bytes, ...],
    sealed_evaluation_raw: bytes,
    sealed_candidate_raw: bytes,
    sealed_execution_raw: bytes,
) -> None:
    """Replay every source-attested candidate from already-verified local bytes.

    This is deliberately called by the common receipt loader, rather than
    only the activation writer.  Consequently pointer creation, a fresh
    Agent start, the automatic pointer reloader, and request-time drift
    checks all require the same raw-candidate score replay.  The source
    module receives bytes already obtained through this runtime's safe
    hash-named path reader; it receives no paths and returns no document/OCR
    text.

    Do not pass a test scope or schema override here.  The source verifier's
    default fixed scope is the approved 6,000-PDF fixture and its 60/20
    packets, which prevents a locally shaped fixture from becoming eligible
    for automatic activation.
    """

    code = "document_preview_source_attested_replay_mismatch"
    if not (
        len(development_evaluation_raws)
        == len(development_candidate_raws)
        == len(development_execution_raws)
        and 1 <= len(development_evaluation_raws) <= len(_ENGINE_KINDS)
    ):
        raise DocumentPreviewRuntimeError(code)
    try:
        # Keep imports local: source evaluation utilities also use preview
        # runner helpers, and a module-level edge would create a startup
        # cycle.  These Pydantic models have extra=forbid and verify their
        # semantic self hashes in addition to this runtime's byte checks.
        from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
        from customs_ai_agent.evaluation.document_ocr_quality import EvaluationPacketV2
        from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
            SourceAttestedSyntheticCandidateExecutionV1,
            SourceAttestedSyntheticDevelopmentSelectionV1,
            SourceAttestedSyntheticLayoutTruthV1,
            SourceAttestedSyntheticOcrEvaluationResultV1,
            SourceAttestedSyntheticSealedRunV1,
            verify_source_attested_synthetic_pdf_evaluation_replay,
        )

        receipt = DatasetReceiptV1.model_validate_json(dataset_receipt_raw)
        development_packet = EvaluationPacketV2.model_validate_json(development_packet_raw)
        sealed_packet = EvaluationPacketV2.model_validate_json(sealed_packet_raw)
        development_layout_truth = SourceAttestedSyntheticLayoutTruthV1.model_validate_json(
            development_layout_truth_raw
        )
        sealed_layout_truth = SourceAttestedSyntheticLayoutTruthV1.model_validate_json(
            sealed_layout_truth_raw
        )
        development_selection = SourceAttestedSyntheticDevelopmentSelectionV1.model_validate_json(
            development_selection_raw
        )
        if len(development_evaluation_raws) != len(development_selection.candidate_results):
            raise DocumentPreviewRuntimeError(code)
        sealed_run = SourceAttestedSyntheticSealedRunV1.model_validate_json(sealed_run_raw)
        development_results = tuple(
            SourceAttestedSyntheticOcrEvaluationResultV1.model_validate_json(raw)
            for raw in development_evaluation_raws
        )
        development_executions = tuple(
            SourceAttestedSyntheticCandidateExecutionV1.model_validate_json(raw)
            for raw in development_execution_raws
        )
        for result, candidate_raw, execution in zip(
            development_results,
            development_candidate_raws,
            development_executions,
            strict=True,
        ):
            verify_source_attested_synthetic_pdf_evaluation_replay(
                result=result,
                receipt=receipt,
                packet=development_packet,
                layout_truth=development_layout_truth,
                candidate_output_raw=candidate_raw,
                candidate_execution=execution,
                engine_kind=result.engine_kind,
                extractor_revision=result.extractor_revision,
                configuration_sha256=result.configuration_sha256,
                model_inventory_sha256=result.model_inventory_sha256,
                parser_revision=result.candidate_parser_revision,
            )

        sealed_result = SourceAttestedSyntheticOcrEvaluationResultV1.model_validate_json(
            sealed_evaluation_raw
        )
        sealed_execution = SourceAttestedSyntheticCandidateExecutionV1.model_validate_json(
            sealed_execution_raw
        )
        verify_source_attested_synthetic_pdf_evaluation_replay(
            result=sealed_result,
            receipt=receipt,
            packet=sealed_packet,
            layout_truth=sealed_layout_truth,
            candidate_output_raw=sealed_candidate_raw,
            candidate_execution=sealed_execution,
            engine_kind=sealed_result.engine_kind,
            extractor_revision=sealed_result.extractor_revision,
            configuration_sha256=sealed_result.configuration_sha256,
            model_inventory_sha256=sealed_result.model_inventory_sha256,
            parser_revision=sealed_result.candidate_parser_revision,
            development_selection=development_selection,
            sealed_run=sealed_run,
        )
    except Exception as error:
        # Source truth errors intentionally contain only compact category
        # codes.  Do not surface exception text because a malformed candidate
        # may contain user document/OCR content.
        raise DocumentPreviewRuntimeError(code) from error


def load_source_attested_synthetic_receipt_binding(
    *,
    root: Path,
    evaluation: SourceAttestedSyntheticSealedEvaluationResultV1,
) -> SourceAttestedSyntheticReceiptBindingV1:
    """Load every immutable input that made automatic activation possible.

    A source-attested sealed result is intentionally insufficient on its own:
    its metrics only become meaningful after the runtime has independently
    loaded the selected source PDFs' receipt, both packets, both distinct
    content-stream layout receipts, all three development candidates/results/
    execution receipts, the sealed candidate/execution receipt, and the
    one-shot reservation.  No data in this path is supplied by a caller or
    inferred from the active pointer.
    """

    binding_code = "document_preview_source_attested_receipt_binding_mismatch"

    # The generic runtime copy is compact JSON.  The source evaluator keeps a
    # separate, hash-named pretty canonical copy.  Require semantic equality
    # so a caller cannot activate a result that was never persisted by the
    # source-attested evaluation workflow.
    sealed_evaluation_path, sealed_evaluation_raw, sealed_evaluation = (
        _load_source_attested_sealed_evaluation(
            root=root, evaluation_result_sha256=evaluation.evaluation_result_sha256
        )
    )
    if sealed_evaluation != evaluation:
        raise DocumentPreviewRuntimeError(binding_code)

    layout_path, layout_raw, layout_value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_LAYOUT_TRUTH_DIRECTORY,
        digest=evaluation.layout_truth_receipt_sha256,
        limit=_MAX_LAYOUT_TRUTH_RECEIPT_BYTES,
        code="document_preview_source_attested_layout_receipt_invalid",
    )
    layout_truth = _parse_source_attested_layout_truth_receipt(
        value=layout_value,
        receipt_sha256=evaluation.layout_truth_receipt_sha256,
    )
    selection_path, selection_raw, selection_value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_DEVELOPMENT_SELECTION_DIRECTORY,
        digest=evaluation.development_selection_receipt_sha256,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_source_attested_development_selection_invalid",
    )
    selection = _parse_source_attested_development_selection_receipt(
        value=selection_value,
        receipt_sha256=evaluation.development_selection_receipt_sha256,
    )
    development_layout_path, development_layout_raw, development_layout_value = (
        _read_canonical_source_attested_artifact(
            root=root,
            directory=_SOURCE_LAYOUT_TRUTH_DIRECTORY,
            digest=selection.layout_truth_receipt_sha256,
            limit=_MAX_LAYOUT_TRUTH_RECEIPT_BYTES,
            code="document_preview_source_attested_layout_receipt_invalid",
        )
    )
    development_layout_truth = _parse_source_attested_layout_truth_receipt(
        value=development_layout_value,
        receipt_sha256=selection.layout_truth_receipt_sha256,
    )
    sealed_run_path, sealed_run_raw, sealed_run_value = _read_canonical_source_attested_artifact(
        root=root,
        directory=_SOURCE_SEALED_RUN_DIRECTORY,
        digest=evaluation.sealed_run_receipt_sha256,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_source_attested_sealed_run_invalid",
    )
    sealed_run = _parse_source_attested_sealed_run_receipt(
        value=sealed_run_value,
        receipt_sha256=evaluation.sealed_run_receipt_sha256,
    )

    dataset_receipt_path, dataset_receipt_raw, dataset_receipt = (
        _load_source_attested_dataset_receipt(
            root=root, dataset_receipt_sha256=layout_truth.dataset_receipt_sha256
        )
    )
    development_packet_path, development_packet_raw, development_packet = (
        _load_source_attested_packet(
            root=root,
            expected_packet_receipt_sha256=selection.development_packet_receipt_sha256,
        )
    )
    sealed_packet_path, sealed_packet_raw, sealed_packet = _load_source_attested_packet(
        root=root, expected_packet_receipt_sha256=evaluation.packet_receipt_sha256
    )
    _validate_source_attested_packet_coverage(
        dataset_receipt=dataset_receipt,
        packet=development_packet,
        layout_truth=development_layout_truth,
        expected_split="development",
    )
    _validate_source_attested_packet_coverage(
        dataset_receipt=dataset_receipt,
        packet=sealed_packet,
        layout_truth=layout_truth,
        expected_split="sealed",
    )

    if (
        layout_truth.packet_split != "sealed"
        or layout_truth.dataset_id != evaluation.dataset_id
        or layout_truth.dataset_revision != evaluation.dataset_revision
        or layout_truth.packet_receipt_sha256 != evaluation.packet_receipt_sha256
        or layout_truth.parser_spec_sha256 != evaluation.layout_truth_parser_spec_sha256
        or development_layout_truth.packet_split != "development"
        or development_layout_truth.dataset_id != evaluation.dataset_id
        or development_layout_truth.dataset_revision != evaluation.dataset_revision
        or development_layout_truth.dataset_receipt_sha256 != layout_truth.dataset_receipt_sha256
        or development_layout_truth.packet_receipt_sha256
        != selection.development_packet_receipt_sha256
        or development_layout_truth.parser_spec_sha256 != selection.layout_truth_parser_spec_sha256
        or selection.dataset_id != evaluation.dataset_id
        or selection.dataset_revision != evaluation.dataset_revision
        # Development and sealed layout *receipt hashes* are intentionally
        # different (60 vs 20 documents), but their parser spec must be one.
        or selection.layout_truth_parser_spec_sha256 != evaluation.layout_truth_parser_spec_sha256
        or selection.candidate_parser_revision != evaluation.candidate_parser_revision
        or selection.selected.engine_kind != evaluation.engine_kind
        or selection.selected.extractor_revision != evaluation.extractor_revision
        or selection.selected.configuration_sha256 != evaluation.configuration_sha256
        or selection.selected.model_inventory_sha256 != evaluation.model_inventory_sha256
        or sealed_run.dataset_id != evaluation.dataset_id
        or sealed_run.dataset_revision != evaluation.dataset_revision
        or sealed_run.sealed_packet_receipt_sha256 != evaluation.packet_receipt_sha256
        or sealed_run.layout_truth_receipt_sha256 != evaluation.layout_truth_receipt_sha256
        or sealed_run.layout_truth_parser_spec_sha256 != evaluation.layout_truth_parser_spec_sha256
        or sealed_run.development_selection_receipt_sha256
        != evaluation.development_selection_receipt_sha256
        or sealed_run.engine_kind != evaluation.engine_kind
        or sealed_run.extractor_revision != evaluation.extractor_revision
        or sealed_run.configuration_sha256 != evaluation.configuration_sha256
        or sealed_run.model_inventory_sha256 != evaluation.model_inventory_sha256
        or sealed_run.selected_development_evaluation_result_sha256
        != selection.selected.evaluation_result_sha256
        or sealed_run.candidate_bundle_sha256 != evaluation.candidate_bundle_sha256
        or sealed_run.candidate_parser_revision != evaluation.candidate_parser_revision
        or sealed_run.candidate_execution_receipt_sha256
        != evaluation.candidate_execution_receipt_sha256
        or {record.asset_id for record in development_layout_truth.records}
        & {record.asset_id for record in layout_truth.records}
        or {record.source_sha256 for record in development_layout_truth.records}
        & {record.source_sha256 for record in layout_truth.records}
    ):
        raise DocumentPreviewRuntimeError(binding_code)

    development_evaluation_paths: list[Path] = []
    development_evaluations: list[SourceAttestedSyntheticDevelopmentEvaluationResultV1] = []
    development_candidate_paths: list[Path] = []
    development_candidates: list[SourceAttestedCandidateBundleV1] = []
    development_evaluation_raws: list[bytes] = []
    development_candidate_raws: list[bytes] = []
    development_execution_paths: list[Path] = []
    development_execution_raws: list[bytes] = []
    for member in selection.candidate_results:
        evaluation_path, development_evaluation_raw, development_result = (
            _load_source_attested_development_evaluation(
                root=root, evaluation_result_sha256=member.evaluation_result_sha256
            )
        )
        candidate_path, development_candidate_raw, development_candidate = (
            _load_source_attested_candidate_bundle(
                root=root, candidate_bundle_sha256_value=member.candidate_bundle_sha256
            )
        )
        execution_path, execution_raw = _load_source_attested_candidate_execution(
            root=root,
            candidate_execution_receipt_sha256=member.candidate_execution_receipt_sha256,
        )
        _validate_source_attested_development_result(
            result=development_result,
            selection_member=member,
            candidate=development_candidate,
            development_packet=development_packet,
            development_layout_truth=development_layout_truth,
            selection=selection,
        )
        development_evaluation_paths.append(evaluation_path)
        development_evaluations.append(development_result)
        development_candidate_paths.append(candidate_path)
        development_candidates.append(development_candidate)
        development_evaluation_raws.append(development_evaluation_raw)
        development_candidate_raws.append(development_candidate_raw)
        development_execution_paths.append(execution_path)
        development_execution_raws.append(execution_raw)
    selected_development_result = _recompute_source_attested_development_selection(
        results=tuple(development_evaluations), selection=selection
    )
    if (
        selected_development_result.evaluation_result_sha256
        != sealed_run.selected_development_evaluation_result_sha256
    ):
        raise DocumentPreviewRuntimeError(binding_code)

    sealed_candidate_path, sealed_candidate_raw, sealed_candidate = (
        _load_source_attested_candidate_bundle(
            root=root, candidate_bundle_sha256_value=sealed_run.candidate_bundle_sha256
        )
    )
    _validate_source_attested_candidate_coverage(
        candidate=sealed_candidate,
        packet=sealed_packet,
        engine_kind=evaluation.engine_kind,
        extractor_revision=evaluation.extractor_revision,
        configuration_sha256=evaluation.configuration_sha256,
        model_inventory_sha256=evaluation.model_inventory_sha256,
        expected_parser_revision=sealed_run.candidate_parser_revision,
    )
    if sealed_candidate.candidate_bundle_sha256 != evaluation.candidate_bundle_sha256:
        raise DocumentPreviewRuntimeError(binding_code)

    sealed_execution_path, sealed_execution_raw = _load_source_attested_candidate_execution(
        root=root,
        candidate_execution_receipt_sha256=evaluation.candidate_execution_receipt_sha256,
    )
    if (
        sealed_run.candidate_execution_receipt_sha256
        != evaluation.candidate_execution_receipt_sha256
        or sealed_run.candidate_parser_revision != evaluation.candidate_parser_revision
        or sealed_candidate.parser_revision != evaluation.candidate_parser_revision
    ):
        raise DocumentPreviewRuntimeError(binding_code)

    sealed_run_state_path = _verify_source_attested_sealed_run_state(
        root=root, sealed_run=sealed_run
    )
    sealed_evaluation_state_path = _verify_source_attested_sealed_evaluation_state(
        root=root, evaluation=evaluation
    )
    reservation_path = _verify_source_attested_sealed_reservation(root=root, sealed_run=sealed_run)
    _verify_source_attested_synthetic_replay(
        dataset_receipt_raw=dataset_receipt_raw,
        development_packet_raw=development_packet_raw,
        sealed_packet_raw=sealed_packet_raw,
        development_layout_truth_raw=development_layout_raw,
        sealed_layout_truth_raw=layout_raw,
        development_selection_raw=selection_raw,
        sealed_run_raw=sealed_run_raw,
        development_evaluation_raws=tuple(development_evaluation_raws),
        development_candidate_raws=tuple(development_candidate_raws),
        development_execution_raws=tuple(development_execution_raws),
        sealed_evaluation_raw=sealed_evaluation_raw,
        sealed_candidate_raw=sealed_candidate_raw,
        sealed_execution_raw=sealed_execution_raw,
    )
    return SourceAttestedSyntheticReceiptBindingV1(
        dataset_receipt_path=dataset_receipt_path,
        dataset_receipt=dataset_receipt,
        development_packet_path=development_packet_path,
        development_packet=development_packet,
        sealed_packet_path=sealed_packet_path,
        sealed_packet=sealed_packet,
        development_layout_truth_path=development_layout_path,
        development_layout_truth=development_layout_truth,
        layout_truth_path=layout_path,
        layout_truth=layout_truth,
        development_selection_path=selection_path,
        development_selection=selection,
        development_evaluation_paths=tuple(development_evaluation_paths),
        development_evaluations=tuple(development_evaluations),
        development_candidate_paths=tuple(development_candidate_paths),
        development_candidates=tuple(development_candidates),
        development_execution_paths=tuple(development_execution_paths),
        sealed_run_path=sealed_run_path,
        sealed_run=sealed_run,
        sealed_run_state_path=sealed_run_state_path,
        sealed_candidate_path=sealed_candidate_path,
        sealed_candidate=sealed_candidate,
        sealed_execution_path=sealed_execution_path,
        sealed_evaluation_path=sealed_evaluation_path,
        sealed_evaluation_state_path=sealed_evaluation_state_path,
        sealed_reservation_path=reservation_path,
    )


def parse_pdf_preview_evaluation_result(raw: bytes) -> PdfPreviewEvaluationResultV1:
    """Parse exactly one sealed evaluation basis; never infer or convert it."""

    value = _strict_json_object(raw, code="document_preview_evaluation_invalid")
    if "evaluation_basis" in value:
        return parse_source_attested_synthetic_sealed_evaluation_result(raw)
    return parse_sealed_evaluation_result(raw)


def _artifact_path(root: Path, relative_path: str) -> Path:
    destination = root.joinpath(*Path(relative_path).parts)
    _assert_no_symlink(destination, code="document_preview_model_artifact_invalid")
    try:
        resolved = destination.resolve(strict=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid") from error
    try:
        resolved.relative_to(root / "models")
    except ValueError as error:
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid") from error
    return resolved


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ArtifactFingerprint:
    path: Path
    sha256: str
    byte_size: int
    mtime_ns: int


def _verify_inventory_artifacts(
    root: Path, inventory: ModelInventoryV1
) -> tuple[_ArtifactFingerprint, ...]:
    fingerprints: list[_ArtifactFingerprint] = []
    for artifact in inventory.artifacts:
        path = _artifact_path(root, artifact.relative_path)
        try:
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != artifact.byte_size:
                raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
            if _hash_file(path) != artifact.sha256:
                raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
        except OSError as error:
            raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid") from error
        fingerprints.append(
            _ArtifactFingerprint(
                path=path,
                sha256=artifact.sha256,
                byte_size=artifact.byte_size,
                mtime_ns=metadata.st_mtime_ns,
            )
        )
    return tuple(fingerprints)


_DOCLING_EXECUTION_PACKAGES = frozenset(
    {
        # The direct local Docling profile instantiates the PDF parser,
        # GLM assembler, layout model, and TableFormer.  These distributions
        # can affect its output even though only ``docling`` is imported by
        # this module.  They must therefore be hash-bound inventory entries,
        # not an unobserved dependency closure supplied by the host.
        "deepsearch-glm",
        "docling",
        "docling-core",
        "docling-ibm-models",
        "docling-parse",
        "numpy",
        "pillow",
        "pypdf",
        "pypdfium2",
        "rtree",
        "safetensors",
        "scipy",
        "torch",
        "torchvision",
        "transformers",
    }
)


_PADDLE_EXECUTION_PACKAGES = frozenset(
    {
        # PaddleOCR 2.9.1's declared runtime requirements in the pinned
        # uv.lock, plus the PDF rasterizer/parser and inference runtime that
        # this adapter imports directly. The engine may select different code
        # paths for a document, so do not reduce this to modules loaded by one
        # sample run.
        "albucore",
        "albumentations",
        "beautifulsoup4",
        "cython",
        "fire",
        "fonttools",
        "imgaug",
        "lmdb",
        "numpy",
        # All three distributions can own the same ``cv2`` import tree in a
        # locked Windows environment. Require each recorded version, then
        # separately verify that the actually imported bytes are contributed
        # by the designated contrib wheel; silently accepting an install-order
        # dependent provider would make candidate output non-reproducible.
        "opencv-contrib-python",
        "opencv-python",
        "opencv-python-headless",
        "paddleocr",
        "paddlepaddle",
        "pillow",
        "pyclipper",
        "pypdf",
        "pypdfium2",
        "python-docx",
        "pyyaml",
        "rapidfuzz",
        "requests",
        "scikit-image",
        "shapely",
        "tqdm",
    }
)
_PADDLE_CV2_PROVIDER = "opencv-contrib-python"

# PaddleOCR 3.2 runs the configured OCR path through
# ``paddlex[ocr-core]``.  This is deliberately not a softened version of the
# legacy 2.9 closure: the two lines install the same ``paddleocr``
# distribution name with incompatible APIs and cannot be mixed in one
# profile.
#
# The map is the complete, normalized, *activated* requirement closure of
# PaddleOCR 3.2.0, PaddleX 3.2.0 with ``ocr-core``, PaddlePaddle 3.1.1, and
# the adapter's PDFium rasterizer in the designated ASCII-only CPython 3.11 /
# Windows AMD64 environment.  It was derived from installed distribution
# metadata with current platform markers applied.  Unselected PaddleX extras
# and unrelated service/test packages are intentionally absent; accepting
# either would make this narrow profile depend on host installation choices.
#
# Pin every member, rather than only the three top-level packages.  A library
# such as ``huggingface-hub`` or ``pydantic-core`` can affect construction,
# parsing, or model resolution before inference, so recording just the nine
# directly imported modules would leave a mutable execution path outside the
# inventory hash.
_PADDLEOCR3_PINNED_PACKAGE_VERSIONS: Mapping[str, str] = {
    "aistudio-sdk": "0.3.9",
    "annotated-types": "0.8.0",
    "anyio": "4.14.2",
    "bce-python-sdk": "0.9.76",
    "certifi": "2026.7.22",
    "chardet": "7.6.0",
    "charset-normalizer": "3.5.1",
    "click": "8.4.2",
    "colorama": "0.4.6",
    "colorlog": "6.12.0",
    "crc32c": "2.8",
    "filelock": "3.32.3",
    "fsspec": "2026.7.0",
    "future": "1.0.0",
    "h11": "0.16.0",
    "hf-xet": "1.6.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.27.0",
    "idna": "3.18",
    "imagesize": "2.0.0",
    "modelscope": "1.39.1",
    "modelscope-hub": "0.2.0",
    "networkx": "3.6.1",
    "numpy": "2.4.6",
    "opencv-contrib-python": "4.10.0.84",
    "opt-einsum": "3.3.0",
    "packaging": "26.3",
    "paddleocr": "3.2.0",
    "paddlepaddle": "3.1.1",
    "paddlex": "3.2.0",
    "pandas": "3.0.5",
    "pillow": "12.3.0",
    "prettytable": "3.18.0",
    "protobuf": "7.35.1",
    "psutil": "7.2.2",
    "py-cpuinfo": "9.0.0",
    "pyclipper": "1.4.0",
    "pycryptodome": "3.23.0",
    "pydantic": "2.13.4",
    "pydantic-core": "2.46.4",
    "pypdfium2": "4.30.0",
    "python-dateutil": "2.9.0.post0",
    "pyyaml": "6.0.2",
    "requests": "2.34.2",
    "ruamel-yaml": "0.19.1",
    "setuptools": "84.0.0",
    "shapely": "2.1.2",
    "six": "1.17.0",
    "tqdm": "4.70.0",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.4",
    "tzdata": "2026.3",
    "ujson": "5.13.0",
    "urllib3": "2.7.0",
    "wcwidth": "0.8.2",
}
_PADDLEOCR3_EXECUTION_PACKAGES: Final[frozenset[str]] = frozenset(
    _PADDLEOCR3_PINNED_PACKAGE_VERSIONS
)


def _required_packages(
    kind: ApprovedPdfEngineKind,
    runtime_api: PaddleOcrRuntimeApi | None = None,
) -> frozenset[str]:
    if kind == "pdf_text_layer":
        if runtime_api is not None:
            raise DocumentPreviewRuntimeError("document_preview_inventory_packages_invalid")
        return frozenset({"pypdf"})
    if kind == "paddleocr":
        if runtime_api in {None, _PADDLEOCR2_RUNTIME_API}:
            return _PADDLE_EXECUTION_PACKAGES
        if runtime_api == _PADDLEOCR3_RUNTIME_API:
            return _PADDLEOCR3_EXECUTION_PACKAGES
        raise DocumentPreviewRuntimeError("document_preview_inventory_packages_invalid")
    if runtime_api is not None:
        raise DocumentPreviewRuntimeError("document_preview_inventory_packages_invalid")
    return _DOCLING_EXECUTION_PACKAGES


def _normalized_distribution_name(name: str) -> str:
    return name.casefold().replace("_", "-")


def _recorded_cv2_tree_matches_designated_provider(
    *,
    distribution_reader: Callable[[str], Any] = importlib.metadata.distribution,
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> None:
    """Verify the selected ``cv2`` tree against the designated wheel RECORD.

    ``opencv-contrib-python``, ``opencv-python``, and
    ``opencv-python-headless`` can all install ``cv2``. Distribution versions
    alone cannot prove which wheel's files remain after an install or repair.
    The locked Paddle profile designates contrib as its provider and validates
    the exact regular-file set plus every hash-bearing ``cv2/`` record entry
    before a profile can be used. Links, Windows reparse points, missing
    files, and extra files are all rejected.

    This intentionally verifies the imported provider tree, not an unbounded
    site-packages SBOM. The remaining interpreter environment is governed by
    the operator's ``uv sync --locked`` installation boundary.
    """

    try:
        distribution = distribution_reader(_PADDLE_CV2_PROVIDER)
        # Some OpenCV builds can load optional resources while importing.
        # Treat that import as model-runtime work and keep it inside the same
        # per-thread network denial boundary as the adapter itself. Disable
        # Python bytecode output while importing so a pristine RECORD-bound
        # tree is not changed by the verifier itself.
        with _BYTECODE_GUARD_LOCK:
            previous_dont_write_bytecode = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                with local_files_only():
                    module = module_importer("cv2")
            finally:
                sys.dont_write_bytecode = previous_dont_write_bytecode
        module_file = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        installation_root = Path(str(distribution.locate_file(""))).resolve(strict=True)
        provider_root = Path(str(distribution.locate_file("cv2"))).resolve(strict=True)
        if (
            not provider_root.is_dir()
            or module_file != provider_root / "__init__.py"
            or module_file.parent != provider_root
            or provider_root.parent != installation_root
        ):
            raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")

        record = distribution.read_text("RECORD")
        if not record:
            raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
        expected_entries: dict[str, tuple[str, int]] = {}
        for row in csv.reader(io.StringIO(record)):
            if len(row) != 3:
                raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
            relative_path, expected_hash, expected_size = row
            if not relative_path.startswith("cv2/"):
                continue
            record_path = PurePosixPath(relative_path)
            if (
                len(record_path.parts) < 2
                or record_path.parts[0] != "cv2"
                or record_path.is_absolute()
                or relative_path != record_path.as_posix()
                or any(part in {"", ".", ".."} for part in record_path.parts)
                or not expected_hash.startswith("sha256=")
                or not expected_size.isdecimal()
            ):
                raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
            path_from_cv2 = PurePosixPath(*record_path.parts[1:]).as_posix()
            if path_from_cv2 in expected_entries:
                raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
            expected_entries[path_from_cv2] = (expected_hash, int(expected_size))
        if not expected_entries:
            raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")

        provider_metadata = provider_root.lstat()
        if _is_link_or_reparse_point(provider_metadata) or not stat.S_ISDIR(
            provider_metadata.st_mode
        ):
            raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
        actual_entries: dict[str, tuple[Path, os.stat_result]] = {}
        directories = [provider_root]
        while directories:
            current = directories.pop()
            current_metadata = current.lstat()
            if _is_link_or_reparse_point(current_metadata) or not stat.S_ISDIR(
                current_metadata.st_mode
            ):
                raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
            for child in current.iterdir():
                metadata = child.lstat()
                if _is_link_or_reparse_point(metadata):
                    raise DocumentPreviewRuntimeError(
                        "document_preview_runtime_dependency_unavailable"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    directories.append(child)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise DocumentPreviewRuntimeError(
                        "document_preview_runtime_dependency_unavailable"
                    )
                relative_path = child.relative_to(provider_root).as_posix()
                if relative_path in actual_entries:
                    raise DocumentPreviewRuntimeError(
                        "document_preview_runtime_dependency_unavailable"
                    )
                actual_entries[relative_path] = (child, metadata)

        if set(actual_entries) != set(expected_entries):
            raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
        for relative_path, (artifact_path, metadata) in actual_entries.items():
            expected_hash, expected_byte_size = expected_entries[relative_path]
            if metadata.st_size != expected_byte_size:
                raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
            actual_hash = (
                base64.urlsafe_b64encode(bytes.fromhex(_hash_file(artifact_path)))
                .decode("ascii")
                .rstrip("=")
            )
            if expected_hash != f"sha256={actual_hash}":
                raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
    except DocumentPreviewRuntimeError:
        raise
    except Exception as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_runtime_dependency_unavailable"
        ) from error


def verify_inventory_packages(
    inventory: ModelInventoryV1,
    *,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> None:
    packages = {_normalized_distribution_name(item.name): item for item in inventory.packages}
    if not _required_packages(inventory.engine_kind, inventory.runtime_api).issubset(packages):
        raise DocumentPreviewRuntimeError("document_preview_inventory_packages_invalid")
    if inventory.runtime_api == _PADDLEOCR3_RUNTIME_API:
        if set(packages) != _PADDLEOCR3_EXECUTION_PACKAGES:
            raise DocumentPreviewRuntimeError("document_preview_inventory_packages_invalid")
        for name, expected_version in _PADDLEOCR3_PINNED_PACKAGE_VERSIONS.items():
            if packages[name].version != expected_version:
                raise DocumentPreviewRuntimeError("document_preview_inventory_packages_invalid")
    for item in inventory.packages:
        try:
            installed = package_version_reader(item.name)
        except importlib.metadata.PackageNotFoundError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_runtime_dependency_unavailable"
            ) from error
        except Exception as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_runtime_dependency_unavailable"
            ) from error
        if installed != item.version:
            raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
    if inventory.engine_kind == "paddleocr":
        _recorded_cv2_tree_matches_designated_provider()


def _configuration_directories_exist(
    root: Path, configuration: RuntimeConfigurationV1, inventory: ModelInventoryV1
) -> tuple[tuple[str, Path], ...]:
    if configuration.engine_kind == "pdf_text_layer":
        return ()
    configured = [
        value
        for key, value in configuration.options.items()
        if key.endswith("_directory") and isinstance(value, str)
    ]
    artifact_paths = [item.relative_path for item in inventory.artifacts]
    directories: list[tuple[str, Path]] = []
    for relative in configured:
        path = _artifact_path(root, relative)
        try:
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or _is_link_or_reparse_point(metadata):
                raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        except OSError as error:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid") from error
        prefix = f"{relative.rstrip('/')}/"
        if not any(item == relative or item.startswith(prefix) for item in artifact_paths):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        directories.append((relative, path))
    return tuple(directories)


def _configuration_inventory_profile_matches(
    configuration: RuntimeConfigurationV1, inventory: ModelInventoryV1
) -> bool:
    """Require both the engine and its non-interchangeable API family."""

    return (
        configuration.engine_kind == inventory.engine_kind
        and configuration.runtime_api == inventory.runtime_api
    )


def _verify_supervised_template_inventory_binding(
    configuration: RuntimeConfigurationV1,
    inventory: ModelInventoryV1,
) -> None:
    """Bind the local supervised model to its exact runtime configuration.

    The trained selector is not an externally licensed download, so it must
    never be represented by a made-up HTTPS source or a copied model-license
    field.  Its inventory entry instead carries the immutable training receipt
    digest.  Every other engine/profile remains external-artifact-only.
    """

    locally_trained = tuple(
        item
        for item in inventory.artifacts
        if item.provenance_kind == "locally_trained_source_attested"
    )
    is_supervised_template = (
        configuration.engine_kind == "pdf_text_layer"
        and configuration.extractor_revision
        == _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION
    )
    if not is_supervised_template:
        if locally_trained:
            raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
        return
    options = configuration.options
    model_directory = options.get("semantic_model_directory")
    model_artifact = options.get("semantic_model_artifact")
    training_receipt_sha256 = options.get("model_training_receipt_sha256")
    if not all(
        isinstance(value, str)
        for value in (model_directory, model_artifact, training_receipt_sha256)
    ):
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
    if len(inventory.artifacts) != 1 or len(locally_trained) != 1:
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
    artifact = locally_trained[0]
    model_directory_text = cast(str, model_directory)
    model_artifact_text = cast(str, model_artifact)
    training_receipt_sha256_text = cast(str, training_receipt_sha256)
    model_sha256 = Path(model_directory_text).name
    if (
        artifact.relative_path != model_artifact_text
        or artifact.sha256 != model_sha256
        or artifact.model_training_receipt_sha256 != training_receipt_sha256_text
        or artifact.artifact_id != f"{SUPERVISED_TEMPLATE_MODEL_KIND}-{model_sha256}"
        or artifact.license_id is not None
        or artifact.source_uri is not None
    ):
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")


def _is_supervised_template_configuration(configuration: RuntimeConfigurationV1) -> bool:
    return (
        configuration.engine_kind == "pdf_text_layer"
        and configuration.extractor_revision
        == _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION
    )


def _load_r1_supervised_template_provenance_binding(
    *,
    root: Path,
    configuration: RuntimeConfigurationV1,
    inventory: ModelInventoryV1,
) -> "R1SupervisedTemplateProvenanceBinding | None":
    """Load the model only after its inventory/configuration chain is exact."""

    if not _is_supervised_template_configuration(configuration):
        return None
    _verify_supervised_template_inventory_binding(configuration, inventory)
    model_artifact = configuration.options.get("semantic_model_artifact")
    training_receipt_sha256 = configuration.options.get("model_training_receipt_sha256")
    if not isinstance(model_artifact, str) or not isinstance(training_receipt_sha256, str):
        raise DocumentPreviewRuntimeError("document_preview_supervised_template_provenance_invalid")
    model_path = _artifact_path(root, model_artifact)
    expected_model_sha256 = Path(model_artifact).stem
    try:
        # Delay the import to preserve the runtime/evaluation module boundary:
        # the candidate runner imports this service module, while provenance is
        # intentionally usable by both candidate and serving paths.
        from customs_ai_agent.evaluation.supervised_template_provenance import (
            SupervisedTemplateProvenanceError,
            load_r1_supervised_template_provenance,
        )

        binding = load_r1_supervised_template_provenance(
            document_preview_root=root,
            model_artifact_path=model_path,
            expected_model_training_receipt_sha256=training_receipt_sha256,
            expected_model_artifact_sha256=expected_model_sha256,
        )
        binding.verified_model()
    except (SupervisedTemplateProvenanceError, OSError, ValueError) as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_supervised_template_provenance_invalid"
        ) from error
    return binding


def _model_relative_path_key(relative_path: str) -> tuple[str, ...]:
    """Normalize a declared local relative path for the host filesystem."""

    parts = Path(relative_path).parts
    if os.name == "nt":
        return tuple(part.casefold() for part in parts)
    return tuple(parts)


def _is_relative_descendant(candidate: tuple[str, ...], directory: tuple[str, ...]) -> bool:
    return len(candidate) > len(directory) and candidate[: len(directory)] == directory


def _walk_regular_model_files(*, root: Path, directory: Path) -> frozenset[tuple[str, ...]]:
    """Return every regular file below one configured model directory.

    The model folder is an immutable inventory boundary, not a cache.  A
    symlink, junction, device, or other non-regular entry is therefore a
    provenance failure even when its target happens to be inside ``_local``.
    """

    discovered: set[tuple[str, ...]] = set()
    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            for child in current.iterdir():
                metadata = child.lstat()
                if _is_link_or_reparse_point(metadata):
                    raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(child)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
                try:
                    relative = child.relative_to(root).as_posix()
                except ValueError as error:
                    raise DocumentPreviewRuntimeError(
                        "document_preview_model_artifact_invalid"
                    ) from error
                discovered.add(_model_relative_path_key(relative))
        except DocumentPreviewRuntimeError:
            raise
        except OSError as error:
            raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid") from error
    return frozenset(discovered)


def _verify_configured_model_directory_inventory(
    root: Path,
    configuration: RuntimeConfigurationV1,
    inventory: ModelInventoryV1,
    *,
    configured_directories: tuple[tuple[str, Path], ...] | None = None,
) -> None:
    """Require an exact declared-file set in every configured model directory.

    Hash/size validation is performed separately by
    :func:`_verify_inventory_artifacts`.  This companion check closes the
    important gap that a declared subset could otherwise leave arbitrary,
    unverified model files available to PaddleOCR or Docling at runtime.
    """

    directories = configured_directories
    if directories is None:
        directories = _configuration_directories_exist(root, configuration, inventory)
    declared = {_model_relative_path_key(item.relative_path) for item in inventory.artifacts}
    if len(declared) != len(inventory.artifacts):
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
    for relative, directory in directories:
        directory_key = _model_relative_path_key(relative)
        expected = {
            artifact_path
            for artifact_path in declared
            if _is_relative_descendant(artifact_path, directory_key)
        }
        if not expected or _walk_regular_model_files(root=root, directory=directory) != expected:
            raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")


def verify_configured_model_directory_artifacts(
    root: Path, configuration: RuntimeConfigurationV1, inventory: ModelInventoryV1
) -> tuple[_ArtifactFingerprint, ...]:
    """Verify the exact, fully hashed local model material for one profile.

    This shared internal boundary is used by both evaluation and activation so
    an unapproved regular file cannot be seen by an evaluator but rejected
    only later by the serving runtime.
    """

    if not _configuration_inventory_profile_matches(configuration, inventory):
        raise DocumentPreviewRuntimeError("document_preview_provisioning_kind_mismatch")
    _verify_supervised_template_inventory_binding(configuration, inventory)
    directories = _configuration_directories_exist(root, configuration, inventory)
    fingerprints = _verify_inventory_artifacts(root, inventory)
    _verify_configured_model_directory_inventory(
        root, configuration, inventory, configured_directories=directories
    )
    return fingerprints


def _check_evaluation_binding(
    approval: PdfPreviewActivationV1,
    evaluation: PdfPreviewEvaluationResultV1,
    configuration: RuntimeConfigurationV1,
    inventory: ModelInventoryV1,
) -> None:
    if isinstance(approval, PdfPreviewApprovalV1) and isinstance(
        evaluation, SealedEvaluationResultV1
    ):
        provenance_matches = (
            approval.dataset_revision == evaluation.dataset_revision
            and approval.gold_receipt_sha256 == evaluation.gold_receipt_sha256
        )
    elif isinstance(approval, SourceAttestedSyntheticPdfPreviewActivationV1) and isinstance(
        evaluation, SourceAttestedSyntheticSealedEvaluationResultV1
    ):
        provenance_matches = (
            approval.dataset_id == evaluation.dataset_id
            and approval.dataset_revision == evaluation.dataset_revision
            and approval.packet_receipt_sha256 == evaluation.packet_receipt_sha256
            and approval.evaluation_scope == evaluation.evaluation_scope
            and approval.layout_truth_kind == evaluation.layout_truth_kind
            and approval.layout_truth_receipt_sha256 == evaluation.layout_truth_receipt_sha256
            and approval.layout_truth_parser_spec_sha256
            == evaluation.layout_truth_parser_spec_sha256
            and approval.development_selection_receipt_sha256
            == evaluation.development_selection_receipt_sha256
            and approval.sealed_run_receipt_sha256 == evaluation.sealed_run_receipt_sha256
        )
    else:
        # A human-reviewed result can never activate the source-attested
        # pointer (or the other way around), even when every engine hash is
        # coincidentally identical.
        raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")
    if (
        not provenance_matches
        or approval.evaluation_result_sha256 != evaluation.evaluation_result_sha256
        or approval.engine_kind != evaluation.engine_kind
        or approval.extractor_revision != evaluation.extractor_revision
        or approval.configuration_sha256 != evaluation.configuration_sha256
        or approval.model_inventory_sha256 != evaluation.model_inventory_sha256
        or approval.engine_kind != configuration.engine_kind
        or approval.extractor_revision != configuration.extractor_revision
        or approval.engine_kind != inventory.engine_kind
    ):
        raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")


@dataclass(frozen=True, slots=True)
class ApprovedPdfPreviewBinding:
    """Verified immutable state captured when the Agent process starts."""

    root: Path
    approval_path: Path
    approval_sha256: str
    approval: PdfPreviewActivationV1
    configuration_path: Path
    inventory_path: Path
    evaluation_path: Path
    configuration: RuntimeConfigurationV1
    inventory: ModelInventoryV1
    evaluation: PdfPreviewEvaluationResultV1
    source_attested_receipt_binding: SourceAttestedSyntheticReceiptBindingV1 | None
    supervised_template_provenance_binding: "R1SupervisedTemplateProvenanceBinding | None"
    artifact_fingerprints: tuple[_ArtifactFingerprint, ...]

    def verify_runtime_state(self) -> None:
        """Fail closed on pointer/sidecar drift without rehashing huge models.

        Model artifacts are fully rehashed at activation.  Request-time checks
        require the configured model directories to retain the exact declared
        regular-file set, then compare declared files' size and nanosecond
        mtime to the approved fingerprints.  Pointer/config/inventory/
        evaluation sidecars are small and are re-read and rehashed on every
        preview request.
        """

        approval_raw = _read_regular_file(
            self.approval_path,
            limit=_MAX_POINTER_BYTES,
            code="document_preview_approval_runtime_changed",
        )
        if (
            _sha256_bytes(approval_raw) != self.approval_sha256
            or parse_pdf_preview_activation(approval_raw) != self.approval
        ):
            raise DocumentPreviewRuntimeError("document_preview_approval_runtime_changed")
        configuration_raw = _read_regular_file(
            self.configuration_path,
            limit=_MAX_SIDECAR_BYTES,
            code="document_preview_configuration_runtime_changed",
        )
        if (
            _sha256_bytes(configuration_raw) != self.approval.configuration_sha256
            or parse_runtime_configuration(configuration_raw) != self.configuration
        ):
            raise DocumentPreviewRuntimeError("document_preview_configuration_runtime_changed")
        inventory_raw = _read_regular_file(
            self.inventory_path,
            limit=_MAX_SIDECAR_BYTES,
            code="document_preview_inventory_runtime_changed",
        )
        if (
            _sha256_bytes(inventory_raw) != self.approval.model_inventory_sha256
            or parse_model_inventory(inventory_raw) != self.inventory
        ):
            raise DocumentPreviewRuntimeError("document_preview_inventory_runtime_changed")
        evaluation_raw = _read_regular_file(
            self.evaluation_path,
            limit=_MAX_EVALUATION_BYTES,
            code="document_preview_evaluation_runtime_changed",
        )
        if parse_pdf_preview_evaluation_result(evaluation_raw) != self.evaluation:
            raise DocumentPreviewRuntimeError("document_preview_evaluation_runtime_changed")
        if isinstance(self.evaluation, SourceAttestedSyntheticSealedEvaluationResultV1):
            if (
                self.source_attested_receipt_binding is None
                or load_source_attested_synthetic_receipt_binding(
                    root=self.root, evaluation=self.evaluation
                )
                != self.source_attested_receipt_binding
            ):
                raise DocumentPreviewRuntimeError(
                    "document_preview_source_attested_receipts_runtime_changed"
                )
        elif self.source_attested_receipt_binding is not None:
            raise DocumentPreviewRuntimeError(
                "document_preview_source_attested_receipts_runtime_changed"
            )
        if _is_supervised_template_configuration(self.configuration):
            if self.supervised_template_provenance_binding is None:
                raise DocumentPreviewRuntimeError(
                    "document_preview_supervised_template_runtime_changed"
                )
            try:
                self.supervised_template_provenance_binding.verify_runtime_state()
            except Exception as error:
                raise DocumentPreviewRuntimeError(
                    "document_preview_supervised_template_runtime_changed"
                ) from error
        elif self.supervised_template_provenance_binding is not None:
            raise DocumentPreviewRuntimeError("document_preview_supervised_template_runtime_changed")
        try:
            _verify_configured_model_directory_inventory(
                self.root, self.configuration, self.inventory
            )
        except DocumentPreviewRuntimeError as error:
            raise DocumentPreviewRuntimeError("document_preview_model_runtime_changed") from error
        for fingerprint in self.artifact_fingerprints:
            _assert_no_symlink(fingerprint.path, code="document_preview_model_runtime_changed")
            try:
                metadata = fingerprint.path.stat()
            except OSError as error:
                raise DocumentPreviewRuntimeError(
                    "document_preview_model_runtime_changed"
                ) from error
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != fingerprint.byte_size
                or metadata.st_mtime_ns != fingerprint.mtime_ns
            ):
                raise DocumentPreviewRuntimeError("document_preview_model_runtime_changed")


def load_approved_pdf_preview_binding(
    *,
    approval_path: Path,
    approval_sha256: str,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> ApprovedPdfPreviewBinding:
    """Load one exact local approval and all of its immutable dependencies."""

    if _SHA256.fullmatch(approval_sha256) is None:
        raise DocumentPreviewRuntimeError("document_preview_approval_hash_invalid")
    root, canonical_approval_path = validate_approval_pointer_path(approval_path)
    approval_raw = _read_regular_file(
        canonical_approval_path,
        limit=_MAX_POINTER_BYTES,
        code="document_preview_approval_invalid",
    )
    if _sha256_bytes(approval_raw) != approval_sha256:
        raise DocumentPreviewRuntimeError("document_preview_approval_hash_mismatch")
    approval = parse_pdf_preview_activation(approval_raw)
    configuration_path = _sidecar_path(
        root, "runtime-configurations", approval.configuration_sha256
    )
    inventory_path = _sidecar_path(root, "model-inventories", approval.model_inventory_sha256)
    evaluation_path = _sidecar_path(root, "evaluation-results", approval.evaluation_result_sha256)
    configuration_raw = _read_regular_file(
        configuration_path,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_configuration_invalid",
    )
    inventory_raw = _read_regular_file(
        inventory_path,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_inventory_invalid",
    )
    evaluation_raw = _read_regular_file(
        evaluation_path,
        limit=_MAX_EVALUATION_BYTES,
        code="document_preview_evaluation_invalid",
    )
    if _sha256_bytes(configuration_raw) != approval.configuration_sha256:
        raise DocumentPreviewRuntimeError("document_preview_configuration_hash_mismatch")
    if _sha256_bytes(inventory_raw) != approval.model_inventory_sha256:
        raise DocumentPreviewRuntimeError("document_preview_inventory_hash_mismatch")
    configuration = parse_runtime_configuration(configuration_raw)
    inventory = parse_model_inventory(inventory_raw)
    evaluation = parse_pdf_preview_evaluation_result(evaluation_raw)
    _check_evaluation_binding(approval, evaluation, configuration, inventory)
    source_attested_receipt_binding = (
        load_source_attested_synthetic_receipt_binding(root=root, evaluation=evaluation)
        if isinstance(evaluation, SourceAttestedSyntheticSealedEvaluationResultV1)
        else None
    )
    verify_inventory_packages(inventory, package_version_reader=package_version_reader)
    fingerprints = verify_configured_model_directory_artifacts(root, configuration, inventory)
    supervised_template_provenance_binding = _load_r1_supervised_template_provenance_binding(
        root=root,
        configuration=configuration,
        inventory=inventory,
    )
    return ApprovedPdfPreviewBinding(
        root=root,
        approval_path=canonical_approval_path,
        approval_sha256=approval_sha256,
        approval=approval,
        configuration_path=configuration_path,
        inventory_path=inventory_path,
        evaluation_path=evaluation_path,
        configuration=configuration,
        inventory=inventory,
        evaluation=evaluation,
        source_attested_receipt_binding=source_attested_receipt_binding,
        supervised_template_provenance_binding=supervised_template_provenance_binding,
        artifact_fingerprints=fingerprints,
    )


def _chunks(text: str, *, size: int = 8_000) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    return [normalized[index : index + size] for index in range(0, len(normalized), size)]


@dataclass(frozen=True, slots=True)
class _DoclingPageGeometry:
    """Validated Docling page geometry used for page-relative provenance."""

    page_number: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _DoclingStructuredTableCell:
    """One direct table-grid cell retained for the evaluation-only adapter."""

    page_number: int
    table_index: int
    row_index: int
    column_index: int
    text: str


@dataclass(frozen=True, slots=True)
class _DoclingStructuredExtraction:
    """One in-memory Docling result shared by preview and candidate paths.

    The public preview contract deliberately exposes only pages and spans.
    Table cells stay internal here so the evaluation adapter can score the
    exact same conversion without re-parsing a Markdown export.
    """

    pages: tuple[DocumentPreviewPageV1, ...]
    table_cells: tuple[_DoclingStructuredTableCell, ...]


def _docling_structured_output_invalid() -> DocumentPreviewExecutionError:
    """Return one content-free failure for malformed converter output."""

    return DocumentPreviewExecutionError("Docling structured output is invalid")


def _docling_page_dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _docling_structured_output_invalid()
    dimension = float(value)
    if not math.isfinite(dimension) or not 0 < dimension <= 100_000:
        raise _docling_structured_output_invalid()
    return math.ceil(dimension)


def _docling_page_geometries(document: object) -> dict[int, _DoclingPageGeometry]:
    """Accept only contiguous Docling pages with explicit finite dimensions."""

    page_values = getattr(document, "pages", None)
    if not isinstance(page_values, Mapping) or not page_values:
        raise _docling_structured_output_invalid()
    geometries: dict[int, _DoclingPageGeometry] = {}
    for raw_page_number, page in page_values.items():
        if (
            isinstance(raw_page_number, bool)
            or not isinstance(raw_page_number, int)
            or raw_page_number < 1
            or raw_page_number in geometries
        ):
            raise _docling_structured_output_invalid()
        size = getattr(page, "size", None)
        width = _docling_page_dimension(getattr(size, "width", None))
        height = _docling_page_dimension(getattr(size, "height", None))
        geometries[raw_page_number] = _DoclingPageGeometry(
            page_number=raw_page_number,
            width=width,
            height=height,
        )
    page_numbers = sorted(geometries)
    if page_numbers != list(range(1, len(page_numbers) + 1)):
        raise _docling_structured_output_invalid()
    return geometries


def _docling_coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _docling_structured_output_invalid()
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise _docling_structured_output_invalid()
    return coordinate


def _docling_normalized_bbox(bbox: object, *, geometry: _DoclingPageGeometry) -> BoundingBoxV1:
    """Convert one verified Docling bbox without clipping or guessing it."""

    left = _docling_coordinate(getattr(bbox, "l", None))
    top = _docling_coordinate(getattr(bbox, "t", None))
    right = _docling_coordinate(getattr(bbox, "r", None))
    bottom = _docling_coordinate(getattr(bbox, "b", None))
    raw_origin = getattr(bbox, "coord_origin", None)
    origin = getattr(raw_origin, "value", raw_origin)
    if origin == "BOTTOMLEFT":
        top, bottom = geometry.height - top, geometry.height - bottom
    elif origin != "TOPLEFT":
        raise _docling_structured_output_invalid()
    if not (0 <= left < right <= geometry.width and 0 <= top < bottom <= geometry.height):
        raise _docling_structured_output_invalid()
    return BoundingBoxV1(
        x=left / geometry.width,
        y=top / geometry.height,
        width=(right - left) / geometry.width,
        height=(bottom - top) / geometry.height,
    )


def _docling_single_provenance(
    item: object, *, geometries: Mapping[int, _DoclingPageGeometry]
) -> tuple[int, BoundingBoxV1]:
    """Require one unambiguous page and bbox for every emitted item."""

    provenance_values = getattr(item, "prov", None)
    if (
        not isinstance(provenance_values, Sequence)
        or isinstance(provenance_values, (str, bytes))
        or len(provenance_values) != 1
    ):
        raise _docling_structured_output_invalid()
    provenance = provenance_values[0]
    page_number = getattr(provenance, "page_no", None)
    if (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number not in geometries
    ):
        raise _docling_structured_output_invalid()
    return page_number, _docling_normalized_bbox(
        getattr(provenance, "bbox", None), geometry=geometries[page_number]
    )


def _docling_append_span(
    *,
    spans_by_page: Mapping[int, list[DocumentSpanV1]],
    page_number: int,
    text: str,
    bbox: BoundingBoxV1 | None,
) -> None:
    """Append bounded, raw Docling text while preserving page reading order."""

    for chunk in _chunks(text):
        spans = spans_by_page[page_number]
        ordinal = len(spans) + 1
        spans.append(
            DocumentSpanV1(
                span_id=f"page-{page_number}-span-{ordinal}",
                ordinal=ordinal,
                reading_order=ordinal,
                text=chunk,
                bbox=bbox,
            )
        )


def _is_docling_table_item(item: object) -> bool:
    """Recognize the optional dependency's TableItem without importing it."""

    data = getattr(item, "data", None)
    return data is not None and all(
        hasattr(data, attribute)
        # Do not probe the computed ``grid`` property here: it materializes a
        # row/column matrix.  Dimensions are bounded before accessing it.
        for attribute in ("table_cells", "num_rows", "num_cols")
    )


def _docling_table_dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000:
        raise _docling_structured_output_invalid()
    return value


def _docling_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _docling_structured_output_invalid()
    return value


def _docling_table_cell_text(cell: object) -> str:
    text = getattr(cell, "text", None)
    if not isinstance(text, str):
        raise _docling_structured_output_invalid()
    normalized = text.strip()
    if len(normalized) > 8_000:
        raise _docling_structured_output_invalid()
    return normalized


def _docling_table_cell_bbox(
    cell: object, *, geometry: _DoclingPageGeometry
) -> BoundingBoxV1 | None:
    bbox = getattr(cell, "bbox", None)
    if bbox is None:
        return None
    return _docling_normalized_bbox(bbox, geometry=geometry)


def _docling_table_cell_offsets(
    cell: object, *, rows: int, columns: int
) -> tuple[int, int, int, int]:
    values = (
        getattr(cell, "start_row_offset_idx", None),
        getattr(cell, "end_row_offset_idx", None),
        getattr(cell, "start_col_offset_idx", None),
        getattr(cell, "end_col_offset_idx", None),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise _docling_structured_output_invalid()
    start_row = cast(int, values[0])
    end_row = cast(int, values[1])
    start_column = cast(int, values[2])
    end_column = cast(int, values[3])
    if not (0 <= start_row < end_row <= rows and 0 <= start_column < end_column <= columns):
        raise _docling_structured_output_invalid()
    return start_row, end_row, start_column, end_column


def _docling_table_cells(
    *,
    item: object,
    table_index: int,
    page_number: int,
    geometry: _DoclingPageGeometry,
    spans_by_page: Mapping[int, list[DocumentSpanV1]],
) -> list[_DoclingStructuredTableCell]:
    """Read Docling's real grid and raw cells without Markdown reconstruction."""

    data = getattr(item, "data", None)
    rows = _docling_table_dimension(getattr(data, "num_rows", None))
    columns = _docling_table_dimension(getattr(data, "num_cols", None))
    if (rows == 0) != (columns == 0) or rows * columns > 10_000:
        raise _docling_structured_output_invalid()
    raw_cells = _docling_sequence(getattr(data, "table_cells", None))
    if len(raw_cells) > rows * columns:
        raise _docling_structured_output_invalid()
    raw_spans: list[tuple[tuple[int, int, int, int], str, BoundingBoxV1 | None]] = []
    raw_grid_cells: dict[tuple[int, int], str] = {}
    for raw_cell in raw_cells:
        offsets = _docling_table_cell_offsets(raw_cell, rows=rows, columns=columns)
        text = _docling_table_cell_text(raw_cell)
        bbox = _docling_table_cell_bbox(raw_cell, geometry=geometry)
        start_row, end_row, start_column, end_column = offsets
        for row_index in range(start_row, end_row):
            for column_index in range(start_column, end_column):
                coordinate = (row_index, column_index)
                if coordinate in raw_grid_cells:
                    raise _docling_structured_output_invalid()
                raw_grid_cells[coordinate] = text
        if text:
            raw_spans.append((offsets, text, bbox))

    grid = _docling_sequence(getattr(data, "grid", None))
    if len(grid) != rows:
        raise _docling_structured_output_invalid()
    cells: list[_DoclingStructuredTableCell] = []
    for row_index, row in enumerate(grid):
        row_values = _docling_sequence(row)
        if len(row_values) != columns:
            raise _docling_structured_output_invalid()
        for column_index, cell in enumerate(row_values):
            text = _docling_table_cell_text(cell)
            # The grid cell's bbox remains a required validity check even
            # though table-cell F1 retains only the coordinate/text tuple.
            _docling_table_cell_bbox(cell, geometry=geometry)
            if raw_grid_cells.get((row_index, column_index), "") != text:
                raise _docling_structured_output_invalid()
            if text:
                cells.append(
                    _DoclingStructuredTableCell(
                        page_number=page_number,
                        table_index=table_index,
                        row_index=row_index,
                        column_index=column_index,
                        text=text,
                    )
                )
    for _offsets, text, bbox in sorted(raw_spans, key=lambda value: value[0]):
        _docling_append_span(
            spans_by_page=spans_by_page,
            page_number=page_number,
            text=text,
            bbox=bbox,
        )
    return cells


def _extract_docling_structured_document(document: object) -> _DoclingStructuredExtraction:
    """Extract page-bound text and table cells from Docling's object model.

    This is intentionally a narrow, local-only bridge.  It reads no source
    truth, CSV, content stream, or Markdown representation.  Every emitted
    text value is tied to one Docling provenance record; malformed structure
    is an execution failure rather than an inferred page or table coordinate.
    """

    try:
        geometries = _docling_page_geometries(document)
        spans_by_page: dict[int, list[DocumentSpanV1]] = {
            page_number: [] for page_number in geometries
        }
        iterate_items = getattr(document, "iterate_items", None)
        if not callable(iterate_items):
            raise _docling_structured_output_invalid()
        table_index = 0
        table_cells: list[_DoclingStructuredTableCell] = []
        for entry in iterate_items():
            if (
                not isinstance(entry, Sequence)
                or isinstance(entry, (str, bytes))
                or len(entry) != 2
            ):
                raise _docling_structured_output_invalid()
            item = entry[0]
            if _is_docling_table_item(item):
                table_index += 1
                page_number, _ = _docling_single_provenance(item, geometries=geometries)
                table_cells.extend(
                    _docling_table_cells(
                        item=item,
                        table_index=table_index,
                        page_number=page_number,
                        geometry=geometries[page_number],
                        spans_by_page=spans_by_page,
                    )
                )
                continue
            text = getattr(item, "text", None)
            if text is None:
                continue
            if not isinstance(text, str):
                raise _docling_structured_output_invalid()
            if not text.strip():
                continue
            page_number, bbox = _docling_single_provenance(item, geometries=geometries)
            _docling_append_span(
                spans_by_page=spans_by_page,
                page_number=page_number,
                text=text,
                bbox=bbox,
            )
        pages = tuple(
            DocumentPreviewPageV1(
                page_number=geometry.page_number,
                width=geometry.width,
                height=geometry.height,
                coordinate_unit="pdf_points",
                spans=spans_by_page[geometry.page_number],
            )
            for geometry in (geometries[page_number] for page_number in sorted(geometries))
        )
        return _DoclingStructuredExtraction(pages=pages, table_cells=tuple(table_cells))
    except DocumentPreviewExecutionError:
        raise
    except Exception as error:
        raise _docling_structured_output_invalid() from error


def _pdf_page_dimensions(page: object) -> tuple[int, int]:
    try:
        mediabox = getattr(page, "mediabox")
        width = math.ceil(float(mediabox.width))
        height = math.ceil(float(mediabox.height))
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise DocumentPreviewInputError("PDF page dimensions are invalid") from error
    if not 1 <= width <= 100_000 or not 1 <= height <= 100_000:
        raise DocumentPreviewInputError("PDF page dimensions are invalid")
    return width, height


def _extract_pdf_text_layer_legacy(preview_input: PreviewInputV1) -> PreviewCandidateV1 | None:
    """Preserve the approved v1 aggregate ``pypdf`` extraction semantics.

    This path intentionally keeps the original ``page.extract_text()`` call
    and its whole-page ``_chunks`` projection.  It is separate from the v2
    visitor path because the two algorithms can produce different text spans
    for exactly the same PDF and must therefore never share an engine
    revision.
    """

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as error:
        raise DocumentPreviewUnavailable("pypdf is not provisioned") from error
    try:
        with local_files_only():
            reader = PdfReader(io.BytesIO(preview_input.content), strict=True)
            if reader.is_encrypted:
                raise DocumentPreviewInputError("encrypted PDF preview is unavailable")
            pages: list[DocumentPreviewPageV1] = []
            has_text = False
            for index, page in enumerate(reader.pages, start=1):
                width, height = _pdf_page_dimensions(page)
                extracted = page.extract_text() or ""
                spans = [
                    DocumentSpanV1(
                        span_id=f"page-{index}-span-{ordinal}",
                        ordinal=ordinal,
                        reading_order=ordinal,
                        text=text,
                    )
                    for ordinal, text in enumerate(_chunks(extracted), start=1)
                ]
                has_text = has_text or bool(spans)
                pages.append(
                    DocumentPreviewPageV1(
                        page_number=index,
                        width=width,
                        height=height,
                        coordinate_unit="pdf_points",
                        spans=spans,
                    )
                )
    except DocumentPreviewInputError:
        raise
    except Exception:
        # A parser exception can include source text.  Preserve the legacy
        # extraction output while keeping the existing content-free error
        # boundary for logs and transcripts.
        raise DocumentPreviewInputError("PDF text layer cannot be safely decoded") from None
    if not pages:
        raise DocumentPreviewInputError("PDF has no pages")
    if not has_text:
        return None
    return PreviewCandidateV1(
        status="preview_ready",
        extractor_revision=_PDF_TEXT_LAYER_LEGACY_EXTRACTOR_REVISION,
        pages=pages,
        warnings=[DocumentExtractionWarningV1(code="text_layer_used")],
    )


def _pdf_text_visitor_fragment(value: object) -> str | None:
    """Keep a visitor fragment verbatim except for its terminal line break.

    ``pypdf`` uses a trailing LF (and sometimes CRLF) to delimit text-showing
    operations.  It is an extraction artifact rather than content that belongs
    in the individual callback fragment.  Leading whitespace and all other
    characters remain untouched so this path cannot silently rewrite a native
    text layer.
    """

    if not isinstance(value, str):
        raise DocumentPreviewInputError("PDF text visitor output is invalid")
    trimmed = value.rstrip("\r\n")
    return trimmed or None


_PDF_TEXT_VISITOR_IDENTITY_MATRIX: Final[tuple[float, float, float, float, float, float]] = (
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
)


@dataclass(frozen=True, slots=True)
class _PdfTextLayerVisitorAnchor:
    """One internal-only normalized text start point, with top-origin ``y``."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _PdfTextLayerVisitorFragment:
    """One local visitor fragment before it is projected to public preview spans."""

    page_number: int
    ordinal: int
    text: str
    anchor: _PdfTextLayerVisitorAnchor | None


@dataclass(frozen=True, slots=True)
class _PdfTextLayerVisitorPage:
    """One internal-only page result shared by preview and candidate paths."""

    page_number: int
    width: int
    height: int
    fragments: tuple[_PdfTextLayerVisitorFragment, ...]


def _pdf_text_visitor_matrix(
    value: object,
) -> tuple[float, float, float, float, float, float] | None:
    """Accept one finite six-value PDF matrix without coercing arbitrary values."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        return None
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            return None
        numeric = float(coordinate)
        if not math.isfinite(numeric):
            return None
        coordinates.append(numeric)
    return cast(tuple[float, float, float, float, float, float], tuple(coordinates))


def _pdf_text_visitor_anchor(
    *, current_matrix: object, text_matrix: object, width: int, height: int
) -> _PdfTextLayerVisitorAnchor | None:
    """Return a strict local anchor without inventing a public text rectangle.

    ``pypdf`` gives the text matrix's start point, but only an identity CTM
    keeps that point directly aligned to the unrotated page coordinates used
    here.  Any other transformation, malformed matrix, or out-of-page point
    is deliberately omitted.  The internal anchor is never placed on the
    public ``DocumentSpanV1`` wire model.
    """

    current = _pdf_text_visitor_matrix(current_matrix)
    text = _pdf_text_visitor_matrix(text_matrix)
    if current != _PDF_TEXT_VISITOR_IDENTITY_MATRIX or text is None:
        return None
    x, y = text[4], text[5]
    if not 0 <= x <= width or not 0 <= y <= height:
        return None
    return _PdfTextLayerVisitorAnchor(x=x / width, y=(height - y) / height)


def _pdf_text_visitor_fragments(
    *, page: object, page_number: int, width: int, height: int
) -> tuple[_PdfTextLayerVisitorFragment, ...]:
    """Extract local callback fragments and only their proven start anchors.

    A start anchor belongs only to the first emitted chunk of one callback;
    later chunks have no proven advance and therefore receive ``None``.
    """

    fragments: list[_PdfTextLayerVisitorFragment] = []

    def visitor(
        text: object,
        current_matrix: object,
        text_matrix: object,
        _font_dictionary: object,
        _font_size: object,
    ) -> None:
        fragment = _pdf_text_visitor_fragment(text)
        if fragment is None:
            return
        anchor = _pdf_text_visitor_anchor(
            current_matrix=current_matrix,
            text_matrix=text_matrix,
            width=width,
            height=height,
        )
        for offset in range(0, len(fragment), 8_000):
            if len(fragments) >= 10_000:
                raise DocumentPreviewInputError("PDF text layer has too many fragments")
            ordinal = len(fragments) + 1
            fragments.append(
                _PdfTextLayerVisitorFragment(
                    page_number=page_number,
                    ordinal=ordinal,
                    text=fragment[offset : offset + 8_000],
                    anchor=anchor if offset == 0 else None,
                )
            )

    extract_text = getattr(page, "extract_text", None)
    if not callable(extract_text):
        raise DocumentPreviewInputError("PDF text layer cannot be safely decoded")
    extract_text(visitor_text=visitor)
    return tuple(fragments)


def _extract_pdf_text_layer_visitor_pages(
    preview_input: PreviewInputV1,
) -> tuple[_PdfTextLayerVisitorPage, ...]:
    """Run the one local pypdf visitor conversion shared by both consumers."""

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as error:
        raise DocumentPreviewUnavailable("pypdf is not provisioned") from error
    try:
        with local_files_only():
            reader = PdfReader(io.BytesIO(preview_input.content), strict=True)
            if reader.is_encrypted:
                raise DocumentPreviewInputError("encrypted PDF preview is unavailable")
            pages: list[_PdfTextLayerVisitorPage] = []
            for index, page in enumerate(reader.pages, start=1):
                width, height = _pdf_page_dimensions(page)
                pages.append(
                    _PdfTextLayerVisitorPage(
                        page_number=index,
                        width=width,
                        height=height,
                        fragments=_pdf_text_visitor_fragments(
                            page=page,
                            page_number=index,
                            width=width,
                            height=height,
                        ),
                    )
                )
    except DocumentPreviewInputError:
        raise
    except Exception:
        # A parser exception may include a raw text fragment.  Do not retain it
        # in the normal preview error chain where a transcript or general log
        # could expose document content.
        raise DocumentPreviewInputError("PDF text layer cannot be safely decoded") from None
    if not pages:
        raise DocumentPreviewInputError("PDF has no pages")
    return tuple(pages)


@dataclass(slots=True)
class PdfTextLayerPreviewAdapter:
    """Pinned local ``pypdf`` extraction selected by approved revision only."""

    extractor_revision: str

    def extract_pdf_text_layer_fragments(
        self, preview_input: PreviewInputV1
    ) -> tuple[_PdfTextLayerVisitorFragment, ...]:
        """Return local-only visitor output for candidate evaluation reuse.

        This deliberately does not change the public preview wire contract.
        It shares the exact pypdf conversion path with ``extract_pdf_text_layer``
        but retains a fail-closed, internal start anchor for a later local
        candidate projection.
        """

        if self.extractor_revision != _PDF_TEXT_LAYER_VISITOR_EXTRACTOR_REVISION:
            raise DocumentPreviewUnavailable(
                "PDF text visitor is unavailable for this extractor revision"
            )
        if preview_input.metadata.media_type != "application/pdf":
            return ()
        pages = _extract_pdf_text_layer_visitor_pages(preview_input)
        return tuple(fragment for page in pages for fragment in page.fragments)

    def extract_pdf_native_vector_grid(
        self, preview_input: PreviewInputV1
    ) -> "NativePdfVectorGridExtraction | None":
        """Return direct candidate table evidence for the explicit v1 profile only.

        This method is deliberately a separate seam from the visitor callback:
        a vector-grid profile never falls back to visitor geometry, and an
        older text-layer profile cannot accidentally acquire direct cells.
        """

        if self.extractor_revision != _PDF_TEXT_LAYER_NATIVE_VECTOR_GRID_EXTRACTOR_REVISION:
            return None
        if preview_input.metadata.media_type != "application/pdf":
            return None
        try:
            from customs_ai_agent.evaluation.native_pdf_vector_grid import (
                NativePdfVectorGridError,
                extract_native_pdf_vector_grid,
            )

            with local_files_only():
                return extract_native_pdf_vector_grid(preview_input=preview_input)
        except NativePdfVectorGridError:
            raise DocumentPreviewInputError(
                "PDF native vector grid cannot be safely decoded"
            ) from None
        except Exception:
            # Third-party parser exceptions can embed text-show operands.  Keep
            # the same content-free runtime boundary used by the visitor path.
            raise DocumentPreviewInputError(
                "PDF native vector grid cannot be safely decoded"
            ) from None

    def extract_pdf_text_layer(self, preview_input: PreviewInputV1) -> PreviewCandidateV1 | None:
        if self.extractor_revision not in _PDF_TEXT_LAYER_EXTRACTOR_REVISIONS:
            raise DocumentPreviewUnavailable("PDF text layer extractor revision is unavailable")
        if preview_input.metadata.media_type != "application/pdf":
            return None
        if self.extractor_revision == _PDF_TEXT_LAYER_LEGACY_EXTRACTOR_REVISION:
            return _extract_pdf_text_layer_legacy(preview_input)
        if self.extractor_revision == _PDF_TEXT_LAYER_NATIVE_VECTOR_GRID_EXTRACTOR_REVISION:
            extraction = self.extract_pdf_native_vector_grid(preview_input)
            if extraction is None or not extraction.spans:
                return None
            spans_by_page: dict[int, list[DocumentSpanV1]] = {
                page.page_number: [] for page in extraction.pages
            }
            for span in extraction.spans:
                spans_by_page[span.page_number].append(
                    DocumentSpanV1(
                        span_id=f"page-{span.page_number}-span-{span.reading_order}",
                        ordinal=span.reading_order,
                        reading_order=span.reading_order,
                        text=span.text,
                    )
                )
            return PreviewCandidateV1(
                status="preview_ready",
                extractor_revision=self.extractor_revision,
                pages=[
                    DocumentPreviewPageV1(
                        page_number=page.page_number,
                        width=page.width,
                        height=page.height,
                        coordinate_unit="pdf_points",
                        spans=spans_by_page[page.page_number],
                    )
                    for page in extraction.pages
                ],
                warnings=[DocumentExtractionWarningV1(code="text_layer_used")],
            )
        visitor_pages = _extract_pdf_text_layer_visitor_pages(preview_input)
        if not any(page.fragments for page in visitor_pages):
            return None
        return PreviewCandidateV1(
            status="preview_ready",
            extractor_revision=self.extractor_revision,
            pages=[
                DocumentPreviewPageV1(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    coordinate_unit="pdf_points",
                    spans=[
                        DocumentSpanV1(
                            span_id=f"page-{fragment.page_number}-span-{fragment.ordinal}",
                            ordinal=fragment.ordinal,
                            reading_order=fragment.ordinal,
                            text=fragment.text,
                        )
                        for fragment in page.fragments
                    ],
                )
                for page in visitor_pages
            ],
            warnings=[DocumentExtractionWarningV1(code="text_layer_used")],
        )


@dataclass(frozen=True, slots=True)
class _LegacyPdfTextLayerFallbackAdapter:
    """Keep human-approved raster profiles on their historic native-text path.

    A human approval for PaddleOCR or Docling historically reported its
    selected profile revision even when a native PDF text layer satisfied the
    request before raster OCR.  The parser is nevertheless explicitly fixed
    to text-layer v1 here; a raster profile revision must never select the
    visitor parser by accident.
    """

    extractor_revision: str

    def extract_pdf_text_layer(self, preview_input: PreviewInputV1) -> PreviewCandidateV1 | None:
        candidate = _extract_pdf_text_layer_legacy(preview_input)
        if candidate is None:
            return None
        return replace(candidate, extractor_revision=self.extractor_revision)


def _normalized_bbox(points: object, *, width: int, height: int) -> dict[str, float] | None:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return None
    coordinates: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            return None
        x, y = point
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
        ):
            return None
        coordinates.append((float(x), float(y)))
    if not coordinates:
        return None
    left = max(0.0, min(x for x, _ in coordinates) / width)
    top = max(0.0, min(y for _, y in coordinates) / height)
    right = min(1.0, max(x for x, _ in coordinates) / width)
    bottom = min(1.0, max(y for _, y in coordinates) / height)
    if right <= left or bottom <= top:
        return None
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


@dataclass(slots=True)
class PaddleOcrPdfPreviewAdapter:
    """Pinned local PaddleOCR PDF adapter; all model directories are explicit."""

    extractor_revision: str
    model_root: Path
    options: Mapping[str, object]
    _engine: object | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _paddle_engine(self) -> object:
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-not-found]
            except ImportError as error:
                raise DocumentPreviewUnavailable("PaddleOCR is not provisioned") from error
            try:
                kwargs: dict[str, object] = {
                    "det_model_dir": str(
                        self.model_root / str(self.options["det_model_directory"])
                    ),
                    "rec_model_dir": str(
                        self.model_root / str(self.options["rec_model_directory"])
                    ),
                    "cls_model_dir": str(
                        self.model_root / str(self.options["cls_model_directory"])
                    ),
                    "ocr_version": self.options["ocr_version"],
                    "use_gpu": False,
                    "cpu_threads": 1,
                    "lang": self.options["language"],
                    "use_angle_cls": self.options["use_angle_cls"],
                    "show_log": False,
                }
                # Every model directory was inventory-hashed before this
                # constructor is reached.  No default model path is accepted.
                with local_files_only():
                    self._engine = PaddleOCR(**kwargs)
            except Exception as error:
                raise DocumentPreviewUnavailable(
                    "PaddleOCR local model cannot be loaded"
                ) from error
            return self._engine

    def extract_raster_or_image(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        if preview_input.metadata.media_type != "application/pdf":
            raise DocumentPreviewUnavailable("PDF-only approval")
        try:
            import numpy as np  # type: ignore[import-not-found]
            import pypdfium2 as pdfium  # type: ignore[import-not-found]
        except ImportError as error:
            raise DocumentPreviewUnavailable("PaddleOCR renderer is not provisioned") from error
        engine: Any = self._paddle_engine()
        try:
            document = pdfium.PdfDocument(preview_input.content)
            pages: list[DocumentPreviewPageV1] = []
            has_text = False
            for index in range(len(document)):
                rendered = document[index].render(scale=1).to_pil().convert("RGB")
                width, height = rendered.size
                if width < 1 or height < 1 or width * height > 40_000_000:
                    raise DocumentPreviewInputError("PDF raster dimensions are invalid")
                with local_files_only():
                    output = engine.ocr(
                        np.asarray(rendered), cls=bool(self.options["use_angle_cls"])
                    )
                page_lines = output[0] if isinstance(output, list) and output else []
                spans: list[DocumentSpanV1] = []
                for ordinal, line in enumerate(page_lines, start=1):
                    if not isinstance(line, Sequence) or len(line) < 2:
                        continue
                    result = line[1]
                    if (
                        not isinstance(result, Sequence)
                        or not result
                        or not isinstance(result[0], str)
                    ):
                        continue
                    text = result[0].strip()
                    if not text:
                        continue
                    bbox_value = _normalized_bbox(line[0], width=width, height=height)
                    spans.append(
                        DocumentSpanV1(
                            span_id=f"page-{index + 1}-span-{len(spans) + 1}",
                            ordinal=len(spans) + 1,
                            reading_order=len(spans) + 1,
                            text=text[:8_000],
                            bbox=(
                                BoundingBoxV1.model_validate(bbox_value)
                                if bbox_value is not None
                                else None
                            ),
                        )
                    )
                has_text = has_text or bool(spans)
                pages.append(
                    DocumentPreviewPageV1(
                        page_number=index + 1,
                        width=width,
                        height=height,
                        coordinate_unit="pixels",
                        spans=spans,
                    )
                )
        except DocumentPreviewInputError:
            raise
        except DocumentPreviewUnavailable:
            raise
        except Exception as error:
            raise DocumentPreviewExecutionError("PaddleOCR execution failed") from error
        if not pages:
            raise DocumentPreviewInputError("PDF has no pages")
        return PreviewCandidateV1(
            status="preview_ready" if has_text else "needs_review",
            extractor_revision=self.extractor_revision,
            pages=pages,
            warnings=[] if has_text else [DocumentExtractionWarningV1(code="ocr_text_not_found")],
        )


def _paddleocr3_sequence(value: object) -> list[object]:
    """Convert an official P3 list/array without accepting arbitrary iterables."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    try:
        converted = tolist()
    except Exception as error:
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid") from error
    if not isinstance(converted, Sequence) or isinstance(converted, (str, bytes, bytearray)):
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    return list(converted)


def _strict_paddleocr3_normalized_bbox(
    polygon: object,
    *,
    width: int,
    height: int,
) -> BoundingBoxV1:
    """Validate a P3 ``rec_polys`` polygon before normalizing its rectangle."""

    if width < 1 or height < 1:
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    points = _paddleocr3_sequence(polygon)
    if not 4 <= len(points) <= 64:
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    coordinates: list[tuple[float, float]] = []
    for point in points:
        pair = _paddleocr3_sequence(point)
        if len(pair) != 2:
            raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
        x, y = pair
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
        x_float, y_float = float(x), float(y)
        if x_float < 0 or y_float < 0 or x_float > width or y_float > height:
            raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
        coordinates.append((x_float, y_float))
    left = min(x for x, _ in coordinates) / width
    top = min(y for _, y in coordinates) / height
    right = max(x for x, _ in coordinates) / width
    bottom = max(y for _, y in coordinates) / height
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    try:
        return BoundingBoxV1.model_validate(
            {"x": left, "y": top, "width": right - left, "height": bottom - top}
        )
    except ValidationError as error:
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid") from error


def _paddleocr3_page_spans(
    prediction: object,
    *,
    page_number: int,
    width: int,
    height: int,
) -> list[DocumentSpanV1]:
    """Translate exactly one P3 ``predict`` result to safe public spans."""

    if isinstance(prediction, Mapping) or isinstance(prediction, (str, bytes, bytearray)):
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    try:
        iterator = iter(prediction)  # type: ignore[call-overload]
        result = next(iterator)
    except (TypeError, StopIteration) as error:
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid") from error
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    if not isinstance(result, Mapping):
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    texts = _paddleocr3_sequence(result.get("rec_texts"))
    polygons = _paddleocr3_sequence(result.get("rec_polys"))
    if len(texts) != len(polygons):
        raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
    spans: list[DocumentSpanV1] = []
    for text_value, polygon in zip(texts, polygons, strict=True):
        if not isinstance(text_value, str):
            raise DocumentPreviewExecutionError("PaddleOCR 3 result is invalid")
        bbox = _strict_paddleocr3_normalized_bbox(polygon, width=width, height=height)
        text = text_value.strip()
        if not text:
            continue
        spans.append(
            DocumentSpanV1(
                span_id=f"page-{page_number}-span-{len(spans) + 1}",
                ordinal=len(spans) + 1,
                reading_order=len(spans) + 1,
                text=text[:8_000],
                bbox=bbox,
            )
        )
    return spans


def _windows_short_path(path: Path) -> str:
    """Return the Win32 8.3 spelling for an existing local path.

    PaddlePaddle 3.1's native inference loader on Windows cannot open a
    Unicode model-file path even though Python can.  Use Windows' own
    pre-existing short alias rather than creating a junction, copying a model,
    or accepting a lossy encoding.
    """

    try:
        import ctypes

        windll = getattr(ctypes, "windll", None)
        kernel32 = None if windll is None else getattr(windll, "kernel32", None)
        get_short_path_name = (
            None if kernel32 is None else getattr(kernel32, "GetShortPathNameW", None)
        )
        if not callable(get_short_path_name):
            raise OSError("GetShortPathNameW is unavailable")
        required = get_short_path_name(str(path), None, 0)
        if not isinstance(required, int) or required < 1:
            raise OSError("GetShortPathNameW failed")
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = get_short_path_name(str(path), buffer, len(buffer))
        if not isinstance(written, int) or written < 1 or written >= len(buffer):
            raise OSError("GetShortPathNameW failed")
    except (AttributeError, OSError) as error:
        raise DocumentPreviewUnavailable(
            "document_preview_paddleocr3_native_runtime_unavailable"
        ) from error
    if not buffer.value or not buffer.value.isascii():
        raise DocumentPreviewUnavailable("document_preview_paddleocr3_native_runtime_unavailable")
    return buffer.value


def _paddleocr3_native_model_directory(*, model_root: Path, relative_directory: str) -> str:
    """Resolve a P3 model directory into a native-safe local path.

    The configuration parser has already limited ``relative_directory`` to a
    relative local path.  Resolve both levels again at the native boundary so
    a missing directory or a link escaping the approved model root never
    reaches PaddleX.  On Windows there is deliberately no Unicode-path
    fallback: without an existing ASCII 8.3 alias, the profile is unavailable.
    """

    try:
        resolved_root = model_root.resolve(strict=True)
        resolved_directory = (resolved_root / relative_directory).resolve(strict=True)
        resolved_directory.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise DocumentPreviewUnavailable(
            "document_preview_paddleocr3_native_runtime_unavailable"
        ) from error
    if sys.platform != "win32":
        return str(resolved_directory)
    return _windows_short_path(resolved_directory)


@dataclass(slots=True)
class PaddleOcr3PdfPreviewAdapter:
    """Pinned offline PaddleOCR 3.2/PaddleX PDF adapter without P2 fallback."""

    extractor_revision: str
    model_root: Path
    options: Mapping[str, object]
    _engine: object | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _native_runtime_verified: bool = field(default=False, init=False, repr=False)
    _native_runtime_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def _paddle_engine(self) -> object:
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                kwargs: dict[str, object] = {
                    "text_detection_model_name": self.options["text_detection_model_name"],
                    "text_detection_model_dir": _paddleocr3_native_model_directory(
                        model_root=self.model_root,
                        relative_directory=cast(
                            str, self.options["text_detection_model_directory"]
                        ),
                    ),
                    "text_recognition_model_name": self.options["text_recognition_model_name"],
                    "text_recognition_model_dir": _paddleocr3_native_model_directory(
                        model_root=self.model_root,
                        relative_directory=cast(
                            str, self.options["text_recognition_model_directory"]
                        ),
                    ),
                    "device": "cpu",
                    "cpu_threads": 1,
                    "enable_hpi": False,
                    "enable_mkldnn": False,
                    "use_tensorrt": False,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                }
                # Import and construction can both trigger PaddleX's model
                # resolver.  The per-thread guard turns a missing local model
                # into an unavailable result rather than an implicit download.
                with local_files_only():
                    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

                    self._engine = PaddleOCR(**kwargs)
            except Exception as error:
                raise DocumentPreviewUnavailable(
                    "document_preview_paddleocr3_native_runtime_unavailable"
                ) from error
            return self._engine

    @staticmethod
    def _prediction_kwargs() -> dict[str, bool]:
        return {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

    def verify_native_runtime(self) -> None:
        """Run P3 once on a blank ndarray before any source PDF is loaded."""

        with self._native_runtime_lock:
            if self._native_runtime_verified:
                return
            try:
                import numpy as np  # type: ignore[import-not-found]

                blank = np.zeros((64, 64, 3), dtype=np.uint8)
                engine: Any = self._paddle_engine()
                with local_files_only():
                    prediction = engine.predict(blank, **self._prediction_kwargs())
                _paddleocr3_page_spans(
                    prediction,
                    page_number=1,
                    width=64,
                    height=64,
                )
            except Exception as error:
                raise DocumentPreviewUnavailable(
                    "document_preview_paddleocr3_native_runtime_unavailable"
                ) from error
            self._native_runtime_verified = True

    def extract_raster_or_image(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        if preview_input.metadata.media_type != "application/pdf":
            raise DocumentPreviewUnavailable("PDF-only approval")
        self.verify_native_runtime()
        try:
            import numpy as np  # type: ignore[import-not-found]
            import pypdfium2 as pdfium  # type: ignore[import-not-found]
        except ImportError as error:
            raise DocumentPreviewUnavailable("PaddleOCR renderer is not provisioned") from error
        engine: Any = self._paddle_engine()
        try:
            document = pdfium.PdfDocument(preview_input.content)
            pages: list[DocumentPreviewPageV1] = []
            has_text = False
            for index in range(len(document)):
                # The P3 profile parser pins raster_scale=3.  Do not fall
                # back to pdfium's default scale because the extractor
                # revision explicitly represents 216-dpi input.
                rendered = document[index].render(scale=3).to_pil().convert("RGB")
                width, height = rendered.size
                if width < 1 or height < 1 or width * height > 40_000_000:
                    raise DocumentPreviewInputError("PDF raster dimensions are invalid")
                with local_files_only():
                    prediction = engine.predict(
                        np.asarray(rendered), **self._prediction_kwargs()
                    )
                spans = _paddleocr3_page_spans(
                    prediction,
                    page_number=index + 1,
                    width=width,
                    height=height,
                )
                has_text = has_text or bool(spans)
                pages.append(
                    DocumentPreviewPageV1(
                        page_number=index + 1,
                        width=width,
                        height=height,
                        coordinate_unit="pixels",
                        spans=spans,
                    )
                )
        except DocumentPreviewInputError:
            raise
        except DocumentPreviewUnavailable:
            raise
        except DocumentPreviewExecutionError:
            raise
        except Exception as error:
            raise DocumentPreviewExecutionError("PaddleOCR 3 execution failed") from error
        if not pages:
            raise DocumentPreviewInputError("PDF has no pages")
        return PreviewCandidateV1(
            status="preview_ready" if has_text else "needs_review",
            extractor_revision=self.extractor_revision,
            pages=pages,
            warnings=[] if has_text else [DocumentExtractionWarningV1(code="ocr_text_not_found")],
        )


@dataclass(slots=True)
class DoclingPdfPreviewAdapter:
    """Pinned local Docling adapter using an in-memory PDF stream only."""

    extractor_revision: str
    model_root: Path
    options: Mapping[str, object]
    _converter: object | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _native_runtime_verified: bool = field(default=False, init=False, repr=False)
    _native_runtime_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def _docling_converter(self) -> object:
        with self._lock:
            if self.options.get("pdf_backend") != _DOCLING_PDF_BACKEND:
                raise DocumentPreviewUnavailable("Docling PDF backend configuration is invalid")
            table_structure_mode = self.options.get("table_structure_mode")
            if (
                not isinstance(table_structure_mode, str)
                or table_structure_mode not in _DOCLING_TABLE_STRUCTURE_MODES
            ):
                raise DocumentPreviewUnavailable("Docling table structure configuration is invalid")
            if self._converter is not None:
                return self._converter
            _require_docling_native_ascii_package_paths()
            try:
                from docling.backend.pypdfium2_backend import (  # type: ignore[import-not-found]
                    PyPdfiumDocumentBackend,
                )
                from docling.datamodel.base_models import DocumentStream, InputFormat  # type: ignore[import-not-found]
                from docling.datamodel.pipeline_options import (  # type: ignore[import-not-found]
                    AcceleratorDevice,
                    AcceleratorOptions,
                    PdfPipelineOptions,
                    TableFormerMode,
                )
                from docling.document_converter import (  # type: ignore[import-not-found]
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError as error:
                raise DocumentPreviewUnavailable("Docling is not provisioned") from error
            try:
                # Docling 2.15.1 provides this default at runtime, while its
                # Pydantic type stub treats ``ocr_options`` as required.
                pipeline = PdfPipelineOptions()  # type: ignore[call-arg]
                # The configuration parser permits only this fixed offline
                # Docling structure/table profile.  Do not let its default
                # EasyOCR cache or downloader escape the local inventory.
                pipeline.do_ocr = False
                pipeline.do_table_structure = True
                pipeline.table_structure_options.mode = (
                    TableFormerMode.FAST
                    if table_structure_mode == "fast"
                    else TableFormerMode.ACCURATE
                )
                pipeline.table_structure_options.do_cell_matching = True
                pipeline.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice.CPU, num_threads=1
                )
                # Current pinned Docling releases use this local artifact
                # directory.  If a future pinned API removes it, activation
                # fails closed instead of accepting a downloader default.
                if not hasattr(pipeline, "artifacts_path"):
                    raise DocumentPreviewUnavailable("Docling local artifact API is unavailable")
                pipeline.artifacts_path = str(
                    self.model_root / str(self.options["artifacts_directory"])
                )
                with local_files_only():
                    self._converter = (
                        DocumentConverter(
                            format_options={
                                InputFormat.PDF: PdfFormatOption(
                                    backend=PyPdfiumDocumentBackend,
                                    pipeline_options=pipeline,
                                )
                            }
                        ),
                        DocumentStream,
                    )
            except DocumentPreviewUnavailable:
                raise
            except Exception as error:
                raise DocumentPreviewUnavailable("Docling local model cannot be loaded") from error
            return self._converter

    @staticmethod
    def _native_probe_pdf_bytes() -> bytes:
        """Build a disposable blank PDF wholly in memory for native preflight."""

        try:
            from pypdf import PdfWriter  # type: ignore[import-not-found]
        except ImportError as error:
            raise DocumentPreviewUnavailable(
                "document_preview_docling_native_runtime_unavailable"
            ) from error
        stream = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(stream)
        return stream.getvalue()

    def _convert_pdf_bytes(self, content: bytes) -> Any:
        converter_value = self._docling_converter()
        if not isinstance(converter_value, tuple) or len(converter_value) != 2:
            raise DocumentPreviewUnavailable("Docling local converter is invalid")
        converter, document_stream_type = converter_value
        try:
            with local_files_only():
                return converter.convert(
                    document_stream_type(name="document.pdf", stream=io.BytesIO(content))
                )
        except DocumentPreviewUnavailable:
            raise
        except Exception as error:
            raise DocumentPreviewExecutionError("Docling execution failed") from error

    def verify_native_runtime(self) -> None:
        """Run the native Docling stack before any selected source PDF is read.

        The result is intentionally discarded.  It proves only that the
        exact local package/model runtime can convert a blank one-page PDF
        while sockets are denied; it neither evaluates nor records OCR text.
        """

        with self._native_runtime_lock:
            if self._native_runtime_verified:
                return
            try:
                _require_docling_native_ascii_package_paths()
                result = self._convert_pdf_bytes(self._native_probe_pdf_bytes())
                document = getattr(result, "document", None)
                pages = getattr(document, "pages", None)
                if not pages or len(pages) != 1:
                    raise ValueError("blank Docling PDF was not converted")
            except Exception as error:
                raise DocumentPreviewUnavailable(
                    "document_preview_docling_native_runtime_unavailable"
                ) from error
            self._native_runtime_verified = True

    def extract_structured_pdf(self, preview_input: PreviewInputV1) -> _DoclingStructuredExtraction:
        """Convert one PDF into the shared, page-bound Docling representation."""

        if preview_input.metadata.media_type != "application/pdf":
            raise DocumentPreviewUnavailable("PDF-only approval")
        self.verify_native_runtime()
        try:
            result = self._convert_pdf_bytes(preview_input.content)
            document = result.document
            return _extract_docling_structured_document(document)
        except DocumentPreviewUnavailable:
            raise
        except DocumentPreviewExecutionError:
            raise
        except Exception as error:
            raise DocumentPreviewExecutionError("Docling execution failed") from error

    def extract_raster_or_image(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        """Return the public preview projection of the shared structured result."""

        extraction = self.extract_structured_pdf(preview_input)
        pages = list(extraction.pages)
        has_text = any(page.spans for page in pages)
        return PreviewCandidateV1(
            status="preview_ready" if has_text else "needs_review",
            extractor_revision=self.extractor_revision,
            pages=pages,
            warnings=[] if has_text else [DocumentExtractionWarningV1(code="ocr_text_not_found")],
        )


def preflight_provisioned_docling_native_runtime(
    *, document_preview_root: Path, configuration: RuntimeConfigurationV1
) -> None:
    """Verify the explicit Docling profile without ever opening a source asset."""

    if configuration.engine_kind != "docling":
        return
    adapter = DoclingPdfPreviewAdapter(
        extractor_revision=configuration.extractor_revision,
        model_root=document_preview_root,
        options=configuration.options,
    )
    try:
        adapter.verify_native_runtime()
    except DocumentPreviewUnavailable as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_native_runtime_unavailable"
        ) from error


def preflight_provisioned_paddleocr3_native_runtime(
    *, document_preview_root: Path, configuration: RuntimeConfigurationV1
) -> None:
    """Verify an explicit PaddleOCR 3 profile without opening a source asset."""

    if (
        configuration.engine_kind != "paddleocr"
        or configuration.runtime_api != _PADDLEOCR3_RUNTIME_API
    ):
        return
    adapter = PaddleOcr3PdfPreviewAdapter(
        extractor_revision=configuration.extractor_revision,
        model_root=document_preview_root,
        options=configuration.options,
    )
    try:
        adapter.verify_native_runtime()
    except DocumentPreviewUnavailable as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_paddleocr3_native_runtime_unavailable"
        ) from error


@dataclass(slots=True)
class ApprovedPdfOnlyPreviewEngine:
    """Block images and runtime drift before invoking the selected local profile."""

    binding: ApprovedPdfPreviewBinding
    delegate: LocalDocumentPreviewEngine

    def extract(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        if preview_input.metadata.media_type != "application/pdf":
            raise DocumentPreviewUnavailable("PDF-only approval")
        try:
            self.binding.verify_runtime_state()
        except DocumentPreviewRuntimeError as error:
            raise DocumentPreviewUnavailable(error.code) from error
        candidate = self.delegate.extract(preview_input)
        if candidate.extractor_revision != self.binding.approval.extractor_revision:
            raise DocumentPreviewUnavailable("approved extractor revision changed")
        return candidate


class _AdapterFactory(Protocol):
    def __call__(self, binding: ApprovedPdfPreviewBinding) -> LocalDocumentPreviewEngine: ...


def build_default_pdf_preview_adapter(
    binding: ApprovedPdfPreviewBinding,
) -> LocalDocumentPreviewEngine:
    """Build exactly one approved PDF profile; there is no environment fallback.

    The source-attested synthetic evaluator measures PaddleOCR and Docling as
    *direct* profiles.  Serving such a pointer through the legacy text-layer
    fallback would silently serve a different extractor for born-digital
    PDFs, so that provenance is deliberately direct at runtime as well.
    Human-approved pointers retain the historic text-layer-first behavior.
    """

    revision = binding.approval.extractor_revision
    direct_source_attested_profile = isinstance(
        binding.approval, SourceAttestedSyntheticPdfPreviewActivationV1
    )
    if binding.approval.engine_kind == "pdf_text_layer":
        if revision == _PDF_TEXT_LAYER_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION:
            if binding.supervised_template_provenance_binding is None:
                raise DocumentPreviewRuntimeError(
                    "document_preview_supervised_template_provenance_invalid"
                )
            return _SupervisedNativeTemplatePdfEngine(
                extractor_revision=revision,
                provenance_binding=binding.supervised_template_provenance_binding,
            )
        return _PdfTextLayerEngine(PdfTextLayerPreviewAdapter(extractor_revision=revision))
    if binding.approval.engine_kind == "paddleocr":
        runtime_api = getattr(binding.configuration, "runtime_api", _PADDLEOCR2_RUNTIME_API)
        if runtime_api == _PADDLEOCR2_RUNTIME_API:
            paddle: PaddleOcrPdfPreviewAdapter | PaddleOcr3PdfPreviewAdapter = (
                PaddleOcrPdfPreviewAdapter(
                    extractor_revision=revision,
                    model_root=binding.root,
                    options=binding.configuration.options,
                )
            )
        elif runtime_api == _PADDLEOCR3_RUNTIME_API:
            paddle = PaddleOcr3PdfPreviewAdapter(
                extractor_revision=revision,
                model_root=binding.root,
                options=binding.configuration.options,
            )
        else:
            raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")
        if direct_source_attested_profile:
            return _DirectRasterOcrEngine(paddle)
        return LayeredLocalPreviewEngine(
            _LegacyPdfTextLayerFallbackAdapter(extractor_revision=revision), paddle
        )
    if binding.approval.engine_kind == "docling":
        docling = DoclingPdfPreviewAdapter(
            extractor_revision=revision,
            model_root=binding.root,
            options=binding.configuration.options,
        )
        if direct_source_attested_profile:
            return _DirectRasterOcrEngine(docling)
        return LayeredLocalPreviewEngine(
            _LegacyPdfTextLayerFallbackAdapter(extractor_revision=revision), docling
        )
    raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")


@dataclass(frozen=True, slots=True)
class _PdfTextLayerEngine:
    adapter: PdfTextLayerPreviewAdapter

    def extract(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        candidate = self.adapter.extract_pdf_text_layer(preview_input)
        if candidate is None:
            raise DocumentPreviewUnavailable("PDF text layer is unavailable")
        return candidate


@dataclass(frozen=True, slots=True)
class _SupervisedNativeTemplatePdfEngine:
    """Serve the exact learned r1 selector rather than a raw-vector fallback."""

    extractor_revision: str
    provenance_binding: "R1SupervisedTemplateProvenanceBinding"

    def extract(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        try:
            # Keep this import lazy: the semantic adapter imports the runtime
            # network guard, while the service imports this class at startup.
            from customs_ai_agent.evaluation.native_pdf_semantic_adapter import (
                NativePdfSemanticAdapter,
            )
            from customs_ai_agent.evaluation.supervised_template_contract import (
                R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS,
            )

            output = NativePdfSemanticAdapter(
                model=self.provenance_binding.verified_model()
            ).extract_pdf_bytes(preview_input.content)
        except Exception as error:
            raise DocumentPreviewUnavailable("supervised native template unavailable") from error
        if output.document_type not in R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS:
            raise DocumentPreviewUnavailable("supervised native template unavailable")
        expected_field_names = {
            field_name
            for field_name, _normalizer in R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS[
                cast(Literal["bill_of_lading", "invoice", "packing_list"], output.document_type)
            ]
        }
        observed_field_names = [field.field_name for field in output.fields]
        if (
            len(observed_field_names) != len(expected_field_names)
            or set(observed_field_names) != set(expected_field_names)
            or len(observed_field_names) != len(set(observed_field_names))
            or output.model_artifact_sha256
            != self.provenance_binding.expected_model_artifact_sha256
        ):
            raise DocumentPreviewUnavailable("supervised native template unavailable")
        spans_by_page: dict[int, list[DocumentSpanV1]] = {
            page.page_number: [] for page in output.pages
        }
        for span in output.spans:
            spans = spans_by_page.get(span.page_number)
            if spans is None:
                raise DocumentPreviewUnavailable("supervised native template unavailable")
            ordinal = len(spans) + 1
            spans.append(
                DocumentSpanV1(
                    span_id=f"page-{span.page_number}-span-{ordinal}",
                    ordinal=ordinal,
                    reading_order=ordinal,
                    text=span.text,
                    bbox=span.bbox,
                )
            )
        pages = [
            DocumentPreviewPageV1(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                coordinate_unit="pdf_points",
                spans=spans_by_page[page.page_number],
            )
            for page in output.pages
        ]
        if not pages or not any(page.spans for page in pages):
            raise DocumentPreviewUnavailable("supervised native template unavailable")
        return PreviewCandidateV1(
            status="preview_ready",
            extractor_revision=self.extractor_revision,
            pages=pages,
            warnings=[DocumentExtractionWarningV1(code="supervised_native_template_used")],
        )


@dataclass(frozen=True, slots=True)
class _DirectRasterOcrEngine:
    """Serve the exact direct PaddleOCR or Docling profile that was evaluated."""

    adapter: PaddleOcrPdfPreviewAdapter | PaddleOcr3PdfPreviewAdapter | DoclingPdfPreviewAdapter

    def extract(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        return self.adapter.extract_raster_or_image(preview_input)


@dataclass(slots=True)
class AutoActivatingSourceAttestedPdfPreviewEngine:
    """Adopt only a newly verified synthetic activation on a PDF request.

    The normal static path receives an approval hash from the process
    environment and intentionally never changes.  This separate, explicit
    local-only mode is for the source-attested synthetic workflow: it derives
    a fresh hash only from the one fixed active-pointer path, validates the
    complete immutable evidence chain, then swaps the cached delegate.  A
    missing, changed, or malformed pointer never falls back to a previously
    cached engine.
    """

    approval_path: Path
    adapter_factory: _AdapterFactory
    package_version_reader: Callable[[str], str]
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _active_pointer_sha256: str | None = field(default=None, init=False, repr=False)
    _active_engine: ApprovedPdfOnlyPreviewEngine | None = field(
        default=None, init=False, repr=False
    )

    def _clear_active(self) -> None:
        self._active_pointer_sha256 = None
        self._active_engine = None

    def _read_source_attested_pointer_snapshot(self) -> tuple[bytes, str]:
        """Read one exact canonical automatic pointer without accepting human shape."""

        code = "document_preview_source_attested_auto_pointer_invalid"
        # The settings layer enforces the lexical fixed path.  Repeat the
        # runtime root/symlink check before every read because the pointer is
        # deliberately allowed to appear after this process starts.
        _, canonical_approval_path = validate_approval_pointer_path(self.approval_path)
        raw = _read_regular_file(canonical_approval_path, limit=_MAX_POINTER_BYTES, code=code)
        value = _strict_json_object(raw, code=code)
        if raw != _canonical_json_bytes(value, trailing_newline=True):
            raise DocumentPreviewRuntimeError(code)
        parse_source_attested_synthetic_pdf_preview_activation(raw)
        return raw, _sha256_bytes(raw)

    def _current_engine(self) -> ApprovedPdfOnlyPreviewEngine:
        """Return the exact current pointer engine, or reject it before use."""

        with self._lock:
            initial_raw, pointer_sha256 = self._read_source_attested_pointer_snapshot()
            if self._active_engine is not None and self._active_pointer_sha256 == pointer_sha256:
                return self._active_engine

            # Pointer bytes changed.  Drop the previous profile before doing
            # any work so a new invalid pointer cannot keep serving old OCR.
            self._clear_active()
            binding = load_approved_pdf_preview_binding(
                approval_path=self.approval_path,
                approval_sha256=pointer_sha256,
                package_version_reader=self.package_version_reader,
            )
            if not isinstance(binding.approval, SourceAttestedSyntheticPdfPreviewActivationV1):
                raise DocumentPreviewRuntimeError(
                    "document_preview_source_attested_auto_pointer_invalid"
                )

            # A writer uses atomic replacement, but re-read after the complete
            # chain validation to reject a pointer that changed while loading.
            confirmed_raw, confirmed_sha256 = self._read_source_attested_pointer_snapshot()
            if confirmed_sha256 != pointer_sha256 or confirmed_raw != initial_raw:
                raise DocumentPreviewRuntimeError(
                    "document_preview_source_attested_auto_pointer_changed"
                )
            binding.verify_runtime_state()
            candidate = ApprovedPdfOnlyPreviewEngine(
                binding=binding,
                delegate=self.adapter_factory(binding),
            )
            self._active_pointer_sha256 = pointer_sha256
            self._active_engine = candidate
            return candidate

    def extract(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        if preview_input.metadata.media_type != "application/pdf":
            raise DocumentPreviewUnavailable("PDF-only source-attested activation")
        try:
            engine = self._current_engine()
            return engine.extract(preview_input)
        except DocumentPreviewRuntimeError as error:
            with self._lock:
                self._clear_active()
            raise DocumentPreviewUnavailable(error.code) from error
        except DocumentPreviewUnavailable:
            # This includes pointer/sidecar drift caught by the cached binding.
            # Do not retain an old profile after any unavailable outcome.
            with self._lock:
                self._clear_active()
            raise


@dataclass(frozen=True, slots=True)
class DocumentPreviewEngineResolution:
    engine: LocalDocumentPreviewEngine
    status: Literal["disabled", "approved", "auto-pending"]
    reason_code: str | None = None


def resolve_approved_pdf_preview_engine(
    *,
    approval_path: Path | None,
    approval_sha256: str | None,
    adapter_factory: _AdapterFactory = build_default_pdf_preview_adapter,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> DocumentPreviewEngineResolution:
    """Resolve an opt-in local approval without stopping the text-only service."""

    if approval_path is None and approval_sha256 is None:
        return DocumentPreviewEngineResolution(DisabledDocumentPreviewEngine(), "disabled")
    if approval_path is None or approval_sha256 is None:
        return DocumentPreviewEngineResolution(
            DisabledDocumentPreviewEngine(), "disabled", "document_preview_approval_incomplete"
        )
    try:
        binding = load_approved_pdf_preview_binding(
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            package_version_reader=package_version_reader,
        )
        return DocumentPreviewEngineResolution(
            ApprovedPdfOnlyPreviewEngine(binding=binding, delegate=adapter_factory(binding)),
            "approved",
        )
    except DocumentPreviewRuntimeError as error:
        return DocumentPreviewEngineResolution(
            DisabledDocumentPreviewEngine(), "disabled", error.code
        )


def resolve_document_preview_engine(
    *,
    approval_path: Path | None,
    approval_sha256: str | None,
    source_attested_auto_activation: bool,
    adapter_factory: _AdapterFactory = build_default_pdf_preview_adapter,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> DocumentPreviewEngineResolution:
    """Resolve either the preserved static mode or explicit dynamic synthetic mode."""

    if not source_attested_auto_activation:
        return resolve_approved_pdf_preview_engine(
            approval_path=approval_path,
            approval_sha256=approval_sha256,
            adapter_factory=adapter_factory,
            package_version_reader=package_version_reader,
        )
    if approval_path is None or approval_sha256 is not None:
        return DocumentPreviewEngineResolution(
            DisabledDocumentPreviewEngine(),
            "disabled",
            "document_preview_source_attested_auto_activation_incomplete",
        )
    return DocumentPreviewEngineResolution(
        AutoActivatingSourceAttestedPdfPreviewEngine(
            approval_path=approval_path,
            adapter_factory=adapter_factory,
            package_version_reader=package_version_reader,
        ),
        "auto-pending",
        "document_preview_source_attested_auto_pending",
    )


def write_runtime_sidecars(
    *,
    document_preview_root: Path,
    configuration_value: Mapping[str, object],
    inventory_value: Mapping[str, object],
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> tuple[str, str]:
    """Validate and immutably register one explicit local provisioning record."""

    root = validate_document_preview_root(document_preview_root, create=True)
    configuration_raw = _canonical_json_bytes(configuration_value, trailing_newline=True)
    inventory_raw = _canonical_json_bytes(inventory_value, trailing_newline=True)
    configuration = parse_runtime_configuration(configuration_raw)
    inventory = parse_model_inventory(inventory_raw)
    if not _configuration_inventory_profile_matches(configuration, inventory):
        raise DocumentPreviewRuntimeError("document_preview_provisioning_kind_mismatch")
    verify_inventory_packages(inventory, package_version_reader=package_version_reader)
    verify_configured_model_directory_artifacts(root, configuration, inventory)
    _load_r1_supervised_template_provenance_binding(
        root=root,
        configuration=configuration,
        inventory=inventory,
    )
    configuration_sha256 = _sha256_bytes(configuration_raw)
    inventory_sha256 = _sha256_bytes(inventory_raw)
    _write_immutable_json(
        root / "runtime-configurations" / f"{configuration_sha256}.json",
        configuration_value,
        code="document_preview_provisioning_write_failed",
    )
    _write_immutable_json(
        root / "model-inventories" / f"{inventory_sha256}.json",
        inventory_value,
        code="document_preview_provisioning_write_failed",
    )
    return configuration_sha256, inventory_sha256


def write_approved_pdf_preview_pointer(
    *,
    document_preview_root: Path,
    approval_id: str,
    evaluation_raw: bytes,
    configuration_sha256: str,
    model_inventory_sha256: str,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> tuple[str, PdfPreviewApprovalV1]:
    """Atomically activate one human-approved sealed PDF evaluation result."""

    root = validate_document_preview_root(document_preview_root, create=True)
    configuration_sha256 = _sha256(
        configuration_sha256, code="document_preview_approval_arguments_invalid"
    )
    model_inventory_sha256 = _sha256(
        model_inventory_sha256, code="document_preview_approval_arguments_invalid"
    )
    try:
        (root / "evaluation-results").mkdir(parents=True, exist_ok=True)
        _assert_no_symlink(
            root / "evaluation-results", code="document_preview_approval_write_failed"
        )
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_approval_write_failed") from error
    configuration_path = _sidecar_path(root, "runtime-configurations", configuration_sha256)
    inventory_path = _sidecar_path(root, "model-inventories", model_inventory_sha256)
    configuration_raw = _read_regular_file(
        configuration_path,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_configuration_invalid",
    )
    inventory_raw = _read_regular_file(
        inventory_path,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_inventory_invalid",
    )
    if (
        _sha256_bytes(configuration_raw) != configuration_sha256
        or _sha256_bytes(inventory_raw) != model_inventory_sha256
    ):
        raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")
    configuration = parse_runtime_configuration(configuration_raw)
    inventory = parse_model_inventory(inventory_raw)
    verify_inventory_packages(inventory, package_version_reader=package_version_reader)
    verify_configured_model_directory_artifacts(root, configuration, inventory)
    evaluation = parse_sealed_evaluation_result(evaluation_raw)
    if (
        configuration.engine_kind != evaluation.engine_kind
        or configuration.extractor_revision != evaluation.extractor_revision
        or inventory.engine_kind != evaluation.engine_kind
        or configuration_sha256 != evaluation.configuration_sha256
        or model_inventory_sha256 != evaluation.model_inventory_sha256
    ):
        raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")
    normalized_evaluation_raw = _canonical_json_bytes(
        _strict_json_object(evaluation_raw, code="document_preview_evaluation_invalid"),
        trailing_newline=True,
    )
    evaluation_path = root / "evaluation-results" / f"{evaluation.evaluation_result_sha256}.json"
    _write_exact_immutable_bytes(
        evaluation_path,
        normalized_evaluation_raw,
        limit=_MAX_EVALUATION_BYTES,
        code="document_preview_approval_write_failed",
    )
    if (
        _read_regular_file(
            evaluation_path,
            limit=_MAX_EVALUATION_BYTES,
            code="document_preview_approval_write_failed",
        )
        != normalized_evaluation_raw
    ):
        raise DocumentPreviewRuntimeError("document_preview_approval_write_failed")
    approval_value: dict[str, object] = {
        "schema_version": "1.0",
        "approval_id": _identifier(approval_id, code="document_preview_approval_arguments_invalid"),
        "status": "approved",
        "media_type": "application/pdf",
        "engine": {
            "kind": evaluation.engine_kind,
            "extractor_revision": evaluation.extractor_revision,
            "configuration_sha256": configuration_sha256,
            "model_inventory_sha256": model_inventory_sha256,
        },
        "evaluation": {
            "dataset_revision": evaluation.dataset_revision,
            "gold_receipt_sha256": evaluation.gold_receipt_sha256,
            "evaluation_result_sha256": evaluation.evaluation_result_sha256,
            "packet_kind": "sealed",
        },
    }
    approval_raw = _canonical_json_bytes(approval_value, trailing_newline=True)
    approval = parse_pdf_preview_approval(approval_raw)
    _check_evaluation_binding(approval, evaluation, configuration, inventory)
    destination = root / "approvals" / _APPROVAL_FILENAME
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink(destination.parent, code="document_preview_approval_write_failed")
        temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
        with temporary.open("xb") as stream:
            stream.write(approval_raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_approval_write_failed") from error
    finally:
        try:
            if "temporary" in locals() and temporary.exists():
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return _sha256_bytes(approval_raw), approval


def write_source_attested_synthetic_pdf_preview_pointer(
    *,
    document_preview_root: Path,
    evaluation_raw: bytes,
    configuration_sha256: str,
    model_inventory_sha256: str,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> tuple[str, SourceAttestedSyntheticPdfPreviewActivationV1]:
    """Atomically activate one passing source-attested synthetic result.

    This is not a shortcut around the human-approved path.  It accepts only a
    distinct, canonical sealed result that is bound to the synthetic PDF
    content-stream layout receipt.  A manual Gold result, a mixed provenance
    result, or any failed threshold remains unable to write the active pointer.
    """

    code = "document_preview_source_attested_activation_write_failed"
    root = validate_document_preview_root(document_preview_root, create=True)
    configuration_sha256 = _sha256(
        configuration_sha256, code="document_preview_source_attested_activation_arguments_invalid"
    )
    model_inventory_sha256 = _sha256(
        model_inventory_sha256,
        code="document_preview_source_attested_activation_arguments_invalid",
    )
    try:
        (root / "evaluation-results").mkdir(parents=True, exist_ok=True)
        _assert_no_symlink(root / "evaluation-results", code=code)
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    configuration_path = _sidecar_path(root, "runtime-configurations", configuration_sha256)
    inventory_path = _sidecar_path(root, "model-inventories", model_inventory_sha256)
    configuration_raw = _read_regular_file(
        configuration_path,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_configuration_invalid",
    )
    inventory_raw = _read_regular_file(
        inventory_path,
        limit=_MAX_SIDECAR_BYTES,
        code="document_preview_inventory_invalid",
    )
    if (
        _sha256_bytes(configuration_raw) != configuration_sha256
        or _sha256_bytes(inventory_raw) != model_inventory_sha256
    ):
        raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")
    configuration = parse_runtime_configuration(configuration_raw)
    inventory = parse_model_inventory(inventory_raw)
    verify_inventory_packages(inventory, package_version_reader=package_version_reader)
    verify_configured_model_directory_artifacts(root, configuration, inventory)
    _load_r1_supervised_template_provenance_binding(
        root=root,
        configuration=configuration,
        inventory=inventory,
    )
    evaluation = parse_source_attested_synthetic_sealed_evaluation_result(evaluation_raw)
    if (
        configuration.engine_kind != evaluation.engine_kind
        or configuration.extractor_revision != evaluation.extractor_revision
        or inventory.engine_kind != evaluation.engine_kind
        or configuration_sha256 != evaluation.configuration_sha256
        or model_inventory_sha256 != evaluation.model_inventory_sha256
    ):
        raise DocumentPreviewRuntimeError("document_preview_approval_binding_mismatch")
    # The automatic path can write nothing until its full layout → development
    # selection → sealed-run → one-shot-reservation evidence chain is present
    # and mutually bound.  It is deliberately reloaded again by service
    # startup and before every preview request.
    load_source_attested_synthetic_receipt_binding(root=root, evaluation=evaluation)
    normalized_evaluation_raw = _canonical_json_bytes(
        _strict_json_object(
            evaluation_raw, code="document_preview_source_attested_evaluation_invalid"
        ),
        trailing_newline=True,
    )
    evaluation_path = root / "evaluation-results" / f"{evaluation.evaluation_result_sha256}.json"
    _write_exact_immutable_bytes(
        evaluation_path,
        normalized_evaluation_raw,
        limit=_MAX_EVALUATION_BYTES,
        code=code,
    )
    if (
        _read_regular_file(evaluation_path, limit=_MAX_EVALUATION_BYTES, code=code)
        != normalized_evaluation_raw
    ):
        raise DocumentPreviewRuntimeError(code)
    activation_value: dict[str, object] = {
        "schema_version": "1.0",
        "activation_kind": "source_attested_synthetic",
        "status": "activated",
        "media_type": "application/pdf",
        "engine": {
            "kind": evaluation.engine_kind,
            "extractor_revision": evaluation.extractor_revision,
            "configuration_sha256": configuration_sha256,
            "model_inventory_sha256": model_inventory_sha256,
        },
        "evaluation": {
            "dataset_id": evaluation.dataset_id,
            "dataset_revision": evaluation.dataset_revision,
            "packet_kind": "sealed",
            "packet_receipt_sha256": evaluation.packet_receipt_sha256,
            "evaluation_scope": "synthetic_r1_supervised_native_template_compatibility",
            "layout_truth_kind": "pdf_content_stream_layout_v1",
            "layout_truth_receipt_sha256": evaluation.layout_truth_receipt_sha256,
            "layout_truth_parser_spec_sha256": evaluation.layout_truth_parser_spec_sha256,
            "development_selection_receipt_sha256": (
                evaluation.development_selection_receipt_sha256
            ),
            "sealed_run_receipt_sha256": evaluation.sealed_run_receipt_sha256,
            "evaluation_basis": "source_attested_synthetic",
            "evaluation_result_sha256": evaluation.evaluation_result_sha256,
        },
    }
    activation_raw = _canonical_json_bytes(activation_value, trailing_newline=True)
    activation = parse_source_attested_synthetic_pdf_preview_activation(activation_raw)
    _check_evaluation_binding(activation, evaluation, configuration, inventory)
    destination = root / "approvals" / _APPROVAL_FILENAME
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink(destination.parent, code=code)
        temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
        with temporary.open("xb") as stream:
            stream.write(activation_raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    finally:
        try:
            if "temporary" in locals() and temporary.exists():
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return _sha256_bytes(activation_raw), activation


__all__ = [
    "AutoActivatingSourceAttestedPdfPreviewEngine",
    "ApprovedPdfOnlyPreviewEngine",
    "ApprovedPdfPreviewBinding",
    "DoclingPdfPreviewAdapter",
    "DocumentPreviewEngineResolution",
    "DocumentPreviewRuntimeError",
    "ModelArtifactV1",
    "ModelInventoryV1",
    "ModelPackageV1",
    "PaddleOcr3PdfPreviewAdapter",
    "PaddleOcrPdfPreviewAdapter",
    "PdfPreviewActivationV1",
    "PdfPreviewApprovalV1",
    "PdfPreviewEvaluationResultV1",
    "PdfTextLayerPreviewAdapter",
    "RuntimeConfigurationV1",
    "SealedEvaluationResultV1",
    "SourceAttestedDevelopmentSelectionReceiptV1",
    "SourceAttestedLayoutTruthReceiptV1",
    "SourceAttestedSealedRunReceiptV1",
    "SourceAttestedSyntheticPdfPreviewActivationV1",
    "SourceAttestedSyntheticReceiptBindingV1",
    "SourceAttestedSyntheticSealedEvaluationResultV1",
    "build_default_pdf_preview_adapter",
    "load_approved_pdf_preview_binding",
    "load_source_attested_synthetic_receipt_binding",
    "local_files_only",
    "parse_model_inventory",
    "parse_pdf_preview_activation",
    "parse_pdf_preview_approval",
    "parse_pdf_preview_evaluation_result",
    "parse_runtime_configuration",
    "parse_sealed_evaluation_result",
    "parse_source_attested_synthetic_pdf_preview_activation",
    "parse_source_attested_synthetic_sealed_evaluation_result",
    "preflight_provisioned_docling_native_runtime",
    "preflight_provisioned_paddleocr3_native_runtime",
    "resolve_document_preview_engine",
    "resolve_approved_pdf_preview_engine",
    "validate_approval_pointer_path",
    "validate_document_preview_root",
    "verify_inventory_packages",
    "write_approved_pdf_preview_pointer",
    "write_runtime_sidecars",
    "write_source_attested_synthetic_pdf_preview_pointer",
]
