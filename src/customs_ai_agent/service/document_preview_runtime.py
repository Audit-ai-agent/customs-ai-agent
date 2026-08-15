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

import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import socket
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol, Sequence, cast
from urllib.parse import urlparse

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
_SOURCE_DEVELOPMENT_SELECTION_DIRECTORY = "source-attested-development-selections"
_SOURCE_SEALED_RUN_DIRECTORY = "source-attested-sealed-runs"
_SEALED_RESERVATION_DIRECTORY = "candidate-run-reservations"
_SOURCE_LAYOUT_RECORD_COUNTS: Mapping[str, int] = {"development": 60, "sealed": 20}
_FIELD_EXACT_MATCH_MIN = 0.99
_OCR_CER_MAX = 0.01
_TABLE_CELL_F1_MIN = 0.98
_NETWORK_GUARD_LOCK = threading.Lock()
_NETWORK_GUARD_STATE = threading.local()
_NETWORK_GUARD_INSTALLED = False
_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_SOCKET_CONNECT_EX = socket.socket.connect_ex

ApprovedPdfEngineKind = Literal["pdf_text_layer", "paddleocr", "docling"]


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


def _canonical_json_bytes(value: object, *, trailing_newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_newline else b"")


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


def _assert_no_symlink(path: Path, *, code: str) -> None:
    """Reject a symlink in any existing component from its volume root."""

    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.exists() and current.is_symlink():
                raise DocumentPreviewRuntimeError(code)
        except OSError as error:
            raise DocumentPreviewRuntimeError(code) from error


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
    license_id: str
    source_uri: str


@dataclass(frozen=True, slots=True)
class ModelInventoryV1:
    engine_kind: ApprovedPdfEngineKind
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
    evaluation_scope: Literal["synthetic_content_stream_compatibility"]
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
    evaluation_scope: Literal["synthetic_content_stream_compatibility"]
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"]
    layout_truth_receipt_sha256: str
    layout_truth_parser_spec_sha256: str
    development_selection_receipt_sha256: str
    sealed_run_receipt_sha256: str
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    evaluation_result_sha256: str
    field_exact_match: float
    ocr_cer: float
    table_cell_f1: float


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
    parser_spec_sha256: str
    records: tuple[SourceAttestedLayoutRecordV1, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAttestedDevelopmentCandidateResultV1:
    engine_kind: ApprovedPdfEngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    evaluation_result_sha256: str
    passed: bool


@dataclass(frozen=True, slots=True)
class SourceAttestedDevelopmentSelectionReceiptV1:
    dataset_id: str
    dataset_revision: str
    development_packet_receipt_sha256: str
    layout_truth_receipt_sha256: str
    layout_truth_parser_spec_sha256: str
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
    sealed_candidate_reservation_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAttestedSyntheticReceiptBindingV1:
    development_layout_truth_path: Path
    development_layout_truth: SourceAttestedLayoutTruthReceiptV1
    layout_truth_path: Path
    layout_truth: SourceAttestedLayoutTruthReceiptV1
    development_selection_path: Path
    development_selection: SourceAttestedDevelopmentSelectionReceiptV1
    sealed_run_path: Path
    sealed_run: SourceAttestedSealedRunReceiptV1
    sealed_reservation_path: Path


PdfPreviewActivationV1 = PdfPreviewApprovalV1 | SourceAttestedSyntheticPdfPreviewActivationV1
PdfPreviewEvaluationResultV1 = (
    SealedEvaluationResultV1 | SourceAttestedSyntheticSealedEvaluationResultV1
)


def _parse_options(value: object, *, kind: ApprovedPdfEngineKind) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
    options = cast(dict[str, object], value)
    if kind == "pdf_text_layer":
        if options:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        return {}
    if kind == "paddleocr":
        allowed = {
            "det_model_directory",
            "rec_model_directory",
            "cls_model_directory",
            "language",
            "use_angle_cls",
        }
        if (
            not {"det_model_directory", "rec_model_directory"}.issubset(options)
            or set(options) - allowed
        ):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        normalized: dict[str, object] = {
            "det_model_directory": _relative_model_path(
                options["det_model_directory"], code="document_preview_configuration_invalid"
            ),
            "rec_model_directory": _relative_model_path(
                options["rec_model_directory"], code="document_preview_configuration_invalid"
            ),
            "language": "en",
            "use_angle_cls": False,
        }
        if "cls_model_directory" in options:
            normalized["cls_model_directory"] = _relative_model_path(
                options["cls_model_directory"], code="document_preview_configuration_invalid"
            )
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
    if kind == "docling":
        allowed = {"artifacts_directory", "ocr_enabled"}
        if "artifacts_directory" not in options or set(options) - allowed:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        ocr_enabled = options.get("ocr_enabled", True)
        if not isinstance(ocr_enabled, bool):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        return {
            "artifacts_directory": _relative_model_path(
                options["artifacts_directory"], code="document_preview_configuration_invalid"
            ),
            "ocr_enabled": ocr_enabled,
        }
    raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")


def parse_runtime_configuration(raw: bytes) -> RuntimeConfigurationV1:
    value = _strict_json_object(raw, code="document_preview_configuration_invalid")
    _require_exact_keys(
        value,
        {"schema_version", "engine_kind", "extractor_revision", "options"},
        code="document_preview_configuration_invalid",
    )
    if value["schema_version"] != "1.0":
        raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
    kind = _engine_kind(value["engine_kind"], code="document_preview_configuration_invalid")
    return RuntimeConfigurationV1(
        engine_kind=kind,
        extractor_revision=_identifier(
            value["extractor_revision"], code="document_preview_configuration_invalid"
        ),
        options=_parse_options(value["options"], kind=kind),
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
    _require_exact_keys(
        item,
        {"artifact_id", "relative_path", "sha256", "byte_size", "license_id", "source_uri"},
        code="document_preview_inventory_invalid",
    )
    byte_size = item["byte_size"]
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or not 1 <= byte_size <= 8 * 1024**3
    ):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
    return ModelArtifactV1(
        artifact_id=_identifier(item["artifact_id"], code="document_preview_inventory_invalid"),
        relative_path=_relative_model_path(
            item["relative_path"], code="document_preview_inventory_invalid"
        ),
        sha256=_sha256(item["sha256"], code="document_preview_inventory_invalid"),
        byte_size=byte_size,
        license_id=_identifier(item["license_id"], code="document_preview_inventory_invalid"),
        source_uri=_validate_https_source(
            item["source_uri"], code="document_preview_inventory_invalid"
        ),
    )


def parse_model_inventory(raw: bytes) -> ModelInventoryV1:
    value = _strict_json_object(raw, code="document_preview_inventory_invalid")
    _require_exact_keys(
        value,
        {"schema_version", "engine_kind", "packages", "artifacts"},
        code="document_preview_inventory_invalid",
    )
    if (
        value["schema_version"] != "1.0"
        or not isinstance(value["packages"], list)
        or not isinstance(value["artifacts"], list)
    ):
        raise DocumentPreviewRuntimeError("document_preview_inventory_invalid")
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
        engine_kind=_engine_kind(value["engine_kind"], code="document_preview_inventory_invalid"),
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
        or evaluation["evaluation_scope"] != "synthetic_content_stream_compatibility"
        or evaluation["layout_truth_kind"] != "pdf_content_stream_layout_v1"
    ):
        raise DocumentPreviewRuntimeError(code)
    return SourceAttestedSyntheticPdfPreviewActivationV1(
        engine_kind=_engine_kind(engine["kind"], code=code),
        extractor_revision=_identifier(engine["extractor_revision"], code=code),
        configuration_sha256=_sha256(engine["configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(engine["model_inventory_sha256"], code=code),
        dataset_id=_identifier(evaluation["dataset_id"], code=code),
        dataset_revision=_identifier(evaluation["dataset_revision"], code=code),
        packet_receipt_sha256=_sha256(evaluation["packet_receipt_sha256"], code=code),
        evaluation_scope=cast(
            Literal["synthetic_content_stream_compatibility"], evaluation["evaluation_scope"]
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
    if raw != _canonical_json_bytes(value, trailing_newline=True):
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
            "evaluation_result_sha256",
            "passed",
            "metrics",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["evaluation_basis"] != "source_attested_synthetic"
        or value["packet_kind"] != "sealed"
        or value["evaluation_scope"] != "synthetic_content_stream_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
        or value["passed"] is not True
    ):
        raise DocumentPreviewRuntimeError(code)
    metrics_value = value["metrics"]
    if not isinstance(metrics_value, dict):
        raise DocumentPreviewRuntimeError(code)
    metrics = cast(dict[str, object], metrics_value)
    _require_exact_keys(
        metrics,
        {"field_exact_match", "ocr_cer", "table_cell_f1"},
        code=code,
    )
    embedded_sha = _sha256(value["evaluation_result_sha256"], code=code)
    unsigned = dict(value)
    unsigned.pop("evaluation_result_sha256")
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != embedded_sha:
        raise DocumentPreviewRuntimeError(code)
    field_exact_match = _finite_number(metrics["field_exact_match"], code=code)
    ocr_cer = _finite_number(metrics["ocr_cer"], code=code)
    table_cell_f1 = _finite_number(metrics["table_cell_f1"], code=code)
    if not (
        0.0 <= field_exact_match <= 1.0 and 0.0 <= ocr_cer <= 1.0 and 0.0 <= table_cell_f1 <= 1.0
    ):
        raise DocumentPreviewRuntimeError(code)
    if (
        field_exact_match < _FIELD_EXACT_MATCH_MIN
        or ocr_cer > _OCR_CER_MAX
        or table_cell_f1 < _TABLE_CELL_F1_MIN
    ):
        raise DocumentPreviewRuntimeError("document_preview_source_attested_evaluation_gate_failed")
    return SourceAttestedSyntheticSealedEvaluationResultV1(
        dataset_id=_identifier(value["dataset_id"], code=code),
        dataset_revision=_identifier(value["dataset_revision"], code=code),
        packet_receipt_sha256=_sha256(value["packet_receipt_sha256"], code=code),
        evaluation_scope=cast(
            Literal["synthetic_content_stream_compatibility"], value["evaluation_scope"]
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
        engine_kind=_engine_kind(value["engine_kind"], code=code),
        extractor_revision=_identifier(value["extractor_revision"], code=code),
        configuration_sha256=_sha256(value["configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(value["model_inventory_sha256"], code=code),
        evaluation_result_sha256=embedded_sha,
        field_exact_match=field_exact_match,
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
        or value["evaluation_scope"] != "synthetic_content_stream_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
        or value["packet_kind"] != "stratified-v2"
        or not isinstance(value["packet_split"], str)
        or value["packet_split"] not in _SOURCE_LAYOUT_RECORD_COUNTS
        or not isinstance(value["source_manifest_mismatch_count"], int)
        or isinstance(value["source_manifest_mismatch_count"], bool)
        or value["source_manifest_mismatch_count"] != 0
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
            "evaluation_result_sha256",
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
        evaluation_result_sha256=_sha256(candidate["evaluation_result_sha256"], code=code),
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
        or value["evaluation_scope"] != "synthetic_content_stream_compatibility"
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
    development_packet_receipt_sha256 = _sha256(
        value["development_packet_receipt_sha256"], code=code
    )
    layout_truth_receipt_sha256 = _sha256(value["layout_truth_receipt_sha256"], code=code)
    candidates = tuple(
        _source_attested_development_candidate(item) for item in value["candidate_results"]
    )
    if (
        len(candidates) != len(_ENGINE_KINDS)
        or {item.engine_kind for item in candidates} != _ENGINE_KINDS
    ):
        raise DocumentPreviewRuntimeError(code)
    selected = SourceAttestedDevelopmentCandidateResultV1(
        engine_kind=_engine_kind(value["selected_engine_kind"], code=code),
        extractor_revision=_identifier(value["selected_extractor_revision"], code=code),
        configuration_sha256=_sha256(value["selected_configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(value["selected_model_inventory_sha256"], code=code),
        evaluation_result_sha256=_sha256(value["selected_evaluation_result_sha256"], code=code),
        passed=True,
    )
    if selected not in candidates:
        raise DocumentPreviewRuntimeError(code)
    return SourceAttestedDevelopmentSelectionReceiptV1(
        dataset_id=_identifier(value["dataset_id"], code=code),
        dataset_revision=_identifier(value["dataset_revision"], code=code),
        development_packet_receipt_sha256=development_packet_receipt_sha256,
        layout_truth_receipt_sha256=layout_truth_receipt_sha256,
        layout_truth_parser_spec_sha256=_sha256(
            value["layout_truth_parser_spec_sha256"], code=code
        ),
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
            "sealed_candidate_reservation_sha256",
            "sealed_run_receipt_sha256",
        },
        code=code,
    )
    if (
        value["schema_version"] != "1.0"
        or value["artifact_kind"] != "source_attested_synthetic_pdf_sealed_run"
        or value["evaluation_basis"] != "source_attested_synthetic"
        or value["evaluation_scope"] != "synthetic_content_stream_compatibility"
        or value["layout_truth_kind"] != "pdf_content_stream_layout_v1"
    ):
        raise DocumentPreviewRuntimeError(code)
    _verify_self_excluding_receipt_digest(
        value=value,
        field_name="sealed_run_receipt_sha256",
        expected_digest=receipt_sha256,
        code=code,
    )
    return SourceAttestedSealedRunReceiptV1(
        dataset_id=_identifier(value["dataset_id"], code=code),
        dataset_revision=_identifier(value["dataset_revision"], code=code),
        sealed_packet_receipt_sha256=_sha256(value["sealed_packet_receipt_sha256"], code=code),
        layout_truth_receipt_sha256=_sha256(value["layout_truth_receipt_sha256"], code=code),
        layout_truth_parser_spec_sha256=_sha256(
            value["layout_truth_parser_spec_sha256"], code=code
        ),
        development_selection_receipt_sha256=_sha256(
            value["development_selection_receipt_sha256"], code=code
        ),
        engine_kind=_engine_kind(value["selected_engine_kind"], code=code),
        extractor_revision=_identifier(value["selected_extractor_revision"], code=code),
        configuration_sha256=_sha256(value["selected_configuration_sha256"], code=code),
        model_inventory_sha256=_sha256(value["selected_model_inventory_sha256"], code=code),
        selected_development_evaluation_result_sha256=_sha256(
            value["selected_development_evaluation_result_sha256"], code=code
        ),
        candidate_bundle_sha256=_sha256(value["candidate_bundle_sha256"], code=code),
        candidate_parser_revision=_identifier(value["candidate_parser_revision"], code=code),
        sealed_candidate_reservation_sha256=_sha256(
            value["sealed_candidate_reservation_sha256"], code=code
        ),
        receipt_sha256=receipt_sha256,
    )


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
    if raw != _canonical_json_bytes(value, trailing_newline=True):
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
        or value["parser_revision"] != sealed_run.candidate_parser_revision
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


def load_source_attested_synthetic_receipt_binding(
    *,
    root: Path,
    evaluation: SourceAttestedSyntheticSealedEvaluationResultV1,
) -> SourceAttestedSyntheticReceiptBindingV1:
    """Load the complete immutable evidence chain for automatic activation."""

    layout_path, _layout_raw, layout_value = _read_canonical_hash_named_receipt(
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
    selection_path, _selection_raw, selection_value = _read_canonical_hash_named_receipt(
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
    development_layout_path, _development_layout_raw, development_layout_value = (
        _read_canonical_hash_named_receipt(
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
    sealed_run_path, _sealed_run_raw, sealed_run_value = _read_canonical_hash_named_receipt(
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
        or selection.layout_truth_parser_spec_sha256 != evaluation.layout_truth_parser_spec_sha256
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
        or {record.asset_id for record in development_layout_truth.records}
        & {record.asset_id for record in layout_truth.records}
        or {record.source_sha256 for record in development_layout_truth.records}
        & {record.source_sha256 for record in layout_truth.records}
    ):
        raise DocumentPreviewRuntimeError(
            "document_preview_source_attested_receipt_binding_mismatch"
        )
    reservation_path = _verify_source_attested_sealed_reservation(root=root, sealed_run=sealed_run)
    return SourceAttestedSyntheticReceiptBindingV1(
        development_layout_truth_path=development_layout_path,
        development_layout_truth=development_layout_truth,
        layout_truth_path=layout_path,
        layout_truth=layout_truth,
        development_selection_path=selection_path,
        development_selection=selection,
        sealed_run_path=sealed_run_path,
        sealed_run=sealed_run,
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


def _required_packages(kind: ApprovedPdfEngineKind) -> frozenset[str]:
    if kind == "pdf_text_layer":
        return frozenset({"pypdf"})
    if kind == "paddleocr":
        return frozenset({"pypdf", "paddleocr", "paddlepaddle", "pypdfium2"})
    return frozenset({"pypdf", "docling"})


def _normalized_distribution_name(name: str) -> str:
    return name.casefold().replace("_", "-")


def verify_inventory_packages(
    inventory: ModelInventoryV1,
    *,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> None:
    packages = {_normalized_distribution_name(item.name): item for item in inventory.packages}
    if not _required_packages(inventory.engine_kind).issubset(packages):
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


def _configuration_directories_exist(
    root: Path, configuration: RuntimeConfigurationV1, inventory: ModelInventoryV1
) -> None:
    if configuration.engine_kind == "pdf_text_layer":
        return
    configured = [
        value
        for key, value in configuration.options.items()
        if key.endswith("_directory") and isinstance(value, str)
    ]
    artifact_paths = [item.relative_path for item in inventory.artifacts]
    for relative in configured:
        path = _artifact_path(root, relative)
        try:
            if not path.is_dir() or path.is_symlink():
                raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")
        except OSError as error:
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid") from error
        prefix = f"{relative.rstrip('/')}/"
        if not any(item == relative or item.startswith(prefix) for item in artifact_paths):
            raise DocumentPreviewRuntimeError("document_preview_configuration_invalid")


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
    artifact_fingerprints: tuple[_ArtifactFingerprint, ...]

    def verify_runtime_state(self) -> None:
        """Fail closed on pointer/sidecar drift without rehashing huge models.

        Model artifacts are fully rehashed at activation.  Request-time checks
        compare their regular-file size and nanosecond mtime to the approved
        fingerprints; pointer/config/inventory/evaluation sidecars are small
        and are re-read and rehashed on every preview request.
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
    _configuration_directories_exist(root, configuration, inventory)
    fingerprints = _verify_inventory_artifacts(root, inventory)
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
        artifact_fingerprints=fingerprints,
    )


def _chunks(text: str, *, size: int = 8_000) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    return [normalized[index : index + size] for index in range(0, len(normalized), size)]


def _docling_preview_pages(
    *, page_numbers: Sequence[int], markdown_text: str
) -> list[DocumentPreviewPageV1]:
    """Preserve every bounded Docling markdown chunk without inventing pages.

    ``export_to_markdown`` is document-global in the pinned Docling API, so it
    does not safely expose a page boundary for every chunk.  Assigning all
    chunks to the first admitted page is conservative and retains the complete
    candidate text; assigning later chunks to guessed pages would corrupt
    page/span provenance.  Table cells remain unavailable until an adapter
    exposes real structured coordinates.
    """

    chunks = _chunks(markdown_text)
    return [
        DocumentPreviewPageV1(
            page_number=page_number,
            width=1,
            height=1,
            coordinate_unit="pdf_points",
            spans=(
                [
                    DocumentSpanV1(
                        span_id=f"page-1-span-{ordinal}",
                        ordinal=ordinal,
                        reading_order=ordinal,
                        text=chunk,
                    )
                    for ordinal, chunk in enumerate(chunks, start=1)
                ]
                if page_number == 1
                else []
            ),
        )
        for page_number in page_numbers
    ]


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


@dataclass(slots=True)
class PdfTextLayerPreviewAdapter:
    """Pinned local ``pypdf`` extraction with no raster/OCR fallback."""

    extractor_revision: str

    def extract_pdf_text_layer(self, preview_input: PreviewInputV1) -> PreviewCandidateV1 | None:
        if preview_input.metadata.media_type != "application/pdf":
            return None
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as error:
            raise DocumentPreviewUnavailable("pypdf is not provisioned") from error
        try:
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
        except Exception as error:
            raise DocumentPreviewInputError("PDF text layer cannot be safely decoded") from error
        if not pages:
            raise DocumentPreviewInputError("PDF has no pages")
        if not has_text:
            return None
        return PreviewCandidateV1(
            status="preview_ready",
            extractor_revision=self.extractor_revision,
            pages=pages,
            warnings=[DocumentExtractionWarningV1(code="text_layer_used")],
        )


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
                    "lang": self.options["language"],
                    "use_angle_cls": self.options["use_angle_cls"],
                    "show_log": False,
                }
                if "cls_model_directory" in self.options:
                    kwargs["cls_model_dir"] = str(
                        self.model_root / str(self.options["cls_model_directory"])
                    )
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


@dataclass(slots=True)
class DoclingPdfPreviewAdapter:
    """Pinned local Docling adapter using an in-memory PDF stream only."""

    extractor_revision: str
    model_root: Path
    options: Mapping[str, object]
    _converter: object | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _docling_converter(self) -> object:
        with self._lock:
            if self._converter is not None:
                return self._converter
            try:
                from docling.datamodel.base_models import DocumentStream, InputFormat  # type: ignore[import-not-found]
                from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore[import-not-found]
                from docling.document_converter import (  # type: ignore[import-not-found]
                    DocumentConverter,
                    PdfFormatOption,
                )
            except ImportError as error:
                raise DocumentPreviewUnavailable("Docling is not provisioned") from error
            try:
                pipeline = PdfPipelineOptions()
                pipeline.do_ocr = bool(self.options["ocr_enabled"])
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
                                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)
                            }
                        ),
                        DocumentStream,
                    )
            except DocumentPreviewUnavailable:
                raise
            except Exception as error:
                raise DocumentPreviewUnavailable("Docling local model cannot be loaded") from error
            return self._converter

    def extract_raster_or_image(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        if preview_input.metadata.media_type != "application/pdf":
            raise DocumentPreviewUnavailable("PDF-only approval")
        converter_value = self._docling_converter()
        if not isinstance(converter_value, tuple) or len(converter_value) != 2:
            raise DocumentPreviewUnavailable("Docling local converter is invalid")
        converter, document_stream_type = converter_value
        try:
            with local_files_only():
                result = converter.convert(
                    document_stream_type(
                        name="document.pdf", stream=io.BytesIO(preview_input.content)
                    )
                )
            document = result.document
            page_values = getattr(document, "pages", {})
            page_numbers = sorted(int(number) for number in page_values) if page_values else [1]
            pages = _docling_preview_pages(
                page_numbers=page_numbers,
                markdown_text=str(document.export_to_markdown()),
            )
        except DocumentPreviewUnavailable:
            raise
        except Exception as error:
            raise DocumentPreviewExecutionError("Docling execution failed") from error
        if not pages or page_numbers != list(range(1, len(page_numbers) + 1)):
            raise DocumentPreviewInputError("Docling PDF pages are invalid")
        has_text = any(page.spans for page in pages)
        return PreviewCandidateV1(
            status="preview_ready" if has_text else "needs_review",
            extractor_revision=self.extractor_revision,
            pages=pages,
            warnings=[] if has_text else [DocumentExtractionWarningV1(code="ocr_text_not_found")],
        )


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
    """Build exactly one approved PDF profile; there is no environment fallback."""

    revision = binding.approval.extractor_revision
    text_layer = PdfTextLayerPreviewAdapter(extractor_revision=revision)
    if binding.approval.engine_kind == "pdf_text_layer":
        return _PdfTextLayerEngine(text_layer)
    if binding.approval.engine_kind == "paddleocr":
        return LayeredLocalPreviewEngine(
            text_layer,
            PaddleOcrPdfPreviewAdapter(
                extractor_revision=revision,
                model_root=binding.root,
                options=binding.configuration.options,
            ),
        )
    if binding.approval.engine_kind == "docling":
        return LayeredLocalPreviewEngine(
            text_layer,
            DoclingPdfPreviewAdapter(
                extractor_revision=revision,
                model_root=binding.root,
                options=binding.configuration.options,
            ),
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
class DocumentPreviewEngineResolution:
    engine: LocalDocumentPreviewEngine
    status: Literal["disabled", "approved"]
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
    if configuration.engine_kind != inventory.engine_kind:
        raise DocumentPreviewRuntimeError("document_preview_provisioning_kind_mismatch")
    verify_inventory_packages(inventory, package_version_reader=package_version_reader)
    _configuration_directories_exist(root, configuration, inventory)
    _verify_inventory_artifacts(root, inventory)
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
    _configuration_directories_exist(root, configuration, inventory)
    _verify_inventory_artifacts(root, inventory)
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
    _configuration_directories_exist(root, configuration, inventory)
    _verify_inventory_artifacts(root, inventory)
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
            "evaluation_scope": "synthetic_content_stream_compatibility",
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
    "ApprovedPdfOnlyPreviewEngine",
    "ApprovedPdfPreviewBinding",
    "DoclingPdfPreviewAdapter",
    "DocumentPreviewEngineResolution",
    "DocumentPreviewRuntimeError",
    "ModelArtifactV1",
    "ModelInventoryV1",
    "ModelPackageV1",
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
    "resolve_approved_pdf_preview_engine",
    "validate_approval_pointer_path",
    "validate_document_preview_root",
    "verify_inventory_packages",
    "write_approved_pdf_preview_pointer",
    "write_runtime_sidecars",
    "write_source_attested_synthetic_pdf_preview_pointer",
]
