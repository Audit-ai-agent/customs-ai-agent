"""Deterministic, local-only scaffolding for ``synthetic-raster-v1``.

The compiler intentionally receives final-PDF content-stream geometry and a
renderer result as separate inputs.  It never uses OCR/VLM output, manifests,
or JSON fixture values as Gold.  A real PDFium renderer can be supplied later;
the pure functions below remain testable without optional OCR/PDFium packages.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    ContextManager,
    Final,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.schemas.document_extraction import BoundingBoxV1
from customs_ai_agent.schemas.raster_document_extraction import RasterDocumentExtractionResultV2


_SHA256 = r"^[a-f0-9]{64}$"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
# The original static raster profile used the bare label.  Automatic raster
# activation derives a distinct profile identity from the immutable offline
# bundle fingerprint.  Keep this narrow rather than accepting arbitrary
# profile names: bundle/execution/runtime provenance then proves that the
# accepted candidate belongs to the exact active fingerprint.
_SYNTHETIC_RASTER_PROFILE_REVISION = re.compile(
    r"^(?:synthetic-raster-v1|synthetic-raster-v1:[a-f0-9]{24})$"
)
RasterProfileId = Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
RasterSplit = Literal["train", "development", "calibration", "sealed"]


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PdfBoundingBoxV1(_WireModel):
    """PDF points in lower-left origin coordinates from an independent parser."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class SourceTextGeometryV1(_WireModel):
    span_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    page_number: int = Field(ge=1, le=1_000)
    text: str = Field(min_length=1, max_length=8_000)
    # The content-stream parser may know only the text origin.  Gold geometry
    # remains absent until the separate PDFium observation is matched 1:1.
    pdf_bbox: PdfBoundingBoxV1 | None = None
    content_stream_x: float | None = Field(default=None, ge=0)
    content_stream_y: float | None = Field(default=None, ge=0)
    reading_order: int = Field(ge=1, le=100_000)

    @model_validator(mode="after")
    def _content_stream_origin_is_complete(self) -> "SourceTextGeometryV1":
        if (self.content_stream_x is None) != (self.content_stream_y is None):
            raise ValueError("source text origin must include both coordinates")
        return self


class SourceFieldGoldV1(_WireModel):
    """Field truth from the final content stream, anchored to source spans."""

    field_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    value: str = Field(min_length=1, max_length=8_000)
    evidence_span_ids: list[str] = Field(min_length=1, max_length=128)


class SourceTableCellGoldV1(_WireModel):
    """Table-cell truth from the final content stream.

    ``pdf_bbox`` is the authored table-cell rectangle, not a union of the
    glyph boxes contained in the cell.  The latter is useful OCR evidence, but
    would turn a text-span detector into a false table-cell success during the
    table gate.
    """

    table_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    row_index: int = Field(ge=0, le=100_000)
    column_index: int = Field(ge=0, le=100_000)
    text: str = Field(min_length=1, max_length=8_000)
    pdf_bbox: PdfBoundingBoxV1
    evidence_span_ids: list[str] = Field(min_length=1, max_length=128)


class SourcePdfGeometryV1(_WireModel):
    """Source-attested final PDF content stream geometry, never OCR output."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    family_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    source_pdf_sha256: str = Field(pattern=_SHA256)
    page_width_points: float = Field(gt=0)
    page_height_points: float = Field(gt=0)
    source_truth_kind: Literal["final_pdf_content_stream"] = "final_pdf_content_stream"
    spans: list[SourceTextGeometryV1] = Field(min_length=1, max_length=10_000)
    fields: list[SourceFieldGoldV1] = Field(default_factory=list, max_length=512)
    table_cells: list[SourceTableCellGoldV1] = Field(default_factory=list, max_length=10_000)

    @model_validator(mode="after")
    def _source_geometry_is_consistent(self) -> "SourcePdfGeometryV1":
        if any(
            span.pdf_bbox is not None
            and (
                span.pdf_bbox.x + span.pdf_bbox.width > self.page_width_points
                or span.pdf_bbox.y + span.pdf_bbox.height > self.page_height_points
            )
            for span in self.spans
        ):
            raise ValueError("source PDF geometry escapes page bounds")
        if any(
            span.content_stream_x is not None
            and (
                span.content_stream_x > self.page_width_points
                or span.content_stream_y is None
                or span.content_stream_y > self.page_height_points
            )
            for span in self.spans
        ):
            raise ValueError("source text origin escapes page bounds")
        if len({span.span_id for span in self.spans}) != len(self.spans):
            raise ValueError("source geometry span IDs must be unique")
        span_ids = {span.span_id for span in self.spans}
        if any(not set(field.evidence_span_ids) <= span_ids for field in self.fields):
            raise ValueError("source field evidence must refer to source spans")
        if any(not set(cell.evidence_span_ids) <= span_ids for cell in self.table_cells):
            raise ValueError("source table-cell evidence must refer to source spans")
        if any(
            cell.pdf_bbox.x + cell.pdf_bbox.width > self.page_width_points
            or cell.pdf_bbox.y + cell.pdf_bbox.height > self.page_height_points
            for cell in self.table_cells
        ):
            raise ValueError("source table-cell geometry escapes page bounds")
        if len({field.field_id for field in self.fields}) != len(self.fields):
            raise ValueError("source field IDs must be unique")
        cell_positions = [
            (cell.table_id, cell.row_index, cell.column_index) for cell in self.table_cells
        ]
        if len(cell_positions) != len(set(cell_positions)):
            raise ValueError("source table-cell positions must be unique")
        return self


class PdfiumGeometrySpanV1(_WireModel):
    """Independent PDFium geometry observation for a final PDF text span."""

    page_number: int = Field(ge=1, le=1_000)
    text: str = Field(min_length=1, max_length=8_000)
    pdf_bbox: PdfBoundingBoxV1
    reading_order: int = Field(ge=1, le=100_000)
    source_pdf_sha256: str = Field(pattern=_SHA256)


class RasterizedOutputV1(_WireModel):
    """One already-rendered output passed in-memory from a pinned renderer."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    profile_id: RasterProfileId
    media_type: Literal["image/png", "image/jpeg", "application/pdf"]
    content_sha256: str = Field(pattern=_SHA256)
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)

    @model_validator(mode="after")
    def _profile_media_type_is_fixed(self) -> "RasterizedOutputV1":
        expected = RASTER_PROFILES[self.profile_id].media_type
        if self.media_type != expected:
            raise ValueError("raster output media type does not match profile")
        return self


class RasterProfileV1(_WireModel):
    profile_id: RasterProfileId
    media_type: Literal["image/png", "image/jpeg", "application/pdf"]
    dpi: int = Field(ge=72, le=600)
    jpeg_quality: int | None = Field(default=None, ge=1, le=100)
    image_only_pdf: bool = False


RASTER_PROFILES: Mapping[RasterProfileId, RasterProfileV1] = {
    "png-216": RasterProfileV1(profile_id="png-216", media_type="image/png", dpi=216),
    "jpeg-216-q75": RasterProfileV1(
        profile_id="jpeg-216-q75", media_type="image/jpeg", dpi=216, jpeg_quality=75
    ),
    "jpeg-150-q75": RasterProfileV1(
        profile_id="jpeg-150-q75", media_type="image/jpeg", dpi=150, jpeg_quality=75
    ),
    "scan-pdf-216-q75": RasterProfileV1(
        profile_id="scan-pdf-216-q75",
        media_type="application/pdf",
        dpi=216,
        jpeg_quality=75,
        image_only_pdf=True,
    ),
}


# These names are deliberately fixed instead of being supplied as dataset CLI
# arguments.  The input manifest can point only to local configuration files;
# it cannot redirect a supposedly sealed build to another source tree.
SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH: Final[Literal["통관서류_합성데이터셋_2000세트"]] = (
    "통관서류_합성데이터셋_2000세트"
)
SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH = "synthetic-raster-v1/source-receipt.json"
SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH = "synthetic-raster-v1/synthetic-raster-receipt.json"
SYNTHETIC_RASTER_BUILD_INPUTS_RELATIVE_PATH = "synthetic-raster-build-v2/inputs.json"
_SYNTHETIC_RASTER_BUILD_DIRECTORY_RELATIVE_PATH = "synthetic-raster-build-v2"
_SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME = "synthetic-raster-v1"
_SYNTHETIC_RASTER_STAGING_PREFIX = ".synthetic-raster-v1-staging-"
_SYNTHETIC_RASTER_BUILD_LOCK_NAME = ".synthetic-raster-v1-build.lock"
_SYNTHETIC_RASTER_SOURCE_DATASET_ID = "synthetic-customs-pdf-2000"
_FINAL_PDF_FONT_RESOURCE_POLICY: Final[Literal["final_pdf_font_resources_only"]] = (
    "final_pdf_font_resources_only"
)
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_MAX_LOCAL_BUILD_INPUT_BYTES = 1024 * 1024
_MAX_LOCAL_RASTER_ARTIFACT_BYTES = 32 * 1024 * 1024
_CONTENT_STREAM_ORIGIN_X_TOLERANCE_POINTS: Final[float] = 10.0
_CONTENT_STREAM_ORIGIN_Y_TOLERANCE_POINTS: Final[float] = 4.0
_PDF_FONT_RESOURCE_REFERENCE_RE = re.compile(
    rb"/(?P<resource>F[1-9][0-9]{0,2})\s+"
    rb"(?P<object>[1-9][0-9]*)\s+(?P<generation>[0-9]+)\s+R"
)
_PDF_INDIRECT_OBJECT_RE = re.compile(
    rb"(?P<object>[1-9][0-9]*)\s+(?P<generation>[0-9]+)\s+obj\s*"
    rb"(?P<body>.*?)\s*endobj",
    re.DOTALL,
)
_PDF_BASE_FONT_RE = re.compile(rb"/BaseFont\s*/(?P<font>[A-Za-z0-9-]+)(?=\s|/|>)")
_PDF_ENCODING_RE = re.compile(rb"/Encoding\s*/(?P<encoding>[A-Za-z0-9-]+)(?=\s|/|>)")
_WIN_ANSI_BASE_FONTS = frozenset({"Helvetica", "Helvetica-Bold", "Helvetica-Oblique"})
# The only non-WinAnsi font in the fixed synthetic B/L template.  The PDF
# character codes are not Unicode: they are the final PDF's ZapfDingbats
# codes.  This narrow table is deliberately limited to the glyphs observed
# in every fixed source PDF; a new glyph/template must fail rather than be
# guessed from OCR, VLM, an older JSON, or a host font.
_FIXED_ZAPF_DINGBATS_UNICODE_BY_CODE: Mapping[int, str] = {
    110: "\u25a0",
    172: "\u2460",
    173: "\u2461",
    174: "\u2462",
    175: "\u2463",
    176: "\u2464",
    177: "\u2465",
    178: "\u2466",
    179: "\u2467",
    180: "\u2468",
    181: "\u2469",
}


def _validate_document_preview_relative_path(value: str, *, label: str) -> str:
    """Accept one normalized POSIX path below the document-preview root only."""

    try:
        path = PurePosixPath(value)
    except TypeError:
        raise ValueError(f"{label} must be a local relative path") from None
    if (
        not value
        or "\\" in value
        or ":" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized local relative path")
    return value


class SyntheticRasterBuildInputsV1(_WireModel):
    """The only caller-provided build input: paths local to document-preview.

    Hashes are intentionally absent.  The builder reads the exact local bytes
    at these paths and records their SHA-256 values itself.  With ``extra``
    forbidden, a manifest cannot smuggle an alternative source receipt, host
    font location, network URL, or caller-selected hash into the chain.
    """

    schema_version: Literal["1.0"] = "1.0"
    source_root_relative_path: str = Field(min_length=1, max_length=240)
    renderer_configuration_relative_path: str = Field(min_length=1, max_length=240)
    font_manifest_relative_path: str = Field(min_length=1, max_length=240)
    split_seed_relative_path: str = Field(min_length=1, max_length=240)
    render_recipe_relative_path: str = Field(min_length=1, max_length=240)
    render_seed_relative_path: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def _uses_only_fixed_local_build_inputs(self) -> "SyntheticRasterBuildInputsV1":
        paths = {
            "source root": self.source_root_relative_path,
            "renderer configuration": self.renderer_configuration_relative_path,
            "font manifest": self.font_manifest_relative_path,
            "split seed": self.split_seed_relative_path,
            "render recipe": self.render_recipe_relative_path,
            "render seed": self.render_seed_relative_path,
        }
        for label, value in paths.items():
            _validate_document_preview_relative_path(value, label=label)
        if self.source_root_relative_path != SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH:
            raise ValueError("synthetic raster source root is not the fixed 6,000-PDF tree")
        if len(set(paths.values())) != len(paths):
            raise ValueError("synthetic raster build input paths must be distinct")
        return self


class SyntheticRasterRendererConfigurationV1(_WireModel):
    """Pinned PDFium renderer semantics read from a local configuration file."""

    schema_version: Literal["1.0"] = "1.0"
    renderer_kind: Literal["pdfium"] = "pdfium"
    renderer_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    pdfium_version: str = Field(min_length=1, max_length=128)
    geometry_compiler_revision: Literal[
        "source-content-stream-geometry-v1",
        "source-content-stream-geometry-v2-font-encoding",
    ] = "source-content-stream-geometry-v1"
    coordinate_transform: Literal["pdf-points-lower-left-to-normalized-top-left-v1"] = (
        "pdf-points-lower-left-to-normalized-top-left-v1"
    )
    profiles: list[RasterProfileV1] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _pins_the_actual_renderer_semantics(self) -> "SyntheticRasterRendererConfigurationV1":
        configured = {
            profile.profile_id: profile.model_dump(mode="json") for profile in self.profiles
        }
        expected = {
            profile_id: profile.model_dump(mode="json")
            for profile_id, profile in RASTER_PROFILES.items()
        }
        if len(configured) != len(self.profiles) or configured != expected:
            raise ValueError("renderer configuration does not bind the fixed raster profiles")
        return self


class SyntheticRasterFontManifestV1(_WireModel):
    """Forbid host-font inputs; PDFium receives only final PDF bytes.

    The present source PDFs legitimately use Base14 font resources, so this is
    not a ``/FontFile`` requirement.  Their resource declarations are bound by
    each source-PDF SHA.  The manifest is deliberately a one-field policy:
    file paths, URLs, inline font bytes, and fallback controls are rejected.
    """

    schema_version: Literal["1.0"] = "1.0"
    font_resource_policy: Literal["final_pdf_font_resources_only"] = _FINAL_PDF_FONT_RESOURCE_POLICY


@dataclass(frozen=True, slots=True)
class ResolvedSyntheticRasterBuildInputsV1:
    """Manifest values after every path is resolved and every byte hash is read."""

    document_preview_root: Path
    source_root: Path
    build_inputs_relative_path: str
    build_inputs_sha256: str
    renderer_configuration: SyntheticRasterRendererConfigurationV1
    renderer_configuration_sha256: str
    font_manifest: SyntheticRasterFontManifestV1
    font_manifest_sha256: str
    split_seed_sha256: str
    render_recipe_sha256: str
    render_seed_sha256: str


_RASTER_ASSET_SUFFIX: Mapping[RasterProfileId, str] = {
    "png-216": ".png",
    "jpeg-216-q75": ".jpg",
    "jpeg-150-q75": ".jpg",
    "scan-pdf-216-q75": ".pdf",
}


def _raster_asset_relative_path(*, asset_id: str, profile_id: RasterProfileId) -> str:
    """Return the only permitted local byte path for a raster receipt row."""

    return f"synthetic-raster-v1/assets/{asset_id}-{profile_id}{_RASTER_ASSET_SUFFIX[profile_id]}"


def _resolve_document_preview_root(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_dir()
            or resolved.is_symlink()
            or tuple(part.casefold() for part in resolved.parts[-3:])
            != _DOCUMENT_PREVIEW_ROOT_SUFFIX
        ):
            raise ValueError("document preview root is invalid")
    except OSError as error:
        raise ValueError("document preview root is invalid") from error
    return resolved


def _resolve_document_preview_local_path(
    *,
    document_preview_root: Path,
    relative_path: str,
    require_directory: bool,
) -> Path:
    """Resolve one local manifest path without following a symlink component."""

    _validate_document_preview_relative_path(relative_path, label="build input")
    candidate = document_preview_root
    try:
        for part in PurePosixPath(relative_path).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("synthetic raster build input path is invalid")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(document_preview_root)
        if (
            resolved.is_symlink()
            or (require_directory and not resolved.is_dir())
            or (not require_directory and not resolved.is_file())
        ):
            raise ValueError("synthetic raster build input path is invalid")
    except (OSError, ValueError):
        raise ValueError("synthetic raster build input path is invalid") from None
    return resolved


def _read_local_document_preview_bytes(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("synthetic raster build input is invalid")
        value = path.read_bytes()
    except OSError:
        raise ValueError("synthetic raster build input is invalid") from None
    if not value or len(value) > _MAX_LOCAL_BUILD_INPUT_BYTES:
        raise ValueError("synthetic raster build input is invalid")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _local_output_path_below_document_preview_root(
    *, document_preview_root: Path, relative_path: str
) -> Path:
    """Create checked parent directories for one fixed local output path.

    This helper is deliberately narrower than a generic file writer.  It is
    used only for the fixed build-input files below the Git-excluded preview
    root and rejects a linked component before and after directory creation.
    """

    _validate_document_preview_relative_path(relative_path, label="local output")
    candidate = document_preview_root
    parts = PurePosixPath(relative_path).parts
    for part in parts[:-1]:
        candidate = candidate / part
        try:
            if candidate.exists() and (candidate.is_symlink() or not candidate.is_dir()):
                raise ValueError("synthetic raster local output path is invalid")
            candidate.mkdir(mode=0o700, exist_ok=True)
            if candidate.is_symlink() or not candidate.is_dir():
                raise ValueError("synthetic raster local output path is invalid")
        except OSError:
            raise ValueError("synthetic raster local output path is invalid") from None
    destination = candidate / parts[-1]
    if destination.is_symlink():
        raise ValueError("synthetic raster local output path is invalid")
    return destination


def _write_or_verify_fixed_local_file(
    *, document_preview_root: Path, relative_path: str, payload: bytes
) -> None:
    """Create one fixed input once, or prove an existing file has same bytes."""

    if not payload or len(payload) > _MAX_LOCAL_BUILD_INPUT_BYTES:
        raise ValueError("synthetic raster fixed build input is invalid")
    destination = _local_output_path_below_document_preview_root(
        document_preview_root=document_preview_root, relative_path=relative_path
    )
    try:
        if destination.exists():
            if _read_local_document_preview_bytes(destination) != payload:
                raise ValueError("synthetic raster fixed build input conflicts with local bytes")
            return
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # A concurrent invocation may have completed exactly the same fixed
        # write.  Re-read rather than treating that benign race as success.
        if _read_local_document_preview_bytes(destination) != payload:
            raise ValueError("synthetic raster fixed build input conflicts with local bytes")
        return
    except OSError:
        raise ValueError("synthetic raster fixed build input is invalid") from None
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


def _fixed_synthetic_raster_build_input_payloads(*, pdfium_version: str) -> Mapping[str, bytes]:
    """Return the only permitted six local inputs for a raster v1 build."""

    if not pdfium_version or len(pdfium_version) > 128:
        raise SyntheticRasterCompilerUnavailable("pinned PDFium runtime is unavailable")
    directory = _SYNTHETIC_RASTER_BUILD_DIRECTORY_RELATIVE_PATH
    renderer_configuration = {
        "schema_version": "1.0",
        "renderer_kind": "pdfium",
        "renderer_revision": "pdfium-raster-v2-font-encoding",
        "pdfium_version": pdfium_version,
        "geometry_compiler_revision": "source-content-stream-geometry-v2-font-encoding",
        "coordinate_transform": "pdf-points-lower-left-to-normalized-top-left-v1",
        "profiles": [profile.model_dump(mode="json") for profile in RASTER_PROFILES.values()],
    }
    build_inputs = {
        "schema_version": "1.0",
        "source_root_relative_path": SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH,
        "renderer_configuration_relative_path": f"{directory}/renderer.json",
        "font_manifest_relative_path": f"{directory}/font.json",
        "split_seed_relative_path": f"{directory}/split.seed",
        "render_recipe_relative_path": f"{directory}/recipe.json",
        "render_seed_relative_path": f"{directory}/render.seed",
    }
    return {
        SYNTHETIC_RASTER_BUILD_INPUTS_RELATIVE_PATH: _canonical_json_bytes(build_inputs),
        f"{directory}/renderer.json": _canonical_json_bytes(renderer_configuration),
        f"{directory}/font.json": _canonical_json_bytes(
            SyntheticRasterFontManifestV1().model_dump(mode="json")
        ),
        f"{directory}/split.seed": b"synthetic-raster-v1-family-split-seed\n",
        f"{directory}/recipe.json": _canonical_json_bytes(
            {
                "schema_version": "1.0",
                "renderer_kind": "pdfium",
                "profiles": [profile.profile_id for profile in RASTER_PROFILES.values()],
                "png_optimize": False,
                "jpeg_quality": 75,
                "jpeg_progressive": False,
                "jpeg_subsampling": 0,
            }
        ),
        f"{directory}/render.seed": b"synthetic-raster-v1-render-seed\n",
    }


def ensure_fixed_synthetic_raster_build_inputs(
    *, document_preview_root: Path
) -> ResolvedSyntheticRasterBuildInputsV1:
    """Create-or-verify the six fixed build inputs without caller hashes.

    The PDFium version is read from the provisioned local environment before
    writing anything.  A missing wheel therefore leaves the preview root
    untouched, while a package drift conflicts with the existing immutable
    renderer configuration instead of silently changing a later build.
    """

    root = _resolve_document_preview_root(document_preview_root)
    pdfium_version = _installed_pdfium_version()
    for relative_path, payload in _fixed_synthetic_raster_build_input_payloads(
        pdfium_version=pdfium_version
    ).items():
        _write_or_verify_fixed_local_file(
            document_preview_root=root, relative_path=relative_path, payload=payload
        )
    return resolve_synthetic_raster_build_inputs(
        document_preview_root=root,
        build_inputs_relative_path=SYNTHETIC_RASTER_BUILD_INPUTS_RELATIVE_PATH,
        pdfium_version_provider=lambda: pdfium_version,
    )


def _read_local_json_model(
    *,
    document_preview_root: Path,
    relative_path: str,
    model_type: type[_WireModel],
) -> tuple[_WireModel, str]:
    path = _resolve_document_preview_local_path(
        document_preview_root=document_preview_root,
        relative_path=relative_path,
        require_directory=False,
    )
    raw = _read_local_document_preview_bytes(path)
    try:
        model = model_type.model_validate_json(raw)
    except Exception:
        raise ValueError("synthetic raster build JSON is invalid") from None
    return model, _sha256_bytes(raw)


def _installed_pdfium_version() -> str:
    try:
        return importlib.metadata.version("pypdfium2")
    except Exception:
        raise SyntheticRasterCompilerUnavailable("pinned PDFium runtime is unavailable") from None


def resolve_synthetic_raster_build_inputs(
    *,
    document_preview_root: Path,
    build_inputs_relative_path: str,
    pdfium_version_provider: Callable[[], str] | None = None,
) -> ResolvedSyntheticRasterBuildInputsV1:
    """Read one local manifest and derive every build-provenance hash from bytes.

    ``pdfium_version_provider`` is a narrow unit-test seam.  The dataset CLI
    always uses the installed pinned package and has no version argument.
    """

    root = _resolve_document_preview_root(document_preview_root)
    _validate_document_preview_relative_path(build_inputs_relative_path, label="build inputs")
    inputs_model, inputs_sha256 = _read_local_json_model(
        document_preview_root=root,
        relative_path=build_inputs_relative_path,
        model_type=SyntheticRasterBuildInputsV1,
    )
    if not isinstance(inputs_model, SyntheticRasterBuildInputsV1):  # defensive for generic helper
        raise ValueError("synthetic raster build inputs are invalid")
    source_root = _resolve_document_preview_local_path(
        document_preview_root=root,
        relative_path=inputs_model.source_root_relative_path,
        require_directory=True,
    )
    renderer_model, renderer_sha256 = _read_local_json_model(
        document_preview_root=root,
        relative_path=inputs_model.renderer_configuration_relative_path,
        model_type=SyntheticRasterRendererConfigurationV1,
    )
    font_model, font_sha256 = _read_local_json_model(
        document_preview_root=root,
        relative_path=inputs_model.font_manifest_relative_path,
        model_type=SyntheticRasterFontManifestV1,
    )
    if not isinstance(renderer_model, SyntheticRasterRendererConfigurationV1) or not isinstance(
        font_model, SyntheticRasterFontManifestV1
    ):
        raise ValueError("synthetic raster build configuration is invalid")
    seed_hashes: dict[str, str] = {}
    for name, relative_path in (
        ("split", inputs_model.split_seed_relative_path),
        ("recipe", inputs_model.render_recipe_relative_path),
        ("render", inputs_model.render_seed_relative_path),
    ):
        local_path = _resolve_document_preview_local_path(
            document_preview_root=root,
            relative_path=relative_path,
            require_directory=False,
        )
        seed_hashes[name] = _sha256_bytes(_read_local_document_preview_bytes(local_path))
    installed_version = (pdfium_version_provider or _installed_pdfium_version)()
    if installed_version != renderer_model.pdfium_version:
        raise SyntheticRasterCompilerUnavailable("pinned PDFium version is unavailable")
    return ResolvedSyntheticRasterBuildInputsV1(
        document_preview_root=root,
        source_root=source_root,
        build_inputs_relative_path=build_inputs_relative_path,
        build_inputs_sha256=inputs_sha256,
        renderer_configuration=renderer_model,
        renderer_configuration_sha256=renderer_sha256,
        font_manifest=font_model,
        font_manifest_sha256=font_sha256,
        split_seed_sha256=seed_hashes["split"],
        render_recipe_sha256=seed_hashes["recipe"],
        render_seed_sha256=seed_hashes["render"],
    )


@dataclass(frozen=True, slots=True)
class RenderedRasterBytesV1:
    """Ephemeral renderer output verified before its compact receipt is made."""

    asset_id: str
    profile_id: RasterProfileId
    content: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            re.fullmatch(_IDENTIFIER, self.asset_id) is None
            or not self.content
            or len(self.content) > 32 * 1024 * 1024
            or self.width < 1
            or self.height < 1
        ):
            raise ValueError("rendered raster output is invalid")


class SyntheticRasterCompilerUnavailable(RuntimeError):
    """A required local PDFium/Pillow capability is not provisioned.

    This is intentionally distinct from an invalid source document.  Callers
    can keep a profile unavailable without mistaking a missing optional wheel
    for a successful, empty raster fixture.
    """


def _final_pdf_font_resource_bindings(
    source_pdf: bytes,
) -> Mapping[str, tuple[str, str | None]]:
    """Resolve the fixed PDF's ``/F*`` resources from final PDF bytes only.

    The source-attested content parser intentionally handles a very small PDF
    subset.  It already rejects unknown streams and operators; this companion
    reader is equally narrow and accepts only direct Type1 font dictionaries
    used by the fixed synthetic templates.  It does not ask PDFium, OCR, a
    VLM, or a host font to identify text.
    """

    object_fonts: dict[tuple[int, int], tuple[str, str | None]] = {}
    for match in _PDF_INDIRECT_OBJECT_RE.finditer(source_pdf):
        body = match.group("body")
        if b"/BaseFont" not in body:
            continue
        base_font = _PDF_BASE_FONT_RE.search(body)
        if base_font is None:
            raise ValueError("final PDF font resource is invalid")
        encoding = _PDF_ENCODING_RE.search(body)
        key = (int(match.group("object")), int(match.group("generation")))
        value = (
            base_font.group("font").decode("ascii"),
            encoding.group("encoding").decode("ascii") if encoding is not None else None,
        )
        previous = object_fonts.get(key)
        if previous is not None and previous != value:
            raise ValueError("final PDF font resource is ambiguous")
        object_fonts[key] = value

    references: dict[str, set[tuple[int, int]]] = {}
    for match in _PDF_FONT_RESOURCE_REFERENCE_RE.finditer(source_pdf):
        references.setdefault(match.group("resource").decode("ascii"), set()).add(
            (int(match.group("object")), int(match.group("generation")))
        )
    bindings: dict[str, tuple[str, str | None]] = {}
    for resource_name, references_for_name in references.items():
        if len(references_for_name) != 1:
            raise ValueError("final PDF font resource is ambiguous")
        font = object_fonts.get(next(iter(references_for_name)))
        if font is None:
            raise ValueError("final PDF font resource is invalid")
        bindings[resource_name] = font
    if not bindings:
        raise ValueError("final PDF font resource is invalid")
    return bindings


def _final_pdf_text_run_decoder(source_pdf: bytes) -> Callable[[str, str], str]:
    """Return the one safe byte-to-Unicode decoder for a final source PDF."""

    bindings = _final_pdf_font_resource_bindings(source_pdf)

    def _decode(font_resource_name: str, encoded_text: str) -> str:
        font = bindings.get(font_resource_name)
        if font is None:
            raise ValueError("final PDF text font is not declared")
        base_font, encoding = font
        raw = encoded_text.encode("latin-1")
        if base_font in _WIN_ANSI_BASE_FONTS and encoding == "WinAnsiEncoding":
            return raw.decode("cp1252")
        if base_font == "ZapfDingbats" and encoding is None:
            try:
                return "".join(_FIXED_ZAPF_DINGBATS_UNICODE_BY_CODE[value] for value in raw)
            except KeyError:
                raise ValueError("final PDF ZapfDingbats glyph is unsupported") from None
        raise ValueError("final PDF text font is unsupported")

    return _decode


class FinalPdfContentStreamParserV1(Protocol):
    """Trusted, source-only truth reader supplied by the data preparation job.

    The parser is deliberately outside this module: it may read only the
    final PDF content stream and must never consume OCR/VLM/manifest output.
    Its normalized result is then independently checked against PDFium before
    any Gold or raster asset is emitted.
    """

    def parse_final_pdf_content_stream(
        self, *, source_pdf: bytes, asset_id: str
    ) -> SourcePdfGeometryV1: ...


@dataclass(frozen=True, slots=True)
class SourceAttestedSyntheticContentStreamParserV1:
    """Concrete parser for the known synthetic B/L, Invoice, and P/L templates.

    It reuses the existing strict final-content-stream parser, not its prior
    JSON/CSV outputs.  The template profile contributes only immutable slot
    positions/identifiers; every value and evidence span is read again from
    the supplied final PDF bytes.  Unsupported source PDFs fail closed.
    """

    family_id: str
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]

    def parse_final_pdf_content_stream(
        self, *, source_pdf: bytes, asset_id: str
    ) -> SourcePdfGeometryV1:
        try:
            from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
                _fields_from_profile,
                _parse_pdf_content_stream_v1,
                _require_profile,
                _text_in_rectangle,
            )

            source_document_type = (
                "invoice" if self.document_type == "commercial_invoice" else self.document_type
            )
            parsed = _parse_pdf_content_stream_v1(
                source_pdf,
                text_run_decoder=_final_pdf_text_run_decoder(source_pdf),
            )
            profile = _require_profile(document_type=source_document_type, parsed=parsed)
            fields = _fields_from_profile(parsed=parsed, profile=profile)
            spans = [
                SourceTextGeometryV1(
                    span_id=f"span-{run.reading_order}",
                    page_number=1,
                    text=run.text,
                    content_stream_x=run.x,
                    content_stream_y=run.y,
                    reading_order=run.reading_order,
                )
                for run in parsed.text_runs
            ]
            source_fields = [
                SourceFieldGoldV1(
                    field_id=slot.field_name,
                    value=field.value,
                    evidence_span_ids=[
                        f"span-{anchor.reading_order}" for anchor in slot.source_runs
                    ],
                )
                for slot, field in zip(profile.field_slots, fields, strict=True)
            ]
            source_cells: list[SourceTableCellGoldV1] = []
            rectangle_rows = (
                profile.table.header_rectangle_indices,
                profile.table.body_rectangle_indices,
            )
            for row_index, rectangle_indexes in enumerate(rectangle_rows):
                for column_index, rectangle_index in enumerate(rectangle_indexes):
                    rectangle = parsed.rectangles[rectangle_index]
                    matching_runs = [
                        run
                        for run in parsed.text_runs
                        if _text_in_rectangle(run=run, rectangle=rectangle)
                    ]
                    if not matching_runs:
                        raise ValueError("final content-stream table cell is ambiguous")
                    source_cells.append(
                        SourceTableCellGoldV1(
                            table_id="items",
                            row_index=row_index,
                            column_index=column_index,
                            text="\n".join(run.text for run in matching_runs).strip(),
                            pdf_bbox=PdfBoundingBoxV1(
                                x=rectangle.x,
                                y=rectangle.y,
                                width=rectangle.width,
                                height=rectangle.height,
                            ),
                            evidence_span_ids=[
                                f"span-{run.reading_order}" for run in matching_runs
                            ],
                        )
                    )
            x0, y0, x1, y1 = parsed.media_box
            if x1 <= x0 or y1 <= y0:
                raise ValueError("final content-stream page dimensions are invalid")
            return SourcePdfGeometryV1(
                asset_id=asset_id,
                family_id=self.family_id,
                document_type=self.document_type,
                source_pdf_sha256=hashlib.sha256(source_pdf).hexdigest(),
                page_width_points=x1 - x0,
                page_height_points=y1 - y0,
                spans=spans,
                fields=source_fields,
                table_cells=source_cells,
            )
        except ValueError:
            raise
        except Exception:
            # The source parser's detailed exception may contain fixture
            # structure details.  This boundary emits only a stable code.
            raise ValueError("final content-stream parser rejected source PDF") from None


@dataclass(frozen=True, slots=True)
class PdfiumFinalPdfGeometryExtractorV1:
    """Concrete local PDFium text/geometry reader used by the fixture compiler.

    PDFium gives character boxes and a renderer-specific reading order.  The
    final content stream supplies each source run's text and authored ``Tm``
    origin.  We match exact non-whitespace character sequences, then accept a
    candidate only when its PDFium glyph box has one compatible origin.  This
    handles a renderer that orders independent columns differently from the
    content stream without guessing through duplicate text.  Any extra or
    missing text, non-unique spatial candidate, unsupported API, or geometry
    mismatch aborts generation.
    """

    pdfium_version: str
    max_pages: int = 20

    def _pdfium(self) -> object:
        try:
            installed = importlib.metadata.version("pypdfium2")
            if installed != self.pdfium_version:
                raise SyntheticRasterCompilerUnavailable("pinned PDFium version is unavailable")
            import pypdfium2  # type: ignore[import-not-found]
        except SyntheticRasterCompilerUnavailable:
            raise
        except Exception:
            raise SyntheticRasterCompilerUnavailable(
                "pinned PDFium runtime is unavailable"
            ) from None
        return pypdfium2

    @staticmethod
    def _page_size(page: object) -> tuple[float, float]:
        try:
            size = page.get_size()  # type: ignore[attr-defined]
            width, height = float(size[0]), float(size[1])
        except Exception:
            raise SyntheticRasterCompilerUnavailable(
                "PDFium page-size API is unavailable"
            ) from None
        if width <= 0 or height <= 0:
            raise ValueError("PDFium page size is invalid")
        return width, height

    @staticmethod
    def _span_box(text_page: object, *, character_indexes: Sequence[int]) -> PdfBoundingBoxV1:
        boxes: list[tuple[float, float, float, float]] = []
        for index in character_indexes:
            try:
                left, bottom, right, top = text_page.get_charbox(index)  # type: ignore[attr-defined]
                left, bottom, right, top = map(float, (left, bottom, right, top))
            except Exception:
                raise SyntheticRasterCompilerUnavailable(
                    "PDFium character-geometry API is unavailable"
                ) from None
            if right <= left or top <= bottom:
                raise ValueError("PDFium character geometry is invalid")
            boxes.append((left, bottom, right, top))
        if not boxes:
            raise ValueError("PDFium source span has no drawable characters")
        left = min(box[0] for box in boxes)
        bottom = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        top = max(box[3] for box in boxes)
        return PdfBoundingBoxV1(x=left, y=bottom, width=right - left, height=top - bottom)

    @staticmethod
    def _layout_text(value: str) -> str:
        """Compare only PDFium layout whitespace, never letters/punctuation/case.

        PDFium may inject a line boundary or normalize a non-breaking space
        between adjacent content-stream text runs.  For this English fixture
        scope, whitespace is the sole allowed extraction difference.  Every
        non-whitespace Unicode code point remains ordered and exact.
        """

        return "".join(
            character
            for character in unicodedata.normalize("NFC", value).replace("\u00a0", " ")
            if not character.isspace()
        )

    @classmethod
    def _non_whitespace_pdfium_characters(cls, page_text: str) -> tuple[tuple[str, int], ...]:
        """Return one normalized, drawable character and original index at a time."""

        values: list[tuple[str, int]] = []
        for index, character in enumerate(page_text):
            normalized = cls._layout_text(character)
            if not normalized:
                continue
            if len(normalized) != 1:
                raise ValueError("PDFium text normalization is ambiguous")
            values.append((normalized, index))
        if not values:
            raise ValueError("PDFium page has no drawable text")
        return tuple(values)

    @classmethod
    def _matching_character_index_sequences(
        cls,
        *,
        page_characters: Sequence[tuple[str, int]],
        source_text: str,
    ) -> tuple[tuple[int, ...], ...]:
        """Enumerate exact source-text matches without selecting by text alone."""

        expected = tuple(cls._layout_text(source_text))
        if not expected:
            raise ValueError("final content-stream span has no drawable text")
        if len(expected) > len(page_characters):
            return ()
        candidates: list[tuple[int, ...]] = []
        for start in range(len(page_characters) - len(expected) + 1):
            if (
                tuple(
                    character
                    for character, _index in page_characters[start : start + len(expected)]
                )
                != expected
            ):
                continue
            candidates.append(
                tuple(index for _character, index in page_characters[start : start + len(expected)])
            )
        return tuple(dict.fromkeys(candidates))

    @staticmethod
    def _is_content_stream_origin_compatible(
        *, source_span: SourceTextGeometryV1, pdfium_bbox: PdfBoundingBoxV1
    ) -> bool:
        if source_span.content_stream_x is None or source_span.content_stream_y is None:
            raise ValueError("source text span lacks final content-stream origin")
        return (
            abs(pdfium_bbox.x - source_span.content_stream_x)
            <= _CONTENT_STREAM_ORIGIN_X_TOLERANCE_POINTS
            and abs(pdfium_bbox.y - source_span.content_stream_y)
            <= _CONTENT_STREAM_ORIGIN_Y_TOLERANCE_POINTS
        )

    def _match_spans_by_content_stream_origin(
        self,
        *,
        text_page: object,
        page_text: str,
        source_spans: Sequence[SourceTextGeometryV1],
    ) -> tuple[tuple[SourceTextGeometryV1, PdfBoundingBoxV1], ...]:
        """Prove one exact non-overlapping PDFium character assignment per span."""

        page_characters = self._non_whitespace_pdfium_characters(page_text)
        used_character_indexes: set[int] = set()
        matches: list[tuple[SourceTextGeometryV1, PdfBoundingBoxV1]] = []
        for source_span in source_spans:
            compatible: list[tuple[tuple[int, ...], PdfBoundingBoxV1]] = []
            for indexes in self._matching_character_index_sequences(
                page_characters=page_characters,
                source_text=source_span.text,
            ):
                bbox = self._span_box(text_page, character_indexes=indexes)
                if self._is_content_stream_origin_compatible(
                    source_span=source_span, pdfium_bbox=bbox
                ):
                    compatible.append((indexes, bbox))
            if len(compatible) != 1:
                raise ValueError("PDFium text does not match final content-stream truth")
            indexes, bbox = compatible[0]
            if used_character_indexes.intersection(indexes):
                raise ValueError("PDFium text does not match final content-stream truth")
            used_character_indexes.update(indexes)
            matches.append((source_span, bbox))
        if used_character_indexes != {index for _character, index in page_characters}:
            raise ValueError("PDFium text does not match final content-stream truth")
        return tuple(matches)

    def extract(
        self,
        *,
        source_pdf: bytes,
        source: SourcePdfGeometryV1,
    ) -> tuple[PdfiumGeometrySpanV1, ...]:
        if hashlib.sha256(source_pdf).hexdigest() != source.source_pdf_sha256:
            raise ValueError("source PDF bytes do not match source geometry hash")
        pdfium = self._pdfium()
        document: Any = None
        observations: list[PdfiumGeometrySpanV1] = []
        try:
            document = pdfium.PdfDocument(source_pdf)  # type: ignore[attr-defined]
            page_count = len(document)
            # Raster v1 has one output file and one normalized coordinate
            # canvas per source PDF.  A blank second page is still a
            # multi-page document, so it must not slip through merely because
            # it contains no text runs.
            if page_count != 1:
                raise ValueError("synthetic raster v1 asset must contain exactly one page")
            if max(span.page_number for span in source.spans) > page_count:
                raise ValueError("source content-stream span refers to a missing page")
            for page_number in range(1, page_count + 1):
                expected = sorted(
                    (span for span in source.spans if span.page_number == page_number),
                    key=lambda span: span.reading_order,
                )
                page = document[page_number - 1]
                width, height = self._page_size(page)
                # Source model v1 deliberately has one fixed canvas.  A
                # mixed-size PDF is not safely transformable under this v1
                # contract, so reject rather than distort a bbox.
                if (
                    abs(width - source.page_width_points) > 0.01
                    or abs(height - source.page_height_points) > 0.01
                ):
                    raise ValueError("PDFium page size does not match source geometry")
                text_page = page.get_textpage()
                try:
                    page_text = text_page.get_text_bounded()  # type: ignore[attr-defined]
                except AttributeError:
                    page_text = text_page.get_text_range()  # type: ignore[attr-defined]
                if not isinstance(page_text, str):
                    raise SyntheticRasterCompilerUnavailable("PDFium text API is unavailable")
                for expected_span, bbox in self._match_spans_by_content_stream_origin(
                    text_page=text_page,
                    page_text=page_text,
                    source_spans=expected,
                ):
                    observations.append(
                        PdfiumGeometrySpanV1(
                            page_number=page_number,
                            text=expected_span.text,
                            pdf_bbox=bbox,
                            reading_order=expected_span.reading_order,
                            source_pdf_sha256=source.source_pdf_sha256,
                        )
                    )
        except SyntheticRasterCompilerUnavailable:
            raise
        except ValueError:
            raise
        except Exception:
            raise SyntheticRasterCompilerUnavailable(
                "PDFium text geometry extraction failed"
            ) from None
        finally:
            if document is not None:
                try:
                    document.close()
                except Exception:
                    # The source bytes and observations are already in memory
                    # at this point.  Cleanup failure must not cause a partial
                    # data write (preflight has no output directory yet).
                    pass
        verify_pdfium_geometry(source, pdfium_spans=observations)
        return tuple(observations)


@dataclass(frozen=True, slots=True)
class SyntheticRasterPdfiumCompiler:
    """Render the four fixed profiles with local pinned PDFium and Pillow only.

    ``compile`` accepts source geometry produced by a caller-supplied final
    content-stream parser.  It always performs the independent PDFium check
    first, so no bytes are written when source text and PDFium disagree.
    """

    pdfium_version: str
    rasterizer_revision: str = "pdfium-raster-v1"
    max_pages: int = 20

    def _pdfium(self) -> object:
        return PdfiumFinalPdfGeometryExtractorV1(
            self.pdfium_version, max_pages=self.max_pages
        )._pdfium()

    def compile(
        self,
        *,
        source_pdf: bytes,
        source: SourcePdfGeometryV1,
        profile_id: RasterProfileId,
    ) -> tuple[RenderedRasterBytesV1, tuple[PdfiumGeometrySpanV1, ...]]:
        """Return one verified in-memory asset; callers choose an immutable writer."""

        geometry = PdfiumFinalPdfGeometryExtractorV1(
            self.pdfium_version, max_pages=self.max_pages
        ).extract(source_pdf=source_pdf, source=source)
        return (
            self.render(source_pdf=source_pdf, source=source, profile_id=profile_id),
            geometry,
        )

    def render(
        self,
        *,
        source_pdf: bytes,
        source: SourcePdfGeometryV1,
        profile_id: RasterProfileId,
    ) -> RenderedRasterBytesV1:
        """Render one already-preflighted single-page source PDF.

        ``compile`` remains the safe all-in-one primitive for small callers.
        The full dataset builder uses this method only after every source has
        independently passed content-stream and PDFium geometry preflight, so
        it does not repeat that expensive check four times per source.
        """

        if hashlib.sha256(source_pdf).hexdigest() != source.source_pdf_sha256:
            raise ValueError("source PDF bytes do not match source geometry hash")
        profile = RASTER_PROFILES[profile_id]
        pdfium = self._pdfium()
        document: Any = None
        try:
            document = pdfium.PdfDocument(source_pdf)  # type: ignore[attr-defined]
            page_count = len(document)
            if page_count != 1:
                # JPEG/PNG profile output has a single-file contract.  The
                # dataset compiler invokes this once per source page for any
                # future multi-page fixture rather than silently concatenate.
                raise ValueError("synthetic raster v1 asset must contain exactly one page")
            page = document[0]
            image = page.render(scale=profile.dpi / 72).to_pil().convert("RGB")
            width, height = image.size
            encoded = io.BytesIO()
            if profile.media_type == "image/png":
                image.save(encoded, format="PNG", optimize=False)
            elif profile.media_type == "image/jpeg":
                image.save(
                    encoded,
                    format="JPEG",
                    quality=profile.jpeg_quality,
                    optimize=False,
                    progressive=False,
                    subsampling=0,
                )
            else:
                # Pillow emits image XObjects for a raster PDF.  Keep the JPEG
                # recipe explicit: the profile name is evidence, not a default
                # inherited from whatever Pillow version happens to be local.
                image.save(
                    encoded,
                    format="PDF",
                    resolution=profile.dpi,
                    quality=profile.jpeg_quality,
                )
            content = encoded.getvalue()
        except SyntheticRasterCompilerUnavailable:
            raise
        except ValueError:
            raise
        except Exception:
            raise SyntheticRasterCompilerUnavailable("PDFium raster rendering failed") from None
        finally:
            if document is not None:
                try:
                    document.close()  # type: ignore[attr-defined]
                except Exception:
                    # PDFium cleanup is best-effort.  The rendered bytes have
                    # already been fully materialized above, and a cleanup
                    # error must not convert a valid asset into a different
                    # partial output state.
                    pass
        rendered = RenderedRasterBytesV1(
            asset_id=source.asset_id,
            profile_id=profile_id,
            content=content,
            width=width,
            height=height,
        )
        verify_rendered_raster_output(rendered, pdfium_version=self.pdfium_version)
        return rendered

    def compile_from_content_stream(
        self,
        *,
        source_pdf: bytes,
        asset_id: str,
        parser: FinalPdfContentStreamParserV1,
        profile_id: RasterProfileId,
    ) -> tuple[RenderedRasterBytesV1, tuple[PdfiumGeometrySpanV1, ...]]:
        """Use an explicitly injected final-content-stream parser, then compile.

        This is the runnable boundary for the future source parser.  It makes
        it impossible for the renderer to substitute a manifest, OCR output,
        or VLM answer for the source truth.
        """

        source = parser.parse_final_pdf_content_stream(source_pdf=source_pdf, asset_id=asset_id)
        if source.asset_id != asset_id:
            raise ValueError("final content-stream parser asset binding is invalid")
        return self.compile(source_pdf=source_pdf, source=source, profile_id=profile_id)


def write_immutable_rendered_raster_output(
    rendered: RenderedRasterBytesV1,
    *,
    output_path: Path,
    pdfium_version: str,
) -> RasterizedOutputV1:
    """Write one local fixture byte stream with create-only semantics.

    The caller must point at the Git-excluded local data directory.  This
    lower-level writer intentionally supports a temporary test directory but
    never overwrites an existing output or follows a symlink.
    """

    expected_suffix = _RASTER_ASSET_SUFFIX[rendered.profile_id]
    if output_path.suffix.casefold() != expected_suffix or output_path.is_symlink():
        raise ValueError("synthetic raster output path is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("synthetic raster output already exists") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(rendered.content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return verify_rendered_raster_output(rendered, pdfium_version=pdfium_version)


@dataclass(frozen=True, slots=True)
class SyntheticRasterSourceAssetV1:
    """One deterministic source path; path is local execution state, not receipt data."""

    asset_id: str
    family_id: str
    fixture_kind: Literal["clean", "error"]
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    source_path: Path


def _expected_source_asset_location(
    *,
    asset_id: str,
) -> tuple[
    str,
    str,
    Literal["clean", "error"],
    Literal["bill_of_lading", "commercial_invoice", "packing_list"],
]:
    match = re.fullmatch(r"(clean|error)-(\d{4})-(bl|inv|pl)", asset_id)
    if match is None:
        raise ValueError("synthetic raster source receipt asset ID is invalid")
    fixture_kind, number, suffix = match.groups()
    document_type_by_suffix: dict[
        str, Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    ] = {
        "bl": "bill_of_lading",
        "inv": "commercial_invoice",
        "pl": "packing_list",
    }
    suffix_by_document = {"bl": "BL", "inv": "INV", "pl": "PL"}
    directory = "clean_sets" if fixture_kind == "clean" else "error_sets"
    prefix = "C" if fixture_kind == "clean" else "E"
    return (
        f"{directory}/set_{number}/{prefix}{number}_{suffix_by_document[suffix]}.pdf",
        f"family-{number}",
        fixture_kind,  # type: ignore[return-value]
        document_type_by_suffix[suffix],
    )


class SyntheticRasterSourceAssetReceiptV1(_WireModel):
    """Hash-only evidence for one final source PDF in the fixed 6,000-file tree."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    family_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    fixture_kind: Literal["clean", "error"]
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    source_relative_path: str = Field(min_length=1, max_length=240)
    source_pdf_sha256: str = Field(pattern=_SHA256)
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)

    @model_validator(mode="after")
    def _uses_the_fixed_source_tree_location(self) -> "SyntheticRasterSourceAssetReceiptV1":
        expected_path, expected_family, expected_fixture, expected_document_type = (
            _expected_source_asset_location(asset_id=self.asset_id)
        )
        if (
            self.source_relative_path != expected_path
            or self.family_id != expected_family
            or self.fixture_kind != expected_fixture
            or self.document_type != expected_document_type
        ):
            raise ValueError(
                "synthetic raster source receipt asset does not bind the fixed source tree"
            )
        return self


class SyntheticRasterSourceReceiptV1(_WireModel):
    """Canonical, create-only hash receipt for the direct fixed source tree."""

    schema_version: Literal["1.0"] = "1.0"
    source_root_relative_path: Literal["통관서류_합성데이터셋_2000세트"] = (
        SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH
    )
    source_tree_layout: Literal["clean-error-1000-family-6000-pdf-v1"] = (
        "clean-error-1000-family-6000-pdf-v1"
    )
    font_resource_policy: Literal["final_pdf_font_resources_only"] = _FINAL_PDF_FONT_RESOURCE_POLICY
    font_manifest_sha256: str = Field(pattern=_SHA256)
    assets: list[SyntheticRasterSourceAssetReceiptV1] = Field(min_length=6_000, max_length=6_000)

    @model_validator(mode="after")
    def _has_unique_source_assets(self) -> "SyntheticRasterSourceReceiptV1":
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("synthetic raster source receipt asset IDs must be unique")
        if tuple(asset_ids) != _expected_fixed_source_asset_ids():
            raise ValueError(
                "synthetic raster source receipt must cover exactly the fixed 6,000 PDFs"
            )
        return self


def _expected_fixed_source_asset_ids() -> tuple[str, ...]:
    return tuple(
        f"{fixture_kind}-{index:04d}-{suffix}"
        for index in range(1, 1_001)
        for fixture_kind in ("clean", "error")
        for suffix in ("bl", "inv", "pl")
    )


def validate_fixed_synthetic_raster_source_receipt(receipt: SyntheticRasterSourceReceiptV1) -> None:
    """Require the canonical receipt to cover exactly the direct 6,000-PDF tree."""

    if (
        len(receipt.assets) != 6_000
        or tuple(asset.asset_id for asset in receipt.assets) != _expected_fixed_source_asset_ids()
    ):
        raise ValueError("synthetic raster source receipt must cover exactly the fixed 6,000 PDFs")


def _verify_final_pdf_font_resources_only(
    source_pdf: bytes,
    *,
    font_manifest: SyntheticRasterFontManifestV1,
) -> None:
    """Reject a source without final-PDF font resources before rendering it.

    The renderer never receives a host font path, URL, or byte buffer.  Base14
    resources are allowed because they are declarations inside the final PDF;
    the PDF SHA later binds those declarations to every raster output.
    """

    if font_manifest.font_resource_policy != _FINAL_PDF_FONT_RESOURCE_POLICY:
        raise ValueError("synthetic raster font resource policy is invalid")
    if b"/BaseFont" not in source_pdf:
        raise ValueError("final source PDF has no declared font resources")


def build_synthetic_raster_source_receipt(
    *,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1,
) -> SyntheticRasterSourceReceiptV1:
    """Hash the actual direct source PDFs, without reading any old JSON/CSV Gold."""

    source_assets = enumerate_synthetic_raster_source_assets(build_inputs.source_root)
    receipt_assets: list[SyntheticRasterSourceAssetReceiptV1] = []
    for asset in source_assets:
        source_pdf = _read_local_source_pdf(asset.source_path)
        _verify_final_pdf_font_resources_only(
            source_pdf,
            font_manifest=build_inputs.font_manifest,
        )
        try:
            source_relative_path = asset.source_path.relative_to(
                build_inputs.source_root
            ).as_posix()
        except ValueError:
            raise ValueError(
                "synthetic raster source asset escapes the fixed source root"
            ) from None
        receipt_assets.append(
            SyntheticRasterSourceAssetReceiptV1(
                asset_id=asset.asset_id,
                family_id=asset.family_id,
                fixture_kind=asset.fixture_kind,
                document_type=asset.document_type,
                source_relative_path=source_relative_path,
                source_pdf_sha256=_sha256_bytes(source_pdf),
                byte_size=len(source_pdf),
            )
        )
    receipt = SyntheticRasterSourceReceiptV1(
        font_manifest_sha256=build_inputs.font_manifest_sha256,
        assets=receipt_assets,
    )
    validate_fixed_synthetic_raster_source_receipt(receipt)
    return receipt


def synthetic_raster_source_receipt_sha256(receipt: SyntheticRasterSourceReceiptV1) -> str:
    """Digest canonical source-receipt bytes before the raster receipt refers to it."""

    return _sha256_bytes(
        json.dumps(
            receipt.model_dump(mode="json", warnings=False),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def write_immutable_synthetic_raster_source_receipt(
    receipt: SyntheticRasterSourceReceiptV1,
    *,
    output_path: Path,
) -> str:
    """Create the canonical source receipt exactly once; never replace evidence."""

    if output_path.name != "source-receipt.json" or output_path.is_symlink():
        raise ValueError("synthetic raster source receipt output path is invalid")
    validate_fixed_synthetic_raster_source_receipt(receipt)
    return _write_immutable_local_json(
        output_path=output_path,
        payload=receipt.model_dump(mode="json", warnings=False),
    )


def enumerate_synthetic_raster_source_assets(
    source_root: Path, *, expected_family_count: int = 1_000
) -> tuple[SyntheticRasterSourceAssetV1, ...]:
    """Enumerate only the fixed clean/error PDF tree, never CSV/JSON labels.

    Each ``set_NNNN`` owns its six clean/error B/L, Invoice, Packing List PDFs
    and therefore one split.  A missing, extra, redirected, or non-regular
    source file fails before any renderer sees it.
    """

    try:
        root = source_root.resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("synthetic raster source root is invalid")
    except OSError as error:
        raise ValueError("synthetic raster source root is invalid") from error
    assets: list[SyntheticRasterSourceAssetV1] = []

    def _resolved_regular_source_path(path: Path) -> Path:
        try:
            current = root
            for part in path.relative_to(root).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError("synthetic raster source asset is invalid")
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if resolved.is_symlink() or not resolved.is_file():
                raise ValueError("synthetic raster source asset is invalid")
        except (OSError, ValueError):
            raise ValueError("synthetic raster source asset is invalid") from None
        return resolved

    type_by_suffix: dict[str, Literal["bill_of_lading", "commercial_invoice", "packing_list"]] = {
        "BL": "bill_of_lading",
        "INV": "commercial_invoice",
        "PL": "packing_list",
    }
    for index in range(1, expected_family_count + 1):
        set_id = f"set_{index:04d}"
        family_id = f"family-{index:04d}"
        fixture_groups: tuple[tuple[Literal["clean", "error"], str, str], ...] = (
            ("clean", "clean_sets", "C"),
            ("error", "error_sets", "E"),
        )
        for fixture_kind, directory, prefix in fixture_groups:
            for suffix, document_type in type_by_suffix.items():
                path = root / directory / set_id / f"{prefix}{index:04d}_{suffix}.pdf"
                resolved = _resolved_regular_source_path(path)
                assets.append(
                    SyntheticRasterSourceAssetV1(
                        asset_id=f"{fixture_kind}-{index:04d}-{suffix.lower()}",
                        family_id=family_id,
                        fixture_kind=fixture_kind,
                        document_type=document_type,
                        source_path=resolved,
                    )
                )
    if len(assets) != expected_family_count * 6:
        raise ValueError("synthetic raster source family count is invalid")
    expected_relative_paths = {asset.source_path.relative_to(root).as_posix() for asset in assets}
    observed_relative_paths: set[str] = set()
    try:
        for directory in ("clean_sets", "error_sets"):
            source_directory = root / directory
            if source_directory.is_symlink() or not source_directory.is_dir():
                raise ValueError("synthetic raster source tree is invalid")
            for candidate in source_directory.rglob("*.pdf"):
                resolved = _resolved_regular_source_path(candidate)
                observed_relative_paths.add(resolved.relative_to(root).as_posix())
    except (OSError, ValueError):
        raise ValueError("synthetic raster source tree is invalid") from None
    if observed_relative_paths != expected_relative_paths:
        raise ValueError("synthetic raster source tree is not the fixed PDF set")
    return tuple(assets)


def _read_local_source_pdf(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
            raise ValueError("synthetic raster source asset is invalid")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("synthetic raster source asset is invalid")
        return path.read_bytes()
    except OSError:
        raise ValueError("synthetic raster source asset is invalid") from None


def _write_immutable_local_json(*, output_path: Path, payload: object) -> str:
    if output_path.suffix != ".json" or output_path.is_symlink():
        raise ValueError("synthetic raster JSON output path is invalid")
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("synthetic raster JSON output already exists") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticRasterPreflightedSourceV1:
    """One source that passed all source-only and PDFium checks before writes."""

    asset: SyntheticRasterSourceAssetV1
    source: SourcePdfGeometryV1
    pdfium_spans: tuple[PdfiumGeometrySpanV1, ...]


def _validate_synthetic_raster_table_scope(source: SourcePdfGeometryV1) -> None:
    """Keep raster v1 limited to the known single-page table contract."""

    if {span.page_number for span in source.spans} != {1}:
        raise ValueError("synthetic raster v1 source must contain exactly one page")
    if not source.fields or not source.table_cells:
        raise ValueError("synthetic raster v1 source lacks required field or table truth")
    if {cell.table_id for cell in source.table_cells} != {"items"}:
        raise ValueError("synthetic raster v1 source table scope is unsupported")
    rows: dict[int, list[int]] = {}
    for cell in source.table_cells:
        rows.setdefault(cell.row_index, []).append(cell.column_index)
    if sorted(rows) != [0, 1] or any(
        sorted(columns) != list(range(len(columns))) for columns in rows.values()
    ):
        raise ValueError("synthetic raster v1 source table structure is unsupported")


def preflight_synthetic_raster_sources(
    *,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1,
    source_assets: Sequence[SyntheticRasterSourceAssetV1],
    source_receipt: SyntheticRasterSourceReceiptV1,
) -> tuple[SyntheticRasterPreflightedSourceV1, ...]:
    """Validate every source before a staging directory or output is created.

    This pass intentionally does not render or write any asset.  It parses
    final PDF content streams, proves the source hash and known table scope,
    and independently matches all geometry through PDFium for the entire
    fixed source tree.  An unsupported/mismatched one therefore cannot leave
    a partial final dataset behind.
    """

    validate_fixed_synthetic_raster_source_receipt(source_receipt)
    if (
        len(source_assets) != 6_000
        or tuple(asset.asset_id for asset in source_assets) != _expected_fixed_source_asset_ids()
    ):
        raise ValueError("synthetic raster source preflight must cover the fixed 6,000 PDFs")
    source_binding_by_id = {binding.asset_id: binding for binding in source_receipt.assets}
    if len(source_binding_by_id) != len(source_receipt.assets) or {
        asset.asset_id for asset in source_assets
    } != set(source_binding_by_id):
        raise ValueError("synthetic raster source preflight does not bind the canonical receipt")
    geometry_extractor = PdfiumFinalPdfGeometryExtractorV1(
        build_inputs.renderer_configuration.pdfium_version
    )
    preflighted: list[SyntheticRasterPreflightedSourceV1] = []
    for asset in source_assets:
        source_pdf = _read_local_source_pdf(asset.source_path)
        binding = source_binding_by_id[asset.asset_id]
        if (
            binding.family_id != asset.family_id
            or binding.fixture_kind != asset.fixture_kind
            or binding.document_type != asset.document_type
            or binding.byte_size != len(source_pdf)
            or binding.source_pdf_sha256 != _sha256_bytes(source_pdf)
        ):
            raise ValueError("synthetic raster source preflight does not bind source PDF bytes")
        _verify_final_pdf_font_resources_only(source_pdf, font_manifest=build_inputs.font_manifest)
        source = SourceAttestedSyntheticContentStreamParserV1(
            family_id=asset.family_id,
            document_type=asset.document_type,
        ).parse_final_pdf_content_stream(source_pdf=source_pdf, asset_id=asset.asset_id)
        if (
            source.asset_id != asset.asset_id
            or source.family_id != asset.family_id
            or source.document_type != asset.document_type
            or source.source_pdf_sha256 != binding.source_pdf_sha256
        ):
            raise ValueError("synthetic raster source parser does not bind the canonical receipt")
        _validate_synthetic_raster_table_scope(source)
        pdfium_spans = geometry_extractor.extract(source_pdf=source_pdf, source=source)
        # Compiling these in memory proves every source field/table evidence
        # span can be transformed from independently observed PDFium boxes;
        # no result is retained or written during preflight.
        compile_raster_field_gold(source, profile_id="png-216", pdfium_spans=pdfium_spans)
        compile_raster_table_cell_gold(source, profile_id="png-216", pdfium_spans=pdfium_spans)
        preflighted.append(
            SyntheticRasterPreflightedSourceV1(
                asset=asset,
                source=source,
                pdfium_spans=pdfium_spans,
            )
        )
    return tuple(preflighted)


def _create_synthetic_raster_staging_directory(*, document_preview_root: Path) -> tuple[Path, Path]:
    """Create one owned staging directory on the same filesystem as publish."""

    staging_parent = document_preview_root / f"{_SYNTHETIC_RASTER_STAGING_PREFIX}{uuid.uuid4().hex}"
    try:
        staging_parent.mkdir(mode=0o700)
        if staging_parent.is_symlink() or not staging_parent.is_dir():
            raise ValueError("synthetic raster staging directory is invalid")
        dataset_directory = staging_parent / _SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME
        dataset_directory.mkdir(mode=0o700)
    except OSError:
        raise ValueError("synthetic raster staging directory is unavailable") from None
    return staging_parent, dataset_directory


def _discard_owned_synthetic_raster_staging_directory(
    *, document_preview_root: Path, staging_parent: Path
) -> None:
    """Remove only a staging directory created by this invocation."""

    try:
        if (
            staging_parent.parent.resolve(strict=True) != document_preview_root
            or not staging_parent.name.startswith(_SYNTHETIC_RASTER_STAGING_PREFIX)
            or staging_parent.is_symlink()
        ):
            raise ValueError("synthetic raster staging directory is invalid")
        shutil.rmtree(staging_parent)
    except FileNotFoundError:
        return
    except OSError:
        # An unrecoverable scratch directory is not a dataset and is never
        # reused.  Do not broaden cleanup to a discovered/stale directory.
        return


def _publish_synthetic_raster_staging_directory(
    *, document_preview_root: Path, staging_parent: Path, dataset_directory: Path
) -> None:
    """Publish a verified staged dataset by a same-volume directory rename."""

    final_directory = document_preview_root / _SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME
    if (
        dataset_directory.parent != staging_parent
        or dataset_directory.name != _SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME
        or staging_parent.parent != document_preview_root
        or staging_parent.is_symlink()
        or final_directory.is_symlink()
        or final_directory.exists()
    ):
        raise ValueError("synthetic raster dataset publish target is invalid")
    try:
        # The build lock serializes supported writers.  ``rename`` rather
        # than replace intentionally fails on Windows if an unexpected final
        # directory appeared, leaving the current published data untouched.
        os.rename(dataset_directory, final_directory)
    except OSError:
        raise ValueError("synthetic raster dataset publish failed") from None
    try:
        staging_parent.rmdir()
    except OSError:
        # The published dataset is already complete and durable.  A leftover
        # empty/scratch parent is never reused and must not turn a successful
        # atomic publish into a reported failed activation.
        pass


def _acquire_synthetic_raster_build_lock(*, document_preview_root: Path) -> ContextManager[None]:
    """Acquire an OS-owned lock that a later automatic retry can reclaim."""

    lock_path = document_preview_root / _SYNTHETIC_RASTER_BUILD_LOCK_NAME
    try:
        lock = exclusive_local_process_lock(lock_path)
        lock.__enter__()
    except LocalProcessLockBusy:
        raise ValueError("synthetic raster dataset build is already running") from None
    except LocalProcessLockInvalid:
        raise ValueError("synthetic raster dataset build lock is unavailable") from None
    return lock


def _release_owned_synthetic_raster_build_lock(lock: ContextManager[None]) -> None:
    try:
        lock.__exit__(None, None, None)
    except OSError:
        # Closing the owning descriptor still releases the OS lock.  Do not
        # remove its inert marker: process death and normal retry are both
        # safely handled by the operating system rather than a stale file.
        pass


def compile_synthetic_raster_dataset(
    *,
    document_preview_root: Path,
    build_inputs_relative_path: str,
) -> SyntheticRasterReceiptV1:
    """Run the 6,000 source PDFs × 4 profiles local-only generation chain.

    The caller supplies a single document-preview-relative input manifest.
    Every provenance hash, the PDFium pin/revision, renderer semantics, source
    tree, and font policy are then read locally and bound from actual bytes.
    Source truth is re-parsed per PDF from the final content stream.  The
    complete source tree passes parser/PDFium/table preflight before one
    staging directory is created.  Gold and image bytes are then create-only
    in that staging directory, fully re-verified, and atomically published as
    ``synthetic-raster-v1``.  A matching existing final dataset is verified
    and reused; a partial or drifted one is never overwritten.
    """

    build_inputs = resolve_synthetic_raster_build_inputs(
        document_preview_root=document_preview_root,
        build_inputs_relative_path=build_inputs_relative_path,
    )
    output_root = build_inputs.document_preview_root
    lock_path = _acquire_synthetic_raster_build_lock(document_preview_root=output_root)
    staging_parent: Path | None = None
    try:
        existing = output_root / _SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME
        if existing.exists() or existing.is_symlink():
            receipt = read_verified_synthetic_raster_dataset(
                document_preview_root=output_root,
                expected_build_inputs_relative_path=build_inputs_relative_path,
            )
            return receipt

        # No final output exists.  Do the entire 6,000-PDF parser/PDFium
        # preflight before creating any staging/output directory.
        assets = enumerate_synthetic_raster_source_assets(build_inputs.source_root)
        source_receipt = build_synthetic_raster_source_receipt(build_inputs=build_inputs)
        preflighted_sources = preflight_synthetic_raster_sources(
            build_inputs=build_inputs,
            source_assets=assets,
            source_receipt=source_receipt,
        )

        staging_parent, staged_dataset = _create_synthetic_raster_staging_directory(
            document_preview_root=output_root
        )
        compiler = SyntheticRasterPdfiumCompiler(
            pdfium_version=build_inputs.renderer_configuration.pdfium_version,
            rasterizer_revision=build_inputs.renderer_configuration.renderer_revision,
        )
        outputs: list[RasterizedOutputV1] = []
        gold_artifacts: list[RasterGoldArtifactBindingV1] = []
        for preflighted in preflighted_sources:
            asset = preflighted.asset
            source = preflighted.source
            source_pdf = _read_local_source_pdf(asset.source_path)
            if _sha256_bytes(source_pdf) != source.source_pdf_sha256:
                raise ValueError("source PDF changed after synthetic raster preflight")
            for profile_id in RASTER_PROFILES:
                rendered = compiler.render(
                    source_pdf=source_pdf, source=source, profile_id=profile_id
                )
                destination = (
                    staged_dataset
                    / "assets"
                    / (f"{asset.asset_id}-{profile_id}{_RASTER_ASSET_SUFFIX[profile_id]}")
                )
                outputs.append(
                    write_immutable_rendered_raster_output(
                        rendered,
                        output_path=destination,
                        pdfium_version=compiler.pdfium_version,
                    )
                )
                gold = RasterGoldArtifactV1(
                    asset_id=asset.asset_id,
                    family_id=asset.family_id,
                    document_type=asset.document_type,
                    source_pdf_sha256=source.source_pdf_sha256,
                    profile_id=profile_id,
                    spans=compile_raster_gold(
                        source, profile_id=profile_id, pdfium_spans=preflighted.pdfium_spans
                    ),
                    fields=compile_raster_field_gold(
                        source, profile_id=profile_id, pdfium_spans=preflighted.pdfium_spans
                    ),
                    table_cells=compile_raster_table_cell_gold(
                        source, profile_id=profile_id, pdfium_spans=preflighted.pdfium_spans
                    ),
                )
                gold_sha256 = _write_immutable_local_json(
                    output_path=staged_dataset / "gold" / f"{asset.asset_id}-{profile_id}.json",
                    payload=gold.model_dump(mode="json", warnings=False),
                )
                gold_artifacts.append(
                    RasterGoldArtifactBindingV1(
                        asset_id=asset.asset_id,
                        profile_id=profile_id,
                        gold_relative_path=(
                            f"synthetic-raster-v1/gold/{asset.asset_id}-{profile_id}.json"
                        ),
                        gold_sha256=gold_sha256,
                    )
                )
        receipt = build_synthetic_raster_receipt(
            build_inputs=build_inputs,
            source_receipt=source_receipt,
            sources=[item.source for item in preflighted_sources],
            outputs=outputs,
            gold_artifacts=gold_artifacts,
        )
        _validate_complete_synthetic_raster_dataset_receipt(receipt)
        expected_source_receipt_sha256 = synthetic_raster_source_receipt_sha256(source_receipt)
        if (
            write_immutable_synthetic_raster_source_receipt(
                source_receipt,
                output_path=staged_dataset / "source-receipt.json",
            )
            != expected_source_receipt_sha256
        ):
            raise ValueError("synthetic raster source receipt digest is inconsistent")
        write_immutable_synthetic_raster_receipt(
            receipt,
            output_path=staged_dataset / "synthetic-raster-receipt.json",
        )
        verify_synthetic_raster_receipt_materials(
            document_preview_root=output_root,
            receipt=receipt,
            required_splits=("train", "development", "calibration", "sealed"),
            dataset_directory=staged_dataset,
        )
        _publish_synthetic_raster_staging_directory(
            document_preview_root=output_root,
            staging_parent=staging_parent,
            dataset_directory=staged_dataset,
        )
        staging_parent = None
        return receipt
    finally:
        if staging_parent is not None:
            _discard_owned_synthetic_raster_staging_directory(
                document_preview_root=output_root, staging_parent=staging_parent
            )
        _release_owned_synthetic_raster_build_lock(lock_path)


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if (
        len(content) < 24
        or not content.startswith(b"\x89PNG\r\n\x1a\n")
        or content[12:16] != b"IHDR"
    ):
        raise ValueError("PNG renderer output is invalid")
    width = int.from_bytes(content[16:20], "big")
    height = int.from_bytes(content[20:24], "big")
    if width < 1 or height < 1:
        raise ValueError("PNG renderer output is invalid")
    return width, height


def _jpeg_dimensions(content: bytes) -> tuple[int, int]:
    """Read the SOF dimensions without accepting a PNG/PDF cross-profile file."""

    if len(content) < 4 or not content.startswith(b"\xff\xd8") or not content.endswith(b"\xff\xd9"):
        raise ValueError("JPEG renderer output is invalid")
    cursor = 2
    while cursor < len(content):
        if content[cursor] != 0xFF:
            raise ValueError("JPEG renderer output is invalid")
        while cursor < len(content) and content[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(content):
            break
        marker = content[cursor]
        cursor += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:
            break
        if cursor + 2 > len(content):
            break
        segment_length = int.from_bytes(content[cursor : cursor + 2], "big")
        if segment_length < 2 or cursor + segment_length > len(content):
            break
        if 0xC0 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
            if segment_length < 8:
                break
            height = int.from_bytes(content[cursor + 3 : cursor + 5], "big")
            width = int.from_bytes(content[cursor + 5 : cursor + 7], "big")
            if width < 1 or height < 1:
                break
            return width, height
        cursor += segment_length
    raise ValueError("JPEG renderer output is invalid")


def verify_rendered_raster_output(
    rendered: RenderedRasterBytesV1,
    *,
    pdfium_version: str | None = None,
    pdfium_module: object | None = None,
) -> RasterizedOutputV1:
    """Validate the exact profile bytes before recording their digest.

    Every evaluator invocation repeats this validation over bytes re-read from
    the immutable receipt path.  A matching declarative SHA alone is therefore
    insufficient to make an asset count as a PNG, JPEG, or image-only scan PDF.
    """

    profile = RASTER_PROFILES[rendered.profile_id]
    if profile.media_type == "image/png":
        if _png_dimensions(rendered.content) != (rendered.width, rendered.height):
            raise ValueError("PNG renderer dimensions do not match receipt")
    elif profile.media_type == "image/jpeg":
        if _jpeg_dimensions(rendered.content) != (rendered.width, rendered.height):
            raise ValueError("JPEG renderer dimensions do not match receipt")
    else:
        if pdfium_version is None:
            raise ValueError("scan PDF validation requires a pinned PDFium version")
        validate_image_only_pdf(
            rendered.content,
            pdfium_version=pdfium_version,
            expected_width=rendered.width,
            expected_height=rendered.height,
            dpi=profile.dpi,
            pdfium_module=pdfium_module,
        )
    return RasterizedOutputV1(
        asset_id=rendered.asset_id,
        profile_id=rendered.profile_id,
        media_type=profile.media_type,
        content_sha256=hashlib.sha256(rendered.content).hexdigest(),
        byte_size=len(rendered.content),
        width=rendered.width,
        height=rendered.height,
    )


class RasterGoldSpanV1(_WireModel):
    """Local Gold payload; text is intentionally not copied into the receipt."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    profile_id: RasterProfileId
    page_number: int = Field(ge=1, le=1_000)
    span_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    text: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1
    reading_order: int = Field(ge=1, le=100_000)
    source_pdf_sha256: str = Field(pattern=_SHA256)


class RasterGoldFieldV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    profile_id: RasterProfileId
    field_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    value: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1
    evidence_span_ids: list[str] = Field(min_length=1, max_length=128)
    source_pdf_sha256: str = Field(pattern=_SHA256)


class RasterGoldTableCellV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    profile_id: RasterProfileId
    table_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    row_index: int = Field(ge=0, le=100_000)
    column_index: int = Field(ge=0, le=100_000)
    text: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1
    evidence_span_ids: list[str] = Field(min_length=1, max_length=128)
    source_pdf_sha256: str = Field(pattern=_SHA256)


class RasterGoldArtifactV1(_WireModel):
    """The create-only Gold file for one rendered source/profile pair.

    This model deliberately contains the text truth.  It is local evidence,
    not an activation receipt.  The receipt stores only its deterministic
    relative location and SHA-256 so a case file cannot substitute its own
    convenient Gold payload during calibration or sealed evaluation.
    """

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    family_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    source_pdf_sha256: str = Field(pattern=_SHA256)
    profile_id: RasterProfileId
    spans: list[RasterGoldSpanV1] = Field(min_length=1, max_length=20_000)
    fields: list[RasterGoldFieldV1] = Field(max_length=512)
    table_cells: list[RasterGoldTableCellV1] = Field(max_length=10_000)

    @model_validator(mode="after")
    def _binds_one_asset_profile_and_source(self) -> "RasterGoldArtifactV1":
        for collection in (self.spans, self.fields, self.table_cells):
            if any(
                item.asset_id != self.asset_id
                or item.profile_id != self.profile_id
                or item.source_pdf_sha256 != self.source_pdf_sha256
                for item in collection
            ):
                raise ValueError("raster Gold artifact binding is invalid")
        return self


class RasterGoldArtifactBindingV1(_WireModel):
    """Immutable reference to a local Gold file, without embedding Gold text."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    profile_id: RasterProfileId
    gold_relative_path: str = Field(min_length=1, max_length=240)
    gold_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _uses_the_fixed_gold_location(self) -> "RasterGoldArtifactBindingV1":
        expected = f"synthetic-raster-v1/gold/{self.asset_id}-{self.profile_id}.json"
        if self.gold_relative_path != expected:
            raise ValueError("raster Gold artifact path is not deterministic")
        return self


class RasterAssetReceiptV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    family_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    split: RasterSplit
    profile_id: RasterProfileId
    source_pdf_sha256: str = Field(pattern=_SHA256)
    image_sha256: str = Field(pattern=_SHA256)
    asset_relative_path: str = Field(min_length=1, max_length=240)
    media_type: Literal["image/png", "image/jpeg", "application/pdf"]
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    coordinate_transform: Literal["pdf-points-lower-left-to-normalized-top-left-v1"]
    render_recipe_sha256: str = Field(pattern=_SHA256)
    render_seed_sha256: str = Field(pattern=_SHA256)
    gold_relative_path: str = Field(min_length=1, max_length=240)
    gold_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _uses_the_fixed_local_locations(self) -> "RasterAssetReceiptV1":
        expected_asset = _raster_asset_relative_path(
            asset_id=self.asset_id, profile_id=self.profile_id
        )
        expected_gold = f"synthetic-raster-v1/gold/{self.asset_id}-{self.profile_id}.json"
        if self.asset_relative_path != expected_asset:
            raise ValueError("raster receipt asset path is not deterministic")
        if self.gold_relative_path != expected_gold:
            raise ValueError("raster receipt Gold path is not deterministic")
        return self


class SyntheticRasterReceiptV1(_WireModel):
    """Compact immutable receipt with hashes/provenance only, never Gold text."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    # Retained for parsing already-created local evidence.  The v1 builder no
    # longer accepts either value: it derives both from the fixed source tree
    # and its canonical source receipt.
    source_dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_receipt_sha256: str = Field(pattern=_SHA256)
    source_receipt_relative_path: str | None = Field(default=None, min_length=1, max_length=240)
    build_inputs_relative_path: str | None = Field(default=None, min_length=1, max_length=240)
    build_inputs_sha256: str | None = Field(default=None, pattern=_SHA256)
    renderer_kind: Literal["pdfium"] = "pdfium"
    renderer_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    renderer_configuration_sha256: str = Field(pattern=_SHA256)
    pdfium_version: str = Field(min_length=1, max_length=128)
    font_manifest_sha256: str = Field(pattern=_SHA256)
    font_resource_policy: Literal["final_pdf_font_resources_only"] = _FINAL_PDF_FONT_RESOURCE_POLICY
    geometry_compiler_revision: Literal[
        "source-content-stream-geometry-v1",
        "source-content-stream-geometry-v2-font-encoding",
    ] = "source-content-stream-geometry-v1"
    split_seed_sha256: str = Field(pattern=_SHA256)
    render_recipe_sha256: str = Field(pattern=_SHA256)
    render_seed_sha256: str = Field(pattern=_SHA256)
    coordinate_transform: Literal["pdf-points-lower-left-to-normalized-top-left-v1"] = (
        "pdf-points-lower-left-to-normalized-top-left-v1"
    )
    profiles: list[RasterProfileV1] = Field(min_length=4, max_length=4)
    assets: list[RasterAssetReceiptV1] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def _receipt_has_unique_profile_assets(self) -> "SyntheticRasterReceiptV1":
        pairs = [(asset.asset_id, asset.profile_id) for asset in self.assets]
        if len(pairs) != len(set(pairs)):
            raise ValueError("raster receipt asset/profile pairs must be unique")
        profile_ids = [profile.profile_id for profile in self.profiles]
        if set(profile_ids) != set(RASTER_PROFILES) or len(profile_ids) != len(set(profile_ids)):
            raise ValueError("raster receipt must bind all fixed raster profiles")
        if any(asset.profile_id not in set(profile_ids) for asset in self.assets):
            raise ValueError("raster receipt asset profile is not declared")
        if self.source_receipt_relative_path is not None:
            _validate_document_preview_relative_path(
                self.source_receipt_relative_path, label="source receipt"
            )
            if self.source_receipt_relative_path != SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH:
                raise ValueError("raster receipt source receipt path is not deterministic")
        if (self.build_inputs_relative_path is None) != (self.build_inputs_sha256 is None):
            raise ValueError("raster receipt build-input provenance is incomplete")
        if self.build_inputs_relative_path is not None:
            _validate_document_preview_relative_path(
                self.build_inputs_relative_path, label="build inputs"
            )
        return self


def _read_local_raster_artifact(path: Path, *, maximum_size: int) -> bytes:
    """Read one regular local evaluation artifact without following links."""

    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size < 1
            or path.stat().st_size > maximum_size
        ):
            raise ValueError("synthetic raster local artifact is invalid")
        return path.read_bytes()
    except OSError:
        raise ValueError("synthetic raster local artifact is invalid") from None


def _canonical_local_json(model: _WireModel) -> bytes:
    return _canonical_json_bytes(model.model_dump(mode="json", warnings=False))


def _validate_complete_synthetic_raster_dataset_receipt(receipt: SyntheticRasterReceiptV1) -> None:
    """Require the exact 6,000-source × 4-profile dataset before publish/reuse."""

    expected_pairs = {
        (asset_id, profile_id)
        for asset_id in _expected_fixed_source_asset_ids()
        for profile_id in RASTER_PROFILES
    }
    observed_pairs = {(asset.asset_id, asset.profile_id) for asset in receipt.assets}
    if len(receipt.assets) != 24_000 or observed_pairs != expected_pairs:
        raise ValueError("synthetic raster receipt is not a complete fixed dataset")
    expected_family_counts = {
        "train": 750,
        "development": 100,
        "calibration": 50,
        "sealed": 100,
    }
    split_by_family: dict[str, RasterSplit] = {}
    for asset in receipt.assets:
        _expected_path, expected_family, _fixture_kind, expected_document_type = (
            _expected_source_asset_location(asset_id=asset.asset_id)
        )
        if asset.family_id != expected_family or asset.document_type != expected_document_type:
            raise ValueError("synthetic raster receipt source family binding is invalid")
        existing_split = split_by_family.setdefault(asset.family_id, asset.split)
        if existing_split != asset.split:
            raise ValueError("synthetic raster receipt leaks a family across splits")
    if {
        split: sum(value == split for value in split_by_family.values())
        for split in expected_family_counts
    } != (expected_family_counts):
        raise ValueError("synthetic raster receipt family split counts are invalid")


def _resolve_synthetic_raster_dataset_directory(
    *, document_preview_root: Path, dataset_directory: Path | None
) -> Path:
    candidate = dataset_directory or (
        document_preview_root / _SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME
    )
    try:
        if candidate.is_symlink():
            raise ValueError("synthetic raster dataset directory is invalid")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(document_preview_root)
        if (
            resolved.name != _SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME
            or resolved.is_symlink()
            or not resolved.is_dir()
        ):
            raise ValueError("synthetic raster dataset directory is invalid")
    except (OSError, ValueError):
        raise ValueError("synthetic raster dataset directory is invalid") from None
    return resolved


def _resolve_dataset_artifact_path(*, dataset_directory: Path, relative_path: str) -> Path:
    """Map a receipt's fixed final path onto final or staged dataset bytes."""

    _validate_document_preview_relative_path(relative_path, label="raster artifact")
    prefix = f"{_SYNTHETIC_RASTER_DATASET_DIRECTORY_NAME}/"
    if not relative_path.startswith(prefix):
        raise ValueError("synthetic raster artifact path is invalid")
    parts = PurePosixPath(relative_path).parts[1:]
    if not parts:
        raise ValueError("synthetic raster artifact path is invalid")
    candidate = dataset_directory
    try:
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("synthetic raster artifact path is invalid")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(dataset_directory)
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError("synthetic raster artifact path is invalid")
    except (OSError, ValueError):
        raise ValueError("synthetic raster artifact path is invalid") from None
    return resolved


def read_verified_synthetic_raster_dataset(
    *,
    document_preview_root: Path,
    expected_build_inputs_relative_path: str | None = None,
) -> SyntheticRasterReceiptV1:
    """Return an existing dataset only after full immutable-material replay."""

    root = _resolve_document_preview_root(document_preview_root)
    if expected_build_inputs_relative_path is not None:
        _validate_document_preview_relative_path(
            expected_build_inputs_relative_path, label="build inputs"
        )
    dataset_directory = _resolve_synthetic_raster_dataset_directory(
        document_preview_root=root, dataset_directory=None
    )
    raw = _read_local_raster_artifact(
        dataset_directory / "synthetic-raster-receipt.json",
        maximum_size=_MAX_LOCAL_RASTER_ARTIFACT_BYTES,
    )
    try:
        receipt = SyntheticRasterReceiptV1.model_validate_json(raw)
    except ValueError:
        raise ValueError("synthetic raster dataset receipt is invalid") from None
    if raw != _canonical_local_json(receipt):
        raise ValueError("synthetic raster dataset receipt is not canonical")
    _validate_complete_synthetic_raster_dataset_receipt(receipt)
    if (
        expected_build_inputs_relative_path is not None
        and receipt.build_inputs_relative_path != expected_build_inputs_relative_path
    ):
        raise ValueError("synthetic raster dataset build inputs do not match requested build")
    verify_synthetic_raster_receipt_materials(
        document_preview_root=root,
        receipt=receipt,
        required_splits=("train", "development", "calibration", "sealed"),
        dataset_directory=dataset_directory,
    )
    return receipt


def verify_synthetic_raster_receipt_materials(
    *,
    document_preview_root: Path,
    receipt: SyntheticRasterReceiptV1,
    required_splits: Sequence[RasterSplit],
    dataset_directory: Path | None = None,
) -> None:
    """Re-read every source/build/raster/Gold byte that can justify activation.

    Content-addressed receipts bind their original bytes, but they cannot
    detect a later local replacement on their own.  This expensive verifier is
    intentionally for evaluation/activation boundaries, never a customer OCR
    request.  It recreates the fixed source receipt, rebinds the local build
    inputs, and validates every requested raster/Gold pair before an approval
    resolver can select the profile.
    """

    requested_splits = set(required_splits)
    if not requested_splits or not requested_splits <= {
        "train",
        "development",
        "calibration",
        "sealed",
    }:
        raise ValueError("synthetic raster verification split is invalid")
    if (
        receipt.source_receipt_relative_path is None
        or receipt.build_inputs_relative_path is None
        or receipt.build_inputs_sha256 is None
        or receipt.source_receipt_relative_path != SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH
        or receipt.source_dataset_id != _SYNTHETIC_RASTER_SOURCE_DATASET_ID
        or receipt.source_dataset_revision != receipt.source_receipt_sha256
        or [profile.model_dump(mode="json") for profile in receipt.profiles]
        != [profile.model_dump(mode="json") for profile in RASTER_PROFILES.values()]
    ):
        raise ValueError("synthetic raster receipt provenance is invalid")
    root = _resolve_document_preview_root(document_preview_root)
    artifact_directory = _resolve_synthetic_raster_dataset_directory(
        document_preview_root=root, dataset_directory=dataset_directory
    )
    build_inputs = resolve_synthetic_raster_build_inputs(
        document_preview_root=root,
        build_inputs_relative_path=receipt.build_inputs_relative_path,
    )
    if (
        build_inputs.build_inputs_sha256 != receipt.build_inputs_sha256
        or build_inputs.renderer_configuration_sha256 != receipt.renderer_configuration_sha256
        or build_inputs.renderer_configuration.renderer_revision != receipt.renderer_revision
        or build_inputs.renderer_configuration.geometry_compiler_revision
        != receipt.geometry_compiler_revision
        or build_inputs.renderer_configuration.pdfium_version != receipt.pdfium_version
        or build_inputs.font_manifest_sha256 != receipt.font_manifest_sha256
        or build_inputs.font_manifest.font_resource_policy != receipt.font_resource_policy
        or build_inputs.split_seed_sha256 != receipt.split_seed_sha256
        or build_inputs.render_recipe_sha256 != receipt.render_recipe_sha256
        or build_inputs.render_seed_sha256 != receipt.render_seed_sha256
    ):
        raise ValueError("synthetic raster build inputs do not bind the receipt")
    source_receipt_raw = _read_local_raster_artifact(
        _resolve_dataset_artifact_path(
            dataset_directory=artifact_directory,
            relative_path=receipt.source_receipt_relative_path,
        ),
        maximum_size=_MAX_LOCAL_RASTER_ARTIFACT_BYTES,
    )
    try:
        source_receipt = SyntheticRasterSourceReceiptV1.model_validate_json(source_receipt_raw)
    except ValueError:
        raise ValueError("synthetic raster source receipt is invalid") from None
    if (
        source_receipt_raw != _canonical_local_json(source_receipt)
        or synthetic_raster_source_receipt_sha256(source_receipt) != receipt.source_receipt_sha256
        or source_receipt.font_manifest_sha256 != receipt.font_manifest_sha256
        or build_synthetic_raster_source_receipt(build_inputs=build_inputs) != source_receipt
    ):
        raise ValueError("synthetic raster source receipt does not bind local source PDFs")
    source_by_id = {asset.asset_id: asset for asset in source_receipt.assets}
    for asset in receipt.assets:
        if asset.split not in requested_splits:
            continue
        source_asset = source_by_id.get(asset.asset_id)
        if (
            source_asset is None
            or source_asset.family_id != asset.family_id
            or source_asset.document_type != asset.document_type
            or source_asset.source_pdf_sha256 != asset.source_pdf_sha256
            or asset.render_recipe_sha256 != receipt.render_recipe_sha256
            or asset.render_seed_sha256 != receipt.render_seed_sha256
        ):
            raise ValueError("synthetic raster asset does not bind source provenance")
        raster_raw = _read_local_raster_artifact(
            _resolve_dataset_artifact_path(
                dataset_directory=artifact_directory,
                relative_path=asset.asset_relative_path,
            ),
            maximum_size=_MAX_LOCAL_RASTER_ARTIFACT_BYTES,
        )
        observed = verify_rendered_raster_output(
            RenderedRasterBytesV1(
                asset_id=asset.asset_id,
                profile_id=asset.profile_id,
                content=raster_raw,
                width=asset.width,
                height=asset.height,
            ),
            pdfium_version=receipt.pdfium_version,
        )
        if (
            observed.content_sha256 != asset.image_sha256
            or observed.media_type != asset.media_type
            or observed.byte_size != asset.byte_size
            or observed.width != asset.width
            or observed.height != asset.height
        ):
            raise ValueError("synthetic raster asset bytes do not bind the receipt")
        gold_raw = _read_local_raster_artifact(
            _resolve_dataset_artifact_path(
                dataset_directory=artifact_directory,
                relative_path=asset.gold_relative_path,
            ),
            maximum_size=_MAX_LOCAL_RASTER_ARTIFACT_BYTES,
        )
        try:
            gold = RasterGoldArtifactV1.model_validate_json(gold_raw)
        except ValueError:
            raise ValueError("synthetic raster Gold artifact is invalid") from None
        if (
            gold_raw != _canonical_local_json(gold)
            or _sha256_bytes(gold_raw) != asset.gold_sha256
            or gold.asset_id != asset.asset_id
            or gold.family_id != asset.family_id
            or gold.document_type != asset.document_type
            or gold.profile_id != asset.profile_id
            or gold.source_pdf_sha256 != asset.source_pdf_sha256
        ):
            raise ValueError("synthetic raster Gold bytes do not bind the receipt")


def transform_pdf_bbox_to_normalized_top_left(
    pdf_bbox: PdfBoundingBoxV1,
    *,
    page_width_points: float,
    page_height_points: float,
) -> BoundingBoxV1:
    """Convert bottom-left PDF points into a top-left normalized image bbox."""

    if page_width_points <= 0 or page_height_points <= 0:
        raise ValueError("PDF page dimensions are invalid")
    return BoundingBoxV1(
        x=pdf_bbox.x / page_width_points,
        y=1 - (pdf_bbox.y + pdf_bbox.height) / page_height_points,
        width=pdf_bbox.width / page_width_points,
        height=pdf_bbox.height / page_height_points,
    )


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def verify_pdfium_geometry(
    source: SourcePdfGeometryV1,
    *,
    pdfium_spans: Sequence[PdfiumGeometrySpanV1],
) -> dict[str, PdfiumGeometrySpanV1]:
    """Bind PDFium geometry to source text 1:1, or fail before Gold exists.

    The ordered source content stream supplies text and stable source span IDs.
    PDFium supplies only the independent page geometry.  Neither side is
    allowed to fill a missing item or approximate a near match.
    """

    expected = sorted(source.spans, key=lambda span: (span.page_number, span.reading_order))
    observed = sorted(pdfium_spans, key=lambda span: (span.page_number, span.reading_order))
    if len(expected) != len(observed):
        raise ValueError("source/PDFium span count does not match")
    bound: dict[str, PdfiumGeometrySpanV1] = {}
    for source_span, pdfium_span in zip(expected, observed, strict=True):
        if (
            pdfium_span.source_pdf_sha256 != source.source_pdf_sha256
            or pdfium_span.page_number != source_span.page_number
            or pdfium_span.reading_order != source_span.reading_order
            or _normalized_text(pdfium_span.text) != _normalized_text(source_span.text)
        ):
            raise ValueError("source/PDFium text geometry does not match 1:1")
        if (
            pdfium_span.pdf_bbox.x + pdfium_span.pdf_bbox.width > source.page_width_points
            or pdfium_span.pdf_bbox.y + pdfium_span.pdf_bbox.height > source.page_height_points
        ):
            raise ValueError("PDFium geometry escapes source page bounds")
        bound[source_span.span_id] = pdfium_span
    return bound


def _union_bbox(boxes: Sequence[BoundingBoxV1]) -> BoundingBoxV1:
    if not boxes:
        raise ValueError("Gold geometry requires at least one evidence span")
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    # PDF point division can introduce insignificant binary tails.  Canonical
    # rounding makes repeated receipt/Gold compilation deterministic.
    return BoundingBoxV1(
        x=round(left, 12),
        y=round(top, 12),
        width=round(right - left, 12),
        height=round(bottom - top, 12),
    )


def _verified_table_cell_bbox(
    *,
    source: SourcePdfGeometryV1,
    cell: SourceTableCellGoldV1,
    verified_spans: Mapping[str, PdfiumGeometrySpanV1],
) -> BoundingBoxV1:
    """Use the authored cell rectangle after PDFium verifies its evidence.

    The final content stream supplies the stable rectangle.  The separately
    extracted PDFium glyph boxes must still have their centres inside that
    rectangle, otherwise the template-to-PDF correspondence is ambiguous and
    the entire Gold asset fails closed.  The emitted bbox remains the cell,
    never the smaller union of glyph boxes.
    """

    for span_id in cell.evidence_span_ids:
        evidence = verified_spans[span_id].pdf_bbox
        center_x = evidence.x + evidence.width / 2
        center_y = evidence.y + evidence.height / 2
        if not (
            cell.pdf_bbox.x <= center_x <= cell.pdf_bbox.x + cell.pdf_bbox.width
            and cell.pdf_bbox.y <= center_y <= cell.pdf_bbox.y + cell.pdf_bbox.height
        ):
            raise ValueError("PDFium table-cell evidence escapes source rectangle")
    return transform_pdf_bbox_to_normalized_top_left(
        cell.pdf_bbox,
        page_width_points=source.page_width_points,
        page_height_points=source.page_height_points,
    )


def compile_raster_gold(
    source: SourcePdfGeometryV1,
    *,
    profile_id: RasterProfileId,
    pdfium_spans: Sequence[PdfiumGeometrySpanV1],
) -> list[RasterGoldSpanV1]:
    """Compile span Gold only after a source/PDFium 1:1 geometry check."""

    if source.source_truth_kind != "final_pdf_content_stream":
        raise ValueError("raster Gold requires final PDF content-stream truth")
    geometry_by_span_id = verify_pdfium_geometry(source, pdfium_spans=pdfium_spans)
    return [
        RasterGoldSpanV1(
            asset_id=source.asset_id,
            profile_id=profile_id,
            page_number=span.page_number,
            span_id=span.span_id,
            text=span.text,
            bbox=transform_pdf_bbox_to_normalized_top_left(
                geometry_by_span_id[span.span_id].pdf_bbox,
                page_width_points=source.page_width_points,
                page_height_points=source.page_height_points,
            ),
            reading_order=span.reading_order,
            source_pdf_sha256=source.source_pdf_sha256,
        )
        for span in source.spans
    ]


def compile_raster_field_gold(
    source: SourcePdfGeometryV1,
    *,
    profile_id: RasterProfileId,
    pdfium_spans: Sequence[PdfiumGeometrySpanV1],
) -> list[RasterGoldFieldV1]:
    """Compile field Gold with bbox unioned only from verified PDFium spans."""

    verified = verify_pdfium_geometry(source, pdfium_spans=pdfium_spans)
    bbox_by_span_id = {
        span_id: transform_pdf_bbox_to_normalized_top_left(
            observation.pdf_bbox,
            page_width_points=source.page_width_points,
            page_height_points=source.page_height_points,
        )
        for span_id, observation in verified.items()
    }
    return [
        RasterGoldFieldV1(
            asset_id=source.asset_id,
            profile_id=profile_id,
            field_id=field.field_id,
            value=field.value,
            bbox=_union_bbox([bbox_by_span_id[span_id] for span_id in field.evidence_span_ids]),
            evidence_span_ids=field.evidence_span_ids,
            source_pdf_sha256=source.source_pdf_sha256,
        )
        for field in source.fields
    ]


def compile_raster_table_cell_gold(
    source: SourcePdfGeometryV1,
    *,
    profile_id: RasterProfileId,
    pdfium_spans: Sequence[PdfiumGeometrySpanV1],
) -> list[RasterGoldTableCellV1]:
    """Compile table-cell Gold with bbox unioned from verified PDFium spans."""

    verified = verify_pdfium_geometry(source, pdfium_spans=pdfium_spans)
    return [
        RasterGoldTableCellV1(
            asset_id=source.asset_id,
            profile_id=profile_id,
            table_id=cell.table_id,
            row_index=cell.row_index,
            column_index=cell.column_index,
            text=cell.text,
            bbox=_verified_table_cell_bbox(source=source, cell=cell, verified_spans=verified),
            evidence_span_ids=cell.evidence_span_ids,
            source_pdf_sha256=source.source_pdf_sha256,
        )
        for cell in source.table_cells
    ]


def deterministic_family_splits(
    family_ids: Sequence[str], *, split_seed_sha256: str
) -> dict[str, RasterSplit]:
    """Assign the fixed 750/100/50/100 family split without sample leakage."""

    if len(family_ids) != 1_000 or len(set(family_ids)) != 1_000:
        raise ValueError("synthetic-raster-v1 requires exactly 1,000 unique families")
    if re.fullmatch(_SHA256, split_seed_sha256) is None:
        raise ValueError("split seed must be a SHA-256")
    ranked = sorted(
        family_ids,
        key=lambda family_id: (
            hashlib.sha256(f"{split_seed_sha256}:{family_id}".encode("ascii")).hexdigest(),
            family_id,
        ),
    )
    assignments: dict[str, RasterSplit] = {}
    for index, family_id in enumerate(ranked):
        assignments[family_id] = (
            "train"
            if index < 750
            else "development"
            if index < 850
            else "calibration"
            if index < 900
            else "sealed"
        )
    return assignments


def validate_image_only_pdf(
    content: bytes,
    *,
    pdfium_version: str,
    expected_width: int,
    expected_height: int,
    dpi: int,
    pdfium_module: object | None = None,
) -> None:
    """Require a one-page, text-free PDF through the pinned PDFium runtime.

    Raw token scanning is not authoritative: text operators can live inside a
    compressed stream.  PDFium instead opens the produced PDF, extracts its
    page text, and re-renders it at the profile DPI so both the no-text-layer
    claim and the recorded image dimensions are independently checked.
    """

    if (
        len(content) < 9
        or not content.startswith(b"%PDF-")
        or b"%%EOF" not in content[-4096:]
        or expected_width < 1
        or expected_height < 1
        or dpi < 1
    ):
        raise ValueError("scan PDF bytes are invalid")
    pdfium = (
        pdfium_module
        if pdfium_module is not None
        else PdfiumFinalPdfGeometryExtractorV1(pdfium_version)._pdfium()
    )
    document: object | None = None
    try:
        document = pdfium.PdfDocument(content)  # type: ignore[attr-defined]
        if len(document) != 1:  # type: ignore[arg-type]
            raise ValueError("scan PDF must contain exactly one page")
        page = document[0]  # type: ignore[index]
        text_page = page.get_textpage()  # type: ignore[attr-defined]
        try:
            page_text = text_page.get_text_bounded()  # type: ignore[attr-defined]
        except AttributeError:
            page_text = text_page.get_text_range()  # type: ignore[attr-defined]
        if not isinstance(page_text, str):
            raise SyntheticRasterCompilerUnavailable("PDFium text API is unavailable")
        if page_text:
            raise ValueError("scan PDF must not contain a text layer")
        rendered_page = page.render(scale=dpi / 72)  # type: ignore[attr-defined]
        image = rendered_page.to_pil()  # type: ignore[attr-defined]
        if image.size != (expected_width, expected_height):
            raise ValueError("scan PDF dimensions do not match raster receipt")
    except SyntheticRasterCompilerUnavailable:
        raise
    except ValueError:
        raise
    except Exception:
        raise SyntheticRasterCompilerUnavailable(
            "PDFium image-only scan validation failed"
        ) from None
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class RasterEvaluationCaseV1(_WireModel):
    """One local-only candidate/Gold comparison input for a raster profile."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: RasterProfileId
    source_pdf_sha256: str = Field(pattern=_SHA256)
    image_sha256: str = Field(pattern=_SHA256)
    gold_spans: list[RasterGoldSpanV1] = Field(min_length=1, max_length=20_000)
    gold_fields: list[RasterGoldFieldV1] = Field(max_length=512)
    gold_table_cells: list[RasterGoldTableCellV1] = Field(max_length=10_000)
    candidate: RasterDocumentExtractionResultV2

    @model_validator(mode="after")
    def _binds_one_asset_profile_and_source(self) -> "RasterEvaluationCaseV1":
        if (
            self.candidate.document_asset_id != self.asset_id
            or _SYNTHETIC_RASTER_PROFILE_REVISION.fullmatch(self.candidate.raster_profile_revision)
            is None
            # Agent input is the rendered raster asset.  Its source digest
            # must therefore bind the immutable image/PDF derivative, while
            # source_pdf_sha256 separately binds final-content-stream truth.
            or self.candidate.source_sha256 != self.image_sha256
        ):
            raise ValueError("raster evaluation candidate binding is invalid")
        for collection in (self.gold_spans, self.gold_fields, self.gold_table_cells):
            if any(
                item.asset_id != self.asset_id
                or item.profile_id != self.profile_id
                or item.source_pdf_sha256 != self.source_pdf_sha256
                for item in collection
            ):
                raise ValueError("raster evaluation Gold binding is invalid")
        return self


@dataclass(frozen=True, slots=True)
class RasterMetricCountsV1:
    """Micro-average numerators retained until a sealed profile score is made."""

    field_matches: int = 0
    field_gold: int = 0
    field_candidate: int = 0
    character_errors: int = 0
    character_reference: int = 0
    table_matches: int = 0
    table_gold: int = 0
    table_candidate: int = 0
    span_matches: int = 0
    span_gold: int = 0
    span_candidate: int = 0

    def __add__(self, other: "RasterMetricCountsV1") -> "RasterMetricCountsV1":
        return RasterMetricCountsV1(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )


class RasterGateScoreV1(_WireModel):
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"]
    profile_id: RasterProfileId
    field_exact_match: float = Field(ge=0, le=1)
    # CER is edit-distance/reference-length.  Insertions can legitimately
    # make a failed candidate score greater than 1.0; clipping or rejecting
    # that raw evaluation would hide an ordinary Gate failure instead of
    # preserving a sealed ``failed`` receipt.  The activation Gate still
    # enforces the much tighter ``<= 0.02`` threshold below.
    character_error_rate: float = Field(ge=0)
    table_cell_f1: float = Field(ge=0, le=1)
    span_localization_f1: float = Field(ge=0, le=1)
    # Python's typing.Literal cannot represent floats.  Equal numeric bounds
    # retain the same fail-closed 0.5 contract for parsed evaluation records.
    bbox_iou_threshold: float = Field(default=0.5, ge=0.5, le=0.5)


def _bbox_iou(left: BoundingBoxV1 | None, right: BoundingBoxV1 | None) -> float:
    if left is None or right is None:
        return 0.0
    intersection_left = max(left.x, right.x)
    intersection_top = max(left.y, right.y)
    intersection_right = min(left.x + left.width, right.x + right.width)
    intersection_bottom = min(left.y + left.height, right.y + right.height)
    width = max(0.0, intersection_right - intersection_left)
    height = max(0.0, intersection_bottom - intersection_top)
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return 0.0 if union <= 0 else intersection / union


def _bounded_levenshtein(reference: str, candidate: str) -> int:
    """Exact CER distance with a strict work cap instead of an approximation."""

    if len(reference) * len(candidate) > 10_000_000:
        raise ValueError("raster CER input is too large for exact local evaluation")
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1] + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


_VisualCerSpanV1 = tuple[int, BoundingBoxV1 | None, int, str]
_PositionedVisualCerSpanV1 = tuple[int, BoundingBoxV1, int, str]


def _canonical_visual_cer_text(*, spans: Sequence[_VisualCerSpanV1]) -> str:
    """Return deterministic physical reading text for a raster CER comparison.

    Final-PDF content streams are an attested *text* source, but their write
    order is not a visual reading order: a renderer can draw a lower table
    row before a title or another table column. A raster provider necessarily
    observes the rendered page instead. CER therefore compares two sequences
    rebuilt from their independently observed, normalized page geometry.

    The ordering rule deliberately reads no span text. It groups nearby
    vertical centres into a physical row, orders that row left-to-right, and
    uses the already bound ordinal only for exact geometric ties. A candidate
    without a bbox cannot establish visual placement, so it is retained after
    positioned spans in provider order; it will separately fail the
    localization Gate rather than being silently discarded.
    """

    by_page: dict[int, list[_VisualCerSpanV1]] = {}
    for entry in spans:
        by_page.setdefault(entry[0], []).append(entry)

    ordered_text: list[str] = []
    for page_number in sorted(by_page):
        positioned: list[_PositionedVisualCerSpanV1] = []
        unpositioned: list[_VisualCerSpanV1] = []
        for entry in by_page[page_number]:
            page, bbox, ordinal, text = entry
            if bbox is None:
                unpositioned.append(entry)
            else:
                positioned.append((page, bbox, ordinal, text))
        positioned.sort(
            key=lambda entry: (
                entry[1].y + entry[1].height / 2,
                entry[1].x,
                entry[2],
            )
        )

        rows: list[list[_PositionedVisualCerSpanV1]] = []
        row_center_y: list[float] = []
        row_height: list[float] = []
        for entry in positioned:
            bbox = entry[1]
            center_y = bbox.y + bbox.height / 2
            if rows and abs(center_y - row_center_y[-1]) <= 0.6 * max(bbox.height, row_height[-1]):
                rows[-1].append(entry)
                row_center_y[-1] = sum(item[1].y + item[1].height / 2 for item in rows[-1]) / len(
                    rows[-1]
                )
                row_height[-1] = sum(item[1].height for item in rows[-1]) / len(rows[-1])
            else:
                rows.append([entry])
                row_center_y.append(center_y)
                row_height.append(bbox.height)

        for row in rows:
            ordered_text.extend(
                _normalized_text(entry[3])
                for entry in sorted(
                    row,
                    key=lambda entry: (
                        entry[1].x,
                        entry[1].y + entry[1].height / 2,
                        entry[2],
                    ),
                )
            )
        ordered_text.extend(
            _normalized_text(entry[3]) for entry in sorted(unpositioned, key=lambda entry: entry[2])
        )
    return "\n".join(ordered_text)


GoldItem = TypeVar("GoldItem")
CandidateItem = TypeVar("CandidateItem")


def _matched_pairs(
    *,
    gold: Sequence[GoldItem],
    candidate: Sequence[CandidateItem],
    is_match: Callable[[GoldItem, CandidateItem], bool],
    iou: Callable[[GoldItem, CandidateItem], float],
) -> int:
    """Use deterministic highest-IoU one-to-one matching for layout metrics."""

    pairs = [
        (iou(gold_item, candidate_item), gold_index, candidate_index)
        for gold_index, gold_item in enumerate(gold)
        for candidate_index, candidate_item in enumerate(candidate)
        if is_match(gold_item, candidate_item) and iou(gold_item, candidate_item) >= 0.5
    ]
    used_gold: set[int] = set()
    used_candidate: set[int] = set()
    matches = 0
    for _score, gold_index, candidate_index in sorted(
        pairs, key=lambda item: (-item[0], item[1], item[2])
    ):
        if gold_index not in used_gold and candidate_index not in used_candidate:
            used_gold.add(gold_index)
            used_candidate.add(candidate_index)
            matches += 1
    return matches


def evaluate_synthetic_raster_case(case: RasterEvaluationCaseV1) -> RasterMetricCountsV1:
    """Calculate exact local metrics; no metric can be supplied by a candidate."""

    candidate_fields = case.candidate.field_candidates
    field_matches = sum(
        1
        for gold in case.gold_fields
        if any(
            candidate.field_id == gold.field_id
            and _normalized_text(candidate.value) == _normalized_text(gold.value)
            for candidate in candidate_fields
        )
    )
    candidate_spans = [
        (page.page_number, span) for page in case.candidate.pages for span in page.spans
    ]
    span_matches = _matched_pairs(
        gold=case.gold_spans,
        candidate=candidate_spans,
        is_match=lambda gold, candidate: (
            gold.page_number == candidate[0]
            and _normalized_text(gold.text) == _normalized_text(candidate[1].text)
        ),
        iou=lambda gold, candidate: _bbox_iou(gold.bbox, candidate[1].bbox),
    )
    candidate_cells = [
        (table.table_id, table.page_number, cell)
        for table in case.candidate.tables
        for row in table.rows
        for cell in row.cells
    ]
    table_matches = _matched_pairs(
        gold=case.gold_table_cells,
        candidate=candidate_cells,
        is_match=lambda gold, candidate: (
            gold.table_id == candidate[0]
            and gold.row_index == candidate[2].row_index
            and gold.column_index == candidate[2].column_index
            and _normalized_text(gold.text) == _normalized_text(candidate[2].text)
        ),
        iou=lambda gold, candidate: _bbox_iou(gold.bbox, candidate[2].bbox),
    )
    reference = _canonical_visual_cer_text(
        spans=[
            (span.page_number, span.bbox, span.reading_order, span.text) for span in case.gold_spans
        ]
    )
    observed = _canonical_visual_cer_text(
        spans=[
            (page_number, span.bbox, span.reading_order, span.text)
            for page_number, span in candidate_spans
        ]
    )
    return RasterMetricCountsV1(
        field_matches=field_matches,
        field_gold=len(case.gold_fields),
        field_candidate=len(candidate_fields),
        character_errors=_bounded_levenshtein(reference, observed),
        character_reference=len(reference),
        table_matches=table_matches,
        table_gold=len(case.gold_table_cells),
        table_candidate=len(candidate_cells),
        span_matches=span_matches,
        span_gold=len(case.gold_spans),
        span_candidate=len(candidate_spans),
    )


def _f1(matches: int, gold: int, candidate: int) -> float:
    denominator = gold + candidate
    return 0.0 if denominator == 0 else 2 * matches / denominator


def raster_gate_score(
    *,
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"],
    profile_id: RasterProfileId,
    counts: RasterMetricCountsV1,
) -> RasterGateScoreV1:
    return RasterGateScoreV1(
        document_type=document_type,
        profile_id=profile_id,
        field_exact_match=(
            0.0
            if max(counts.field_gold, counts.field_candidate) == 0
            else counts.field_matches / max(counts.field_gold, counts.field_candidate)
        ),
        character_error_rate=(
            1.0
            if counts.character_reference == 0
            else counts.character_errors / counts.character_reference
        ),
        table_cell_f1=_f1(counts.table_matches, counts.table_gold, counts.table_candidate),
        span_localization_f1=_f1(counts.span_matches, counts.span_gold, counts.span_candidate),
    )


def evaluate_synthetic_raster_gates(
    cases: Sequence[RasterEvaluationCaseV1],
) -> list[RasterGateScoreV1]:
    """Produce exactly the 3×4 sealed gate matrix or fail before activation."""

    buckets: dict[
        tuple[Literal["bill_of_lading", "commercial_invoice", "packing_list"], RasterProfileId],
        RasterMetricCountsV1,
    ] = {}
    for case in cases:
        key = (case.document_type, case.profile_id)
        buckets[key] = buckets.get(key, RasterMetricCountsV1()) + evaluate_synthetic_raster_case(
            case
        )
    expected = {
        (document_type, profile_id)
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in RASTER_PROFILES
    }
    if set(buckets) != expected:
        raise ValueError("raster evaluation requires every document type and profile")
    return [
        raster_gate_score(
            document_type=document_type,
            profile_id=profile_id,
            counts=buckets[(document_type, profile_id)],
        )
        for document_type, profile_id in sorted(buckets)
    ]


def synthetic_raster_gate_passes(scores: Sequence[RasterGateScoreV1]) -> bool:
    """Return the immutable v1 thresholds; values are never relaxed at runtime."""

    return len(scores) == 12 and all(
        score.field_exact_match >= 0.98
        and score.character_error_rate <= 0.02
        and score.table_cell_f1 >= 0.95
        and score.span_localization_f1 >= 0.90
        and score.bbox_iou_threshold == 0.5
        for score in scores
    )


def synthetic_raster_case_manifest_sha256(cases: Sequence[RasterEvaluationCaseV1]) -> str:
    """Hash local candidate/Gold inputs without copying them to a receipt."""

    return hashlib.sha256(
        json.dumps(
            [case.model_dump(mode="json", warnings=False) for case in cases],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _lowest_calibrated_threshold(
    observations: Sequence[tuple[float, bool]], *, minimum_precision: float
) -> float:
    """Choose the lowest confidence cutoff meeting precision on labeled cases."""

    if not observations:
        raise ValueError("calibration has no confidence observations")
    for threshold in sorted({confidence for confidence, _correct in observations}):
        accepted = [correct for confidence, correct in observations if confidence >= threshold]
        if accepted and sum(accepted) / len(accepted) >= minimum_precision:
            return threshold
    raise ValueError("calibration cannot derive a safe confidence threshold")


def derive_synthetic_raster_confidence_thresholds(
    cases: Sequence[RasterEvaluationCaseV1],
) -> dict[str, float]:
    """Derive frozen review cutoffs from calibration cases, never from sealed data.

    A score is accepted only if every lower threshold would violate the stated
    precision target.  Empty/ambiguous calibration evidence fails rather than
    emitting a cosmetic threshold receipt.
    """

    classification: list[tuple[float, bool]] = []
    spans: list[tuple[float, bool]] = []
    fields: list[tuple[float, bool]] = []
    cells: list[tuple[float, bool]] = []
    for case in cases:
        candidate = case.candidate
        if candidate.classification_confidence is not None:
            classification.append(
                (candidate.classification_confidence, candidate.document_type == case.document_type)
            )
        for page in candidate.pages:
            for span in page.spans:
                spans.append(
                    (
                        span.confidence,
                        any(
                            gold.page_number == page.page_number
                            and _normalized_text(gold.text) == _normalized_text(span.text)
                            and _bbox_iou(gold.bbox, span.bbox) >= 0.5
                            for gold in case.gold_spans
                        ),
                    )
                )
        for field in candidate.field_candidates:
            fields.append(
                (
                    field.confidence,
                    any(
                        gold.field_id == field.field_id
                        and _normalized_text(gold.value) == _normalized_text(field.value)
                        for gold in case.gold_fields
                    ),
                )
            )
        for table in candidate.tables:
            for row in table.rows:
                for cell in row.cells:
                    cells.append(
                        (
                            cell.confidence,
                            any(
                                gold.table_id == table.table_id
                                and gold.row_index == cell.row_index
                                and gold.column_index == cell.column_index
                                and _normalized_text(gold.text) == _normalized_text(cell.text)
                                and _bbox_iou(gold.bbox, cell.bbox) >= 0.5
                                for gold in case.gold_table_cells
                            ),
                        )
                    )
    return {
        "classification_confidence": _lowest_calibrated_threshold(
            classification, minimum_precision=0.98
        ),
        "span_confidence": _lowest_calibrated_threshold(spans, minimum_precision=0.90),
        "field_confidence": _lowest_calibrated_threshold(fields, minimum_precision=0.98),
        "table_cell_confidence": _lowest_calibrated_threshold(cells, minimum_precision=0.95),
    }


def build_synthetic_raster_receipt(
    *,
    build_inputs: ResolvedSyntheticRasterBuildInputsV1,
    source_receipt: SyntheticRasterSourceReceiptV1,
    sources: Sequence[SourcePdfGeometryV1],
    outputs: Sequence[RasterizedOutputV1],
    gold_artifacts: Sequence[RasterGoldArtifactBindingV1],
) -> SyntheticRasterReceiptV1:
    """Bind deterministic renderer and Gold outputs to actual local provenance."""

    source_by_id = {source.asset_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("source asset IDs must be unique")
    source_receipt_by_id = {asset.asset_id: asset for asset in source_receipt.assets}
    if len(source_receipt_by_id) != len(source_receipt.assets):
        raise ValueError("source receipt asset IDs must be unique")
    if (
        source_receipt.font_manifest_sha256 != build_inputs.font_manifest_sha256
        or source_receipt.font_resource_policy != build_inputs.font_manifest.font_resource_policy
    ):
        raise ValueError("source receipt does not bind the resolved font policy")
    gold_by_asset_profile = {
        (artifact.asset_id, artifact.profile_id): artifact for artifact in gold_artifacts
    }
    if len(gold_by_asset_profile) != len(gold_artifacts):
        raise ValueError("Gold artifact asset/profile pairs must be unique")
    splits = deterministic_family_splits(
        sorted({source.family_id for source in sources}),
        split_seed_sha256=build_inputs.split_seed_sha256,
    )
    assets: list[RasterAssetReceiptV1] = []
    for output in outputs:
        source = source_by_id.get(output.asset_id)
        if source is None:
            raise ValueError("renderer output does not bind a source PDF")
        source_receipt_asset = source_receipt_by_id.get(source.asset_id)
        if (
            source_receipt_asset is None
            or source_receipt_asset.family_id != source.family_id
            or source_receipt_asset.document_type != source.document_type
            or source_receipt_asset.source_pdf_sha256 != source.source_pdf_sha256
        ):
            raise ValueError("renderer output source does not bind the source receipt")
        gold = gold_by_asset_profile.get((output.asset_id, output.profile_id))
        if gold is None:
            raise ValueError("renderer output does not bind a Gold artifact")
        assets.append(
            RasterAssetReceiptV1(
                asset_id=source.asset_id,
                family_id=source.family_id,
                document_type=source.document_type,
                split=splits[source.family_id],
                profile_id=output.profile_id,
                source_pdf_sha256=source.source_pdf_sha256,
                image_sha256=output.content_sha256,
                asset_relative_path=_raster_asset_relative_path(
                    asset_id=source.asset_id, profile_id=output.profile_id
                ),
                media_type=output.media_type,
                byte_size=output.byte_size,
                width=output.width,
                height=output.height,
                coordinate_transform="pdf-points-lower-left-to-normalized-top-left-v1",
                render_recipe_sha256=build_inputs.render_recipe_sha256,
                render_seed_sha256=build_inputs.render_seed_sha256,
                gold_relative_path=gold.gold_relative_path,
                gold_sha256=gold.gold_sha256,
            )
        )
    if set(gold_by_asset_profile) != {(output.asset_id, output.profile_id) for output in outputs}:
        raise ValueError("Gold artifacts do not bind renderer output exactly")
    source_receipt_sha256 = synthetic_raster_source_receipt_sha256(source_receipt)
    return SyntheticRasterReceiptV1(
        source_dataset_id=_SYNTHETIC_RASTER_SOURCE_DATASET_ID,
        # The old generic revision field is retained for local receipt parsing,
        # but is no longer caller-controlled: it is the actual source receipt
        # digest for this build.
        source_dataset_revision=source_receipt_sha256,
        source_receipt_sha256=source_receipt_sha256,
        source_receipt_relative_path=SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH,
        build_inputs_relative_path=build_inputs.build_inputs_relative_path,
        build_inputs_sha256=build_inputs.build_inputs_sha256,
        renderer_revision=build_inputs.renderer_configuration.renderer_revision,
        renderer_configuration_sha256=build_inputs.renderer_configuration_sha256,
        pdfium_version=build_inputs.renderer_configuration.pdfium_version,
        font_manifest_sha256=build_inputs.font_manifest_sha256,
        font_resource_policy=build_inputs.font_manifest.font_resource_policy,
        geometry_compiler_revision=build_inputs.renderer_configuration.geometry_compiler_revision,
        split_seed_sha256=build_inputs.split_seed_sha256,
        render_recipe_sha256=build_inputs.render_recipe_sha256,
        render_seed_sha256=build_inputs.render_seed_sha256,
        profiles=list(RASTER_PROFILES.values()),
        assets=assets,
    )


def synthetic_raster_receipt_sha256(receipt: SyntheticRasterReceiptV1) -> str:
    """Digest immutable receipt bytes with deterministic JSON formatting."""

    return hashlib.sha256(
        json.dumps(
            receipt.model_dump(mode="json", warnings=False),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def write_immutable_synthetic_raster_receipt(
    receipt: SyntheticRasterReceiptV1,
    *,
    output_path: Path,
) -> str:
    """Atomically write one compact local receipt and never replace evidence."""

    if output_path.name != "synthetic-raster-receipt.json" or output_path.is_symlink():
        raise ValueError("synthetic raster receipt output path is invalid")
    payload = json.dumps(
        receipt.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("synthetic raster receipt already exists") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return synthetic_raster_receipt_sha256(receipt)


@dataclass(frozen=True, slots=True)
class SyntheticRasterFixturePlanV1:
    """Small non-mutating compiler plan used by a local dataset tool."""

    profile_ids: tuple[RasterProfileId, ...] = tuple(RASTER_PROFILES)
    geometry_compiler_revision: str = "source-content-stream-geometry-v1"


__all__ = [
    "RASTER_PROFILES",
    "SYNTHETIC_RASTER_BUILD_INPUTS_RELATIVE_PATH",
    "SYNTHETIC_RASTER_RECEIPT_RELATIVE_PATH",
    "SYNTHETIC_RASTER_SOURCE_RECEIPT_RELATIVE_PATH",
    "SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH",
    "PdfBoundingBoxV1",
    "PdfiumGeometrySpanV1",
    "RasterAssetReceiptV1",
    "RasterGoldArtifactBindingV1",
    "RasterGoldArtifactV1",
    "RasterGoldFieldV1",
    "RasterGoldSpanV1",
    "RasterGoldTableCellV1",
    "RasterEvaluationCaseV1",
    "RasterGateScoreV1",
    "RasterMetricCountsV1",
    "RasterProfileId",
    "RasterProfileV1",
    "RasterSplit",
    "RasterizedOutputV1",
    "RenderedRasterBytesV1",
    "FinalPdfContentStreamParserV1",
    "PdfiumFinalPdfGeometryExtractorV1",
    "SourceFieldGoldV1",
    "SourcePdfGeometryV1",
    "SourceAttestedSyntheticContentStreamParserV1",
    "SourceTableCellGoldV1",
    "SourceTextGeometryV1",
    "ResolvedSyntheticRasterBuildInputsV1",
    "SyntheticRasterBuildInputsV1",
    "SyntheticRasterFixturePlanV1",
    "SyntheticRasterFontManifestV1",
    "SyntheticRasterCompilerUnavailable",
    "SyntheticRasterPdfiumCompiler",
    "SyntheticRasterPreflightedSourceV1",
    "SyntheticRasterRendererConfigurationV1",
    "SyntheticRasterReceiptV1",
    "SyntheticRasterSourceAssetReceiptV1",
    "SyntheticRasterSourceReceiptV1",
    "build_synthetic_raster_receipt",
    "build_synthetic_raster_source_receipt",
    "compile_synthetic_raster_dataset",
    "compile_raster_gold",
    "compile_raster_field_gold",
    "compile_raster_table_cell_gold",
    "derive_synthetic_raster_confidence_thresholds",
    "deterministic_family_splits",
    "evaluate_synthetic_raster_case",
    "evaluate_synthetic_raster_gates",
    "ensure_fixed_synthetic_raster_build_inputs",
    "preflight_synthetic_raster_sources",
    "raster_gate_score",
    "resolve_synthetic_raster_build_inputs",
    "read_verified_synthetic_raster_dataset",
    "synthetic_raster_case_manifest_sha256",
    "synthetic_raster_gate_passes",
    "synthetic_raster_receipt_sha256",
    "synthetic_raster_source_receipt_sha256",
    "transform_pdf_bbox_to_normalized_top_left",
    "validate_image_only_pdf",
    "validate_fixed_synthetic_raster_source_receipt",
    "verify_synthetic_raster_receipt_materials",
    "verify_pdfium_geometry",
    "verify_rendered_raster_output",
    "write_immutable_synthetic_raster_receipt",
    "write_immutable_synthetic_raster_source_receipt",
    "write_immutable_rendered_raster_output",
]
