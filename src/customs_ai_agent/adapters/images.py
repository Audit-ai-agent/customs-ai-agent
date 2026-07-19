from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final, Protocol, Sequence, cast
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from customs_ai_agent.schemas.models import (
    ImagePageProvenance,
    ImageMediaType,
    ImagePageInput,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_BYTES_PER_RUN,
    MAX_IMAGE_LONG_EDGE,
    MAX_IMAGE_PAGES_PER_RUN,
    PageTransform,
    PdfPagesInput,
)

_PDF_WORKER_PROTOCOL: Final = "customs-pdf-raster-worker/v1"
_PDF_WORKER_TIMEOUT_SECONDS: Final = 30.0
_MAX_PDF_WORKER_REQUEST_BYTES: Final = 30 * 1024 * 1024
_MAX_PDF_WORKER_RESPONSE_BYTES: Final = 30 * 1024 * 1024
_MAX_PDF_TOTAL_PIXELS: Final = MAX_IMAGE_LONG_EDGE**2 * MAX_IMAGE_PAGES_PER_RUN
_PDF_WORKER_SHUTDOWN_SECONDS: Final = 2.0
_SAFE_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_PDF_WORKER_ENV_ALLOWLIST: Final = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)


class ImageInputError(ValueError):
    """A content-free error for unsafe or unsupported local image input."""


class PdfRasterizerError(RuntimeError):
    """Content-free provider failure raised by a PDF rasterizer adapter."""


@dataclass(frozen=True)
class RasterizedPdfPage:
    page_number: int
    media_type: ImageMediaType
    data: bytes


@dataclass(frozen=True)
class PdfRasterizationResult:
    rasterizer_id: str
    rasterizer_version: str
    dpi: int
    pages: Sequence[RasterizedPdfPage]


class PdfRasterizer(Protocol):
    def rasterize(
        self,
        pdf_bytes: bytes,
        *,
        dpi: int,
        max_pages: int,
    ) -> PdfRasterizationResult: ...


class IsolatedPdfium2Rasterizer:
    """One-shot, bounded pypdfium2 rasterizer for untrusted local PDFs."""

    def __init__(
        self,
        *,
        python_executable: str | os.PathLike[str] | None = None,
        timeout_seconds: float = _PDF_WORKER_TIMEOUT_SECONDS,
    ) -> None:
        executable = Path(
            os.fspath(python_executable) if python_executable is not None else sys.executable
        ).expanduser()
        try:
            executable = executable.resolve(strict=True)
        except OSError:
            raise PdfRasterizerError("PDF worker Python is unavailable") from None
        if not executable.is_file() or timeout_seconds <= 0:
            raise PdfRasterizerError("PDF worker configuration is invalid")
        self._python_executable = executable
        self._timeout_seconds = timeout_seconds

    def rasterize(
        self,
        pdf_bytes: bytes,
        *,
        dpi: int,
        max_pages: int,
    ) -> PdfRasterizationResult:
        if (
            not isinstance(pdf_bytes, bytes)
            or not pdf_bytes
            or len(pdf_bytes) > MAX_IMAGE_BYTES_PER_RUN
            or not pdf_bytes.startswith(b"%PDF-")
            or not isinstance(dpi, int)
            or isinstance(dpi, bool)
            or not 72 <= dpi <= 600
            or not isinstance(max_pages, int)
            or isinstance(max_pages, bool)
            or not 1 <= max_pages <= MAX_IMAGE_PAGES_PER_RUN
        ):
            raise PdfRasterizerError("PDF worker request is invalid")
        request_id = "pdfreq_" + uuid4().hex
        source_hash = hashlib.sha256(pdf_bytes).hexdigest()
        request = {
            "protocol": _PDF_WORKER_PROTOCOL,
            "request_id": request_id,
            "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            "byte_length": len(pdf_bytes),
            "sha256": source_hash,
            "dpi": dpi,
            "max_pages": max_pages,
        }
        encoded_request = (
            json.dumps(
                request,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(encoded_request) > _MAX_PDF_WORKER_REQUEST_BYTES:
            raise PdfRasterizerError("PDF worker request exceeds the safety limit")
        raw_response = _invoke_pdf_worker(
            encoded_request,
            python_executable=self._python_executable,
            timeout_seconds=self._timeout_seconds,
        )
        return _parse_pdf_worker_response(
            raw_response,
            request_id=request_id,
            source_hash=source_hash,
            dpi=dpi,
            max_pages=max_pages,
        )


def _invoke_pdf_worker(
    encoded_request: bytes,
    *,
    python_executable: Path,
    timeout_seconds: float,
) -> bytes:
    try:
        # The interpreter path was resolved as a regular file and argv is the
        # fixed isolated PDF worker module; no shell is involved.
        process = subprocess.Popen(  # noqa: S603
            _pdf_worker_command(python_executable),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # PDFium diagnostics may include local paths or untrusted document
            # content and therefore never enter application logs.
            stderr=subprocess.DEVNULL,
            env=_pdf_worker_environment(),
            cwd=_pdf_worker_cwd(),
            bufsize=0,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
    except (OSError, ValueError) as exc:
        raise PdfRasterizerError("PDF worker failed to start") from exc
    if process.stdin is None or process.stdout is None:
        _terminate_pdf_worker(process)
        raise PdfRasterizerError("PDF worker pipes are unavailable")
    stdin = process.stdin
    stdout = process.stdout

    output: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=1)
    write_result: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

    def read_bounded() -> None:
        try:
            output.put(stdout.read(_MAX_PDF_WORKER_RESPONSE_BYTES + 1))
        except (OSError, ValueError) as exc:
            output.put(exc)

    def write_request() -> None:
        error: BaseException | None = None
        try:
            stdin.write(encoded_request)
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            error = exc
        finally:
            try:
                stdin.close()
            except OSError as exc:
                error = error or exc
            write_result.put(error)

    reader = threading.Thread(
        target=read_bounded,
        name="customs-pdf-worker-stdout",
        daemon=True,
    )
    writer = threading.Thread(
        target=write_request,
        name="customs-pdf-worker-stdin",
        daemon=True,
    )
    deadline = time.monotonic() + timeout_seconds
    reader.start()
    writer.start()
    writer.join(timeout=max(0.0, deadline - time.monotonic()))
    if writer.is_alive():
        _terminate_pdf_worker(process)
        raise PdfRasterizerError("PDF worker timed out")
    write_error = write_result.get_nowait()
    if write_error is not None:
        _terminate_pdf_worker(process)
        raise PdfRasterizerError("PDF worker request failed") from write_error

    reader.join(timeout=max(0.0, deadline - time.monotonic()))
    if reader.is_alive():
        _terminate_pdf_worker(process)
        raise PdfRasterizerError("PDF worker timed out")
    read_value = output.get_nowait()
    if isinstance(read_value, BaseException):
        _terminate_pdf_worker(process)
        raise PdfRasterizerError("PDF worker response failed") from read_value
    if len(read_value) > _MAX_PDF_WORKER_RESPONSE_BYTES:
        _terminate_pdf_worker(process)
        raise PdfRasterizerError("PDF worker response exceeds the safety limit")

    try:
        return_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        _terminate_pdf_worker(process)
        raise PdfRasterizerError("PDF worker timed out") from None
    finally:
        try:
            stdout.close()
        except OSError:
            pass
    if return_code != 0:
        raise PdfRasterizerError("PDF worker exited unsuccessfully")
    return read_value


def _parse_pdf_worker_response(
    raw_response: bytes,
    *,
    request_id: str,
    source_hash: str,
    dpi: int,
    max_pages: int,
) -> PdfRasterizationResult:
    if (
        not raw_response
        or len(raw_response) > _MAX_PDF_WORKER_RESPONSE_BYTES
        or not raw_response.endswith(b"\n")
        or raw_response.count(b"\n") != 1
    ):
        raise PdfRasterizerError("PDF worker response is malformed")
    try:
        response = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PdfRasterizerError("PDF worker response is malformed") from None
    if (
        not isinstance(response, dict)
        or response.get("protocol") != _PDF_WORKER_PROTOCOL
        or response.get("request_id") != request_id
        or response.get("status") not in {"ok", "error"}
        or set(response)
        not in (
            {"protocol", "request_id", "status", "result"},
            {"protocol", "request_id", "status", "error"},
        )
    ):
        raise PdfRasterizerError("PDF worker response is malformed")
    if response["status"] == "error":
        error = response.get("error")
        if (
            not isinstance(error, dict)
            or set(error) != {"code", "retryable"}
            or not isinstance(error.get("code"), str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error["code"]) is None
            or not isinstance(error.get("retryable"), bool)
        ):
            raise PdfRasterizerError("PDF worker response is malformed")
        raise PdfRasterizerError("PDF worker rejected the document safely")

    result = response.get("result")
    if (
        not isinstance(result, dict)
        or set(result)
        != {
            "rasterizer_id",
            "rasterizer_version",
            "dpi",
            "source_sha256",
            "pages",
        }
        or result.get("rasterizer_id") != "pypdfium2"
        or not isinstance(result.get("rasterizer_version"), str)
        or _SAFE_VERSION_PATTERN.fullmatch(result["rasterizer_version"]) is None
        or result.get("dpi") != dpi
        or result.get("source_sha256") != source_hash
        or not isinstance(result.get("pages"), list)
        or not 1 <= len(result["pages"]) <= max_pages
    ):
        raise PdfRasterizerError("PDF worker result provenance is invalid")

    pages: list[RasterizedPdfPage] = []
    total_bytes = 0
    total_pixels = 0
    for expected_number, raw_page in enumerate(result["pages"], start=1):
        if (
            not isinstance(raw_page, dict)
            or set(raw_page)
            != {
                "page_number",
                "media_type",
                "data_base64",
                "byte_length",
                "sha256",
                "width_px",
                "height_px",
            }
            or raw_page.get("page_number") != expected_number
            or raw_page.get("media_type") != "image/png"
            or not isinstance(raw_page.get("data_base64"), str)
            or not isinstance(raw_page.get("byte_length"), int)
            or isinstance(raw_page.get("byte_length"), bool)
            or not 1 <= raw_page["byte_length"] <= MAX_IMAGE_BYTES
            or not isinstance(raw_page.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_page["sha256"]) is None
            or not isinstance(raw_page.get("width_px"), int)
            or isinstance(raw_page.get("width_px"), bool)
            or not 1 <= raw_page["width_px"] <= MAX_IMAGE_LONG_EDGE
            or not isinstance(raw_page.get("height_px"), int)
            or isinstance(raw_page.get("height_px"), bool)
            or not 1 <= raw_page["height_px"] <= MAX_IMAGE_LONG_EDGE
        ):
            raise PdfRasterizerError("PDF worker page result is invalid")
        try:
            page_bytes = base64.b64decode(raw_page["data_base64"], validate=True)
        except (binascii.Error, ValueError):
            raise PdfRasterizerError("PDF worker page result is invalid") from None
        total_bytes += len(page_bytes)
        page_pixels = raw_page["width_px"] * raw_page["height_px"]
        total_pixels += page_pixels
        if (
            len(page_bytes) != raw_page["byte_length"]
            or hashlib.sha256(page_bytes).hexdigest() != raw_page["sha256"]
            or total_bytes > MAX_IMAGE_BYTES_PER_RUN
            or page_pixels > MAX_IMAGE_LONG_EDGE**2
            or total_pixels > _MAX_PDF_TOTAL_PIXELS
        ):
            raise PdfRasterizerError("PDF worker page result is invalid")
        try:
            with Image.open(BytesIO(page_bytes)) as image:
                image_format = str(image.format or "").upper()
                image_size = image.size
                image.verify()
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
            raise PdfRasterizerError("PDF worker page result is invalid") from None
        if image_format != "PNG" or image_size != (
            raw_page["width_px"],
            raw_page["height_px"],
        ):
            raise PdfRasterizerError("PDF worker page result is invalid")
        pages.append(
            RasterizedPdfPage(
                page_number=expected_number,
                media_type="image/png",
                data=page_bytes,
            )
        )
    return PdfRasterizationResult(
        rasterizer_id="pypdfium2",
        rasterizer_version=cast(str, result["rasterizer_version"]),
        dpi=dpi,
        pages=pages,
    )


def _pdf_worker_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in _PDF_WORKER_ENV_ALLOWLIST
    }
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _pdf_worker_command(python_executable: Path) -> list[str]:
    return [
        str(python_executable),
        "-I",
        "-m",
        "customs_ai_agent.pdf_worker",
    ]


def _pdf_worker_cwd() -> str:
    root = Path(tempfile.gettempdir()) / "customs-ai-pdf-worker"
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve(strict=True)
    except OSError:
        raise PdfRasterizerError("PDF worker directory is unavailable") from None
    if not resolved.is_dir() or resolved.is_symlink():
        raise PdfRasterizerError("PDF worker directory is unavailable")
    return str(resolved)


def _terminate_pdf_worker(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=_PDF_WORKER_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=_PDF_WORKER_SHUTDOWN_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            return
    except OSError:
        return


def _read_regular_file(path: str | Path, *, byte_limit: int, label: str) -> bytes:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise ImageInputError(f"{label} path does not identify a readable file") from None
    if not resolved.is_file() or resolved.is_symlink():
        raise ImageInputError(f"{label} path must be a regular non-symlink file")

    try:
        with resolved.open("rb") as handle:
            raw = handle.read(byte_limit + 1)
    except OSError:
        raise ImageInputError(f"{label} file could not be read") from None
    if not raw or len(raw) > byte_limit:
        raise ImageInputError(f"{label} file is empty or exceeds the local byte limit")
    return raw


def _image_metadata(raw: bytes) -> tuple[str, int, int, int]:
    try:
        with Image.open(BytesIO(raw)) as image:
            detected_format = str(image.format or "").upper()
            width, height = image.size
            orientation = int(image.getexif().get(274, 1) or 1)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
        raise ImageInputError("image file is not a supported decodable image") from None
    return detected_format, width, height, orientation


def _media_type(detected_format: str) -> ImageMediaType:
    media_types: dict[str, ImageMediaType] = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    media_type = media_types.get(detected_format)
    if media_type is None:
        raise ImageInputError("image format is not supported")
    return media_type


def _metadata_stripped_image_bytes(raw: bytes, detected_format: str) -> bytes:
    """Decode pixels and emit a metadata-free, single-frame image container.

    The source container is never forwarded to an external OCR/VLM provider,
    even when its EXIF orientation is already one.  Re-encoding in the same
    supported media type keeps the bounded wire contract practical for scanned
    documents while omitting EXIF, XMP, ICC, comments, PNG text and other
    source-controlled non-visual metadata.
    """

    try:
        with Image.open(BytesIO(raw)) as image:
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ImageInputError("animated or multi-frame images are not supported")
            normalized = ImageOps.exif_transpose(image)
            normalized.load()
            has_alpha = "A" in normalized.getbands() or "transparency" in normalized.info
            if detected_format == "JPEG":
                if normalized.mode not in {"RGB", "L"}:
                    normalized = normalized.convert("RGB")
                save_options: dict[str, object] = {
                    "quality": 95,
                    "subsampling": 0,
                    "optimize": False,
                    "progressive": False,
                }
            elif detected_format == "PNG":
                if has_alpha:
                    normalized = normalized.convert("RGBA")
                elif normalized.mode not in {"RGB", "L"}:
                    normalized = normalized.convert("RGB")
                save_options = {
                    "compress_level": 6,
                    "optimize": False,
                }
            elif detected_format == "WEBP":
                normalized = normalized.convert("RGBA" if has_alpha else "RGB")
                save_options = {
                    "lossless": True,
                    "quality": 100,
                    "method": 6,
                    "exact": True,
                }
            else:
                raise ImageInputError("image format is not supported")
            normalized.info.clear()
            output = BytesIO()
            normalized.save(output, format=detected_format, **save_options)
            return output.getvalue()
    except ImageInputError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        raise ImageInputError("image pixel sanitization failed") from None


def _page_from_encoded_image(
    raw: bytes,
    *,
    page_number: int,
    media_type: ImageMediaType,
    provenance: ImagePageProvenance,
) -> ImagePageInput:
    _, width, height, _ = _image_metadata(raw)
    try:
        return ImagePageInput(
            page_number=page_number,
            media_type=media_type,
            data_base64=base64.b64encode(raw).decode("ascii"),
            byte_length=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            width_px=width,
            height_px=height,
            provenance=provenance,
        )
    except ValueError:
        raise ImageInputError("image file failed the image-page safety contract") from None


def image_page_from_file(
    path: str | Path,
    *,
    page_number: int = 1,
) -> ImagePageInput:
    """Load one local evaluation/CLI image into the provider-neutral wire model.

    Paths are an edge adapter concern only. Agent requests contain validated bytes
    and provenance, never an arbitrary server-side path or URL.
    """

    source = _read_regular_file(path, byte_limit=MAX_IMAGE_BYTES, label="image")
    detected_format, _, _, orientation = _image_metadata(source)
    media_type = _media_type(detected_format)
    source_sha256 = hashlib.sha256(source).hexdigest()
    rendered = _metadata_stripped_image_bytes(source, detected_format)
    transforms: list[PageTransform] = ["metadata_strip"]
    if orientation != 1:
        transforms = ["exif_transpose", "metadata_strip"]
    return _page_from_encoded_image(
        rendered,
        page_number=page_number,
        media_type=media_type,
        provenance=ImagePageProvenance(
            source_media_type=media_type,
            source_sha256=source_sha256,
            source_page_number=1,
            transforms=transforms,
            original_exif_orientation=orientation,
        ),
    )


def _image_pages_from_pdf_bytes(
    source: bytes,
    rasterizer: PdfRasterizer,
    *,
    dpi: int = 200,
    max_pages: int = MAX_IMAGE_PAGES_PER_RUN,
) -> list[ImagePageInput]:
    if not 72 <= dpi <= 600:
        raise ImageInputError("PDF rasterization DPI is outside the safety range")
    if not 1 <= max_pages <= MAX_IMAGE_PAGES_PER_RUN:
        raise ImageInputError("PDF rasterization page limit is invalid")
    if not source.startswith(b"%PDF-"):
        raise ImageInputError("PDF file signature is invalid")
    source_sha256 = hashlib.sha256(source).hexdigest()
    try:
        result = rasterizer.rasterize(
            source,
            dpi=dpi,
            max_pages=max_pages,
        )
    except PdfRasterizerError:
        raise ImageInputError("PDF rasterizer failed") from None
    except Exception:
        raise ImageInputError("PDF rasterizer failed unexpectedly") from None
    if not isinstance(result, PdfRasterizationResult) or any(
        not isinstance(page, RasterizedPdfPage) or not isinstance(page.data, bytes)
        for page in getattr(result, "pages", ())
    ):
        raise ImageInputError("PDF rasterizer returned an invalid result type")
    if (
        not result.rasterizer_id
        or not result.rasterizer_version
        or result.dpi != dpi
        or not result.pages
        or len(result.pages) > max_pages
    ):
        raise ImageInputError("PDF rasterizer returned invalid provenance")
    page_numbers = [page.page_number for page in result.pages]
    if page_numbers != list(range(1, len(result.pages) + 1)):
        raise ImageInputError("PDF rasterizer pages must be consecutive and source ordered")

    pages = [
        _page_from_encoded_image(
            page.data,
            page_number=page.page_number,
            media_type=page.media_type,
            provenance=ImagePageProvenance(
                source_media_type="application/pdf",
                source_sha256=source_sha256,
                source_page_number=page.page_number,
                transforms=["pdf_rasterize"],
                original_exif_orientation=None,
                rasterizer_id=result.rasterizer_id,
                rasterizer_version=result.rasterizer_version,
                raster_dpi=result.dpi,
            ),
        )
        for page in result.pages
    ]
    if sum(page.byte_length for page in pages) > MAX_IMAGE_BYTES_PER_RUN:
        raise ImageInputError("PDF rasterized pages exceed the run byte limit")
    return pages


def pdf_pages_from_pdf_file(
    path: str | Path,
    rasterizer: PdfRasterizer,
    *,
    dpi: int = 200,
    max_pages: int = MAX_IMAGE_PAGES_PER_RUN,
) -> PdfPagesInput:
    """Validate/rasterize an ephemeral local PDF into the public V2 PDF envelope."""

    source = _read_regular_file(
        path,
        byte_limit=MAX_IMAGE_BYTES_PER_RUN,
        label="PDF",
    )
    pages = _image_pages_from_pdf_bytes(
        source,
        rasterizer,
        dpi=dpi,
        max_pages=max_pages,
    )
    provenance = pages[0].provenance
    try:
        return PdfPagesInput(
            source_pdf_sha256=hashlib.sha256(source).hexdigest(),
            source_pdf_byte_length=len(source),
            page_count=len(pages),
            rasterizer_id=str(provenance.rasterizer_id),
            rasterizer_version=str(provenance.rasterizer_version),
            raster_dpi=int(provenance.raster_dpi or 0),
            pages=pages,
        )
    except ValueError:
        raise ImageInputError("PDF failed the normalized V2 PDF contract") from None


def image_pages_from_pdf_file(
    path: str | Path,
    rasterizer: PdfRasterizer,
    *,
    dpi: int = 200,
    max_pages: int = MAX_IMAGE_PAGES_PER_RUN,
) -> list[ImagePageInput]:
    """Compatibility edge helper returning only the normalized raster pages."""

    return pdf_pages_from_pdf_file(
        path,
        rasterizer,
        dpi=dpi,
        max_pages=max_pages,
    ).pages


__all__ = [
    "ImageInputError",
    "IsolatedPdfium2Rasterizer",
    "PdfRasterizationResult",
    "PdfRasterizer",
    "PdfRasterizerError",
    "RasterizedPdfPage",
    "image_page_from_file",
    "image_pages_from_pdf_file",
    "pdf_pages_from_pdf_file",
]
