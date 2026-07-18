from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import sys
from contextlib import redirect_stdout
from io import BytesIO, TextIOWrapper
from typing import Any, Mapping, Sequence

from customs_ai_agent.schemas.models import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_BYTES_PER_RUN,
    MAX_IMAGE_LONG_EDGE,
    MAX_IMAGE_PAGES_PER_RUN,
)

_PROTOCOL = "customs-pdf-raster-worker/v1"
_MAX_REQUEST_BYTES = 30 * 1024 * 1024
_MAX_RESPONSE_BYTES = 30 * 1024 * 1024
_MAX_TOTAL_PIXELS = MAX_IMAGE_LONG_EDGE**2 * MAX_IMAGE_PAGES_PER_RUN
_REQUEST_ID_PATTERN = re.compile(r"^pdfreq_[0-9a-f]{32}$")
_PDF_HEADER_PATTERN = re.compile(rb"^%PDF-[12]\.[0-9](?:\r\n|\r|\n)")


class _SafePdfError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _validate_request(request: object) -> tuple[str, bytes, int, int]:
    if not isinstance(request, dict):
        raise _SafePdfError("pdf_request_invalid")
    expected_keys = {
        "protocol",
        "request_id",
        "pdf_base64",
        "byte_length",
        "sha256",
        "dpi",
        "max_pages",
    }
    request_id = request.get("request_id")
    encoded = request.get("pdf_base64")
    byte_length = request.get("byte_length")
    expected_hash = request.get("sha256")
    dpi = request.get("dpi")
    max_pages = request.get("max_pages")
    if (
        set(request) != expected_keys
        or request.get("protocol") != _PROTOCOL
        or not isinstance(request_id, str)
        or _REQUEST_ID_PATTERN.fullmatch(request_id) is None
        or not isinstance(encoded, str)
        or len(encoded) > ((MAX_IMAGE_BYTES_PER_RUN + 2) // 3) * 4
        or not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or not 1 <= byte_length <= MAX_IMAGE_BYTES_PER_RUN
        or not isinstance(expected_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        or not isinstance(dpi, int)
        or isinstance(dpi, bool)
        or not 72 <= dpi <= 600
        or not isinstance(max_pages, int)
        or isinstance(max_pages, bool)
        or not 1 <= max_pages <= MAX_IMAGE_PAGES_PER_RUN
    ):
        raise _SafePdfError("pdf_request_invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _SafePdfError("pdf_request_invalid") from exc
    if len(raw) != byte_length or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise _SafePdfError("pdf_request_mismatch")
    _validate_pdf_container(raw)
    return request_id, raw, dpi, max_pages


def _validate_pdf_container(raw: bytes) -> None:
    if _PDF_HEADER_PATTERN.match(raw) is None:
        raise _SafePdfError("pdf_signature_invalid")
    eof_offset = raw.rfind(b"%%EOF")
    if eof_offset < 0:
        raise _SafePdfError("pdf_structure_invalid")
    # A PDF may contain arbitrary bytes in streams, but non-whitespace bytes
    # appended after the final EOF marker are not needed for this workflow and
    # create an avoidable PDF/polyglot ambiguity.
    if raw[eof_offset + len(b"%%EOF") :].strip(b"\x00\t\n\f\r "):
        raise _SafePdfError("pdf_polyglot_rejected")
    trailer_window = raw[max(0, eof_offset - 4096) : eof_offset]
    if re.search(rb"startxref\s+[0-9]+\s*$", trailer_window) is None:
        raise _SafePdfError("pdf_structure_invalid")


def _render_pdf(
    raw: bytes,
    *,
    dpi: int,
    max_pages: int,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_raw

        document = pdfium.PdfDocument(raw)
    except Exception:
        raise _SafePdfError("pdf_encrypted_or_corrupt") from None

    pages: list[dict[str, Any]] = []
    try:
        try:
            if int(pdfium_raw.FPDF_GetSecurityHandlerRevision(document.raw)) != -1:
                raise _SafePdfError("pdf_encrypted")
        except _SafePdfError:
            raise
        except Exception:
            raise _SafePdfError("pdf_security_check_failed") from None

        page_count = len(document)
        if not 1 <= page_count <= max_pages:
            raise _SafePdfError("pdf_page_count_invalid")

        total_pixels = 0
        # Check every page's MediaBox and effective rendered size before
        # allocating the first bitmap. A later hostile page must therefore
        # fail without partially rendering earlier pages.
        for page_index in range(page_count):
            page = document[page_index]
            try:
                media_box = page.get_mediabox()
                if (
                    not isinstance(media_box, Sequence)
                    or len(media_box) != 4
                    or not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in media_box
                    )
                ):
                    raise _SafePdfError("pdf_media_box_invalid")
                left, bottom, right, top = (float(value) for value in media_box)
                if right <= left or top <= bottom:
                    raise _SafePdfError("pdf_media_box_invalid")
                width_points, height_points = page.get_size()
                if (
                    not math.isfinite(float(width_points))
                    or not math.isfinite(float(height_points))
                    or width_points <= 0
                    or height_points <= 0
                ):
                    raise _SafePdfError("pdf_page_dimensions_invalid")
                width_px = math.ceil(float(width_points) * dpi / 72)
                height_px = math.ceil(float(height_points) * dpi / 72)
                page_pixels = width_px * height_px
                if (
                    width_px < 1
                    or height_px < 1
                    or width_px > MAX_IMAGE_LONG_EDGE
                    or height_px > MAX_IMAGE_LONG_EDGE
                    or page_pixels > MAX_IMAGE_LONG_EDGE**2
                ):
                    raise _SafePdfError("pdf_page_dimensions_invalid")
                total_pixels += page_pixels
                if total_pixels > _MAX_TOTAL_PIXELS:
                    raise _SafePdfError("pdf_total_pixels_exceeded")
            finally:
                page.close()

        total_encoded_bytes = 0
        for page_index in range(page_count):
            page = document[page_index]
            bitmap = None
            image = None
            try:
                bitmap = page.render(scale=dpi / 72)
                image = bitmap.to_pil()
                width_px, height_px = image.size
                if (
                    width_px < 1
                    or height_px < 1
                    or width_px > MAX_IMAGE_LONG_EDGE
                    or height_px > MAX_IMAGE_LONG_EDGE
                ):
                    raise _SafePdfError("pdf_render_dimensions_invalid")
                encoded = BytesIO()
                image.save(encoded, format="PNG")
                png_bytes = encoded.getvalue()
                total_encoded_bytes += len(png_bytes)
                if (
                    not png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                    or len(png_bytes) > MAX_IMAGE_BYTES
                    or total_encoded_bytes > MAX_IMAGE_BYTES_PER_RUN
                ):
                    raise _SafePdfError("pdf_render_output_too_large")
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "media_type": "image/png",
                        "data_base64": base64.b64encode(png_bytes).decode("ascii"),
                        "byte_length": len(png_bytes),
                        "sha256": hashlib.sha256(png_bytes).hexdigest(),
                        "width_px": width_px,
                        "height_px": height_px,
                    }
                )
            finally:
                if image is not None:
                    image.close()
                if bitmap is not None:
                    bitmap.close()
                page.close()
    except _SafePdfError:
        raise
    except Exception:
        raise _SafePdfError("pdf_render_failed") from None
    finally:
        document.close()

    version = str(pdfium.PYPDFIUM_INFO)
    if not version or len(version) > 128:
        raise _SafePdfError("pdf_rasterizer_version_invalid")
    return version, pages


def _process_request(request: object) -> dict[str, Any]:
    request_id: str | None = None
    if isinstance(request, Mapping) and isinstance(request.get("request_id"), str):
        request_id = request["request_id"]
    envelope: dict[str, Any] = {
        "protocol": _PROTOCOL,
        "request_id": request_id,
    }
    try:
        request_id, raw, dpi, max_pages = _validate_request(request)
        version, pages = _render_pdf(raw, dpi=dpi, max_pages=max_pages)
        envelope["request_id"] = request_id
        envelope.update(
            {
                "status": "ok",
                "result": {
                    "rasterizer_id": "pypdfium2",
                    "rasterizer_version": version,
                    "dpi": dpi,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "pages": pages,
                },
            }
        )
    except _SafePdfError as exc:
        envelope.update(
            {
                "status": "error",
                "error": {
                    "code": exc.code,
                    "retryable": False,
                },
            }
        )
    except Exception:
        envelope.update(
            {
                "status": "error",
                "error": {
                    "code": "pdf_worker_internal_error",
                    "retryable": False,
                },
            }
        )
    return envelope


def _open_protocol_stdout() -> TextIOWrapper:
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return TextIOWrapper(
        os.fdopen(protocol_fd, "wb", closefd=True),
        encoding="utf-8",
        newline="\n",
        write_through=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.parse_args(argv)
    protocol_stdout = _open_protocol_stdout()
    raw_line = sys.stdin.buffer.readline(_MAX_REQUEST_BYTES + 1)
    extra = sys.stdin.buffer.read(1)
    if len(raw_line) > _MAX_REQUEST_BYTES or extra or not raw_line.endswith(b"\n"):
        response = {
            "protocol": _PROTOCOL,
            "request_id": None,
            "status": "error",
            "error": {
                "code": "pdf_worker_request_too_large",
                "retryable": False,
            },
        }
    else:
        try:
            request = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            request = None
        with redirect_stdout(sys.stderr):
            response = _process_request(request)
    encoded = (
        json.dumps(
            response,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > _MAX_RESPONSE_BYTES:
        encoded = (
            json.dumps(
                {
                    "protocol": _PROTOCOL,
                    "request_id": response.get("request_id"),
                    "status": "error",
                    "error": {
                        "code": "pdf_worker_response_too_large",
                        "retryable": False,
                    },
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    protocol_stdout.buffer.write(encoded)
    protocol_stdout.flush()
    protocol_stdout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
