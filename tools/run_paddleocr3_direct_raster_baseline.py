"""Run one local, non-approvable PaddleOCR 3 direct-raster baseline candidate.

This is a comparison helper for synthetic-raster-v1 evaluation.  It accepts
only hash-pinned local configuration/model state, writes one create-only local
candidate JSON, and never writes an approval pointer or downloads anything.
The output is expected to remain ``needs_review`` because direct P3 OCR has no
document classification, field mapping, or table-cell layout contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Sequence

from pydantic import SecretStr

from customs_ai_agent.schemas.raster_document_extraction import (
    RasterDocumentExtractionRequestMetadataV2,
    RasterPreviewInputV2,
)
from customs_ai_agent.service.document_input import inspect_raster_document_structure
from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.raster_document_extraction import extract_raster_document_preview
from customs_ai_agent.service.raster_document_preview_runtime import (
    resolve_paddleocr3_direct_raster_baseline,
)
from customs_ai_agent.service.settings import ServiceSettings


_MAX_DOCUMENT_BYTES = 32 * 1024 * 1024


def _preview_root(path: Path) -> Path:
    try:
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise ValueError("document preview root is invalid")
        return root
    except OSError as error:
        raise ValueError("document preview root is invalid") from error


def _relative_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts or ":" in value or "\\" in value:
        raise ValueError("local baseline path is invalid")
    path = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("local baseline path is invalid") from error
    return path


def _read_local(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("local baseline input is invalid")
        content = path.read_bytes()
    except OSError as error:
        raise ValueError("local baseline input is invalid") from error
    if not content or len(content) > _MAX_DOCUMENT_BYTES:
        raise ValueError("local baseline input is invalid")
    return content


def _write_create_only(path: Path, value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("local baseline output already exists") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _settings() -> ServiceSettings:
    # Only decode limits are used.  This tool never initializes the review or
    # evidence runtime and does not read environment credentials.
    token = "PaddleOcr3RasterBaselineLocalOnlyToken000000000000"
    return ServiceSettings(SecretStr(token), "http://127.0.0.1:18083", SecretStr(token))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--configuration-relative-path", required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--input-relative-path", required=True)
    parser.add_argument("--output-relative-path", required=True)
    parser.add_argument("--document-asset-id", required=True)
    parser.add_argument("--extraction-request-id", required=True)
    parser.add_argument("--raster-profile-revision", required=True)
    parser.add_argument(
        "--media-type", choices=["application/pdf", "image/jpeg", "image/png"], required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        root = _preview_root(arguments.document_preview_root)
        resolution = resolve_paddleocr3_direct_raster_baseline(
            document_preview_root=root,
            configuration_path=_relative_path(root, arguments.configuration_relative_path),
            configuration_sha256=arguments.configuration_sha256,
        )
        if resolution.engine is None or resolution.status != "available":
            raise ValueError("direct baseline is unavailable")
        content = _read_local(_relative_path(root, arguments.input_relative_path))
        digest = hashlib.sha256(content).hexdigest()
        metadata = RasterDocumentExtractionRequestMetadataV2(
            document_asset_id=arguments.document_asset_id,
            extraction_request_id=arguments.extraction_request_id,
            source_sha256=digest,
            media_type=arguments.media_type,
            raster_profile_revision=arguments.raster_profile_revision,
        )
        settings = _settings()
        result = extract_raster_document_preview(
            resolution.engine,
            raster_input=RasterPreviewInputV2(
                metadata=metadata,
                byte_size=len(content),
                content=content,
            ),
            structure=inspect_raster_document_structure(content, arguments.media_type, settings),
            settings=settings,
        )
        output_path = _relative_path(root, arguments.output_relative_path)
        digest = _write_create_only(output_path, result.model_dump(mode="json", warnings=False))
    except (OSError, ValueError, ServiceHttpError):
        print(json.dumps({"status": "failed", "code": "paddleocr3_direct_baseline_failed"}, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {"status": result.status, "candidate_sha256": digest, "engine": "paddleocr3_direct_baseline"},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
