"""Render one fail-closed ``synthetic-raster-v1`` asset below local data.

The command accepts only a final-PDF content-stream geometry record produced
by the built-in trusted source parser for the known synthetic templates.  It
independently re-reads text and geometry through a pinned local PDFium runtime
before rendering.  It never downloads a model/package and never writes source
text or OCR output to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.synthetic_raster import (
    RASTER_PROFILES,
    SourceAttestedSyntheticContentStreamParserV1,
    SyntheticRasterCompilerUnavailable,
    SyntheticRasterPdfiumCompiler,
    resolve_synthetic_raster_build_inputs,
    write_immutable_rendered_raster_output,
)


_MAX_INPUT_BYTES = 32 * 1024 * 1024


def _preview_root(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_dir()
            or resolved.is_symlink()
            or tuple(part.casefold() for part in resolved.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise ValueError("document preview root is invalid")
    except OSError as error:
        raise ValueError("document preview root is invalid") from error
    return resolved


def _read_input(path: Path, *, limit: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("synthetic raster input is invalid")
        value = path.read_bytes()
    except OSError as error:
        raise ValueError("synthetic raster input is invalid") from error
    if not value or len(value) > limit:
        raise ValueError("synthetic raster input is invalid")
    return value


def build_fixture(
    *,
    document_preview_root: Path,
    source_pdf_path: Path,
    asset_id: str,
    family_id: str,
    document_type: str,
    profile_id: str,
    build_inputs_relative_path: str,
) -> dict[str, object]:
    root = _preview_root(document_preview_root)
    build_inputs = resolve_synthetic_raster_build_inputs(
        document_preview_root=root,
        build_inputs_relative_path=build_inputs_relative_path,
    )
    if profile_id not in RASTER_PROFILES:
        raise ValueError("synthetic raster profile is invalid")
    source_pdf = _read_input(source_pdf_path, limit=_MAX_INPUT_BYTES)
    if document_type not in {"bill_of_lading", "commercial_invoice", "packing_list"}:
        raise ValueError("synthetic raster source parser arguments are invalid")
    source = SourceAttestedSyntheticContentStreamParserV1(
        family_id=family_id,
        document_type=document_type,
    ).parse_final_pdf_content_stream(source_pdf=source_pdf, asset_id=asset_id)
    compiler = SyntheticRasterPdfiumCompiler(
        pdfium_version=build_inputs.renderer_configuration.pdfium_version,
        rasterizer_revision=build_inputs.renderer_configuration.renderer_revision,
    )
    rendered, geometry = compiler.compile(
        source_pdf=source_pdf,
        source=source,
        profile_id=profile_id,  # type: ignore[arg-type]
    )
    extension = {"image/png": "png", "image/jpeg": "jpg", "application/pdf": "pdf"}[
        RASTER_PROFILES[profile_id].media_type
    ]
    output_path = (
        root / "synthetic-raster-v1" / "assets" / f"{source.asset_id}-{profile_id}.{extension}"
    )
    output = write_immutable_rendered_raster_output(
        rendered,
        output_path=output_path,
        pdfium_version=build_inputs.renderer_configuration.pdfium_version,
    )
    return {
        "status": "synthetic_raster_asset_written",
        "asset_id": source.asset_id,
        "profile_id": profile_id,
        "image_sha256": output.content_sha256,
        "byte_size": output.byte_size,
        "pdfium_geometry_span_count": len(geometry),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument(
        "--document-type",
        choices=["bill_of_lading", "commercial_invoice", "packing_list"],
    )
    parser.add_argument("--profile", choices=sorted(RASTER_PROFILES), required=True)
    parser.add_argument("--build-inputs-relative-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = build_fixture(
            document_preview_root=arguments.document_preview_root,
            source_pdf_path=arguments.source_pdf,
            asset_id=arguments.asset_id,
            family_id=arguments.family_id,
            document_type=arguments.document_type,
            profile_id=arguments.profile,
            build_inputs_relative_path=arguments.build_inputs_relative_path,
        )
    except (OSError, ValueError, SyntheticRasterCompilerUnavailable):
        print(
            json.dumps(
                {"status": "failed", "code": "synthetic_raster_fixture_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
