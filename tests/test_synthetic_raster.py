from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
from typing import Literal

import pytest

import customs_ai_agent.evaluation.synthetic_raster as synthetic_raster
from customs_ai_agent.evaluation.synthetic_raster import (
    PdfBoundingBoxV1,
    PdfiumFinalPdfGeometryExtractorV1,
    PdfiumGeometrySpanV1,
    RASTER_PROFILES,
    RasterAssetReceiptV1,
    RasterGoldArtifactV1,
    RasterGoldArtifactBindingV1,
    RasterizedOutputV1,
    ResolvedSyntheticRasterBuildInputsV1,
    RasterEvaluationCaseV1,
    SourcePdfGeometryV1,
    SourceTextGeometryV1,
    SyntheticRasterBuildInputsV1,
    SyntheticRasterCompilerUnavailable,
    SyntheticRasterFontManifestV1,
    SyntheticRasterRendererConfigurationV1,
    SyntheticRasterReceiptV1,
    SyntheticRasterSourceAssetReceiptV1,
    SyntheticRasterSourceReceiptV1,
    SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH,
    build_synthetic_raster_receipt,
    build_synthetic_raster_source_receipt,
    compile_synthetic_raster_dataset,
    compile_raster_field_gold,
    compile_raster_gold,
    compile_raster_table_cell_gold,
    ensure_fixed_synthetic_raster_build_inputs,
    deterministic_family_splits,
    derive_synthetic_raster_confidence_thresholds,
    evaluate_synthetic_raster_case,
    evaluate_synthetic_raster_gates,
    enumerate_synthetic_raster_source_assets,
    resolve_synthetic_raster_build_inputs,
    synthetic_raster_gate_passes,
    synthetic_raster_source_receipt_sha256,
    transform_pdf_bbox_to_normalized_top_left,
    validate_image_only_pdf,
    verify_pdfium_geometry,
    verify_rendered_raster_output,
    RenderedRasterBytesV1,
    write_immutable_synthetic_raster_source_receipt,
    write_immutable_synthetic_raster_receipt,
)
from customs_ai_agent.schemas.raster_document_extraction import (
    RasterDocumentExtractionResultV2,
    RasterDocumentPreviewPageV2,
    RasterFieldCandidateV2,
    RasterTableV2,
    canonical_raster_preview_text_sha256,
)
from customs_ai_agent.service.raster_document_preview_runtime import (
    PpStructureV3OfflineProfileV2,
    RasterCandidateEngineResolutionV1,
)


_HASH = "a" * 64
_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9J4j8AAAAASUVORK5CYII="
)


class _FakeScanPdfium:
    def __init__(self, *, text: str, width: int = 200, height: int = 400) -> None:
        self._text = text
        self._width = width
        self._height = height

    def PdfDocument(self, _content: bytes) -> object:  # noqa: N802 - mirrors PDFium
        fake = self

        class _TextPage:
            def get_text_range(self) -> str:
                return fake._text

        class _RenderedPage:
            def to_pil(self) -> object:
                class _Image:
                    size = (fake._width, fake._height)

                return _Image()

        class _Page:
            def get_textpage(self) -> object:
                return _TextPage()

            def render(self, *, scale: float) -> object:
                assert scale == 3.0
                return _RenderedPage()

        class _Document:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> object:
                assert index == 0
                return _Page()

            def close(self) -> None:
                return None

        return _Document()


def _source(index: int) -> SourcePdfGeometryV1:
    return SourcePdfGeometryV1(
        asset_id=f"asset-{index}",
        family_id=f"family-{index}",
        document_type="commercial_invoice",
        source_pdf_sha256=_HASH,
        page_width_points=100,
        page_height_points=200,
        spans=[
            SourceTextGeometryV1(
                span_id="span-1",
                page_number=1,
                text="COMMERCIAL INVOICE",
                pdf_bbox=PdfBoundingBoxV1(x=10, y=20, width=30, height=40),
                reading_order=1,
            )
        ],
        fields=[
            {
                "field_id": "invoice_number",
                "value": "COMMERCIAL INVOICE",
                "evidence_span_ids": ["span-1"],
            }
        ],
        table_cells=[
            {
                "table_id": "items",
                "row_index": 0,
                "column_index": 0,
                "text": "COMMERCIAL INVOICE",
                "pdf_bbox": {"x": 5, "y": 10, "width": 50, "height": 60},
                "evidence_span_ids": ["span-1"],
            }
        ],
    )


def _pdfium_span() -> list[PdfiumGeometrySpanV1]:
    return [
        PdfiumGeometrySpanV1(
            page_number=1,
            text="COMMERCIAL INVOICE",
            pdf_bbox=PdfBoundingBoxV1(x=10, y=20, width=30, height=40),
            reading_order=1,
            source_pdf_sha256=_HASH,
        )
    ]


def _resolved_build_inputs(
    tmp_path: Path,
    *,
    geometry_compiler_revision: Literal[
        "source-content-stream-geometry-v1",
        "source-content-stream-geometry-v2-font-encoding",
    ] = "source-content-stream-geometry-v1",
) -> ResolvedSyntheticRasterBuildInputsV1:
    return ResolvedSyntheticRasterBuildInputsV1(
        document_preview_root=tmp_path,
        source_root=tmp_path / SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH,
        build_inputs_relative_path="synthetic-raster-build/inputs.json",
        build_inputs_sha256="b" * 64,
        renderer_configuration=SyntheticRasterRendererConfigurationV1(
            renderer_revision="pdfium-raster-v1",
            pdfium_version="4.30.0",
            geometry_compiler_revision=geometry_compiler_revision,
            profiles=list(RASTER_PROFILES.values()),
        ),
        renderer_configuration_sha256="e" * 64,
        font_manifest=SyntheticRasterFontManifestV1(),
        font_manifest_sha256="c" * 64,
        split_seed_sha256=_HASH,
        render_recipe_sha256="f" * 64,
        render_seed_sha256="1" * 64,
    )


def _fixed_source_receipt(
    *, font_manifest_sha256: str, source_pdf_sha256: str
) -> SyntheticRasterSourceReceiptV1:
    documents = {
        "bl": ("BL", "bill_of_lading"),
        "inv": ("INV", "commercial_invoice"),
        "pl": ("PL", "packing_list"),
    }
    assets: list[SyntheticRasterSourceAssetReceiptV1] = []
    for index in range(1, 1_001):
        for fixture_kind, directory, prefix in (
            ("clean", "clean_sets", "C"),
            ("error", "error_sets", "E"),
        ):
            for short_type, (filename_type, document_type) in documents.items():
                assets.append(
                    SyntheticRasterSourceAssetReceiptV1(
                        asset_id=f"{fixture_kind}-{index:04d}-{short_type}",
                        family_id=f"family-{index:04d}",
                        fixture_kind=fixture_kind,
                        document_type=document_type,
                        source_relative_path=(
                            f"{directory}/set_{index:04d}/{prefix}{index:04d}_{filename_type}.pdf"
                        ),
                        source_pdf_sha256=source_pdf_sha256,
                        byte_size=1,
                    )
                )
    return SyntheticRasterSourceReceiptV1(
        font_manifest_sha256=font_manifest_sha256,
        assets=assets,
    )


def _write_local_build_inputs(root: Path) -> str:
    relative_directory = "synthetic-raster-build"
    directory = root / relative_directory
    directory.mkdir(parents=True)
    renderer_payload = {
        "schema_version": "1.0",
        "renderer_kind": "pdfium",
        "renderer_revision": "pdfium-raster-v1",
        "pdfium_version": "4.30.0",
        "geometry_compiler_revision": "source-content-stream-geometry-v1",
        "coordinate_transform": "pdf-points-lower-left-to-normalized-top-left-v1",
        "profiles": [profile.model_dump(mode="json") for profile in RASTER_PROFILES.values()],
    }
    files: dict[str, bytes] = {
        "renderer.json": json.dumps(renderer_payload, separators=(",", ":")).encode("utf-8"),
        "font.json": b'{"schema_version":"1.0","font_resource_policy":"final_pdf_font_resources_only"}',
        "split.seed": b"fixed-family-split-seed-v1",
        "recipe.json": b'{"renderer":"pdfium-raster-v1"}',
        "render.seed": b"fixed-render-seed-v1",
    }
    for name, content in files.items():
        (directory / name).write_bytes(content)
    build_inputs_relative_path = f"{relative_directory}/inputs.json"
    inputs = {
        "schema_version": "1.0",
        "source_root_relative_path": SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH,
        "renderer_configuration_relative_path": f"{relative_directory}/renderer.json",
        "font_manifest_relative_path": f"{relative_directory}/font.json",
        "split_seed_relative_path": f"{relative_directory}/split.seed",
        "render_recipe_relative_path": f"{relative_directory}/recipe.json",
        "render_seed_relative_path": f"{relative_directory}/render.seed",
    }
    (root / build_inputs_relative_path).write_bytes(
        json.dumps(inputs, separators=(",", ":")).encode("utf-8")
    )
    return build_inputs_relative_path


def test_geometry_compiler_uses_source_content_stream_and_flips_pdf_y_axis() -> None:
    bbox = transform_pdf_bbox_to_normalized_top_left(
        PdfBoundingBoxV1(x=10, y=20, width=30, height=40),
        page_width_points=100,
        page_height_points=200,
    )
    assert bbox.model_dump() == {"x": 0.1, "y": 0.7, "width": 0.3, "height": 0.2}

    source = _source(1)
    gold = compile_raster_gold(source, profile_id="jpeg-216-q75", pdfium_spans=_pdfium_span())
    assert gold[0].text == "COMMERCIAL INVOICE"
    assert gold[0].bbox == bbox
    assert gold[0].source_pdf_sha256 == _HASH
    assert (
        compile_raster_field_gold(source, profile_id="jpeg-216-q75", pdfium_spans=_pdfium_span())[
            0
        ].bbox
        == bbox
    )
    cell_gold = compile_raster_table_cell_gold(
        source, profile_id="jpeg-216-q75", pdfium_spans=_pdfium_span()
    )[0]
    assert cell_gold.row_index == 0
    assert cell_gold.bbox == transform_pdf_bbox_to_normalized_top_left(
        PdfBoundingBoxV1(x=5, y=10, width=50, height=60),
        page_width_points=100,
        page_height_points=200,
    )
    assert cell_gold.bbox != bbox
    with pytest.raises(ValueError, match="table-cell evidence escapes"):
        compile_raster_table_cell_gold(
            source,
            profile_id="jpeg-216-q75",
            pdfium_spans=[
                PdfiumGeometrySpanV1(
                    page_number=1,
                    text="COMMERCIAL INVOICE",
                    pdf_bbox=PdfBoundingBoxV1(x=90, y=20, width=5, height=5),
                    reading_order=1,
                    source_pdf_sha256=_HASH,
                )
            ],
        )
    with pytest.raises(ValueError, match="does not match 1:1"):
        verify_pdfium_geometry(
            source,
            pdfium_spans=[
                PdfiumGeometrySpanV1(
                    page_number=1,
                    text="WRONG",
                    pdf_bbox=PdfBoundingBoxV1(x=10, y=20, width=30, height=40),
                    reading_order=1,
                    source_pdf_sha256=_HASH,
                )
            ],
        )


def test_pdfium_geometry_alignment_matches_duplicate_text_by_content_stream_origin() -> None:
    extractor = PdfiumFinalPdfGeometryExtractorV1(pdfium_version="4.30.0")
    assert extractor._layout_text("ITEM\u00a0\nNO.") == "ITEMNO."

    class _TextPage:
        def __init__(self, boxes: dict[int, tuple[float, float, float, float]]) -> None:
            self._boxes = boxes

        def get_charbox(self, index: int) -> tuple[float, float, float, float]:
            return self._boxes[index]

    boxes = {
        0: (0, 0, 1, 1),
        1: (1, 0, 2, 1),
        3: (50, 0, 51, 1),
        4: (51, 0, 52, 1),
        6: (100, 0, 101, 1),
        7: (101, 0, 102, 1),
    }
    spans = [
        SourceTextGeometryV1(
            span_id="span-1",
            page_number=1,
            text="AB",
            content_stream_x=100,
            content_stream_y=0,
            reading_order=1,
        ),
        SourceTextGeometryV1(
            span_id="span-2",
            page_number=1,
            text="CD",
            content_stream_x=50,
            content_stream_y=0,
            reading_order=2,
        ),
        SourceTextGeometryV1(
            span_id="span-3",
            page_number=1,
            text="AB",
            content_stream_x=0,
            content_stream_y=0,
            reading_order=3,
        ),
    ]
    matched = extractor._match_spans_by_content_stream_origin(
        text_page=_TextPage(boxes), page_text="AB CD AB", source_spans=spans
    )
    assert [(span.span_id, bbox.x) for span, bbox in matched] == [
        ("span-1", 100),
        ("span-2", 50),
        ("span-3", 0),
    ]

    ambiguous_boxes = {**boxes, 6: (0, 0, 1, 1), 7: (1, 0, 2, 1)}
    with pytest.raises(ValueError, match="does not match"):
        extractor._match_spans_by_content_stream_origin(
            text_page=_TextPage(ambiguous_boxes),
            page_text="AB CD AB",
            source_spans=[spans[2]],
        )


def test_final_pdf_text_decoder_uses_final_pdf_font_resources_without_host_fonts() -> None:
    source_pdf = b"""
        %PDF-1.3
        2 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >> endobj
        3 0 obj << /Type /Font /Subtype /Type1 /BaseFont /ZapfDingbats >> endobj
        4 0 obj << /Font << /F1 2 0 R /F2 3 0 R >> >> endobj
    """
    decoder = synthetic_raster._final_pdf_text_run_decoder(source_pdf)

    assert decoder("F1", "ABC") == "ABC"
    assert decoder("F2", chr(172) + "n") == "①■"
    with pytest.raises(ValueError, match="glyph is unsupported"):
        decoder("F2", chr(182))
    with pytest.raises(ValueError, match="font is not declared"):
        decoder("F9", "ABC")


def test_fixed_family_split_is_deterministic_and_prevents_cross_family_leakage() -> None:
    families = [f"family-{index}" for index in range(1_000)]
    first = deterministic_family_splits(families, split_seed_sha256=_HASH)
    second = deterministic_family_splits(list(reversed(families)), split_seed_sha256=_HASH)

    assert first == second
    assert Counter(first.values()) == {
        "train": 750,
        "development": 100,
        "calibration": 50,
        "sealed": 100,
    }
    with pytest.raises(ValueError, match="exactly 1,000"):
        deterministic_family_splits(families[:-1], split_seed_sha256=_HASH)


def test_receipt_binds_output_hash_profile_and_family_split_without_gold_text(
    tmp_path: Path,
) -> None:
    bound_source = _source(0).model_copy(
        update={
            "asset_id": "clean-0001-bl",
            "family_id": "family-0001",
            "document_type": "bill_of_lading",
        }
    )
    sources = [bound_source, *[_source(index) for index in range(1, 1_000)]]
    source_receipt = _fixed_source_receipt(
        font_manifest_sha256="c" * 64,
        source_pdf_sha256=_HASH,
    )
    receipt = build_synthetic_raster_receipt(
        build_inputs=_resolved_build_inputs(
            tmp_path,
            geometry_compiler_revision="source-content-stream-geometry-v2-font-encoding",
        ),
        source_receipt=source_receipt,
        sources=sources,
        outputs=[
            RasterizedOutputV1(
                asset_id="clean-0001-bl",
                profile_id="png-216",
                media_type="image/png",
                content_sha256="d" * 64,
                byte_size=10,
                width=300,
                height=600,
            )
        ],
        gold_artifacts=[
            RasterGoldArtifactBindingV1(
                asset_id="clean-0001-bl",
                profile_id="png-216",
                gold_relative_path="synthetic-raster-v1/gold/clean-0001-bl-png-216.json",
                gold_sha256="9" * 64,
            )
        ],
    )

    asset = receipt.assets[0]
    assert asset.source_pdf_sha256 == _HASH
    assert asset.image_sha256 == "d" * 64
    assert asset.asset_relative_path == "synthetic-raster-v1/assets/clean-0001-bl-png-216.png"
    assert asset.gold_sha256 == "9" * 64
    assert asset.gold_relative_path == "synthetic-raster-v1/gold/clean-0001-bl-png-216.json"
    assert asset.coordinate_transform == "pdf-points-lower-left-to-normalized-top-left-v1"
    assert asset.render_recipe_sha256 == "f" * 64
    assert receipt.renderer_configuration_sha256 == "e" * 64
    assert receipt.geometry_compiler_revision == "source-content-stream-geometry-v2-font-encoding"
    assert receipt.source_receipt_sha256 == synthetic_raster_source_receipt_sha256(source_receipt)
    assert receipt.build_inputs_sha256 == "b" * 64
    assert {profile.profile_id for profile in receipt.profiles} == {
        "png-216",
        "jpeg-216-q75",
        "jpeg-150-q75",
        "scan-pdf-216-q75",
    }
    assert "COMMERCIAL INVOICE" not in receipt.model_dump_json()
    output_path = tmp_path / "synthetic-raster-receipt.json"
    assert len(write_immutable_synthetic_raster_receipt(receipt, output_path=output_path)) == 64
    with pytest.raises(ValueError, match="already exists"):
        write_immutable_synthetic_raster_receipt(receipt, output_path=output_path)


def test_receipt_replay_rejects_a_geometry_compiler_revision_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    (root / "synthetic-raster-v1").mkdir(parents=True)
    build_inputs = _resolved_build_inputs(
        root,
        geometry_compiler_revision="source-content-stream-geometry-v2-font-encoding",
    )
    bound_source = _source(0).model_copy(
        update={
            "asset_id": "clean-0001-bl",
            "family_id": "family-0001",
            "document_type": "bill_of_lading",
        }
    )
    source_receipt = _fixed_source_receipt(
        font_manifest_sha256="c" * 64,
        source_pdf_sha256=_HASH,
    )
    receipt = build_synthetic_raster_receipt(
        build_inputs=build_inputs,
        source_receipt=source_receipt,
        sources=[bound_source, *[_source(index) for index in range(1, 1_000)]],
        outputs=[
            RasterizedOutputV1(
                asset_id="clean-0001-bl",
                profile_id="png-216",
                media_type="image/png",
                content_sha256="d" * 64,
                byte_size=10,
                width=300,
                height=600,
            )
        ],
        gold_artifacts=[
            RasterGoldArtifactBindingV1(
                asset_id="clean-0001-bl",
                profile_id="png-216",
                gold_relative_path="synthetic-raster-v1/gold/clean-0001-bl-png-216.json",
                gold_sha256="9" * 64,
            )
        ],
    ).model_copy(update={"geometry_compiler_revision": "source-content-stream-geometry-v1"})
    monkeypatch.setattr(
        synthetic_raster,
        "resolve_synthetic_raster_build_inputs",
        lambda **_kwargs: build_inputs,
    )

    with pytest.raises(ValueError, match="build inputs do not bind"):
        synthetic_raster.verify_synthetic_raster_receipt_materials(
            document_preview_root=root,
            receipt=receipt,
            required_splits=("sealed",),
        )


def test_build_inputs_resolve_only_local_actual_bytes_and_fixed_renderer_semantics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    source_root = root / SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH
    source_root.mkdir(parents=True)
    inputs_relative_path = _write_local_build_inputs(root)

    resolved = resolve_synthetic_raster_build_inputs(
        document_preview_root=root,
        build_inputs_relative_path=inputs_relative_path,
        pdfium_version_provider=lambda: "4.30.0",
    )
    assert resolved.source_root == source_root.resolve()
    assert resolved.renderer_configuration.pdfium_version == "4.30.0"
    assert resolved.renderer_configuration.renderer_revision == "pdfium-raster-v1"
    assert resolved.split_seed_sha256 == hashlib.sha256(b"fixed-family-split-seed-v1").hexdigest()
    assert resolved.font_manifest.font_resource_policy == "final_pdf_font_resources_only"

    font_path = root / "synthetic-raster-build" / "font.json"
    font_path.write_bytes(
        b'{"schema_version":"1.0","font_resource_policy":"final_pdf_font_resources_only",'
        b'"host_font_path":"C:/fonts/forbidden.ttf"}'
    )
    with pytest.raises(ValueError, match="build JSON is invalid"):
        resolve_synthetic_raster_build_inputs(
            document_preview_root=root,
            build_inputs_relative_path=inputs_relative_path,
            pdfium_version_provider=lambda: "4.30.0",
        )
    font_path.write_bytes(
        b'{"schema_version":"1.0","font_resource_policy":"final_pdf_font_resources_only"}'
    )
    renderer_path = root / "synthetic-raster-build" / "renderer.json"
    renderer_payload = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer_payload["coordinate_transform"] = "wrong-coordinate-transform"
    renderer_path.write_text(json.dumps(renderer_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="build JSON is invalid"):
        resolve_synthetic_raster_build_inputs(
            document_preview_root=root,
            build_inputs_relative_path=inputs_relative_path,
            pdfium_version_provider=lambda: "4.30.0",
        )
    with pytest.raises(ValueError, match="fixed 6,000-PDF tree"):
        SyntheticRasterBuildInputsV1.model_validate(
            {
                "source_root_relative_path": "other-source-root",
                "renderer_configuration_relative_path": "renderer.json",
                "font_manifest_relative_path": "font.json",
                "split_seed_relative_path": "split.seed",
                "render_recipe_relative_path": "recipe.json",
                "render_seed_relative_path": "render.seed",
            }
        )
    with pytest.raises(ValueError):
        SyntheticRasterBuildInputsV1.model_validate(
            {
                "source_root_relative_path": SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH,
                "renderer_configuration_relative_path": "renderer.json",
                "font_manifest_relative_path": "font.json",
                "split_seed_relative_path": "split.seed",
                "render_recipe_relative_path": "recipe.json",
                "render_seed_relative_path": "render.seed",
                "source_receipt_sha256": "a" * 64,
            }
        )
    builder_source = (
        Path(__file__).parents[1] / "tools" / "build_synthetic_raster_dataset.py"
    ).read_text(encoding="utf-8")
    for forbidden_flag in (
        "--pdfium-version",
        "--source-root",
        "--source-dataset-id",
        "--source-dataset-revision",
        "--source-receipt-sha256",
        "--renderer-configuration-sha256",
        "--font-manifest-sha256",
        "--split-seed-sha256",
        "--render-recipe-sha256",
        "--render-seed-sha256",
    ):
        assert forbidden_flag not in builder_source
    fixture_source = (
        Path(__file__).parents[1] / "tools" / "build_synthetic_raster_fixture.py"
    ).read_text(encoding="utf-8")
    assert "--pdfium-version" not in fixture_source


def test_fixed_build_inputs_are_create_or_verify_without_caller_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    (root / SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH).mkdir(parents=True)
    legacy_marker = root / "synthetic-raster-build" / "inputs.json"
    legacy_marker.parent.mkdir()
    legacy_marker.write_bytes(b'{"legacy":"immutable"}')
    monkeypatch.setattr(synthetic_raster, "_installed_pdfium_version", lambda: "4.30.0")

    first = ensure_fixed_synthetic_raster_build_inputs(document_preview_root=root)
    assert first.build_inputs_relative_path == "synthetic-raster-build-v2/inputs.json"
    assert first.renderer_configuration.geometry_compiler_revision == (
        "source-content-stream-geometry-v2-font-encoding"
    )
    assert (root / "synthetic-raster-build-v2" / "renderer.json").is_file()
    assert legacy_marker.read_bytes() == b'{"legacy":"immutable"}'
    assert ensure_fixed_synthetic_raster_build_inputs(document_preview_root=root) == first

    (root / "synthetic-raster-build-v2" / "split.seed").write_bytes(b"drift")
    with pytest.raises(ValueError, match="conflicts with local bytes"):
        ensure_fixed_synthetic_raster_build_inputs(document_preview_root=root)

    unavailable_root = tmp_path / "missing" / "_local" / "data" / "document-preview"
    unavailable_root.mkdir(parents=True)

    def _unavailable() -> str:
        raise SyntheticRasterCompilerUnavailable("pinned PDFium runtime is unavailable")

    monkeypatch.setattr(synthetic_raster, "_installed_pdfium_version", _unavailable)
    with pytest.raises(SyntheticRasterCompilerUnavailable):
        ensure_fixed_synthetic_raster_build_inputs(document_preview_root=unavailable_root)
    assert not (unavailable_root / "synthetic-raster-build-v2").exists()


def test_dataset_builder_reports_offline_pdfium_dependency_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)

    def _unavailable() -> str:
        raise SyntheticRasterCompilerUnavailable("pinned PDFium runtime is unavailable")

    monkeypatch.setattr(synthetic_raster, "_installed_pdfium_version", _unavailable)
    builder = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "build_synthetic_raster_dataset.py")
    )
    assert builder["main"](["--document-preview-root", str(root)]) == 2
    assert not (root / "synthetic-raster-build").exists()
    assert not (root / "synthetic-raster-v1").exists()


def test_dataset_preflight_failure_creates_no_staging_or_final_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    resolved = _resolved_build_inputs(root)
    monkeypatch.setattr(
        synthetic_raster, "resolve_synthetic_raster_build_inputs", lambda **_kwargs: resolved
    )
    monkeypatch.setattr(
        synthetic_raster, "enumerate_synthetic_raster_source_assets", lambda _root: ()
    )
    monkeypatch.setattr(
        synthetic_raster, "build_synthetic_raster_source_receipt", lambda **_kwargs: object()
    )

    def _reject_preflight(**_kwargs: object) -> tuple[object, ...]:
        raise ValueError("preflight rejected source")

    monkeypatch.setattr(synthetic_raster, "preflight_synthetic_raster_sources", _reject_preflight)
    with pytest.raises(ValueError, match="preflight rejected source"):
        compile_synthetic_raster_dataset(
            document_preview_root=root,
            build_inputs_relative_path=resolved.build_inputs_relative_path,
        )
    assert not (root / "synthetic-raster-v1").exists()
    assert not list(root.glob(".synthetic-raster-v1-staging-*"))


def test_dataset_build_lock_is_reusable_without_stale_marker_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)

    first = synthetic_raster._acquire_synthetic_raster_build_lock(document_preview_root=root)
    synthetic_raster._release_owned_synthetic_raster_build_lock(first)
    marker = root / ".synthetic-raster-v1-build.lock"
    assert marker.read_bytes() == b"\0"

    second = synthetic_raster._acquire_synthetic_raster_build_lock(document_preview_root=root)
    synthetic_raster._release_owned_synthetic_raster_build_lock(second)


def test_staged_dataset_publish_is_atomic_and_failed_publish_keeps_final_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    staging_parent, staged_dataset = synthetic_raster._create_synthetic_raster_staging_directory(
        document_preview_root=root
    )
    (staged_dataset / "marker").write_bytes(b"verified")
    synthetic_raster._publish_synthetic_raster_staging_directory(
        document_preview_root=root,
        staging_parent=staging_parent,
        dataset_directory=staged_dataset,
    )
    assert (root / "synthetic-raster-v1" / "marker").read_bytes() == b"verified"
    assert not staging_parent.exists()

    failed_root = tmp_path / "failed" / "_local" / "data" / "document-preview"
    failed_root.mkdir(parents=True)
    failed_parent, failed_dataset = synthetic_raster._create_synthetic_raster_staging_directory(
        document_preview_root=failed_root
    )
    original_rename = synthetic_raster.os.rename

    def _fail_rename(_source: object, _destination: object) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(synthetic_raster.os, "rename", _fail_rename)
    with pytest.raises(ValueError, match="publish failed"):
        synthetic_raster._publish_synthetic_raster_staging_directory(
            document_preview_root=failed_root,
            staging_parent=failed_parent,
            dataset_directory=failed_dataset,
        )
    assert not (failed_root / "synthetic-raster-v1").exists()
    monkeypatch.setattr(synthetic_raster.os, "rename", original_rename)
    synthetic_raster._discard_owned_synthetic_raster_staging_directory(
        document_preview_root=failed_root, staging_parent=failed_parent
    )


def test_existing_partial_dataset_is_never_overwritten_or_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    partial = root / "synthetic-raster-v1"
    partial.mkdir(parents=True)
    marker = partial / "partial-marker"
    marker.write_bytes(b"do-not-replace")
    resolved = _resolved_build_inputs(root)
    monkeypatch.setattr(
        synthetic_raster, "resolve_synthetic_raster_build_inputs", lambda **_kwargs: resolved
    )
    with pytest.raises(ValueError):
        compile_synthetic_raster_dataset(
            document_preview_root=root,
            build_inputs_relative_path=resolved.build_inputs_relative_path,
        )
    assert marker.read_bytes() == b"do-not-replace"


def test_fixed_6000_source_tree_is_hashed_into_a_create_only_canonical_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    source_root = root / SYNTHETIC_RASTER_SOURCE_ROOT_RELATIVE_PATH
    source_pdf = b"%PDF-1.7\n/Font << /F1 1 0 R >>\n/BaseFont /Helvetica\n%%EOF"
    for index in range(1, 1_001):
        for directory, prefix in (("clean_sets", "C"), ("error_sets", "E")):
            output_directory = source_root / directory / f"set_{index:04d}"
            output_directory.mkdir(parents=True, exist_ok=True)
            for suffix in ("BL", "INV", "PL"):
                (output_directory / f"{prefix}{index:04d}_{suffix}.pdf").write_bytes(source_pdf)
    resolved = _resolved_build_inputs(root)

    receipt = build_synthetic_raster_source_receipt(build_inputs=resolved)
    assert len(receipt.assets) == 6_000
    assert receipt.assets[0].source_relative_path == "clean_sets/set_0001/C0001_BL.pdf"
    assert receipt.assets[-1].source_relative_path == "error_sets/set_1000/E1000_PL.pdf"
    assert receipt.assets[0].source_pdf_sha256 == hashlib.sha256(source_pdf).hexdigest()
    receipt_path = root / "synthetic-raster-v1" / "source-receipt.json"
    assert write_immutable_synthetic_raster_source_receipt(
        receipt, output_path=receipt_path
    ) == synthetic_raster_source_receipt_sha256(receipt)
    with pytest.raises(ValueError, match="already exists"):
        write_immutable_synthetic_raster_source_receipt(receipt, output_path=receipt_path)


def test_scan_pdf_guard_uses_pdfium_text_extraction_not_raw_tokens() -> None:
    # The raw PDF bytes deliberately contain text operators.  A fake PDFium
    # response with no extracted text proves the guard does not use raw token
    # scanning; a compressed real stream receives the same authoritative path.
    validate_image_only_pdf(
        b"%PDF-1.7\nBT /F1 12 Tf (opaque) Tj ET\n%%EOF",
        pdfium_version="4.30.0",
        expected_width=200,
        expected_height=400,
        dpi=216,
        pdfium_module=_FakeScanPdfium(text=""),
    )
    with pytest.raises(ValueError, match="text layer"):
        validate_image_only_pdf(
            b"%PDF-1.7\n/Image /XObject\n%%EOF",
            pdfium_version="4.30.0",
            expected_width=200,
            expected_height=400,
            dpi=216,
            pdfium_module=_FakeScanPdfium(text="hello"),
        )


def test_rendered_output_validates_each_profile_before_receipt() -> None:
    scan_pdf = verify_rendered_raster_output(
        RenderedRasterBytesV1(
            asset_id="asset-1",
            profile_id="scan-pdf-216-q75",
            content=b"%PDF-1.7\n/Image /XObject\n%%EOF",
            width=200,
            height=400,
        ),
        pdfium_version="4.30.0",
        pdfium_module=_FakeScanPdfium(text=""),
    )
    assert scan_pdf.media_type == "application/pdf"
    with pytest.raises(ValueError, match="text layer"):
        verify_rendered_raster_output(
            RenderedRasterBytesV1(
                asset_id="asset-1",
                profile_id="scan-pdf-216-q75",
                content=b"%PDF-1.7\nBT (bad) Tj ET\n%%EOF",
                width=200,
                height=400,
            ),
            pdfium_version="4.30.0",
            pdfium_module=_FakeScanPdfium(text="bad"),
        )


def _perfect_candidate(*, source: SourcePdfGeometryV1) -> RasterDocumentExtractionResultV2:
    profile = "png-216"
    spans = compile_raster_gold(source, profile_id=profile, pdfium_spans=_pdfium_span())
    fields = compile_raster_field_gold(source, profile_id=profile, pdfium_spans=_pdfium_span())
    cells = compile_raster_table_cell_gold(source, profile_id=profile, pdfium_spans=_pdfium_span())
    pages = [
        {
            "page_number": 1,
            "width": 100,
            "height": 200,
            "coordinate_unit": "pixels",
            "spans": [
                {
                    "span_id": item.span_id,
                    "ordinal": item.reading_order,
                    "text": item.text,
                    "bbox": item.bbox.model_dump(),
                    "polygon": None,
                    "confidence": 0.99,
                    "reading_order": item.reading_order,
                }
                for item in spans
            ],
        }
    ]
    page_models = [RasterDocumentPreviewPageV2.model_validate(item) for item in pages]
    field_models = [
        RasterFieldCandidateV2.model_validate(
            {
                "field_id": item.field_id,
                "value": item.value,
                "confidence": 0.99,
                "evidence_span_ids": item.evidence_span_ids,
                "bbox": item.bbox.model_dump(),
            }
        )
        for item in fields
    ]
    table_models = [
        RasterTableV2.model_validate(
            {
                "table_id": "items",
                "page_number": 1,
                "bbox": cells[0].bbox.model_dump(),
                "rows": [
                    {
                        "row_index": 0,
                        "cells": [
                            {
                                "cell_id": "cell-0",
                                "row_index": 0,
                                "column_index": 0,
                                "text": cells[0].text,
                                "bbox": cells[0].bbox.model_dump(),
                                "confidence": 0.99,
                                "evidence_span_ids": cells[0].evidence_span_ids,
                            }
                        ],
                    }
                ],
            }
        )
    ]
    return RasterDocumentExtractionResultV2(
        document_asset_id=source.asset_id,
        extraction_request_id="request-1",
        raster_profile_revision="synthetic-raster-v1",
        status="preview_ready",
        media_type="image/png",
        byte_size=10,
        source_sha256=source.source_pdf_sha256,
        canonical_text_sha256=canonical_raster_preview_text_sha256(page_models),
        extractor_revision="ppstructurev3-local-v1",
        document_type=source.document_type,
        classification_confidence=0.99,
        pages=page_models,
        field_candidates=field_models,
        tables=table_models,
        warnings=[],
        needs_review_reasons=[],
    )


def test_local_metric_evaluator_calculates_gate_values_from_candidate_and_gold() -> None:
    source = _source(1)
    candidate = _perfect_candidate(source=source)
    case = RasterEvaluationCaseV1(
        asset_id=source.asset_id,
        document_type=source.document_type,
        profile_id="png-216",
        source_pdf_sha256=source.source_pdf_sha256,
        image_sha256=source.source_pdf_sha256,
        gold_spans=compile_raster_gold(source, profile_id="png-216", pdfium_spans=_pdfium_span()),
        gold_fields=compile_raster_field_gold(
            source, profile_id="png-216", pdfium_spans=_pdfium_span()
        ),
        gold_table_cells=compile_raster_table_cell_gold(
            source, profile_id="png-216", pdfium_spans=_pdfium_span()
        ),
        candidate=candidate,
    )
    counts = evaluate_synthetic_raster_case(case)
    assert (
        counts.field_matches,
        counts.character_errors,
        counts.table_matches,
        counts.span_matches,
    ) == (
        1,
        0,
        1,
        1,
    )
    cases = [
        case.model_copy(
            update={
                "document_type": document_type,
                "profile_id": profile_id,
                "candidate": _perfect_candidate(
                    source=source.model_copy(update={"document_type": document_type})
                ),
                "gold_spans": compile_raster_gold(
                    source.model_copy(update={"document_type": document_type}),
                    profile_id=profile_id,
                    pdfium_spans=_pdfium_span(),
                ),
                "gold_fields": compile_raster_field_gold(
                    source.model_copy(update={"document_type": document_type}),
                    profile_id=profile_id,
                    pdfium_spans=_pdfium_span(),
                ),
                "gold_table_cells": compile_raster_table_cell_gold(
                    source.model_copy(update={"document_type": document_type}),
                    profile_id=profile_id,
                    pdfium_spans=_pdfium_span(),
                ),
            }
        )
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
    ]
    scores = evaluate_synthetic_raster_gates(cases)
    assert synthetic_raster_gate_passes(scores)
    assert derive_synthetic_raster_confidence_thresholds(cases) == {
        "classification_confidence": 0.99,
        "span_confidence": 0.99,
        "field_confidence": 0.99,
        "table_cell_confidence": 0.99,
    }


def test_local_metric_evaluator_uses_visual_geometry_for_cer_reading_order() -> None:
    """CER must not confuse final-PDF drawing order with visible text order."""

    source = SourcePdfGeometryV1(
        asset_id="asset-visual-order",
        family_id="family-visual-order",
        document_type="commercial_invoice",
        source_pdf_sha256=_HASH,
        page_width_points=100,
        page_height_points=100,
        # Deliberately authored in a non-visual content-stream order.
        spans=[
            SourceTextGeometryV1(
                span_id="span-lower-right",
                page_number=1,
                text="LOWER RIGHT",
                pdf_bbox=PdfBoundingBoxV1(x=60, y=20, width=20, height=10),
                reading_order=1,
            ),
            SourceTextGeometryV1(
                span_id="span-top-left",
                page_number=1,
                text="TOP LEFT",
                pdf_bbox=PdfBoundingBoxV1(x=10, y=70, width=20, height=10),
                reading_order=2,
            ),
            SourceTextGeometryV1(
                span_id="span-lower-left",
                page_number=1,
                text="LOWER LEFT",
                pdf_bbox=PdfBoundingBoxV1(x=10, y=20, width=20, height=10),
                reading_order=3,
            ),
        ],
    )
    pdfium_spans = [
        PdfiumGeometrySpanV1(
            page_number=span.page_number,
            text=span.text,
            pdf_bbox=span.pdf_bbox,
            reading_order=span.reading_order,
            source_pdf_sha256=_HASH,
        )
        for span in source.spans
    ]
    gold = compile_raster_gold(source, profile_id="png-216", pdfium_spans=pdfium_spans)
    visual_order = [gold[1], gold[2], gold[0]]
    pages = [
        RasterDocumentPreviewPageV2.model_validate(
            {
                "page_number": 1,
                "width": 100,
                "height": 100,
                "coordinate_unit": "pixels",
                "spans": [
                    {
                        "span_id": f"candidate-{index}",
                        "ordinal": index,
                        "text": span.text,
                        "bbox": span.bbox.model_dump(),
                        "polygon": None,
                        "confidence": 0.99,
                        "reading_order": index,
                    }
                    for index, span in enumerate(visual_order, start=1)
                ],
            }
        )
    ]
    candidate = RasterDocumentExtractionResultV2(
        document_asset_id=source.asset_id,
        extraction_request_id="request-visual-order",
        raster_profile_revision="synthetic-raster-v1",
        status="preview_ready",
        media_type="image/png",
        byte_size=10,
        source_sha256=_HASH,
        canonical_text_sha256=canonical_raster_preview_text_sha256(pages),
        extractor_revision="ppstructurev3-local-v1",
        document_type=source.document_type,
        classification_confidence=0.99,
        pages=pages,
        field_candidates=[],
        tables=[],
        warnings=[],
        needs_review_reasons=[],
    )
    case = RasterEvaluationCaseV1(
        asset_id=source.asset_id,
        document_type=source.document_type,
        profile_id="png-216",
        source_pdf_sha256=_HASH,
        image_sha256=_HASH,
        gold_spans=gold,
        gold_fields=[],
        gold_table_cells=[],
        candidate=candidate,
    )

    raw_content_stream_text = "\n".join(span.text for span in gold)
    raw_provider_text = "\n".join(span.text for page in candidate.pages for span in page.spans)
    assert synthetic_raster._bounded_levenshtein(raw_content_stream_text, raw_provider_text) > 0
    assert evaluate_synthetic_raster_case(case).character_errors == 0


def test_local_metric_evaluator_accepts_only_the_fingerprint_bound_v2_profile_revision() -> None:
    """A v2 candidate reaches evaluation; runtime provenance binds its exact fingerprint.

    ``RasterEvaluationCaseV1`` is intentionally only the per-case boundary.
    The enclosing candidate bundle/execution/runtime comparison in the
    evaluator proves equality to the selected profile, while this boundary
    prevents a malformed arbitrary revision from entering metric calculation.
    """

    source = _source(1)
    fingerprint_revision = "synthetic-raster-v1:" + "a" * 24
    candidate = _perfect_candidate(source=source).model_copy(
        update={"raster_profile_revision": fingerprint_revision}
    )
    case = RasterEvaluationCaseV1(
        asset_id=source.asset_id,
        document_type=source.document_type,
        profile_id="png-216",
        source_pdf_sha256=source.source_pdf_sha256,
        image_sha256=source.source_pdf_sha256,
        gold_spans=compile_raster_gold(source, profile_id="png-216", pdfium_spans=_pdfium_span()),
        gold_fields=compile_raster_field_gold(
            source, profile_id="png-216", pdfium_spans=_pdfium_span()
        ),
        gold_table_cells=compile_raster_table_cell_gold(
            source, profile_id="png-216", pdfium_spans=_pdfium_span()
        ),
        candidate=candidate,
    )

    assert evaluate_synthetic_raster_case(case).field_matches == 1

    malformed = candidate.model_copy(
        update={"raster_profile_revision": "synthetic-raster-v1:not-a-fingerprint"}
    )
    with pytest.raises(ValueError, match="candidate binding"):
        RasterEvaluationCaseV1(
            asset_id=source.asset_id,
            document_type=source.document_type,
            profile_id="png-216",
            source_pdf_sha256=source.source_pdf_sha256,
            image_sha256=source.source_pdf_sha256,
            gold_spans=compile_raster_gold(
                source, profile_id="png-216", pdfium_spans=_pdfium_span()
            ),
            gold_fields=compile_raster_field_gold(
                source, profile_id="png-216", pdfium_spans=_pdfium_span()
            ),
            gold_table_cells=compile_raster_table_cell_gold(
                source, profile_id="png-216", pdfium_spans=_pdfium_span()
            ),
            candidate=malformed,
        )


def test_failed_raster_gate_preserves_a_cer_above_one_instead_of_crashing() -> None:
    """A long OCR insertion is a failed Gate result, not an invalid metric."""

    score = synthetic_raster.raster_gate_score(
        document_type="bill_of_lading",
        profile_id="png-216",
        counts=synthetic_raster.RasterMetricCountsV1(
            character_errors=3,
            character_reference=1,
        ),
    )

    assert score.character_error_rate == 3.0
    assert not synthetic_raster_gate_passes([score])


def test_evaluator_rejects_case_gold_that_does_not_match_receipt_bound_local_gold(
    tmp_path: Path,
) -> None:
    source = _source(1)
    candidate = _perfect_candidate(source=source)
    gold = RasterGoldArtifactV1(
        asset_id=source.asset_id,
        family_id=source.family_id,
        document_type=source.document_type,
        source_pdf_sha256=source.source_pdf_sha256,
        profile_id="png-216",
        spans=compile_raster_gold(source, profile_id="png-216", pdfium_spans=_pdfium_span()),
        fields=compile_raster_field_gold(source, profile_id="png-216", pdfium_spans=_pdfium_span()),
        table_cells=compile_raster_table_cell_gold(
            source, profile_id="png-216", pdfium_spans=_pdfium_span()
        ),
    )
    root = tmp_path / "_local" / "data" / "document-preview"
    gold_path = root / "synthetic-raster-v1" / "gold" / "asset-1-png-216.json"
    gold_path.parent.mkdir(parents=True)
    gold_raw = json.dumps(
        gold.model_dump(mode="json", warnings=False), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    gold_path.write_bytes(gold_raw)
    raster_path = root / "synthetic-raster-v1" / "assets" / "asset-1-png-216.png"
    raster_path.parent.mkdir(parents=True)
    raster_path.write_bytes(_ONE_BY_ONE_PNG)
    image_sha256 = hashlib.sha256(_ONE_BY_ONE_PNG).hexdigest()
    candidate = candidate.model_copy(update={"source_sha256": image_sha256})
    receipt = SyntheticRasterReceiptV1(
        source_dataset_id="synthetic-customs-pdf-2000",
        source_dataset_revision="20260811-r1",
        source_receipt_sha256="b" * 64,
        renderer_revision="pdfium-raster-v1",
        renderer_configuration_sha256="c" * 64,
        pdfium_version="4.30.0",
        font_manifest_sha256="d" * 64,
        split_seed_sha256="e" * 64,
        render_recipe_sha256="f" * 64,
        render_seed_sha256="1" * 64,
        profiles=list(RASTER_PROFILES.values()),
        assets=[
            RasterAssetReceiptV1(
                asset_id=source.asset_id,
                family_id=source.family_id,
                document_type=source.document_type,
                split="calibration",
                profile_id="png-216",
                source_pdf_sha256=source.source_pdf_sha256,
                image_sha256=image_sha256,
                asset_relative_path="synthetic-raster-v1/assets/asset-1-png-216.png",
                media_type="image/png",
                byte_size=len(_ONE_BY_ONE_PNG),
                width=1,
                height=1,
                coordinate_transform="pdf-points-lower-left-to-normalized-top-left-v1",
                render_recipe_sha256="f" * 64,
                render_seed_sha256="1" * 64,
                gold_relative_path="synthetic-raster-v1/gold/asset-1-png-216.json",
                gold_sha256=hashlib.sha256(gold_raw).hexdigest(),
            )
        ],
    )
    case = RasterEvaluationCaseV1(
        asset_id=source.asset_id,
        document_type=source.document_type,
        profile_id="png-216",
        source_pdf_sha256=source.source_pdf_sha256,
        image_sha256=image_sha256,
        gold_spans=gold.spans,
        gold_fields=gold.fields,
        gold_table_cells=gold.table_cells,
        candidate=candidate,
    )
    verifier = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "evaluate_synthetic_raster.py")
    )["_verify_cases_against_receipt"]
    verifier(root=root, cases=[case], receipt=receipt, split="calibration")
    forged = case.model_copy(
        update={"gold_spans": [case.gold_spans[0].model_copy(update={"text": "FORGED"})]}
    )
    with pytest.raises(ValueError, match="Gold does not bind"):
        verifier(root=root, cases=[forged], receipt=receipt, split="calibration")
    raster_path.write_bytes(b"not-a-png")
    with pytest.raises(ValueError, match="evaluation asset does not bind"):
        verifier(root=root, cases=[case], receipt=receipt, split="calibration")


def test_source_enumerator_uses_only_the_fixed_clean_error_pdf_tree(tmp_path: Path) -> None:
    for index in (1, 2):
        for directory, prefix in (("clean_sets", "C"), ("error_sets", "E")):
            target = tmp_path / directory / f"set_{index:04d}"
            target.mkdir(parents=True)
            for suffix in ("BL", "INV", "PL"):
                (target / f"{prefix}{index:04d}_{suffix}.pdf").write_bytes(b"%PDF-1.3\n%%EOF")
    assets = enumerate_synthetic_raster_source_assets(tmp_path, expected_family_count=2)
    assert len(assets) == 12
    assert {asset.family_id for asset in assets} == {"family-0001", "family-0002"}
    assert {
        (asset.fixture_kind, asset.document_type)
        for asset in assets
        if asset.family_id == "family-0001"
    } == {
        ("clean", "bill_of_lading"),
        ("clean", "commercial_invoice"),
        ("clean", "packing_list"),
        ("error", "bill_of_lading"),
        ("error", "commercial_invoice"),
        ("error", "packing_list"),
    }
    (tmp_path / "clean_sets" / "set_0001" / "unexpected.pdf").write_bytes(b"%PDF-1.3\n%%EOF")
    with pytest.raises(ValueError, match="fixed PDF set"):
        enumerate_synthetic_raster_source_assets(tmp_path, expected_family_count=2)


def test_raster_candidate_and_evaluation_tools_do_not_accept_free_result_or_hash_flags() -> None:
    evaluator = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "evaluate_synthetic_raster.py")
    )["_parser"]()
    evaluator_options = {
        option for action in evaluator._actions for option in action.option_strings
    }
    assert {
        "--runtime-configuration-relative-path",
        "--synthetic-raster-receipt-relative-path",
        "--candidate-bundle-relative-path",
        "--candidate-execution-relative-path",
    } <= evaluator_options
    assert (
        not {
            "--cases",
            "--raster-profile-revision",
            "--runtime-configuration-sha256",
            "--package-inventory-sha256",
            "--model-inventory-sha256",
        }
        & evaluator_options
    )

    candidate_runner = runpy.run_path(
        str(
            Path(__file__).parents[1] / "tools" / "run_ppstructurev3_synthetic_raster_candidates.py"
        )
    )["_parser"]()
    runner_options = {
        option for action in candidate_runner._actions for option in action.option_strings
    }
    assert {
        "--runtime-configuration-relative-path",
        "--synthetic-raster-receipt-relative-path",
        "--split",
    } <= runner_options
    assert "--pdfium-version" not in runner_options


def test_evaluator_artifact_writer_reuses_only_identical_canonical_bytes(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    evaluator = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "evaluate_synthetic_raster.py")
    )
    writer = evaluator["_write_create_only"]
    relative_path = "synthetic-raster-v1/evaluation/calibration.json"
    first_path, first_hash = writer(root, relative_path=relative_path, value={"kind": "same"})
    second_path, second_hash = writer(root, relative_path=relative_path, value={"kind": "same"})
    assert (second_path, second_hash) == (first_path, first_hash)
    first_path.write_bytes(b"different")
    with pytest.raises(ValueError, match="immutable conflict"):
        writer(root, relative_path=relative_path, value={"kind": "same"})


def test_evaluator_uses_split_inventory_receipt_fields_for_fingerprint_profile() -> None:
    evaluator = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "evaluate_synthetic_raster.py")
    )
    profile = PpStructureV3OfflineProfileV2.model_construct(
        raster_profile_revision="synthetic-raster-v1:" + "a" * 24,
        parent_renderer_package_inventory_sha256="b" * 64,
        sidecar_package_inventory_sha256="c" * 64,
        model_inventory_sha256="d" * 64,
    )
    resolution = RasterCandidateEngineResolutionV1(
        engine=None,
        status="available",
        runtime_configuration_sha256="e" * 64,
        profile=profile,
    )
    assert evaluator["_runtime_artifact_schema_version"](resolution) == "2.0"
    assert evaluator["_runtime_inventory_bindings"](resolution) == {
        "parent_renderer_package_inventory_sha256": "b" * 64,
        "sidecar_package_inventory_sha256": "c" * 64,
        "model_inventory_sha256": "d" * 64,
    }
