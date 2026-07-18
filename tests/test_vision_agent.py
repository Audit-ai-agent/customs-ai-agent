from __future__ import annotations

import base64
import copy
import hashlib
import json
from io import BytesIO

import pytest
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image, ImageOps, PngImagePlugin
from pydantic import SecretStr, ValidationError

from customs_ai_agent.adapters.images import (
    ImageInputError,
    PdfRasterizationResult,
    RasterizedPdfPage,
    image_page_from_file,
    image_pages_from_pdf_file,
    pdf_pages_from_pdf_file,
)
from customs_ai_agent.adapters.ocr import OcrLayoutProviderError
from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    ClaudeTransportResult,
    CropReadBudget,
    LLMOutputValidationError,
    LLMProviderError,
    analysis_to_extracted_document,
)
from customs_ai_agent.adapters.retrieval import MockRegulationRetriever
from customs_ai_agent.graph import build_review_graph
from customs_ai_agent.graph.workflow import _compiled_graph_delegate
from customs_ai_agent.nodes.classify import classify_documents
from customs_ai_agent.nodes.extract import extract_documents
from customs_ai_agent.nodes.self_correct import self_correct_extraction
from customs_ai_agent.payloads import SensitiveDocumentPayloadStore
from customs_ai_agent.runner import (
    _build_initial_state,
    run_review,
    run_review_v2,
    to_review_run_result_v2,
)
from customs_ai_agent.schemas.models import (
    BBox,
    ImagePageProvenance,
    ImagePageInput,
    MAX_IMAGE_BYTES,
    OcrLayoutLedger,
    OcrLayoutPage,
    OcrLayoutSpan,
    ReviewRunRequest,
    ReviewRunRequestV2,
    ReviewRunResultV2,
    stable_ocr_ledger_id,
)


def _image_bytes(
    *,
    size: tuple[int, int] = (400, 600),
    image_format: str = "JPEG",
    exif_orientation: int | None = None,
) -> bytes:
    image = Image.new("RGB", size, color=(245, 245, 245))
    output = BytesIO()
    save_kwargs: dict[str, object] = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        save_kwargs["exif"] = exif
    image.save(output, format=image_format, **save_kwargs)
    return output.getvalue()


def _page_from_bytes(
    raw: bytes | None = None,
    *,
    page_number: int = 1,
    size: tuple[int, int] = (400, 600),
    media_type: str = "image/jpeg",
) -> ImagePageInput:
    raw = raw if raw is not None else _image_bytes(size=size)
    return ImagePageInput(
        page_number=page_number,
        media_type=media_type,
        data_base64=base64.b64encode(raw).decode("ascii"),
        byte_length=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        width_px=size[0],
        height_px=size[1],
        provenance=ImagePageProvenance(
            source_media_type=media_type,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_page_number=1,
            transforms=["metadata_strip"],
            original_exif_orientation=1,
        ),
    )


def _vision_evidence(
    source_text: str,
    *,
    page: int = 1,
    box: tuple[int, int, int, int] = (10, 10, 200, 40),
) -> dict[str, object]:
    span_id = "span_" + hashlib.sha256(f"{page}:{box}:{source_text}".encode()).hexdigest()[:16]
    return {
        "source_text": source_text,
        "model_observed_bbox": {
            "page": page,
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
        },
        "ocr_span_ids": [span_id],
    }


def _line_item(
    *,
    y1: int = 200,
    description: str = "Stainless bolts",
    hs_code: str = "731815",
) -> dict[str, object]:
    return {
        "row_evidence": _vision_evidence(
            f"1 {description} {hs_code} 10 KG 12000",
            box=(10, y1, 390, y1 + 50),
        ),
        "fields": [
            {
                "name": "item_no",
                "value": "1",
                "evidence": _vision_evidence(
                    "1",
                    box=(15, y1 + 5, 30, y1 + 35),
                ),
            },
            {
                "name": "description",
                "value": description,
                "evidence": _vision_evidence(
                    description,
                    box=(40, y1 + 5, 180, y1 + 35),
                ),
            },
            {
                "name": "hs_code",
                "value": hs_code,
                "evidence": _vision_evidence(
                    hs_code,
                    box=(190, y1 + 5, 250, y1 + 35),
                ),
            },
            {
                "name": "quantity",
                "value": "10 KG",
                "evidence": _vision_evidence(
                    "10 KG",
                    box=(260, y1 + 5, 305, y1 + 35),
                ),
            },
            {
                "name": "amount",
                "value": "12000",
                "evidence": _vision_evidence(
                    "12000",
                    box=(315, y1 + 5, 375, y1 + 35),
                ),
            },
        ],
    }


def _vision_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "document_type": "invoice",
        "classification_evidence": _vision_evidence(
            "COMMERCIAL INVOICE",
            box=(80, 20, 320, 60),
        ),
        "fields": [
            {
                "name": "invoice_no",
                "value": "INV-TEST-001",
                "evidence": _vision_evidence(
                    "INVOICE NO INV-TEST-001",
                    box=(20, 80, 250, 110),
                ),
            },
            {
                "name": "total_amount",
                "value": "12000",
                "evidence": _vision_evidence(
                    "TOTAL AMOUNT 12000",
                    box=(210, 500, 380, 535),
                ),
            },
        ],
        "line_items": [_line_item()],
    }


class FakeVisionTransport:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        crop_payload: dict[str, object] | None = None,
    ) -> None:
        self.payload = payload
        self.crop_payload = crop_payload
        self.calls: list[dict[str, object]] = []

    def create_document_analysis(self, **kwargs: object) -> ClaudeTransportResult:
        self.calls.append(kwargs)
        schema = kwargs["output_schema"]
        assert isinstance(schema, dict)
        payload = (
            self.crop_payload
            if schema.get("title") == "ClaudeCropReadCandidate" and self.crop_payload is not None
            else self.payload
        )
        call_number = len(self.calls)
        return ClaudeTransportResult(
            text=json.dumps(payload),
            request_id=f"req_vision{call_number:08d}",
            message_id=f"msg_vision{call_number:08d}",
            model="claude-sonnet-5",
            stop_reason="end_turn",
            input_tokens=300,
            output_tokens=100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            latency_ms=12,
        )


class FailingCropTransport(FakeVisionTransport):
    def create_document_analysis(self, **kwargs: object) -> ClaudeTransportResult:
        schema = kwargs["output_schema"]
        assert isinstance(schema, dict)
        if schema.get("title") == "ClaudeCropReadCandidate":
            self.calls.append(kwargs)
            raise LLMProviderError(
                "provider_unavailable",
                "Synthetic crop provider failure.",
                retryable=True,
            )
        return super().create_document_analysis(**kwargs)


def _all_vision_evidence(payload: dict[str, object]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    classification = payload.get("classification_evidence")
    if isinstance(classification, dict):
        evidence.append(classification)
    for field in payload.get("fields", []):
        if isinstance(field, dict) and isinstance(field.get("evidence"), dict):
            evidence.append(field["evidence"])
    for item in payload.get("line_items", []):
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("row_evidence"), dict):
            evidence.append(item["row_evidence"])
        for field in item.get("fields", []):
            if isinstance(field, dict) and isinstance(field.get("evidence"), dict):
                evidence.append(field["evidence"])
    return evidence


class FakeOcrLayoutProvider:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        bbox_overrides: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.payload = payload
        self.bbox_overrides = bbox_overrides or {}

    def analyze(
        self,
        *,
        document_id: str,
        pages: list[ImagePageInput],
    ) -> OcrLayoutLedger:
        source_page = pages[0]
        observations = _all_vision_evidence(self.payload)
        spans: list[OcrLayoutSpan] = []
        seen: set[str] = set()
        sortable: list[tuple[int, int, int, dict[str, object], str]] = []
        for observation in observations:
            span_id = str(observation["ocr_span_ids"][0])
            if span_id in seen:
                continue
            seen.add(span_id)
            box = observation["model_observed_bbox"]
            assert isinstance(box, dict)
            coordinates = self.bbox_overrides.get(
                span_id,
                (
                    int(box["x1"]),
                    int(box["y1"]),
                    min(int(box["x2"]), source_page.width_px),
                    min(int(box["y2"]), source_page.height_px),
                ),
            )
            sortable.append(
                (
                    coordinates[1],
                    coordinates[0],
                    len(sortable),
                    observation,
                    span_id,
                )
            )
        for reading_order, (_, _, _, observation, span_id) in enumerate(sorted(sortable)):
            box = observation["model_observed_bbox"]
            assert isinstance(box, dict)
            x1, y1, x2, y2 = self.bbox_overrides.get(
                span_id,
                (
                    int(box["x1"]),
                    int(box["y1"]),
                    min(int(box["x2"]), source_page.width_px),
                    min(int(box["y2"]), source_page.height_px),
                ),
            )
            spans.append(
                OcrLayoutSpan(
                    span_id=span_id,
                    page=source_page.page_number,
                    reading_order=reading_order,
                    text=str(observation["source_text"]),
                    bbox=BBox(
                        page=source_page.page_number,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                    confidence=0.99,
                )
            )
        ledger_page = OcrLayoutPage(
            page_number=source_page.page_number,
            image_sha256=source_page.sha256,
            width_px=source_page.width_px,
            height_px=source_page.height_px,
            spans=spans,
        )
        values = {
            "document_id": document_id,
            "provider_id": "fake-independent-ocr",
            "engine_id": "fake-layout",
            "engine_version": "1.0",
            "pages": [ledger_page],
        }
        return OcrLayoutLedger(
            ledger_id=stable_ocr_ledger_id(**values),
            **values,
        )


def _analyze_image(
    payload: dict[str, object] | None = None,
    *,
    pages: list[ImagePageInput] | None = None,
    ocr_provider: FakeOcrLayoutProvider | None = None,
    transport: FakeVisionTransport | None = None,
    crop_budget: CropReadBudget | None = None,
):
    active_payload = payload or _vision_payload()
    active_transport = transport or FakeVisionTransport(active_payload)
    analyzer = ClaudeDocumentAnalyzer(
        active_transport,
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=ocr_provider or FakeOcrLayoutProvider(active_payload),
    )
    page_items = pages or [_page_from_bytes()]
    document = {
        "document_id": "doc_image",
        "filename": "invoice.jpg",
        "content_text": None,
        "image_pages": [page.model_dump(mode="json") for page in page_items],
        "declared_type": None,
    }
    if crop_budget is not None:
        document["_runtime_crop_budget"] = crop_budget
    return analyzer.analyze(document), active_transport, document


def test_valid_image_page_and_file_adapter_preserve_provenance(tmp_path):
    raw = _image_bytes()
    path = tmp_path / "invoice.jpg"
    path.write_bytes(raw)

    page = image_page_from_file(path)

    assert page.decoded_bytes() != raw
    assert page.media_type == "image/jpeg"
    assert (page.width_px, page.height_px) == (400, 600)
    assert page.byte_length == len(page.decoded_bytes())
    assert page.sha256 == hashlib.sha256(page.decoded_bytes()).hexdigest()
    assert page.provenance.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert page.provenance.source_sha256 != page.sha256
    assert page.provenance.source_media_type == "image/jpeg"
    assert page.provenance.transforms == ["metadata_strip"]
    assert page.provenance.original_exif_orientation == 1
    assert "data_base64" not in repr(page)


def test_file_adapter_normalizes_exif_and_records_transform(tmp_path):
    raw = _image_bytes(size=(400, 600), exif_orientation=6)
    path = tmp_path / "rotated.jpg"
    path.write_bytes(raw)

    page = image_page_from_file(path)

    assert (page.width_px, page.height_px) == (600, 400)
    assert page.sha256 != hashlib.sha256(raw).hexdigest()
    assert page.provenance.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert page.provenance.original_exif_orientation == 6
    assert page.provenance.transforms == ["exif_transpose", "metadata_strip"]
    with Image.open(BytesIO(page.decoded_bytes())) as normalized:
        assert int(normalized.getexif().get(274, 1) or 1) == 1


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_file_adapter_reencodes_pixels_and_strips_nonvisual_metadata(
    tmp_path,
    image_format,
):
    secret_marker = "private-origin-metadata-48291"
    image = Image.new("RGB", (12, 8), color=(20, 40, 60))
    exif = Image.Exif()
    exif[315] = secret_marker
    output = BytesIO()
    suffix = image_format.lower().replace("jpeg", "jpg")
    save_options: dict[str, object] = {}
    if image_format == "JPEG":
        save_options = {"exif": exif, "comment": secret_marker.encode()}
    elif image_format == "PNG":
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("Comment", secret_marker)
        save_options = {"pnginfo": png_info, "exif": exif}
    else:
        save_options = {
            "exif": exif.tobytes(),
            "xmp": f"<private>{secret_marker}</private>".encode(),
        }
    image.save(output, format=image_format, **save_options)
    source = output.getvalue()
    path = tmp_path / f"source.{suffix}"
    path.write_bytes(source)

    page = image_page_from_file(path)
    sanitized = page.decoded_bytes()

    assert page.provenance.source_sha256 == hashlib.sha256(source).hexdigest()
    assert page.sha256 == hashlib.sha256(sanitized).hexdigest()
    assert page.sha256 != page.provenance.source_sha256
    assert page.provenance.transforms == ["metadata_strip"]
    assert secret_marker.encode() not in sanitized
    with Image.open(BytesIO(sanitized)) as decoded:
        assert decoded.size == image.size
        assert list(decoded.getexif().items()) == []
        assert not {"comment", "exif", "icc_profile", "xmp"}.intersection(
            key.lower() for key in decoded.info
        )


def test_file_adapter_preserves_oriented_pixels_while_stripping_metadata(tmp_path):
    image = Image.new("RGB", (3, 2))
    image.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
    )
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "private-orientation-author"
    source_buffer = BytesIO()
    image.save(source_buffer, format="PNG", exif=exif)
    source = source_buffer.getvalue()
    path = tmp_path / "oriented.png"
    path.write_bytes(source)

    with Image.open(BytesIO(source)) as original:
        expected = ImageOps.exif_transpose(original).convert("RGB")
        expected_pixels = list(expected.get_flattened_data())

    page = image_page_from_file(path)

    assert page.provenance.source_sha256 == hashlib.sha256(source).hexdigest()
    assert page.sha256 == hashlib.sha256(page.decoded_bytes()).hexdigest()
    assert page.provenance.transforms == ["exif_transpose", "metadata_strip"]
    assert page.provenance.original_exif_orientation == 6
    with Image.open(BytesIO(page.decoded_bytes())) as sanitized:
        assert sanitized.size == (2, 3)
        assert list(sanitized.convert("RGB").get_flattened_data()) == expected_pixels
        assert list(sanitized.getexif().items()) == []


@pytest.mark.parametrize(
    ("orientation", "transforms"),
    [
        (1, []),
        (1, ["exif_transpose", "metadata_strip"]),
        (6, ["metadata_strip"]),
        (6, ["exif_transpose"]),
    ],
)
def test_direct_image_provenance_requires_metadata_sanitization(
    orientation,
    transforms,
):
    with pytest.raises(ValidationError):
        ImagePageProvenance(
            source_media_type="image/png",
            source_sha256="a" * 64,
            source_page_number=1,
            transforms=transforms,
            original_exif_orientation=orientation,
        )


@pytest.mark.parametrize(
    ("image_format", "media_type", "metadata_options"),
    [
        (
            "JPEG",
            "image/jpeg",
            lambda marker: {
                "exif": _exif_with_artist(marker),
            },
        ),
        (
            "JPEG",
            "image/jpeg",
            lambda marker: {
                "icc_profile": marker.encode(),
            },
        ),
        (
            "JPEG",
            "image/jpeg",
            lambda marker: {
                "comment": marker.encode(),
            },
        ),
        (
            "PNG",
            "image/png",
            lambda marker: {
                "pnginfo": _png_text_metadata(marker),
            },
        ),
        (
            "WEBP",
            "image/webp",
            lambda marker: {
                "xmp": f"<private>{marker}</private>".encode(),
            },
        ),
    ],
)
def test_forged_metadata_strip_provenance_cannot_bypass_byte_validation(
    image_format,
    media_type,
    metadata_options,
):
    marker = "forged-private-metadata-92741"
    output = BytesIO()
    Image.new("RGB", (12, 8), color=(20, 40, 60)).save(
        output,
        format=image_format,
        **metadata_options(marker),
    )
    raw = output.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    assert marker.encode() in raw

    with pytest.raises(ValidationError, match="non-visual metadata"):
        ImagePageInput(
            page_number=1,
            media_type=media_type,
            data_base64=base64.b64encode(raw).decode(),
            byte_length=len(raw),
            sha256=digest,
            width_px=12,
            height_px=8,
            provenance=ImagePageProvenance(
                source_media_type=media_type,
                source_sha256=digest,
                source_page_number=1,
                transforms=["metadata_strip"],
                original_exif_orientation=1,
            ),
        )


def _exif_with_artist(marker: str) -> Image.Exif:
    exif = Image.Exif()
    exif[315] = marker
    return exif


def _png_text_metadata(marker: str) -> PngImagePlugin.PngInfo:
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", marker)
    return metadata


class FakePdfRasterizer:
    def __init__(self, pages: list[bytes]) -> None:
        self.pages = pages

    def rasterize(
        self,
        pdf_bytes: bytes,
        *,
        dpi: int,
        max_pages: int,
    ) -> PdfRasterizationResult:
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(self.pages) <= max_pages
        return PdfRasterizationResult(
            rasterizer_id="fake-pdf-renderer",
            rasterizer_version="1.2.3",
            dpi=dpi,
            pages=[
                RasterizedPdfPage(
                    page_number=index,
                    media_type="image/png",
                    data=data,
                )
                for index, data in enumerate(self.pages, start=1)
            ],
        )


def test_pdf_adapter_records_source_and_rasterization_provenance(tmp_path):
    pdf_bytes = b"%PDF-1.7\nsynthetic-test-only\n%%EOF"
    path = tmp_path / "shipment.pdf"
    path.write_bytes(pdf_bytes)
    rendered = [
        _image_bytes(size=(300, 500), image_format="PNG"),
        _image_bytes(size=(300, 500), image_format="PNG"),
    ]

    pages = image_pages_from_pdf_file(
        path,
        FakePdfRasterizer(rendered),
        dpi=200,
    )

    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.media_type == "image/png" for page in pages)
    assert all(
        page.provenance.source_sha256 == hashlib.sha256(pdf_bytes).hexdigest() for page in pages
    )
    assert [page.provenance.source_page_number for page in pages] == [1, 2]
    assert all(page.provenance.transforms == ["pdf_rasterize"] for page in pages)
    assert all(
        page.provenance.rasterizer_id == "fake-pdf-renderer"
        and page.provenance.rasterizer_version == "1.2.3"
        and page.provenance.raster_dpi == 200
        for page in pages
    )


def test_public_v2_pdf_modality_is_explicit_and_normalized_before_graph(tmp_path):
    pdf_bytes = b"%PDF-1.7\nsynthetic-test-only\n%%EOF"
    path = tmp_path / "shipment.pdf"
    path.write_bytes(pdf_bytes)
    envelope = pdf_pages_from_pdf_file(
        path,
        FakePdfRasterizer([_image_bytes(size=(300, 500), image_format="PNG")]),
        dpi=200,
        max_pages=5,
    )
    request = ReviewRunRequestV2(
        case_id="case_pdf_modality",
        documents=[
            {
                "document_id": "doc_pdf",
                "filename": "shipment.pdf",
                "pdf_pages": envelope,
            }
        ],
        llm_mode="llm",
    )

    assert request.documents[0].image_pages == []
    assert request.documents[0].pdf_pages is not None
    assert request.documents[0].pdf_pages.source_pdf_sha256 == hashlib.sha256(pdf_bytes).hexdigest()
    initial_state = _build_initial_state(request)
    assert "pdf_pages" not in initial_state["documents"][0]
    assert (
        initial_state["documents"][0]["image_pages"][0]["provenance"]["source_media_type"]
        == "application/pdf"
    )
    assert "data_base64" not in json.dumps(initial_state)


def test_pdf_pages_cannot_be_smuggled_through_the_direct_image_modality(tmp_path):
    path = tmp_path / "shipment.pdf"
    path.write_bytes(b"%PDF-1.7\nsynthetic-test-only\n%%EOF")
    envelope = pdf_pages_from_pdf_file(
        path,
        FakePdfRasterizer([_image_bytes(size=(300, 500), image_format="PNG")]),
        dpi=200,
        max_pages=5,
    )

    with pytest.raises(ValidationError, match="explicit pdf_pages modality"):
        ReviewRunRequestV2(
            case_id="case_pdf_smuggling",
            documents=[
                {
                    "document_id": "doc_pdf",
                    "filename": "shipment.pdf",
                    "image_pages": envelope.pages,
                }
            ],
            llm_mode="llm",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"sha256": "0" * 64}, "sha256"),
        ({"media_type": "image/png"}, "media_type"),
        ({"width_px": 401}, "dimensions"),
        ({"data_base64": "!!!!"}, "base64"),
        ({"byte_length": MAX_IMAGE_BYTES + 1}, "less than or equal"),
    ],
)
def test_image_page_rejects_forged_hash_media_dimensions_or_encoding(mutation, message):
    page = _page_from_bytes()
    values = page.model_dump(mode="json")
    values.update(mutation)

    with pytest.raises(ValidationError, match=message):
        ImagePageInput.model_validate(values)


def test_image_page_rejects_corrupt_bytes_even_with_matching_hash_and_length():
    corrupt = b"not-an-image"

    with pytest.raises(ValidationError, match="safe decodable image"):
        _page_from_bytes(corrupt, size=(1, 1))


@pytest.mark.parametrize(
    ("image_format", "media_type", "appended"),
    [
        ("PNG", "image/png", b"PK\x03\x04FORGED-ZIP-PAYLOAD"),
        ("JPEG", "image/jpeg", b"<script>forged</script>"),
        ("WEBP", "image/webp", b"\x89PNG\r\n\x1a\n"),
    ],
)
def test_image_page_rejects_trailing_polyglot_payload(
    image_format: str,
    media_type: str,
    appended: bytes,
) -> None:
    raw = _image_bytes(image_format=image_format) + appended

    with pytest.raises(ValidationError, match="trailing or embedded payload"):
        _page_from_bytes(raw, media_type=media_type)


def test_image_page_rejects_visual_token_oversize_before_provider_call():
    raw = _image_bytes(size=(2576, 2576))

    with pytest.raises(ValidationError, match="upstream resize"):
        _page_from_bytes(raw, size=(2576, 2576))


def test_image_page_rejects_exif_rotation_before_coordinate_mapping():
    raw = _image_bytes(exif_orientation=6)

    with pytest.raises(ValidationError, match="orientation normalization"):
        _page_from_bytes(raw)


def test_file_adapter_redacts_corrupt_input_details(tmp_path):
    path = tmp_path / "secret-shipment-name.jpg"
    path.write_bytes(b"not-an-image")

    with pytest.raises(ImageInputError) as exc_info:
        image_page_from_file(path)

    assert "secret-shipment-name" not in str(exc_info.value)
    assert "supported decodable image" in str(exc_info.value)


@pytest.mark.parametrize(
    "values",
    [
        {
            "document_id": "doc_none",
            "filename": "none",
        },
        {
            "document_id": "doc_both",
            "filename": "both",
            "content_text": "TEXT",
            "image_pages": [_page_from_bytes()],
        },
    ],
)
def test_document_input_requires_exactly_one_modality(values):
    with pytest.raises(ValidationError, match="exactly one"):
        ReviewRunRequestV2(
            case_id="case_modalities",
            documents=[values],
            llm_mode="llm",
        )


def test_document_input_rejects_nonconsecutive_or_reordered_pages():
    page_one = _page_from_bytes(page_number=1)
    page_three = _page_from_bytes(page_number=3)

    with pytest.raises(ValidationError, match="consecutive and source ordered"):
        ReviewRunRequestV2(
            case_id="case_pages",
            documents=[
                {
                    "document_id": "doc_pages",
                    "filename": "pages.jpg",
                    "image_pages": [page_one, page_three],
                }
            ],
            llm_mode="llm",
        )


def test_image_documents_are_rejected_in_mock_mode():
    with pytest.raises(ValidationError, match="require llm mode"):
        ReviewRunRequestV2(
            case_id="case_mock_image",
            documents=[
                {
                    "document_id": "doc_image",
                    "filename": "invoice.jpg",
                    "image_pages": [_page_from_bytes()],
                }
            ],
            llm_mode="mock",
        )


def test_combined_image_page_limit_is_enforced_before_model_call():
    documents = [
        {
            "document_id": f"doc_{index}",
            "filename": f"page_{index}.jpg",
            "image_pages": [_page_from_bytes()],
        }
        for index in range(11)
    ]

    with pytest.raises(ValidationError, match="combined image pages"):
        ReviewRunRequestV2(
            case_id="case_too_many_pages",
            documents=documents,
            llm_mode="llm",
        )


def test_graph_initial_state_never_contains_raw_image_payload():
    page = _page_from_bytes()
    request = ReviewRunRequestV2(
        case_id="case_state_boundary",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "invoice.jpg",
                "image_pages": [page],
            }
        ],
        llm_mode="llm",
    )

    initial_state = _build_initial_state(request)

    serialized = json.dumps(initial_state, default=str)
    assert "data_base64" not in serialized
    state_page = initial_state["documents"][0]["image_pages"][0]
    assert state_page["sha256"] == page.sha256
    assert state_page["provenance"]["source_sha256"] == page.sha256
    assert initial_state["sensitive_inputs_redacted"] is False


def test_graph_checkpoints_never_contain_raw_image_payload():
    payload = _vision_payload()
    transport = FakeVisionTransport(payload)
    analyzer = ClaudeDocumentAnalyzer(
        transport,
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(payload),
    )
    page = _page_from_bytes()
    request = ReviewRunRequestV2(
        case_id="case_checkpoint_boundary",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "invoice.jpg",
                "image_pages": [page],
            }
        ],
        llm_mode="llm",
    )
    initial_state = _build_initial_state(request)
    # Simulate an unsafe direct graph caller. The graph facade must strip this
    # value before LangGraph or its checkpointer sees the input state.
    initial_state["documents"][0]["image_pages"][0]["data_base64"] = page.data_base64
    sensitive_documents = {"doc_image": request.documents[0].model_dump(mode="json")}
    payload_store = SensitiveDocumentPayloadStore(sensitive_documents)
    checkpointer = MemorySaver()
    graph = build_review_graph(
        MockRegulationRetriever(),
        analyzer,
        payload_store=payload_store,
        checkpointer=checkpointer,
    )
    config = {
        "configurable": {"thread_id": "checkpoint-boundary"},
        "recursion_limit": 20,
    }

    result = graph.invoke(initial_state, config=config)
    checkpoints = list(checkpointer.list(config))

    assert result["sensitive_inputs_redacted"] is True
    assert payload_store.is_empty
    assert request.documents[0].image_pages[0].data_base64 == page.data_base64
    assert "data_base64" not in json.dumps(sensitive_documents, default=str)
    assert "data_base64" in initial_state["documents"][0]["image_pages"][0]
    assert checkpoints
    assert all(
        "data_base64" not in json.dumps(item.checkpoint, default=str) for item in checkpoints
    )
    assert all(
        page.data_base64 not in json.dumps(item.checkpoint, default=str) for item in checkpoints
    )


def test_graph_facade_rejects_direct_checkpoint_state_mutation():
    checkpointer = MemorySaver()
    graph = build_review_graph(
        MockRegulationRetriever(),
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "direct-state-mutation"}}

    with pytest.raises(RuntimeError, match="direct_graph_state_mutation_forbidden"):
        graph.update_state(
            config,
            {"documents": [{"data_base64": "SECRET_RAW_MEDIA_SENTINEL"}]},
            as_node="load_case",
        )

    with pytest.raises(AttributeError):
        _ = graph._compiled_graph
    assert list(checkpointer.list(config)) == []


def test_checkpoint_sink_redacts_even_a_recovered_raw_graph_update():
    checkpointer = MemorySaver()
    graph = build_review_graph(
        MockRegulationRetriever(),
        checkpointer=checkpointer,
    )
    config = {
        "configurable": {"thread_id": "raw-graph-checkpoint-sink"},
        "recursion_limit": 20,
    }
    graph.invoke(
        _build_initial_state(ReviewRunRequest(scenario_id="happy_path")),
        config=config,
    )

    raw_graph = _compiled_graph_delegate(graph)
    raw_graph.update_state(
        config,
        {
            "documents": [
                {
                    "document_id": "doc-unsafe",
                    "image_pages": [
                        {
                            "data_base64": "SECRET_RAW_MEDIA_SENTINEL",
                        }
                    ],
                }
            ]
        },
        as_node="load_case",
    )
    serialized = json.dumps(
        [item.checkpoint for item in checkpointer.list(config)],
        default=str,
    )

    assert "data_base64" not in serialized
    assert "SECRET_RAW_MEDIA_SENTINEL" not in serialized


def test_one_shot_payload_is_empty_before_extract_and_extraction_uses_snapshot():
    payload = _vision_payload()
    analyzer = ClaudeDocumentAnalyzer(
        FakeVisionTransport(payload),
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(payload),
    )
    page = _page_from_bytes()
    request = ReviewRunRequestV2(
        case_id="case_one_shot_extract",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "invoice.jpg",
                "image_pages": [page],
            }
        ],
        llm_mode="llm",
    )
    state = _build_initial_state(request)
    owned_payload = request.documents[0].model_dump(mode="json")
    payload_store = SensitiveDocumentPayloadStore({"doc_image": owned_payload})

    state.update(
        classify_documents(
            state,
            analyzer,
            payload_store=payload_store,
        )
    )

    assert payload_store.is_empty
    assert "data_base64" not in json.dumps(owned_payload, default=str)
    assert "data_base64" not in json.dumps(state, default=str)
    assert page.data_base64 not in json.dumps(state, default=str)
    # Caller ownership is explicit: the workflow never mutates this request.
    assert request.documents[0].image_pages[0].data_base64 == page.data_base64

    state.update(extract_documents(state))

    assert state["extracted_documents"]["doc_image"]["document_type"] == "invoice"
    assert "data_base64" not in json.dumps(state, default=str)
    assert page.data_base64 not in json.dumps(state, default=str)


def test_payload_store_is_cleared_and_owned_copy_scrubbed_on_analysis_exception():
    class UnexpectedFailureAnalyzer:
        def analyze(self, _document):
            raise RuntimeError("unexpected classification failure")

    page = _page_from_bytes()
    request = ReviewRunRequestV2(
        case_id="case_analysis_exception",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "invoice.jpg",
                "image_pages": [page],
            }
        ],
        llm_mode="llm",
    )
    state = _build_initial_state(request)
    owned_payload = request.documents[0].model_dump(mode="json")
    payload_store = SensitiveDocumentPayloadStore({"doc_image": owned_payload})

    with pytest.raises(RuntimeError, match="unexpected classification failure"):
        classify_documents(
            state,
            UnexpectedFailureAnalyzer(),
            payload_store=payload_store,
        )

    assert payload_store.is_empty
    assert "data_base64" not in json.dumps(owned_payload, default=str)
    assert request.documents[0].image_pages[0].data_base64 == page.data_base64


def test_payload_store_is_empty_when_evidence_retrieval_begins():
    payload = _vision_payload()
    analyzer = ClaudeDocumentAnalyzer(
        FakeVisionTransport(payload),
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(payload),
    )
    page = _page_from_bytes()
    request = ReviewRunRequestV2(
        case_id="case_retrieval_payload_boundary",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "invoice.jpg",
                "image_pages": [page],
            }
        ],
        llm_mode="llm",
    )
    owned_payload = request.documents[0].model_dump(mode="json")
    payload_store = SensitiveDocumentPayloadStore({"doc_image": owned_payload})
    retrieval_observed = False

    class StoreAssertingRetriever:
        def retrieve(self, retrieval_request):
            nonlocal retrieval_observed
            retrieval_observed = True
            assert payload_store.is_empty
            assert "data_base64" not in json.dumps(owned_payload, default=str)
            return MockRegulationRetriever().retrieve(retrieval_request)

    graph = build_review_graph(
        StoreAssertingRetriever(),
        analyzer,
        payload_store=payload_store,
    )
    result = graph.invoke(_build_initial_state(request))
    public = to_review_run_result_v2(result)

    assert payload_store.is_empty
    assert retrieval_observed
    assert "data_base64" not in json.dumps(result, default=str)
    assert "data_base64" not in public.model_dump_json()
    assert page.data_base64 not in json.dumps(result, default=str)
    assert page.data_base64 not in public.model_dump_json()
    assert request.documents[0].image_pages[0].data_base64 == page.data_base64


def test_analyzer_places_images_before_metadata_and_uses_vision_contract():
    analysis, transport, _ = _analyze_image()

    request = transport.calls[0]
    content = request["user_message"]
    assert isinstance(content, list)
    assert [block["type"] for block in content] == ["image", "text"]
    assert content[0]["source"]["type"] == "base64"
    metadata_text = content[-1]["text"]
    metadata = json.loads(metadata_text)
    assert metadata["protocol"] == "customs_document_vision_analysis_input/v1"
    assert metadata["untrusted_metadata"]["pages"][0]["sha256"]
    assert metadata["untrusted_metadata"]["pages"][0]["provenance"]["source_sha256"]
    assert "data_base64" not in metadata_text
    assert content[0]["source"]["data"] not in metadata_text
    assert metadata["independent_ocr_layout_ledger"]["provider_id"] == ("fake-independent-ocr")
    assert analysis.metadata.prompt_id == "customs_document_vision_analysis"
    assert analysis.metadata.prompt_version == "1.3.0"
    assert (
        analysis.metadata.prompt_sha256
        == hashlib.sha256(request["system_prompt"].encode("utf-8")).hexdigest()
    )
    assert analysis.metadata.input_modality == "image"
    assert "independent OCR/layout engine" in request["system_prompt"]
    schema = request["output_schema"]
    assert schema["title"] == "ClaudeVisionDocumentCandidate"
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ClaudeVisionEvidence"]["additionalProperties"] is False


def test_image_analysis_requires_independent_ocr_provider():
    payload = _vision_payload()
    transport = FakeVisionTransport(payload)
    analyzer = ClaudeDocumentAnalyzer(
        transport,
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
    )
    page = _page_from_bytes()

    with pytest.raises(LLMOutputValidationError) as exc_info:
        analyzer.analyze(
            {
                "document_id": "doc_image",
                "filename": "invoice.jpg",
                "content_text": None,
                "image_pages": [page.model_dump(mode="json")],
            }
        )

    assert exc_info.value.code == "ocr_layout_provider_missing"
    assert exc_info.value.provider == "ocr_layout"
    assert transport.calls == []


def test_valid_vision_candidate_maps_bbox_page_hash_dimensions_and_modality():
    analysis, _, document = _analyze_image()

    extracted = analysis_to_extracted_document(document, analysis)
    field = extracted["fields"]["invoice_no"]
    evidence = field["evidence"]

    assert field["value"] == "INV-TEST-001"
    assert field["confidence_basis"] == "model_mapped_to_independent_ocr_uncalibrated"
    assert evidence["raw_text"] == "INVOICE NO INV-TEST-001"
    assert evidence["source_modality"] == "image"
    assert evidence["page"] == 1
    assert evidence["bbox"] == {
        "page": 1,
        "x1": 20.0,
        "y1": 80.0,
        "x2": 250.0,
        "y2": 110.0,
    }
    assert evidence["image_sha256"] == document["image_pages"][0]["sha256"]
    assert evidence["image_width_px"] == 400
    assert evidence["image_height_px"] == 600
    assert evidence["coordinate_space"] == "original_image_pixels_top_left"
    assert evidence["ocr_ledger_id"].startswith("ocr_")
    assert evidence["ocr_span_ids"]
    assert evidence["text_start"] is None
    assert evidence["text_end"] is None
    assert extracted["input_modality"] == "image"
    assert extracted["line_items"][0]["item_id"] == "doc_image_line_1"


def test_public_field_uses_independent_ocr_bbox_not_claude_observed_bbox():
    payload = _vision_payload()
    invoice_evidence = payload["fields"][0]["evidence"]
    span_id = invoice_evidence["ocr_span_ids"][0]
    provider = FakeOcrLayoutProvider(
        payload,
        bbox_overrides={span_id: (25, 85, 275, 120)},
    )

    analysis, _, document = _analyze_image(payload, ocr_provider=provider)
    extracted = analysis_to_extracted_document(document, analysis)
    evidence = extracted["fields"]["invoice_no"]["evidence"]

    assert invoice_evidence["model_observed_bbox"] == {
        "page": 1,
        "x1": 20,
        "y1": 80,
        "x2": 250,
        "y2": 110,
    }
    assert evidence["bbox"] == {
        "page": 1,
        "x1": 25.0,
        "y1": 85.0,
        "x2": 275.0,
        "y2": 120.0,
    }
    assert evidence["ocr_span_ids"] == [span_id]


def test_claude_reference_outside_independent_ocr_ledger_fails_closed():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_payload["fields"][0]["evidence"]["ocr_span_ids"] = ["span_missing"]

    with pytest.raises(LLMOutputValidationError) as exc_info:
        _analyze_image(
            candidate_payload,
            ocr_provider=FakeOcrLayoutProvider(trusted_payload),
        )

    assert exc_info.value.code == "ocr_span_not_found"


def test_unresolved_crop_conflict_is_preserved_and_field_is_omitted():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    field = candidate_payload["fields"][0]
    field["value"] = "INV-FORGED-999"
    field["evidence"]["source_text"] = "INVOICE NO INV-FORGED-999"
    span_ids = trusted_payload["fields"][0]["evidence"]["ocr_span_ids"]
    transport = FakeVisionTransport(
        candidate_payload,
        crop_payload={
            "schema_version": "1.0",
            "value": "INV-FORGED-999",
            "source_text": "INV-FORGED-999",
            "ocr_span_ids": span_ids,
        },
    )

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=FakeOcrLayoutProvider(trusted_payload),
        transport=transport,
    )
    extracted = analysis_to_extracted_document(document, analysis)
    conflict = analysis.model_conflicts[0]

    assert "invoice_no" not in extracted["fields"]
    assert conflict["field_path"] == "fields.invoice_no"
    assert conflict["status"] == "crop_unresolved"
    assert conflict["crop_attempted"] is True
    assert conflict["candidate_value_sha256"] == hashlib.sha256(b"INV-FORGED-999").hexdigest()
    assert conflict["crop_source_image_sha256"] == document["image_pages"][0]["sha256"]
    assert conflict["crop_image_sha256"]
    assert conflict["crop_transform_sha256"]
    assert analysis.additional_model_calls[0].input_modality == "image_crop"
    assert len(transport.calls) == 2
    assert transport.calls[1]["output_schema"]["title"] == "ClaudeCropReadCandidate"
    assert transport.calls[1]["system_prompt"] != transport.calls[0]["system_prompt"]
    assert "data_base64" not in json.dumps(conflict)
    assert "base64" not in json.dumps(analysis.additional_model_calls[0].as_audit_dict())


def test_crop_value_is_adopted_only_when_it_matches_independent_ocr():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    field = candidate_payload["fields"][0]
    field["value"] = "INV-TEST-OO1"
    field["evidence"]["source_text"] = "INVOICE NO INV-TEST-OO1"
    span_ids = trusted_payload["fields"][0]["evidence"]["ocr_span_ids"]
    provider = FakeOcrLayoutProvider(
        trusted_payload,
        bbox_overrides={span_ids[0]: (100, 100, 220, 140)},
    )
    transport = FakeVisionTransport(
        candidate_payload,
        crop_payload={
            "schema_version": "1.0",
            "value": "INV-TEST-001",
            "source_text": "INV-TEST-001",
            "ocr_span_ids": span_ids,
        },
    )

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=provider,
        transport=transport,
    )
    extracted = analysis_to_extracted_document(document, analysis)

    assert extracted["fields"]["invoice_no"]["value"] == "INV-TEST-001"
    assert analysis.model_conflicts[0]["status"] == "resolved"
    assert analysis.model_conflicts[0]["crop_attempted"] is True
    assert analysis.model_conflicts[0]["crop_bbox"] == {
        "page": 1,
        "x1": 88.0,
        "y1": 88.0,
        "x2": 232.0,
        "y2": 152.0,
    }
    assert analysis.additional_model_calls[0].prompt_id == ("customs_document_crop_reconciliation")
    assert analysis.additional_model_calls[0].prompt_sha256
    assert (
        analysis.additional_model_calls[0].prompt_sha256
        == hashlib.sha256(transport.calls[1]["system_prompt"].encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize(
    ("ocr_identifier", "candidate_identifier"),
    [
        ("ABC123EE", "ABC123"),
        ("ABC123 EE", "ABC123"),
        ("ABC123-EE", "ABC123"),
        ("ABC 123 EE", "ABC 123"),
    ],
)
def test_identifier_suffix_is_not_silently_truncated_from_selected_ocr(
    ocr_identifier,
    candidate_identifier,
):
    trusted_payload = _vision_payload()
    trusted_field = trusted_payload["fields"][0]
    trusted_field["value"] = ocr_identifier
    trusted_field["evidence"]["source_text"] = f"INVOICE NO: {ocr_identifier}"
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_field = candidate_payload["fields"][0]
    candidate_field["value"] = candidate_identifier
    candidate_field["evidence"]["source_text"] = candidate_identifier

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=FakeOcrLayoutProvider(trusted_payload),
    )
    extracted = analysis_to_extracted_document(document, analysis)

    assert "invoice_no" not in extracted["fields"]
    assert analysis.model_conflicts[0]["field_path"] == "fields.invoice_no"
    assert analysis.model_conflicts[0]["status"] != "resolved"


def test_crop_reread_cannot_resolve_identifier_by_dropping_ocr_suffix():
    trusted_payload = _vision_payload()
    trusted_field = trusted_payload["fields"][0]
    trusted_field["value"] = "ABC 123 EE"
    trusted_field["evidence"]["source_text"] = "INVOICE NO: ABC 123 EE"
    span_ids = trusted_field["evidence"]["ocr_span_ids"]
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_field = candidate_payload["fields"][0]
    candidate_field["value"] = "WRONG"
    candidate_field["evidence"]["source_text"] = "WRONG"
    transport = FakeVisionTransport(
        candidate_payload,
        crop_payload={
            "schema_version": "1.0",
            "value": "ABC 123",
            "source_text": "ABC 123",
            "ocr_span_ids": span_ids,
        },
    )

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=FakeOcrLayoutProvider(trusted_payload),
        transport=transport,
    )
    extracted = analysis_to_extracted_document(document, analysis)

    assert "invoice_no" not in extracted["fields"]
    assert analysis.model_conflicts[0]["status"] == "crop_unresolved"
    assert len(analysis.additional_model_calls) == 1


def test_complete_identifier_allows_known_label_prefix():
    payload = _vision_payload()
    field = payload["fields"][0]
    field["value"] = "ABC123"
    field["evidence"]["source_text"] = "INVOICE NO: ABC123"

    analysis, _, document = _analyze_image(payload)
    extracted = analysis_to_extracted_document(document, analysis)

    assert extracted["fields"]["invoice_no"]["value"] == "ABC123"
    assert analysis.model_conflicts == ()


def test_free_form_name_keeps_existing_bounded_substring_grounding():
    trusted_payload = _vision_payload()
    trusted_payload["fields"].append(
        {
            "name": "seller",
            "value": "ACME TRADING COMPANY",
            "evidence": _vision_evidence(
                "SELLER: ACME TRADING COMPANY",
                box=(20, 120, 320, 150),
            ),
        }
    )
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_field = candidate_payload["fields"][-1]
    candidate_field["value"] = "ACME"
    candidate_field["evidence"]["source_text"] = "ACME"

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=FakeOcrLayoutProvider(trusted_payload),
    )
    extracted = analysis_to_extracted_document(document, analysis)

    assert extracted["fields"]["seller"]["value"] == "ACME"
    assert analysis.model_conflicts == ()


def test_crop_budget_is_shared_and_capped_at_six_calls_per_run():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_payload["fields"][0]["value"] = "INV-CONFLICT"
    candidate_payload["fields"][0]["evidence"]["source_text"] = "INV-CONFLICT"
    span_ids = trusted_payload["fields"][0]["evidence"]["ocr_span_ids"]
    budget = CropReadBudget()
    statuses: list[str] = []
    attempted_calls = 0

    for _ in range(7):
        transport = FakeVisionTransport(
            candidate_payload,
            crop_payload={
                "schema_version": "1.0",
                "value": None,
                "source_text": "unreadable",
                "ocr_span_ids": span_ids,
            },
        )
        analysis, _, _ = _analyze_image(
            copy.deepcopy(candidate_payload),
            ocr_provider=FakeOcrLayoutProvider(trusted_payload),
            transport=transport,
            crop_budget=budget,
        )
        statuses.append(analysis.model_conflicts[0]["status"])
        attempted_calls += len(analysis.additional_model_calls)

    assert statuses == ["crop_unresolved"] * 6 + ["crop_budget_exhausted"]
    assert attempted_calls == 6
    assert budget.used_calls == 6


def test_workflow_enforces_one_shared_six_crop_budget_across_documents():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_payload["fields"][0]["value"] = "INV-CONFLICT"
    candidate_payload["fields"][0]["evidence"]["source_text"] = "INV-CONFLICT"
    span_ids = trusted_payload["fields"][0]["evidence"]["ocr_span_ids"]
    transport = FakeVisionTransport(
        candidate_payload,
        crop_payload={
            "schema_version": "1.0",
            "value": None,
            "source_text": "unreadable",
            "ocr_span_ids": span_ids,
        },
    )
    analyzer = ClaudeDocumentAnalyzer(
        transport,
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(trusted_payload),
    )
    page = _page_from_bytes()
    request = ReviewRunRequestV2(
        case_id="case_crop_run_budget",
        documents=[
            {
                "document_id": f"doc_image_{index}",
                "filename": f"invoice_{index}.jpg",
                "image_pages": [page],
            }
            for index in range(7)
        ],
        llm_mode="llm",
    )

    state = run_review_v2(
        request,
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )

    assert [item["status"] for item in state["model_conflicts"]] == (
        ["crop_unresolved"] * 6 + ["crop_budget_exhausted"]
    )
    assert sum(call["input_modality"] == "image_crop" for call in state["model_calls"]) == 6
    assert len(transport.calls) == 13
    assert state["final_status"] == "needs_review"


def test_crop_provider_failure_preserves_hashed_conflict_and_safe_error():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_payload["fields"][0]["value"] = "INV-CONFLICT"
    candidate_payload["fields"][0]["evidence"]["source_text"] = "INV-CONFLICT"
    transport = FailingCropTransport(candidate_payload)

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=FakeOcrLayoutProvider(trusted_payload),
        transport=transport,
    )
    extracted = analysis_to_extracted_document(document, analysis)

    assert "invoice_no" not in extracted["fields"]
    assert analysis.model_conflicts[0]["status"] == "crop_provider_error"
    assert analysis.additional_model_calls == ()
    assert analysis.additional_model_errors[0]["code"] == ("crop_provider_unavailable")
    assert analysis.additional_model_errors[0]["input_modality"] == "image_crop"
    assert "INV-CONFLICT" not in json.dumps(analysis.model_conflicts)
    assert "data" not in analysis.additional_model_errors[0]


def test_vision_workflow_records_modality_and_cannot_auto_complete():
    analysis, transport, document = _analyze_image()
    analyzer = ClaudeDocumentAnalyzer(
        transport,
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(_vision_payload()),
    )
    request = ReviewRunRequestV2(
        case_id="case_image",
        documents=[document],
        llm_mode="llm",
    )

    state = run_review_v2(
        request,
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )
    public = to_review_run_result_v2(state)

    assert analysis.metadata.input_modality == "image"
    assert state["final_status"] == "needs_review"
    assert state["requires_human_review"] is True
    assert state["model_errors"] == []
    assert state["model_calls"][0]["input_modality"] == "image"
    assert public.model_calls[0].input_modality == "image"
    assert public.model_calls[0].ocr_ledger_id
    assert public.model_calls[0].ocr_provider_id == "fake-independent-ocr"
    assert (
        state["extracted_documents"]["doc_image"]["fields"]["invoice_no"]["evidence"][
            "source_modality"
        ]
        == "image"
    )
    assert all("data_base64" not in json.dumps(call) for call in state["model_calls"])
    assert "data_base64" not in json.dumps(state, default=str)
    assert state["sensitive_inputs_redacted"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.update({"document_id": "doc_other"}),
        lambda source: source.update({"ocr_ledger_id": "ocr_" + "f" * 64}),
        lambda source: source["bbox"].update({"x2": int(source["image_width_px"]) + 1}),
    ],
)
def test_v2_image_field_provenance_must_match_parent_document_and_ocr_ledger(mutate):
    _, transport, document = _analyze_image()
    analyzer = ClaudeDocumentAnalyzer(
        transport,
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(_vision_payload()),
    )
    request = ReviewRunRequestV2(
        case_id="case_image_provenance",
        documents=[document],
        llm_mode="llm",
    )
    state = run_review_v2(
        request,
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )
    payload = to_review_run_result_v2(state).model_dump(mode="python")
    source = payload["document_results"][0]["fields"][0]["source"]
    mutate(source)

    with pytest.raises(ValidationError):
        ReviewRunResultV2.model_validate(payload)


def test_workflow_preserves_unresolved_crop_conflict_without_raw_crop_bytes():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_payload["fields"][0]["value"] = "INV-CONFLICT"
    candidate_payload["fields"][0]["evidence"]["source_text"] = "INV-CONFLICT"
    span_ids = trusted_payload["fields"][0]["evidence"]["ocr_span_ids"]
    transport = FakeVisionTransport(
        candidate_payload,
        crop_payload={
            "schema_version": "1.0",
            "value": None,
            "source_text": "unreadable",
            "ocr_span_ids": span_ids,
        },
    )
    analyzer = ClaudeDocumentAnalyzer(
        transport,
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(trusted_payload),
    )
    page = _page_from_bytes()
    request = ReviewRunRequestV2(
        case_id="case_crop_conflict",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "invoice.jpg",
                "image_pages": [page],
            }
        ],
        llm_mode="llm",
    )

    state = run_review_v2(
        request,
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )
    public = to_review_run_result_v2(state)

    assert state["final_status"] == "needs_review"
    assert state["requires_human_review"] is True
    assert state["model_conflicts"][0]["status"] == "crop_unresolved"
    assert public.model_conflicts[0].status == "crop_unresolved"
    assert [call["input_modality"] for call in state["model_calls"]] == [
        "image",
        "image_crop",
    ]
    serialized_state = json.dumps(state, default=str)
    assert "data_base64" not in serialized_state
    assert '"data"' not in json.dumps(state["model_calls"], default=str)
    assert "INV-CONFLICT" not in json.dumps(state["model_conflicts"])


@pytest.mark.parametrize(
    ("mutate", "error_code"),
    [
        (
            lambda payload: payload["fields"][0]["evidence"]["model_observed_bbox"].update(
                {"x2": 401}
            ),
            "vision_bbox_out_of_bounds",
        ),
        (
            lambda payload: payload["classification_evidence"]["model_observed_bbox"].update(
                {"page": 2}
            ),
            "vision_page_not_supplied",
        ),
        (
            lambda payload: payload["fields"][0].update({"name": "severity"}),
            "field_not_allowed",
        ),
        (
            lambda payload: payload.update(
                {
                    "line_items": [
                        _line_item(y1=300, description="First row"),
                        _line_item(y1=200, description="Second row"),
                    ]
                }
            ),
            "line_item_source_order_invalid",
        ),
        (
            lambda payload: payload["line_items"][0]["fields"][0]["evidence"][
                "model_observed_bbox"
            ].update({"x1": 10, "x2": 30, "y1": 300, "y2": 330}),
            "line_item_bbox_not_in_row",
        ),
    ],
)
def test_invalid_vision_grounding_fails_closed(mutate, error_code):
    payload = _vision_payload()
    mutate(payload)

    with pytest.raises(LLMOutputValidationError) as exc_info:
        _analyze_image(payload)

    assert exc_info.value.code == error_code


def test_value_not_present_in_model_source_is_dropped_as_a_binding_conflict():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    candidate_payload["fields"][0]["value"] = "HALLUCINATED"

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=FakeOcrLayoutProvider(trusted_payload),
    )
    extracted = analysis_to_extracted_document(document, analysis)

    assert "invoice_no" not in extracted["fields"]
    assert extracted["fields"]["total_amount"]["value"] == "12000"
    assert analysis.model_conflicts[0]["field_path"] == "fields.invoice_no"
    assert analysis.model_conflicts[0]["status"] == "crop_schema_error"


def test_classification_requires_decisive_type_label_from_selected_ocr():
    payload = _vision_payload()
    evidence = payload["classification_evidence"]
    assert isinstance(evidence, dict)
    evidence["source_text"] = "INVOICE NO"

    with pytest.raises(LLMOutputValidationError) as exc_info:
        _analyze_image(payload)

    assert exc_info.value.code == "classification_label_not_grounded"


def test_unbound_row_is_removed_and_preserved_as_a_hashed_conflict():
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    line_items = candidate_payload["line_items"]
    assert isinstance(line_items, list)
    row = line_items[0]
    assert isinstance(row, dict)
    row_evidence = row["row_evidence"]
    assert isinstance(row_evidence, dict)
    row_evidence["source_text"] = "1 FORGED ROW 731815 10 KG 12000"

    analysis, _, document = _analyze_image(
        candidate_payload,
        ocr_provider=FakeOcrLayoutProvider(trusted_payload),
    )
    extracted = analysis_to_extracted_document(document, analysis)
    conflict = analysis.model_conflicts[0]

    assert extracted["line_items"] == []
    assert conflict["field_path"] == "line_items.0.__row__"
    assert conflict["status"] == "crop_not_eligible"
    assert conflict["crop_attempted"] is False
    assert "FORGED ROW" not in json.dumps(conflict)


class _FailingOcrLayoutProvider:
    def analyze(self, *, document_id, pages):
        raise OcrLayoutProviderError(
            "ocr_runtime_failed",
            "Synthetic content-free OCR failure.",
        )


def test_image_ocr_failure_is_a_serializable_failed_v2_document():
    payload = _vision_payload()
    analyzer = ClaudeDocumentAnalyzer(
        FakeVisionTransport(payload),
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=_FailingOcrLayoutProvider(),
    )
    request = ReviewRunRequestV2(
        case_id="case_ocr_failure",
        review_run_id="run_ocr_failure",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "private-customer-invoice-name.jpg",
                "image_pages": [_page_from_bytes()],
            }
        ],
        llm_mode="llm",
    )

    state = run_review_v2(
        request,
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )
    public = to_review_run_result_v2(state)
    document = public.document_results[0]
    serialized = public.model_dump_json()

    assert public.status == "needs_review"
    assert public.requires_human_review is True
    assert document.document_type == "unknown"
    assert document.input_modality == "image"
    assert document.extraction_status == "failed"
    assert document.extraction_confidence == 0
    assert document.fields == []
    assert document.line_items == []
    assert document.ocr_layout is None
    assert state["extracted_documents"]["doc_image"]["extraction_status"] == "failed"
    assert "doc_image:extraction_failed" in state["report_json"]["summary"]["outcome_reasons"]
    assert "private-customer-invoice-name.jpg" not in serialized


@pytest.mark.parametrize("failure_kind", ["schema", "grounding"])
def test_image_schema_or_grounding_failure_is_a_serializable_failed_v2_document(
    failure_kind,
):
    trusted_payload = _vision_payload()
    candidate_payload = copy.deepcopy(trusted_payload)
    if failure_kind == "schema":
        candidate_payload["unexpected_provider_field"] = True
    else:
        classification_evidence = candidate_payload["classification_evidence"]
        assert isinstance(classification_evidence, dict)
        classification_evidence["source_text"] = "FORGED CLASSIFICATION"
    analyzer = ClaudeDocumentAnalyzer(
        FakeVisionTransport(candidate_payload),
        ClaudeSettings(api_key=SecretStr("unit-test-key")),
        ocr_layout_provider=FakeOcrLayoutProvider(trusted_payload),
    )
    request = ReviewRunRequestV2(
        case_id=f"case_{failure_kind}_failure",
        review_run_id=f"run_{failure_kind}_failure",
        documents=[
            {
                "document_id": "doc_image",
                "filename": "private-customer-document.jpg",
                "image_pages": [_page_from_bytes()],
            }
        ],
        llm_mode="llm",
    )

    state = run_review_v2(
        request,
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )
    public = to_review_run_result_v2(state)
    document = public.document_results[0]

    assert public.status == "needs_review"
    assert document.extraction_status == "failed"
    assert document.document_type == "unknown"
    assert document.fields == []
    assert document.line_items == []
    assert document.ocr_layout is None
    assert state["model_errors"][0]["code"] == (
        "structured_output_invalid" if failure_kind == "schema" else "vision_source_text_not_in_ocr"
    )


def test_v2_report_rejects_unknown_nested_keys_and_structured_issue_values():
    state = run_review(ReviewRunRequest(scenario_id="invoice_total_mismatch"))
    public = to_review_run_result_v2(state)
    payload = public.model_dump(mode="python")
    ReviewRunResultV2.model_validate(payload)

    nested_targets = (
        ("filename", ("report", "summary")),
        ("raw_response", ("report", "document_summary", 0)),
        ("data_base64", ("report", "issues", 0)),
        ("raw_response", ("report", "rag_evidence", 0)),
        ("data_base64", ("report", "rag_evidence", 0, "raw_scores", 0)),
    )
    for forbidden_key, path in nested_targets:
        malicious = copy.deepcopy(payload)
        target = malicious
        for part in path:
            target = target[part]
        target[forbidden_key] = "must-not-enter-public-contract"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ReviewRunResultV2.model_validate(malicious)

    for forbidden_value in (
        {"filename": "private.pdf"},
        ["data_base64", "private-image-content"],
    ):
        malicious = copy.deepcopy(payload)
        malicious["report"]["issues"][0]["expected_value"] = forbidden_value
        with pytest.raises(ValidationError):
            ReviewRunResultV2.model_validate(malicious)


def test_v2_state_conversion_rejects_unknown_nested_report_keys():
    state = run_review(ReviewRunRequest(scenario_id="happy_path"))
    state["report_json"]["summary"]["raw_response"] = {"secret": True}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        to_review_run_result_v2(state)


def test_image_self_correction_does_not_rewrite_observation_or_value():
    original_field = {
        "name": "total_amount",
        "value": "12OO",
        "confidence": 0.70,
        "confidence_basis": "model_mapped_to_independent_ocr_uncalibrated",
        "evidence": {
            "document_id": "doc_image",
            "page": 1,
            "bbox": {"page": 1, "x1": 10, "y1": 10, "x2": 80, "y2": 30},
            "raw_text": "TOTAL 12OO",
            "text_start": None,
            "text_end": None,
            "confidence": 0.70,
            "source_modality": "image",
            "image_sha256": "a" * 64,
            "image_width_px": 400,
            "image_height_px": 600,
            "coordinate_space": "original_image_pixels_top_left",
            "ocr_ledger_id": "ocr_" + "b" * 64,
            "ocr_span_ids": ["span_total_amount"],
        },
    }
    state = {
        "extracted_documents": {
            "doc_image": {
                "fields": {"total_amount": copy.deepcopy(original_field)},
                "line_items": [],
                "extraction_confidence": 0.70,
            }
        },
        "retry_count": 0,
        "requires_human_review": True,
        "audit_trail": [],
    }

    update = self_correct_extraction(state)
    corrected = update["extracted_documents"]["doc_image"]["fields"]["total_amount"]

    assert corrected == original_field
    assert "value_source" not in corrected
    assert update["retry_count"] == 1
