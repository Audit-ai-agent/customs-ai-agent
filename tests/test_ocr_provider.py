from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import queue
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw

from customs_ai_agent.adapters import build_ocr_layout_provider
from customs_ai_agent.adapters import ocr as ocr_module
from customs_ai_agent.adapters.ocr import OcrLayoutProviderError
from customs_ai_agent import ocr_worker as worker_module
from customs_ai_agent.ocr_reading_order import (
    MAX_OCR_SPANS_PER_PAGE,
    OcrSpanLimitExceededError,
    preserve_provider_span_order,
)
from customs_ai_agent.ocr_worker import (
    _PaddleBackend,
    _confidence,
    _docling_document_to_spans,
    _paddle_result_to_spans,
    _process_request,
)
from customs_ai_agent.schemas.models import (
    ImagePageInput,
    ImagePageProvenance,
    OcrLayoutLedger,
    stable_ocr_ledger_id,
)


def _image_page(*, side_px: int = 100) -> ImagePageInput:
    output = io.BytesIO()
    Image.new("RGB", (side_px, side_px), "white").save(output, format="PNG")
    raw = output.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    return ImagePageInput(
        page_number=1,
        media_type="image/png",
        data_base64=base64.b64encode(raw).decode("ascii"),
        byte_length=len(raw),
        sha256=digest,
        width_px=side_px,
        height_px=side_px,
        provenance=ImagePageProvenance(
            source_media_type="image/png",
            source_sha256=digest,
            source_page_number=1,
            transforms=["metadata_strip"],
            original_exif_orientation=1,
        ),
    )


def _ok_response(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "customs-ocr-worker/v2",
        "request_id": request["request_id"],
        "status": "ok",
        "result": {
            "provider_id": request["provider"],
            "engine_id": "test-engine",
            "engine_version": "1.2.3",
            "reading_order_policy_id": "worker-emitted-sequence/v1",
            "model_id": "test-model-set",
            "model_assets_lock_sha256": "a" * 64,
            "pages": [
                {
                    "page_number": 1,
                    "spans": [
                        {
                            "role": "text",
                            "text": "Invoice",
                            "bbox": [5, 5, 45, 15],
                            "confidence": 0.99,
                        },
                        {
                            "role": "table",
                            "text": "Widget | 2",
                            "bbox": [10, 20, 90, 80],
                            "confidence": 0.95,
                            "table_key": "table-0",
                        },
                        {
                            "role": "table_row",
                            "text": "Widget | 2",
                            "bbox": [10, 20, 90, 50],
                            "confidence": 0.95,
                            "table_key": "table-0",
                            "row_index": 0,
                        },
                        {
                            "role": "table_cell",
                            "text": "Widget",
                            "bbox": [10, 20, 60, 50],
                            "confidence": 0.96,
                            "table_key": "table-0",
                            "row_index": 0,
                            "column_index": 0,
                        },
                        {
                            "role": "table_cell",
                            "text": "2",
                            "bbox": [60, 20, 90, 50],
                            "confidence": 0.94,
                            "table_key": "table-0",
                            "row_index": 0,
                            "column_index": 1,
                        },
                    ],
                }
            ],
        },
    }


class _FakeWorker:
    def __init__(
        self,
        responder: Callable[[dict[str, Any]], dict[str, Any]] = _ok_response,
    ) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def request(self, payload: dict[str, object]) -> dict[str, Any]:
        request = cast(dict[str, Any], payload)
        self.requests.append(request)
        return self.responder(request)

    def close(self) -> None:
        self.closed = True


def _provider_with_worker(worker: _FakeWorker) -> Any:
    return ocr_module._SubprocessOcrLayoutProvider(
        provider="paddle",
        python_executable=Path(sys.executable),
        worker=cast(Any, worker),
    )


def test_builder_rejects_unknown_provider_and_missing_python() -> None:
    with pytest.raises(OcrLayoutProviderError) as provider_error:
        build_ocr_layout_provider(cast(Any, "unknown"))
    assert provider_error.value.code == "ocr_provider_invalid"

    with pytest.raises(OcrLayoutProviderError) as python_error:
        build_ocr_layout_provider(
            "paddle",
            python_executable=Path("Z:/definitely-missing/custom-python.exe"),
        )
    assert python_error.value.code == "ocr_python_unavailable"


def test_provider_builds_deterministic_provenance_bound_ledger() -> None:
    page = _image_page()
    worker = _FakeWorker()
    provider = _provider_with_worker(worker)

    first = provider.analyze(document_id="invoice-001", pages=[page])
    second = provider.analyze(document_id="invoice-001", pages=[page])
    provider.close()

    assert first == second
    assert first.provider_id == "paddle"
    assert first.engine_id == "test-engine"
    assert first.engine_version == "1.2.3"
    assert first.reading_order_policy_id == "worker-emitted-sequence/v1"
    assert first.pages[0].image_sha256 == page.sha256
    assert first.pages[0].width_px == 100
    assert first.pages[0].height_px == 100
    assert [span.reading_order for span in first.pages[0].spans] == list(range(5))
    cells = [span for span in first.pages[0].spans if span.role == "table_cell"]
    assert [(cell.row_span, cell.column_span) for cell in cells] == [(1, 1), (1, 1)]
    table_ids = {span.table_id for span in first.pages[0].spans if span.table_id is not None}
    assert len(table_ids) == 1
    assert len({span.span_id for span in first.pages[0].spans}) == 5
    assert len(worker.requests) == 2
    assert worker.requests[0]["pages"][0]["data_base64"] == page.data_base64
    assert page.data_base64 not in first.model_dump_json()
    assert worker.closed


def test_merged_cell_topology_is_preserved_and_hash_bound() -> None:
    def response_with_topology(
        *,
        merged: bool,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def responder(request: dict[str, Any]) -> dict[str, Any]:
            response = _ok_response(request)
            page = response["result"]["pages"][0]
            common = {
                "confidence": 0.95,
                "table_key": "table-0",
            }
            cells = [
                {
                    **common,
                    "role": "table_cell",
                    "text": "Group",
                    "bbox": [10, 20, 50, 80] if merged else [10, 20, 50, 50],
                    "row_index": 0,
                    "column_index": 0,
                    "row_span": 2 if merged else 1,
                    "column_span": 1,
                },
                {
                    **common,
                    "role": "table_cell",
                    "text": "First",
                    "bbox": [50, 20, 90, 50],
                    "row_index": 0,
                    "column_index": 1,
                    "row_span": 1,
                    "column_span": 1,
                },
            ]
            if not merged:
                cells.append(
                    {
                        **common,
                        "role": "table_cell",
                        "text": "Group",
                        "bbox": [10, 50, 50, 80],
                        "row_index": 1,
                        "column_index": 0,
                        "row_span": 1,
                        "column_span": 1,
                    }
                )
            cells.append(
                {
                    **common,
                    "role": "table_cell",
                    "text": "Second",
                    "bbox": [50, 50, 90, 80],
                    "row_index": 1,
                    "column_index": 1,
                    "row_span": 1,
                    "column_span": 1,
                }
            )
            page["spans"] = [
                {
                    **common,
                    "role": "table",
                    "text": "Group | First | Second",
                    "bbox": [10, 20, 90, 80],
                },
                {
                    **common,
                    "role": "table_row",
                    "text": "Group | First",
                    "bbox": [10, 20, 90, 50],
                    "row_index": 0,
                },
                {
                    **common,
                    "role": "table_row",
                    "text": "Group | Second",
                    "bbox": [10, 50, 90, 80],
                    "row_index": 1,
                },
                *cells,
            ]
            return response

        return responder

    merged = _provider_with_worker(_FakeWorker(response_with_topology(merged=True))).analyze(
        document_id="invoice-merged-topology",
        pages=[_image_page()],
    )
    unmerged = _provider_with_worker(_FakeWorker(response_with_topology(merged=False))).analyze(
        document_id="invoice-merged-topology",
        pages=[_image_page()],
    )
    merged_cell = next(span for span in merged.pages[0].spans if span.role == "table_cell")
    unmerged_cell = next(span for span in unmerged.pages[0].spans if span.role == "table_cell")

    assert (merged_cell.row_span, merged_cell.column_span) == (2, 1)
    assert (unmerged_cell.row_span, unmerged_cell.column_span) == (1, 1)
    assert merged_cell.span_id != unmerged_cell.span_id
    assert merged_cell.table_id != unmerged_cell.table_id
    assert merged.ledger_id != unmerged.ledger_id


@pytest.mark.parametrize(
    "mutation",
    [
        "span_beyond_rows",
        "overlapping_logical_cells",
        "overlapping_physical_cells",
        "cell_outside_row_band",
        "ragged_row",
        "unkeyed_table",
    ],
)
def test_provider_rejects_invalid_table_graph_topology(mutation: str) -> None:
    def malformed_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        spans = response["result"]["pages"][0]["spans"]
        cells = [span for span in spans if span["role"] == "table_cell"]
        if mutation == "span_beyond_rows":
            cells[0]["row_span"] = 2
        elif mutation == "overlapping_logical_cells":
            cells[1]["column_index"] = 0
        elif mutation == "overlapping_physical_cells":
            cells[1]["bbox"] = [50, 20, 90, 50]
            cells[0]["bbox"] = [10, 20, 70, 50]
        elif mutation == "cell_outside_row_band":
            cells[0]["bbox"] = [10, 51, 60, 70]
        elif mutation == "ragged_row":
            cells[1]["column_index"] = 2
        else:
            for span in spans:
                if span["role"] in {"table", "table_row", "table_cell"}:
                    span.pop("table_key")
        return response

    with pytest.raises(OcrLayoutProviderError) as captured:
        _provider_with_worker(_FakeWorker(malformed_response)).analyze(
            document_id="invoice-invalid-table-topology",
            pages=[_image_page()],
        )

    assert captured.value.code == "ocr_worker_result_invalid"


def test_provider_rejects_logical_rows_in_reversed_physical_order() -> None:
    def malformed_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        spans = response["result"]["pages"][0]["spans"]
        table = next(span for span in spans if span["role"] == "table")
        row = next(span for span in spans if span["role"] == "table_row")
        cells = [span for span in spans if span["role"] == "table_cell"]
        table["bbox"] = [10, 10, 90, 80]
        row["bbox"] = [10, 40, 90, 60]
        for cell in cells:
            cell["bbox"][1] = 40
            cell["bbox"][3] = 60
        common = {"confidence": 0.95, "table_key": "table-0"}
        spans.extend(
            [
                {
                    **common,
                    "role": "table_row",
                    "text": "Bolt | 4",
                    "bbox": [10, 20, 90, 40],
                    "row_index": 1,
                },
                {
                    **common,
                    "role": "table_cell",
                    "text": "Bolt",
                    "bbox": [10, 20, 60, 40],
                    "row_index": 1,
                    "column_index": 0,
                },
                {
                    **common,
                    "role": "table_cell",
                    "text": "4",
                    "bbox": [60, 20, 90, 40],
                    "row_index": 1,
                    "column_index": 1,
                },
            ]
        )
        return response

    with pytest.raises(OcrLayoutProviderError) as captured:
        _provider_with_worker(_FakeWorker(malformed_response)).analyze(
            document_id="invoice-reversed-physical-rows",
            pages=[_image_page()],
        )

    assert captured.value.code == "ocr_worker_result_invalid"


def test_provider_rejects_rowspan_cell_that_does_not_cover_claimed_rows() -> None:
    def malformed_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        spans = response["result"]["pages"][0]["spans"]
        first_cell = next(
            span for span in spans if span["role"] == "table_cell" and span["column_index"] == 0
        )
        first_cell["row_span"] = 2
        common = {"confidence": 0.95, "table_key": "table-0"}
        spans.extend(
            [
                {
                    **common,
                    "role": "table_row",
                    "text": "4",
                    "bbox": [10, 50, 90, 80],
                    "row_index": 1,
                },
                {
                    **common,
                    "role": "table_cell",
                    "text": "4",
                    "bbox": [60, 50, 90, 80],
                    "row_index": 1,
                    "column_index": 1,
                },
            ]
        )
        return response

    with pytest.raises(OcrLayoutProviderError) as captured:
        _provider_with_worker(_FakeWorker(malformed_response)).analyze(
            document_id="invoice-truncated-rowspan",
            pages=[_image_page()],
        )

    assert captured.value.code == "ocr_worker_result_invalid"


@pytest.mark.parametrize(("field", "value"), [("row_span", 0), ("column_span", 101)])
def test_provider_rejects_out_of_bounds_cell_span(field: str, value: int) -> None:
    def malformed_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        first_cell = next(
            span for span in response["result"]["pages"][0]["spans"] if span["role"] == "table_cell"
        )
        first_cell[field] = value
        return response

    with pytest.raises(OcrLayoutProviderError) as captured:
        _provider_with_worker(_FakeWorker(malformed_response)).analyze(
            document_id="invoice-invalid-span",
            pages=[_image_page()],
        )

    assert captured.value.code == "ocr_worker_result_invalid"


def test_reading_order_policy_is_hash_bound_in_the_ledger_identity() -> None:
    ledger = _provider_with_worker(_FakeWorker()).analyze(
        document_id="invoice-policy-hash",
        pages=[_image_page()],
    )

    different_policy_id = stable_ocr_ledger_id(
        document_id=ledger.document_id,
        provider_id=ledger.provider_id,
        engine_id=ledger.engine_id,
        engine_version=ledger.engine_version,
        reading_order_policy_id="global-yx/v1",
        model_id=ledger.model_id,
        model_assets_lock_sha256=ledger.model_assets_lock_sha256,
        pages=ledger.pages,
    )

    assert different_policy_id != ledger.ledger_id


@pytest.mark.parametrize("policy", [None, "global-yx/v1", "worker-emitted-sequence/v2"])
def test_provider_rejects_missing_or_unexpected_worker_order_policy(
    policy: str | None,
) -> None:
    page = _image_page()

    def response_with_policy(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        result = response["result"]
        if policy is None:
            result.pop("reading_order_policy_id")
        else:
            result["reading_order_policy_id"] = policy
        return response

    with pytest.raises(OcrLayoutProviderError) as captured:
        _provider_with_worker(_FakeWorker(response_with_policy)).analyze(
            document_id="invoice-order-policy",
            pages=[page],
        )

    assert captured.value.code == "ocr_worker_result_invalid"


def test_worker_success_envelope_emits_the_exact_reading_order_policy() -> None:
    page = _image_page()

    class Backend:
        provider_id = "paddle"
        engine_id = "fixture-engine"
        engine_version = "1.0"
        model_id = "fixture-model"
        model_assets_lock_sha256 = "a" * 64

        def analyze_page(self, **_kwargs: object) -> list[dict[str, Any]]:
            return [
                {
                    "role": "text",
                    "text": "Invoice",
                    "bbox": [5, 5, 45, 15],
                    "confidence": 0.99,
                }
            ]

    response, returned_backend = _process_request(
        {
            "protocol": "customs-ocr-worker/v2",
            "request_id": "ocrreq_policy",
            "provider": "paddle",
            "document_id": "invoice-policy-worker",
            "pages": [
                {
                    "page_number": page.page_number,
                    "media_type": page.media_type,
                    "data_base64": page.data_base64,
                    "byte_length": page.byte_length,
                    "sha256": page.sha256,
                    "width_px": page.width_px,
                    "height_px": page.height_px,
                }
            ],
        },
        provider="paddle",
        backend=cast(Any, Backend()),
    )

    assert returned_backend is not None
    assert response["status"] == "ok"
    assert response["result"]["reading_order_policy_id"] == "worker-emitted-sequence/v1"


def test_provider_rejects_pre_reading_order_worker_protocol() -> None:
    page = _image_page()

    def legacy_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        response["protocol"] = "customs-ocr-worker/v1"
        return response

    worker = _FakeWorker(legacy_response)
    provider = _provider_with_worker(worker)

    with pytest.raises(OcrLayoutProviderError) as captured:
        provider.analyze(document_id="invoice-legacy-protocol", pages=[page])

    assert captured.value.code == "ocr_worker_response_malformed"
    assert worker.closed


def test_paddle_converter_preserves_provider_native_sequence() -> None:
    result = {
        "res": {
            "overall_ocr_res": {
                # This explicit provider sequence intentionally differs from
                # bbox y/x order. Geometry must not silently replace it.
                "rec_texts": ["Right 2", "Header", "Left 1", "Footer", "Right 1", "Left 2"],
                "rec_scores": [0.9] * 6,
                "rec_boxes": [
                    [65, 32, 100, 50],
                    [0, 0, 100, 21],
                    [0, 20, 35, 38],
                    [0, 70, 100, 80],
                    [65, 20, 100, 38],
                    [0, 32, 35, 50],
                ],
            }
        }
    }

    spans = _paddle_result_to_spans(
        result,
        page_number=1,
        width_px=100,
        height_px=100,
    )

    assert [span["text"] for span in spans] == [
        "Right 2",
        "Header",
        "Left 1",
        "Footer",
        "Right 1",
        "Left 2",
    ]


def test_docling_converter_preserves_provider_native_sequence() -> None:
    class BoundingBox:
        coord_origin = "TOPLEFT"

        def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
            self.l = left
            self.t = top
            self.r = right
            self.b = bottom

    class Provenance:
        confidence = 0.99

        def __init__(self, bbox: BoundingBox) -> None:
            self.bbox = bbox

    class TextItem:
        label = "text"

        def __init__(self, text: str, bbox: BoundingBox, *, heading: bool = False) -> None:
            self.text = text
            self.prov = [Provenance(bbox)]
            self.label = "section_heading" if heading else "text"

    items = [
        TextItem("Right 2", BoundingBox(65, 32, 100, 50)),
        TextItem("Left 1", BoundingBox(0, 20, 35, 38)),
        TextItem("Header", BoundingBox(0, 0, 100, 21), heading=True),
        TextItem("Right 1", BoundingBox(65, 20, 100, 38)),
        TextItem("Left 2", BoundingBox(0, 32, 35, 50)),
    ]

    class Document:
        def iterate_items(self) -> list[tuple[object, int]]:
            return [(item, 0) for item in items]

    spans = _docling_document_to_spans(
        Document(),
        page_number=1,
        width_px=100,
        height_px=100,
    )

    assert [span["text"] for span in spans] == [
        "Right 2",
        "Left 1",
        "Header",
        "Right 1",
        "Left 2",
    ]


def test_adapter_preserves_validated_worker_sequence_and_ledger_schema() -> None:
    page = _image_page()

    def shuffled_table_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        response["result"]["pages"][0]["spans"] = [
            {
                "role": "table_cell",
                "text": "4",
                "bbox": [60, 40, 90, 60],
                "confidence": 0.94,
                "table_key": "table-0",
                "row_index": 1,
                "column_index": 1,
            },
            {
                "role": "text",
                "text": "After table",
                # Provider boxes may overlap the table edge slightly.
                "bbox": [5, 59, 55, 75],
                "confidence": 0.98,
            },
            {
                "role": "table_row",
                "text": "Widget | 2",
                "bbox": [10, 20, 90, 40],
                "confidence": 0.95,
                "table_key": "table-0",
                "row_index": 0,
            },
            {
                "role": "table_cell",
                "text": "Widget",
                "bbox": [10, 20, 60, 40],
                "confidence": 0.96,
                "table_key": "table-0",
                "row_index": 0,
                "column_index": 0,
            },
            {
                "role": "table",
                "text": "Widget | 2 | Bolt | 4",
                "bbox": [10, 20, 90, 60],
                "confidence": 0.95,
                "table_key": "table-0",
            },
            {
                "role": "table_cell",
                "text": "2",
                "bbox": [60, 20, 90, 40],
                "confidence": 0.94,
                "table_key": "table-0",
                "row_index": 0,
                "column_index": 1,
            },
            {
                "role": "text",
                "text": "Before table",
                "bbox": [5, 5, 55, 21],
                "confidence": 0.99,
            },
            {
                "role": "table_cell",
                "text": "Bolt",
                "bbox": [10, 40, 60, 60],
                "confidence": 0.96,
                "table_key": "table-0",
                "row_index": 1,
                "column_index": 0,
            },
            {
                "role": "table_row",
                "text": "Bolt | 4",
                "bbox": [10, 40, 90, 60],
                "confidence": 0.95,
                "table_key": "table-0",
                "row_index": 1,
            },
        ]
        return response

    ledger = _provider_with_worker(_FakeWorker(shuffled_table_response)).analyze(
        document_id="invoice-table-order",
        pages=[page],
    )
    spans = ledger.pages[0].spans

    assert [(span.role, span.text) for span in spans] == [
        ("table_cell", "4"),
        ("text", "After table"),
        ("table_row", "Widget | 2"),
        ("table_cell", "Widget"),
        ("table", "Widget | 2 | Bolt | 4"),
        ("table_cell", "2"),
        ("text", "Before table"),
        ("table_cell", "Bolt"),
        ("table_row", "Bolt | 4"),
    ]
    assert [span.reading_order for span in spans] == list(range(len(spans)))
    assert len({span.table_id for span in spans if span.table_id is not None}) == 1
    assert OcrLayoutLedger.model_validate(ledger.model_dump(mode="json")) == ledger


def test_adapter_does_not_replace_worker_sequence_with_geometry() -> None:
    page = _image_page()

    def table_band_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        response["result"]["pages"][0]["spans"] = [
            {
                "role": "text",
                "text": "Right 2",
                "bbox": [65, 32, 100, 50],
                "confidence": 0.9,
            },
            {
                "role": "table_cell",
                "text": "Widget",
                "bbox": [0, 49, 50, 70],
                "confidence": 0.95,
                "table_key": "table-band",
                "row_index": 0,
                "column_index": 0,
            },
            {
                "role": "text",
                "text": "Left 1",
                "bbox": [0, 20, 35, 38],
                "confidence": 0.9,
            },
            {
                "role": "table",
                "text": "Widget | 2",
                "bbox": [0, 49, 100, 70],
                "confidence": 0.95,
                "table_key": "table-band",
            },
            {
                "role": "text",
                "text": "Right 1",
                "bbox": [65, 20, 100, 38],
                "confidence": 0.9,
            },
            {
                "role": "table_cell",
                "text": "2",
                "bbox": [50, 49, 100, 70],
                "confidence": 0.95,
                "table_key": "table-band",
                "row_index": 0,
                "column_index": 1,
            },
            {
                "role": "text",
                "text": "Left 2",
                "bbox": [0, 32, 35, 50],
                "confidence": 0.9,
            },
            {
                "role": "table_row",
                "text": "Widget | 2",
                "bbox": [0, 49, 100, 70],
                "confidence": 0.95,
                "table_key": "table-band",
                "row_index": 0,
            },
        ]
        return response

    ledger = _provider_with_worker(_FakeWorker(table_band_response)).analyze(
        document_id="invoice-table-band",
        pages=[page],
    )

    assert [(span.role, span.text) for span in ledger.pages[0].spans] == [
        ("text", "Right 2"),
        ("table_cell", "Widget"),
        ("text", "Left 1"),
        ("table", "Widget | 2"),
        ("text", "Right 1"),
        ("table_cell", "2"),
        ("text", "Left 2"),
        ("table_row", "Widget | 2"),
    ]


def test_adapter_rejects_span_count_before_normalization_or_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _image_page()

    def oversized_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        response["result"]["pages"][0]["spans"] = [
            {
                "role": "text",
                "text": f"span-{index}",
                "bbox": [1, 1 + index, 20, 2 + index],
                "confidence": 0.9,
            }
            for index in range(3)
        ]
        return response

    monkeypatch.setattr(ocr_module, "MAX_OCR_SPANS_PER_PAGE", 2)
    monkeypatch.setattr(
        ocr_module,
        "preserve_provider_span_order",
        lambda *_args, **_kwargs: pytest.fail("ordering must not run past the span cap"),
    )

    with pytest.raises(OcrLayoutProviderError) as captured:
        _provider_with_worker(_FakeWorker(oversized_response)).analyze(
            document_id="invoice-span-limit",
            pages=[page],
        )

    assert captured.value.code == "ocr_span_limit_exceeded"


def test_worker_rejects_span_count_before_reading_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "res": {
            "overall_ocr_res": {
                "rec_texts": ["one", "two", "three"],
                "rec_scores": [0.9, 0.9, 0.9],
                "rec_boxes": [[0, 0, 10, 10], [0, 20, 10, 30], [0, 40, 10, 50]],
            }
        }
    }
    monkeypatch.setattr(worker_module, "MAX_OCR_SPANS_PER_PAGE", 2)
    monkeypatch.setattr(
        worker_module,
        "preserve_provider_span_order",
        lambda *_args, **_kwargs: pytest.fail("ordering must not run past the span cap"),
    )

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _paddle_result_to_spans(
            result,
            page_number=1,
            width_px=100,
            height_px=100,
        )

    assert captured.value.code == "ocr_span_limit_exceeded"


def test_provider_order_accepts_exact_span_boundary_and_rejects_one_more() -> None:
    spans = [
        {
            "role": "text",
            "text": f"line-{index}",
            "bbox": [0, index * 2, 10, index * 2 + 1],
        }
        for index in range(MAX_OCR_SPANS_PER_PAGE)
    ]

    ordered = preserve_provider_span_order(spans)
    assert len(ordered) == MAX_OCR_SPANS_PER_PAGE
    assert ordered[0]["text"] == "line-0"
    assert ordered[-1]["text"] == f"line-{MAX_OCR_SPANS_PER_PAGE - 1}"

    with pytest.raises(OcrSpanLimitExceededError):
        preserve_provider_span_order(
            [*spans, {"role": "text", "text": "overflow", "bbox": [0, 0, 1, 1]}]
        )


def test_provider_rejects_raw_echo_and_never_leaks_it_in_error() -> None:
    page = _image_page()

    def echo_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        response["result"]["raw_bytes"] = request["pages"][0]["data_base64"]
        return response

    provider = _provider_with_worker(_FakeWorker(echo_response))
    with pytest.raises(OcrLayoutProviderError) as captured:
        provider.analyze(document_id="invoice-secret", pages=[page])
    assert captured.value.code == "ocr_worker_raw_content_rejected"
    assert page.data_base64 not in str(captured.value)
    assert "invoice-secret" not in str(captured.value)


@pytest.mark.parametrize("position", ["prefix", "middle", "suffix"])
def test_provider_rejects_substantial_raw_fragments(position: str) -> None:
    page = _image_page(side_px=500)
    starts = {
        "prefix": 0,
        "middle": (len(page.data_base64) - 500) // 2,
        "suffix": len(page.data_base64) - 500,
    }
    fragment = page.data_base64[starts[position] : starts[position] + 500]

    def fragment_response(request: dict[str, Any]) -> dict[str, Any]:
        response = _ok_response(request)
        response["result"]["pages"][0]["spans"][0]["text"] = f"worker-output:{fragment}:end"
        return response

    provider = _provider_with_worker(_FakeWorker(fragment_response))
    with pytest.raises(OcrLayoutProviderError) as captured:
        provider.analyze(document_id="invoice-secret", pages=[page])

    assert captured.value.code == "ocr_worker_raw_content_rejected"
    assert fragment not in str(captured.value)


def test_provider_fail_closed_for_malformed_and_backend_error_envelopes() -> None:
    page = _image_page()

    malformed_worker = _FakeWorker(lambda request: {"request_id": request["request_id"]})
    malformed = _provider_with_worker(malformed_worker)
    with pytest.raises(OcrLayoutProviderError) as malformed_error:
        malformed.analyze(document_id="doc-1", pages=[page])
    assert malformed_error.value.code == "ocr_worker_response_malformed"
    assert malformed_worker.closed

    def failed(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocol": "customs-ocr-worker/v2",
            "request_id": request["request_id"],
            "status": "error",
            "error": {
                "code": "ocr_backend_inference_failed",
                "retryable": True,
            },
        }

    backend_failure = _provider_with_worker(_FakeWorker(failed))
    with pytest.raises(OcrLayoutProviderError) as backend_error:
        backend_failure.analyze(document_id="doc-1", pages=[page])
    assert backend_error.value.code == "ocr_backend_inference_failed"
    assert backend_error.value.retryable is True


class _QueueStdout:
    def __init__(self) -> None:
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        self.closed = False

    def readline(self, _limit: int = -1) -> bytes:
        value = self.lines.get(timeout=1)
        return b"" if value is None else value

    def close(self) -> None:
        self.closed = True
        self.lines.put(None)


class _ReactiveStdin:
    def __init__(
        self,
        process: "_FakeProcess",
        response_factory: Callable[[bytes], bytes | None],
    ) -> None:
        self.process = process
        self.response_factory = response_factory
        self.closed = False

    def write(self, value: bytes) -> int:
        response = self.response_factory(value)
        if response is not None:
            self.process.stdout.lines.put(response)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, response_factory: Callable[[bytes], bytes | None]) -> None:
        self.pid = 4321
        self.stdout = _QueueStdout()
        self.stdin = _ReactiveStdin(self, response_factory)
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 1

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_jsonl_client_reuses_one_persistent_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(value: bytes) -> bytes:
        request = json.loads(value)
        return (
            json.dumps(
                {
                    "protocol": "customs-ocr-worker/v2",
                    "request_id": request["request_id"],
                    "status": "ok",
                    "result": {},
                }
            ).encode("utf-8")
            + b"\n"
        )

    process = _FakeProcess(respond)
    starts = 0

    def popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        nonlocal starts
        starts += 1
        return process

    monkeypatch.setattr(ocr_module.subprocess, "Popen", popen)
    client = ocr_module._JsonlWorkerClient(
        provider="docling",
        python_executable=Path(sys.executable),
        timeout_seconds=0.2,
    )

    first = client.request({"request_id": "ocrreq_1"})
    second = client.request({"request_id": "ocrreq_2"})
    assert client.runtime_process_id() == process.pid
    client.close()

    assert first["request_id"] == "ocrreq_1"
    assert second["request_id"] == "ocrreq_2"
    assert starts == 1
    assert process.terminated
    assert client.runtime_process_id() is None


def test_jsonl_client_rejects_malformed_output_and_cleans_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(lambda _request: b"not-json\n")
    monkeypatch.setattr(
        ocr_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    client = ocr_module._JsonlWorkerClient(
        provider="docling",
        python_executable=Path(sys.executable),
        timeout_seconds=0.2,
    )
    with pytest.raises(ocr_module._WorkerProtocolError) as captured:
        client.request({"request_id": "ocrreq_1"})
    assert captured.value.code == "ocr_worker_response_malformed"
    assert process.terminated
    assert process.stdin.closed
    assert process.stdout.closed


def test_jsonl_client_times_out_and_cleans_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(lambda _request: None)
    monkeypatch.setattr(
        ocr_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    client = ocr_module._JsonlWorkerClient(
        provider="docling",
        python_executable=Path(sys.executable),
        timeout_seconds=0.01,
    )
    with pytest.raises(ocr_module._WorkerProtocolError) as captured:
        client.request({"request_id": "ocrreq_1"})
    assert captured.value.code == "ocr_worker_timeout"
    assert captured.value.retryable
    assert process.terminated
    assert process.stdin.closed
    assert process.stdout.closed


def test_jsonl_client_rejects_oversized_output_and_cleans_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocr_module, "_MAX_WORKER_RESPONSE_BYTES", 32)
    process = _FakeProcess(lambda _request: (b"x" * 33) + b"\n")
    monkeypatch.setattr(
        ocr_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    client = ocr_module._JsonlWorkerClient(
        provider="docling",
        python_executable=Path(sys.executable),
        timeout_seconds=0.2,
    )
    with pytest.raises(ocr_module._WorkerProtocolError) as captured:
        client.request({"request_id": "ocrreq_1"})
    assert captured.value.code == "ocr_worker_response_too_large"
    assert process.terminated
    assert process.stdin.closed
    assert process.stdout.closed


def test_worker_environment_is_allowlisted_and_disables_paddle_mkldnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-boundary")
    monkeypatch.setenv("LAW_API_OC", "must-not-cross-boundary")
    monkeypatch.setenv("DATABASE_URL", "must-not-cross-boundary")
    monkeypatch.setenv("PYTHONPATH", "must-not-control-imports")
    monkeypatch.setenv("SAFE_SETTING", "not-explicitly-allowed")
    monkeypatch.setenv("PATH", "C:\\SAFE_BIN")
    monkeypatch.setattr(ocr_module, "_paddle_cache_root", lambda: "C:\\OCR_CACHE")

    environment = ocr_module._worker_environment("paddle")

    assert "ANTHROPIC_API_KEY" not in environment
    assert "LAW_API_OC" not in environment
    assert "DATABASE_URL" not in environment
    assert "PYTHONPATH" not in environment
    assert "SAFE_SETTING" not in environment
    assert environment["PATH"] == "C:\\SAFE_BIN"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["FLAGS_use_mkldnn"] == "0"
    assert environment["FLAGS_enable_pir_api"] == "0"
    assert environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"
    assert environment["PADDLE_HOME"] == "C:\\OCR_CACHE"
    assert environment["PADDLEX_HOME"] == "C:\\OCR_CACHE"
    assert environment["PADDLE_PDX_CACHE_HOME"] == "C:\\OCR_CACHE"


def test_docling_worker_environment_forces_verified_cache_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "C:\\VERIFIED_HF_CACHE")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "false")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-boundary")

    environment = ocr_module._worker_environment("docling")

    assert environment["HF_HOME"] == "C:\\VERIFIED_HF_CACHE"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert "ANTHROPIC_API_KEY" not in environment


def test_jsonl_worker_uses_isolated_imports_neutral_cwd_and_suppressed_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _FakeProcess(lambda _request: None)
    captured: dict[str, Any] = {}

    def popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        return process

    monkeypatch.setattr(ocr_module.subprocess, "Popen", popen)
    monkeypatch.setattr(ocr_module, "_worker_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(
        ocr_module,
        "_worker_environment",
        lambda _provider: {"PATH": "C:\\SAFE_BIN"},
    )
    client = ocr_module._JsonlWorkerClient(
        provider="docling",
        python_executable=Path(sys.executable),
        timeout_seconds=0.01,
    )

    with pytest.raises(ocr_module._WorkerProtocolError):
        client.request({"request_id": "ocrreq_isolated"})

    assert captured["command"][:3] == [
        str(Path(sys.executable)),
        "-I",
        "-m",
    ]
    assert captured["command"][3:] == [
        "customs_ai_agent.ocr_worker",
        "--provider",
        "docling",
    ]
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"] == {"PATH": "C:\\SAFE_BIN"}
    assert captured["stderr"] is ocr_module.subprocess.DEVNULL


def test_paddle_backend_uses_pinned_accuracy_first_cpu_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    paddleocr = types.ModuleType("paddleocr")

    class FakePPStructureV3:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    paddleocr.PPStructureV3 = FakePPStructureV3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
    expected_versions = {
        "paddlepaddle": "3.3.1",
        "paddleocr": "3.7.0",
        "paddlex": "3.7.2",
    }
    monkeypatch.setattr(
        "customs_ai_agent.ocr_worker.importlib.metadata.version",
        lambda package: expected_versions[package],
    )

    _PaddleBackend()

    assert captured["layout_detection_model_name"] == "PP-DocLayout-S"
    assert captured["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert captured["text_det_unclip_ratio"] == 0.75
    assert captured["text_recognition_model_name"] == "en_PP-OCRv5_mobile_rec"
    assert captured["use_table_recognition"] is True
    assert captured["use_formula_recognition"] is False
    assert captured["use_chart_recognition"] is False
    assert captured["use_seal_recognition"] is False
    assert captured["enable_mkldnn"] is False
    assert captured["device"] == "cpu"
    assert os.environ["FLAGS_use_mkldnn"] == "0"
    assert os.environ["FLAGS_enable_pir_api"] == "0"
    assert os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"


def test_paddle_converter_maps_text_and_table_cells() -> None:
    result = {
        "res": {
            "overall_ocr_res": {
                "rec_texts": ["Invoice"],
                "rec_scores": [0.98],
                "rec_boxes": [[1, 1, 40, 10]],
            },
            "table_res_list": [
                {
                    "cell_box_list": [
                        [5, 20, 45, 40],
                        [45, 20, 80, 40],
                    ],
                    "table_ocr_pred": {
                        "rec_boxes": [
                            [5, 20, 45, 40],
                            [45, 20, 80, 40],
                        ],
                        "rec_texts": ["Item", "Qty"],
                        "rec_scores": [0.97, 0.96],
                    },
                    "pred_html": (
                        "<html><body><table><tr>"
                        "<td>Item</td><td>Qty</td>"
                        "</tr></table></body></html>"
                    ),
                }
            ],
        }
    }

    spans = _paddle_result_to_spans(
        result,
        page_number=1,
        width_px=100,
        height_px=100,
    )

    assert any(span["role"] == "text" and span["text"] == "Invoice" for span in spans)
    assert sum(span["role"] == "table" for span in spans) == 1
    assert sum(span["role"] == "table_row" for span in spans) == 1
    cells = [span for span in spans if span["role"] == "table_cell"]
    assert [(cell["row_index"], cell["column_index"]) for cell in cells] == [
        (0, 0),
        (0, 1),
    ]


def _paddle_merged_table() -> dict[str, Any]:
    return {
        "cell_box_list": [
            [0, 0, 20, 40],
            [20, 0, 100, 20],
            [20, 20, 60, 40],
            [60, 20, 100, 40],
        ],
        "table_ocr_pred": {
            "rec_boxes": [
                [1, 1, 19, 39],
                [21, 1, 99, 19],
                [21, 21, 59, 39],
                [61, 21, 99, 39],
            ],
            "rec_texts": ["Group", "Header", "Left", "Right"],
            "rec_scores": [0.99, 0.98, 0.97, 0.96],
        },
        "pred_html": (
            "<html><body><table>"
            '<tr><td rowspan="2">Group</td><td colspan="2">Header</td></tr>'
            "<tr><td>Left</td><td>Right</td></tr>"
            "</table></body></html>"
        ),
    }


def test_paddle_converter_validates_explicit_row_and_column_spans() -> None:
    spans = _paddle_result_to_spans(
        _paddle_full_result(table=_paddle_merged_table()),
        page_number=1,
        width_px=100,
        height_px=100,
    )

    cells = [span for span in spans if span["role"] == "table_cell"]
    assert [
        (
            cell["text"],
            cell["row_index"],
            cell["column_index"],
            cell["row_span"],
            cell["column_span"],
            cell["bbox"],
        )
        for cell in cells
    ] == [
        ("Group", 0, 0, 2, 1, [0.0, 0.0, 20.0, 40.0]),
        ("Header", 0, 1, 1, 2, [20.0, 0.0, 100.0, 20.0]),
        ("Left", 1, 1, 1, 1, [20.0, 20.0, 60.0, 40.0]),
        ("Right", 1, 2, 1, 1, [60.0, 20.0, 100.0, 40.0]),
    ]


@pytest.mark.parametrize(
    ("right_edge", "accepted"),
    [
        (61.0, False),
        (88.0, False),
        (88.04, True),
    ],
)
def test_paddle_colspan_requires_strict_axis_coverage(
    right_edge: float,
    accepted: bool,
) -> None:
    table = _paddle_merged_table()
    table["cell_box_list"][1] = [20, 0, right_edge, 20]
    table["table_ocr_pred"]["rec_boxes"][1] = [21, 1, right_edge - 1, 19]

    def convert() -> list[dict[str, Any]]:
        return _paddle_result_to_spans(
            _paddle_full_result(table=table),
            page_number=1,
            width_px=100,
            height_px=100,
        )

    if accepted:
        assert any(span["role"] == "table_cell" for span in convert())
    else:
        with pytest.raises(worker_module._SafeWorkerError) as captured:
            convert()
        assert captured.value.code == "ocr_table_structure_invalid"


def test_paddle_rowspan_rejects_one_pixel_covered_row_overlap() -> None:
    table = _paddle_merged_table()
    table["cell_box_list"][0] = [0, 0, 20, 21]
    table["table_ocr_pred"]["rec_boxes"][0] = [1, 1, 19, 20]

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _paddle_result_to_spans(
            _paddle_full_result(table=table),
            page_number=1,
            width_px=100,
            height_px=100,
        )

    assert captured.value.code == "ocr_table_structure_invalid"


@pytest.mark.parametrize(
    "invalid_confidence",
    [None, True, "0.99", -0.1, 1.1, float("nan"), float("inf")],
)
def test_missing_or_invalid_provider_confidence_is_fail_low(invalid_confidence: object) -> None:
    assert _confidence(invalid_confidence) == 0.0


@pytest.mark.parametrize("confidence", [0, 0.25, 1])
def test_valid_provider_confidence_is_preserved(confidence: float) -> None:
    assert _confidence(confidence) == float(confidence)


def test_paddle_converter_does_not_invent_confidence_when_scores_are_missing() -> None:
    spans = _paddle_result_to_spans(
        {
            "res": {
                "overall_ocr_res": {
                    "rec_texts": ["Invoice"],
                    "rec_scores": [],
                    "rec_boxes": [[1, 1, 40, 10]],
                }
            }
        },
        page_number=1,
        width_px=100,
        height_px=100,
    )

    assert spans == [
        {
            "role": "text",
            "text": "Invoice",
            "bbox": [1.0, 1.0, 40.0, 10.0],
            "confidence": 0.0,
        }
    ]


def test_paddle_text_bbox_uses_provider_coordinates_without_raster_resegmentation() -> None:
    spans = _paddle_result_to_spans(
        {
            "res": {
                "overall_ocr_res": {
                    "rec_texts": ["Invoice"],
                    "rec_scores": [0.99],
                    "rec_boxes": [[10, 5, 50, 30]],
                }
            }
        },
        page_number=1,
        width_px=100,
        height_px=50,
    )

    assert spans[0]["bbox"] == [10.0, 5.0, 50.0, 30.0]


@pytest.mark.parametrize(
    "invalid_bbox",
    [
        [float("-inf"), 1, float("inf"), 10],
        [1, 1, float("nan"), 10],
        [-1, 1, 40, 10],
        [1, 1, 101, 10],
    ],
)
def test_paddle_converter_rejects_invalid_text_geometry(
    invalid_bbox: list[float],
) -> None:
    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _paddle_result_to_spans(
            {
                "res": {
                    "overall_ocr_res": {
                        "rec_texts": ["Invoice"],
                        "rec_scores": [0.99],
                        "rec_boxes": [invalid_bbox],
                    }
                }
            },
            page_number=1,
            width_px=100,
            height_px=50,
        )

    assert captured.value.code == "ocr_text_geometry_invalid"


def test_docling_converter_maps_text_provenance_and_table_cells() -> None:
    class BoundingBox:
        def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
            self.l = left
            self.t = top
            self.r = right
            self.b = bottom
            self.coord_origin = "TOPLEFT"

    class Provenance:
        def __init__(self, bbox: BoundingBox) -> None:
            self.bbox = bbox
            self.confidence = 0.99

    class TextItem:
        text = "Packing List"
        # This is Docling's observed runtime label, not a synthetic alias.
        label = "section_header"
        prov = [Provenance(BoundingBox(1, 1, 50, 12))]

    class Cell:
        def __init__(self, text: str, column: int, bbox: BoundingBox) -> None:
            self.text = text
            self.start_row_offset_idx = 0
            self.end_row_offset_idx = 1
            self.start_col_offset_idx = column
            self.end_col_offset_idx = column + 1
            self.row_span = 1
            self.col_span = 1
            self.bbox = bbox

    class TableData:
        table_cells = [
            Cell("Item", 0, BoundingBox(5, 20, 45, 40)),
            Cell("Qty", 1, BoundingBox(45, 20, 80, 40)),
        ]

    class TableItem:
        data = TableData()
        prov = [Provenance(BoundingBox(5, 20, 80, 40))]

    class Document:
        def iterate_items(self) -> list[tuple[object, int]]:
            return [(TextItem(), 0), (TableItem(), 0)]

    spans = _docling_document_to_spans(
        Document(),
        page_number=1,
        width_px=100,
        height_px=100,
    )

    assert any(span["role"] == "heading" and span["text"] == "Packing List" for span in spans)
    assert sum(span["role"] == "table_cell" for span in spans) == 2


def test_docling_converter_rejects_cell_without_cell_level_bbox() -> None:
    class Provenance:
        bbox = {
            "left": 5,
            "top": 20,
            "right": 80,
            "bottom": 40,
            "coord_origin": "TOPLEFT",
        }

    class Cell:
        text = "Item"
        start_row_offset_idx = 0
        end_row_offset_idx = 1
        start_col_offset_idx = 0
        end_col_offset_idx = 1
        row_span = 1
        col_span = 1

    class TableData:
        table_cells = [Cell()]

    class TableItem:
        data = TableData()
        prov = [Provenance()]

    class Document:
        items = [TableItem()]

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _docling_document_to_spans(
            Document(),
            page_number=1,
            width_px=100,
            height_px=100,
        )

    assert captured.value.code == "ocr_table_structure_invalid"


@pytest.mark.parametrize("cells_source", [None, []])
def test_docling_converter_rejects_observed_table_without_cells(
    cells_source: list[object] | None,
) -> None:
    class TextItem:
        text = "Packing List"
        label = "section_header"
        bbox = [1, 1, 50, 12]

    class TableItem:
        data = (
            types.SimpleNamespace()
            if cells_source is None
            else types.SimpleNamespace(table_cells=cells_source)
        )

    class Document:
        items = [TextItem(), TableItem()]

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _docling_document_to_spans(
            Document(),
            page_number=1,
            width_px=100,
            height_px=100,
        )

    assert captured.value.code == "ocr_table_structure_invalid"


@pytest.mark.parametrize(
    "invalid_bbox",
    [
        [float("-inf"), 1, float("inf"), 10],
        [1, 1, float("nan"), 10],
        [-1, 1, 40, 10],
        [1, 1, 101, 10],
    ],
)
def test_docling_converter_rejects_invalid_text_geometry(
    invalid_bbox: list[float],
) -> None:
    class TextItem:
        text = "Invoice"
        bbox = invalid_bbox

    class Document:
        items = [TextItem()]

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _docling_document_to_spans(
            Document(),
            page_number=1,
            width_px=100,
            height_px=50,
        )

    assert captured.value.code == "ocr_text_geometry_invalid"


_EMPTY_PADDLE_PREDICTION = object()


class _FakePaddlePipeline:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.inputs: list[Any] = []
        self.ocr_inputs: list[Any] = []
        self.paddlex_pipeline = self
        self.general_ocr_options: list[dict[str, object]] = []
        self.table_crop_options: list[dict[str, object]] = []

    def predict(self, *, input: Any) -> object:
        return self._respond(input)

    def table_recognition_pipeline(self, input: Any, **kwargs: object) -> object:
        self.table_crop_options.append(kwargs)
        return self._respond(input)

    def general_ocr_pipeline(self, input: Any, **kwargs: object) -> object:
        self.ocr_inputs.append(input.copy())
        self.general_ocr_options.append(kwargs)
        return [{"provider_ocr": "observed"}]

    def _respond(self, input: Any) -> object:
        self.inputs.append(input.copy())
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if response is _EMPTY_PADDLE_PREDICTION:
            return []
        return [response]


class _FakeRgbArray:
    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)
        self.size = width * height * 3

    def __getitem__(self, key: tuple[slice, slice]) -> "_FakeRgbArray":
        y_slice, x_slice = key
        return _FakeRgbArray(
            cast(int, x_slice.stop) - cast(int, x_slice.start),
            cast(int, y_slice.stop) - cast(int, y_slice.start),
        )

    def copy(self) -> "_FakeRgbArray":
        return _FakeRgbArray(self.shape[1], self.shape[0])

    def tobytes(self, *, order: str) -> bytes:
        assert order == "C"
        return b"\xff" * self.size


class _FakeNumpy(types.ModuleType):
    uint8 = object()

    def asarray(self, image: Image.Image, *, dtype: object) -> _FakeRgbArray:
        assert dtype is self.uint8
        return _FakeRgbArray(*image.size)

    def ascontiguousarray(self, image: _FakeRgbArray) -> _FakeRgbArray:
        return image


def _paddle_full_result(
    *,
    image_boxes: list[list[int]] | None = None,
    table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root: dict[str, Any] = {
        "overall_ocr_res": {
            "rec_texts": ["Invoice"],
            "rec_scores": [0.99],
            "rec_boxes": [[1, 1, 9, 9]],
        }
    }
    if image_boxes is not None:
        root["parsing_res_list"] = [
            {"block_label": "image", "block_bbox": box} for box in image_boxes
        ]
    if table is not None:
        root["table_res_list"] = [table]
    return {"res": root}


def _paddle_two_by_four_table() -> dict[str, Any]:
    boxes = [
        [0, 0, 20, 20],
        [20, 0, 40, 20],
        [40, 0, 60, 20],
        [60, 0, 80, 20],
        [0, 20, 20, 40],
        [20, 20, 40, 40],
        [40, 20, 60, 40],
        [60, 20, 80, 40],
    ]
    texts = [f"cell-{index}" for index in range(8)]
    return {
        "cell_box_list": boxes,
        "table_ocr_pred": {
            "rec_boxes": boxes,
            "rec_texts": texts,
            "rec_scores": [0.9] * 8,
        },
        "pred_html": (
            "<html><body><table>"
            "<tr>"
            + "".join(f"<td>{text}</td>" for text in texts[:4])
            + "</tr><tr>"
            + "".join(f"<td>{text}</td>" for text in texts[4:])
            + "</tr></table></body></html>"
        ),
    }


def _ruled_grid_fixture() -> tuple[Image.Image, dict[str, Any]]:
    image = Image.new("RGB", (200, 160), "white")
    drawing = ImageDraw.Draw(image)
    metadata_rows = [
        [(5, 5, 95, 45), (105, 5, 195, 45)],
        [(5, 50, 95, 90), (105, 50, 195, 90)],
    ]
    item_rows = [
        [(5, 100, 70, 125), (70, 100, 130, 125), (130, 100, 195, 125)],
        [(5, 125, 70, 155), (70, 125, 130, 155), (130, 125, 195, 155)],
    ]
    cells = [cell for row in [*metadata_rows, *item_rows] for cell in row]
    for cell in metadata_rows[0] + metadata_rows[1]:
        drawing.rectangle(cell, outline="black", width=1)
    for y in (100, 125, 155):
        drawing.line((5, y, 195, y), fill="black", width=1)
    for x in (5, 70, 130, 195):
        drawing.line((x, 100, x, 155), fill="black", width=1)

    token_boxes = [[left + 5, top + 5, right - 5, bottom - 5] for left, top, right, bottom in cells]
    root = {
        "overall_ocr_res": {
            "rec_boxes": token_boxes,
            "rec_texts": [f"cell-{index}" for index in range(len(cells))],
            "rec_scores": [0.99] * len(cells),
        }
    }
    return image, root


def test_paddle_ruled_grid_keeps_connected_table_and_excludes_disconnected_form_fields() -> None:
    image, root = _ruled_grid_fixture()

    spans = worker_module._paddle_ruled_grid_table_spans(
        image=image,
        ocr_root=root,
        table_key="p1-fixture",
        crop_bbox=(0, 0, 200, 160),
        page_width_px=200,
        page_height_px=160,
    )

    assert [span["role"] for span in spans].count("table") == 1
    assert [span["role"] for span in spans].count("table_row") == 2
    cells = [span for span in spans if span["role"] == "table_cell"]
    assert len(cells) == 6
    table_keys = {cast(str, cell["table_key"]) for cell in cells}
    assert len(table_keys) == 1
    assert [(cell["row_index"], cell["column_index"]) for cell in cells] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert [cell["text"] for cell in cells] == [f"cell-{index}" for index in range(4, 10)]


def test_paddle_ruled_grid_ignores_unruled_image() -> None:
    spans = worker_module._paddle_ruled_grid_table_spans(
        image=Image.new("RGB", (200, 160), "white"),
        ocr_root={"overall_ocr_res": {}},
        table_key="p1-plain",
        crop_bbox=(0, 0, 200, 160),
        page_width_px=200,
        page_height_px=160,
    )

    assert spans == []


def test_paddle_ruled_grid_rejects_connected_two_by_one_form() -> None:
    sections = worker_module._ruled_grid_sections(
        [
            [[0.0, 0.0, 40.0, 20.0]],
            [[0.0, 20.0, 40.0, 40.0]],
        ],
        width=100,
    )

    assert sections == []


def test_paddle_ruled_grid_separates_side_by_side_connected_tables() -> None:
    sections = worker_module._ruled_grid_sections(
        [
            [
                [0.0, 0.0, 40.0, 20.0],
                [40.0, 0.0, 80.0, 20.0],
                [100.0, 0.0, 140.0, 20.0],
                [140.0, 0.0, 180.0, 20.0],
            ],
            [
                [0.0, 20.0, 40.0, 40.0],
                [40.0, 20.0, 80.0, 40.0],
                [100.0, 20.0, 140.0, 40.0],
                [140.0, 20.0, 180.0, 40.0],
            ],
        ],
        width=200,
    )

    assert len(sections) == 2
    assert all([len(row) for row in section] == [2, 2] for section in sections)


def test_paddle_ruled_grid_separates_vertically_disconnected_tables() -> None:
    sections = worker_module._ruled_grid_sections(
        [
            [
                [0.0, 0.0, 40.0, 20.0],
                [40.0, 0.0, 80.0, 20.0],
            ],
            [
                [0.0, 20.0, 40.0, 40.0],
                [40.0, 20.0, 80.0, 40.0],
            ],
            [
                [0.0, 60.0, 40.0, 80.0],
                [40.0, 60.0, 80.0, 80.0],
            ],
            [
                [0.0, 80.0, 40.0, 100.0],
                [40.0, 80.0, 80.0, 100.0],
            ],
        ],
        width=100,
    )

    assert len(sections) == 2
    assert [section[0][0][1] for section in sections] == [0.0, 60.0]


def test_paddle_ruled_grid_ignores_wider_adjacent_form_rows() -> None:
    sections = worker_module._ruled_grid_sections(
        [
            [[0.0, 0.0, 100.0, 20.0]],
            [
                [5.0, 20.0, 35.0, 40.0],
                [35.0, 20.0, 65.0, 40.0],
                [65.0, 20.0, 95.0, 40.0],
            ],
            [
                [5.0, 40.0, 35.0, 60.0],
                [35.0, 40.0, 65.0, 60.0],
                [65.0, 40.0, 95.0, 60.0],
            ],
            [[0.0, 60.0, 100.0, 80.0]],
        ],
        width=100,
    )

    assert len(sections) == 1
    assert [len(row) for row in sections[0]] == [3, 3]
    assert worker_module._union_bbox([cell for row in sections[0] for cell in row]) == [
        5.0,
        20.0,
        95.0,
        60.0,
    ]


@pytest.mark.parametrize("shape", ["page_frame", "underlines", "checkboxes", "broken_corner"])
def test_paddle_ruled_grid_negative_shapes_do_not_create_tables(shape: str) -> None:
    image = Image.new("RGB", (200, 160), "white")
    drawing = ImageDraw.Draw(image)
    if shape == "page_frame":
        drawing.rectangle((5, 5, 195, 155), outline="black", width=1)
    elif shape == "underlines":
        for y in (30, 60, 90, 120):
            drawing.line((20, y, 180, y), fill="black", width=1)
    elif shape == "checkboxes":
        for x, y in ((20, 20), (80, 20), (20, 80), (80, 80)):
            drawing.rectangle((x, y, x + 20, y + 20), outline="black", width=1)
    else:
        for y in (20, 70, 120):
            drawing.line((20, y, 180, y), fill="black", width=1)
        for x in (20, 100, 180):
            drawing.line((x, 20, x, 120), fill="black", width=1)
        drawing.rectangle((96, 66, 104, 74), fill="white")

    spans = worker_module._paddle_ruled_grid_table_spans(
        image=image,
        ocr_root={"overall_ocr_res": {}},
        table_key=f"p1-{shape}",
        crop_bbox=(0, 0, 200, 160),
        page_width_px=200,
        page_height_px=160,
    )

    assert spans == []


@pytest.mark.parametrize("mutation", ["empty_cell", "unassigned_grid_token"])
def test_paddle_ruled_grid_fails_closed_on_incomplete_grounding(mutation: str) -> None:
    image, root = _ruled_grid_fixture()
    if mutation == "empty_cell":
        for key in ("rec_boxes", "rec_texts", "rec_scores"):
            root["overall_ocr_res"][key] = root["overall_ocr_res"][key][:-1]
    else:
        root["overall_ocr_res"]["rec_boxes"].append([68, 105, 72, 115])
        root["overall_ocr_res"]["rec_texts"].append("gap-token")
        root["overall_ocr_res"]["rec_scores"].append(0.99)

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        worker_module._paddle_ruled_grid_table_spans(
            image=image,
            ocr_root=root,
            table_key="p1-malformed",
            crop_bbox=(0, 0, 200, 160),
            page_width_px=200,
            page_height_px=160,
        )

    assert captured.value.code == "ocr_table_fallback_structure_invalid"


def _paddle_backend_with_pipeline(
    *responses: object,
) -> tuple[_PaddleBackend, _FakePaddlePipeline]:
    backend = object.__new__(_PaddleBackend)
    pipeline = _FakePaddlePipeline(*responses)
    backend._pipeline = pipeline
    return backend, pipeline


def _run_fake_paddle(
    backend: _PaddleBackend,
    *,
    image: Image.Image | None = None,
) -> list[dict[str, Any]]:
    output = io.BytesIO()
    source = image or Image.new("RGB", (100, 100), "white")
    source.save(output, format="PNG")
    previous_numpy = sys.modules.get("numpy")
    sys.modules["numpy"] = _FakeNumpy("numpy")
    try:
        return backend.analyze_page(
            raw=output.getvalue(),
            page_number=1,
            media_type="image/png",
            width_px=source.width,
            height_px=source.height,
        )
    finally:
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy


def test_paddle_bounded_fallback_prefers_grounded_ruled_grid() -> None:
    image, root = _ruled_grid_fixture()
    full = {
        "res": {
            **root,
            "parsing_res_list": [
                {
                    "block_label": "image",
                    "block_bbox": [3, 3, 197, 157],
                }
            ],
        }
    }
    backend, pipeline = _paddle_backend_with_pipeline(full)

    spans = _run_fake_paddle(backend, image=image)

    assert len(pipeline.inputs) == 1
    assert pipeline.ocr_inputs == []
    assert [span["role"] for span in spans].count("table") == 1
    assert [span["role"] for span in spans].count("table_cell") == 6


def test_paddle_page_ruled_grid_recovers_table_beyond_truncated_layout_candidate() -> None:
    image, root = _ruled_grid_fixture()
    full = {
        "res": {
            **root,
            "parsing_res_list": [
                {
                    "block_label": "image",
                    # The layout observation overlaps most of the real table but
                    # cuts through its final row. Page pixels must provide the
                    # complete geometry; this bbox only corroborates it.
                    "block_bbox": [3, 3, 197, 135],
                }
            ],
        }
    }
    backend, pipeline = _paddle_backend_with_pipeline(full)

    spans = _run_fake_paddle(backend, image=image)

    assert len(pipeline.inputs) == 1
    assert [span["role"] for span in spans].count("table") == 1
    assert [span["role"] for span in spans].count("table_row") == 2
    assert [span["role"] for span in spans].count("table_cell") == 6
    assert next(span["bbox"] for span in spans if span["role"] == "table") == [
        5.0,
        100.0,
        195.0,
        155.0,
    ]


def test_paddle_page_ruled_grid_recovers_without_layout_candidate() -> None:
    image, root = _ruled_grid_fixture()
    backend, pipeline = _paddle_backend_with_pipeline({"res": root})

    spans = _run_fake_paddle(backend, image=image)

    assert len(pipeline.inputs) == 1
    assert [span["role"] for span in spans].count("table") == 1
    assert [span["role"] for span in spans].count("table_row") == 2
    assert [span["role"] for span in spans].count("table_cell") == 6


def test_paddle_full_page_table_does_not_enter_image_region_fallback() -> None:
    table = _paddle_two_by_four_table()
    backend, pipeline = _paddle_backend_with_pipeline(
        _paddle_full_result(image_boxes=[[10, 20, 90, 80]], table=table)
    )

    spans = _run_fake_paddle(backend)

    assert len(pipeline.inputs) == 1
    assert [span["role"] for span in spans].count("table") == 1
    assert _PaddleBackend.engine_id.endswith("layout-table-v4")


def test_paddle_image_region_fallback_remaps_two_by_four_table_deterministically() -> None:
    full = _paddle_full_result(image_boxes=[[10, 20, 90, 80]])
    crop = _paddle_full_result(table=_paddle_two_by_four_table())
    first_backend, first_pipeline = _paddle_backend_with_pipeline(full, crop)
    second_backend, _ = _paddle_backend_with_pipeline(full, crop)

    first = _run_fake_paddle(first_backend)
    second = _run_fake_paddle(second_backend)

    assert len(first_pipeline.inputs) == 2
    assert first_pipeline.inputs[1].shape[:2] == (60, 80)
    assert len(first_pipeline.ocr_inputs) == 1
    assert first_pipeline.ocr_inputs[0].shape[:2] == (60, 80)
    assert first_pipeline.general_ocr_options == [{}]
    assert first_pipeline.table_crop_options == [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_layout_detection": False,
            "use_ocr_model": False,
            "overall_ocr_res": {"provider_ocr": "observed"},
            "cell_sort_by_y_projection": True,
            "use_wired_table_cells_trans_to_html": False,
            "use_wireless_table_cells_trans_to_html": False,
            "use_table_orientation_classify": False,
            "use_ocr_results_with_table_cells": True,
            "use_e2e_wired_table_rec_model": False,
            "use_e2e_wireless_table_rec_model": True,
        }
    ]
    assert [span["role"] for span in first].count("table") == 1
    assert [span["role"] for span in first].count("table_row") == 2
    assert [span["role"] for span in first].count("table_cell") == 8
    first_table_key = next(span["table_key"] for span in first if span["role"] == "table")
    second_table_key = next(span["table_key"] for span in second if span["role"] == "table")
    assert first_table_key == second_table_key
    assert next(span["bbox"] for span in first if span["role"] == "table") == [
        10.0,
        20.0,
        90.0,
        60.0,
    ]
    assert next(span["bbox"] for span in first if span["role"] == "table_cell") == [
        10.0,
        20.0,
        30.0,
        40.0,
    ]


def test_paddle_fallback_table_key_binds_crop_pixels_bbox_and_revision() -> None:
    key = worker_module._paddle_fallback_table_key(
        page_number=1,
        crop_bbox=(10, 20, 90, 80),
        crop_sha256="a" * 64,
    )

    assert key == worker_module._paddle_fallback_table_key(
        page_number=1,
        crop_bbox=(10, 20, 90, 80),
        crop_sha256="a" * 64,
    )
    assert key != worker_module._paddle_fallback_table_key(
        page_number=1,
        crop_bbox=(11, 20, 90, 80),
        crop_sha256="a" * 64,
    )
    assert key != worker_module._paddle_fallback_table_key(
        page_number=1,
        crop_bbox=(10, 20, 90, 80),
        crop_sha256="b" * 64,
    )
    assert (
        worker_module._PADDLE_TABLE_FALLBACK_ALGORITHM_REVISION == "image-region-table-recovery/v3"
    )


def test_paddle_no_image_candidate_preserves_full_page_observations() -> None:
    backend, pipeline = _paddle_backend_with_pipeline(_paddle_full_result())

    spans = _run_fake_paddle(backend)

    assert len(pipeline.inputs) == 1
    assert [span["role"] for span in spans] == ["text"]


def test_paddle_non_table_image_region_does_not_create_table_spans() -> None:
    backend, pipeline = _paddle_backend_with_pipeline(
        _paddle_full_result(image_boxes=[[10, 20, 90, 80]]),
        _paddle_full_result(),
    )

    spans = _run_fake_paddle(backend)

    assert len(pipeline.inputs) == 2
    assert [span["text"] for span in spans] == ["Invoice"]
    assert not any(span["role"].startswith("table") for span in spans)


@pytest.mark.parametrize(
    ("boxes", "error_code"),
    [
        ([[10, 20, 10, 80]], "ocr_table_fallback_candidate_invalid"),
        (
            [
                [1, 1, 11, 11],
                [20, 1, 30, 11],
                [40, 1, 50, 11],
                [60, 1, 70, 11],
                [80, 1, 90, 11],
            ],
            "ocr_table_fallback_candidate_limit_exceeded",
        ),
        (
            [[10, 20, 90, 80], [10, 20, 90, 80]],
            "ocr_table_fallback_candidate_duplicate",
        ),
    ],
)
def test_paddle_invalid_excess_or_duplicate_candidates_fail_closed(
    boxes: list[list[int]],
    error_code: str,
) -> None:
    backend, _ = _paddle_backend_with_pipeline(_paddle_full_result(image_boxes=boxes))

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _run_fake_paddle(backend)

    assert captured.value.code == error_code


def test_paddle_multiple_successful_image_regions_fail_closed() -> None:
    first_table = _paddle_two_by_four_table()
    second_table = _paddle_two_by_four_table()
    for table in (first_table, second_table):
        scaled_boxes = [
            [value[0] / 2, value[1], value[2] / 2, value[3]] for value in table["cell_box_list"]
        ]
        table["cell_box_list"] = scaled_boxes
        table["table_ocr_pred"]["rec_boxes"] = scaled_boxes
    backend, pipeline = _paddle_backend_with_pipeline(
        _paddle_full_result(image_boxes=[[1, 1, 41, 61], [50, 1, 99, 61]]),
        _paddle_full_result(table=first_table),
        _paddle_full_result(table=second_table),
    )

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _run_fake_paddle(backend)

    assert captured.value.code == "ocr_table_fallback_ambiguous"
    assert len(pipeline.inputs) == 3


def test_paddle_image_region_recall_error_fails_closed_without_raw_content() -> None:
    backend, _ = _paddle_backend_with_pipeline(
        _paddle_full_result(image_boxes=[[10, 20, 90, 80]]),
        RuntimeError("secret crop content"),
    )

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _run_fake_paddle(backend)

    assert captured.value.code == "ocr_table_fallback_inference_failed"
    assert captured.value.retryable is True
    assert "secret crop content" not in str(captured.value)


def test_paddle_empty_crop_result_fails_closed() -> None:
    backend, _ = _paddle_backend_with_pipeline(
        _paddle_full_result(image_boxes=[[10, 20, 90, 80]]),
        _EMPTY_PADDLE_PREDICTION,
    )

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _run_fake_paddle(backend)

    assert captured.value.code == "ocr_table_fallback_result_empty"


def test_paddle_malformed_crop_table_structure_fails_closed() -> None:
    malformed = _paddle_two_by_four_table()
    malformed["table_ocr_pred"]["rec_texts"] = malformed["table_ocr_pred"]["rec_texts"][:-1]
    backend, _ = _paddle_backend_with_pipeline(
        _paddle_full_result(image_boxes=[[10, 20, 90, 80]]),
        _paddle_full_result(table=malformed),
    )

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _run_fake_paddle(backend)

    assert captured.value.code == "ocr_table_fallback_structure_invalid"


def test_paddle_nested_tokens_are_grounded_by_bbox_not_position() -> None:
    table = {
        "cell_box_list": [[0, 0, 50, 50], [50, 0, 100, 50]],
        "table_ocr_pred": {
            # Provider token order intentionally starts with the right cell.
            "rec_boxes": [[55, 5, 95, 20], [5, 5, 20, 20], [5, 25, 40, 40]],
            "rec_texts": ["Right", "Alpha", "Beta"],
            "rec_scores": [0.91, 0.96, 0.93],
        },
        "pred_html": (
            "<html><body><table><tr><td>Alpha Beta</td><td>Right</td></tr></table></body></html>"
        ),
    }

    spans = _paddle_result_to_spans(
        _paddle_full_result(table=table),
        page_number=1,
        width_px=100,
        height_px=50,
    )

    cells = [span for span in spans if span["role"] == "table_cell"]
    assert [(cell["text"], cell["bbox"]) for cell in cells] == [
        ("Alpha Beta", [0.0, 0.0, 50.0, 50.0]),
        ("Right", [50.0, 0.0, 100.0, 50.0]),
    ]
    assert cells[0]["confidence"] == 0.93


def test_paddle_equal_token_and_cell_counts_still_use_spatial_grounding() -> None:
    table = {
        "cell_box_list": [[0, 0, 50, 50], [50, 0, 100, 50]],
        "table_ocr_pred": {
            "rec_boxes": [[55, 5, 95, 20], [5, 5, 45, 20]],
            "rec_texts": ["Right", "Left"],
            "rec_scores": [0.91, 0.96],
        },
        "pred_html": (
            "<html><body><table><tr><td>Left</td><td>Right</td></tr></table></body></html>"
        ),
    }

    spans = _paddle_result_to_spans(
        _paddle_full_result(table=table),
        page_number=1,
        width_px=100,
        height_px=50,
    )

    assert [span["text"] for span in spans if span["role"] == "table_cell"] == ["Left", "Right"]


def test_paddle_parsing_table_label_is_not_structured_table_evidence() -> None:
    full = _paddle_full_result(image_boxes=[[10, 20, 90, 80]])
    full["res"]["parsing_res_list"].append(
        {
            "block_label": "table",
            "block_bbox": [10, 20, 90, 80],
            "block_content": "<table><tr><td>presentation-only</td></tr></table>",
        }
    )
    backend, pipeline = _paddle_backend_with_pipeline(
        full,
        _paddle_full_result(table=_paddle_two_by_four_table()),
    )

    spans = _run_fake_paddle(backend)

    assert len(pipeline.inputs) == 2
    assert [span["role"] for span in spans].count("table") == 1
    assert [span["role"] for span in spans].count("table_cell") == 8


def test_paddle_malformed_full_page_table_fails_closed_without_partial_spans() -> None:
    malformed = _paddle_two_by_four_table()
    malformed["table_ocr_pred"]["rec_boxes"] = malformed["table_ocr_pred"]["rec_boxes"][:-1]
    backend, pipeline = _paddle_backend_with_pipeline(
        _paddle_full_result(image_boxes=[[10, 20, 90, 80]], table=malformed)
    )

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _run_fake_paddle(backend)

    assert captured.value.code == "ocr_table_structure_invalid"
    assert len(pipeline.inputs) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "unassigned_token",
        "empty_cell",
        "html_text_mismatch",
        "inconsistent_merged_cell",
        "invalid_span",
        "span_beyond_rows",
        "duplicate_span_attribute",
        "overlapping_cells",
    ],
)
def test_paddle_ambiguous_or_inconsistent_table_grounding_fails_closed(
    mutation: str,
) -> None:
    table = _paddle_two_by_four_table()
    if mutation == "unassigned_token":
        table["table_ocr_pred"]["rec_boxes"][0] = [19, 0, 21, 20]
    elif mutation == "empty_cell":
        for key in ("rec_boxes", "rec_texts", "rec_scores"):
            table["table_ocr_pred"][key] = table["table_ocr_pred"][key][:-1]
    elif mutation == "html_text_mismatch":
        table["pred_html"] = table["pred_html"].replace("cell-0", "different", 1)
    elif mutation == "inconsistent_merged_cell":
        table["pred_html"] = table["pred_html"].replace("<td>cell-0", '<td colspan="2">cell-0', 1)
    elif mutation == "invalid_span":
        table["pred_html"] = table["pred_html"].replace("<td>cell-0", '<td colspan="0">cell-0', 1)
    elif mutation == "span_beyond_rows":
        table["pred_html"] = table["pred_html"].replace("<td>cell-0", '<td rowspan="3">cell-0', 1)
    elif mutation == "duplicate_span_attribute":
        table["pred_html"] = table["pred_html"].replace(
            "<td>cell-0",
            '<td colspan="1" colspan="1">cell-0',
            1,
        )
    else:
        table["cell_box_list"][1] = [19, 0, 40, 20]

    with pytest.raises(worker_module._SafeWorkerError) as captured:
        _paddle_result_to_spans(
            _paddle_full_result(table=table),
            page_number=1,
            width_px=100,
            height_px=100,
        )

    assert captured.value.code == "ocr_table_structure_invalid"
