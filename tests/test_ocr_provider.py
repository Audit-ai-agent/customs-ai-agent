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
from PIL import Image

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


def _image_page() -> ImagePageInput:
    output = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(output, format="PNG")
    raw = output.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    return ImagePageInput(
        page_number=1,
        media_type="image/png",
        data_base64=base64.b64encode(raw).decode("ascii"),
        byte_length=len(raw),
        sha256=digest,
        width_px=100,
        height_px=100,
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
    table_ids = {span.table_id for span in first.pages[0].spans if span.table_id is not None}
    assert len(table_ids) == 1
    assert len({span.span_id for span in first.pages[0].spans}) == 5
    assert len(worker.requests) == 2
    assert worker.requests[0]["pages"][0]["data_base64"] == page.data_base64
    assert page.data_base64 not in first.model_dump_json()
    assert worker.closed


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
                    "rec_texts": ["Item", "Qty"],
                    "rec_scores": [0.97, 0.96],
                    "row_indices": [0, 0],
                    "column_indices": [0, 1],
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
        label = "section_heading"
        prov = [Provenance(BoundingBox(1, 1, 50, 12))]

    class Cell:
        def __init__(self, text: str, column: int, bbox: BoundingBox) -> None:
            self.text = text
            self.start_row_offset_idx = 0
            self.start_col_offset_idx = column
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
