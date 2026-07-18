"""Isolated, persistent OCR/layout worker.

Protocol frames are the only data written to stdout. Provider diagnostics,
including output emitted by native libraries, are redirected to stderr before
provider imports occur. Error frames intentionally contain codes, not exception
messages or document content.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from io import BytesIO, TextIOWrapper
from statistics import fmean
from typing import Any, Literal, Protocol, cast

from PIL import Image, UnidentifiedImageError

from customs_ai_agent.ocr_identity import (
    OCR_MODEL_ASSETS_LOCK_SHA256,
    OCR_MODEL_SET_IDS,
)
from customs_ai_agent.ocr_reading_order import (
    MAX_OCR_SPANS_PER_PAGE,
    OCR_READING_ORDER_POLICY_ID,
    OcrSpanLimitExceededError,
    preserve_provider_span_order,
)

_PROTOCOL = "customs-ocr-worker/v2"
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MEDIA_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_PADDLE_PACKAGE_VERSIONS = {
    "paddlepaddle": "3.3.1",
    "paddleocr": "3.7.0",
    "paddlex": "3.7.2",
}

ProviderName = Literal["paddle", "docling"]


class _SafeWorkerError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class _Backend(Protocol):
    provider_id: str
    engine_id: str
    engine_version: str
    model_id: str
    model_assets_lock_sha256: str

    def analyze_page(
        self,
        *,
        raw: bytes,
        page_number: int,
        media_type: str,
        width_px: int,
        height_px: int,
    ) -> list[dict[str, Any]]: ...


class _PaddleBackend:
    provider_id = "paddle"
    engine_id = "PP-StructureV3:customs-paddle-layout-table-v1"
    model_id = OCR_MODEL_SET_IDS["paddle"]
    model_assets_lock_sha256 = OCR_MODEL_ASSETS_LOCK_SHA256

    def __init__(self) -> None:
        # These must be set before importing Paddle.
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        # Release model assets are pre-provisioned and verified from the
        # code-owned lock.  Do not make worker availability depend on a model
        # host connectivity probe or allow an implicit download at startup.
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        _require_package_versions(_PADDLE_PACKAGE_VERSIONS)
        try:
            from paddleocr import PPStructureV3
        except Exception as exc:
            raise _SafeWorkerError("ocr_backend_unavailable") from exc
        try:
            self._pipeline = PPStructureV3(
                layout_detection_model_name="PP-DocLayout-S",
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                table_classification_model_name="PP-LCNet_x1_0_table_cls",
                wired_table_structure_recognition_model_name="SLANeXt_wired",
                wireless_table_structure_recognition_model_name="SLANet_plus",
                wired_table_cells_detection_model_name=("RT-DETR-L_wired_table_cell_det"),
                wireless_table_cells_detection_model_name=("RT-DETR-L_wireless_table_cell_det"),
                table_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_table_recognition=True,
                use_formula_recognition=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                use_region_detection=False,
                enable_mkldnn=False,
                device="cpu",
            )
        except Exception as exc:
            raise _SafeWorkerError(
                "ocr_backend_initialization_failed",
                retryable=True,
            ) from exc
        self.engine_version = _version_string(("paddleocr", "paddlepaddle", "paddlex"))

    def analyze_page(
        self,
        *,
        raw: bytes,
        page_number: int,
        media_type: str,
        width_px: int,
        height_px: int,
    ) -> list[dict[str, Any]]:
        del media_type
        try:
            import numpy as np

            with Image.open(BytesIO(raw)) as source:
                image = np.asarray(source.convert("RGB"))
            prediction = self._pipeline.predict(input=image)
            results = (
                prediction
                if isinstance(prediction, list)
                else list(cast(Iterable[Any], prediction))
            )
        except Exception as exc:
            raise _SafeWorkerError(
                "ocr_backend_inference_failed",
                retryable=True,
            ) from exc
        if not results:
            return []
        return _paddle_result_to_spans(
            results[0],
            page_number=page_number,
            width_px=width_px,
            height_px=height_px,
        )


class _DoclingBackend:
    provider_id = "docling"
    engine_id = "DocumentConverter:default-image-pipeline"
    model_id = OCR_MODEL_SET_IDS["docling"]
    model_assets_lock_sha256 = OCR_MODEL_ASSETS_LOCK_SHA256

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:
            raise _SafeWorkerError("ocr_backend_unavailable") from exc
        try:
            self._converter = DocumentConverter()
        except Exception as exc:
            raise _SafeWorkerError(
                "ocr_backend_initialization_failed",
                retryable=True,
            ) from exc
        self.engine_version = _version_string(("docling",))

    def analyze_page(
        self,
        *,
        raw: bytes,
        page_number: int,
        media_type: str,
        width_px: int,
        height_px: int,
    ) -> list[dict[str, Any]]:
        try:
            from docling.datamodel.base_models import DocumentStream

            suffix = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }[media_type]
            stream = DocumentStream(
                name=f"page-{page_number}{suffix}",
                stream=BytesIO(raw),
            )
            conversion = self._converter.convert(stream)
            document = conversion.document
        except Exception as exc:
            raise _SafeWorkerError(
                "ocr_backend_inference_failed",
                retryable=True,
            ) from exc
        return _docling_document_to_spans(
            document,
            page_number=page_number,
            width_px=width_px,
            height_px=height_px,
        )


def _paddle_result_to_spans(
    result: object,
    *,
    page_number: int,
    width_px: int,
    height_px: int,
) -> list[dict[str, Any]]:
    root = _object_mapping(result)
    wrapped = _lookup(root, "res")
    if isinstance(_object_mapping(wrapped), Mapping):
        root = _object_mapping(wrapped)
    overall = _lookup(root, "overall_ocr_res")
    if overall is None:
        overall = result
    texts = _sequence(_lookup(overall, "rec_texts", "texts"))
    scores = _sequence(_lookup(overall, "rec_scores", "scores"))
    boxes = _sequence(_lookup(overall, "rec_boxes", "rec_polys", "boxes"))

    spans: list[dict[str, Any]] = []
    for index, text_value in enumerate(texts):
        text = str(text_value).strip() if text_value is not None else ""
        box = _bbox_from_value(
            boxes[index] if index < len(boxes) else None,
            width_px=width_px,
            height_px=height_px,
        )
        if not text or box is None:
            continue
        score = _confidence(scores[index] if index < len(scores) else None)
        spans.append(
            {
                "role": "text",
                "text": text,
                "bbox": box,
                "confidence": score,
            }
        )

    for table_index, table in enumerate(_paddle_tables(root)):
        spans.extend(
            _paddle_table_to_spans(
                table,
                table_key=f"p{page_number}-table-{table_index}",
                width_px=width_px,
                height_px=height_px,
            )
        )
    return _provider_span_order(spans)


def _paddle_tables(root: object) -> list[object]:
    direct = _lookup(root, "table_res_list", "tables")
    if direct is not None:
        return _sequence(direct)
    single = _lookup(root, "table_res")
    if single is not None:
        return _sequence(single) or [single]
    parsing = _sequence(_lookup(root, "parsing_res_list"))
    tables: list[object] = []
    for item in parsing:
        label = str(_lookup(item, "label", "block_label", "type") or "").lower()
        if "table" in label:
            tables.append(item)
    return tables


def _paddle_table_to_spans(
    table: object,
    *,
    table_key: str,
    width_px: int,
    height_px: int,
) -> list[dict[str, Any]]:
    nested = _lookup(table, "table_ocr_pred", "ocr_result")
    text_source = nested if nested is not None else table
    boxes = _sequence(
        _lookup(
            table,
            "cell_box_list",
            "cell_boxes",
            "cell_bboxes",
            "rec_boxes",
        )
    )
    if not boxes and nested is not None:
        boxes = _sequence(_lookup(nested, "rec_boxes", "rec_polys", "boxes"))
    texts = _sequence(_lookup(text_source, "rec_texts", "cell_texts", "texts", "cell_text"))
    scores = _sequence(_lookup(text_source, "rec_scores", "cell_scores", "scores"))
    row_values = _sequence(_lookup(table, "row_indices", "row_indexes", "cell_row_indices"))
    column_values = _sequence(
        _lookup(table, "column_indices", "column_indexes", "cell_column_indices")
    )

    cells: list[dict[str, Any]] = []
    for index, box_value in enumerate(boxes):
        box = _bbox_from_value(
            box_value,
            width_px=width_px,
            height_px=height_px,
        )
        text = str(texts[index]).strip() if index < len(texts) and texts[index] is not None else ""
        if box is None or not text:
            continue
        cells.append(
            {
                "text": text,
                "bbox": box,
                "confidence": _confidence(scores[index] if index < len(scores) else None),
                "provided_row": _non_negative_int(
                    row_values[index] if index < len(row_values) else None
                ),
                "provided_column": _non_negative_int(
                    column_values[index] if index < len(column_values) else None
                ),
            }
        )
    if not cells:
        return []

    inferred = _infer_cell_coordinates([cell["bbox"] for cell in cells])
    for index, cell in enumerate(cells):
        inferred_row, inferred_column = inferred[index]
        cell["row_index"] = (
            cell["provided_row"] if cell["provided_row"] is not None else inferred_row
        )
        cell["column_index"] = (
            cell["provided_column"] if cell["provided_column"] is not None else inferred_column
        )

    table_bbox = _union_bbox([cell["bbox"] for cell in cells])
    table_confidence = fmean(cell["confidence"] for cell in cells)
    spans: list[dict[str, Any]] = [
        {
            "role": "table",
            "text": " | ".join(cell["text"] for cell in cells),
            "bbox": table_bbox,
            "confidence": table_confidence,
            "table_key": table_key,
        }
    ]
    rows = sorted({cast(int, cell["row_index"]) for cell in cells})
    for row_index in rows:
        row_cells = sorted(
            (cell for cell in cells if cell["row_index"] == row_index),
            key=lambda cell: (cell["column_index"], cell["bbox"][0]),
        )
        spans.append(
            {
                "role": "table_row",
                "text": " | ".join(cell["text"] for cell in row_cells),
                "bbox": _union_bbox([cell["bbox"] for cell in row_cells]),
                "confidence": fmean(cell["confidence"] for cell in row_cells),
                "table_key": table_key,
                "row_index": row_index,
            }
        )
    for cell in cells:
        spans.append(
            {
                "role": "table_cell",
                "text": cell["text"],
                "bbox": cell["bbox"],
                "confidence": cell["confidence"],
                "table_key": table_key,
                "row_index": cell["row_index"],
                "column_index": cell["column_index"],
            }
        )
    return spans


def _docling_document_to_spans(
    document: object,
    *,
    page_number: int,
    width_px: int,
    height_px: int,
) -> list[dict[str, Any]]:
    items = _docling_items(document)
    spans: list[dict[str, Any]] = []
    table_number = 0
    for item in items:
        class_name = item.__class__.__name__.lower()
        if "table" in class_name or _lookup(item, "table_cells") is not None:
            spans.extend(
                _docling_table_to_spans(
                    item,
                    table_key=f"p{page_number}-table-{table_number}",
                    width_px=width_px,
                    height_px=height_px,
                )
            )
            table_number += 1
            continue
        text = str(_lookup(item, "text") or "").strip()
        box = _docling_item_bbox(
            item,
            width_px=width_px,
            height_px=height_px,
        )
        if not text or box is None:
            continue
        label = str(_lookup(item, "label") or "").lower()
        role = "heading" if "heading" in label or "title" in label else "text"
        spans.append(
            {
                "role": role,
                "text": text,
                "bbox": box,
                "confidence": _docling_confidence(item),
            }
        )
    return _provider_span_order(spans)


def _docling_items(document: object) -> list[object]:
    iterate = getattr(document, "iterate_items", None)
    if callable(iterate):
        values: list[object] = []
        for value in iterate():
            if isinstance(value, tuple) and value:
                values.append(value[0])
            else:
                values.append(value)
        return values
    values = []
    for attribute in ("texts", "pictures", "tables", "items"):
        values.extend(_sequence(_lookup(document, attribute)))
    return values


def _docling_table_to_spans(
    table: object,
    *,
    table_key: str,
    width_px: int,
    height_px: int,
) -> list[dict[str, Any]]:
    data = _lookup(table, "data")
    cells_source = _lookup(data, "table_cells", "cells")
    if cells_source is None:
        cells_source = _lookup(table, "table_cells", "cells")
    cells: list[dict[str, Any]] = []
    for cell in _sequence(cells_source):
        text = str(_lookup(cell, "text") or "").strip()
        box = _docling_item_bbox(
            cell,
            width_px=width_px,
            height_px=height_px,
        )
        if box is None:
            box = _docling_item_bbox(
                table,
                width_px=width_px,
                height_px=height_px,
            )
        if not text or box is None:
            continue
        cells.append(
            {
                "text": text,
                "bbox": box,
                "confidence": _docling_confidence(cell),
                "row_index": _non_negative_int(
                    _lookup(
                        cell,
                        "start_row_offset_idx",
                        "row_index",
                        "row_span_offset",
                    )
                ),
                "column_index": _non_negative_int(
                    _lookup(
                        cell,
                        "start_col_offset_idx",
                        "column_index",
                        "col_span_offset",
                    )
                ),
            }
        )
    if not cells:
        return []
    inferred = _infer_cell_coordinates([cell["bbox"] for cell in cells])
    for index, cell in enumerate(cells):
        if cell["row_index"] is None:
            cell["row_index"] = inferred[index][0]
        if cell["column_index"] is None:
            cell["column_index"] = inferred[index][1]
    table_box = _docling_item_bbox(
        table,
        width_px=width_px,
        height_px=height_px,
    ) or _union_bbox([cell["bbox"] for cell in cells])
    spans: list[dict[str, Any]] = [
        {
            "role": "table",
            "text": " | ".join(cell["text"] for cell in cells),
            "bbox": table_box,
            "confidence": fmean(cell["confidence"] for cell in cells),
            "table_key": table_key,
        }
    ]
    for row_index in sorted({cast(int, cell["row_index"]) for cell in cells}):
        row_cells = sorted(
            (cell for cell in cells if cell["row_index"] == row_index),
            key=lambda cell: (cell["column_index"], cell["bbox"][0]),
        )
        spans.append(
            {
                "role": "table_row",
                "text": " | ".join(cell["text"] for cell in row_cells),
                "bbox": _union_bbox([cell["bbox"] for cell in row_cells]),
                "confidence": fmean(cell["confidence"] for cell in row_cells),
                "table_key": table_key,
                "row_index": row_index,
            }
        )
    for cell in cells:
        spans.append(
            {
                "role": "table_cell",
                "text": cell["text"],
                "bbox": cell["bbox"],
                "confidence": cell["confidence"],
                "table_key": table_key,
                "row_index": cell["row_index"],
                "column_index": cell["column_index"],
            }
        )
    return spans


def _docling_item_bbox(
    item: object,
    *,
    width_px: int,
    height_px: int,
) -> list[float] | None:
    provenance = _sequence(_lookup(item, "prov", "provenance"))
    candidate = _lookup(item, "bbox")
    if candidate is None and provenance:
        candidate = _lookup(provenance[0], "bbox")
    if candidate is None:
        return None
    to_top_left = getattr(candidate, "to_top_left_origin", None)
    if callable(to_top_left):
        try:
            candidate = to_top_left(page_height=height_px)
        except TypeError:
            try:
                candidate = to_top_left(height_px)
            except Exception:
                return None
        except Exception:
            return None
    left = _lookup(candidate, "l", "left", "x1")
    top = _lookup(candidate, "t", "top", "y1")
    right = _lookup(candidate, "r", "right", "x2")
    bottom = _lookup(candidate, "b", "bottom", "y2")
    if all(value is not None for value in (left, top, right, bottom)):
        origin = str(_lookup(candidate, "coord_origin", "origin") or "").lower()
        if "bottom" in origin:
            converted = [
                float(cast(float, left)),
                height_px - float(cast(float, top)),
                float(cast(float, right)),
                height_px - float(cast(float, bottom)),
            ]
            return _bbox_from_value(
                converted,
                width_px=width_px,
                height_px=height_px,
            )
    return _bbox_from_value(
        candidate,
        width_px=width_px,
        height_px=height_px,
    )


def _docling_confidence(item: object) -> float:
    confidence = _lookup(item, "confidence", "score")
    provenance = _sequence(_lookup(item, "prov", "provenance"))
    if confidence is None and provenance:
        confidence = _lookup(provenance[0], "confidence", "score")
    return _confidence(confidence)


def _object_mapping(value: object) -> object:
    for attribute in ("json", "to_dict", "model_dump"):
        candidate = getattr(value, attribute, None)
        if candidate is None:
            continue
        try:
            converted = candidate() if callable(candidate) else candidate
        except Exception:  # noqa: S112 - incompatible optional SDK representation
            continue
        if isinstance(converted, str):
            try:
                converted = json.loads(converted)
            except json.JSONDecodeError:
                continue
        if isinstance(converted, Mapping):
            return converted
    return value


def _lookup(value: object, *names: str) -> object | None:
    value = _object_mapping(value)
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            try:
                return getattr(value, name)
            except Exception:
                return None
    return None


def _sequence(value: object | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return list(value)
    try:
        shape = getattr(value, "shape")
        if shape is not None:
            to_list = getattr(value, "tolist")
            return cast(list[Any], to_list())
    except Exception:  # noqa: S110 - optional array protocols may raise arbitrary errors
        pass
    return []


def _bbox_from_value(
    value: object,
    *,
    width_px: int,
    height_px: int,
) -> list[float] | None:
    if value is None:
        return None
    left = _lookup(value, "l", "left", "x1")
    top = _lookup(value, "t", "top", "y1")
    right = _lookup(value, "r", "right", "x2")
    bottom = _lookup(value, "b", "bottom", "y2")
    if all(item is not None for item in (left, top, right, bottom)):
        coordinates: list[object] = [left, top, right, bottom]
    else:
        coordinates = _flatten_coordinates(value)
        if not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in coordinates
        ):
            return None
        numeric_coordinates = [float(cast(int | float, item)) for item in coordinates]
        if len(coordinates) == 4:
            coordinates = [cast(object, item) for item in numeric_coordinates]
        elif len(coordinates) >= 8 and len(coordinates) % 2 == 0:
            xs = numeric_coordinates[0::2]
            ys = numeric_coordinates[1::2]
            coordinates = [min(xs), min(ys), max(xs), max(ys)]
        else:
            return None
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in coordinates
    ):
        return None
    x1, y1, x2, y2 = (float(cast(int | float, item)) for item in coordinates)
    x1 = max(0.0, min(float(width_px), x1))
    x2 = max(0.0, min(float(width_px), x2))
    y1 = max(0.0, min(float(height_px), y1))
    y2 = max(0.0, min(float(height_px), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _flatten_coordinates(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        flattened: list[object] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flattened.extend(_flatten_coordinates(item))
            else:
                flattened.append(item)
        return flattened
    try:
        to_list = getattr(value, "tolist")
        converted = to_list()
    except Exception:
        return []
    return _flatten_coordinates(converted)


def _confidence(value: object | None) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= float(value) <= 1:
        return float(value)
    # Missing or malformed provider confidence is unknown, never perfect.
    # Preserve the span for review/evaluation while forcing downstream policy
    # to treat the observation as untrusted.
    return 0.0


def _non_negative_int(value: object | None) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _infer_cell_coordinates(
    boxes: Sequence[Sequence[float]],
) -> list[tuple[int, int]]:
    indices = sorted(
        range(len(boxes)),
        key=lambda index: (
            (boxes[index][1] + boxes[index][3]) / 2,
            boxes[index][0],
        ),
    )
    rows: list[list[int]] = []
    for index in indices:
        center_y = (boxes[index][1] + boxes[index][3]) / 2
        height = boxes[index][3] - boxes[index][1]
        if not rows:
            rows.append([index])
            continue
        current = rows[-1]
        row_center = fmean((boxes[item][1] + boxes[item][3]) / 2 for item in current)
        row_height = fmean(boxes[item][3] - boxes[item][1] for item in current)
        if abs(center_y - row_center) <= max(height, row_height) * 0.5:
            current.append(index)
        else:
            rows.append([index])
    coordinates: list[tuple[int, int]] = [(0, 0)] * len(boxes)
    for row_index, row in enumerate(rows):
        for column_index, item in enumerate(sorted(row, key=lambda index: boxes[index][0])):
            coordinates[item] = (row_index, column_index)
    return coordinates


def _union_bbox(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _provider_span_order(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(spans) > MAX_OCR_SPANS_PER_PAGE:
        raise _SafeWorkerError("ocr_span_limit_exceeded")
    try:
        return preserve_provider_span_order(spans)
    except OcrSpanLimitExceededError as exc:
        raise _SafeWorkerError("ocr_span_limit_exceeded") from exc


def _version_string(packages: Sequence[str]) -> str:
    versions: list[str] = []
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        versions.append(f"{package}={version}")
    return ";".join(versions)[:128]


def _require_package_versions(expected: Mapping[str, str]) -> None:
    for package, required_version in expected.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise _SafeWorkerError("ocr_backend_unavailable") from exc
        if actual_version != required_version:
            raise _SafeWorkerError("ocr_backend_version_mismatch")


def _validate_page(page: object) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(page, dict):
        raise _SafeWorkerError("ocr_request_invalid")
    required = {
        "page_number",
        "media_type",
        "data_base64",
        "byte_length",
        "sha256",
        "width_px",
        "height_px",
    }
    if set(page) != required:
        raise _SafeWorkerError("ocr_request_invalid")
    media_type = page.get("media_type")
    encoded = page.get("data_base64")
    expected_length = page.get("byte_length")
    expected_hash = page.get("sha256")
    width = page.get("width_px")
    height = page.get("height_px")
    page_number = page.get("page_number")
    if (
        media_type not in _MEDIA_FORMATS
        or not isinstance(encoded, str)
        or not isinstance(expected_length, int)
        or isinstance(expected_length, bool)
        or expected_length < 1
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height < 1
        or not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or page_number < 1
    ):
        raise _SafeWorkerError("ocr_request_invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _SafeWorkerError("ocr_request_invalid") from exc
    if len(raw) != expected_length or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise _SafeWorkerError("ocr_request_image_mismatch")
    try:
        with Image.open(BytesIO(raw)) as image:
            detected_format = str(image.format or "").upper()
            detected_size = image.size
        with Image.open(BytesIO(raw)) as verified:
            verified.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise _SafeWorkerError("ocr_request_image_invalid") from exc
    if detected_format != _MEDIA_FORMATS[cast(str, media_type)] or detected_size != (width, height):
        raise _SafeWorkerError("ocr_request_image_mismatch")
    metadata = {
        "page_number": page_number,
        "media_type": media_type,
        "width_px": width,
        "height_px": height,
    }
    return raw, metadata


def _process_request(
    request: object,
    *,
    provider: ProviderName,
    backend: _Backend | None,
) -> tuple[dict[str, Any], _Backend | None]:
    request_id: str | None = None
    if isinstance(request, dict) and isinstance(request.get("request_id"), str):
        request_id = request["request_id"]
    envelope: dict[str, Any] = {
        "protocol": _PROTOCOL,
        "request_id": request_id,
    }
    try:
        if (
            not isinstance(request, dict)
            or request.get("protocol") != _PROTOCOL
            or request.get("provider") != provider
            or not isinstance(request_id, str)
            or not request_id.startswith("ocrreq_")
            or not isinstance(request.get("document_id"), str)
            or not request["document_id"]
            or not isinstance(request.get("pages"), list)
            or not request["pages"]
            or set(request) != {"protocol", "request_id", "provider", "document_id", "pages"}
        ):
            raise _SafeWorkerError("ocr_request_invalid")
        validated = [_validate_page(page) for page in request["pages"]]
        if backend is None:
            backend = _build_backend(provider)
        result_pages: list[dict[str, Any]] = []
        for raw, metadata in validated:
            spans = backend.analyze_page(raw=raw, **metadata)
            if len(spans) > MAX_OCR_SPANS_PER_PAGE:
                raise _SafeWorkerError("ocr_span_limit_exceeded")
            result_pages.append(
                {
                    "page_number": metadata["page_number"],
                    "spans": spans,
                }
            )
        envelope.update(
            {
                "status": "ok",
                "result": {
                    "provider_id": backend.provider_id,
                    "engine_id": backend.engine_id,
                    "engine_version": backend.engine_version,
                    "reading_order_policy_id": OCR_READING_ORDER_POLICY_ID,
                    "model_id": backend.model_id,
                    "model_assets_lock_sha256": backend.model_assets_lock_sha256,
                    "pages": result_pages,
                },
            }
        )
    except _SafeWorkerError as exc:
        envelope.update(
            {
                "status": "error",
                "error": {
                    "code": exc.code,
                    "retryable": exc.retryable,
                },
            }
        )
    except Exception:
        envelope.update(
            {
                "status": "error",
                "error": {
                    "code": "ocr_worker_internal_error",
                    "retryable": False,
                },
            }
        )
    return envelope, backend


def _build_backend(provider: ProviderName) -> _Backend:
    if provider == "paddle":
        return _PaddleBackend()
    return _DoclingBackend()


def _open_protocol_stdout() -> TextIOWrapper:
    """Duplicate stdout, then send all ordinary fd-1 writes to stderr."""

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
    parser.add_argument("--provider", choices=("paddle", "docling"), required=True)
    args = parser.parse_args(argv)
    provider = cast(ProviderName, args.provider)
    protocol_stdout = _open_protocol_stdout()
    backend: _Backend | None = None
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > _MAX_REQUEST_BYTES:
            response = {
                "protocol": _PROTOCOL,
                "request_id": None,
                "status": "error",
                "error": {
                    "code": "ocr_worker_request_too_large",
                    "retryable": False,
                },
            }
        else:
            try:
                request = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                request = None
            with redirect_stdout(sys.stderr):
                response, backend = _process_request(
                    request,
                    provider=provider,
                    backend=backend,
                )
        protocol_stdout.write(
            json.dumps(
                response,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        protocol_stdout.flush()
    protocol_stdout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
