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
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from html.parser import HTMLParser
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
_PADDLE_TABLE_FALLBACK_ALGORITHM_REVISION = "image-region-table-recovery/v3"
_PADDLE_TABLE_FALLBACK_MAX_CANDIDATES = 4
_PADDLE_TABLE_FALLBACK_MIN_SIDE_PX = 8
_PADDLE_TABLE_FALLBACK_MIN_AREA_RATIO = 0.001
_PADDLE_TABLE_FALLBACK_MAX_AREA_RATIO = 0.95
_PADDLE_TABLE_AXIS_OVERLAP_THRESHOLD = 0.7
_PADDLE_RULED_GRID_DARK_THRESHOLD_MAX = 200
_PADDLE_RULED_GRID_HORIZONTAL_RUN = 0.08
_PADDLE_RULED_GRID_VERTICAL_COVERAGE = 0.72
_PADDLE_RULED_GRID_BORDER_COVERAGE = 0.9
_PADDLE_RULED_GRID_STRICT_HORIZONTAL_RUN = 0.1
_PADDLE_RULED_GRID_STRICT_VERTICAL_COVERAGE = 0.78
_PADDLE_RULED_GRID_STRICT_BORDER_COVERAGE = 0.94
_PADDLE_RULED_GRID_THRESHOLD_DELTA = 8
_PADDLE_RULED_GRID_CORROBORATION_COVERAGE = 0.5
_PADDLE_RULED_GRID_MAX_ROWS = 100
_PADDLE_RULED_GRID_MAX_CELLS = 1_000
_PADDLE_RULED_GRID_MAX_TABLES = 8
_PADDLE_TEXT_DETECTION_UNCLIP_RATIO = 0.75

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
    engine_id = "PP-StructureV3:customs-paddle-layout-table-v4"
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
                text_det_unclip_ratio=_PADDLE_TEXT_DETECTION_UNCLIP_RATIO,
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
                rgb_source = source.convert("RGB")
                # RGB uint8 in C order gives the crop hash a platform-stable
                # byte representation (there is no multi-byte endianness).
                image = np.ascontiguousarray(np.asarray(rgb_source, dtype=np.uint8))
        except Exception as exc:
            raise _SafeWorkerError(
                "ocr_backend_inference_failed",
                retryable=True,
            ) from exc
        results = _paddle_predict(
            self._pipeline,
            image,
            error_code="ocr_backend_inference_failed",
        )
        if not results:
            return []
        spans = _paddle_result_to_spans(
            results[0],
            page_number=page_number,
            width_px=width_px,
            height_px=height_px,
        )
        root = _paddle_result_root(results[0])
        if _paddle_tables(root):
            return spans

        candidates = _paddle_image_region_candidates(
            root,
            width_px=width_px,
            height_px=height_px,
        )
        page_bbox = (0, 0, width_px, height_px)
        page_sha256 = hashlib.sha256(image.tobytes(order="C")).hexdigest()
        page_table_key = _paddle_fallback_table_key(
            page_number=page_number,
            crop_bbox=page_bbox,
            crop_sha256=page_sha256,
        )
        page_ruled_spans = _paddle_ruled_grid_table_spans(
            image=rgb_source,
            ocr_root=root,
            table_key=page_table_key,
            crop_bbox=page_bbox,
            page_width_px=width_px,
            page_height_px=height_px,
            corroborating_bboxes=candidates or None,
        )
        if page_ruled_spans:
            return _provider_span_order([*spans, *page_ruled_spans])

        successful_tables: list[list[dict[str, Any]]] = []
        for crop_bbox in candidates:
            x1, y1, x2, y2 = crop_bbox
            crop = np.ascontiguousarray(image[y1:y2, x1:x2])
            if crop.size == 0 or crop.shape[:2] != (y2 - y1, x2 - x1):
                raise _SafeWorkerError("ocr_table_fallback_crop_invalid")
            crop_sha256 = hashlib.sha256(crop.tobytes(order="C")).hexdigest()
            table_key = _paddle_fallback_table_key(
                page_number=page_number,
                crop_bbox=crop_bbox,
                crop_sha256=crop_sha256,
            )
            ruled_spans = _paddle_ruled_grid_table_spans(
                image=rgb_source.crop(crop_bbox),
                ocr_root=root,
                table_key=table_key,
                crop_bbox=crop_bbox,
                page_width_px=width_px,
                page_height_px=height_px,
            )
            if ruled_spans:
                successful_tables.append(ruled_spans)
                if len(successful_tables) > 1:
                    raise _SafeWorkerError("ocr_table_fallback_ambiguous")
                continue
            crop_results = _paddle_predict_table_crop(
                self._pipeline,
                crop,
            )
            if not crop_results:
                raise _SafeWorkerError("ocr_table_fallback_result_empty")
            crop_root = _paddle_result_root(crop_results[0])
            crop_tables = _paddle_tables(crop_root)
            if not crop_tables:
                continue
            if len(crop_tables) != 1:
                raise _SafeWorkerError("ocr_table_fallback_ambiguous")
            crop_spans = _paddle_fallback_table_spans(
                crop_tables[0],
                table_key=table_key,
                crop_bbox=crop_bbox,
                page_width_px=width_px,
                page_height_px=height_px,
            )
            successful_tables.append(crop_spans)
            if len(successful_tables) > 1:
                raise _SafeWorkerError("ocr_table_fallback_ambiguous")
        if not successful_tables:
            return spans
        return _provider_span_order([*spans, *successful_tables[0]])


class _DoclingBackend:
    provider_id = "docling"
    engine_id = "DocumentConverter:default-image-pipeline-offline-v2"
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
    root = _paddle_result_root(result)
    overall = _lookup(root, "overall_ocr_res")
    if overall is None:
        overall = result
    texts = _sequence(_lookup(overall, "rec_texts", "texts"))
    scores = _sequence(_lookup(overall, "rec_scores", "scores"))
    boxes = _sequence(_lookup(overall, "rec_boxes", "rec_polys", "boxes"))
    spans: list[dict[str, Any]] = []
    for index, text_value in enumerate(texts):
        text = str(text_value).strip() if text_value is not None else ""
        if not text:
            continue
        box = _bbox_from_value(
            boxes[index] if index < len(boxes) else None,
            width_px=width_px,
            height_px=height_px,
        )
        if box is None:
            raise _SafeWorkerError("ocr_text_geometry_invalid")
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


def _paddle_predict(
    pipeline: object,
    image: object,
    *,
    error_code: str,
) -> list[Any]:
    try:
        predict = getattr(pipeline, "predict")
        prediction = predict(input=image)
        return prediction if isinstance(prediction, list) else list(cast(Iterable[Any], prediction))
    except _SafeWorkerError:
        raise
    except Exception as exc:
        raise _SafeWorkerError(error_code, retryable=True) from exc


def _paddle_predict_table_crop(
    pipeline: object,
    image: object,
) -> list[Any]:
    try:
        paddlex_pipeline = getattr(pipeline, "paddlex_pipeline")
        general_ocr_pipeline = getattr(paddlex_pipeline, "general_ocr_pipeline")
        table_pipeline = getattr(paddlex_pipeline, "table_recognition_pipeline")
        ocr_prediction = general_ocr_pipeline(image)
        ocr_results = (
            ocr_prediction
            if isinstance(ocr_prediction, list)
            else list(cast(Iterable[Any], ocr_prediction))
        )
        if len(ocr_results) != 1:
            raise _SafeWorkerError(
                "ocr_table_fallback_inference_failed",
                retryable=True,
            )
        prediction = table_pipeline(
            image,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            use_ocr_model=False,
            overall_ocr_res=ocr_results[0],
            cell_sort_by_y_projection=True,
            use_wired_table_cells_trans_to_html=False,
            use_wireless_table_cells_trans_to_html=False,
            use_table_orientation_classify=False,
            use_ocr_results_with_table_cells=True,
            use_e2e_wired_table_rec_model=False,
            use_e2e_wireless_table_rec_model=True,
        )
        results = (
            prediction if isinstance(prediction, list) else list(cast(Iterable[Any], prediction))
        )
        if any(_lookup(_paddle_result_root(result), "error") for result in results):
            raise _SafeWorkerError(
                "ocr_table_fallback_inference_failed",
                retryable=True,
            )
        return results
    except _SafeWorkerError:
        raise
    except Exception as exc:
        raise _SafeWorkerError(
            "ocr_table_fallback_inference_failed",
            retryable=True,
        ) from exc


def _paddle_result_root(result: object) -> object:
    root = _object_mapping(result)
    wrapped = _lookup(root, "res")
    if isinstance(_object_mapping(wrapped), Mapping):
        return _object_mapping(wrapped)
    return root


def _paddle_image_region_candidates(
    root: object,
    *,
    width_px: int,
    height_px: int,
) -> list[tuple[int, int, int, int]]:
    parsing = _sequence(_lookup(root, "parsing_res_list"))
    observed = [
        item
        for item in parsing
        if str(_lookup(item, "label", "block_label", "type") or "").strip().lower() == "image"
    ]
    if not observed:
        layout = _lookup(root, "layout_det_res", "layout_res")
        observed = [
            item
            for item in _sequence(_lookup(layout, "boxes"))
            if str(_lookup(item, "label", "block_label", "type") or "").strip().lower() == "image"
        ]
    if len(observed) > _PADDLE_TABLE_FALLBACK_MAX_CANDIDATES:
        raise _SafeWorkerError("ocr_table_fallback_candidate_limit_exceeded")

    candidates: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    page_area = width_px * height_px
    for item in observed:
        value = _lookup(item, "block_bbox", "bbox", "coordinate", "box")
        bbox = _strict_integer_bbox(value, width_px=width_px, height_px=height_px)
        if bbox is None:
            raise _SafeWorkerError("ocr_table_fallback_candidate_invalid")
        x1, y1, x2, y2 = bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / page_area
        if (
            x2 - x1 < _PADDLE_TABLE_FALLBACK_MIN_SIDE_PX
            or y2 - y1 < _PADDLE_TABLE_FALLBACK_MIN_SIDE_PX
            or area_ratio < _PADDLE_TABLE_FALLBACK_MIN_AREA_RATIO
            or area_ratio > _PADDLE_TABLE_FALLBACK_MAX_AREA_RATIO
        ):
            raise _SafeWorkerError("ocr_table_fallback_candidate_invalid")
        if bbox in seen:
            raise _SafeWorkerError("ocr_table_fallback_candidate_duplicate")
        seen.add(bbox)
        candidates.append(bbox)
    return sorted(candidates, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _strict_integer_bbox(
    value: object,
    *,
    width_px: int,
    height_px: int,
) -> tuple[int, int, int, int] | None:
    coordinates = _strict_bbox_coordinates(
        value,
        width_px=width_px,
        height_px=height_px,
    )
    if coordinates is None:
        return None
    x1, y1, x2, y2 = coordinates
    canonical = (
        math.floor(x1),
        math.floor(y1),
        math.ceil(x2),
        math.ceil(y2),
    )
    if canonical[2] > width_px or canonical[3] > height_px:
        return None
    return canonical


def _strict_bbox_coordinates(
    value: object,
    *,
    width_px: int,
    height_px: int,
) -> tuple[float, float, float, float] | None:
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
        if len(coordinates) >= 8 and len(coordinates) % 2 == 0:
            if not all(_is_finite_number(item) for item in coordinates):
                return None
            numeric = [float(cast(int | float, item)) for item in coordinates]
            coordinates = [
                min(numeric[0::2]),
                min(numeric[1::2]),
                max(numeric[0::2]),
                max(numeric[1::2]),
            ]
        elif len(coordinates) != 4:
            return None
    if not all(_is_finite_number(item) for item in coordinates):
        return None
    x1, y1, x2, y2 = (float(cast(int | float, item)) for item in coordinates)
    if x1 < 0 or y1 < 0 or x2 > width_px or y2 > height_px or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _paddle_fallback_table_key(
    *,
    page_number: int,
    crop_bbox: tuple[int, int, int, int],
    crop_sha256: str,
) -> str:
    payload = {
        "algorithm_revision": _PADDLE_TABLE_FALLBACK_ALGORITHM_REVISION,
        "crop_bbox": list(crop_bbox),
        "crop_sha256": crop_sha256,
        "page_number": page_number,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"p{page_number}-fallback-{digest}"


@dataclass(frozen=True)
class _RuledLine:
    start: int
    end: int

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2


def _ruled_line_groups(
    counts: Sequence[int],
    *,
    minimum_count: int,
    maximum_thickness: int,
) -> list[_RuledLine]:
    observed = [index for index, count in enumerate(counts) if count >= minimum_count]
    if not observed:
        return []
    groups: list[_RuledLine] = []
    start = observed[0]
    previous = observed[0]
    for index in observed[1:]:
        if index == previous + 1:
            previous = index
            continue
        if previous - start + 1 <= maximum_thickness:
            groups.append(_RuledLine(start=start, end=previous))
        start = previous = index
    if previous - start + 1 <= maximum_thickness:
        groups.append(_RuledLine(start=start, end=previous))
    return groups


def _ruled_horizontal_border_coverage(
    pixels: bytes,
    *,
    width: int,
    line: _RuledLine,
    left: int,
    right: int,
    dark_threshold: int,
) -> float:
    interval = right - left
    if interval <= 0:
        return 0.0
    return max(
        (
            sum(
                value <= dark_threshold
                for value in pixels[row * width + left : row * width + right]
            )
            / interval
            for row in range(line.start, line.end + 1)
        ),
        default=0.0,
    )


def _ruled_lines_intersect(
    pixels: bytes,
    *,
    width: int,
    horizontal: _RuledLine,
    vertical: _RuledLine,
    dark_threshold: int,
) -> bool:
    return any(
        pixels[row * width + column] <= dark_threshold
        for row in range(horizontal.start, horizontal.end + 1)
        for column in range(vertical.start, vertical.end + 1)
    )


def _ruled_otsu_threshold(pixels: bytes) -> int:
    histogram = [0] * 256
    for value in pixels:
        histogram[value] += 1
    total = len(pixels)
    if total == 0:
        return 0
    weighted_total = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_threshold = 0
    best_variance = -1.0
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return min(_PADDLE_RULED_GRID_DARK_THRESHOLD_MAX, best_threshold)


def _longest_dark_run(values: bytes, *, dark_threshold: int) -> int:
    longest = 0
    current = 0
    for value in values:
        if value <= dark_threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _ruled_grid_rows_at_threshold(
    grayscale: Image.Image,
    *,
    dark_threshold: int,
    horizontal_run: float,
    vertical_coverage: float,
    border_coverage: float,
) -> list[list[list[float]]]:
    width, height = grayscale.size
    if width < 24 or height < 24:
        return []
    pixels = grayscale.tobytes()
    maximum_thickness = max(3, min(12, min(width, height) // 50))
    horizontal_counts = [
        _longest_dark_run(
            pixels[row * width : (row + 1) * width],
            dark_threshold=dark_threshold,
        )
        for row in range(height)
    ]
    horizontal = _ruled_line_groups(
        horizontal_counts,
        minimum_count=max(
            16,
            math.ceil(width * horizontal_run),
        ),
        maximum_thickness=maximum_thickness,
    )
    if len(horizontal) < 3:
        return []
    if len(horizontal) > (_PADDLE_RULED_GRID_MAX_ROWS * 2) + 2:
        raise _SafeWorkerError("ocr_table_fallback_candidate_limit_exceeded")

    minimum_cell_height = max(8, height // 100)
    minimum_cell_width = max(8, width // 100)
    rows: list[list[list[float]]] = []
    for top, bottom in zip(horizontal, horizontal[1:], strict=False):
        interior_top = top.end + 1
        interior_bottom = bottom.start
        interior_height = interior_bottom - interior_top
        if interior_height < minimum_cell_height:
            continue
        vertical_counts = [
            _longest_dark_run(
                bytes(pixels[row * width + column] for row in range(interior_top, interior_bottom)),
                dark_threshold=dark_threshold,
            )
            for column in range(width)
        ]
        vertical = _ruled_line_groups(
            vertical_counts,
            minimum_count=max(
                3,
                math.ceil(interior_height * vertical_coverage),
            ),
            maximum_thickness=maximum_thickness,
        )
        if len(vertical) < 2:
            continue
        if len(vertical) > _PADDLE_RULED_GRID_MAX_ROWS + 1:
            raise _SafeWorkerError("ocr_table_fallback_candidate_limit_exceeded")

        cells: list[list[float]] = []
        for left_line, right_line in zip(vertical, vertical[1:], strict=False):
            left = left_line.center
            right = right_line.center
            if right - left < minimum_cell_width:
                continue
            top_coverage = _ruled_horizontal_border_coverage(
                pixels,
                width=width,
                line=top,
                left=left,
                right=right,
                dark_threshold=dark_threshold,
            )
            bottom_coverage = _ruled_horizontal_border_coverage(
                pixels,
                width=width,
                line=bottom,
                left=left,
                right=right,
                dark_threshold=dark_threshold,
            )
            if (
                top_coverage > border_coverage
                and bottom_coverage > border_coverage
                and _ruled_lines_intersect(
                    pixels,
                    width=width,
                    horizontal=top,
                    vertical=left_line,
                    dark_threshold=dark_threshold,
                )
                and _ruled_lines_intersect(
                    pixels,
                    width=width,
                    horizontal=top,
                    vertical=right_line,
                    dark_threshold=dark_threshold,
                )
                and _ruled_lines_intersect(
                    pixels,
                    width=width,
                    horizontal=bottom,
                    vertical=left_line,
                    dark_threshold=dark_threshold,
                )
                and _ruled_lines_intersect(
                    pixels,
                    width=width,
                    horizontal=bottom,
                    vertical=right_line,
                    dark_threshold=dark_threshold,
                )
            ):
                cells.append(
                    [
                        float(left),
                        float(top.center),
                        float(right),
                        float(bottom.center),
                    ]
                )
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return []
    if len(rows) > _PADDLE_RULED_GRID_MAX_ROWS:
        raise _SafeWorkerError("ocr_table_fallback_candidate_limit_exceeded")
    if sum(len(row) for row in rows) > _PADDLE_RULED_GRID_MAX_CELLS:
        raise _SafeWorkerError("ocr_table_fallback_candidate_limit_exceeded")
    return rows


def _same_ruled_grid_rows(
    first: Sequence[Sequence[Sequence[float]]],
    second: Sequence[Sequence[Sequence[float]]],
    *,
    tolerance: float,
) -> bool:
    return len(first) == len(second) and all(
        len(first_row) == len(second_row)
        and all(
            all(
                abs(first_coordinate - second_coordinate) <= tolerance
                for first_coordinate, second_coordinate in zip(
                    first_cell,
                    second_cell,
                    strict=True,
                )
            )
            for first_cell, second_cell in zip(first_row, second_row, strict=True)
        )
        for first_row, second_row in zip(first, second, strict=True)
    )


def _ruled_grid_rows(image: Image.Image) -> list[list[list[float]]]:
    grayscale = image.convert("L")
    pixels = grayscale.tobytes()
    otsu_threshold = _ruled_otsu_threshold(pixels)
    nominal_threshold = max(
        0,
        otsu_threshold - _PADDLE_RULED_GRID_THRESHOLD_DELTA,
    )
    strict_threshold = min(
        _PADDLE_RULED_GRID_DARK_THRESHOLD_MAX,
        otsu_threshold + _PADDLE_RULED_GRID_THRESHOLD_DELTA,
    )
    nominal = _ruled_grid_rows_at_threshold(
        grayscale,
        dark_threshold=nominal_threshold,
        horizontal_run=_PADDLE_RULED_GRID_HORIZONTAL_RUN,
        vertical_coverage=_PADDLE_RULED_GRID_VERTICAL_COVERAGE,
        border_coverage=_PADDLE_RULED_GRID_BORDER_COVERAGE,
    )
    strict = _ruled_grid_rows_at_threshold(
        grayscale,
        dark_threshold=strict_threshold,
        horizontal_run=_PADDLE_RULED_GRID_STRICT_HORIZONTAL_RUN,
        vertical_coverage=_PADDLE_RULED_GRID_STRICT_VERTICAL_COVERAGE,
        border_coverage=_PADDLE_RULED_GRID_STRICT_BORDER_COVERAGE,
    )
    if not nominal or not strict:
        return []
    tolerance = max(2.0, min(image.size) * 0.002)
    if not _same_ruled_grid_rows(nominal, strict, tolerance=tolerance):
        return []
    return nominal


def _same_ruled_grid_signature(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    *,
    tolerance: float,
) -> bool:
    return len(first) == len(second) and all(
        abs(first_cell[0] - second_cell[0]) <= tolerance
        and abs(first_cell[2] - second_cell[2]) <= tolerance
        for first_cell, second_cell in zip(first, second, strict=True)
    )


def _ruled_grid_sections(
    rows: Sequence[Sequence[Sequence[float]]],
    *,
    width: int,
) -> list[list[Sequence[Sequence[float]]]]:
    tolerance = max(3.0, width * 0.005)
    cells = [cell for row in rows for cell in row]
    neighbours: list[set[int]] = [set() for _ in cells]
    for first_index, first in enumerate(cells):
        for second_index, second in enumerate(
            cells[first_index + 1 :],
            start=first_index + 1,
        ):
            vertical_overlap = max(
                0.0,
                min(first[3], second[3]) - max(first[1], second[1]),
            )
            horizontal_overlap = max(
                0.0,
                min(first[2], second[2]) - max(first[0], second[0]),
            )
            vertical_join = (
                min(abs(first[2] - second[0]), abs(second[2] - first[0])) <= 1.0
                and vertical_overlap / min(first[3] - first[1], second[3] - second[1])
                > _PADDLE_TABLE_AXIS_OVERLAP_THRESHOLD
            )
            horizontal_join = (
                min(abs(first[3] - second[1]), abs(second[3] - first[1])) <= 1.0
                and horizontal_overlap / min(first[2] - first[0], second[2] - second[0])
                > _PADDLE_TABLE_AXIS_OVERLAP_THRESHOLD
                and (
                    abs(first[0] - second[0]) <= tolerance or abs(first[2] - second[2]) <= tolerance
                )
            )
            if vertical_join or horizontal_join:
                neighbours[first_index].add(second_index)
                neighbours[second_index].add(first_index)

    unseen = set(range(len(cells)))
    components: list[list[Sequence[float]]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        member_indices = [start]
        while stack:
            current = stack.pop()
            for neighbour in neighbours[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
                    member_indices.append(neighbour)
        components.append([cells[index] for index in member_indices])

    sections: list[list[Sequence[Sequence[float]]]] = []
    for component in components:
        cells_by_top: dict[float, list[Sequence[float]]] = {}
        for cell in component:
            cells_by_top.setdefault(cell[1], []).append(cell)
        section = cast(
            list[Sequence[Sequence[float]]],
            [
                sorted(row, key=lambda cell: (cell[0], cell[2]))
                for _, row in sorted(cells_by_top.items())
            ],
        )
        if (
            len(section) < 2
            or any(len(row) < 2 for row in section)
            or any(
                not _same_ruled_grid_signature(
                    section[0],
                    row,
                    tolerance=tolerance,
                )
                for row in section[1:]
            )
        ):
            continue
        sections.append(section)
    sections.sort(key=lambda section: (section[0][0][1], section[0][0][0]))
    if len(sections) > _PADDLE_RULED_GRID_MAX_TABLES:
        raise _SafeWorkerError("ocr_table_fallback_candidate_limit_exceeded")
    return sections


def _paddle_ruled_grid_table_spans(
    *,
    image: Image.Image,
    ocr_root: object,
    table_key: str,
    crop_bbox: tuple[int, int, int, int],
    page_width_px: int,
    page_height_px: int,
    corroborating_bboxes: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    rows = _ruled_grid_rows(image)
    if not rows:
        return []
    sections = _ruled_grid_sections(rows, width=image.width)
    if not sections:
        return []

    x_offset, y_offset, x2, y2 = crop_bbox
    if image.size != (x2 - x_offset, y2 - y_offset):
        raise _SafeWorkerError("ocr_table_fallback_crop_invalid")
    if corroborating_bboxes is not None:
        filtered_sections: list[list[Sequence[Sequence[float]]]] = []
        for section in sections:
            local_bbox = _union_bbox([cell for row in section for cell in row])
            full_bbox = [
                local_bbox[0] + x_offset,
                local_bbox[1] + y_offset,
                local_bbox[2] + x_offset,
                local_bbox[3] + y_offset,
            ]
            table_area = (full_bbox[2] - full_bbox[0]) * (full_bbox[3] - full_bbox[1])
            if table_area <= 0:
                raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
            if any(
                _intersection_area(full_bbox, candidate) / table_area
                > _PADDLE_RULED_GRID_CORROBORATION_COVERAGE
                for candidate in corroborating_bboxes
            ):
                filtered_sections.append(section)
        sections = filtered_sections
        if not sections:
            return []

    section_cells: list[list[dict[str, Any]]] = []
    row_envelopes: list[list[float]] = []
    for section in sections:
        cells: list[dict[str, Any]] = []
        for row_index, row in enumerate(section):
            full_row = [
                [
                    cell[0] + x_offset,
                    cell[1] + y_offset,
                    cell[2] + x_offset,
                    cell[3] + y_offset,
                ]
                for cell in row
            ]
            row_envelopes.append(_union_bbox(full_row))
            for column_index, bbox in enumerate(full_row):
                cells.append(
                    {
                        "bbox": bbox,
                        "row_index": row_index,
                        "column_index": column_index,
                        "row_span": 1,
                        "column_span": 1,
                        "tokens": [],
                    }
                )
        section_cells.append(cells)
    all_cells = [cell for cells in section_cells for cell in cells]

    token_source = _lookup(ocr_root, "overall_ocr_res")
    token_box_values = _sequence(_lookup(token_source, "rec_boxes"))
    token_text_values = _sequence(_lookup(token_source, "rec_texts"))
    token_score_values = _sequence(_lookup(token_source, "rec_scores"))
    if (
        not token_box_values
        or len(token_box_values) != len(token_text_values)
        or len(token_box_values) != len(token_score_values)
    ):
        raise _SafeWorkerError("ocr_table_fallback_structure_invalid")

    for token_index, (box_value, text_value, score_value) in enumerate(
        zip(
            token_box_values,
            token_text_values,
            token_score_values,
            strict=True,
        )
    ):
        coordinates = _strict_bbox_coordinates(
            box_value,
            width_px=page_width_px,
            height_px=page_height_px,
        )
        if coordinates is None:
            raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
        token_box = [float(value) for value in coordinates]
        token_area = (token_box[2] - token_box[0]) * (token_box[3] - token_box[1])
        overlap_ratios = [
            _intersection_area(cast(Sequence[float], cell["bbox"]), token_box) / token_area
            for cell in all_cells
        ]
        assignments = [
            cell
            for cell, overlap_ratio in zip(all_cells, overlap_ratios, strict=True)
            if overlap_ratio > 0.7
        ]
        if len(assignments) > 1:
            raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
        if not assignments:
            center_x = (token_box[0] + token_box[2]) / 2
            center_y = (token_box[1] + token_box[3]) / 2
            if (
                any(
                    envelope[0] < center_x < envelope[2] and envelope[1] < center_y < envelope[3]
                    for envelope in row_envelopes
                )
                or max(overlap_ratios, default=0.0) > 0.1
            ):
                raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
            continue
        text = str(text_value).strip() if text_value is not None else ""
        if (
            not text
            or not _is_finite_number(score_value)
            or not 0 <= float(cast(int | float, score_value)) <= 1
        ):
            raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
        assignments[0]["tokens"].append(
            (
                token_index,
                text,
                float(cast(int | float, score_value)),
            )
        )

    spans: list[dict[str, Any]] = []
    for section_index, cells in enumerate(section_cells):
        normalized_cells: list[dict[str, Any]] = []
        for cell in cells:
            tokens = cast(list[tuple[int, str, float]], cell.pop("tokens"))
            if not tokens:
                raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
            ordered = sorted(tokens, key=lambda token: token[0])
            normalized_cells.append(
                {
                    **cell,
                    "text": " ".join(token[1] for token in ordered),
                    "confidence": min(token[2] for token in ordered),
                }
            )
        signature = [
            [
                cell["row_index"],
                cell["column_index"],
                cell["bbox"],
            ]
            for cell in normalized_cells
        ]
        digest = hashlib.sha256(
            json.dumps(
                signature,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        spans.extend(
            _table_cells_to_spans(
                normalized_cells,
                table_key=f"{table_key}-ruled-{section_index}-{digest}",
            )
        )
    return spans


def _paddle_fallback_table_spans(
    table: object,
    *,
    table_key: str,
    crop_bbox: tuple[int, int, int, int],
    page_width_px: int,
    page_height_px: int,
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = crop_bbox
    crop_width = x2 - x1
    crop_height = y2 - y1
    spans = _paddle_table_to_spans(
        table,
        table_key=table_key,
        width_px=crop_width,
        height_px=crop_height,
        error_code="ocr_table_fallback_structure_invalid",
    )
    roles = [span.get("role") for span in spans]
    rows = [span for span in spans if span.get("role") == "table_row"]
    cells = [span for span in spans if span.get("role") == "table_cell"]
    if roles.count("table") != 1 or not rows or not cells:
        raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
    row_indices = {row.get("row_index") for row in rows}
    cell_coordinates = [(cell.get("row_index"), cell.get("column_index")) for cell in cells]
    if (
        None in row_indices
        or len(row_indices) != len(rows)
        or any(cell.get("row_index") not in row_indices for cell in cells)
        or any(None in coordinate for coordinate in cell_coordinates)
        or len(set(cell_coordinates)) != len(cell_coordinates)
        or any(
            span.get("table_key") != table_key
            for span in spans
            if span.get("role") in {"table", "table_row", "table_cell"}
        )
    ):
        raise _SafeWorkerError("ocr_table_fallback_structure_invalid")

    remapped: list[dict[str, Any]] = []
    for span in spans:
        bbox = span.get("bbox")
        local = _strict_bbox_coordinates(
            bbox,
            width_px=crop_width,
            height_px=crop_height,
        )
        if local is None:
            raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
        left, top, right, bottom = local
        full_bbox = [float(left + x1), float(top + y1), float(right + x1), float(bottom + y1)]
        if (
            full_bbox[0] < x1
            or full_bbox[1] < y1
            or full_bbox[2] > x2
            or full_bbox[3] > y2
            or full_bbox[2] > page_width_px
            or full_bbox[3] > page_height_px
        ):
            raise _SafeWorkerError("ocr_table_fallback_structure_invalid")
        remapped.append({**span, "bbox": full_bbox})
    return remapped


def _paddle_tables(root: object) -> list[object]:
    direct = _lookup(root, "table_res_list", "tables")
    if direct is not None:
        tables = _sequence(direct)
        if not tables and (
            not isinstance(direct, Sequence) or isinstance(direct, (str, bytes, bytearray))
        ):
            raise _SafeWorkerError("ocr_table_structure_invalid")
        return tables
    single = _lookup(root, "table_res")
    if single is not None:
        return _sequence(single) or [single]
    # parsing_res_list is a presentation/layout view.  Paddle serializes its
    # table-labelled blocks without the recognition cell/token relationship,
    # so they are not structured table evidence and must not suppress the
    # bounded image-region recovery path.
    return []


@dataclass(frozen=True)
class _PaddleHtmlCell:
    text: str
    row_span: int
    column_span: int


@dataclass(frozen=True)
class _PositionedPaddleHtmlCell:
    text: str
    row_index: int
    column_index: int
    row_span: int
    column_span: int


class _PaddleTableHtmlParser(HTMLParser):
    """Parse a bounded subset of Paddle's table HTML, including explicit spans."""

    _OUTER_TAGS = {"html", "body", "thead", "tbody", "tfoot"}
    _INLINE_TAGS = {"b", "strong", "span", "i", "em"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_PaddleHtmlCell]] = []
        self.invalid = False
        self._table_seen = False
        self._in_table = False
        self._row: list[_PaddleHtmlCell] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_row_span = 1
        self._cell_column_span = 1

    @staticmethod
    def _span_value(value: str | None) -> int | None:
        if value is None or not value.isascii() or not value.isdigit():
            return None
        parsed = int(value)
        if not 1 <= parsed <= 100:
            return None
        return parsed

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            if self._table_seen or self._in_table or self._row is not None:
                self.invalid = True
                return
            self._table_seen = True
            self._in_table = True
            return
        if tag == "tr":
            if not self._in_table or self._row is not None or self._cell_parts is not None:
                self.invalid = True
                return
            self._row = []
            return
        if tag in {"td", "th"}:
            if not self._in_table or self._row is None or self._cell_parts is not None:
                self.invalid = True
                return
            spans: dict[str, str | None] = {}
            for name, value in attrs:
                normalized_name = name.lower()
                if normalized_name not in {"rowspan", "colspan"}:
                    continue
                if normalized_name in spans:
                    self.invalid = True
                    return
                spans[normalized_name] = value
            row_span = self._span_value(spans["rowspan"]) if "rowspan" in spans else 1
            column_span = self._span_value(spans["colspan"]) if "colspan" in spans else 1
            if row_span is None or column_span is None:
                self.invalid = True
                return
            self._cell_row_span = row_span
            self._cell_column_span = column_span
            self._cell_parts = []
            return
        if tag == "br":
            if self._cell_parts is None:
                self.invalid = True
            else:
                self._cell_parts.append(" ")
            return
        if tag in self._INLINE_TAGS:
            if self._cell_parts is None:
                self.invalid = True
            return
        if tag in self._OUTER_TAGS:
            if self._cell_parts is not None:
                self.invalid = True
            return
        self.invalid = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"}:
            if self._cell_parts is None or self._row is None:
                self.invalid = True
                return
            self._row.append(
                _PaddleHtmlCell(
                    text="".join(self._cell_parts),
                    row_span=self._cell_row_span,
                    column_span=self._cell_column_span,
                )
            )
            self._cell_parts = None
            self._cell_row_span = 1
            self._cell_column_span = 1
            return
        if tag == "tr":
            if self._row is None or self._cell_parts is not None or not self._row:
                self.invalid = True
                return
            self.rows.append(self._row)
            self._row = None
            return
        if tag == "table":
            if not self._in_table or self._row is not None or self._cell_parts is not None:
                self.invalid = True
                return
            self._in_table = False
            return
        if tag in self._INLINE_TAGS or tag in self._OUTER_TAGS or tag == "br":
            return
        self.invalid = True

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        elif data.strip():
            self.invalid = True

    def handle_comment(self, data: str) -> None:
        del data
        self.invalid = True

    def validated_rows(self) -> list[list[_PaddleHtmlCell]]:
        if (
            self.invalid
            or not self._table_seen
            or self._in_table
            or self._row is not None
            or self._cell_parts is not None
            or not self.rows
        ):
            return []
        return self.rows


def _paddle_html_rows(value: object) -> list[list[_PaddleHtmlCell]]:
    if not isinstance(value, str) or not value.strip():
        return []
    parser = _PaddleTableHtmlParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return []
    return parser.validated_rows()


def _positioned_paddle_html_cells(
    rows: Sequence[Sequence[_PaddleHtmlCell]],
) -> list[_PositionedPaddleHtmlCell]:
    if not rows or len(rows) > 100:
        return []

    occupied: set[tuple[int, int]] = set()
    positioned: list[_PositionedPaddleHtmlCell] = []
    width = 0
    for row_index, row in enumerate(rows):
        if not row:
            return []
        column_index = 0
        for cell in row:
            while (row_index, column_index) in occupied:
                column_index += 1
            row_end = row_index + cell.row_span
            column_end = column_index + cell.column_span
            if row_end > len(rows) or column_end > 100:
                return []
            coordinates = {
                (covered_row, covered_column)
                for covered_row in range(row_index, row_end)
                for covered_column in range(column_index, column_end)
            }
            if occupied.intersection(coordinates):
                return []
            occupied.update(coordinates)
            positioned.append(
                _PositionedPaddleHtmlCell(
                    text=cell.text,
                    row_index=row_index,
                    column_index=column_index,
                    row_span=cell.row_span,
                    column_span=cell.column_span,
                )
            )
            width = max(width, column_end)
            column_index = column_end

    if width == 0 or not positioned:
        return []
    # HTML permits a row to end before the table's widest row. Those omitted
    # trailing slots are unambiguous implicit empty cells. Interior holes are
    # not: every row's occupied coordinates must therefore remain a contiguous
    # prefix from column zero.
    for row_index in range(len(rows)):
        columns = {column for row, column in occupied if row == row_index}
        if not columns or columns != set(range(max(columns) + 1)):
            return []
    return positioned


def _normalized_paddle_cell_text(value: object) -> str:
    text = str(value).replace("<b>", "").replace("</b>", "")
    return " ".join(text.split())


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _strict_paddle_table_cells(
    table: object,
    *,
    width_px: int,
    height_px: int,
    error_code: str,
) -> list[dict[str, Any]]:
    cell_values = _sequence(_lookup(table, "cell_box_list"))
    token_source = _lookup(table, "table_ocr_pred")
    token_box_values = _sequence(_lookup(token_source, "rec_boxes"))
    token_text_values = _sequence(_lookup(token_source, "rec_texts"))
    token_score_values = _sequence(_lookup(token_source, "rec_scores"))
    html_rows = _paddle_html_rows(_lookup(table, "pred_html"))
    html_cells = _positioned_paddle_html_cells(html_rows)

    if (
        not cell_values
        or not token_box_values
        or len(token_box_values) != len(token_text_values)
        or len(token_box_values) != len(token_score_values)
        or not html_rows
        or len(html_cells) != len(cell_values)
    ):
        raise _SafeWorkerError(error_code)

    cell_boxes: list[list[float]] = []
    for value in cell_values:
        coordinates = _strict_bbox_coordinates(
            value,
            width_px=width_px,
            height_px=height_px,
        )
        if coordinates is None:
            raise _SafeWorkerError(error_code)
        cell_boxes.append([float(item) for item in coordinates])
    if len({tuple(box) for box in cell_boxes}) != len(cell_boxes):
        raise _SafeWorkerError(error_code)
    for first_index, first in enumerate(cell_boxes):
        for second in cell_boxes[first_index + 1 :]:
            if _intersection_area(first, second) > 0:
                raise _SafeWorkerError(error_code)

    for first_index, (first_cell, first_box) in enumerate(zip(html_cells, cell_boxes, strict=True)):
        first_row_end = first_cell.row_index + first_cell.row_span
        first_column_end = first_cell.column_index + first_cell.column_span
        for second_cell, second_box in zip(
            html_cells[first_index + 1 :],
            cell_boxes[first_index + 1 :],
            strict=True,
        ):
            second_row_end = second_cell.row_index + second_cell.row_span
            second_column_end = second_cell.column_index + second_cell.column_span
            rows_overlap = max(first_cell.row_index, second_cell.row_index) < min(
                first_row_end, second_row_end
            )
            columns_overlap = max(first_cell.column_index, second_cell.column_index) < min(
                first_column_end, second_column_end
            )
            horizontal_overlap = min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])
            vertical_overlap = min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
            horizontal_coverage = max(0.0, horizontal_overlap) / min(
                first_box[2] - first_box[0],
                second_box[2] - second_box[0],
            )
            vertical_coverage = max(0.0, vertical_overlap) / min(
                first_box[3] - first_box[1],
                second_box[3] - second_box[1],
            )

            if rows_overlap:
                if columns_overlap or vertical_coverage <= _PADDLE_TABLE_AXIS_OVERLAP_THRESHOLD:
                    raise _SafeWorkerError(error_code)
                if (
                    first_column_end <= second_cell.column_index and first_box[2] > second_box[0]
                ) or (
                    second_column_end <= first_cell.column_index and second_box[2] > first_box[0]
                ):
                    raise _SafeWorkerError(error_code)
            if columns_overlap:
                if rows_overlap or horizontal_coverage <= _PADDLE_TABLE_AXIS_OVERLAP_THRESHOLD:
                    raise _SafeWorkerError(error_code)
                if (first_row_end <= second_cell.row_index and first_box[3] > second_box[1]) or (
                    second_row_end <= first_cell.row_index and second_box[3] > first_box[1]
                ):
                    raise _SafeWorkerError(error_code)

    tokens_by_cell: list[list[tuple[int, str, float]]] = [[] for _ in cell_boxes]
    for token_index, (box_value, text_value, score_value) in enumerate(
        zip(
            token_box_values,
            token_text_values,
            token_score_values,
            strict=True,
        )
    ):
        token_coordinates = _strict_bbox_coordinates(
            box_value,
            width_px=width_px,
            height_px=height_px,
        )
        text = str(text_value).strip() if text_value is not None else ""
        if (
            token_coordinates is None
            or not text
            or not _is_finite_number(score_value)
            or not 0 <= float(cast(int | float, score_value)) <= 1
        ):
            raise _SafeWorkerError(error_code)
        token_box = [float(item) for item in token_coordinates]
        token_area = (token_box[2] - token_box[0]) * (token_box[3] - token_box[1])
        assignments = [
            index
            for index, cell_box in enumerate(cell_boxes)
            if _intersection_area(cell_box, token_box) / token_area > 0.7
        ]
        if len(assignments) != 1:
            raise _SafeWorkerError(error_code)
        tokens_by_cell[assignments[0]].append(
            (token_index, text, float(cast(int | float, score_value)))
        )

    cells: list[dict[str, Any]] = []
    for cell_index, tokens in enumerate(tokens_by_cell):
        if not tokens:
            raise _SafeWorkerError(error_code)
        ordered = sorted(tokens, key=lambda token: token[0])
        text = " ".join(token[1] for token in ordered)
        if _normalized_paddle_cell_text(text) != _normalized_paddle_cell_text(
            html_cells[cell_index].text
        ):
            raise _SafeWorkerError(error_code)
        html_cell = html_cells[cell_index]
        cells.append(
            {
                "text": text,
                "bbox": cell_boxes[cell_index],
                "confidence": min(token[2] for token in ordered),
                "row_index": html_cell.row_index,
                "column_index": html_cell.column_index,
                "row_span": html_cell.row_span,
                "column_span": html_cell.column_span,
            }
        )
    return cells


def _table_cells_to_spans(
    cells: Sequence[Mapping[str, Any]],
    *,
    table_key: str,
    table_bbox: Sequence[float] | None = None,
    error_code: str = "ocr_table_structure_invalid",
) -> list[dict[str, Any]]:
    resolved_table_bbox = (
        [float(value) for value in table_bbox]
        if table_bbox is not None
        else _union_bbox([cell["bbox"] for cell in cells])
    )
    table_confidence = fmean(cell["confidence"] for cell in cells)
    spans: list[dict[str, Any]] = [
        {
            "role": "table",
            "text": " | ".join(cell["text"] for cell in cells),
            "bbox": resolved_table_bbox,
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
        row_band_cells = [cell for cell in row_cells if cast(int, cell["row_span"]) == 1]
        if not row_band_cells:
            raise _SafeWorkerError(error_code)
        row_y1 = min(float(cast(Sequence[float], cell["bbox"])[1]) for cell in row_band_cells)
        row_y2 = max(float(cast(Sequence[float], cell["bbox"])[3]) for cell in row_band_cells)
        if row_y1 >= row_y2:
            raise _SafeWorkerError(error_code)
        spans.append(
            {
                "role": "table_row",
                "text": " | ".join(cell["text"] for cell in row_cells),
                # A logical row is a full-width physical band. Vertically
                # merged cells are excluded from the y-band derivation so
                # adjacent rows remain ordered and non-overlapping.
                "bbox": [
                    resolved_table_bbox[0],
                    row_y1,
                    resolved_table_bbox[2],
                    row_y2,
                ],
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
                "row_span": cell["row_span"],
                "column_span": cell["column_span"],
            }
        )
    return spans


def _paddle_table_to_spans(
    table: object,
    *,
    table_key: str,
    width_px: int,
    height_px: int,
    error_code: str = "ocr_table_structure_invalid",
) -> list[dict[str, Any]]:
    cells = _strict_paddle_table_cells(
        table,
        width_px=width_px,
        height_px=height_px,
        error_code=error_code,
    )
    return _table_cells_to_spans(
        cells,
        table_key=table_key,
        error_code=error_code,
    )


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
        if not text:
            continue
        if box is None:
            raise _SafeWorkerError("ocr_text_geometry_invalid")
        label = str(_lookup(item, "label") or "").lower()
        role = (
            "heading"
            if "heading" in label or "title" in label or label == "section_header"
            else "text"
        )
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
        if not text or box is None:
            raise _SafeWorkerError("ocr_table_structure_invalid")
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
                "row_span": _docling_cell_span(
                    cell,
                    direct_name="row_span",
                    start_name="start_row_offset_idx",
                    end_name="end_row_offset_idx",
                ),
                "column_span": _docling_cell_span(
                    cell,
                    direct_name="col_span",
                    start_name="start_col_offset_idx",
                    end_name="end_col_offset_idx",
                ),
            }
        )
    if not cells:
        raise _SafeWorkerError("ocr_table_structure_invalid")
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
    return _table_cells_to_spans(
        cells,
        table_key=table_key,
        table_bbox=table_box,
    )


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
        if not all(_is_finite_number(item) for item in coordinates):
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
    if not all(_is_finite_number(item) for item in coordinates):
        return None
    x1, y1, x2, y2 = (float(cast(int | float, item)) for item in coordinates)
    if x1 < 0 or y1 < 0 or x2 > width_px or y2 > height_px or x2 <= x1 or y2 <= y1:
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


def _docling_cell_span(
    cell: object,
    *,
    direct_name: str,
    start_name: str,
    end_name: str,
) -> int:
    direct_value = _lookup(cell, direct_name)
    start_value = _lookup(cell, start_name)
    end_value = _lookup(cell, end_name)

    direct = (
        direct_value
        if isinstance(direct_value, int)
        and not isinstance(direct_value, bool)
        and 1 <= direct_value <= 100
        else None
    )
    if direct_value is not None and direct is None:
        raise _SafeWorkerError("ocr_table_structure_invalid")

    inferred: int | None = None
    if start_value is not None or end_value is not None:
        start = _non_negative_int(start_value)
        end = _non_negative_int(end_value)
        if start is None or end is None or not 1 <= end - start <= 100:
            raise _SafeWorkerError("ocr_table_structure_invalid")
        inferred = end - start
    if direct is not None and inferred is not None and direct != inferred:
        raise _SafeWorkerError("ocr_table_structure_invalid")
    return direct or inferred or 1


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
