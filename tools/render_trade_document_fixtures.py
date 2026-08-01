"""Render fixed-layout synthetic trade documents with exact layout Gold.

The renderer uses ReportLab for invariant single-page PDFs and pypdfium2 for
the PDF-to-PNG path that the Agent will actually consume.  Imports are checked
at runtime so environments without the optional rendering group fail closed
with an actionable message instead of silently changing the rendering path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from customs_ai_agent.schemas.models import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_LONG_EDGE,
    MAX_IMAGE_VISUAL_TOKENS,
)
from tools.dataset_provenance import DatasetProvenanceError, verify_dataset_provenance


RENDER_SCHEMA = "trade-document-render-manifest/v2"
LAYOUT_SCHEMA = "trade-document-layout-gold/v2"
TEMPLATE_VERSION = "fixed-trade-document-layout/v2"
RENDER_DPI = 200
REQUIRED_DEPENDENCIES = ("reportlab", "pypdfium2")
FONT_REGULAR = "CustomsVera"
FONT_BOLD = "CustomsVera-Bold"
FONT_ASSETS = {
    FONT_REGULAR: {
        "filename": "Vera.ttf",
        "sha256": "c4c45690b345435b2cba52ecabe275f05e49b389b39fe68ad03afbb551288d3d",
    },
    FONT_BOLD: {
        "filename": "VeraBd.ttf",
        "sha256": "cc037385e4d55bfde89b13e03091ee93bf40c0c52ddd391ff031ab276f13b8e9",
    },
}
DEFAULT_DATASET_LOCK = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "dataset.lock.json"
)
FONT_LICENSE = {
    "name": "Bitstream Vera Fonts Copyright",
    "filename": "bitstream-vera-license.txt",
    "sha256": "3361d054759a2fc686a2c058be82deaf9c2e6fe549be9004d7935a6c1736315d",
}
DOCUMENT_TITLES = {
    "commercial_invoice": "COMMERCIAL INVOICE",
    "packing_list": "PACKING LIST",
    "bill_of_lading": "BILL OF LADING",
    "certificate_of_origin": "CERTIFICATE OF ORIGIN",
}
TABLE_FIELDS = {
    "commercial_invoice": {
        "Goods Description",
        "Quantity",
        "Unit Price",
        "Amount",
    },
    "packing_list": {"Goods Details"},
    "bill_of_lading": {
        "No. & Kinds of Containers or Packages",
        "Description of Goods",
        "Gross Weight",
        "Revenue Tons",
    },
    "certificate_of_origin": {"Goods Details"},
}


class RenderError(RuntimeError):
    """Raised when a document cannot be rendered reproducibly."""


def require_render_dependencies(
    importer: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Import optional renderer dependencies or fail with the exact group needed."""

    modules: dict[str, Any] = {}
    missing = []
    for name in REQUIRED_DEPENDENCIES:
        try:
            modules[name] = importer(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RenderError(
            "document rendering requires optional dependencies: "
            + ", ".join(missing)
            + "; install the locked renderer dependency group"
        )
    modules["canvas"] = importer("reportlab.pdfgen.canvas")
    modules["pagesizes"] = importer("reportlab.lib.pagesizes")
    modules["pdfmetrics"] = importer("reportlab.pdfbase.pdfmetrics")
    modules["ttfonts"] = importer("reportlab.pdfbase.ttfonts")
    reportlab_file = getattr(modules["reportlab"], "__file__", None)
    if not isinstance(reportlab_file, str):
        raise RenderError("reportlab package location is unavailable")
    font_root = Path(reportlab_file).resolve().parent / "fonts"
    font_metadata: list[dict[str, Any]] = []
    for font_name, expected in FONT_ASSETS.items():
        path = font_root / expected["filename"]
        try:
            payload = path.read_bytes()
        except OSError:
            raise RenderError("the pinned renderer font asset is unavailable") from None
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if not path.is_file() or path.is_symlink() or actual_sha256 != expected["sha256"]:
            raise RenderError("the pinned renderer font asset hash does not match")
        modules["pdfmetrics"].registerFont(modules["ttfonts"].TTFont(font_name, str(path)))
        font_metadata.append(
            {
                "font_name": font_name,
                "filename": expected["filename"],
                "sha256": actual_sha256,
                "byte_size": len(payload),
            }
        )
    license_path = font_root / FONT_LICENSE["filename"]
    try:
        license_payload = license_path.read_bytes()
    except OSError:
        raise RenderError("the pinned renderer font license is unavailable") from None
    if (
        not license_path.is_file()
        or license_path.is_symlink()
        or hashlib.sha256(license_payload).hexdigest() != FONT_LICENSE["sha256"]
    ):
        raise RenderError("the pinned renderer font license hash does not match")
    modules["font_assets"] = sorted(font_metadata, key=lambda item: item["font_name"])
    modules["font_license"] = {
        "name": FONT_LICENSE["name"],
        "filename": FONT_LICENSE["filename"],
        "sha256": FONT_LICENSE["sha256"],
        "byte_size": len(license_payload),
    }
    return modules


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot read render source: {path}") from exc
    if not all(isinstance(record, dict) for record in records):
        raise RenderError(f"render source contains a non-object record: {path}")
    return records


def _write_generated(path: Path, content: bytes, *, refresh: bool) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise RenderError(f"render output is not a regular file: {path}")
        if path.read_bytes() == content:
            return
        if not refresh:
            raise RenderError(f"render output differs: {path}; use --refresh after review")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary = Path(file_handle.name)
            file_handle.write(content)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _round_bbox(bbox: list[float]) -> list[float]:
    return [round(value, 3) for value in bbox]


def _union_bbox(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return _round_bbox(
        [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    )


class LayoutRecorder:
    """Draw text and retain exact PDF-point coordinates for every visible word."""

    def __init__(self, pdfmetrics: Any, page_width: float, page_height: float) -> None:
        self.pdfmetrics = pdfmetrics
        self.page_width = float(page_width)
        self.page_height = float(page_height)
        self.words: list[dict[str, Any]] = []
        self.fields: dict[str, dict[str, Any]] = {}
        self.tables: list[dict[str, Any]] = []

    def _wrapped_lines(
        self,
        text: str,
        font_name: str,
        font_size: float,
        max_width: float,
    ) -> list[list[str]]:
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return []
        lines: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            proposed = " ".join([*current, token])
            if current and self.pdfmetrics.stringWidth(proposed, font_name, font_size) > max_width:
                lines.append(current)
                current = [token]
            else:
                current.append(token)
        if current:
            lines.append(current)
        return lines

    def draw_text(
        self,
        pdf: Any,
        *,
        text: str,
        x: float,
        first_baseline_y: float,
        max_width: float,
        font_name: str,
        font_size: float,
        role: str,
        field_id: str | None = None,
        line_gap: float = 2.0,
    ) -> list[list[float]]:
        pdf.setFont(font_name, font_size)
        lines = self._wrapped_lines(text, font_name, font_size, max_width)
        boxes: list[list[float]] = []
        space_width = self.pdfmetrics.stringWidth(" ", font_name, font_size)
        for line_index, tokens in enumerate(lines):
            baseline = first_baseline_y - line_index * (font_size + line_gap)
            cursor = x
            for token_index, token in enumerate(tokens):
                if token_index:
                    cursor += space_width
                pdf.drawString(cursor, baseline, token)
                width = self.pdfmetrics.stringWidth(token, font_name, font_size)
                bbox = _round_bbox(
                    [
                        cursor,
                        baseline - font_size * 0.25,
                        cursor + width,
                        baseline + font_size * 0.82,
                    ]
                )
                word = {
                    "word_id": f"word-{len(self.words) + 1:05d}",
                    "text": token,
                    "role": role,
                    "bbox_pdf_points": bbox,
                }
                if field_id is not None:
                    word["field_id"] = field_id
                self.words.append(word)
                boxes.append(bbox)
                cursor += width
        if field_id is not None:
            field = self.fields.setdefault(
                field_id,
                {
                    "field_id": field_id,
                    "raw_value": text,
                    "rendered": bool(lines),
                    "word_ids": [],
                    "bbox_pdf_points": None,
                },
            )
            if field["raw_value"] != text:
                raise RenderError(f"field rendered with conflicting values: {field_id}")
            field["word_ids"].extend(
                word["word_id"]
                for word in self.words[-sum(len(line) for line in lines) :]
                if word.get("field_id") == field_id
            )
            field_boxes = [
                word["bbox_pdf_points"] for word in self.words if word.get("field_id") == field_id
            ]
            field["bbox_pdf_points"] = _union_bbox(field_boxes)
        return boxes

    def record_empty_field(self, field_id: str) -> None:
        self.fields[field_id] = {
            "field_id": field_id,
            "raw_value": "",
            "rendered": False,
            "word_ids": [],
            "bbox_pdf_points": None,
        }


def _draw_field_grid(
    pdf: Any,
    recorder: LayoutRecorder,
    document_type: str,
    record: dict[str, Any],
    *,
    start_top: float,
) -> float:
    simple_fields = [
        (key, value) for key, value in record.items() if key not in TABLE_FIELDS[document_type]
    ]
    if not all(isinstance(value, str) for _, value in simple_fields):
        raise RenderError(f"{document_type} simple fields must all be strings")
    left = 28.0
    total_width = recorder.page_width - 56.0
    gap = 6.0
    cell_width = (total_width - gap) / 2
    cell_height = 27.0 if document_type == "bill_of_lading" else 34.0
    row_count = (len(simple_fields) + 1) // 2
    for row_index in range(row_count):
        top = start_top - row_index * cell_height
        bottom = top - cell_height + 2
        for column_index in range(2):
            item_index = row_index * 2 + column_index
            if item_index >= len(simple_fields):
                continue
            key, value = simple_fields[item_index]
            x = left + column_index * (cell_width + gap)
            pdf.rect(x, bottom, cell_width, cell_height - 2, stroke=1, fill=0)
            recorder.draw_text(
                pdf,
                text=key.upper(),
                x=x + 3,
                first_baseline_y=top - 7,
                max_width=cell_width - 6,
                font_name=FONT_BOLD,
                font_size=4.8,
                role="field_label",
            )
            field_id = f"{document_type}:/{_escape_pointer(key)}"
            if value:
                recorder.draw_text(
                    pdf,
                    text=value,
                    x=x + 3,
                    first_baseline_y=top - 17,
                    max_width=cell_width - 6,
                    font_name=FONT_REGULAR,
                    font_size=6.2,
                    role="field_value",
                    field_id=field_id,
                    line_gap=1.4,
                )
            else:
                recorder.record_empty_field(field_id)
    return start_top - row_count * cell_height


def _draw_table(
    pdf: Any,
    recorder: LayoutRecorder,
    *,
    table_id: str,
    top: float,
    columns: list[tuple[str, float]],
    rows: list[list[tuple[str, str]]],
    row_height: float,
) -> float:
    left = 28.0
    header_height = 18.0
    total_width = sum(width for _, width in columns)
    header_bottom = top - header_height
    table_rows: list[dict[str, Any]] = []
    header_cells = []
    cursor = left
    for label, width in columns:
        pdf.rect(cursor, header_bottom, width, header_height, stroke=1, fill=0)
        recorder.draw_text(
            pdf,
            text=label,
            x=cursor + 2,
            first_baseline_y=top - 12,
            max_width=width - 4,
            font_name=FONT_BOLD,
            font_size=5.5,
            role="table_header",
        )
        header_cells.append(
            {
                "column_label": label,
                "field_id": None,
                "bbox_pdf_points": _round_bbox([cursor, header_bottom, cursor + width, top]),
            }
        )
        cursor += width
    table_rows.append(
        {
            "row_index": 0,
            "role": "header",
            "bbox_pdf_points": _round_bbox([left, header_bottom, left + total_width, top]),
            "cells": header_cells,
        }
    )

    current_top = header_bottom
    for row_index, row in enumerate(rows, start=1):
        if len(row) != len(columns):
            raise RenderError(f"table {table_id} row width does not match columns")
        bottom = current_top - row_height
        cursor = left
        cells = []
        for (label, width), (field_id, value) in zip(columns, row, strict=True):
            pdf.rect(cursor, bottom, width, row_height, stroke=1, fill=0)
            if value:
                recorder.draw_text(
                    pdf,
                    text=value,
                    x=cursor + 2,
                    first_baseline_y=current_top - 10,
                    max_width=width - 4,
                    font_name=FONT_REGULAR,
                    font_size=5.8,
                    role="table_value",
                    field_id=field_id,
                    line_gap=1.2,
                )
            else:
                recorder.record_empty_field(field_id)
            cells.append(
                {
                    "column_label": label,
                    "field_id": field_id,
                    "bbox_pdf_points": _round_bbox([cursor, bottom, cursor + width, current_top]),
                }
            )
            cursor += width
        table_rows.append(
            {
                "row_index": row_index,
                "role": "data",
                "bbox_pdf_points": _round_bbox([left, bottom, left + total_width, current_top]),
                "cells": cells,
            }
        )
        current_top = bottom
    recorder.tables.append(
        {
            "table_id": table_id,
            "bbox_pdf_points": _round_bbox([left, current_top, left + total_width, top]),
            "rows": table_rows,
        }
    )
    return current_top


def _draw_document_content(
    pdf: Any,
    recorder: LayoutRecorder,
    document_type: str,
    record: dict[str, Any],
) -> None:
    title = DOCUMENT_TITLES[document_type]
    recorder.draw_text(
        pdf,
        text=title,
        x=28,
        first_baseline_y=804,
        max_width=recorder.page_width - 56,
        font_name=FONT_BOLD,
        font_size=16,
        role="document_title",
    )
    recorder.draw_text(
        pdf,
        text="SYNTHETIC EVALUATION DOCUMENT - SOURCE VALUES RENDERED WITHOUT CORRECTION",
        x=28,
        first_baseline_y=786,
        max_width=recorder.page_width - 56,
        font_name=FONT_REGULAR,
        font_size=6,
        role="static_disclaimer",
    )
    grid_bottom = _draw_field_grid(
        pdf,
        recorder,
        document_type,
        record,
        start_top=771,
    )
    table_top = grid_bottom - 9
    full_width = recorder.page_width - 56

    if document_type == "commercial_invoice":
        columns = [
            ("DESCRIPTION", full_width * 0.46),
            ("QUANTITY", full_width * 0.16),
            ("UNIT PRICE", full_width * 0.18),
            ("AMOUNT", full_width * 0.20),
        ]
        rows = [
            [
                (
                    "commercial_invoice:/Goods Description",
                    record["Goods Description"],
                ),
                ("commercial_invoice:/Quantity", record["Quantity"]),
                ("commercial_invoice:/Unit Price", record["Unit Price"]),
                ("commercial_invoice:/Amount", record["Amount"]),
            ]
        ]
        _draw_table(
            pdf,
            recorder,
            table_id="commercial_invoice:line_items",
            top=table_top,
            columns=columns,
            rows=rows,
            row_height=38,
        )
    elif document_type == "packing_list":
        columns = [
            ("ITEM", full_width * 0.54),
            ("QUANTITY", full_width * 0.22),
            ("WEIGHT", full_width * 0.24),
        ]
        rows = []
        for index, item in enumerate(record["Goods Details"]):
            rows.append(
                [
                    (
                        f"packing_list:/Goods Details/{index}/item",
                        item["item"],
                    ),
                    (
                        f"packing_list:/Goods Details/{index}/quantity",
                        item["quantity"],
                    ),
                    (
                        f"packing_list:/Goods Details/{index}/weight",
                        item["weight"],
                    ),
                ]
            )
        _draw_table(
            pdf,
            recorder,
            table_id="packing_list:goods_details",
            top=table_top,
            columns=columns,
            rows=rows,
            row_height=31,
        )
    elif document_type == "bill_of_lading":
        columns = [
            ("PACKAGES", full_width * 0.24),
            ("DESCRIPTION", full_width * 0.42),
            ("GROSS WEIGHT", full_width * 0.18),
            ("REVENUE TONS", full_width * 0.16),
        ]
        rows = [
            [
                (
                    "bill_of_lading:/No. & Kinds of Containers or Packages",
                    record["No. & Kinds of Containers or Packages"],
                ),
                (
                    "bill_of_lading:/Description of Goods",
                    record["Description of Goods"],
                ),
                ("bill_of_lading:/Gross Weight", record["Gross Weight"]),
                ("bill_of_lading:/Revenue Tons", record["Revenue Tons"]),
            ]
        ]
        _draw_table(
            pdf,
            recorder,
            table_id="bill_of_lading:cargo",
            top=table_top,
            columns=columns,
            rows=rows,
            row_height=48,
        )
    elif document_type == "certificate_of_origin":
        columns = [
            ("ITEM", full_width * 0.40),
            ("QUANTITY", full_width * 0.18),
            ("SYNTHETIC HS", full_width * 0.18),
            ("ORIGIN CRITERION", full_width * 0.24),
        ]
        rows = []
        for index, item in enumerate(record["Goods Details"]):
            rows.append(
                [
                    (
                        f"certificate_of_origin:/Goods Details/{index}/item",
                        item["item"],
                    ),
                    (
                        f"certificate_of_origin:/Goods Details/{index}/quantity",
                        item["quantity"],
                    ),
                    (
                        f"certificate_of_origin:/Goods Details/{index}/hs_code",
                        item["hs_code"],
                    ),
                    (
                        f"certificate_of_origin:/Goods Details/{index}/origin_criterion",
                        item["origin_criterion"],
                    ),
                ]
            )
        _draw_table(
            pdf,
            recorder,
            table_id="certificate_of_origin:goods_details",
            top=table_top,
            columns=columns,
            rows=rows,
            row_height=42,
        )
    else:
        raise RenderError(f"unsupported document type: {document_type}")


def _pdf_to_png(
    pdf_bytes: bytes,
    pdfium: Any,
    *,
    dpi: int,
) -> tuple[bytes, int, int]:
    document = pdfium.PdfDocument(pdf_bytes)
    try:
        if len(document) != 1:
            raise RenderError("rendered fixture PDF must contain exactly one page")
        page = document[0]
        try:
            bitmap = page.render(scale=dpi / 72)
            try:
                image = bitmap.to_pil().convert("RGB")
                output = io.BytesIO()
                image.save(output, format="PNG", compress_level=9, optimize=False)
                return output.getvalue(), image.width, image.height
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()


def _visual_tokens(width: int, height: int) -> int:
    return ((width + 27) // 28) * ((height + 27) // 28)


def _largest_claude_dimensions(width: int, height: int) -> tuple[int, int]:
    maximum_scale = min(
        1.0,
        MAX_IMAGE_LONG_EDGE / max(width, height),
    )
    low = 0.0
    high = maximum_scale
    for _ in range(64):
        scale = (low + high) / 2
        target_width = max(1, int(width * scale))
        target_height = max(1, int(height * scale))
        if (
            max(target_width, target_height) <= MAX_IMAGE_LONG_EDGE
            and _visual_tokens(target_width, target_height) <= MAX_IMAGE_VISUAL_TOKENS
        ):
            low = scale
        else:
            high = scale
    return max(1, int(width * low)), max(1, int(height * low))


def _claude_input_png(
    render_png: bytes,
) -> tuple[bytes, dict[str, Any]]:
    from PIL import Image

    with Image.open(io.BytesIO(render_png)) as source:
        source.load()
        original = source.convert("RGB")
    width, height = original.size
    target_width, target_height = _largest_claude_dimensions(width, height)
    reasons = []
    if max(width, height) > MAX_IMAGE_LONG_EDGE:
        reasons.append("long_edge_limit")
    if _visual_tokens(width, height) > MAX_IMAGE_VISUAL_TOKENS:
        reasons.append("visual_token_limit")

    while True:
        if (target_width, target_height) == (width, height):
            candidate = original
        else:
            candidate = original.resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
        output = io.BytesIO()
        candidate.save(output, format="PNG", compress_level=9, optimize=False)
        encoded = output.getvalue()
        if len(encoded) <= MAX_IMAGE_BYTES:
            break
        reasons.append("byte_limit")
        target_width = max(1, int(target_width * 0.9))
        target_height = max(1, int(target_height * 0.9))
        if target_width == 1 and target_height == 1:
            raise RenderError("Claude input cannot satisfy the locked image byte limit")

    scale_x = target_width / width
    scale_y = target_height / height
    return encoded, {
        "resized": (target_width, target_height) != (width, height),
        "reasons": sorted(set(reasons)),
        "source_width_px": width,
        "source_height_px": height,
        "target_width_px": target_width,
        "target_height_px": target_height,
        "scale_x": round(scale_x, 12),
        "scale_y": round(scale_y, 12),
        "source_visual_tokens": _visual_tokens(width, height),
        "target_visual_tokens": _visual_tokens(target_width, target_height),
        "target_byte_length": len(encoded),
        "limits": {
            "max_long_edge_px": MAX_IMAGE_LONG_EDGE,
            "max_visual_tokens": MAX_IMAGE_VISUAL_TOKENS,
            "max_encoded_bytes": MAX_IMAGE_BYTES,
        },
        "resize_filter": "Pillow.LANCZOS",
        "integer_dimension_policy": "floor_after_uniform_scale_search",
    }


def _to_image_bbox(
    bbox: list[float] | None,
    *,
    page_height: float,
    scale: float,
) -> list[float] | None:
    if bbox is None:
        return None
    return _round_bbox(
        [
            bbox[0] * scale,
            (page_height - bbox[3]) * scale,
            bbox[2] * scale,
            (page_height - bbox[1]) * scale,
        ]
    )


def _attach_image_bboxes(
    layout: dict[str, Any],
    *,
    page_height: float,
    render_scale: float,
    claude_scale_x: float,
    claude_scale_y: float,
) -> None:
    def claude_bbox(render_bbox: list[float] | None) -> list[float] | None:
        if render_bbox is None:
            return None
        return _round_bbox(
            [
                render_bbox[0] * claude_scale_x,
                render_bbox[1] * claude_scale_y,
                render_bbox[2] * claude_scale_x,
                render_bbox[3] * claude_scale_y,
            ]
        )

    for word in layout["words"]:
        word["bbox_image_px"] = _to_image_bbox(
            word["bbox_pdf_points"], page_height=page_height, scale=render_scale
        )
        word["bbox_claude_input_px"] = claude_bbox(word["bbox_image_px"])
    for field in layout["fields"]:
        field["bbox_image_px"] = _to_image_bbox(
            field["bbox_pdf_points"], page_height=page_height, scale=render_scale
        )
        field["bbox_claude_input_px"] = claude_bbox(field["bbox_image_px"])
    for table in layout["tables"]:
        table["bbox_image_px"] = _to_image_bbox(
            table["bbox_pdf_points"], page_height=page_height, scale=render_scale
        )
        table["bbox_claude_input_px"] = claude_bbox(table["bbox_image_px"])
        for row in table["rows"]:
            row["bbox_image_px"] = _to_image_bbox(
                row["bbox_pdf_points"], page_height=page_height, scale=render_scale
            )
            row["bbox_claude_input_px"] = claude_bbox(row["bbox_image_px"])
            for cell in row["cells"]:
                cell["bbox_image_px"] = _to_image_bbox(
                    cell["bbox_pdf_points"],
                    page_height=page_height,
                    scale=render_scale,
                )
                cell["bbox_claude_input_px"] = claude_bbox(cell["bbox_image_px"])


def render_document(
    *,
    document_id: str,
    case_id: str,
    document_type: str,
    record: dict[str, Any],
    source_record_sha256: str,
    dpi: int = RENDER_DPI,
    dependency_importer: Callable[[str], Any] = importlib.import_module,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    """Render one document to deterministic PDF/PNG plus layout Gold."""

    modules = require_render_dependencies(dependency_importer)
    page_width, page_height = modules["pagesizes"].A4
    output = io.BytesIO()
    pdf = modules["canvas"].Canvas(
        output,
        pagesize=(page_width, page_height),
        invariant=1,
        pageCompression=1,
    )
    pdf.setTitle(DOCUMENT_TITLES[document_type])
    pdf.setAuthor("customs-ai-agent deterministic fixture renderer")
    pdf.setCreator(TEMPLATE_VERSION)
    pdf.setLineWidth(0.45)
    pdf.rect(20, 20, page_width - 40, page_height - 40, stroke=1, fill=0)
    recorder = LayoutRecorder(modules["pdfmetrics"], page_width, page_height)
    _draw_document_content(pdf, recorder, document_type, record)
    pdf.showPage()
    pdf.save()
    pdf_bytes = output.getvalue()
    png_bytes, image_width, image_height = _pdf_to_png(
        pdf_bytes,
        modules["pypdfium2"],
        dpi=dpi,
    )
    claude_png_bytes, claude_transform = _claude_input_png(png_bytes)
    render_scale = dpi / 72
    claude_scale_x = claude_transform["scale_x"]
    claude_scale_y = claude_transform["scale_y"]
    pdf_to_render_matrix = [
        [round(render_scale, 12), 0.0, 0.0],
        [0.0, round(-render_scale, 12), round(float(page_height) * render_scale, 12)],
        [0.0, 0.0, 1.0],
    ]
    render_to_claude_matrix = [
        [claude_scale_x, 0.0, 0.0],
        [0.0, claude_scale_y, 0.0],
        [0.0, 0.0, 1.0],
    ]
    layout = {
        "schema_version": LAYOUT_SCHEMA,
        "document_id": document_id,
        "case_id": case_id,
        "document_type": document_type,
        "template_version": TEMPLATE_VERSION,
        "source_record_sha256": source_record_sha256,
        "font_assets": modules["font_assets"],
        "font_license": modules["font_license"],
        "page": {
            "page_number": 1,
            "width_pdf_points": round(float(page_width), 3),
            "height_pdf_points": round(float(page_height), 3),
            "width_image_px": image_width,
            "height_image_px": image_height,
            "render_dpi": dpi,
            "pdf_coordinate_system": "bottom_left_origin_points",
            "image_coordinate_system": "top_left_origin_pixels",
            "claude_input_width_px": claude_transform["target_width_px"],
            "claude_input_height_px": claude_transform["target_height_px"],
            "claude_input_coordinate_system": "top_left_origin_pixels",
            "claude_input_visual_tokens": claude_transform["target_visual_tokens"],
        },
        "coordinate_transform_chain": [
            {
                "transform_id": "pdf_points_to_render_image_pixels",
                "from_space": "pdf_points_bottom_left",
                "to_space": "render_image_pixels_top_left",
                "matrix_3x3": pdf_to_render_matrix,
                "formula": "x'=x*dpi/72; y'=(page_height-y)*dpi/72",
            },
            {
                "transform_id": "render_image_pixels_to_claude_input_pixels",
                "from_space": "render_image_pixels_top_left",
                "to_space": "claude_input_pixels_top_left",
                "matrix_3x3": render_to_claude_matrix,
                "formula": "x'=x*scale_x; y'=y*scale_y",
                "resize": claude_transform,
            },
        ],
        "fields": sorted(recorder.fields.values(), key=lambda item: item["field_id"]),
        "words": recorder.words,
        "tables": recorder.tables,
        "files": {
            "pdf_sha256": _sha256_bytes(pdf_bytes),
            "png_sha256": _sha256_bytes(png_bytes),
            "claude_input_png_sha256": _sha256_bytes(claude_png_bytes),
        },
    }
    _attach_image_bboxes(
        layout,
        page_height=float(page_height),
        render_scale=render_scale,
        claude_scale_x=claude_scale_x,
        claude_scale_y=claude_scale_y,
    )
    return pdf_bytes, png_bytes, claude_png_bytes, layout


def render_dataset(
    dataset_root: Path,
    evaluation_root: Path,
    output_root: Path,
    *,
    max_cases: int | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Render registered triplets and controlled C/O fixtures to artifacts."""

    source = dataset_root.resolve()
    evaluation = evaluation_root.resolve()
    output = output_root.resolve()
    raw_root = (source / "raw").resolve()
    if output == raw_root or output.is_relative_to(raw_root):
        raise RenderError("render output must never be written inside registered raw")
    cases = _load_jsonl(source / "derived" / "cases.jsonl")
    certificates = _load_jsonl(evaluation / "certificate-of-origin" / "cases.jsonl")
    if max_cases is not None:
        if max_cases < 1:
            raise RenderError("max_cases must be positive")
        cases = cases[:max_cases]
        certificates = certificates[:max_cases]

    file_records: list[dict[str, Any]] = []
    document_count = 0
    claude_resized_count = 0

    def store(relative_path: str, content: bytes) -> None:
        _write_generated(output / relative_path, content, refresh=refresh)
        file_records.append(
            {
                "relative_path": relative_path,
                "sha256": _sha256_bytes(content),
                "byte_size": len(content),
            }
        )

    for case in cases:
        case_id = case["case_id"]
        for document_type in (
            "commercial_invoice",
            "packing_list",
            "bill_of_lading",
        ):
            source_document = case["documents"][document_type]
            record = source_document["raw_record"]
            record_hash = _sha256_bytes(_canonical_bytes(record))
            document_id = f"{case_id}:{document_type}"
            pdf_bytes, png_bytes, claude_png_bytes, layout = render_document(
                document_id=document_id,
                case_id=case_id,
                document_type=document_type,
                record=record,
                source_record_sha256=record_hash,
            )
            stem = f"documents/{case_id}/{document_type}"
            layout["files"]["pdf_relative_path"] = f"{stem}.pdf"
            layout["files"]["png_relative_path"] = f"{stem}.png"
            layout["files"]["claude_input_png_relative_path"] = f"{stem}.claude.png"
            store(f"{stem}.pdf", pdf_bytes)
            store(f"{stem}.png", png_bytes)
            store(f"{stem}.claude.png", claude_png_bytes)
            store(f"{stem}.layout-gold.json", _json_bytes(layout))
            if layout["coordinate_transform_chain"][1]["resize"]["resized"]:
                claude_resized_count += 1
            document_count += 1

    for fixture in certificates:
        fixture_id = fixture["fixture_id"]
        record = fixture["record"]
        record_hash = _sha256_bytes(_canonical_bytes(record))
        pdf_bytes, png_bytes, claude_png_bytes, layout = render_document(
            document_id=f"{fixture_id}:certificate_of_origin",
            case_id=fixture_id,
            document_type="certificate_of_origin",
            record=record,
            source_record_sha256=record_hash,
        )
        stem = f"certificate-of-origin/{fixture_id}/certificate_of_origin"
        layout["files"]["pdf_relative_path"] = f"{stem}.pdf"
        layout["files"]["png_relative_path"] = f"{stem}.png"
        layout["files"]["claude_input_png_relative_path"] = f"{stem}.claude.png"
        store(f"{stem}.pdf", pdf_bytes)
        store(f"{stem}.png", png_bytes)
        store(f"{stem}.claude.png", claude_png_bytes)
        store(f"{stem}.layout-gold.json", _json_bytes(layout))
        if layout["coordinate_transform_chain"][1]["resize"]["resized"]:
            claude_resized_count += 1
        document_count += 1

    dependency_versions = {name: importlib.metadata.version(name) for name in REQUIRED_DEPENDENCIES}
    dependency_modules = require_render_dependencies()
    manifest = {
        "schema_version": RENDER_SCHEMA,
        "dataset_id": "trade_document_triplets",
        "dataset_version": "source-v1",
        "template_version": TEMPLATE_VERSION,
        "render_dpi": RENDER_DPI,
        "coordinate_transform_contract": {
            "gold_canonical_space": "pdf_points_bottom_left",
            "render_space": "render_image_pixels_top_left",
            "claude_input_space": "claude_input_pixels_top_left",
            "chain_recorded_per_document": True,
            "bbox_keys": {
                "pdf": "bbox_pdf_points",
                "render_image": "bbox_image_px",
                "claude_input": "bbox_claude_input_px",
            },
        },
        "claude_input_policy": {
            "preserve_aspect_ratio": True,
            "max_long_edge_px": MAX_IMAGE_LONG_EDGE,
            "max_visual_tokens": MAX_IMAGE_VISUAL_TOKENS,
            "max_encoded_bytes": MAX_IMAGE_BYTES,
            "resize_filter": "Pillow.LANCZOS",
        },
        "dependency_versions": dependency_versions,
        "font_assets": dependency_modules["font_assets"],
        "font_license": dependency_modules["font_license"],
        "source_sha256": {
            "registered_cases": _sha256_bytes((source / "derived" / "cases.jsonl").read_bytes()),
            "certificate_of_origin_cases": _sha256_bytes(
                (evaluation / "certificate-of-origin" / "cases.jsonl").read_bytes()
            ),
        },
        "counts": {
            "source_case_count": len(cases),
            "triplet_document_count": len(cases) * 3,
            "certificate_of_origin_document_count": len(certificates),
            "document_count": document_count,
            "generated_document_file_count": len(file_records),
            "claude_input_resized_document_count": claude_resized_count,
        },
        "generated_files": sorted(file_records, key=lambda item: item["relative_path"]),
    }
    manifest_bytes = _json_bytes(manifest)
    store("manifest.json", manifest_bytes)
    checksums = "".join(
        f"{item['sha256']}  {item['relative_path']}\n"
        for item in sorted(file_records, key=lambda item: item["relative_path"])
    ).encode()
    _write_generated(output / "checksums.sha256", checksums, refresh=refresh)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed CI/PL/B/L/C/O PDF+PNG fixtures and layout Gold."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-lock", type=Path, default=DEFAULT_DATASET_LOCK)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        verify_dataset_provenance(
            args.dataset_lock,
            expected_dataset_root=args.dataset_root,
        )
        manifest = render_dataset(
            args.dataset_root,
            args.evaluation_root,
            args.output_root,
            max_cases=args.max_cases,
            refresh=args.refresh,
        )
    except DatasetProvenanceError as exc:
        print(f"document render failed: {exc.code}")
        return 1
    except RenderError as exc:
        print(f"document render failed: {exc}")
        return 1
    print(
        "document render passed: "
        f"{manifest['counts']['document_count']} documents, "
        f"{manifest['counts']['generated_document_file_count']} PDF/PNG/Gold files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
