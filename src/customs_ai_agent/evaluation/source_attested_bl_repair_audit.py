"""Independent source-attested B/L audit for post-sealed OCR repairs.

The original 1,000-family synthetic corpus has no unused families left after
training and evaluation.  A repair selected after observing that corpus must
therefore pass this separately generated audit before it can be activated.

This module does not copy or edit an existing PDF.  It generates a small
deterministic B/L form from a fixed recipe, re-parses the *final PDF content
stream*, and requires a one-to-one PDFium character/geometry match before any
raster or Gold file is emitted.  Generator values are never passed into the
Gold compiler, and the receipt contains hashes only, never field text.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
from typing import Final, Literal, Mapping, Sequence, cast
import uuid
import zlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.evaluation.local_process_lock import exclusive_local_process_lock
from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    _parse_pdf_content_stream_v1,
)
from customs_ai_agent.evaluation.synthetic_raster import (
    PdfiumFinalPdfGeometryExtractorV1,
    RASTER_PROFILES,
    RasterGoldArtifactV1,
    RasterProfileId,
    RenderedRasterBytesV1,
    SourceFieldGoldV1,
    SourcePdfGeometryV1,
    SourceTextGeometryV1,
    SyntheticRasterPdfiumCompiler,
    _final_pdf_text_run_decoder,
    compile_raster_field_gold,
    compile_raster_gold,
    verify_rendered_raster_output,
    write_immutable_rendered_raster_output,
)
from customs_ai_agent.evaluation.synthetic_raster_text import (
    RasterTextEvaluationCaseV1,
    RasterTextMetricCountsV1,
    canonicalize_synthetic_raster_semantic_text_v3,
    evaluate_synthetic_raster_semantic_text_case_v3,
    source_only_text_gold_from_raster_gold,
)
from customs_ai_agent.schemas.raster_text_extraction import RasterTextExtractionResultV1


_SHA256 = r"^[a-f0-9]{64}$"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
BL_REPAIR_AUDIT_DATASET_ID: Final = "synthetic-bl-repair-audit-v1"
BL_REPAIR_AUDIT_FAMILY_COUNT: Final = 100
BL_REPAIR_AUDIT_SOURCE_COUNT: Final = 200
BL_REPAIR_AUDIT_RASTER_COUNT: Final = 800
BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH: Final = f"{BL_REPAIR_AUDIT_DATASET_ID}/receipt.json"
BL_REPAIR_AUDIT_GENERATOR_REVISION: Final = "independent-bl-form-generator-v1"
BL_REPAIR_AUDIT_PARSER_REVISION: Final = "independent-bl-content-stream-parser-v1"
_GENERATION_SEED: Final = b"task-0067-independent-bl-repair-audit-seed-v1\n"
_GENERATION_SEED_SHA256: Final = hashlib.sha256(_GENERATION_SEED).hexdigest()
_RASTER_SUFFIX: Mapping[RasterProfileId, str] = {
    "png-216": ".png",
    "jpeg-216-q75": ".jpg",
    "jpeg-150-q75": ".jpg",
    "scan-pdf-216-q75": ".pdf",
}
_MEDIA_TYPE_BY_PROFILE: Mapping[RasterProfileId, str] = {
    profile_id: profile.media_type for profile_id, profile in RASTER_PROFILES.items()
}


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class _RunSpec:
    x: float
    y: float
    font_size: float
    text: str | None = None
    field_id: str | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.field_id is None):
            raise ValueError("B/L audit run must be static text or one source field")


# The voyage value deliberately retains the original B/L position and 7 pt
# font size.  Other source fields cover the same 13-field surface used by the
# production text Gate, so a targeted retry cannot pass by dropping content.
_RUN_RECIPE: Final[tuple[_RunSpec, ...]] = (
    _RunSpec(254.7003, 796.8898, 15.0, text="Bill of Lading"),
    _RunSpec(42.0, 764.8898, 6.0, text="Shipper/Exporter"),
    _RunSpec(44.0, 755.8898, 7.5, field_id="exporter_name"),
    _RunSpec(322.0, 764.8898, 6.0, text="B/L No."),
    _RunSpec(324.0, 755.8898, 7.5, field_id="bl_no"),
    _RunSpec(42.0, 718.8898, 6.0, text="Consignee"),
    _RunSpec(44.0, 709.8898, 7.5, field_id="consignee"),
    _RunSpec(42.0, 620.8898, 6.0, text="Ocean Vessel"),
    _RunSpec(44.0, 612.8898, 7.0, field_id="vessel"),
    _RunSpec(182.0, 620.8898, 6.0, text="Voyage No."),
    _RunSpec(184.0, 612.8898, 7.0, field_id="voyage"),
    _RunSpec(42.0, 596.8898, 6.0, text="Port of Loading"),
    _RunSpec(44.0, 585.8898, 6.5, field_id="load_port"),
    _RunSpec(170.8189, 596.8898, 6.0, text="Port of Discharge"),
    _RunSpec(172.8189, 585.8898, 6.5, field_id="disch_port"),
    _RunSpec(428.4567, 596.8898, 6.0, text="Final Destination"),
    _RunSpec(430.4567, 585.8898, 6.5, field_id="final_dest"),
    _RunSpec(42.0, 561.8898, 5.4, text="Container No."),
    _RunSpec(120.0, 561.8898, 5.4, text="Seal No"),
    _RunSpec(192.0, 561.8898, 5.4, text="No. of Containers"),
    _RunSpec(277.0, 561.8898, 5.4, text="Description of Goods"),
    _RunSpec(402.0, 561.8898, 5.4, text="Gross Weight"),
    _RunSpec(43.0, 537.8898, 6.5, field_id="container_no"),
    _RunSpec(121.0, 537.8898, 6.0, field_id="seal_no"),
    _RunSpec(193.0, 537.8898, 6.5, field_id="containers"),
    _RunSpec(278.0, 537.8898, 6.5, field_id="good"),
    _RunSpec(403.0, 537.8898, 6.5, field_id="gross_weight"),
)

_RECTANGLES: Final[tuple[tuple[float, float, float, float], ...]] = (
    (40.0, 725.8898, 280.0, 46.0),
    (320.0, 725.8898, 235.2756, 46.0),
    (40.0, 691.8898, 515.2756, 34.0),
    (40.0, 603.8898, 140.0, 24.0),
    (180.0, 603.8898, 140.0, 24.0),
    (40.0, 569.8898, 128.8189, 34.0),
    (168.8189, 569.8898, 128.8189, 34.0),
    (426.4567, 569.8898, 128.8189, 34.0),
    (40.0, 549.8898, 78.0, 20.0),
    (118.0, 549.8898, 72.0, 20.0),
    (190.0, 549.8898, 85.0, 20.0),
    (275.0, 549.8898, 125.0, 20.0),
    (400.0, 549.8898, 155.2756, 20.0),
    (40.0, 453.8898, 78.0, 96.0),
    (118.0, 453.8898, 72.0, 96.0),
    (190.0, 453.8898, 85.0, 96.0),
    (275.0, 453.8898, 125.0, 96.0),
    (400.0, 453.8898, 155.2756, 96.0),
)

_FIELD_PATTERNS: Final[Mapping[str, re.Pattern[str]]] = {
    "exporter_name": re.compile(r"^[A-Z][A-Z .&-]{4,48}$"),
    "consignee": re.compile(r"^[A-Z][A-Z .&-]{4,48}$"),
    "load_port": re.compile(r"^[A-Z][A-Z ,.-]{4,48}$"),
    "disch_port": re.compile(r"^[A-Z][A-Z ,.-]{4,48}$"),
    "final_dest": re.compile(r"^[A-Z][A-Z ,.-]{4,48}$"),
    "vessel": re.compile(r"^[A-Z][A-Z 0-9.-]{3,32}$"),
    "voyage": re.compile(r"^[0-9]{4}[EWNS]$"),
    "bl_no": re.compile(r"^RA[0-9]{4}[A-Z]{2}$"),
    "container_no": re.compile(r"^[A-Z]{4}[0-9]{7}$"),
    "seal_no": re.compile(r"^RS[0-9]{6}$"),
    "good": re.compile(r"^[A-Z][A-Z 0-9'&.-]{4,48}$"),
    "gross_weight": re.compile(r"^[0-9]{1,3}(?:,[0-9]{3})?\.[0-9] KGS$"),
    "containers": re.compile(r"^[1-9] CNTR$"),
}

_RECIPE_PAYLOAD: Final = {
    "schema_version": "1.0",
    "generator_revision": BL_REPAIR_AUDIT_GENERATOR_REVISION,
    "parser_revision": BL_REPAIR_AUDIT_PARSER_REVISION,
    "page_points": [595.2756, 841.8898],
    "font": {"base_font": "Helvetica", "encoding": "WinAnsiEncoding"},
    "runs": [
        {
            "x": item.x,
            "y": item.y,
            "font_size": item.font_size,
            "text": item.text,
            "field_id": item.field_id,
        }
        for item in _RUN_RECIPE
    ],
    "rectangles": [list(item) for item in _RECTANGLES],
    "profiles": list(RASTER_PROFILES),
    "jpeg_quality": 75,
    "jpeg_subsampling": 0,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


BL_REPAIR_AUDIT_RECIPE_SHA256: Final = hashlib.sha256(
    _canonical_json_bytes(_RECIPE_PAYLOAD)
).hexdigest()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _asset_id(family_index: int, fixture_kind: Literal["clean", "error"]) -> str:
    if family_index < 0 or family_index >= BL_REPAIR_AUDIT_FAMILY_COUNT:
        raise ValueError("B/L repair audit family index is invalid")
    return f"blra-{family_index:03d}-{fixture_kind}"


def _family_id(family_index: int) -> str:
    if family_index < 0 or family_index >= BL_REPAIR_AUDIT_FAMILY_COUNT:
        raise ValueError("B/L repair audit family index is invalid")
    return f"blra-family-{family_index:03d}"


def _derived_int(*parts: str) -> int:
    digest = hashlib.sha256(_GENERATION_SEED + ":".join(parts).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _pick(values: Sequence[str], *, family_index: int, field_id: str) -> str:
    return values[_derived_int(str(family_index), field_id) % len(values)]


def _generated_field_values(
    *, family_index: int, fixture_kind: Literal["clean", "error"]
) -> Mapping[str, str]:
    """Return renderer inputs only; the Gold parser never calls this function."""

    directions = ("E", "W", "N", "S")
    voyage_number = 6100 + family_index
    container_number = 2_000_000 + family_index * 37
    gross_tenths = 82_000 + family_index * 113
    # Each error sibling changes exactly one visible field.  Family grouping
    # prevents the paired layout/value surface from being split later.
    if fixture_kind == "error":
        mutation = family_index % 3
        if mutation == 0:
            voyage_number += 1_000
        elif mutation == 1:
            container_number += 1_000_003
        else:
            gross_tenths += 7_003

    exporter_prefix = _pick(
        ("ARCTIC", "BLUE", "CROWN", "DELTA", "EASTERN", "HARBOR", "NOVA", "ORBIT"),
        family_index=family_index,
        field_id="exporter-prefix",
    )
    exporter_suffix = _pick(
        ("MACHINERY", "TRADING", "MARITIME", "INDUSTRIES", "LOGISTICS"),
        family_index=family_index,
        field_id="exporter-suffix",
    )
    consignee = _pick(
        (
            "TO ORDER OF MERIDIAN BANK",
            "NORTH STAR IMPORTS LLC",
            "ATLANTIC MERCHANTS LTD.",
            "PACIFIC BUYERS INC.",
            "WESTERN COMMERCE CO.",
        ),
        family_index=family_index,
        field_id="consignee",
    )
    load_port = _pick(
        ("BUSAN, KOREA", "INCHEON, KOREA", "GWANGYANG, KOREA"),
        family_index=family_index,
        field_id="load-port",
    )
    destination = _pick(
        (
            "ROTTERDAM, NETHERLANDS",
            "HAMBURG, GERMANY",
            "LONG BEACH, USA",
            "SINGAPORE, SINGAPORE",
        ),
        family_index=family_index,
        field_id="destination",
    )
    vessel = _pick(
        ("OCEAN AURORA", "MERIDIAN STAR", "PACIFIC LANTERN", "BLUE HORIZON"),
        family_index=family_index,
        field_id="vessel",
    )
    goods = _pick(
        (
            "COTTON T-SHIRTS",
            "STEEL MACHINE PARTS",
            "POLYESTER FABRIC",
            "ELECTRIC MOTORS",
            "CERAMIC TABLEWARE",
        ),
        family_index=family_index,
        field_id="goods",
    )
    owner = _pick(
        ("RAAU", "RABU", "RACU", "RADU", "RAEU"),
        family_index=family_index,
        field_id="container-owner",
    )
    direction = directions[_derived_int(str(family_index), "voyage-direction") % len(directions)]
    gross_whole, gross_decimal = divmod(gross_tenths, 10)
    return {
        "exporter_name": f"{exporter_prefix} {exporter_suffix} CORP.",
        "consignee": consignee,
        "load_port": load_port,
        "disch_port": destination,
        "final_dest": destination,
        "vessel": vessel,
        "voyage": f"{voyage_number:04d}{direction}",
        "bl_no": f"RA{family_index:04d}{chr(65 + family_index // 26)}{chr(65 + family_index % 26)}",
        "container_no": f"{owner}{container_number:07d}",
        "seal_no": f"RS{300_000 + family_index * 71:06d}",
        "good": goods,
        "gross_weight": f"{gross_whole:,}.{gross_decimal} KGS",
        "containers": f"{1 + family_index % 5} CNTR",
    }


def _escape_pdf_literal(value: str) -> str:
    try:
        value.encode("cp1252")
    except UnicodeEncodeError:
        raise ValueError("B/L repair audit text is outside WinAnsi") from None
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_number(value: float) -> str:
    rendered = f"{value:.10f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _pdf_object(object_number: int, body: bytes) -> bytes:
    return f"{object_number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"


def _assemble_pdf(content: bytes) -> bytes:
    compressed = zlib.compress(content, level=9)
    # PDF's ASCII85Decode filter consumes the stream form, which ends in
    # ``~>`` but does not include PostScript's optional opening ``<~`` token.
    # Python's decoder used by the independent parser accepts that same form.
    encoded = base64.a85encode(compressed, adobe=True)[2:]
    objects = (
        _pdf_object(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        _pdf_object(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        _pdf_object(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.2756 841.8898] "
            b"/Rotate 0 /Resources << /Font << /F1 5 0 R >> "
            b"/ProcSet [/PDF /Text] >> /Contents 4 0 R >>",
        ),
        _pdf_object(
            4,
            f"<< /Length {len(encoded)} /Filter [/ASCII85Decode /FlateDecode] >>\n".encode("ascii")
            + b"stream\n"
            + encoded
            + b"\nendstream",
        ),
        _pdf_object(
            5,
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding /Name /F1 >>",
        ),
    )
    result = bytearray(b"%PDF-1.3\n%\x93\x8c\x8b\x9e\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(result))
        result.extend(item)
    xref_offset = len(result)
    result.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend(
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    return bytes(result)


def build_bl_repair_audit_source_pdf(
    *, family_index: int, fixture_kind: Literal["clean", "error"]
) -> bytes:
    """Generate one new deterministic PDF without reading an existing PDF."""

    values = _generated_field_values(family_index=family_index, fixture_kind=fixture_kind)
    commands = ["1 0 0 1 0 0 cm", ".6 w"]
    commands.extend(
        f"n {_pdf_number(x)} {_pdf_number(y)} {_pdf_number(width)} {_pdf_number(height)} re S"
        for x, y, width, height in _RECTANGLES
    )
    for run in _RUN_RECIPE:
        value = run.text if run.text is not None else values[run.field_id or ""]
        commands.append(
            "BT /F1 "
            f"{_pdf_number(run.font_size)} Tf 1 0 0 1 "
            f"{_pdf_number(run.x)} {_pdf_number(run.y)} Tm "
            f"({_escape_pdf_literal(value)}) Tj ET"
        )
    return _assemble_pdf(("\n".join(commands) + "\n").encode("cp1252"))


@dataclass(frozen=True, slots=True)
class BlRepairAuditContentStreamParserV1:
    """Map fixed positions to fields using only final PDF content bytes."""

    family_id: str

    def parse_final_pdf_content_stream(
        self, *, source_pdf: bytes, asset_id: str
    ) -> SourcePdfGeometryV1:
        parsed = _parse_pdf_content_stream_v1(
            source_pdf,
            text_run_decoder=_final_pdf_text_run_decoder(source_pdf),
        )
        if len(parsed.text_runs) != len(_RUN_RECIPE) or len(parsed.rectangles) != len(_RECTANGLES):
            raise ValueError("B/L repair audit final content-stream shape is invalid")
        fields: list[SourceFieldGoldV1] = []
        spans: list[SourceTextGeometryV1] = []
        for expected, observed in zip(_RUN_RECIPE, parsed.text_runs, strict=True):
            if abs(expected.x - observed.x) > 1e-6 or abs(expected.y - observed.y) > 1e-6:
                raise ValueError("B/L repair audit final content-stream position is invalid")
            if expected.text is not None and observed.text != expected.text:
                raise ValueError("B/L repair audit static source text is invalid")
            if expected.field_id is not None:
                pattern = _FIELD_PATTERNS[expected.field_id]
                if pattern.fullmatch(observed.text) is None:
                    raise ValueError("B/L repair audit source field is invalid")
                fields.append(
                    SourceFieldGoldV1(
                        field_id=expected.field_id,
                        value=observed.text,
                        evidence_span_ids=[f"span-{observed.reading_order}"],
                    )
                )
            spans.append(
                SourceTextGeometryV1(
                    span_id=f"span-{observed.reading_order}",
                    page_number=1,
                    text=observed.text,
                    content_stream_x=observed.x,
                    content_stream_y=observed.y,
                    reading_order=observed.reading_order,
                )
            )
        if set(item.field_id for item in fields) != set(_FIELD_PATTERNS):
            raise ValueError("B/L repair audit source field coverage is invalid")
        return SourcePdfGeometryV1(
            asset_id=asset_id,
            family_id=self.family_id,
            document_type="bill_of_lading",
            source_pdf_sha256=_hash_bytes(source_pdf),
            page_width_points=595.2756,
            page_height_points=841.8898,
            spans=spans,
            fields=fields,
            table_cells=[],
        )


class BlRepairAuditSourceBindingV1(_WireModel):
    asset_id: str = Field(pattern=_IDENTIFIER)
    family_id: str = Field(pattern=_IDENTIFIER)
    fixture_kind: Literal["clean", "error"]
    source_pdf_relative_path: str = Field(min_length=1, max_length=240)
    source_pdf_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _fixed_path_and_identity(self) -> "BlRepairAuditSourceBindingV1":
        expected = f"{BL_REPAIR_AUDIT_DATASET_ID}/sources/{self.asset_id}.pdf"
        if self.source_pdf_relative_path != expected:
            raise ValueError("B/L repair audit source path is invalid")
        return self


class BlRepairAuditRasterBindingV1(_WireModel):
    asset_id: str = Field(pattern=_IDENTIFIER)
    family_id: str = Field(pattern=_IDENTIFIER)
    fixture_kind: Literal["clean", "error"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    source_pdf_sha256: str = Field(pattern=_SHA256)
    image_sha256: str = Field(pattern=_SHA256)
    media_type: Literal["image/png", "image/jpeg", "application/pdf"]
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    asset_relative_path: str = Field(min_length=1, max_length=240)
    gold_relative_path: str = Field(min_length=1, max_length=240)
    gold_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _fixed_paths_and_media(self) -> "BlRepairAuditRasterBindingV1":
        suffix = _RASTER_SUFFIX[self.profile_id]
        expected_asset = (
            f"{BL_REPAIR_AUDIT_DATASET_ID}/assets/{self.asset_id}-{self.profile_id}{suffix}"
        )
        expected_gold = f"{BL_REPAIR_AUDIT_DATASET_ID}/gold/{self.asset_id}-{self.profile_id}.json"
        if (
            self.asset_relative_path != expected_asset
            or self.gold_relative_path != expected_gold
            or self.media_type != _MEDIA_TYPE_BY_PROFILE[self.profile_id]
        ):
            raise ValueError("B/L repair audit raster binding is invalid")
        return self


class BlRepairAuditReceiptV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_bl_repair_audit"] = "source_attested_bl_repair_audit"
    dataset_id: Literal["synthetic-bl-repair-audit-v1"] = BL_REPAIR_AUDIT_DATASET_ID
    generator_revision: Literal["independent-bl-form-generator-v1"] = (
        BL_REPAIR_AUDIT_GENERATOR_REVISION
    )
    parser_revision: Literal["independent-bl-content-stream-parser-v1"] = (
        BL_REPAIR_AUDIT_PARSER_REVISION
    )
    source_truth_kind: Literal["final_pdf_content_stream_pdfium_1_to_1"] = (
        "final_pdf_content_stream_pdfium_1_to_1"
    )
    generation_seed_sha256: str = Field(pattern=_SHA256)
    generation_recipe_sha256: str = Field(pattern=_SHA256)
    pdfium_version: str = Field(min_length=1, max_length=128)
    pillow_version: str = Field(min_length=1, max_length=128)
    sources: list[BlRepairAuditSourceBindingV1] = Field(
        min_length=BL_REPAIR_AUDIT_SOURCE_COUNT,
        max_length=BL_REPAIR_AUDIT_SOURCE_COUNT,
    )
    rasters: list[BlRepairAuditRasterBindingV1] = Field(
        min_length=BL_REPAIR_AUDIT_RASTER_COUNT,
        max_length=BL_REPAIR_AUDIT_RASTER_COUNT,
    )

    @model_validator(mode="after")
    def _complete_fixed_scope(self) -> "BlRepairAuditReceiptV1":
        expected_sources = {
            (_asset_id(index, kind), _family_id(index), kind)
            for index in range(BL_REPAIR_AUDIT_FAMILY_COUNT)
            for kind in ("clean", "error")
        }
        source_keys = {(item.asset_id, item.family_id, item.fixture_kind) for item in self.sources}
        raster_keys = {(item.asset_id, item.profile_id) for item in self.rasters}
        expected_rasters = {
            (asset_id, profile_id)
            for asset_id, _family, _kind in expected_sources
            for profile_id in RASTER_PROFILES
        }
        if (
            self.generation_seed_sha256 != _GENERATION_SEED_SHA256
            or self.generation_recipe_sha256 != BL_REPAIR_AUDIT_RECIPE_SHA256
            or source_keys != expected_sources
            or raster_keys != expected_rasters
            or len(source_keys) != len(self.sources)
            or len(raster_keys) != len(self.rasters)
            or self.sources != sorted(self.sources, key=lambda item: item.asset_id)
            or self.rasters
            != sorted(self.rasters, key=lambda item: (item.asset_id, item.profile_id))
        ):
            raise ValueError("B/L repair audit receipt coverage is invalid")
        source_by_asset = {item.asset_id: item for item in self.sources}
        if any(
            item.family_id != source_by_asset[item.asset_id].family_id
            or item.fixture_kind != source_by_asset[item.asset_id].fixture_kind
            or item.source_pdf_sha256 != source_by_asset[item.asset_id].source_pdf_sha256
            for item in self.rasters
        ):
            raise ValueError("B/L repair audit receipt binding is invalid")
        return self


class BlRepairAuditScoreV1(_WireModel):
    fixture_kind: Literal["clean", "error"]
    profile_id: Literal["png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"]
    character_errors: int = Field(ge=0)
    character_reference: int = Field(ge=1)
    critical_matches: int = Field(ge=0)
    critical_total: Literal[1300] = 1300
    voyage_matches: int = Field(ge=0, le=100)
    voyage_total: Literal[100] = 100
    character_error_rate: float = Field(ge=0)
    critical_value_exact_match: float = Field(ge=0, le=1)
    voyage_exact_match: float = Field(ge=0, le=1)
    gate_passed: bool

    @model_validator(mode="after")
    def _scores_and_gate_match_counts(self) -> "BlRepairAuditScoreV1":
        expected_cer = self.character_errors / self.character_reference
        expected_critical = self.critical_matches / self.critical_total
        expected_voyage = self.voyage_matches / self.voyage_total
        expected_gate = (
            expected_cer <= 0.02 and expected_critical >= 0.98 and expected_voyage >= 0.98
        )
        if (
            self.critical_matches > self.critical_total
            or abs(self.character_error_rate - expected_cer) > 1e-15
            or abs(self.critical_value_exact_match - expected_critical) > 1e-15
            or abs(self.voyage_exact_match - expected_voyage) > 1e-15
            or self.gate_passed != expected_gate
        ):
            raise ValueError("B/L repair audit score is inconsistent")
        return self


class BlRepairAuditEvaluationV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_bl_repair_audit_evaluation"] = (
        "source_attested_bl_repair_audit_evaluation"
    )
    dataset_id: Literal["synthetic-bl-repair-audit-v1"] = BL_REPAIR_AUDIT_DATASET_ID
    metric_revision: Literal["source-attested-semantic-text-v3"] = (
        "source-attested-semantic-text-v3"
    )
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_profile_revision: str = Field(min_length=1, max_length=128)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    candidate_bundle_relative_path: str = Field(min_length=1, max_length=240)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    scores: list[BlRepairAuditScoreV1] = Field(min_length=8, max_length=8)
    gate_passed: bool

    @model_validator(mode="after")
    def _complete_gate_matrix(self) -> "BlRepairAuditEvaluationV1":
        expected = {
            (kind, profile_id) for kind in ("clean", "error") for profile_id in RASTER_PROFILES
        }
        keys = [(item.fixture_kind, item.profile_id) for item in self.scores]
        expected_bundle_path = (
            f"{BL_REPAIR_AUDIT_DATASET_ID}/candidate-bundles/{self.candidate_bundle_sha256}.json"
        )
        if (
            set(keys) != expected
            or keys != sorted(keys)
            or self.candidate_bundle_relative_path != expected_bundle_path
            or self.gate_passed != all(item.gate_passed for item in self.scores)
        ):
            raise ValueError("B/L repair audit evaluation matrix is invalid")
        return self


class BlRepairAuditCandidateBundleV1(_WireModel):
    """Local-only raw OCR observations bound to one immutable audit corpus."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_bl_repair_audit_candidate_bundle"] = (
        "source_attested_bl_repair_audit_candidate_bundle"
    )
    dataset_id: Literal["synthetic-bl-repair-audit-v1"] = BL_REPAIR_AUDIT_DATASET_ID
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    candidate_profile_revision: str = Field(min_length=1, max_length=128)
    runtime_configuration_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    candidates: list[RasterTextExtractionResultV1] = Field(
        min_length=BL_REPAIR_AUDIT_RASTER_COUNT,
        max_length=BL_REPAIR_AUDIT_RASTER_COUNT,
    )

    @model_validator(mode="after")
    def _candidate_inventory_is_fixed(self) -> "BlRepairAuditCandidateBundleV1":
        keys = [(item.document_asset_id, item.source_sha256) for item in self.candidates]
        if (
            len(keys) != len(set(keys))
            or keys != sorted(keys)
            or any(
                item.raster_text_profile_revision != self.candidate_profile_revision
                or (
                    item.extractor_revision is not None
                    and item.extractor_revision != self.extractor_revision
                )
                for item in self.candidates
            )
        ):
            raise ValueError("B/L repair audit candidate bundle is invalid")
        return self


def _write_new(path: Path, content: bytes) -> None:
    if not content or path.is_symlink():
        raise ValueError("B/L repair audit output is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError:
        raise ValueError("B/L repair audit output is invalid") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _read_regular(path: Path, *, maximum: int = 32 * 1024 * 1024) -> bytes:
    try:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
            raise OSError
        return path.read_bytes()
    except OSError:
        raise ValueError("B/L repair audit material is invalid") from None


def _material_path(
    *, document_preview_root: Path, dataset_directory: Path, relative_path: str
) -> Path:
    prefix = f"{BL_REPAIR_AUDIT_DATASET_ID}/"
    if not relative_path.startswith(prefix) or "\\" in relative_path:
        raise ValueError("B/L repair audit relative path is invalid")
    parts = relative_path[len(prefix) :].split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("B/L repair audit relative path is invalid")
    candidate = dataset_directory.joinpath(*parts)
    try:
        dataset_directory.resolve(strict=True).relative_to(document_preview_root)
        candidate.parent.resolve(strict=True).relative_to(dataset_directory.resolve(strict=True))
    except (OSError, ValueError):
        raise ValueError("B/L repair audit relative path is invalid") from None
    return candidate


def bl_repair_audit_receipt_sha256(receipt: BlRepairAuditReceiptV1) -> str:
    return _hash_bytes(_canonical_json_bytes(receipt.model_dump(mode="json", warnings=False)))


def bl_repair_audit_candidate_bundle_sha256(
    bundle: BlRepairAuditCandidateBundleV1,
) -> str:
    return _hash_bytes(_canonical_json_bytes(bundle.model_dump(mode="json", warnings=False)))


def bl_repair_audit_evaluation_sha256(evaluation: BlRepairAuditEvaluationV1) -> str:
    return _hash_bytes(_canonical_json_bytes(evaluation.model_dump(mode="json", warnings=False)))


def _verify_candidate_bundle_bindings(
    *, receipt: BlRepairAuditReceiptV1, bundle: BlRepairAuditCandidateBundleV1
) -> None:
    if bundle.dataset_receipt_sha256 != bl_repair_audit_receipt_sha256(receipt):
        raise ValueError("B/L repair audit candidate dataset binding is invalid")
    bindings = {(item.asset_id, item.image_sha256): item for item in receipt.rasters}
    candidates = {(item.document_asset_id, item.source_sha256): item for item in bundle.candidates}
    if set(bindings) != set(candidates):
        raise ValueError("B/L repair audit candidate coverage is invalid")
    if any(
        candidates[key].media_type != binding.media_type
        or candidates[key].byte_size != binding.byte_size
        for key, binding in bindings.items()
    ):
        raise ValueError("B/L repair audit candidate input binding is invalid")


def _write_or_verify_content_addressed(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        if _read_regular(path, maximum=64 * 1024 * 1024) != raw:
            raise ValueError("B/L repair audit immutable artifact conflicts")
        return
    _write_new(path, raw)


def verify_bl_repair_audit_dataset(
    *, document_preview_root: Path, dataset_directory: Path | None = None
) -> BlRepairAuditReceiptV1:
    root = document_preview_root.resolve(strict=True)
    dataset = (dataset_directory or root / BL_REPAIR_AUDIT_DATASET_ID).resolve(strict=True)
    try:
        dataset.relative_to(root)
        if dataset.is_symlink() or not dataset.is_dir():
            raise OSError
    except (OSError, ValueError):
        raise ValueError("B/L repair audit dataset path is invalid") from None
    raw = _read_regular(dataset / "receipt.json", maximum=4 * 1024 * 1024)
    try:
        receipt = BlRepairAuditReceiptV1.model_validate_json(raw)
    except Exception:
        raise ValueError("B/L repair audit receipt is invalid") from None
    if raw != _canonical_json_bytes(receipt.model_dump(mode="json", warnings=False)):
        raise ValueError("B/L repair audit receipt is not canonical")
    if (
        importlib.metadata.version("pypdfium2") != receipt.pdfium_version
        or importlib.metadata.version("Pillow") != receipt.pillow_version
    ):
        raise ValueError("B/L repair audit renderer inventory drifted")
    sources = {item.asset_id: item for item in receipt.sources}
    source_truths: dict[str, SourcePdfGeometryV1] = {}
    source_geometry = {}
    extractor = PdfiumFinalPdfGeometryExtractorV1(receipt.pdfium_version)
    for source_item in receipt.sources:
        source_path = _material_path(
            document_preview_root=root,
            dataset_directory=dataset,
            relative_path=source_item.source_pdf_relative_path,
        )
        source_pdf = _read_regular(source_path, maximum=8 * 1024 * 1024)
        if _hash_bytes(source_pdf) != source_item.source_pdf_sha256:
            raise ValueError("B/L repair audit source PDF drifted")
        parsed = BlRepairAuditContentStreamParserV1(
            family_id=source_item.family_id
        ).parse_final_pdf_content_stream(source_pdf=source_pdf, asset_id=source_item.asset_id)
        if parsed.source_pdf_sha256 != source_item.source_pdf_sha256:
            raise ValueError("B/L repair audit source truth drifted")
        source_truths[source_item.asset_id] = parsed
        source_geometry[source_item.asset_id] = extractor.extract(
            source_pdf=source_pdf,
            source=parsed,
        )
    for raster_item in receipt.rasters:
        source = sources[raster_item.asset_id]
        if raster_item.source_pdf_sha256 != source.source_pdf_sha256:
            raise ValueError("B/L repair audit raster source binding drifted")
        asset_path = _material_path(
            document_preview_root=root,
            dataset_directory=dataset,
            relative_path=raster_item.asset_relative_path,
        )
        content = _read_regular(asset_path)
        if (
            _hash_bytes(content) != raster_item.image_sha256
            or len(content) != raster_item.byte_size
        ):
            raise ValueError("B/L repair audit raster drifted")
        verified = verify_rendered_raster_output(
            RenderedRasterBytesV1(
                asset_id=raster_item.asset_id,
                profile_id=raster_item.profile_id,
                content=content,
                width=raster_item.width,
                height=raster_item.height,
            ),
            pdfium_version=receipt.pdfium_version,
        )
        if verified.media_type != raster_item.media_type:
            raise ValueError("B/L repair audit raster media drifted")
        gold_path = _material_path(
            document_preview_root=root,
            dataset_directory=dataset,
            relative_path=raster_item.gold_relative_path,
        )
        gold_raw = _read_regular(gold_path, maximum=4 * 1024 * 1024)
        if _hash_bytes(gold_raw) != raster_item.gold_sha256:
            raise ValueError("B/L repair audit Gold drifted")
        try:
            gold = RasterGoldArtifactV1.model_validate_json(gold_raw)
        except Exception:
            raise ValueError("B/L repair audit Gold is invalid") from None
        if (
            gold.asset_id != raster_item.asset_id
            or gold.family_id != raster_item.family_id
            or gold.profile_id != raster_item.profile_id
            or gold.source_pdf_sha256 != raster_item.source_pdf_sha256
            or len(gold.fields) != len(_FIELD_PATTERNS)
        ):
            raise ValueError("B/L repair audit Gold binding drifted")
        source_truth = source_truths[raster_item.asset_id]
        geometry = source_geometry[raster_item.asset_id]
        expected_gold = RasterGoldArtifactV1(
            asset_id=raster_item.asset_id,
            family_id=raster_item.family_id,
            document_type="bill_of_lading",
            source_pdf_sha256=raster_item.source_pdf_sha256,
            profile_id=raster_item.profile_id,
            spans=compile_raster_gold(
                source_truth,
                profile_id=raster_item.profile_id,
                pdfium_spans=geometry,
            ),
            fields=compile_raster_field_gold(
                source_truth,
                profile_id=raster_item.profile_id,
                pdfium_spans=geometry,
            ),
            table_cells=[],
        )
        if gold != expected_gold:
            raise ValueError("B/L repair audit Gold does not replay from source")
    return receipt


def compile_bl_repair_audit_dataset(*, document_preview_root: Path) -> BlRepairAuditReceiptV1:
    """Atomically create or fully verify the fixed 100-family audit corpus."""

    root = document_preview_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("B/L repair audit document preview root is invalid")
    lock_path = root / ".synthetic-bl-repair-audit-v1.lock"
    with exclusive_local_process_lock(lock_path):
        final = root / BL_REPAIR_AUDIT_DATASET_ID
        if final.exists() or final.is_symlink():
            return verify_bl_repair_audit_dataset(document_preview_root=root)
        pdfium_version = importlib.metadata.version("pypdfium2")
        pillow_version = importlib.metadata.version("Pillow")
        staging = root / f".{BL_REPAIR_AUDIT_DATASET_ID}.staging-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        try:
            compiler = SyntheticRasterPdfiumCompiler(pdfium_version=pdfium_version)
            extractor = PdfiumFinalPdfGeometryExtractorV1(pdfium_version=pdfium_version)
            source_bindings: list[BlRepairAuditSourceBindingV1] = []
            raster_bindings: list[BlRepairAuditRasterBindingV1] = []
            for family_index in range(BL_REPAIR_AUDIT_FAMILY_COUNT):
                family_id = _family_id(family_index)
                for fixture_kind in ("clean", "error"):
                    asset_id = _asset_id(family_index, fixture_kind)
                    source_pdf = build_bl_repair_audit_source_pdf(
                        family_index=family_index,
                        fixture_kind=fixture_kind,
                    )
                    source = BlRepairAuditContentStreamParserV1(
                        family_id=family_id
                    ).parse_final_pdf_content_stream(source_pdf=source_pdf, asset_id=asset_id)
                    geometry = extractor.extract(source_pdf=source_pdf, source=source)
                    source_relative = f"{BL_REPAIR_AUDIT_DATASET_ID}/sources/{asset_id}.pdf"
                    _write_new(staging / "sources" / f"{asset_id}.pdf", source_pdf)
                    source_bindings.append(
                        BlRepairAuditSourceBindingV1(
                            asset_id=asset_id,
                            family_id=family_id,
                            fixture_kind=fixture_kind,
                            source_pdf_relative_path=source_relative,
                            source_pdf_sha256=source.source_pdf_sha256,
                        )
                    )
                    for profile_id in RASTER_PROFILES:
                        rendered = compiler.render(
                            source_pdf=source_pdf,
                            source=source,
                            profile_id=profile_id,
                        )
                        suffix = _RASTER_SUFFIX[profile_id]
                        asset_name = f"{asset_id}-{profile_id}{suffix}"
                        output = write_immutable_rendered_raster_output(
                            rendered,
                            output_path=staging / "assets" / asset_name,
                            pdfium_version=pdfium_version,
                        )
                        gold = RasterGoldArtifactV1(
                            asset_id=asset_id,
                            family_id=family_id,
                            document_type="bill_of_lading",
                            source_pdf_sha256=source.source_pdf_sha256,
                            profile_id=profile_id,
                            spans=compile_raster_gold(
                                source,
                                profile_id=profile_id,
                                pdfium_spans=geometry,
                            ),
                            fields=compile_raster_field_gold(
                                source,
                                profile_id=profile_id,
                                pdfium_spans=geometry,
                            ),
                            table_cells=[],
                        )
                        gold_raw = _canonical_json_bytes(
                            gold.model_dump(mode="json", warnings=False)
                        )
                        gold_name = f"{asset_id}-{profile_id}.json"
                        _write_new(staging / "gold" / gold_name, gold_raw)
                        raster_bindings.append(
                            BlRepairAuditRasterBindingV1(
                                asset_id=asset_id,
                                family_id=family_id,
                                fixture_kind=fixture_kind,
                                profile_id=profile_id,
                                source_pdf_sha256=source.source_pdf_sha256,
                                image_sha256=output.content_sha256,
                                media_type=output.media_type,
                                byte_size=output.byte_size,
                                width=output.width,
                                height=output.height,
                                asset_relative_path=(
                                    f"{BL_REPAIR_AUDIT_DATASET_ID}/assets/{asset_name}"
                                ),
                                gold_relative_path=(
                                    f"{BL_REPAIR_AUDIT_DATASET_ID}/gold/{gold_name}"
                                ),
                                gold_sha256=_hash_bytes(gold_raw),
                            )
                        )
            receipt = BlRepairAuditReceiptV1(
                generation_seed_sha256=_GENERATION_SEED_SHA256,
                generation_recipe_sha256=BL_REPAIR_AUDIT_RECIPE_SHA256,
                pdfium_version=pdfium_version,
                pillow_version=pillow_version,
                sources=sorted(source_bindings, key=lambda item: item.asset_id),
                rasters=sorted(
                    raster_bindings,
                    key=lambda item: (item.asset_id, item.profile_id),
                ),
            )
            _write_new(
                staging / "receipt.json",
                _canonical_json_bytes(receipt.model_dump(mode="json", warnings=False)),
            )
            verify_bl_repair_audit_dataset(
                document_preview_root=root,
                dataset_directory=staging,
            )
            os.replace(staging, final)
            return verify_bl_repair_audit_dataset(document_preview_root=root)
        finally:
            if staging.exists():
                try:
                    staging.resolve(strict=True).relative_to(root)
                    if staging.name.startswith(f".{BL_REPAIR_AUDIT_DATASET_ID}.staging-"):
                        shutil.rmtree(staging)
                except (OSError, ValueError):
                    pass


def write_bl_repair_audit_candidate_bundle(
    *,
    document_preview_root: Path,
    receipt: BlRepairAuditReceiptV1,
    bundle: BlRepairAuditCandidateBundleV1,
) -> tuple[str, str]:
    """Canonically persist raw OCR results below the Git-excluded dataset."""

    verified = verify_bl_repair_audit_dataset(document_preview_root=document_preview_root)
    if verified != receipt:
        raise ValueError("B/L repair audit receipt argument drifted")
    _verify_candidate_bundle_bindings(receipt=verified, bundle=bundle)
    raw = _canonical_json_bytes(bundle.model_dump(mode="json", warnings=False))
    digest = _hash_bytes(raw)
    relative = f"{BL_REPAIR_AUDIT_DATASET_ID}/candidate-bundles/{digest}.json"
    dataset = document_preview_root.resolve(strict=True) / BL_REPAIR_AUDIT_DATASET_ID
    _write_or_verify_content_addressed(dataset / "candidate-bundles" / f"{digest}.json", raw)
    return relative, digest


def verify_bl_repair_audit_candidate_bundle(
    *,
    document_preview_root: Path,
    candidate_bundle_relative_path: str,
    candidate_bundle_sha256: str,
    expected_candidate_profile_revision: str,
    expected_runtime_configuration_sha256: str,
) -> BlRepairAuditCandidateBundleV1:
    receipt = verify_bl_repair_audit_dataset(document_preview_root=document_preview_root)
    expected_path = f"{BL_REPAIR_AUDIT_DATASET_ID}/candidate-bundles/{candidate_bundle_sha256}.json"
    if candidate_bundle_relative_path != expected_path:
        raise ValueError("B/L repair audit candidate bundle path is invalid")
    root = document_preview_root.resolve(strict=True)
    dataset = (root / BL_REPAIR_AUDIT_DATASET_ID).resolve(strict=True)
    raw = _read_regular(
        _material_path(
            document_preview_root=root,
            dataset_directory=dataset,
            relative_path=candidate_bundle_relative_path,
        ),
        maximum=64 * 1024 * 1024,
    )
    if _hash_bytes(raw) != candidate_bundle_sha256:
        raise ValueError("B/L repair audit candidate bundle digest is invalid")
    try:
        bundle = BlRepairAuditCandidateBundleV1.model_validate_json(raw)
    except Exception:
        raise ValueError("B/L repair audit candidate bundle is invalid") from None
    if (
        raw != _canonical_json_bytes(bundle.model_dump(mode="json", warnings=False))
        or bundle.candidate_profile_revision != expected_candidate_profile_revision
        or bundle.runtime_configuration_sha256 != expected_runtime_configuration_sha256
    ):
        raise ValueError("B/L repair audit candidate bundle provenance is invalid")
    _verify_candidate_bundle_bindings(receipt=receipt, bundle=bundle)
    return bundle


def evaluate_bl_repair_audit_candidates(
    *,
    document_preview_root: Path,
    receipt: BlRepairAuditReceiptV1,
    bundle: BlRepairAuditCandidateBundleV1,
) -> BlRepairAuditEvaluationV1:
    """Evaluate one complete candidate set over 2 fixture kinds x 4 profiles."""

    _verify_candidate_bundle_bindings(receipt=receipt, bundle=bundle)
    candidates = bundle.candidates
    if len(candidates) != BL_REPAIR_AUDIT_RASTER_COUNT:
        raise ValueError("B/L repair audit candidate coverage is invalid")
    revisions = {item.raster_text_profile_revision for item in candidates}
    if revisions != {bundle.candidate_profile_revision}:
        raise ValueError("B/L repair audit candidate profile is inconsistent")
    by_binding = {(item.asset_id, item.image_sha256): item for item in receipt.rasters}
    candidate_by_key = {(item.document_asset_id, item.source_sha256): item for item in candidates}
    if len(candidate_by_key) != len(candidates) or set(candidate_by_key) != set(by_binding):
        raise ValueError("B/L repair audit candidate binding is invalid")
    root = document_preview_root.resolve(strict=True)
    dataset = (root / BL_REPAIR_AUDIT_DATASET_ID).resolve(strict=True)
    counts: dict[tuple[str, str], RasterTextMetricCountsV1] = {}
    voyage_counts: dict[tuple[str, str], int] = {}
    for key, binding in by_binding.items():
        gold_raw = _read_regular(
            _material_path(
                document_preview_root=root,
                dataset_directory=dataset,
                relative_path=binding.gold_relative_path,
            ),
            maximum=4 * 1024 * 1024,
        )
        if _hash_bytes(gold_raw) != binding.gold_sha256:
            raise ValueError("B/L repair audit Gold drifted")
        gold = RasterGoldArtifactV1.model_validate_json(gold_raw)
        candidate = candidate_by_key[key]
        voyage_fields = [field for field in gold.fields if field.field_id == "voyage"]
        if len(voyage_fields) != 1:
            raise ValueError("B/L repair audit voyage Gold is invalid")
        observed = canonicalize_synthetic_raster_semantic_text_v3(candidate.canonical_text or "")
        voyage = canonicalize_synthetic_raster_semantic_text_v3(voyage_fields[0].value)
        if not voyage or " " in voyage:
            raise ValueError("B/L repair audit voyage Gold is invalid")
        case = RasterTextEvaluationCaseV1(
            asset_id=binding.asset_id,
            document_type="bill_of_lading",
            profile_id=binding.profile_id,
            image_sha256=binding.image_sha256,
            gold=source_only_text_gold_from_raster_gold(gold),
            candidate=candidate,
        )
        bucket = (binding.fixture_kind, binding.profile_id)
        counts[bucket] = counts.get(bucket, RasterTextMetricCountsV1()) + (
            evaluate_synthetic_raster_semantic_text_case_v3(case)
        )
        voyage_counts[bucket] = voyage_counts.get(bucket, 0) + int(voyage in observed.split(" "))
    expected = {(kind, profile_id) for kind in ("clean", "error") for profile_id in RASTER_PROFILES}
    if set(counts) != expected or set(voyage_counts) != expected:
        raise ValueError("B/L repair audit score matrix is incomplete")
    scores: list[BlRepairAuditScoreV1] = []
    for (fixture_kind, profile_id), item in sorted(counts.items()):
        if item.critical_total != 1300:
            raise ValueError("B/L repair audit critical denominator is invalid")
        cer = item.character_errors / item.character_reference
        exact = item.critical_matches / item.critical_total
        voyage_matches = voyage_counts[(fixture_kind, profile_id)]
        voyage_exact = voyage_matches / 100
        scores.append(
            BlRepairAuditScoreV1(
                fixture_kind=cast(Literal["clean", "error"], fixture_kind),
                profile_id=cast(RasterProfileId, profile_id),
                character_errors=item.character_errors,
                character_reference=item.character_reference,
                critical_matches=item.critical_matches,
                critical_total=1300,
                voyage_matches=voyage_matches,
                voyage_total=100,
                character_error_rate=cer,
                critical_value_exact_match=exact,
                voyage_exact_match=voyage_exact,
                gate_passed=cer <= 0.02 and exact >= 0.98 and voyage_exact >= 0.98,
            )
        )
    bundle_sha256 = bl_repair_audit_candidate_bundle_sha256(bundle)
    return BlRepairAuditEvaluationV1(
        dataset_receipt_sha256=bl_repair_audit_receipt_sha256(receipt),
        candidate_profile_revision=bundle.candidate_profile_revision,
        runtime_configuration_sha256=bundle.runtime_configuration_sha256,
        extractor_revision=bundle.extractor_revision,
        candidate_bundle_relative_path=(
            f"{BL_REPAIR_AUDIT_DATASET_ID}/candidate-bundles/{bundle_sha256}.json"
        ),
        candidate_bundle_sha256=bundle_sha256,
        scores=scores,
        gate_passed=all(item.gate_passed for item in scores),
    )


def write_bl_repair_audit_evaluation(
    *,
    document_preview_root: Path,
    receipt: BlRepairAuditReceiptV1,
    evaluation: BlRepairAuditEvaluationV1,
) -> tuple[str, str]:
    """Recompute, then canonically persist one content-addressed evaluation."""

    verified_receipt = verify_bl_repair_audit_dataset(document_preview_root=document_preview_root)
    if verified_receipt != receipt:
        raise ValueError("B/L repair audit receipt argument drifted")
    bundle = verify_bl_repair_audit_candidate_bundle(
        document_preview_root=document_preview_root,
        candidate_bundle_relative_path=evaluation.candidate_bundle_relative_path,
        candidate_bundle_sha256=evaluation.candidate_bundle_sha256,
        expected_candidate_profile_revision=evaluation.candidate_profile_revision,
        expected_runtime_configuration_sha256=evaluation.runtime_configuration_sha256,
    )
    recomputed = evaluate_bl_repair_audit_candidates(
        document_preview_root=document_preview_root,
        receipt=verified_receipt,
        bundle=bundle,
    )
    if recomputed != evaluation or bundle.extractor_revision != evaluation.extractor_revision:
        raise ValueError("B/L repair audit evaluation does not replay")
    raw = _canonical_json_bytes(evaluation.model_dump(mode="json", warnings=False))
    digest = _hash_bytes(raw)
    relative = f"{BL_REPAIR_AUDIT_DATASET_ID}/evaluations/{digest}.json"
    dataset = document_preview_root.resolve(strict=True) / BL_REPAIR_AUDIT_DATASET_ID
    _write_or_verify_content_addressed(dataset / "evaluations" / f"{digest}.json", raw)
    return relative, digest


def verify_bl_repair_audit_evaluation(
    *,
    document_preview_root: Path,
    evaluation_relative_path: str,
    evaluation_sha256: str,
    expected_candidate_profile_revision: str,
    expected_runtime_configuration_sha256: str,
    require_gate_passed: bool = True,
) -> BlRepairAuditEvaluationV1:
    """Replay source, candidate, and Gate evidence before pointer use."""

    expected_path = f"{BL_REPAIR_AUDIT_DATASET_ID}/evaluations/{evaluation_sha256}.json"
    if evaluation_relative_path != expected_path:
        raise ValueError("B/L repair audit evaluation path is invalid")
    receipt = verify_bl_repair_audit_dataset(document_preview_root=document_preview_root)
    root = document_preview_root.resolve(strict=True)
    dataset = (root / BL_REPAIR_AUDIT_DATASET_ID).resolve(strict=True)
    raw = _read_regular(
        _material_path(
            document_preview_root=root,
            dataset_directory=dataset,
            relative_path=evaluation_relative_path,
        ),
        maximum=4 * 1024 * 1024,
    )
    if _hash_bytes(raw) != evaluation_sha256:
        raise ValueError("B/L repair audit evaluation digest is invalid")
    try:
        evaluation = BlRepairAuditEvaluationV1.model_validate_json(raw)
    except Exception:
        raise ValueError("B/L repair audit evaluation is invalid") from None
    if (
        raw != _canonical_json_bytes(evaluation.model_dump(mode="json", warnings=False))
        or evaluation.dataset_receipt_sha256 != bl_repair_audit_receipt_sha256(receipt)
        or evaluation.candidate_profile_revision != expected_candidate_profile_revision
        or evaluation.runtime_configuration_sha256 != expected_runtime_configuration_sha256
    ):
        raise ValueError("B/L repair audit evaluation provenance is invalid")
    bundle = verify_bl_repair_audit_candidate_bundle(
        document_preview_root=document_preview_root,
        candidate_bundle_relative_path=evaluation.candidate_bundle_relative_path,
        candidate_bundle_sha256=evaluation.candidate_bundle_sha256,
        expected_candidate_profile_revision=expected_candidate_profile_revision,
        expected_runtime_configuration_sha256=expected_runtime_configuration_sha256,
    )
    replayed = evaluate_bl_repair_audit_candidates(
        document_preview_root=document_preview_root,
        receipt=receipt,
        bundle=bundle,
    )
    if replayed != evaluation or bundle.extractor_revision != evaluation.extractor_revision:
        raise ValueError("B/L repair audit evaluation replay failed")
    if require_gate_passed and not evaluation.gate_passed:
        raise ValueError("B/L repair audit Gate did not pass")
    return evaluation


__all__ = [
    "BL_REPAIR_AUDIT_DATASET_ID",
    "BL_REPAIR_AUDIT_FAMILY_COUNT",
    "BL_REPAIR_AUDIT_RECEIPT_RELATIVE_PATH",
    "BL_REPAIR_AUDIT_RECIPE_SHA256",
    "BlRepairAuditContentStreamParserV1",
    "BlRepairAuditCandidateBundleV1",
    "BlRepairAuditEvaluationV1",
    "BlRepairAuditReceiptV1",
    "bl_repair_audit_candidate_bundle_sha256",
    "bl_repair_audit_evaluation_sha256",
    "bl_repair_audit_receipt_sha256",
    "build_bl_repair_audit_source_pdf",
    "compile_bl_repair_audit_dataset",
    "evaluate_bl_repair_audit_candidates",
    "verify_bl_repair_audit_dataset",
    "verify_bl_repair_audit_candidate_bundle",
    "verify_bl_repair_audit_evaluation",
    "write_bl_repair_audit_candidate_bundle",
    "write_bl_repair_audit_evaluation",
]
