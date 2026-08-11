"""Local-only receipt and human-Gold packet models for document previews."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.schemas.document_extraction import BoundingBoxV1, DocumentMediaType


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
DatasetMediaType = DocumentMediaType | Literal["application/json"]


class EvaluationWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetAssetReceiptV1(EvaluationWireModel):
    """Immutable, non-content receipt for one local source asset."""

    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    relative_path: str = Field(min_length=1, max_length=512)
    media_type: DatasetMediaType
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    triplet_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_role: Literal["preview_candidate", "structured_reference"]

    @model_validator(mode="after")
    def _relative_path_stays_within_dataset(self) -> "DatasetAssetReceiptV1":
        if (
            self.relative_path.startswith("/")
            or "\\" in self.relative_path
            or ".." in self.relative_path.split("/")
            or "\x00" in self.relative_path
        ):
            raise ValueError("relative_path must not traverse outside the dataset root")
        if self.media_type == "application/json" and self.source_role != "structured_reference":
            raise ValueError("JSON source assets are structured references, not preview candidates")
        if self.media_type != "application/json" and self.source_role != "preview_candidate":
            raise ValueError("binary source assets must be preview candidates")
        return self


class DatasetReceiptV1(EvaluationWireModel):
    """A local-source inventory.  It never stores the source bytes or text."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    received_as_of: date
    assets: list[DatasetAssetReceiptV1] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def _assets_are_unique(self) -> "DatasetReceiptV1":
        asset_ids = [asset.asset_id for asset in self.assets]
        paths = [asset.relative_path for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)) or len(paths) != len(set(paths)):
            raise ValueError("dataset receipt asset IDs and relative paths must be unique")
        return self


class GoldSpanV1(EvaluationWireModel):
    page_number: int = Field(ge=1, le=1_000)
    text: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1 | None = None
    reading_order: int = Field(ge=1, le=100_000)


class TripletRecordMappingV1(EvaluationWireModel):
    """Explicit, content-free record alignment for the three JSON sources."""

    triplet_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    invoice_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packing_list_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    bill_of_lading_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    invoice_index: int = Field(ge=0, le=1_000_000)
    packing_list_index: int = Field(ge=0, le=1_000_000)
    bill_of_lading_index: int = Field(ge=0, le=1_000_000)


class DocumentExtractionGoldV1(EvaluationWireModel):
    """A human-adjudicated target held only in local data storage.

    This model defines the shape but never accepts an implicit approval.  A
    separately controlled human process must change ``adjudication_status``.
    """

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    adjudication_status: Literal["draft", "human_adjudicated"] = "draft"
    spans: list[GoldSpanV1] = Field(default_factory=list, max_length=100_000)
    table_cells: list[dict[str, object]] = Field(default_factory=list, max_length=100_000)


class EvaluationPacketItemV1(EvaluationWireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: DocumentMediaType


class GoldReleaseThresholdsV1(EvaluationWireModel):
    """Fixed release thresholds; draft packets never assert that they passed."""

    field_exact_match_min: float = Field(default=0.99, ge=0.99, le=0.99)
    ocr_cer_max: float = Field(default=0.01, ge=0.01, le=0.01)
    table_cell_f1_min: float = Field(default=0.98, ge=0.98, le=0.98)


class EvaluationPacketV1(EvaluationWireModel):
    """A deterministic selection of assets awaiting human Gold completion."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["awaiting_human_adjudication"] = "awaiting_human_adjudication"
    split: Literal["development", "sealed"]
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prepared_as_of: date
    items: list[EvaluationPacketItemV1] = Field(min_length=1, max_length=100_000)
    required_gates: list[Literal["field_exact_match", "ocr_cer", "table_cell_f1"]] = Field(
        min_length=3,
        max_length=3,
    )
    release_thresholds: GoldReleaseThresholdsV1 = Field(default_factory=GoldReleaseThresholdsV1)

    @model_validator(mode="after")
    def _items_are_unique(self) -> "EvaluationPacketV1":
        asset_ids = [item.asset_id for item in self.items]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("packet asset IDs must be unique")
        if set(self.required_gates) != {"field_exact_match", "ocr_cer", "table_cell_f1"}:
            raise ValueError("packet must retain every required OCR release gate")
        return self


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def receipt_sha256(receipt: DatasetReceiptV1) -> str:
    return hashlib.sha256(
        _canonical_json(receipt.model_dump(mode="json", warnings=False))
    ).hexdigest()


def build_draft_gold_packets(
    receipt: DatasetReceiptV1,
    *,
    prepared_as_of: date,
    development_count: int = 60,
    sealed_count: int = 20,
) -> tuple[EvaluationPacketV1, EvaluationPacketV1]:
    """Build non-overlapping 60+20 draft packets without reading source bytes."""

    if development_count != 60 or sealed_count != 20:
        raise ValueError(
            "the pre-approval Gold packet must be exactly 60 development plus 20 sealed"
        )
    eligible_assets = [
        asset for asset in receipt.assets if asset.source_role == "preview_candidate"
    ]
    if len(eligible_assets) < development_count + sealed_count:
        raise ValueError(
            "dataset receipt does not contain enough immutable assets for the 60+20 packet"
        )
    digest = receipt_sha256(receipt)
    ordered = sorted(
        eligible_assets,
        key=lambda asset: hashlib.sha256(f"{digest}:{asset.asset_id}".encode("utf-8")).hexdigest(),
    )

    def items(source: list[DatasetAssetReceiptV1]) -> list[EvaluationPacketItemV1]:
        values: list[EvaluationPacketItemV1] = []
        for asset in source:
            if asset.media_type == "application/json":
                raise ValueError("structured JSON references cannot enter an OCR Gold packet")
            values.append(
                EvaluationPacketItemV1(
                    asset_id=asset.asset_id,
                    source_sha256=asset.sha256,
                    media_type=cast(DocumentMediaType, asset.media_type),
                )
            )
        return values

    return (
        EvaluationPacketV1(
            split="development",
            dataset_id=receipt.dataset_id,
            dataset_revision=receipt.dataset_revision,
            receipt_sha256=digest,
            prepared_as_of=prepared_as_of,
            items=items(ordered[:development_count]),
            required_gates=["field_exact_match", "ocr_cer", "table_cell_f1"],
        ),
        EvaluationPacketV1(
            split="sealed",
            dataset_id=receipt.dataset_id,
            dataset_revision=receipt.dataset_revision,
            receipt_sha256=digest,
            prepared_as_of=prepared_as_of,
            items=items(ordered[development_count : development_count + sealed_count]),
            required_gates=["field_exact_match", "ocr_cer", "table_cell_f1"],
        ),
    )


__all__ = [
    "DatasetAssetReceiptV1",
    "DatasetReceiptV1",
    "DocumentExtractionGoldV1",
    "EvaluationPacketItemV1",
    "EvaluationPacketV1",
    "GoldReleaseThresholdsV1",
    "GoldSpanV1",
    "build_draft_gold_packets",
    "receipt_sha256",
]
