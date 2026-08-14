"""Local-only, fail-closed OCR Gold and quality-evaluation models.

The public document-preview boundary deliberately has no OCR quality claims.
This module is an offline companion: it binds synthetic source receipts to
human-adjudicated Gold, scores an already-local candidate output, and emits a
small content-free release result.  Raw PDFs, OCR text, field values, and Gold
records stay below ``_local/data`` and are never logged by these helpers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    GoldReleaseThresholdsV1,
    receipt_sha256,
)
from customs_ai_agent.schemas.document_extraction import BoundingBoxV1


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_FIELD_NAME = r"^[a-z][a-z0-9_]{0,127}$"
_SHA256 = r"^[a-f0-9]{64}$"
_SYNTHETIC_PDF_PATH = re.compile(
    r"^(?P<fixture>clean_sets|error_sets)/set_(?P<set_number>\d{4})/"
    r"(?P<record>[CE]\d{4})_(?P<document>BL|INV|PL)\.pdf$"
)

PacketSplit = Literal["development", "sealed"]
FixtureKind = Literal["clean", "error"]
SyntheticDocumentType = Literal["bill_of_lading", "invoice", "packing_list"]
FieldNormalizer = Literal["text-v1", "numeric-v1", "identifier-v1"]
EngineKind = Literal["pdf_text_layer", "paddleocr", "docling"]


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def canonical_json_bytes(value: object) -> bytes:
    """Return the one compact representation used for local digest binding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_file_bytes(value: object) -> bytes:
    """Return deterministic, human-readable local artifact bytes."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a local source without retaining or printing its content."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _field_value(value: object) -> str:
    if value is None:
        raise ValueError("Gold and candidate field values must not be null")
    converted = str(value)
    if not converted.strip():
        raise ValueError("Gold and candidate field values must not be blank")
    return converted


def normalize_field_value(value: str, normalizer: FieldNormalizer) -> str:
    """Apply the fixed, field-owned normalization used by exact-match scoring."""

    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = " ".join(normalized.split())
    if normalizer == "text-v1":
        return normalized.casefold()
    if normalizer == "identifier-v1":
        return normalized.upper()
    if normalizer == "numeric-v1":
        compact = normalized.replace(",", "").replace(" ", "")
        return compact.casefold()
    raise ValueError("unsupported field normalizer")


def normalize_table_cell_text(value: str) -> str:
    return normalize_field_value(value, "text-v1")


def canonical_span_text(spans: Sequence["GoldSpanV2 | CandidateSpanV1"]) -> str:
    """Match the preview contract's NFC/newline/order canonical text rule."""

    ordered = sorted(spans, key=lambda span: (span.page_number, span.reading_order))
    return "\n".join(
        unicodedata.normalize("NFC", span.text).replace("\r\n", "\n").replace("\r", "\n")
        for span in ordered
    )


def _levenshtein_distance(reference: str, candidate: str) -> int:
    """Compute edit distance with O(min(n, m)) memory."""

    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_character in enumerate(candidate, start=1):
            substitution = previous[candidate_index - 1] + (
                reference_character != candidate_character
            )
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


class StratifiedPacketItemV2(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    media_type: Literal["application/pdf"] = "application/pdf"
    fixture_kind: FixtureKind
    document_type: SyntheticDocumentType
    set_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)


class EvaluationPacketV2(_WireModel):
    """An immutable, stratified local selection for synthetic PDF OCR work."""

    schema_version: Literal["2.0"] = "2.0"
    packet_kind: Literal["stratified-v2"] = "stratified-v2"
    split: PacketSplit
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    receipt_sha256: str = Field(pattern=_SHA256)
    error_labels_sha256: str = Field(pattern=_SHA256)
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    prepared_as_of: date
    items: list[StratifiedPacketItemV2] = Field(min_length=1, max_length=100_000)
    required_gates: list[Literal["field_exact_match", "ocr_cer", "table_cell_f1"]] = Field(
        min_length=3,
        max_length=3,
    )
    release_thresholds: GoldReleaseThresholdsV1 = Field(default_factory=GoldReleaseThresholdsV1)

    @model_validator(mode="after")
    def _packet_is_bound_and_unique(self) -> "EvaluationPacketV2":
        asset_ids = [item.asset_id for item in self.items]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("stratified packet asset IDs must be unique")
        if set(self.required_gates) != {"field_exact_match", "ocr_cer", "table_cell_f1"}:
            raise ValueError("stratified packet must retain every required OCR release gate")
        if self.packet_receipt_sha256 != packet_receipt_sha256(self):
            raise ValueError("stratified packet receipt SHA-256 mismatch")
        return self


def _packet_digest_payload(packet: EvaluationPacketV2 | Mapping[str, object]) -> dict[str, object]:
    payload = (
        packet.model_dump(mode="json", warnings=False)
        if isinstance(packet, EvaluationPacketV2)
        else dict(packet)
    )
    payload.pop("packet_receipt_sha256", None)
    return payload


def packet_receipt_sha256(packet: EvaluationPacketV2 | Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(_packet_digest_payload(packet)))


def _make_packet(
    *,
    split: PacketSplit,
    receipt: DatasetReceiptV1,
    error_labels_sha256: str,
    prepared_as_of: date,
    items: list[StratifiedPacketItemV2],
) -> EvaluationPacketV2:
    payload: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "split": split,
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "receipt_sha256": receipt_sha256(receipt),
        "error_labels_sha256": error_labels_sha256,
        "prepared_as_of": prepared_as_of.isoformat(),
        "items": [item.model_dump(mode="json", warnings=False) for item in items],
        "required_gates": ["field_exact_match", "ocr_cer", "table_cell_f1"],
        "release_thresholds": GoldReleaseThresholdsV1().model_dump(mode="json", warnings=False),
    }
    payload["packet_receipt_sha256"] = packet_receipt_sha256(payload)
    return EvaluationPacketV2.model_validate(payload)


def _synthetic_pdf_item(asset: DatasetAssetReceiptV1) -> StratifiedPacketItemV2:
    if asset.source_role != "preview_candidate" or asset.media_type != "application/pdf":
        raise ValueError("stratified synthetic PDF packets only permit PDF preview candidates")
    matched = _SYNTHETIC_PDF_PATH.fullmatch(asset.relative_path)
    if matched is None:
        raise ValueError("synthetic PDF receipt path does not have the required clean/error layout")
    fixture_directory = matched.group("fixture")
    record = matched.group("record")
    set_number = matched.group("set_number")
    if (fixture_directory == "clean_sets" and not record.startswith("C")) or (
        fixture_directory == "error_sets" and not record.startswith("E")
    ):
        raise ValueError("synthetic PDF record prefix does not match clean/error directory")
    if record[1:] != set_number:
        raise ValueError("synthetic PDF record ID does not match directory set ID")
    document_type = cast(
        SyntheticDocumentType,
        {
            "BL": "bill_of_lading",
            "INV": "invoice",
            "PL": "packing_list",
        }[matched.group("document")],
    )
    return StratifiedPacketItemV2(
        asset_id=asset.asset_id,
        source_sha256=asset.sha256,
        fixture_kind="clean" if fixture_directory == "clean_sets" else "error",
        document_type=document_type,
        set_id=record,
    )


def _rank_item(receipt_digest: str, item: StratifiedPacketItemV2) -> tuple[str, str]:
    return (
        sha256_bytes(f"{receipt_digest}:{item.source_sha256}:{item.asset_id}".encode("utf-8")),
        item.asset_id,
    )


def build_stratified_pdf_gold_packets(
    receipt: DatasetReceiptV1,
    *,
    prepared_as_of: date,
    error_labels_path: Path,
) -> tuple[EvaluationPacketV2, EvaluationPacketV2]:
    """Create the fixed six-strata 60 development / 20 sealed PDF packets.

    Development takes exactly ten from each clean/error × B/L/invoice/packing
    list stratum.  Sealed takes the next three from each stratum, then the two
    lowest receipt-SHA ranks across all remaining assets.
    """

    digest = receipt_sha256(receipt)
    labels_by_key = load_synthetic_error_labels(error_labels_path)
    error_labels_digest = sha256_file(error_labels_path)
    by_stratum: dict[tuple[FixtureKind, SyntheticDocumentType], list[StratifiedPacketItemV2]] = (
        defaultdict(list)
    )
    for asset in receipt.assets:
        if asset.media_type == "application/pdf" or asset.source_role == "preview_candidate":
            # A PDF-only packet must reject an unexpected binary candidate,
            # rather than silently ignoring it and claiming full coverage.
            if asset.media_type != "application/pdf":
                raise ValueError("synthetic PDF receipt contains a non-PDF preview candidate")
            item = _synthetic_pdf_item(asset)
            if (
                item.fixture_kind == "error"
                and (item.set_id, item.document_type) not in labels_by_key
            ):
                # Error-set siblings without a matching affected_doc label are
                # rendered clean and must not claim error-stratum coverage.
                continue
            by_stratum[(item.fixture_kind, item.document_type)].append(item)

    expected_strata: tuple[tuple[FixtureKind, SyntheticDocumentType], ...] = (
        ("clean", "bill_of_lading"),
        ("clean", "invoice"),
        ("clean", "packing_list"),
        ("error", "bill_of_lading"),
        ("error", "invoice"),
        ("error", "packing_list"),
    )
    if set(by_stratum) != set(expected_strata):
        raise ValueError("synthetic PDF receipt is missing or contains an unknown OCR stratum")

    development: list[StratifiedPacketItemV2] = []
    sealed: list[StratifiedPacketItemV2] = []
    remaining: list[StratifiedPacketItemV2] = []
    for stratum in expected_strata:
        ordered = sorted(by_stratum[stratum], key=lambda item: _rank_item(digest, item))
        if len(ordered) < 13:
            raise ValueError("each synthetic PDF OCR stratum requires at least thirteen assets")
        development.extend(ordered[:10])
        sealed.extend(ordered[10:13])
        remaining.extend(ordered[13:])
    if len(remaining) < 2:
        raise ValueError(
            "synthetic PDF receipt has no eligible assets for sealed SHA-ranked extras"
        )
    sealed.extend(sorted(remaining, key=lambda item: _rank_item(digest, item))[:2])

    development_ids = {item.asset_id for item in development}
    sealed_ids = {item.asset_id for item in sealed}
    if len(development) != 60 or len(sealed) != 20 or development_ids & sealed_ids:
        raise ValueError("stratified packet selection is incomplete or overlapping")
    return (
        _make_packet(
            split="development",
            receipt=receipt,
            error_labels_sha256=error_labels_digest,
            prepared_as_of=prepared_as_of,
            items=development,
        ),
        _make_packet(
            split="sealed",
            receipt=receipt,
            error_labels_sha256=error_labels_digest,
            prepared_as_of=prepared_as_of,
            items=sealed,
        ),
    )


def verify_receipt_source_assets(*, receipt: DatasetReceiptV1, source_directory: Path) -> None:
    """Fail closed when an on-disk asset no longer matches its immutable receipt."""

    root = source_directory.resolve()
    if not root.is_dir():
        raise ValueError("source dataset directory is unavailable")
    for asset in receipt.assets:
        unresolved = root / Path(asset.relative_path)
        if unresolved.is_symlink():
            raise ValueError("receipt source asset must not be a symbolic link")
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("receipt asset path escapes source dataset directory") from error
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("receipt source asset is unavailable or not a regular file")
        if candidate.stat().st_size != asset.byte_size or sha256_file(candidate) != asset.sha256:
            raise ValueError("receipt source asset SHA-256 or byte size drifted")


class GoldFieldV2(_WireModel):
    field_name: str = Field(min_length=1, max_length=128, pattern=_FIELD_NAME)
    value: str = Field(min_length=1, max_length=8_000)
    normalizer: FieldNormalizer

    @model_validator(mode="after")
    def _value_is_evaluable(self) -> "GoldFieldV2":
        _field_value(self.value)
        return self


class CandidateFieldV1(_WireModel):
    field_name: str = Field(min_length=1, max_length=128, pattern=_FIELD_NAME)
    value: str = Field(min_length=1, max_length=8_000)

    @model_validator(mode="after")
    def _value_is_evaluable(self) -> "CandidateFieldV1":
        _field_value(self.value)
        return self


class GoldSpanV2(_WireModel):
    page_number: int = Field(ge=1, le=1_000)
    text: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1 | None = None
    reading_order: int = Field(ge=1, le=100_000)


class CandidateSpanV1(GoldSpanV2):
    pass


class GoldTableCellV2(_WireModel):
    page_number: int = Field(ge=1, le=1_000)
    table_index: int = Field(ge=1, le=10_000)
    row_index: int = Field(ge=0, le=100_000)
    column_index: int = Field(ge=0, le=100_000)
    text: str = Field(min_length=1, max_length=8_000)


class CandidateTableCellV1(GoldTableCellV2):
    pass


class SourceGroundTruthAttestationV1(_WireModel):
    source_kind: Literal["synthetic_pdf_manifest_v1"] = "synthetic_pdf_manifest_v1"
    clean_manifest_sha256: str = Field(pattern=_SHA256)
    error_labels_sha256: str = Field(pattern=_SHA256)
    source_set_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    error_label_sha256s: list[str] = Field(default_factory=list, max_length=64)
    ground_truth_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _attestation_is_bound(self) -> "SourceGroundTruthAttestationV1":
        if any(re.fullmatch(_SHA256, value) is None for value in self.error_label_sha256s):
            raise ValueError("error label source hashes must be SHA-256 values")
        if len(self.error_label_sha256s) != len(set(self.error_label_sha256s)):
            raise ValueError("error label source hashes must be unique")
        if self.ground_truth_receipt_sha256 != ground_truth_receipt_sha256(self):
            raise ValueError("source ground-truth attestation SHA-256 mismatch")
        return self


def ground_truth_receipt_sha256(
    attestation: SourceGroundTruthAttestationV1 | Mapping[str, object],
) -> str:
    payload = (
        attestation.model_dump(mode="json", warnings=False)
        if isinstance(attestation, SourceGroundTruthAttestationV1)
        else dict(attestation)
    )
    payload.pop("ground_truth_receipt_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


class ReviewerApprovalV1(_WireModel):
    approval_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    reviewer_id_sha256: str = Field(pattern=_SHA256)
    approved_at: datetime
    source_sha256: str = Field(pattern=_SHA256)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    gold_content_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _approved_at_is_aware(self) -> "ReviewerApprovalV1":
        if self.approved_at.tzinfo is None or self.approved_at.utcoffset() is None:
            raise ValueError("reviewer approval timestamp must include a timezone")
        return self


class DocumentExtractionGoldV2(_WireModel):
    """One local Gold record; only explicit human approval makes it scorable."""

    schema_version: Literal["2.0"] = "2.0"
    packet_kind: Literal["stratified-v2"] = "stratified-v2"
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    fixture_kind: FixtureKind
    document_type: SyntheticDocumentType
    set_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    adjudication_status: Literal["draft", "human_adjudicated"] = "draft"
    normalization_revision: Literal["ocr-gold-normalization-v1"] = "ocr-gold-normalization-v1"
    fields: list[GoldFieldV2] = Field(default_factory=list, max_length=1_000)
    spans: list[GoldSpanV2] = Field(default_factory=list, max_length=100_000)
    table_cells: list[GoldTableCellV2] = Field(default_factory=list, max_length=100_000)
    source_attestation: SourceGroundTruthAttestationV1
    gold_content_sha256: str = Field(pattern=_SHA256)
    reviewer_approval: ReviewerApprovalV1 | None = None

    @model_validator(mode="after")
    def _record_is_bound_and_consistent(self) -> "DocumentExtractionGoldV2":
        field_names = [field.field_name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("Gold field names must be unique within an asset")
        span_orders = [(span.page_number, span.reading_order) for span in self.spans]
        if len(span_orders) != len(set(span_orders)):
            raise ValueError("Gold span reading order must be unique within a page")
        cell_keys = [
            (cell.page_number, cell.table_index, cell.row_index, cell.column_index)
            for cell in self.table_cells
        ]
        if len(cell_keys) != len(set(cell_keys)):
            raise ValueError("Gold table cells must have unique page/table/row/column coordinates")
        if self.gold_content_sha256 != gold_content_sha256(self):
            raise ValueError("Gold content SHA-256 mismatch")
        if self.adjudication_status == "draft":
            if self.reviewer_approval is not None:
                raise ValueError("draft Gold must not carry a reviewer approval")
            return self
        if (
            not self.fields
            or not self.spans
            or not self.table_cells
            or self.reviewer_approval is None
        ):
            raise ValueError(
                "human-adjudicated Gold requires fields, spans, table cells, and approval"
            )
        approval = self.reviewer_approval
        if (
            approval.source_sha256 != self.source_sha256
            or approval.dataset_revision != self.dataset_revision
            or approval.gold_content_sha256 != self.gold_content_sha256
        ):
            raise ValueError("reviewer approval does not bind this Gold record")
        return self


def _gold_digest_payload(
    record: DocumentExtractionGoldV2 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        record.model_dump(mode="json", warnings=False)
        if isinstance(record, DocumentExtractionGoldV2)
        else dict(record)
    )
    payload.pop("gold_content_sha256", None)
    payload.pop("reviewer_approval", None)
    return payload


def gold_content_sha256(record: DocumentExtractionGoldV2 | Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(_gold_digest_payload(record)))


def build_draft_gold_record(
    *,
    packet: EvaluationPacketV2,
    item: StratifiedPacketItemV2,
    fields: Sequence[GoldFieldV2],
    source_attestation: SourceGroundTruthAttestationV1,
    spans: Sequence[GoldSpanV2] = (),
    table_cells: Sequence[GoldTableCellV2] = (),
) -> DocumentExtractionGoldV2:
    payload: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "asset_id": item.asset_id,
        "source_sha256": item.source_sha256,
        "fixture_kind": item.fixture_kind,
        "document_type": item.document_type,
        "set_id": item.set_id,
        "adjudication_status": "draft",
        "normalization_revision": "ocr-gold-normalization-v1",
        "fields": [field.model_dump(mode="json", warnings=False) for field in fields],
        "spans": [span.model_dump(mode="json", warnings=False) for span in spans],
        "table_cells": [cell.model_dump(mode="json", warnings=False) for cell in table_cells],
        "source_attestation": source_attestation.model_dump(mode="json", warnings=False),
    }
    payload["gold_content_sha256"] = gold_content_sha256(payload)
    return DocumentExtractionGoldV2.model_validate(payload)


def approve_gold_record(
    record: DocumentExtractionGoldV2,
    *,
    approval_id: str,
    reviewer_id_sha256: str,
    approved_at: datetime | None = None,
) -> DocumentExtractionGoldV2:
    """Create an explicitly bound human approval after external review.

    This helper does not invent spans, table cells, or a reviewer identity.  It
    only makes an already-complete local draft eligible for scoring.
    """

    if record.adjudication_status != "draft":
        raise ValueError("only an unapproved Gold draft can be approved")
    if not record.fields or not record.spans or not record.table_cells:
        raise ValueError("Gold approval requires reviewed fields, spans, and table cells")
    approved_at = approved_at or datetime.now(timezone.utc)
    approval = ReviewerApprovalV1(
        approval_id=approval_id,
        reviewer_id_sha256=reviewer_id_sha256,
        approved_at=approved_at,
        source_sha256=record.source_sha256,
        dataset_revision=record.dataset_revision,
        gold_content_sha256=record.gold_content_sha256,
    )
    payload = record.model_dump(mode="json", warnings=False)
    payload["adjudication_status"] = "human_adjudicated"
    payload["reviewer_approval"] = approval.model_dump(mode="json", warnings=False)
    # Status is part of the bound Gold content, so rederive the digest and the
    # approval together instead of carrying a draft digest across the state edge.
    payload["gold_content_sha256"] = gold_content_sha256(payload)
    payload["reviewer_approval"]["gold_content_sha256"] = payload["gold_content_sha256"]
    return DocumentExtractionGoldV2.model_validate(payload)


class GoldPacketV2(_WireModel):
    """Local-only record bundle handed from review to the evaluator."""

    schema_version: Literal["2.0"] = "2.0"
    packet_kind: Literal["stratified-v2"] = "stratified-v2"
    packet_split: PacketSplit
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    status: Literal["draft", "human_adjudicated"]
    records: list[DocumentExtractionGoldV2] = Field(min_length=1, max_length=100_000)
    gold_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _gold_packet_is_bound(self) -> "GoldPacketV2":
        asset_ids = [record.asset_id for record in self.records]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("Gold packet asset IDs must be unique")
        if self.gold_receipt_sha256 != gold_packet_receipt_sha256(self):
            raise ValueError("Gold packet receipt SHA-256 mismatch")
        statuses = {record.adjudication_status for record in self.records}
        if self.status == "human_adjudicated" and statuses != {"human_adjudicated"}:
            raise ValueError("human-adjudicated Gold packet contains an unapproved record")
        if self.status == "draft" and "human_adjudicated" in statuses:
            raise ValueError("Gold packet status cannot mix draft and approved records")
        return self


def _gold_packet_digest_payload(packet: GoldPacketV2 | Mapping[str, object]) -> dict[str, object]:
    payload = (
        packet.model_dump(mode="json", warnings=False)
        if isinstance(packet, GoldPacketV2)
        else dict(packet)
    )
    payload.pop("gold_receipt_sha256", None)
    return payload


def gold_packet_receipt_sha256(packet: GoldPacketV2 | Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(_gold_packet_digest_payload(packet)))


def build_gold_packet(
    *,
    packet: EvaluationPacketV2,
    records: Sequence[DocumentExtractionGoldV2],
) -> GoldPacketV2:
    packet_items = {item.asset_id: item for item in packet.items}
    if set(packet_items) != {record.asset_id for record in records}:
        raise ValueError("Gold records must cover exactly the selected evaluation packet")
    for record in records:
        _validate_gold_record_against_item(
            record=record, packet=packet, item=packet_items[record.asset_id]
        )
    status: Literal["draft", "human_adjudicated"] = (
        "human_adjudicated"
        if all(record.adjudication_status == "human_adjudicated" for record in records)
        else "draft"
    )
    payload: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "packet_split": packet.split,
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "status": status,
        "records": [record.model_dump(mode="json", warnings=False) for record in records],
    }
    payload["gold_receipt_sha256"] = gold_packet_receipt_sha256(payload)
    return GoldPacketV2.model_validate(payload)


def _validate_gold_record_against_item(
    *,
    record: DocumentExtractionGoldV2,
    packet: EvaluationPacketV2,
    item: StratifiedPacketItemV2,
) -> None:
    if (
        record.dataset_id != packet.dataset_id
        or record.dataset_revision != packet.dataset_revision
        or record.packet_receipt_sha256 != packet.packet_receipt_sha256
        or record.source_sha256 != item.source_sha256
        or record.fixture_kind != item.fixture_kind
        or record.document_type != item.document_type
        or record.set_id != item.set_id
    ):
        raise ValueError("Gold record does not bind the selected packet item")


def validate_gold_packet_for_evaluation(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    gold_packet: GoldPacketV2,
) -> None:
    """Verify all receipt → packet → reviewed-Gold bindings before scoring."""

    if receipt_sha256(receipt) != packet.receipt_sha256:
        raise ValueError("evaluation packet does not bind the supplied dataset receipt")
    if (
        gold_packet.status != "human_adjudicated"
        or gold_packet.packet_split != packet.split
        or gold_packet.dataset_id != packet.dataset_id
        or gold_packet.dataset_revision != packet.dataset_revision
        or gold_packet.packet_receipt_sha256 != packet.packet_receipt_sha256
    ):
        raise ValueError("Gold packet is not an approved exact match for the evaluation packet")
    receipt_assets = {asset.asset_id: asset for asset in receipt.assets}
    packet_items = {item.asset_id: item for item in packet.items}
    if set(packet_items) != {record.asset_id for record in gold_packet.records}:
        raise ValueError("Gold packet does not cover exactly the evaluation packet")
    for asset_id, item in packet_items.items():
        asset = receipt_assets.get(asset_id)
        if (
            asset is None
            or asset.sha256 != item.source_sha256
            or asset.media_type != "application/pdf"
        ):
            raise ValueError("packet source asset is missing or drifted from the dataset receipt")
    for record in gold_packet.records:
        _validate_gold_record_against_item(
            record=record, packet=packet, item=packet_items[record.asset_id]
        )
        if record.adjudication_status != "human_adjudicated":
            raise ValueError("unapproved Gold cannot enter OCR scoring")


class CandidateDocumentOutputV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    fields: list[CandidateFieldV1] = Field(default_factory=list, max_length=1_000)
    spans: list[CandidateSpanV1] = Field(default_factory=list, max_length=100_000)
    table_cells: list[CandidateTableCellV1] = Field(default_factory=list, max_length=100_000)

    @model_validator(mode="after")
    def _candidate_contents_are_unambiguous(self) -> "CandidateDocumentOutputV1":
        field_names = [field.field_name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("candidate field names must be unique within an asset")
        span_orders = [(span.page_number, span.reading_order) for span in self.spans]
        if len(span_orders) != len(set(span_orders)):
            raise ValueError("candidate span reading order must be unique within a page")
        cell_keys = [
            (cell.page_number, cell.table_index, cell.row_index, cell.column_index)
            for cell in self.table_cells
        ]
        if len(cell_keys) != len(set(cell_keys)):
            raise ValueError(
                "candidate table cells must have unique page/table/row/column coordinates"
            )
        return self


class CandidateOutputBundleV1(_WireModel):
    """Local adapter output supplied to an offline scorer, never a service API."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    documents: list[CandidateDocumentOutputV1] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def _documents_are_unique(self) -> "CandidateOutputBundleV1":
        asset_ids = [document.asset_id for document in self.documents]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("candidate output asset IDs must be unique")
        return self


class OcrQualityMetricsV1(_WireModel):
    field_exact_match: float = Field(ge=0, le=1)
    ocr_cer: float = Field(ge=0, le=1)
    table_cell_f1: float = Field(ge=0, le=1)


class OcrEvaluationResultV1(_WireModel):
    """Content-free sealed/development result consumed by runtime approval."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packet_kind: PacketSplit
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    gold_receipt_sha256: str = Field(pattern=_SHA256)
    engine_kind: EngineKind
    extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    configuration_sha256: str = Field(pattern=_SHA256)
    model_inventory_sha256: str = Field(pattern=_SHA256)
    evaluation_result_sha256: str = Field(pattern=_SHA256)
    passed: bool
    metrics: OcrQualityMetricsV1

    @model_validator(mode="after")
    def _result_is_derived_and_bound(self) -> "OcrEvaluationResultV1":
        if self.evaluation_result_sha256 != evaluation_result_sha256(self):
            raise ValueError("evaluation result SHA-256 mismatch")
        thresholds = GoldReleaseThresholdsV1()
        expected_passed = (
            self.metrics.field_exact_match >= thresholds.field_exact_match_min
            and self.metrics.ocr_cer <= thresholds.ocr_cer_max
            and self.metrics.table_cell_f1 >= thresholds.table_cell_f1_min
        )
        if self.passed != expected_passed:
            raise ValueError(
                "evaluation result pass flag does not match fixed OCR release thresholds"
            )
        return self


def _evaluation_result_payload(
    result: OcrEvaluationResultV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        result.model_dump(mode="json", warnings=False)
        if isinstance(result, OcrEvaluationResultV1)
        else dict(result)
    )
    payload.pop("evaluation_result_sha256", None)
    return payload


def evaluation_result_sha256(result: OcrEvaluationResultV1 | Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(_evaluation_result_payload(result)))


def _table_cell_set(
    cells: Iterable[GoldTableCellV2 | CandidateTableCellV1],
) -> set[tuple[int, int, int, int, str]]:
    return {
        (
            cell.page_number,
            cell.table_index,
            cell.row_index,
            cell.column_index,
            normalize_table_cell_text(cell.text),
        )
        for cell in cells
    }


def score_candidate_output(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    gold_packet: GoldPacketV2,
    candidate_output: CandidateOutputBundleV1,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
) -> OcrEvaluationResultV1:
    """Score a fully local candidate output against reviewed Gold.

    The returned record is intentionally content-free.  Missing fields, spans,
    or cells lower the score; a partial candidate cannot silently pass.
    """

    validate_gold_packet_for_evaluation(receipt=receipt, packet=packet, gold_packet=gold_packet)
    if (
        candidate_output.dataset_id != packet.dataset_id
        or candidate_output.dataset_revision != packet.dataset_revision
        or candidate_output.packet_receipt_sha256 != packet.packet_receipt_sha256
    ):
        raise ValueError("candidate output does not bind the selected evaluation packet")
    candidate_by_asset = {document.asset_id: document for document in candidate_output.documents}
    gold_by_asset = {record.asset_id: record for record in gold_packet.records}
    if set(candidate_by_asset) != set(gold_by_asset):
        raise ValueError("candidate output must cover exactly the reviewed Gold packet")

    exact_fields = 0
    total_fields = 0
    total_distance = 0
    total_reference_characters = 0
    gold_cells: set[tuple[str, int, int, int, int, str]] = set()
    candidate_cells: set[tuple[str, int, int, int, int, str]] = set()
    for asset_id, gold in gold_by_asset.items():
        candidate = candidate_by_asset[asset_id]
        if candidate.source_sha256 != gold.source_sha256:
            raise ValueError("candidate output source SHA-256 does not match reviewed Gold")
        candidate_fields = {field.field_name: field.value for field in candidate.fields}
        for field in gold.fields:
            total_fields += 1
            if field.field_name in candidate_fields and normalize_field_value(
                candidate_fields[field.field_name], field.normalizer
            ) == normalize_field_value(field.value, field.normalizer):
                exact_fields += 1
        reference_text = canonical_span_text(gold.spans)
        candidate_text = canonical_span_text(candidate.spans)
        if not reference_text:
            raise ValueError("human-adjudicated Gold must contain canonical text")
        total_distance += _levenshtein_distance(reference_text, candidate_text)
        total_reference_characters += len(reference_text)
        gold_cells.update(
            (asset_id, page, table, row, column, text)
            for page, table, row, column, text in _table_cell_set(gold.table_cells)
        )
        candidate_cells.update(
            (asset_id, page, table, row, column, text)
            for page, table, row, column, text in _table_cell_set(candidate.table_cells)
        )
    if total_fields == 0 or total_reference_characters == 0 or not gold_cells:
        raise ValueError("human-adjudicated Gold is missing a required OCR release target")
    true_positive = len(gold_cells & candidate_cells)
    denominator = len(gold_cells) + len(candidate_cells)
    table_f1 = 0.0 if denominator == 0 else (2.0 * true_positive) / denominator
    metrics = OcrQualityMetricsV1(
        field_exact_match=exact_fields / total_fields,
        ocr_cer=min(1.0, total_distance / total_reference_characters),
        table_cell_f1=table_f1,
    )
    thresholds = packet.release_thresholds
    passed = (
        metrics.field_exact_match >= thresholds.field_exact_match_min
        and metrics.ocr_cer <= thresholds.ocr_cer_max
        and metrics.table_cell_f1 >= thresholds.table_cell_f1_min
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "packet_kind": packet.split,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "gold_receipt_sha256": gold_packet.gold_receipt_sha256,
        "engine_kind": engine_kind,
        "extractor_revision": extractor_revision,
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": model_inventory_sha256,
        "passed": passed,
        "metrics": metrics.model_dump(mode="json", warnings=False),
    }
    payload["evaluation_result_sha256"] = evaluation_result_sha256(payload)
    return OcrEvaluationResultV1.model_validate(payload)


def select_best_development_result(
    results: Sequence[OcrEvaluationResultV1],
) -> OcrEvaluationResultV1:
    """Choose exactly one passing development candidate with deterministic ties."""

    if not results:
        raise ValueError("at least one development evaluation result is required")
    first = results[0]
    if first.packet_kind != "development":
        raise ValueError("only development results can select an OCR engine")
    identity = (
        first.dataset_id,
        first.dataset_revision,
        first.packet_receipt_sha256,
        first.gold_receipt_sha256,
    )
    for result in results:
        if (
            result.packet_kind != "development"
            or (
                result.dataset_id,
                result.dataset_revision,
                result.packet_receipt_sha256,
                result.gold_receipt_sha256,
            )
            != identity
        ):
            raise ValueError("development candidates must share one exact packet and Gold receipt")
    passing = [result for result in results if result.passed]
    if not passing:
        raise ValueError("no development OCR candidate passed every fixed release gate")
    return sorted(
        passing,
        key=lambda result: (
            -result.metrics.field_exact_match,
            result.metrics.ocr_cer,
            -result.metrics.table_cell_f1,
            result.engine_kind,
        ),
    )[0]


class EvaluationRunStateV1(_WireModel):
    """One immutable state record for an evaluator output directory."""

    schema_version: Literal["1.0"] = "1.0"
    packet_kind: PacketSplit
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    gold_receipt_sha256: str = Field(pattern=_SHA256)
    evaluation_result_sha256: str = Field(pattern=_SHA256)


def validate_evaluation_run_state(
    *,
    state: EvaluationRunStateV1,
    result: OcrEvaluationResultV1,
) -> None:
    if (
        state.packet_kind != result.packet_kind
        or state.packet_receipt_sha256 != result.packet_receipt_sha256
        or state.gold_receipt_sha256 != result.gold_receipt_sha256
        or state.evaluation_result_sha256 != result.evaluation_result_sha256
    ):
        raise ValueError("evaluation run state does not bind the result artifact")


def write_evaluation_result(
    *,
    result: OcrEvaluationResultV1,
    output_directory: Path,
) -> Path:
    """Write one immutable result and block a second sealed run in that ledger."""

    resolved = output_directory.resolve()
    parts = [part.casefold() for part in resolved.parts]
    if not any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1)):
        raise ValueError(
            "evaluation output directory must be below the local-only _local/data boundary"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    result_path = resolved / f"{result.evaluation_result_sha256}.json"
    payload = result.model_dump(mode="json", warnings=False)
    expected = canonical_json_file_bytes(payload)
    state_path = resolved / "sealed-evaluation-state.json"
    # Validate the one-time sealed ledger before writing any new result bytes.
    # Otherwise an attempted rerun could leave an unapproved extra artifact.
    if result.packet_kind == "sealed" and state_path.exists():
        existing = EvaluationRunStateV1.model_validate_json(state_path.read_bytes())
        validate_evaluation_run_state(state=existing, result=result)
        raise ValueError("sealed evaluation already recorded; use a new packet and Gold revision")
    if result_path.exists():
        if result_path.read_bytes() != expected:
            raise ValueError("immutable evaluation result path already has different bytes")
    else:
        result_path.write_bytes(expected)
    # Development intentionally permits several candidate engines under the
    # same packet.  The sealed split gets the single-use ledger that prevents
    # tuning/re-running an evaluation after its result is known.
    if result.packet_kind == "development":
        return result_path
    state_payload = EvaluationRunStateV1(
        packet_kind=result.packet_kind,
        packet_receipt_sha256=result.packet_receipt_sha256,
        gold_receipt_sha256=result.gold_receipt_sha256,
        evaluation_result_sha256=result.evaluation_result_sha256,
    )
    state_bytes = canonical_json_file_bytes(state_payload.model_dump(mode="json", warnings=False))
    state_path.write_bytes(state_bytes)
    return result_path


def read_canonical_evaluation_result(path: Path) -> OcrEvaluationResultV1:
    """Reject byte-level edits as well as an invalid embedded semantic digest."""

    raw = path.read_bytes()
    result = OcrEvaluationResultV1.model_validate_json(raw)
    if raw != canonical_json_file_bytes(result.model_dump(mode="json", warnings=False)):
        raise ValueError("evaluation result bytes are not canonical")
    return result


_MANIFEST_FIELDS: dict[SyntheticDocumentType, tuple[tuple[str, FieldNormalizer], ...]] = {
    "bill_of_lading": (
        ("exporter_name", "text-v1"),
        ("consignee", "text-v1"),
        ("load_port", "text-v1"),
        ("disch_port", "text-v1"),
        ("final_dest", "text-v1"),
        ("vessel", "text-v1"),
        ("voyage", "identifier-v1"),
        ("bl_no", "identifier-v1"),
        ("container_no", "identifier-v1"),
        ("seal_no", "identifier-v1"),
        ("good", "text-v1"),
        ("gross_weight", "numeric-v1"),
        ("containers", "numeric-v1"),
    ),
    "invoice": (
        ("exporter_name", "text-v1"),
        ("importer_name", "text-v1"),
        ("consignee", "text-v1"),
        ("load_port", "text-v1"),
        ("disch_port", "text-v1"),
        ("vessel", "text-v1"),
        ("inv_no", "identifier-v1"),
        ("inv_date", "identifier-v1"),
        ("lc_no", "identifier-v1"),
        ("good", "text-v1"),
        ("unit", "text-v1"),
        ("qty", "numeric-v1"),
        ("unit_price", "numeric-v1"),
        ("amount", "numeric-v1"),
        ("total", "numeric-v1"),
    ),
    "packing_list": (
        ("exporter_name", "text-v1"),
        ("consignee", "text-v1"),
        ("inv_no", "identifier-v1"),
        ("lc_no", "identifier-v1"),
        ("good", "text-v1"),
        ("unit", "text-v1"),
        ("qty", "numeric-v1"),
        ("cartons", "numeric-v1"),
        ("containers", "numeric-v1"),
        ("net_weight", "numeric-v1"),
        ("gross_weight", "numeric-v1"),
        ("cbm", "numeric-v1"),
        ("net_gross_weight", "text-v1"),
    ),
}

_ERROR_DOCUMENT_TYPE: dict[str, SyntheticDocumentType] = {
    "BILL_OF_LADING": "bill_of_lading",
    "COMMERCIAL_INVOICE": "invoice",
    "PACKING_LIST": "packing_list",
}
_ERROR_FIELD_NAME: dict[tuple[SyntheticDocumentType, str], str] = {
    ("bill_of_lading", "Container No."): "container_no",
    ("bill_of_lading", "Gross Weight"): "gross_weight",
    ("invoice", "Amount"): "amount",
    ("invoice", "Port of Discharge"): "disch_port",
    ("invoice", "TOTAL"): "total",
    ("invoice", "Vessel/Flight"): "vessel",
    ("packing_list", "Goods Description"): "good",
    ("packing_list", "Net/Gross Weight"): "net_gross_weight",
    ("packing_list", "No. of Cartons / PKGS"): "cartons",
    ("packing_list", "No. of Invoice"): "inv_no",
    ("packing_list", "No. of L/C"): "lc_no",
    ("packing_list", "Quantity"): "qty",
    ("packing_list", "To Applicant / Consignee"): "consignee",
}


def _read_csv_rows(path: Path, expected_headers: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != expected_headers:
            raise ValueError("synthetic ground-truth CSV headers are malformed")
        rows = [dict(row) for row in reader]
    if not rows or any(
        set(row) != expected_headers or any(value is None for value in row.values()) for row in rows
    ):
        raise ValueError("synthetic ground-truth CSV contains malformed rows")
    return rows


_CLEAN_HEADERS = {
    "set_id",
    "exporter_name",
    "importer_name",
    "consignee",
    "bank",
    "load_port",
    "disch_port",
    "final_dest",
    "vessel",
    "voyage",
    "bl_no",
    "inv_no",
    "inv_date",
    "lc_no",
    "container_no",
    "seal_no",
    "good",
    "unit",
    "qty",
    "unit_price",
    "amount",
    "total",
    "cartons",
    "containers",
    "net_weight",
    "gross_weight",
    "cbm",
}
_ERROR_HEADERS = {
    "set_id",
    "num_errors_in_set",
    "error_type",
    "affected_doc",
    "field",
    "correct_value",
    "wrong_value",
    "note",
}


def load_synthetic_error_labels(
    error_labels_path: Path,
) -> dict[tuple[str, SyntheticDocumentType], list[dict[str, str]]]:
    """Load only verified actual-error eligibility keys from the local CSV.

    An ``error_sets`` sibling is not necessarily changed.  The matching
    ``(E set_id, affected_doc)`` label with a nonblank ``wrong_value`` is the
    sole proof that an error PDF belongs in an error evaluation stratum.
    """

    rows = _read_csv_rows(error_labels_path, _ERROR_HEADERS)
    labels_by_key: dict[tuple[str, SyntheticDocumentType], list[dict[str, str]]] = defaultdict(list)
    seen_error_fields: set[tuple[str, SyntheticDocumentType, str]] = set()
    for row in rows:
        document_type = _ERROR_DOCUMENT_TYPE.get(row["affected_doc"])
        if document_type is None:
            raise ValueError("error label has an unsupported affected document type")
        canonical_field = _ERROR_FIELD_NAME.get((document_type, row["field"]))
        if canonical_field is None:
            raise ValueError("error label has an unsupported field mapping")
        if (
            not row["set_id"].startswith("E")
            or not row["correct_value"].strip()
            or not row["wrong_value"].strip()
        ):
            raise ValueError("error label has an invalid set ID or missing source value")
        unique_key = (row["set_id"], document_type, canonical_field)
        if unique_key in seen_error_fields:
            raise ValueError("error labels contain ambiguous duplicate document fields")
        seen_error_fields.add(unique_key)
        labels_by_key[(row["set_id"], document_type)].append(row)
    return labels_by_key


def _label_sha256(row: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(row)))


def _attestation(
    *,
    clean_manifest_sha256: str,
    error_labels_sha256: str,
    source_set_id: str,
    labels: Sequence[Mapping[str, str]],
) -> SourceGroundTruthAttestationV1:
    payload: dict[str, object] = {
        "source_kind": "synthetic_pdf_manifest_v1",
        "clean_manifest_sha256": clean_manifest_sha256,
        "error_labels_sha256": error_labels_sha256,
        "source_set_id": source_set_id,
        "error_label_sha256s": sorted(_label_sha256(label) for label in labels),
    }
    payload["ground_truth_receipt_sha256"] = ground_truth_receipt_sha256(payload)
    return SourceGroundTruthAttestationV1.model_validate(payload)


def _manifest_field_values(
    *,
    row: Mapping[str, str],
    document_type: SyntheticDocumentType,
) -> dict[str, GoldFieldV2]:
    result: dict[str, GoldFieldV2] = {}
    for source_field, normalizer in _MANIFEST_FIELDS[document_type]:
        if source_field == "net_gross_weight":
            value = f"{row['net_weight']} / {row['gross_weight']}"
        else:
            value = row[source_field]
        result[source_field] = GoldFieldV2(
            field_name=source_field,
            value=_field_value(value),
            normalizer=normalizer,
        )
    return result


def build_synthetic_pdf_draft_gold(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    clean_manifest_path: Path,
    error_labels_path: Path,
) -> GoldPacketV2:
    """Turn human-verified synthetic CSV ground truth into source-bound drafts.

    This intentionally seeds field Gold only.  A reviewer must add/verify
    rendered span and table-cell Gold before the records can be approved or
    used for an OCR quality claim.
    """

    if receipt_sha256(receipt) != packet.receipt_sha256:
        raise ValueError("Gold packet seed does not bind the supplied dataset receipt")
    clean_rows = _read_csv_rows(clean_manifest_path, _CLEAN_HEADERS)
    clean_by_id = {row["set_id"]: row for row in clean_rows}
    if len(clean_by_id) != len(clean_rows):
        raise ValueError("clean manifest set IDs must be unique")
    if sha256_file(error_labels_path) != packet.error_labels_sha256:
        raise ValueError("evaluation packet error-label source SHA-256 drifted")
    labels_by_key = load_synthetic_error_labels(error_labels_path)

    records: list[DocumentExtractionGoldV2] = []
    for item in packet.items:
        clean_set_id = f"C{item.set_id[1:]}"
        clean_row = clean_by_id.get(clean_set_id)
        if clean_row is None:
            raise ValueError("selected PDF set has no matching clean manifest source row")
        fields = _manifest_field_values(row=clean_row, document_type=item.document_type)
        labels: list[dict[str, str]] = []
        if item.fixture_kind == "error":
            labels = labels_by_key.get((item.set_id, item.document_type), [])
            for label in labels:
                canonical_field = _ERROR_FIELD_NAME[(item.document_type, label["field"])]
                existing = fields.get(canonical_field)
                if existing is None:
                    raise ValueError(
                        "error label field is outside the selected document Gold mapping"
                    )
                fields[canonical_field] = GoldFieldV2(
                    field_name=canonical_field,
                    value=_field_value(label["wrong_value"]),
                    normalizer=existing.normalizer,
                )
        attestation = _attestation(
            clean_manifest_sha256=sha256_file(clean_manifest_path),
            error_labels_sha256=sha256_file(error_labels_path),
            source_set_id=item.set_id,
            labels=labels,
        )
        records.append(
            build_draft_gold_record(
                packet=packet,
                item=item,
                fields=list(fields.values()),
                source_attestation=attestation,
            )
        )
    return build_gold_packet(packet=packet, records=records)


__all__ = [
    "CandidateDocumentOutputV1",
    "CandidateFieldV1",
    "CandidateOutputBundleV1",
    "CandidateSpanV1",
    "CandidateTableCellV1",
    "DocumentExtractionGoldV2",
    "EngineKind",
    "EvaluationPacketV2",
    "EvaluationRunStateV1",
    "FixtureKind",
    "GoldFieldV2",
    "GoldPacketV2",
    "GoldSpanV2",
    "GoldTableCellV2",
    "OcrEvaluationResultV1",
    "OcrQualityMetricsV1",
    "ReviewerApprovalV1",
    "SourceGroundTruthAttestationV1",
    "StratifiedPacketItemV2",
    "SyntheticDocumentType",
    "approve_gold_record",
    "build_draft_gold_record",
    "build_gold_packet",
    "build_stratified_pdf_gold_packets",
    "build_synthetic_pdf_draft_gold",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "canonical_span_text",
    "evaluation_result_sha256",
    "gold_content_sha256",
    "gold_packet_receipt_sha256",
    "ground_truth_receipt_sha256",
    "load_synthetic_error_labels",
    "normalize_field_value",
    "normalize_table_cell_text",
    "packet_receipt_sha256",
    "read_canonical_evaluation_result",
    "receipt_sha256",
    "score_candidate_output",
    "select_best_development_result",
    "sha256_file",
    "validate_evaluation_run_state",
    "validate_gold_packet_for_evaluation",
    "verify_receipt_source_assets",
    "write_evaluation_result",
]
