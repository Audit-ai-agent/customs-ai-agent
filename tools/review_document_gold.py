"""Loopback-only review UI for local synthetic PDF Gold drafts.

This is deliberately *not* an Agent service endpoint.  It opens a local PDF
from an immutable receipt, lets a reviewer save a local intermediate draft,
and never promotes that draft to human-adjudicated Gold.  PDF bytes and the
reviewer's entered text stay below ``_local/data`` and are not logged.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    EvaluationPacketV1,
    receipt_sha256,
)
from customs_ai_agent.schemas.document_extraction import BoundingBoxV1


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_FIELD_NAME = r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_MAX_DRAFT_BYTES = 4 * 1024 * 1024


class ReviewToolError(ValueError):
    """A deliberately content-free local review error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _LocalReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewPacketItemV2(_LocalReviewModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: Literal["application/pdf"]
    fixture_kind: Literal["clean", "error"]
    document_type: Literal["bill_of_lading", "invoice", "packing_list"]
    set_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)


class ReviewPacketV2(_LocalReviewModel):
    """The core, review-relevant part of the stratified packet v2 contract.

    Selection details such as stratum labels may be added by the Gold pipeline;
    they are intentionally not interpreted by the review UI.  The packet's
    identity, split, receipt binding and asset bindings are mandatory here.
    """

    schema_version: Literal["2.0"]
    packet_kind: Literal["stratified-v2"]
    split: Literal["development", "sealed"]
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    packet_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prepared_as_of: date
    items: list[ReviewPacketItemV2] = Field(min_length=1, max_length=100_000)
    required_gates: list[Literal["field_exact_match", "ocr_cer", "table_cell_f1"]] = Field(
        min_length=3, max_length=3
    )
    release_thresholds: dict[str, float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _items_are_unique(self) -> "ReviewPacketV2":
        asset_ids = [item.asset_id for item in self.items]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("packet asset IDs must be unique")
        if set(self.required_gates) != {"field_exact_match", "ocr_cer", "table_cell_f1"}:
            raise ValueError("packet must retain every required OCR release gate")
        return self


class LocalReviewFieldV1(_LocalReviewModel):
    field_name: str = Field(min_length=1, max_length=128, pattern=_FIELD_NAME)
    value: str = Field(max_length=32_000)
    page_number: int | None = Field(default=None, ge=1, le=1_000)


class LocalReviewSpanV1(_LocalReviewModel):
    page_number: int = Field(ge=1, le=1_000)
    text: str = Field(min_length=1, max_length=32_000)
    reading_order: int = Field(ge=1, le=100_000)
    bbox: BoundingBoxV1 | None = None


class LocalReviewTableCellV1(_LocalReviewModel):
    page_number: int = Field(ge=1, le=1_000)
    table_index: int = Field(ge=0, le=10_000)
    row_index: int = Field(ge=0, le=100_000)
    column_index: int = Field(ge=0, le=10_000)
    text: str = Field(max_length=32_000)


class LocalReviewSubmissionV1(_LocalReviewModel):
    """What the loopback browser may submit; it has no Gold status field."""

    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=_IDENTIFIER)
    approval_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=_IDENTIFIER)
    review_status: Literal["draft", "reviewer_submitted"] = "draft"
    fields: list[LocalReviewFieldV1] = Field(default_factory=list, max_length=10_000)
    spans: list[LocalReviewSpanV1] = Field(default_factory=list, max_length=100_000)
    table_cells: list[LocalReviewTableCellV1] = Field(default_factory=list, max_length=100_000)

    @model_validator(mode="after")
    def _reviewer_submission_has_an_opaque_handoff_identity(self) -> "LocalReviewSubmissionV1":
        if self.review_status == "reviewer_submitted" and (
            self.reviewer_id is None or self.approval_id is None
        ):
            raise ValueError("reviewer_submitted requires reviewer_id and approval_id")
        field_names = [field.field_name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("field names must be unique")
        reading_order = [span.reading_order for span in self.spans]
        if len(reading_order) != len(set(reading_order)):
            raise ValueError("span reading order must be unique")
        cells = [
            (cell.page_number, cell.table_index, cell.row_index, cell.column_index)
            for cell in self.table_cells
        ]
        if len(cells) != len(set(cells)):
            raise ValueError("table cell coordinates must be unique")
        return self


class LocalGoldReviewDraftV1(_LocalReviewModel):
    """Local intermediate only; the Gold pipeline owns final adjudication."""

    schema_version: Literal["local-gold-review-draft-v1"] = "local-gold-review-draft-v1"
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    packet_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    packet_kind: Literal["stratified-v2", "legacy-v1"]
    packet_split: Literal["development", "sealed"]
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=_IDENTIFIER)
    approval_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=_IDENTIFIER)
    review_status: Literal["draft", "reviewer_submitted"] = "draft"
    fields: list[LocalReviewFieldV1] = Field(default_factory=list, max_length=10_000)
    spans: list[LocalReviewSpanV1] = Field(default_factory=list, max_length=100_000)
    table_cells: list[LocalReviewTableCellV1] = Field(default_factory=list, max_length=100_000)


@dataclass(frozen=True)
class ReviewPacket:
    packet_kind: Literal["stratified-v2", "legacy-v1"]
    split: Literal["development", "sealed"]
    dataset_id: str
    dataset_revision: str
    receipt_sha256: str
    packet_receipt_sha256: str
    items: tuple[ReviewPacketItemV2, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _packet_receipt_sha256(value: dict[str, object]) -> str:
    """The Gold pipeline's canonical packet hash excludes its own receipt field."""

    return _sha256_bytes(
        _canonical_json_bytes(
            {key: nested for key, nested in value.items() if key != "packet_receipt_sha256"}
        )
    )


def _is_below_local_data(path: Path) -> bool:
    parts = [part.casefold() for part in path.resolve().parts]
    return any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1))


def _require_local_file(path: Path) -> Path:
    resolved = path.resolve()
    if not _is_below_local_data(resolved) or not resolved.is_file():
        raise ReviewToolError("local_file_required")
    return resolved


def _require_local_directory(path: Path, *, create: bool) -> Path:
    resolved = path.resolve()
    if not _is_below_local_data(resolved):
        raise ReviewToolError("local_data_boundary_required")
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise ReviewToolError("local_directory_required")
    return resolved


def _read_json_file(path: Path, *, code: str) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewToolError(code) from error
    if not isinstance(value, dict):
        raise ReviewToolError(code)
    return value, _sha256_bytes(raw)


def _parse_packet(raw: dict[str, object]) -> ReviewPacket:
    """Read v2 first; v1 is accepted only when its exact old shape validates."""

    if raw.get("packet_kind") == "stratified-v2":
        try:
            parsed = ReviewPacketV2.model_validate(raw)
        except ValidationError as error:
            raise ReviewToolError("packet_invalid") from error
        return ReviewPacket(
            packet_kind="stratified-v2",
            split=parsed.split,
            dataset_id=parsed.dataset_id,
            dataset_revision=parsed.dataset_revision,
            receipt_sha256=parsed.receipt_sha256,
            packet_receipt_sha256=parsed.packet_receipt_sha256,
            items=tuple(parsed.items),
        )
    try:
        parsed_v1 = EvaluationPacketV1.model_validate(raw)
    except ValidationError as error:
        raise ReviewToolError("packet_invalid") from error
    return ReviewPacket(
        packet_kind="legacy-v1",
        split=parsed_v1.split,
            dataset_id=parsed_v1.dataset_id,
            dataset_revision=parsed_v1.dataset_revision,
            receipt_sha256=parsed_v1.receipt_sha256,
            packet_receipt_sha256=_sha256_bytes(_canonical_json_bytes(raw)),
            items=tuple(
                ReviewPacketItemV2(
                    asset_id=item.asset_id,
                    source_sha256=item.source_sha256,
                    media_type=item.media_type,
                    fixture_kind="clean",
                    document_type="invoice",
                    set_id="legacy-v1",
            )
            for item in parsed_v1.items
        ),
    )


def _validate_loopback_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ReviewToolError("loopback_host_required") from error
    if not address.is_loopback:
        raise ReviewToolError("loopback_host_required")
    return str(address)


@dataclass(frozen=True)
class ReviewContext:
    receipt_path: Path
    packet_path: Path
    dataset_root: Path
    output_directory: Path
    draft_directory: Path
    receipt_file_sha256: str
    packet_file_sha256: str
    receipt: DatasetReceiptV1
    packet: ReviewPacket
    assets: dict[str, DatasetAssetReceiptV1]
    packet_items: dict[str, ReviewPacketItemV2]

    @classmethod
    def from_paths(
        cls,
        *,
        receipt_path: Path,
        packet_path: Path,
        dataset_root: Path,
        output_directory: Path,
    ) -> "ReviewContext":
        safe_receipt_path = _require_local_file(receipt_path)
        safe_packet_path = _require_local_file(packet_path)
        safe_dataset_root = _require_local_directory(dataset_root, create=False)
        safe_output_directory = _require_local_directory(output_directory, create=True)
        receipt_raw, receipt_file_sha = _read_json_file(safe_receipt_path, code="receipt_invalid")
        packet_raw, packet_file_sha = _read_json_file(safe_packet_path, code="packet_invalid")
        try:
            receipt = DatasetReceiptV1.model_validate(receipt_raw)
        except ValidationError as error:
            raise ReviewToolError("receipt_invalid") from error
        packet = _parse_packet(packet_raw)
        if (
            packet.dataset_id != receipt.dataset_id
            or packet.dataset_revision != receipt.dataset_revision
            or packet.receipt_sha256 != receipt_sha256(receipt)
        ):
            raise ReviewToolError("packet_receipt_mismatch")
        if packet.packet_kind == "stratified-v2" and packet.packet_receipt_sha256 != _packet_receipt_sha256(
            packet_raw
        ):
            raise ReviewToolError("packet_receipt_mismatch")
        assets = {asset.asset_id: asset for asset in receipt.assets}
        packet_items = {item.asset_id: item for item in packet.items}
        for item in packet.items:
            asset = assets.get(item.asset_id)
            if asset is None or asset.sha256 != item.source_sha256:
                raise ReviewToolError("packet_asset_mismatch")
            if asset.media_type != "application/pdf" or item.media_type != "application/pdf":
                raise ReviewToolError("pdf_packet_required")
        return cls(
            receipt_path=safe_receipt_path,
            packet_path=safe_packet_path,
            dataset_root=safe_dataset_root,
            output_directory=safe_output_directory,
            draft_directory=safe_output_directory / "drafts",
            receipt_file_sha256=receipt_file_sha,
            packet_file_sha256=packet_file_sha,
            receipt=receipt,
            packet=packet,
            assets=assets,
            packet_items=packet_items,
        )

    def _assert_control_files_unchanged(self) -> None:
        try:
            receipt_sha = _sha256_file(self.receipt_path)
            packet_sha = _sha256_file(self.packet_path)
        except OSError as error:
            raise ReviewToolError("control_file_unavailable") from error
        if receipt_sha != self.receipt_file_sha256 or packet_sha != self.packet_file_sha256:
            raise ReviewToolError("control_file_changed")

    def asset(self, asset_id: str) -> tuple[DatasetAssetReceiptV1, ReviewPacketItemV2]:
        self._assert_control_files_unchanged()
        asset = self.assets.get(asset_id)
        item = self.packet_items.get(asset_id)
        if asset is None or item is None:
            raise ReviewToolError("asset_not_in_packet")
        return asset, item

    def _source_path(self, asset: DatasetAssetReceiptV1) -> Path:
        candidate = self.dataset_root.joinpath(*asset.relative_path.split("/"))
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.dataset_root) or not resolved.is_file():
            raise ReviewToolError("source_unavailable")
        return resolved

    def _verify_source(self, asset: DatasetAssetReceiptV1, item: ReviewPacketItemV2) -> Path:
        path = self._source_path(asset)
        try:
            byte_size = path.stat().st_size
            source_sha = _sha256_file(path)
        except OSError as error:
            raise ReviewToolError("source_unavailable") from error
        if byte_size != asset.byte_size or source_sha != asset.sha256 or source_sha != item.source_sha256:
            raise ReviewToolError("source_sha_mismatch")
        return path

    def read_pdf(self, asset_id: str) -> bytes:
        asset, item = self.asset(asset_id)
        path = self._verify_source(asset, item)
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ReviewToolError("source_unavailable") from error
        if (
            len(content) != asset.byte_size
            or _sha256_bytes(content) != asset.sha256
            or not content.startswith(b"%PDF-")
        ):
            raise ReviewToolError("source_sha_mismatch")
        return content

    def verify_submission_source(self, asset_id: str, source_sha256: str) -> DatasetAssetReceiptV1:
        asset, item = self.asset(asset_id)
        if source_sha256 != asset.sha256:
            raise ReviewToolError("submission_source_mismatch")
        self._verify_source(asset, item)
        return asset

    def _safe_draft_directory(self) -> Path:
        try:
            self.draft_directory.mkdir(parents=True, exist_ok=True)
            resolved = self.draft_directory.resolve()
        except OSError as error:
            raise ReviewToolError("draft_write_failed") from error
        if not resolved.is_relative_to(self.output_directory) or not _is_below_local_data(resolved):
            raise ReviewToolError("local_data_boundary_required")
        return resolved

    def draft_path(self, asset_id: str) -> Path:
        # asset IDs have already passed the packet's identifier validation.
        return self._safe_draft_directory() / f"{asset_id}.json"

    def _bound_draft(self, asset_id: str, raw: dict[str, object]) -> LocalGoldReviewDraftV1:
        try:
            draft = LocalGoldReviewDraftV1.model_validate(raw)
        except ValidationError as error:
            raise ReviewToolError("draft_invalid") from error
        asset, _ = self.asset(asset_id)
        if (
            draft.dataset_id != self.receipt.dataset_id
            or draft.dataset_revision != self.receipt.dataset_revision
            or draft.receipt_sha256 != receipt_sha256(self.receipt)
            or draft.packet_receipt_sha256 != self.packet.packet_receipt_sha256
            or draft.packet_kind != self.packet.packet_kind
            or draft.packet_split != self.packet.split
            or draft.asset_id != asset_id
            or draft.source_sha256 != asset.sha256
        ):
            raise ReviewToolError("draft_binding_invalid")
        return draft

    def load_draft(self, asset_id: str) -> LocalGoldReviewDraftV1 | None:
        self.verify_submission_source(asset_id, self.asset(asset_id)[0].sha256)
        path = self.draft_path(asset_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReviewToolError("draft_invalid") from error
        if not isinstance(raw, dict):
            raise ReviewToolError("draft_invalid")
        return self._bound_draft(asset_id, raw)

    def save_draft(self, asset_id: str, raw_submission: object) -> LocalGoldReviewDraftV1:
        try:
            submission = LocalReviewSubmissionV1.model_validate(raw_submission)
        except ValidationError as error:
            raise ReviewToolError("submission_invalid") from error
        asset = self.verify_submission_source(asset_id, submission.source_sha256)
        draft = LocalGoldReviewDraftV1(
            dataset_id=self.receipt.dataset_id,
            dataset_revision=self.receipt.dataset_revision,
            receipt_sha256=receipt_sha256(self.receipt),
            packet_receipt_sha256=self.packet.packet_receipt_sha256,
            packet_kind=self.packet.packet_kind,
            packet_split=self.packet.split,
            asset_id=asset_id,
            source_sha256=asset.sha256,
            reviewer_id=submission.reviewer_id,
            approval_id=submission.approval_id,
            review_status=submission.review_status,
            fields=submission.fields,
            spans=submission.spans,
            table_cells=submission.table_cells,
        )
        encoded = _canonical_json_bytes(draft.model_dump(mode="json", warnings=False)) + b"\n"
        if len(encoded) > _MAX_DRAFT_BYTES:
            raise ReviewToolError("draft_too_large")
        destination = self.draft_path(asset_id)
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.draft_directory, prefix=".draft-", suffix=".tmp", delete=False
            ) as stream:
                stream.write(encoded)
                temporary_path = Path(stream.name)
            temporary_path.replace(destination)
        except OSError as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except (OSError, UnboundLocalError):
                pass
            raise ReviewToolError("draft_write_failed") from error
        try:
            # Close the only meaningful race between the pre-write source check
            # and the local draft becoming visible.
            self._verify_source(asset, self.packet_items[asset_id])
        except ReviewToolError:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return draft

    def handoff(self, asset_id: str) -> dict[str, object]:
        """Return a deterministic local handoff; it never sets Gold status."""

        draft = self.load_draft(asset_id)
        if draft is None:
            raise ReviewToolError("draft_unavailable")
        return {
            "schema_version": "local-gold-review-handoff-v1",
            "handoff_status": "reviewer_submitted"
            if draft.review_status == "reviewer_submitted"
            else "draft",
            "dataset_id": draft.dataset_id,
            "dataset_revision": draft.dataset_revision,
            "receipt_sha256": draft.receipt_sha256,
            "packet_receipt_sha256": draft.packet_receipt_sha256,
            "packet_kind": draft.packet_kind,
            "packet_split": draft.packet_split,
            "asset_id": draft.asset_id,
            "source_sha256": draft.source_sha256,
            "reviewer_id": draft.reviewer_id,
            "approval_id": draft.approval_id,
            "review_status": draft.review_status,
            "fields": [field.model_dump(mode="json") for field in draft.fields],
            "spans": [span.model_dump(mode="json") for span in draft.spans],
            "table_cells": [cell.model_dump(mode="json") for cell in draft.table_cells],
        }


_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>로컬 PDF Gold 검수</title>
  <style>
    body{font-family:system-ui,sans-serif;margin:1rem;max-width:1500px} .layout{display:grid;grid-template-columns:18rem 1fr 1fr;gap:1rem}
    button{display:block;margin:.4rem 0;width:100%;text-align:left} iframe{width:100%;height:75vh;border:1px solid #888}
    textarea,input,select{box-sizing:border-box;width:100%;margin:.3rem 0;font:inherit} textarea{min-height:9rem} .note{color:#555}.status{min-height:1.5rem}
  </style>
</head>
<body>
  <h1>로컬 PDF Gold 검수</h1>
  <p class="note">이 화면은 loopback에서만 열립니다. 저장은 local intermediate draft이며 Gold 승인·OCR 활성화는 수행하지 않습니다.</p>
  <div class="layout">
    <section><h2>대상 PDF</h2><div id="items"></div></section>
    <section><h2 id="title">문서를 선택하세요</h2><iframe id="pdf" title="검수 PDF"></iframe></section>
    <section>
      <h2>검수 초안</h2>
      <label>검수자 식별자<input id="reviewer" maxlength="128"></label>
      <label>승인/제출 식별자<input id="approval" maxlength="128"></label>
      <label>상태<select id="state"><option value="draft">초안</option><option value="reviewer_submitted">검수 제출</option></select></label>
      <label>필드 JSON<textarea id="fields" spellcheck="false">[]</textarea></label>
      <label>span JSON<textarea id="spans" spellcheck="false">[]</textarea></label>
      <label>표 cell JSON<textarea id="cells" spellcheck="false">[]</textarea></label>
      <button id="save" type="button">로컬 검수 초안 저장</button>
      <p id="status" class="status" aria-live="polite"></p>
    </section>
  </div>
  <script>
    let selected = null;
    const byId = (id) => document.getElementById(id);
    const message = (value) => { byId('status').textContent = value; };
    const parseJson = (id) => JSON.parse(byId(id).value);
    const selectItem = async (item) => {
      selected = item; byId('title').textContent = item.asset_id; byId('pdf').src = '/api/items/' + encodeURIComponent(item.asset_id) + '/pdf';
      const response = await fetch('/api/drafts/' + encodeURIComponent(item.asset_id));
      const body = await response.json();
      if (!response.ok) { message('초안을 불러올 수 없습니다: ' + body.code); return; }
      const draft = body.draft;
      byId('reviewer').value = draft ? (draft.reviewer_id || '') : '';
      byId('approval').value = draft ? (draft.approval_id || '') : '';
      byId('state').value = draft ? draft.review_status : 'draft';
      byId('fields').value = JSON.stringify(draft ? draft.fields : [], null, 2);
      byId('spans').value = JSON.stringify(draft ? draft.spans : [], null, 2);
      byId('cells').value = JSON.stringify(draft ? draft.table_cells : [], null, 2);
      message('');
    };
    const loadItems = async () => {
      const response = await fetch('/api/items'); const body = await response.json();
      if (!response.ok) { message('대상 목록을 불러올 수 없습니다: ' + body.code); return; }
      for (const item of body.items) { const button = document.createElement('button'); button.type = 'button'; button.textContent = item.asset_id; button.onclick = () => selectItem(item); byId('items').appendChild(button); }
    };
    byId('save').onclick = async () => {
      if (!selected) { message('먼저 PDF를 선택하세요.'); return; }
      let payload;
      try { payload = { source_sha256: selected.source_sha256, reviewer_id: byId('reviewer').value || null, approval_id: byId('approval').value || null, review_status: byId('state').value, fields: parseJson('fields'), spans: parseJson('spans'), table_cells: parseJson('cells') }; }
      catch (_) { message('JSON 형식을 확인하세요.'); return; }
      const response = await fetch('/api/drafts/' + encodeURIComponent(selected.asset_id), {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
      const body = await response.json(); message(response.ok ? '로컬 초안을 저장했습니다.' : '저장할 수 없습니다: ' + body.code);
    };
    loadItems();
  </script>
</body></html>"""


def create_review_app(context: ReviewContext) -> FastAPI:
    """Build an in-process loopback review app; callers still choose the host."""

    app = FastAPI(
        title="Local PDF Gold review",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def _no_store(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(ReviewToolError)
    async def _review_error(_: Request, error: ReviewToolError) -> JSONResponse:
        status_code = 404 if error.code in {"asset_not_in_packet", "draft_unavailable"} else 409
        return JSONResponse(status_code=status_code, content={"status": "failed", "code": error.code})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/api/items")
    async def items() -> dict[str, object]:
        context._assert_control_files_unchanged()
        return {
            "status": "ready",
            "packet_kind": context.packet.packet_kind,
            "split": context.packet.split,
            "items": [
                {"asset_id": item.asset_id, "source_sha256": item.source_sha256}
                for item in context.packet.items
            ],
        }

    @app.get("/api/items/{asset_id}/pdf")
    async def pdf(asset_id: str) -> Response:
        return Response(
            content=context.read_pdf(asset_id),
            media_type="application/pdf",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/drafts/{asset_id}")
    async def draft(asset_id: str) -> dict[str, object]:
        loaded = context.load_draft(asset_id)
        return {
            "status": "ready",
            "draft": None if loaded is None else loaded.model_dump(mode="json", warnings=False),
        }

    @app.post("/api/drafts/{asset_id}")
    async def save_draft(asset_id: str, request: Request) -> dict[str, object]:
        if request.headers.get("content-type", "").split(";", 1)[0].casefold() != "application/json":
            raise ReviewToolError("submission_invalid")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ReviewToolError("submission_invalid") from error
        saved = context.save_draft(asset_id, payload)
        return {
            "status": "saved",
            "asset_id": saved.asset_id,
            "source_sha256": saved.source_sha256,
            "review_status": saved.review_status,
        }

    @app.get("/api/drafts/{asset_id}/handoff")
    async def handoff(asset_id: str) -> dict[str, object]:
        return context.handoff(asset_id)

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a loopback-only local PDF Gold review UI.")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        host = _validate_loopback_host(arguments.host)
        if not 1024 <= arguments.port <= 65535:
            raise ReviewToolError("port_invalid")
        context = ReviewContext.from_paths(
            receipt_path=arguments.receipt,
            packet_path=arguments.packet,
            dataset_root=arguments.dataset_root,
            output_directory=arguments.output_directory,
        )
    except (OSError, ReviewToolError, ValueError):
        # Do not expose local paths, source text, PDF bytes, or validation details.
        print(json.dumps({"status": "failed", "code": "configuration_invalid"}, separators=(",", ":")))
        return 2
    print(json.dumps({"status": "ready", "listen": "loopback"}, separators=(",", ":")))
    uvicorn.run(create_review_app(context), host=host, port=arguments.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
