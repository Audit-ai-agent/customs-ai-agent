"""Finalize one reviewed synthetic-PDF Gold packet in local-only storage.

The loopback review UI intentionally saves only intermediate submissions.  This
manual command is the separate, explicit human-adjudication edge: it verifies
the immutable receipt, packet, seeded source-attested draft, and every review
submission before it produces a canonical ``human_adjudicated`` Gold packet.

It never invokes OCR, accesses a network, starts a server, prints source
content, or writes outside ``_local/data``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, cast

from pydantic import ValidationError

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1, receipt_sha256
from customs_ai_agent.evaluation.document_ocr_quality import (
    DocumentExtractionGoldV2,
    EvaluationPacketV2,
    GoldFieldV2,
    GoldPacketV2,
    GoldSpanV2,
    GoldTableCellV2,
    approve_gold_record,
    build_draft_gold_record,
    build_gold_packet,
    canonical_json_file_bytes,
    verify_receipt_source_assets,
)

try:  # Direct ``python tools/...`` execution and test-module import both work.
    from tools.review_document_gold import LocalGoldReviewDraftV1
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use.
    from review_document_gold import LocalGoldReviewDraftV1


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_HANDOFF_BYTES = 4 * 1024 * 1024


class GoldFinalizationError(ValueError):
    """Content-free failure intended for a local operator-facing summary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _is_below_local_data(path: Path) -> bool:
    parts = [part.casefold() for part in path.resolve().parts]
    return any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1))


def _require_local_file(path: Path, *, code: str) -> Path:
    try:
        if path.is_symlink():
            raise GoldFinalizationError(code)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GoldFinalizationError(code) from error
    if not _is_below_local_data(resolved) or not resolved.is_file():
        raise GoldFinalizationError(code)
    return resolved


def _require_local_directory(path: Path, *, create: bool, code: str) -> Path:
    try:
        if path.exists() and path.is_symlink():
            raise GoldFinalizationError(code)
        resolved = path.resolve()
        if not _is_below_local_data(resolved):
            raise GoldFinalizationError(code)
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        if not resolved.is_dir():
            raise GoldFinalizationError(code)
    except OSError as error:
        raise GoldFinalizationError(code) from error
    return resolved


def _read_model(path: Path, model: type[object], *, code: str) -> object:
    try:
        raw = path.read_bytes()
        validator = getattr(model, "model_validate_json")
        return validator(raw)
    except (
        OSError,
        AttributeError,
        ValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise GoldFinalizationError(code) from error


def _validate_seeded_gold(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    gold_packet: GoldPacketV2,
) -> None:
    """Prove that the input is the unapproved source-attested draft for this packet."""

    if receipt_sha256(receipt) != packet.receipt_sha256:
        raise GoldFinalizationError("packet_receipt_mismatch")
    if (
        gold_packet.status != "draft"
        or gold_packet.packet_split != packet.split
        or gold_packet.dataset_id != packet.dataset_id
        or gold_packet.dataset_revision != packet.dataset_revision
        or gold_packet.packet_receipt_sha256 != packet.packet_receipt_sha256
    ):
        raise GoldFinalizationError("gold_draft_binding_invalid")
    try:
        rebuilt = build_gold_packet(packet=packet, records=gold_packet.records)
    except ValueError as error:
        raise GoldFinalizationError("gold_draft_binding_invalid") from error
    if rebuilt.gold_receipt_sha256 != gold_packet.gold_receipt_sha256:
        raise GoldFinalizationError("gold_draft_binding_invalid")
    if any(record.adjudication_status != "draft" for record in gold_packet.records):
        raise GoldFinalizationError("gold_draft_binding_invalid")


def _validate_operator_identity(*, approval_id: str, reviewer_id: str) -> None:
    if _IDENTIFIER.fullmatch(approval_id) is None or _IDENTIFIER.fullmatch(reviewer_id) is None:
        raise GoldFinalizationError("approval_identity_invalid")


def _read_review_handoffs(
    *,
    review_draft_directory: Path,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    approval_id: str,
    reviewer_id: str,
) -> dict[str, LocalGoldReviewDraftV1]:
    """Load exactly one loopback submission for every selected PDF asset."""

    directory = _require_local_directory(
        review_draft_directory, create=False, code="review_handoff_directory_invalid"
    )
    expected_items = {item.asset_id: item for item in packet.items}
    submitted: dict[str, LocalGoldReviewDraftV1] = {}
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError as error:
        raise GoldFinalizationError("review_handoff_directory_invalid") from error
    for entry in entries:
        if entry.is_symlink() or not entry.is_file() or entry.suffix.casefold() != ".json":
            raise GoldFinalizationError("review_handoff_invalid")
        try:
            raw = entry.read_bytes()
        except OSError as error:
            raise GoldFinalizationError("review_handoff_invalid") from error
        if len(raw) > _MAX_HANDOFF_BYTES:
            raise GoldFinalizationError("review_handoff_invalid")
        try:
            handoff = LocalGoldReviewDraftV1.model_validate_json(raw)
        except ValidationError as error:
            raise GoldFinalizationError("review_handoff_invalid") from error
        if entry.name != f"{handoff.asset_id}.json":
            # Stable file names make a second, differently named duplicate
            # unable to shadow the submitted record.
            raise GoldFinalizationError("review_handoff_duplicate_or_mixed")
        item = expected_items.get(handoff.asset_id)
        if item is None or handoff.asset_id in submitted:
            raise GoldFinalizationError("review_handoff_duplicate_or_mixed")
        if (
            handoff.review_origin != "loopback_review_v1"
            or handoff.review_status != "reviewer_submitted"
            or handoff.reviewer_id != reviewer_id
            or handoff.approval_id != approval_id
            or handoff.dataset_id != receipt.dataset_id
            or handoff.dataset_revision != receipt.dataset_revision
            or handoff.receipt_sha256 != receipt_sha256(receipt)
            or handoff.packet_receipt_sha256 != packet.packet_receipt_sha256
            or handoff.packet_kind != "stratified-v2"
            or handoff.packet_split != packet.split
            or handoff.source_sha256 != item.source_sha256
        ):
            raise GoldFinalizationError("review_handoff_binding_invalid")
        if not handoff.fields or not handoff.spans or not handoff.table_cells:
            raise GoldFinalizationError("review_handoff_incomplete")
        submitted[handoff.asset_id] = handoff
    if set(submitted) != set(expected_items):
        raise GoldFinalizationError("review_handoff_coverage_incomplete")
    return submitted


def _merge_reviewed_record(
    *,
    source_record: DocumentExtractionGoldV2,
    handoff: LocalGoldReviewDraftV1,
    packet: EvaluationPacketV2,
    approval_id: str,
    reviewer_id_sha256: str,
    approved_at: datetime,
) -> DocumentExtractionGoldV2:
    """Keep source attestation/normalizers while replacing reviewer-facing values."""

    source_fields = {field.field_name: field for field in source_record.fields}
    reviewed_fields = {field.field_name: field for field in handoff.fields}
    if set(source_fields) != set(reviewed_fields):
        raise GoldFinalizationError("review_handoff_field_coverage_invalid")
    try:
        fields = [
            GoldFieldV2(
                field_name=source.field_name,
                value=reviewed_fields[source.field_name].value,
                normalizer=source.normalizer,
            )
            for source in source_record.fields
        ]
        spans = [GoldSpanV2.model_validate(span.model_dump(mode="json")) for span in handoff.spans]
        table_cells = [
            GoldTableCellV2.model_validate(cell.model_dump(mode="json"))
            for cell in handoff.table_cells
        ]
        item = next(item for item in packet.items if item.asset_id == source_record.asset_id)
        reviewed_draft = build_draft_gold_record(
            packet=packet,
            item=item,
            fields=fields,
            spans=spans,
            table_cells=table_cells,
            source_attestation=source_record.source_attestation,
        )
        return approve_gold_record(
            reviewed_draft,
            approval_id=approval_id,
            reviewer_id_sha256=reviewer_id_sha256,
            approved_at=approved_at,
        )
    except (StopIteration, ValidationError, ValueError) as error:
        raise GoldFinalizationError("review_handoff_content_invalid") from error


def _write_immutable_final_gold(*, output_directory: Path, gold_packet: GoldPacketV2) -> Path:
    directory = _require_local_directory(
        output_directory, create=True, code="final_gold_output_invalid"
    )
    destination = (
        directory
        / f"synthetic-pdf-{gold_packet.packet_split}-human-gold-{gold_packet.gold_receipt_sha256}.json"
    )
    payload = canonical_json_file_bytes(gold_packet.model_dump(mode="json", warnings=False))
    try:
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise GoldFinalizationError("final_gold_immutable_conflict")
            return destination
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=directory, prefix=".final-gold-", suffix=".tmp", delete=False
        ) as stream:
            stream.write(payload)
            temporary_path = Path(stream.name)
        try:
            # ``replace`` is atomic within the local-only output directory.
            temporary_path.replace(destination)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise
    except GoldFinalizationError:
        raise
    except OSError as error:
        raise GoldFinalizationError("final_gold_write_failed") from error
    return destination


def finalize_synthetic_pdf_gold(
    *,
    receipt_path: Path,
    packet_path: Path,
    seeded_gold_path: Path,
    source_directory: Path,
    review_draft_directory: Path,
    output_directory: Path,
    approval_id: str,
    reviewer_id: str,
    confirm_human_adjudication: bool,
) -> dict[str, object]:
    """Create one complete, locally immutable human-approved Gold packet."""

    if not confirm_human_adjudication:
        raise GoldFinalizationError("human_adjudication_confirmation_required")
    _validate_operator_identity(approval_id=approval_id, reviewer_id=reviewer_id)
    receipt = cast(
        DatasetReceiptV1,
        _read_model(
            _require_local_file(receipt_path, code="receipt_invalid"),
            DatasetReceiptV1,
            code="receipt_invalid",
        ),
    )
    packet = cast(
        EvaluationPacketV2,
        _read_model(
            _require_local_file(packet_path, code="packet_invalid"),
            EvaluationPacketV2,
            code="packet_invalid",
        ),
    )
    seeded_gold = cast(
        GoldPacketV2,
        _read_model(
            _require_local_file(seeded_gold_path, code="gold_draft_invalid"),
            GoldPacketV2,
            code="gold_draft_invalid",
        ),
    )
    _validate_seeded_gold(receipt=receipt, packet=packet, gold_packet=seeded_gold)
    source_root = _require_local_directory(
        source_directory, create=False, code="source_directory_invalid"
    )
    try:
        verify_receipt_source_assets(receipt=receipt, source_directory=source_root)
    except (OSError, ValueError) as error:
        raise GoldFinalizationError("source_receipt_invalid") from error
    handoffs = _read_review_handoffs(
        review_draft_directory=review_draft_directory,
        receipt=receipt,
        packet=packet,
        approval_id=approval_id,
        reviewer_id=reviewer_id,
    )
    reviewer_id_sha256 = hashlib.sha256(reviewer_id.encode("utf-8")).hexdigest()
    approved_at = datetime.now(timezone.utc)
    reviewed_records = [
        _merge_reviewed_record(
            source_record=record,
            handoff=handoffs[record.asset_id],
            packet=packet,
            approval_id=approval_id,
            reviewer_id_sha256=reviewer_id_sha256,
            approved_at=approved_at,
        )
        for record in seeded_gold.records
    ]
    try:
        final_gold = build_gold_packet(packet=packet, records=reviewed_records)
    except ValueError as error:
        raise GoldFinalizationError("final_gold_invalid") from error
    if final_gold.status != "human_adjudicated":
        raise GoldFinalizationError("final_gold_invalid")
    destination = _write_immutable_final_gold(
        output_directory=output_directory, gold_packet=final_gold
    )
    return {
        "status": "human_adjudicated_gold_created",
        "dataset_id": final_gold.dataset_id,
        "dataset_revision": final_gold.dataset_revision,
        "packet_split": final_gold.packet_split,
        "record_count": len(final_gold.records),
        "gold_receipt_sha256": final_gold.gold_receipt_sha256,
        "gold_file": destination.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually finalize loopback-reviewed synthetic PDF Gold in local-only storage."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--seeded-gold", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--review-draft-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument(
        "--confirm-human-adjudication",
        action="store_true",
        help="Required acknowledgement that the named reviewer completed the rendered-PDF review.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = finalize_synthetic_pdf_gold(
            receipt_path=arguments.receipt,
            packet_path=arguments.packet,
            seeded_gold_path=arguments.seeded_gold,
            source_directory=arguments.source_directory,
            review_draft_directory=arguments.review_draft_directory,
            output_directory=arguments.output_directory,
            approval_id=arguments.approval_id,
            reviewer_id=arguments.reviewer_id,
            confirm_human_adjudication=arguments.confirm_human_adjudication,
        )
    except (GoldFinalizationError, OSError, ValueError, json.JSONDecodeError):
        # Do not expose paths, human identity, source fields, PDFs, or OCR text.
        print(
            json.dumps(
                {"status": "failed", "code": "gold_finalization_failed"}, separators=(",", ":")
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
