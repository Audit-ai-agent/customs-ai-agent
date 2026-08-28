from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from customs_ai_agent.evaluation.synthetic_raster import (
    RASTER_PROFILES,
    RasterAssetReceiptV1,
    SyntheticRasterReceiptV1,
    synthetic_raster_receipt_sha256,
)
from customs_ai_agent.evaluation.synthetic_raster_candidates import (
    RasterCandidateBundleV1,
    RasterCandidateObservationV1,
    SyntheticRasterCandidateError,
    build_raster_candidate_shard,
    build_raster_candidate_shard_checkpoint,
    build_raster_candidate_shard_plan,
    build_raster_candidate_execution,
    candidate_bundle_bytes,
    candidate_shard_bytes,
    candidate_shard_plan_sha256,
    iter_raster_candidate_shard_asset_bindings,
    iter_raster_candidate_shard_definitions,
    read_canonical_raster_candidate_bundle,
    read_canonical_raster_candidate_execution,
    read_canonical_raster_candidate_shard_plan,
    read_verified_raster_candidate_shard_checkpoint,
    validate_raster_candidate_bundle_against_receipt,
    validate_raster_candidate_execution,
    validate_raster_candidate_shard_plan_against_receipt,
    write_canonical_raster_candidate_bundle,
    write_canonical_raster_candidate_execution,
    write_canonical_raster_candidate_shard,
    write_canonical_raster_candidate_shard_checkpoint,
    write_canonical_raster_candidate_shard_plan,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _receipt() -> SyntheticRasterReceiptV1:
    return SyntheticRasterReceiptV1(
        source_dataset_id="synthetic-customs-pdf-2000",
        source_dataset_revision="20260811-r1",
        source_receipt_sha256=_HASH_A,
        renderer_revision="pdfium-raster-v1",
        renderer_configuration_sha256=_HASH_B,
        pdfium_version="4.30.0",
        font_manifest_sha256=_HASH_C,
        split_seed_sha256="d" * 64,
        render_recipe_sha256="e" * 64,
        render_seed_sha256="f" * 64,
        profiles=list(RASTER_PROFILES.values()),
        assets=[
            RasterAssetReceiptV1(
                asset_id="asset-1",
                family_id="family-1",
                document_type="commercial_invoice",
                split="calibration",
                profile_id="png-216",
                source_pdf_sha256=_HASH_A,
                image_sha256=_HASH_B,
                asset_relative_path="synthetic-raster-v1/assets/asset-1-png-216.png",
                media_type="image/png",
                byte_size=10,
                width=100,
                height=200,
                coordinate_transform="pdf-points-lower-left-to-normalized-top-left-v1",
                render_recipe_sha256="e" * 64,
                render_seed_sha256="f" * 64,
                gold_relative_path="synthetic-raster-v1/gold/asset-1-png-216.json",
                gold_sha256="1" * 64,
            )
        ],
    )


def _candidate(*, source_sha256: str = _HASH_B) -> dict[str, object]:
    return {
        "document_asset_id": "asset-1",
        "extraction_request_id": "candidate-asset-1",
        "raster_profile_revision": "synthetic-raster-v1",
        "status": "needs_review",
        "media_type": "image/png",
        "byte_size": 10,
        "source_sha256": source_sha256,
        "canonical_text_sha256": hashlib.sha256(b"COMMERCIAL INVOICE").hexdigest(),
        "extractor_revision": "ppstructurev3-local-v1",
        "document_type": "unknown",
        "classification_confidence": None,
        "pages": [
            {
                "page_number": 1,
                "width": 100,
                "height": 200,
                "coordinate_unit": "pixels",
                "spans": [
                    {
                        "span_id": "span-1",
                        "ordinal": 1,
                        "text": "COMMERCIAL INVOICE",
                        "bbox": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.1},
                        "polygon": None,
                        "confidence": 0.99,
                        "reading_order": 1,
                    }
                ],
            }
        ],
        "field_candidates": [],
        "tables": [],
        "warnings": [],
        "needs_review_reasons": [{"code": "document_type_unknown"}],
    }


def _bundle(receipt: SyntheticRasterReceiptV1) -> RasterCandidateBundleV1:
    return RasterCandidateBundleV1(
        split="calibration",
        synthetic_raster_receipt_sha256=synthetic_raster_receipt_sha256(receipt),
        extractor_revision="ppstructurev3-local-v1",
        candidates=[
            RasterCandidateObservationV1(
                asset_id="asset-1", profile_id="png-216", candidate=_candidate()
            )
        ],
    )


def _shard_receipt() -> SyntheticRasterReceiptV1:
    profile_ids = (
        "png-216",
        "jpeg-216-q75",
        "jpeg-150-q75",
        "scan-pdf-216-q75",
        "png-216",
    )
    suffixes = {
        "png-216": ".png",
        "jpeg-216-q75": ".jpg",
        "jpeg-150-q75": ".jpg",
        "scan-pdf-216-q75": ".pdf",
    }
    assets = [
        RasterAssetReceiptV1(
            asset_id=f"asset-{index:03d}",
            family_id=f"family-{index:03d}",
            document_type="commercial_invoice",
            split="calibration",
            profile_id=profile_id,
            source_pdf_sha256=f"{index + 1:x}" * 64,
            image_sha256=f"{index + 6:x}" * 64,
            asset_relative_path=(
                f"synthetic-raster-v1/assets/asset-{index:03d}-{profile_id}{suffixes[profile_id]}"
            ),
            media_type=RASTER_PROFILES[profile_id].media_type,
            byte_size=100 + index,
            width=100,
            height=200,
            coordinate_transform="pdf-points-lower-left-to-normalized-top-left-v1",
            render_recipe_sha256="e" * 64,
            render_seed_sha256="f" * 64,
            gold_relative_path=f"synthetic-raster-v1/gold/asset-{index:03d}-{profile_id}.json",
            gold_sha256=f"{index + 11:x}" * 64,
        )
        for index, profile_id in enumerate(profile_ids)
    ]
    return _receipt().model_copy(update={"assets": assets})


def _shard_observation(
    *, asset_id: str, profile_id: str, image_sha256: str, media_type: str, byte_size: int
) -> RasterCandidateObservationV1:
    candidate = _candidate(source_sha256=image_sha256)
    candidate.update(
        {
            "document_asset_id": asset_id,
            "extraction_request_id": f"candidate-{asset_id}",
            "media_type": media_type,
            "byte_size": byte_size,
        }
    )
    return RasterCandidateObservationV1(
        asset_id=asset_id,
        profile_id=profile_id,  # type: ignore[arg-type]
        candidate=candidate,
    )


def test_candidate_bundle_and_execution_are_canonical_content_addressed(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    receipt = _receipt()
    receipt_sha256 = synthetic_raster_receipt_sha256(receipt)
    bundle = _bundle(receipt)
    relative_path, bundle_sha256, size = write_canonical_raster_candidate_bundle(
        document_preview_root=root, bundle=bundle
    )
    loaded, raw, loaded_sha256 = read_canonical_raster_candidate_bundle(
        document_preview_root=root, relative_path=relative_path, expected_sha256=bundle_sha256
    )
    assert loaded == bundle
    assert loaded_sha256 == bundle_sha256
    assert size == len(raw)
    validate_raster_candidate_bundle_against_receipt(
        bundle=loaded,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        split="calibration",
    )

    execution = build_raster_candidate_execution(
        bundle=loaded,
        candidate_bundle_relative_path_value=relative_path,
        candidate_bundle_raw=raw,
        runtime_configuration_sha256="2" * 64,
    )
    execution_path, execution_sha256 = write_canonical_raster_candidate_execution(
        document_preview_root=root, execution=execution
    )
    loaded_execution, _execution_raw, loaded_execution_sha256 = (
        read_canonical_raster_candidate_execution(
            document_preview_root=root,
            relative_path=execution_path,
            expected_sha256=execution_sha256,
        )
    )
    assert loaded_execution_sha256 == execution_sha256
    validate_raster_candidate_execution(
        execution=loaded_execution,
        bundle=loaded,
        bundle_raw=raw,
        bundle_sha256=bundle_sha256,
        runtime_configuration_sha256="2" * 64,
    )

    # A one-shot activation rerun can reuse identical content-addressed OCR
    # evidence, but cannot alter a pre-existing hash path.
    assert write_canonical_raster_candidate_bundle(
        document_preview_root=root, bundle=bundle
    ) == (relative_path, bundle_sha256, size)
    assert write_canonical_raster_candidate_execution(
        document_preview_root=root, execution=execution
    ) == (execution_path, execution_sha256)
    (root / relative_path).write_bytes(b"different")
    with pytest.raises(SyntheticRasterCandidateError, match="immutable_conflict"):
        write_canonical_raster_candidate_bundle(document_preview_root=root, bundle=bundle)
    with pytest.raises(SyntheticRasterCandidateError, match="hash_mismatch"):
        read_canonical_raster_candidate_bundle(
            document_preview_root=root, relative_path=relative_path, expected_sha256="3" * 64
        )


def test_candidate_bundle_rejects_wrong_asset_binding_and_execution_swap(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    receipt = _receipt()
    receipt_sha256 = synthetic_raster_receipt_sha256(receipt)
    bundle = _bundle(receipt)
    with pytest.raises(SyntheticRasterCandidateError, match="asset_binding"):
        validate_raster_candidate_bundle_against_receipt(
            bundle=bundle.model_copy(
                update={
                    "candidates": [
                        RasterCandidateObservationV1(
                            asset_id="asset-1",
                            profile_id="png-216",
                            candidate=_candidate(source_sha256="9" * 64),
                        )
                    ]
                }
            ),
            receipt=receipt,
            receipt_sha256=receipt_sha256,
            split="calibration",
        )

    relative_path, bundle_sha256, _size = write_canonical_raster_candidate_bundle(
        document_preview_root=root, bundle=bundle
    )
    raw = candidate_bundle_bytes(bundle)
    execution = build_raster_candidate_execution(
        bundle=bundle,
        candidate_bundle_relative_path_value=relative_path,
        candidate_bundle_raw=raw,
        runtime_configuration_sha256="2" * 64,
    )
    with pytest.raises(SyntheticRasterCandidateError, match="binding_invalid"):
        validate_raster_candidate_execution(
            execution=execution.model_copy(update={"split": "sealed"}),
            bundle=bundle,
            bundle_raw=raw,
            bundle_sha256=bundle_sha256,
            runtime_configuration_sha256="2" * 64,
        )


def test_candidate_bundle_accepts_a_fingerprint_bound_runtime_revision() -> None:
    receipt = _receipt()
    fingerprint_revision = "synthetic-raster-v1:" + "a" * 24
    candidate = _candidate()
    candidate["raster_profile_revision"] = fingerprint_revision
    bundle = RasterCandidateBundleV1(
        split="calibration",
        synthetic_raster_receipt_sha256=synthetic_raster_receipt_sha256(receipt),
        raster_profile_revision=fingerprint_revision,
        extractor_revision="ppstructurev3-local-v1",
        candidates=[
            RasterCandidateObservationV1(
                asset_id="asset-1", profile_id="png-216", candidate=candidate
            )
        ],
    )
    validate_raster_candidate_bundle_against_receipt(
        bundle=bundle,
        receipt=receipt,
        receipt_sha256=synthetic_raster_receipt_sha256(receipt),
        split="calibration",
    )


def test_candidate_shards_are_plan_bound_and_resume_only_from_valid_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    receipt = _shard_receipt()
    receipt_sha256 = synthetic_raster_receipt_sha256(receipt)
    plan = build_raster_candidate_shard_plan(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        split="calibration",
        runtime_configuration_sha256="2" * 64,
        raster_profile_revision="synthetic-raster-v1",
        extractor_revision="ppstructurev3-local-v1",
        runner_implementation_sha256="3" * 64,
        shard_size=4,
        evaluation_batch_size=4,
    )
    assert [len(shard.asset_bindings) for shard in iter_raster_candidate_shard_definitions(plan)] == [4, 1]
    assert len(iter_raster_candidate_shard_asset_bindings(plan=plan)) == 5
    assert "gold" not in plan.model_dump_json().casefold()
    validate_raster_candidate_shard_plan_against_receipt(
        plan=plan, receipt=receipt, receipt_sha256=receipt_sha256, split="calibration"
    )
    plan_path, plan_sha256, _plan_size = write_canonical_raster_candidate_shard_plan(
        document_preview_root=root, plan=plan
    )
    loaded_plan, _plan_raw, loaded_plan_sha256 = read_canonical_raster_candidate_shard_plan(
        document_preview_root=root, relative_path=plan_path, expected_sha256=plan_sha256
    )
    assert loaded_plan_sha256 == candidate_shard_plan_sha256(loaded_plan)
    with pytest.raises(SyntheticRasterCandidateError, match="checkpoint_unavailable"):
        read_verified_raster_candidate_shard_checkpoint(
            document_preview_root=root, plan=loaded_plan, shard_ordinal=0
        )

    definition = iter_raster_candidate_shard_definitions(loaded_plan)[0]
    shard = build_raster_candidate_shard(
        plan=loaded_plan,
        shard_ordinal=0,
        candidates=[
            _shard_observation(
                asset_id=binding.asset_id,
                profile_id=binding.profile_id,
                image_sha256=binding.image_sha256,
                media_type=binding.media_type,
                byte_size=binding.byte_size,
            )
            for binding in definition.asset_bindings
        ],
    )
    shard_path, shard_sha256, _shard_size = write_canonical_raster_candidate_shard(
        document_preview_root=root, shard=shard
    )
    checkpoint = build_raster_candidate_shard_checkpoint(
        plan=loaded_plan,
        shard=shard,
        candidate_shard_relative_path_value=shard_path,
        candidate_shard_raw=candidate_shard_bytes(shard),
    )
    checkpoint_path, _checkpoint_sha256 = write_canonical_raster_candidate_shard_checkpoint(
        document_preview_root=root, checkpoint=checkpoint
    )
    loaded_checkpoint, loaded_shard = read_verified_raster_candidate_shard_checkpoint(
        document_preview_root=root, plan=loaded_plan, shard_ordinal=0
    )
    assert loaded_checkpoint.candidate_shard_sha256 == shard_sha256
    assert loaded_shard == shard

    # The path is deterministic for the plan/shard, so an alternate result
    # cannot silently replace a successful resume marker.
    with pytest.raises(SyntheticRasterCandidateError, match="immutable_conflict"):
        write_canonical_raster_candidate_shard_checkpoint(
            document_preview_root=root,
            checkpoint=checkpoint.model_copy(update={"candidate_shard_sha256": "9" * 64}),
        )
    (root / checkpoint_path).write_bytes(b"{")
    with pytest.raises(SyntheticRasterCandidateError, match="checkpoint_invalid"):
        read_verified_raster_candidate_shard_checkpoint(
            document_preview_root=root, plan=loaded_plan, shard_ordinal=0
        )
