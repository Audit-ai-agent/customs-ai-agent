from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from customs_ai_agent.evaluation.synthetic_raster import (
    RASTER_PROFILES,
    RasterAssetReceiptV1,
    SyntheticRasterReceiptV1,
    synthetic_raster_receipt_sha256,
)
from customs_ai_agent.evaluation.synthetic_raster_candidates import RasterCandidateObservationV1
from customs_ai_agent.service.raster_document_preview_runtime import PpStructureV3OfflineProfileV2

import tools.run_ppstructurev3_synthetic_raster_candidates as runner


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_REVISION = "synthetic-raster-v1:" + "d" * 24
_EXTRACTOR = "ppstructurev3-local-v1"


def _receipt() -> SyntheticRasterReceiptV1:
    return SyntheticRasterReceiptV1(
        source_dataset_id="synthetic-customs-pdf-2000",
        source_dataset_revision="20260811-r1",
        source_receipt_sha256=_HASH_A,
        renderer_revision="pdfium-raster-v1",
        renderer_configuration_sha256=_HASH_B,
        pdfium_version="4.30.0",
        font_manifest_sha256=_HASH_C,
        split_seed_sha256="e" * 64,
        render_recipe_sha256="f" * 64,
        render_seed_sha256="1" * 64,
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
                render_recipe_sha256="f" * 64,
                render_seed_sha256="1" * 64,
                gold_relative_path="synthetic-raster-v1/gold/asset-1-png-216.json",
                gold_sha256="2" * 64,
            )
        ],
    )


def _observation() -> RasterCandidateObservationV1:
    return RasterCandidateObservationV1.model_validate(
        {
            "asset_id": "asset-1",
            "profile_id": "png-216",
            "candidate": {
                "document_asset_id": "asset-1",
                "extraction_request_id": "candidate-asset-1",
                "raster_profile_revision": _REVISION,
                "status": "needs_review",
                "media_type": "image/png",
                "byte_size": 10,
                "source_sha256": _HASH_B,
                "canonical_text_sha256": sha256(b"").hexdigest(),
                "extractor_revision": _EXTRACTOR,
                "document_type": "unknown",
                "classification_confidence": None,
                "pages": [
                    {
                        "page_number": 1,
                        "width": 100,
                        "height": 200,
                        "coordinate_unit": "pixels",
                        "spans": [],
                    }
                ],
                "field_candidates": [],
                "tables": [],
                "warnings": [],
                "needs_review_reasons": [{"code": "document_type_unknown"}],
            },
        }
    )


def test_runner_reuses_only_a_valid_completed_shard(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    receipt = _receipt()
    receipt_sha256 = synthetic_raster_receipt_sha256(receipt)
    profile = PpStructureV3OfflineProfileV2.model_construct(
        raster_profile_revision=_REVISION,
        extractor_revision=_EXTRACTOR,
        pdfium_version="4.30.0",
        evaluation_batch_size=4,
    )
    resolution = SimpleNamespace(
        status="available",
        engine=object(),
        profile=profile,
        runtime_configuration_sha256="3" * 64,
    )
    monkeypatch.setattr(
        runner,
        "_load_receipt",
        lambda **_kwargs: (receipt, receipt_sha256),
    )
    monkeypatch.setattr(runner, "_verified_asset_bytes", lambda **_kwargs: b"synthetic")

    session_calls: list[int] = []

    @contextmanager
    def _evaluation_engine(_resolution):
        session_calls.append(1)
        yield SimpleNamespace(profile=profile)

    evaluation_calls: list[tuple[str, ...]] = []

    def _evaluate_assets(*, assets, **_kwargs):
        evaluation_calls.append(tuple(asset.asset_id for asset in assets))
        return [_observation()]

    monkeypatch.setattr(runner, "_evaluation_engine", _evaluation_engine)
    monkeypatch.setattr(runner, "_evaluate_assets", _evaluate_assets)

    def resolver(**_kwargs):
        return resolution

    first = runner.run_ppstructurev3_synthetic_raster_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path="runtime/profile.json",
        synthetic_raster_receipt_relative_path="synthetic-raster-v1/synthetic-raster-receipt.json",
        split="calibration",
        resolver=resolver,
    )
    assert first["candidate_shard_count"] == 1
    assert first["reused_candidate_shard_count"] == 0
    assert evaluation_calls == [("asset-1",)]
    assert session_calls == [1]

    second = runner.run_ppstructurev3_synthetic_raster_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path="runtime/profile.json",
        synthetic_raster_receipt_relative_path="synthetic-raster-v1/synthetic-raster-receipt.json",
        split="calibration",
        resolver=resolver,
    )
    assert second["candidate_bundle_sha256"] == first["candidate_bundle_sha256"]
    assert second["reused_candidate_shard_count"] == 1
    assert evaluation_calls == [("asset-1",)]
    assert session_calls == [1]

    checkpoint_path = next(
        root.joinpath("synthetic-raster-v1", "candidate-shard-checkpoints").rglob("*.json")
    )
    checkpoint_path.write_bytes(b"{")
    with pytest.raises(runner.SyntheticRasterCandidateRunnerError, match="checkpoint_invalid"):
        runner.run_ppstructurev3_synthetic_raster_candidates(
            document_preview_root=root,
            runtime_configuration_relative_path="runtime/profile.json",
            synthetic_raster_receipt_relative_path="synthetic-raster-v1/synthetic-raster-receipt.json",
            split="calibration",
            resolver=resolver,
        )
    assert evaluation_calls == [("asset-1",)]
    assert session_calls == [1]
