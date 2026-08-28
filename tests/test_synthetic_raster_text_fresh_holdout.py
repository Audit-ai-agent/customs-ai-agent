from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.evaluation.synthetic_raster import RASTER_PROFILES
from customs_ai_agent.evaluation.synthetic_raster_text import RasterTextGoldArtifactV1
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionResultV1,
    RasterTextInputV1,
    RasterTextPageV1,
    canonical_raster_text,
    canonical_raster_text_sha256,
)


_PROFILE_REVISION = "synthetic-raster-text-v1:0123456789abcdef01234567"
_EXTRACTOR_REVISION = "fresh-holdout-test-v1"


def _complete_receipt_assets() -> list[Any]:
    assets: list[Any] = []
    suffixes = (
        ("bl", "bill_of_lading"),
        ("inv", "commercial_invoice"),
        ("pl", "packing_list"),
    )
    for family_number in range(1, 1_001):
        family_id = f"family-{family_number}"
        split = (
            "train"
            if family_number <= 750
            else "development"
            if family_number <= 850
            else "calibration"
            if family_number <= 900
            else "sealed"
        )
        for fixture_kind in ("clean", "error"):
            for suffix, document_type in suffixes:
                asset_id = f"{fixture_kind}-{family_number:04d}-{suffix}"
                for profile_id in RASTER_PROFILES:
                    assets.append(
                        SimpleNamespace(
                            asset_id=asset_id,
                            family_id=family_id,
                            document_type=document_type,
                            split=split,
                            profile_id=profile_id,
                        )
                    )
    return assets


@pytest.fixture(scope="module")
def complete_receipt() -> SimpleNamespace:
    return SimpleNamespace(assets=_complete_receipt_assets())


def test_fresh_selection_is_source_bound_deterministic_and_leak_free(
    complete_receipt: SimpleNamespace,
) -> None:
    first = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    repeated = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    changed_source = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="c" * 64,
    )
    original_non_train = {
        asset.family_id for asset in complete_receipt.assets if asset.split != "train"
    }
    calibration = set(first.fresh_calibration_family_ids)
    sealed = set(first.fresh_sealed_family_ids)

    assert first == repeated
    assert len(calibration) == 50
    assert len(sealed) == 100
    assert not calibration & sealed
    assert not (calibration | sealed) & original_non_train
    assert first.source_receipt_sha256 == "b" * 64
    assert (
        first.fresh_calibration_family_ids,
        first.fresh_sealed_family_ids,
    ) != (
        changed_source.fresh_calibration_family_ids,
        changed_source.fresh_sealed_family_ids,
    )
    assert activation._digest_model(first) != activation._digest_model(  # noqa: SLF001
        changed_source
    )


def test_fresh_v2_selection_is_parent_bound_exact_and_leak_free(
    complete_receipt: SimpleNamespace,
) -> None:
    parent = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    first = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    repeated = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    changed_source = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="c" * 64,
    )
    split_by_family = {
        asset.family_id: asset.split for asset in complete_receipt.assets
    }
    parent_families = set(parent.fresh_calibration_family_ids) | set(
        parent.fresh_sealed_family_ids
    )
    non_train = {
        family_id for family_id, split in split_by_family.items() if split != "train"
    }
    excluded = sorted(non_train | parent_families)
    eligible = sorted(set(split_by_family) - set(excluded))

    assert first == repeated
    assert first.selection_seed_sha256 == (
        "7e13eabee7d20c7cf23e256f1f0d9a9f763d5757eef6337105cc19e0f3b39ccf"
    )
    assert first.parent_selection_receipt_sha256 == activation._digest_model(  # noqa: SLF001
        parent
    )
    assert first.parent_selection_receipt_relative_path.endswith(
        f"/{first.parent_selection_receipt_sha256}.json"
    )
    assert first.fresh_calibration_family_ids == parent.fresh_calibration_family_ids
    assert first.excluded_family_count == len(excluded) == 400
    assert first.eligible_family_count == len(eligible) == 600
    assert first.excluded_family_ids_sha256 == (
        "a55075a682f9b90efdf3f45078014b2d0cdb8b9f086ce32df2a4c6fabd1ae3fb"
    )
    assert first.eligible_family_ids_sha256 == (
        "ab7fdfd4c1428f1e4fe95e47f3cdc8fadea5eabff5ce7067b87a6e1ed58d9dc2"
    )
    assert hashlib.sha256(
        activation._canonical_json(first.fresh_sealed_family_ids)  # noqa: SLF001
    ).hexdigest() == (
        "d65095d4c55883d3195aef3aca076e192faaab819173da8fe18cb1a0964d1816"
    )
    assert activation._digest_model(first) == (  # noqa: SLF001
        "3a32c1dd92d84d43127c6cd186df85bff3aa80e0a98bc7108806d11986c3e145"
    )
    assert set(first.fresh_sealed_family_ids).issubset(eligible)
    assert not set(first.fresh_sealed_family_ids) & set(excluded)
    assert not set(first.fresh_sealed_family_ids) & set(
        first.fresh_calibration_family_ids
    )
    assert changed_source.parent_selection_receipt_sha256 != (
        first.parent_selection_receipt_sha256
    )
    assert changed_source.fresh_sealed_family_ids != first.fresh_sealed_family_ids


@pytest.mark.parametrize(
    ("scope", "expected_assets", "expected_per_stratum"),
    [
        ("fresh-calibration", 1_200, 100),
        ("fresh-sealed", 2_400, 200),
    ],
)
def test_fresh_scope_has_complete_family_and_stratum_coverage(
    complete_receipt: SimpleNamespace,
    scope: str,
    expected_assets: int,
    expected_per_stratum: int,
) -> None:
    selection = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    assets = activation._fresh_scope_assets(  # noqa: SLF001
        receipt=complete_receipt,
        selection=selection,
        scope=scope,
    )

    assert len(assets) == expected_assets
    assert {asset.split for asset in assets} == {"train"}
    assert {
        (document_type, profile_id): sum(
            asset.document_type == document_type and asset.profile_id == profile_id
            for asset in assets
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    } == {
        (document_type, profile_id): expected_per_stratum
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    }


@pytest.mark.parametrize(
    ("scope", "expected_assets", "expected_per_stratum"),
    [
        ("fresh-calibration", 1_200, 100),
        ("fresh-sealed", 2_400, 200),
    ],
)
def test_fresh_v2_scope_has_complete_family_and_stratum_coverage(
    complete_receipt: SimpleNamespace,
    scope: str,
    expected_assets: int,
    expected_per_stratum: int,
) -> None:
    selection = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    assets = activation._fresh_scope_assets(  # noqa: SLF001
        receipt=complete_receipt,
        selection=selection,
        scope=scope,
    )

    assert len(assets) == expected_assets
    assert len({asset.family_id for asset in assets}) * 24 == expected_assets
    assert {asset.split for asset in assets} == {"train"}
    assert {
        (document_type, profile_id): sum(
            asset.document_type == document_type and asset.profile_id == profile_id
            for asset in assets
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    } == {
        (document_type, profile_id): expected_per_stratum
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    }


@pytest.mark.parametrize(
    ("scope", "projection_kind", "expected_assets", "expected_per_stratum"),
    [
        ("fresh-calibration", "calibration-10-family-v1", 240, 20),
        ("fresh-sealed", "sealed-stage1-50-family-v1", 1_200, 100),
    ],
)
def test_fresh_projection_keeps_complete_families_and_full_gold_identity(
    tmp_path: Path,
    complete_receipt: SimpleNamespace,
    scope: str,
    projection_kind: str,
    expected_assets: int,
    expected_per_stratum: int,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    selection = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    selection_sha = activation._digest_model(selection)  # noqa: SLF001
    context = activation._FreshScopeContext(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt=SimpleNamespace(),
        source_receipt_sha256="b" * 64,
        selection=selection,
        selection_relative_path=(
            "synthetic-raster-text-v1/fresh-scope-selections/"
            f"{selection_sha}.json"
        ),
        selection_sha256=selection_sha,
        assets=activation._fresh_scope_assets(  # noqa: SLF001
            receipt=complete_receipt,
            selection=selection,
            scope=scope,
        ),
    )
    projection = activation._build_fresh_evaluation_projection(  # noqa: SLF001
        root=root,
        context=context,
        source_gold_receipt_relative_path=(
            f"synthetic-raster-text-v1/fresh-source-gold-receipts/{scope}/"
            f"{'c' * 64}.json"
        ),
        source_gold_receipt_sha256="c" * 64,
        projection_kind=projection_kind,
    )
    assets = activation._fresh_projection_assets(  # noqa: SLF001
        context=context, projection=projection
    )

    assert len(assets) == expected_assets
    assert projection.full_source_gold_receipt_sha256 == "c" * 64
    assert {asset.family_id for asset in assets} == set(projection.family_ids)
    assert {
        (document_type, profile_id): sum(
            asset.document_type == document_type and asset.profile_id == profile_id
            for asset in assets
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    } == {
        (document_type, profile_id): expected_per_stratum
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in RASTER_PROFILES
    }
    source_families = (
        selection.fresh_calibration_family_ids
        if scope == "fresh-calibration"
        else selection.fresh_sealed_family_ids
    )
    replacement = next(
        family_id for family_id in source_families if family_id not in projection.family_ids
    )
    tampered = projection.model_copy(
        update={"family_ids": sorted([*projection.family_ids[1:], replacement])}
    )
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_projection_invalid$",
    ):
        activation._fresh_projection_assets(  # noqa: SLF001
            context=context, projection=tampered
        )


def test_fresh_v2_projection_binds_only_the_v2_selection(
    tmp_path: Path,
    complete_receipt: SimpleNamespace,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    v1 = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    v2 = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    v2_sha = activation._digest_model(v2)  # noqa: SLF001
    v2_context = activation._FreshScopeContext(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt=SimpleNamespace(),
        source_receipt_sha256="b" * 64,
        selection=v2,
        selection_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{v2_sha}.json"
        ),
        selection_sha256=v2_sha,
        assets=activation._fresh_scope_assets(  # noqa: SLF001
            receipt=complete_receipt,
            selection=v2,
            scope="fresh-sealed",
        ),
    )
    projection = activation._build_fresh_evaluation_projection(  # noqa: SLF001
        root=root,
        context=v2_context,
        source_gold_receipt_relative_path=(
            "synthetic-raster-text-v1/fresh-source-gold-receipts/fresh-sealed/"
            f"{'c' * 64}.json"
        ),
        source_gold_receipt_sha256="c" * 64,
        projection_kind="sealed-stage1-50-family-v1",
    )

    assert projection.selection_receipt_sha256 == v2_sha
    assert projection.selection_receipt_relative_path == (
        v2_context.selection_relative_path
    )
    assert len(
        activation._fresh_projection_assets(  # noqa: SLF001
            context=v2_context, projection=projection
        )
    ) == 1_200

    v1_sha = activation._digest_model(v1)  # noqa: SLF001
    v1_context = activation._FreshScopeContext(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt=SimpleNamespace(),
        source_receipt_sha256="b" * 64,
        selection=v1,
        selection_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{v1_sha}.json"
        ),
        selection_sha256=v1_sha,
        assets=activation._fresh_scope_assets(  # noqa: SLF001
            receipt=complete_receipt,
            selection=v1,
            scope="fresh-sealed",
        ),
    )
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_projection_invalid$",
    ):
        activation._fresh_projection_assets(  # noqa: SLF001
            context=v1_context, projection=projection
        )


def test_fresh_scope_rejects_family_leakage(
    complete_receipt: SimpleNamespace,
) -> None:
    selection = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    leaked = selection.model_copy(
        update={
            "fresh_calibration_family_ids": sorted(
                [*selection.fresh_calibration_family_ids[1:], "family-851"]
            )
        }
    )

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_scope_coverage_invalid$",
    ):
        activation._fresh_scope_assets(  # noqa: SLF001
            receipt=complete_receipt,
            selection=leaked,
            scope="fresh-calibration",
        )


def test_fresh_v2_scope_rejects_parent_family_and_pool_hash_tampering(
    complete_receipt: SimpleNamespace,
) -> None:
    parent = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    selection = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    old_sealed_family = next(
        family_id
        for family_id in parent.fresh_sealed_family_ids
        if family_id not in selection.fresh_sealed_family_ids
    )
    leaked = selection.model_copy(
        update={
            "fresh_sealed_family_ids": sorted(
                [*selection.fresh_sealed_family_ids[1:], old_sealed_family]
            )
        }
    )
    drifted_pool = selection.model_copy(
        update={"eligible_family_ids_sha256": "0" * 64}
    )

    for invalid in (leaked, drifted_pool):
        with pytest.raises(
            activation.SyntheticRasterTextActivationError,
            match="^raster_text_fresh_scope_coverage_invalid$",
        ):
            activation._fresh_scope_assets(  # noqa: SLF001
                receipt=complete_receipt,
                selection=invalid,
                scope="fresh-sealed",
            )


def test_fresh_selection_is_create_or_verify_and_never_overwritten(
    tmp_path: Path,
    complete_receipt: SimpleNamespace,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    selection = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )

    first = activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=selection
    )
    second = activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=selection
    )
    path = root.joinpath(*first[0].split("/"))
    assert first == second
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first[1]

    conflicting = b"{}"
    path.write_bytes(conflicting)
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_selection_write_failed$",
    ):
        activation._write_fresh_scope_selection(  # noqa: SLF001
            root=root, selection=selection
        )
    assert path.read_bytes() == conflicting


def test_fresh_v1_and_v2_selection_receipts_cannot_be_cross_read(
    tmp_path: Path,
    complete_receipt: SimpleNamespace,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    v1 = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    v2 = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    v1_path, _v1_sha = activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=v1
    )
    v2_path, _v2_sha = activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=v2
    )

    activation._read_fresh_scope_selection(  # noqa: SLF001
        root=root, relative_path=v1_path, expected=v1
    )
    activation._read_fresh_scope_selection_v2(  # noqa: SLF001
        root=root, relative_path=v2_path, expected=v2
    )
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_selection_invalid$",
    ):
        activation._read_fresh_scope_selection(  # noqa: SLF001
            root=root, relative_path=v2_path, expected=v1
        )
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_selection_invalid$",
    ):
        activation._read_fresh_scope_selection_v2(  # noqa: SLF001
            root=root, relative_path=v1_path, expected=v2
        )


def test_public_fresh_v2_selection_writer_creates_and_binds_the_v1_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    complete_receipt: SimpleNamespace,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        activation,
        "_load_fresh_dataset_receipts",
        lambda *_args, **_kwargs: (
            complete_receipt,
            "a" * 64,
            SimpleNamespace(assets=[]),
            "b" * 64,
        ),
    )

    first = activation.write_fresh_raster_text_scope_selection_receipt(
        document_preview_root=root,
        synthetic_raster_receipt_relative_path="synthetic-raster-v1/receipt.json",
        selection_revision="source-receipt-train-holdout-v2",
    )
    second = activation.write_fresh_raster_text_scope_selection_receipt(
        document_preview_root=root,
        synthetic_raster_receipt_relative_path="synthetic-raster-v1/receipt.json",
        selection_revision="source-receipt-train-holdout-v2",
    )
    raw = root.joinpath(*first[0].split("/")).read_bytes()
    selection = activation.RasterTextFreshScopeSelectionReceiptV2.model_validate_json(
        raw
    )
    parent_raw = root.joinpath(
        *selection.parent_selection_receipt_relative_path.split("/")
    ).read_bytes()

    assert first == second
    assert hashlib.sha256(raw).hexdigest() == first[1]
    assert hashlib.sha256(parent_raw).hexdigest() == (
        selection.parent_selection_receipt_sha256
    )
    assert activation._fresh_scope_selection_revision_from_path(  # noqa: SLF001
        root=root, relative_path=first[0]
    ) == "source-receipt-train-holdout-v2"


def test_existing_fresh_v2_selection_requires_its_exact_v1_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    complete_receipt: SimpleNamespace,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    v1 = activation._build_fresh_scope_selection(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    v2 = activation._build_fresh_scope_selection_v2(  # noqa: SLF001
        receipt=complete_receipt,
        receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
    )
    parent_relative, _parent_sha = activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=v1
    )
    v2_relative, _v2_sha = activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=v2
    )
    monkeypatch.setattr(
        activation,
        "_load_fresh_dataset_receipts",
        lambda *_args, **_kwargs: (
            complete_receipt,
            "a" * 64,
            SimpleNamespace(assets=[]),
            "b" * 64,
        ),
    )
    root.joinpath(*parent_relative.split("/")).unlink()

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_selection_invalid$",
    ):
        activation._fresh_scope_context(  # noqa: SLF001
            root=root,
            synthetic_raster_receipt_relative_path="synthetic-raster-v1/receipt.json",
            scope="fresh-sealed",
            selection_revision="source-receipt-train-holdout-v2",
            existing_selection_relative_path=v2_relative,
        )


@pytest.mark.parametrize("scope", ["fresh-calibration", "fresh-sealed"])
@pytest.mark.parametrize(
    "selection_revision",
    ["source-receipt-train-holdout-v1", "source-receipt-train-holdout-v2"],
)
def test_candidate_runner_supports_fresh_scope_and_immutable_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scope: str,
    selection_revision: Literal[
        "source-receipt-train-holdout-v1", "source-receipt-train-holdout-v2"
    ],
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    pointer = root / "approvals" / "raster-text-active.json"
    pointer.parent.mkdir()
    pointer.write_bytes(b"existing-pointer")
    asset = SimpleNamespace(
        asset_id="clean-0001-bl",
        family_id="family-1",
        document_type="bill_of_lading",
        split="train",
        profile_id="png-216",
        source_pdf_sha256="1" * 64,
        image_sha256="2" * 64,
        gold_relative_path="synthetic-raster-v1/gold/clean-0001-bl-png-216.json",
        gold_sha256="3" * 64,
        media_type="image/png",
        byte_size=1,
    )
    parent_selection = activation.RasterTextFreshScopeSelectionReceiptV1(
        synthetic_raster_receipt_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
        selection_seed_sha256=activation._FRESH_HOLDOUT_SELECTION_SEED_SHA256,  # noqa: SLF001
        fresh_calibration_family_ids=sorted(
            f"family-{index}" for index in range(1, 51)
        ),
        fresh_sealed_family_ids=sorted(
            f"family-{index}" for index in range(51, 151)
        ),
    )
    selection: (
        activation.RasterTextFreshScopeSelectionReceiptV1
        | activation.RasterTextFreshScopeSelectionReceiptV2
    )
    if selection_revision == "source-receipt-train-holdout-v1":
        selection = parent_selection
    else:
        parent_sha256 = activation._digest_model(parent_selection)  # noqa: SLF001
        selection = activation.RasterTextFreshScopeSelectionReceiptV2(
            synthetic_raster_receipt_sha256="a" * 64,
            source_receipt_sha256="b" * 64,
            parent_selection_receipt_relative_path=(
                "synthetic-raster-text-v1/fresh-scope-selections/"
                f"{parent_sha256}.json"
            ),
            parent_selection_receipt_sha256=parent_sha256,
            selection_seed_sha256=(
                activation._FRESH_HOLDOUT_SELECTION_SEED_SHA256_V2  # noqa: SLF001
            ),
            excluded_family_ids_sha256="e" * 64,
            eligible_family_ids_sha256="f" * 64,
            fresh_calibration_family_ids=list(
                parent_selection.fresh_calibration_family_ids
            ),
            fresh_sealed_family_ids=sorted(
                f"family-{index}" for index in range(151, 251)
            ),
        )
    selection_sha256 = activation._digest_model(selection)  # noqa: SLF001
    selection_relative = (
        "synthetic-raster-text-v1/fresh-scope-selections/"
        f"{selection_sha256}.json"
    )
    activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=parent_selection
    )
    assert activation._write_fresh_scope_selection(  # noqa: SLF001
        root=root, selection=selection
    ) == (selection_relative, selection_sha256)
    context = activation._FreshScopeContext(  # noqa: SLF001
        receipt=SimpleNamespace(pdfium_version="4.30.0"),
        receipt_sha256="a" * 64,
        source_receipt=SimpleNamespace(),
        source_receipt_sha256="b" * 64,
        selection=selection,
        selection_relative_path=selection_relative,
        selection_sha256=selection_sha256,
        assets=[asset],
    )
    session_starts: list[bool] = []
    extraction_calls: list[str] = []

    class Session:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Provider:
        def open_evaluation_session(self, *, parent_materials_verified: bool) -> Session:
            session_starts.append(parent_materials_verified)
            return Session()

    profile = SimpleNamespace(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision=_EXTRACTOR_REVISION,
    )
    resolution = SimpleNamespace(
        status="available",
        engine=SimpleNamespace(input_adapter=object(), provider=Provider()),
        profile=profile,
        runtime_configuration_sha256="c" * 64,
    )
    ticks = iter((1.0, 2.0))

    def extract_candidate(
        _engine: object,
        *,
        raster_input: RasterTextInputV1,
        **_kwargs: object,
    ) -> RasterTextExtractionResultV1:
        extraction_calls.append(raster_input.metadata.document_asset_id)
        pages = [RasterTextPageV1(page_number=1, text="TEXT", confidence=0.9)]
        return RasterTextExtractionResultV1(
            document_asset_id=asset.asset_id,
            extraction_request_id=raster_input.metadata.extraction_request_id,
            raster_text_profile_revision=_PROFILE_REVISION,
            status="preview_ready",
            media_type=asset.media_type,
            byte_size=asset.byte_size,
            source_sha256=asset.image_sha256,
            canonical_text=canonical_raster_text(pages),
            canonical_text_sha256=canonical_raster_text_sha256(pages),
            extractor_revision=_EXTRACTOR_REVISION,
            pages=pages,
            warnings=[],
            needs_review_reasons=[],
        )

    gold = RasterTextGoldArtifactV1(
        asset_id=asset.asset_id,
        family_id=asset.family_id,
        document_type=asset.document_type,
        source_pdf_sha256=asset.source_pdf_sha256,
        profile_id=asset.profile_id,
        source_raster_gold_sha256=asset.gold_sha256,
        canonical_text="TEXT",
        critical_values=["TEXT"],
    )
    context_revisions: list[str] = []

    def fresh_context(**kwargs: object) -> object:
        context_revisions.append(str(kwargs["selection_revision"]))
        return context

    monkeypatch.setattr(activation, "_fresh_scope_context", fresh_context)
    monkeypatch.setattr(
        activation,
        "resolve_paddleocr3_raster_text_candidate",
        lambda **_kwargs: resolution,
    )
    monkeypatch.setattr(activation, "_gold_for_asset", lambda *_args: gold)
    monkeypatch.setattr(
        activation, "_verified_asset", lambda *_args, **_kwargs: b"x"
    )
    monkeypatch.setattr(
        activation,
        "inspect_raster_document_structure",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(activation, "extract_raster_text_preview", extract_candidate)
    monkeypatch.setattr(activation, "_runner_implementation_sha256", lambda: "d" * 64)
    monkeypatch.setattr(activation.time, "monotonic", lambda: next(ticks))

    first = activation.run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path="profiles/runtime.json",
        synthetic_raster_receipt_relative_path=(
            "synthetic-raster-v1/receipt.json"
        ),
        scope=scope,
        fresh_selection_revision=selection_revision,
    )
    source_raw = root.joinpath(*first.source_gold_receipt_relative_path.split("/")).read_bytes()
    source = activation.RasterTextFreshSourceGoldReceiptV1.model_validate_json(source_raw)
    execution_path = root.joinpath(*first.candidate_execution_relative_path.split("/"))
    execution = activation.RasterTextFreshCandidateExecutionV1.model_validate_json(
        execution_path.read_bytes()
    )
    bundle = activation.RasterTextFreshCandidateBundleV1.model_validate_json(
        root.joinpath(*execution.candidate_bundle_relative_path.split("/")).read_bytes()
    )

    assert first.candidate_execution_relative_path.startswith(
        f"synthetic-raster-text-v1/fresh-candidate-executions/{scope}/"
    )
    assert source.selection_receipt_sha256 == selection_sha256
    assert execution.selection_receipt_sha256 == selection_sha256
    assert bundle.selection_receipt_sha256 == selection_sha256
    assert source.selection_receipt_relative_path == selection_relative
    assert execution.selection_receipt_relative_path == selection_relative
    assert bundle.selection_receipt_relative_path == selection_relative
    assert execution.source_gold_receipt_sha256 == first.source_gold_receipt_sha256
    checkpoint_paths = list(
        (
            root
            / "synthetic-raster-text-v1"
            / "candidate-checkpoints"
            / scope
        ).glob("*/*.json")
    )
    assert len(checkpoint_paths) == 1
    checkpoint = activation.RasterTextFreshCandidateCheckpointV1.model_validate_json(
        checkpoint_paths[0].read_bytes()
    )
    assert checkpoint.identity.selection_receipt_sha256 == selection_sha256
    assert session_starts == [True]
    assert extraction_calls == [asset.asset_id]

    second = activation.run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path="profiles/runtime.json",
        synthetic_raster_receipt_relative_path=(
            "synthetic-raster-v1/receipt.json"
        ),
        scope=scope,
        fresh_selection_revision=selection_revision,
    )
    assert second.candidate_execution_sha256 == first.candidate_execution_sha256
    assert session_starts == [True]
    assert extraction_calls == [asset.asset_id]
    assert pointer.read_bytes() == b"existing-pointer"

    _source, _execution, _bundle, cases = (
        activation.verify_source_attested_fresh_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=(
                first.source_gold_receipt_relative_path
            ),
            candidate_execution_relative_path=(
                first.candidate_execution_relative_path
            ),
            scope=scope,
            required_selection_revision=selection_revision,
        )
    )
    assert len(cases) == 1
    assert context_revisions == [selection_revision, selection_revision, selection_revision]

    wrong_revision: Literal[
        "source-receipt-train-holdout-v1", "source-receipt-train-holdout-v2"
    ] = (
        "source-receipt-train-holdout-v2"
        if selection_revision == "source-receipt-train-holdout-v1"
        else "source-receipt-train-holdout-v1"
    )
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_selection_invalid$",
    ):
        activation.verify_source_attested_fresh_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path=(
                first.source_gold_receipt_relative_path
            ),
            candidate_execution_relative_path=(
                first.candidate_execution_relative_path
            ),
            scope=scope,
            required_selection_revision=wrong_revision,
        )


def test_fresh_scope_rejects_smoke_build_shortcuts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        activation,
        "_fresh_scope_context",
        lambda **_kwargs: pytest.fail("fresh context must not be opened"),
    )

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_fresh_selection_invalid$",
    ):
        activation.run_paddleocr3_synthetic_raster_text_candidates(
            document_preview_root=root,
            runtime_configuration_relative_path="profiles/runtime.json",
            synthetic_raster_receipt_relative_path=(
                "synthetic-raster-v1/receipt.json"
            ),
            scope="fresh-sealed",
            preflight_smoke_only=True,
        )
