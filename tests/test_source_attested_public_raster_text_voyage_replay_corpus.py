from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_voyage_replay_corpus as replay,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    TrainingCropV1,
)


def _training_crop(
    index: int,
    *,
    family_id: str,
    split: str = "train",
    label: str = "VALUE",
) -> TrainingCropV1:
    token = hashlib.sha256(f"crop:{split}:{index}".encode()).hexdigest()
    return TrainingCropV1(
        sample_id=token,
        asset_id=f"asset-{split}-{index:04d}",
        family_id=family_id,
        split=split,  # type: ignore[arg-type]
        asset_relative_path=f"assets/{token}.png",
        gold_relative_path=f"gold/{token}.json",
        source_pdf_sha256=hashlib.sha256(f"source:{index}".encode()).hexdigest(),
        raster_sha256="2" * 64,
        gold_sha256="3" * 64,
        span_ids=(f"span-{index}",),
        bbox=(0.2, 0.2, 0.1, 0.05),
        label_sha256=hashlib.sha256(label.encode()).hexdigest(),
        label_length=len(label),
        page_width=100,
        page_height=100,
    )


def _voyage_assets(
    count: int,
) -> dict[
    str,
    tuple[replay._VerifiedVoyageAsset, replay._VerifiedVoyageAsset],
]:
    result: dict[
        str,
        tuple[replay._VerifiedVoyageAsset, replay._VerifiedVoyageAsset],
    ] = {}
    for family_index in range(count):
        family_id = f"family-{family_index:04d}"
        siblings: list[replay._VerifiedVoyageAsset] = []
        for fixture_index, fixture_kind in enumerate(("clean", "error")):
            asset_id = f"{fixture_kind}-{family_index:04d}-bl"
            label = f"{family_index % 10_000:04d}E"
            candidate = SimpleNamespace(
                field_id="voyage",
                candidate=SimpleNamespace(
                    kind="span",
                    span_ids=("span-voyage",),
                    bbox=(0.30, 0.25, 0.05, 0.01),
                    label=label,
                ),
            )
            assets = {
                profile: {
                    "asset_id": asset_id,
                    "asset_relative_path": f"assets/{asset_id}-{profile}",
                    "document_type": "bill_of_lading",
                    "family_id": family_id,
                    "gold_relative_path": f"gold/{asset_id}-{profile}.json",
                    "gold_sha256": "3" * 64,
                    "height": 100,
                    "image_sha256": "2" * 64,
                    "media_type": replay._EXPECTED_MEDIA[profile],
                    "profile_id": profile,
                    "source_pdf_sha256": hashlib.sha256(
                        f"source:{family_index}:{fixture_index}".encode()
                    ).hexdigest(),
                    "split": "train" if count == 500 else "development",
                    "width": 100,
                }
                for profile in replay._ALL_PROFILES
            }
            siblings.append(
                replay._VerifiedVoyageAsset(
                    asset_id=asset_id,
                    family_id=family_id,
                    fixture_kind=fixture_kind,  # type: ignore[arg-type]
                    source_pdf_sha256=assets["png-216"]["source_pdf_sha256"],
                    candidates_by_profile={profile: candidate for profile in replay._ALL_PROFILES},
                    assets_by_profile=assets,
                )
            )
        result[family_id] = siblings[0], siblings[1]
    return result


def _general_corpus() -> SimpleNamespace:
    train: list[TrainingCropV1] = []
    labels: dict[str, str] = {}
    profiles: dict[str, str] = {}
    fields: dict[str, str] = {}
    documents: dict[str, str] = {}
    field_keys = tuple(
        f"{document}:field-{index:02d}"
        for document in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for index in range(13)
    )
    for family_index in range(500):
        family_id = f"family-{family_index:04d}"
        for sibling in range(2):
            index = family_index * 2 + sibling
            crop = _training_crop(index, family_id=family_id)
            train.append(crop)
            labels[crop.sample_id] = "VALUE"
            profiles[crop.sample_id] = replay._PROFILES[index % 3]
            fields[crop.sample_id] = field_keys[index % len(field_keys)]
            documents[crop.sample_id] = fields[crop.sample_id].split(":", 1)[0]
    return SimpleNamespace(
        train_crops=tuple(train),
        labels_by_sample_id=labels,
        profile_by_sample_id=profiles,
        field_key_by_sample_id=fields,
        document_type_by_sample_id=documents,
    )


def test_exact_eight_recipe_contract_is_content_addressed() -> None:
    recipes = replay.VOYAGE_REPLAY_RECIPES_V1
    hashes = replay.voyage_replay_recipe_sha256s_v1()

    assert len(recipes) == len(hashes) == 8
    assert len(set(hashes.values())) == 8
    assert Counter(item.horizontal_height_ratio for item in recipes) == {
        0.32: 4,
        0.64: 4,
    }
    assert Counter(item.source_profile_id for item in recipes) == {
        "png-216": 4,
        "jpeg-216-q75": 2,
        "jpeg-150-q75": 2,
    }
    assert sum(item.degradation == "gaussian0p6-jpeg75-v1" for item in recipes) == 2

    with pytest.raises(ValueError, match="crop recipe"):
        replay.VoyageReplayCropRecipeV1(
            recipe_id="g32-png216-v1",
            horizontal_height_ratio=0.64,
            horizontal_max_pixels=8,
            source_profile_id="png-216",
            degradation="actual-raster-v1",
            gaussian_blur_radius=0.0,
        )


def test_voyage_split_has_exact_family_and_lane_coverage() -> None:
    train, train_labels, train_profiles, train_recipes = replay._compile_voyage_split(
        assets=_voyage_assets(500), split="train", seed=1_937_429
    )
    development, dev_labels, dev_profiles, dev_recipes = replay._compile_voyage_split(
        assets=_voyage_assets(100), split="development", seed=1_937_429
    )

    assert len(train) == len(train_labels) == len(train_profiles) == len(train_recipes) == 500
    assert len({crop.family_id for crop in train}) == 500
    assert set(Counter(crop.recipe_id for crop in train).values()) == {62, 63}
    assert all(value.endswith("E") and len(value) == 5 for value in train_labels.values())

    assert len(development) == len(dev_labels) == len(dev_profiles) == len(dev_recipes) == 200
    assert Counter(crop.recipe_id for crop in development) == {
        recipe.recipe_id: 25 for recipe in replay.VOYAGE_REPLAY_RECIPES_V1
    }
    by_family: dict[str, list[replay.VoyageReplayCropV1]] = {}
    for crop in development:
        by_family.setdefault(crop.family_id, []).append(crop)
    assert {len(items) for items in by_family.values()} == {2}
    assert all(len({item.recipe_id for item in items}) == 2 for items in by_family.values())
    assert all(
        {item.asset_id.split("-", 1)[0] for item in items} == {"clean", "error"}
        for items in by_family.values()
    )


def test_general_anchor_is_250_unique_families_and_covers_every_lane() -> None:
    corpus = _general_corpus()
    selected = replay._select_general_train_anchors(corpus, seed=1_937_429)

    assert len(selected) == len({crop.family_id for crop in selected}) == 250
    assert {corpus.field_key_by_sample_id[crop.sample_id] for crop in selected} == set(
        corpus.field_key_by_sample_id.values()
    )
    assert {corpus.document_type_by_sample_id[crop.sample_id] for crop in selected} == {
        "bill_of_lading",
        "commercial_invoice",
        "packing_list",
    }
    assert {corpus.profile_by_sample_id[crop.sample_id] for crop in selected} == set(
        replay._PROFILES
    )
    assert all(
        corpus.field_key_by_sample_id[crop.sample_id] != "bill_of_lading:voyage"
        for crop in selected
    )


def test_manifest_binds_counts_and_contains_no_plaintext_label() -> None:
    voyage, labels, _profiles, _recipes = replay._compile_voyage_split(
        assets=_voyage_assets(500), split="train", seed=1_937_429
    )
    parent = _general_corpus()
    general = replay._select_general_train_anchors(parent, seed=1_937_429)
    parent.train_family_ids = tuple(f"family-{index:04d}" for index in range(500))
    parent.development_family_ids = tuple(f"dev-{index:04d}" for index in range(100))
    parent.source_receipt_sha256 = "1" * 64
    parent.raster_receipt_sha256 = "2" * 64
    parent.selection_receipt_sha256 = "3" * 64
    parent.selection_receipt_relative_path = (
        "synthetic-raster-text-v1/fresh-scope-selections/" + "3" * 64 + ".json"
    )
    parent.parent_selection_receipt_sha256 = "4" * 64
    parent.excluded_family_ids_sha256 = "5" * 64
    manifest = replay._manifest(
        split="train",
        v2_corpus=parent,
        voyage_crops=voyage,
        general_crops=general,
        audit_family_ids=tuple(f"blra-family-{index:03d}" for index in range(100)),
        audit_source_sha256s=tuple(f"{index:064x}" for index in range(200)),
    )
    serialized = manifest.model_dump_json()

    assert manifest.voyage_crop_count == 500
    assert manifest.general_crop_count == 250
    assert manifest.total_crop_count == 750
    assert manifest.all_field_key_crop_counts["bill_of_lading:voyage"] == 500
    assert not any(label in serialized for label in labels.values())
    assert "labels_by_sample_id" not in serialized

    with pytest.raises(ValueError, match="holdout_overlap_count"):
        manifest.model_copy(update={"holdout_overlap_count": 1}).__class__.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "holdout_overlap_count": 1,
            }
        )


def test_prepare_crop_applies_g32_g64_and_deterministic_blur_jpeg(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    assets = root / "assets"
    assets.mkdir(parents=True)
    page_path = assets / "page.png"
    Image.new("RGB", (200, 100), "white").save(page_path, format="PNG")

    def crop_for(recipe_id: replay.VoyageRecipeId) -> replay.VoyageReplayCropV1:
        recipe = replay._RECIPE_BY_ID[recipe_id]
        return replay.VoyageReplayCropV1(
            sample_id=hashlib.sha256(recipe_id.encode()).hexdigest(),
            asset_id="clean-0001-bl",
            family_id="family-0001",
            split="train",
            asset_relative_path="assets/page.png",
            gold_relative_path="gold/page.json",
            source_pdf_sha256="1" * 64,
            raster_sha256=replay.sha256_file(page_path),
            gold_sha256="3" * 64,
            span_ids=("span-voyage",),
            bbox=(0.4, 0.4, 0.1, 0.1),
            label_sha256=hashlib.sha256(b"1234E").hexdigest(),
            label_length=5,
            page_width=200,
            page_height=100,
            recipe_id=recipe_id,
            recipe_sha256=replay._RECIPE_SHA256S[recipe_id],
            source_profile_id=recipe.source_profile_id,
            field_id_sha256=hashlib.sha256(b"bill_of_lading:voyage").hexdigest(),
        )

    g32 = replay.prepare_voyage_replay_crop_image_v1(root, crop_for("g32-png216-v1"))
    g64 = replay.prepare_voyage_replay_crop_image_v1(root, crop_for("g64-png216-v1"))
    degraded_crop = crop_for("g32-png216-blur0p6-jpeg75-v1")
    first = replay.prepare_voyage_replay_crop_image_v1(root, degraded_crop)
    second = replay.prepare_voyage_replay_crop_image_v1(root, degraded_crop)

    assert g32.height == g64.height == first.height == 48
    assert g32.width < g64.width
    assert first.tobytes() == second.tobytes()


def test_public_builder_rejects_independent_audit_family_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    parent = SimpleNamespace(
        source_receipt_sha256="1" * 64,
        raster_receipt_sha256="2" * 64,
        train_family_ids=tuple(["blra-family-000", *[f"t-{index}" for index in range(499)]]),
        development_family_ids=tuple(f"d-{index}" for index in range(100)),
        excluded_family_ids=tuple(
            [*[f"d-{index}" for index in range(100)], *[f"x-{index}" for index in range(400)]]
        ),
        holdout_overlap_count=0,
    )
    monkeypatch.setattr(replay.v2, "_root", lambda _path: root)
    monkeypatch.setattr(
        replay.v2,
        "build_verified_leakage_safe_public_crop_corpus_v2",
        lambda *_args, **_kwargs: parent,
    )
    monkeypatch.setattr(
        replay,
        "_read_dataset",
        lambda _root: (
            {"assets": []},
            {"assets": []},
            b"raster",
            b"source",
        ),
    )
    parent.source_receipt_sha256 = replay.sha256_bytes(b"source")
    parent.raster_receipt_sha256 = replay.sha256_bytes(b"raster")
    monkeypatch.setattr(
        replay,
        "_audit_exclusions",
        lambda: (
            tuple(f"blra-family-{index:03d}" for index in range(100)),
            tuple(f"{index:064x}" for index in range(200)),
        ),
    )

    with pytest.raises(replay.SourceAttestedVoyageReplayCorpusError) as raised:
        replay.build_verified_public_raster_text_voyage_replay_corpus_v1(root)
    assert raised.value.code == "voyage_replay_independent_audit_leakage"
