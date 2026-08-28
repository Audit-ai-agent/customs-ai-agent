"""Leakage-safe B/L voyage replay crops for the public mobile recognizer.

The compiler extends, but never changes, the immutable V2 fine-tune corpus.
Only the 500 V2 optimizer families may contribute optimizer samples.  The
original 100 development families remain validation-only.  Calibration,
sealed, both historical fresh holdouts, and the independent B/L repair audit
are checked as exclusions and can never provide a label.

Gold text is kept only in the returned in-memory label map.  Manifests and
crop records contain content hashes, counts, and provenance, never label text.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Final, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.evaluation import source_attested_bl_repair_audit as bl_audit
from customs_ai_agent.evaluation import source_attested_public_raster_text_corpus as v2
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    TrainingCropV1,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


_SHA256 = r"^[a-f0-9]{64}$"
_COMPILER_REVISION: Final[Literal["source-attested-voyage-replay-family-lanes-v1"]] = (
    "source-attested-voyage-replay-family-lanes-v1"
)
_DEFAULT_SEED = 1_937_429
_VOYAGE_PATTERN_TEXT = r"^[0-9]{4}[EWNS]$"
_VOYAGE_PATTERN = re.compile(_VOYAGE_PATTERN_TEXT)
_VOYAGE_PATTERN_SHA256 = sha256_bytes(_VOYAGE_PATTERN_TEXT.encode("ascii"))
_PROFILES = ("png-216", "jpeg-216-q75", "jpeg-150-q75")
_ALL_PROFILES = (*_PROFILES, "scan-pdf-216-q75")
_EXPECTED_MEDIA = {
    "png-216": "image/png",
    "jpeg-216-q75": "image/jpeg",
    "jpeg-150-q75": "image/jpeg",
    "scan-pdf-216-q75": "application/pdf",
}

VoyageProfileId = Literal["png-216", "jpeg-216-q75", "jpeg-150-q75"]
VoyageRecipeId = Literal[
    "g32-png216-v1",
    "g32-jpeg216-q75-v1",
    "g32-jpeg150-q75-v1",
    "g32-png216-blur0p6-jpeg75-v1",
    "g64-png216-v1",
    "g64-jpeg216-q75-v1",
    "g64-jpeg150-q75-v1",
    "g64-png216-blur0p6-jpeg75-v1",
]


class SourceAttestedVoyageReplayCorpusError(RuntimeError):
    """Stable, content-free failure raised by the V3 corpus boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VoyageReplayCropRecipeV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    recipe_id: VoyageRecipeId
    horizontal_height_ratio: float = Field(ge=0.32, le=0.64)
    horizontal_min_pixels: Literal[2] = 2
    horizontal_max_pixels: Literal[8, 16]
    vertical_height_ratio: float = Field(default=0.15, ge=0.15, le=0.15)
    source_profile_id: VoyageProfileId
    degradation: Literal["actual-raster-v1", "gaussian0p6-jpeg75-v1"]
    gaussian_blur_radius: float = Field(ge=0.0, le=0.6)
    jpeg_quality: Literal[75] = 75
    jpeg_subsampling: Literal[2] = 2
    jpeg_progressive: Literal[False] = False
    jpeg_optimize: Literal[False] = False
    output_height: Literal[48] = 48
    resize_interpolation: Literal["bilinear-v1"] = "bilinear-v1"
    maximum_output_width: Literal[3200] = 3200

    @model_validator(mode="after")
    def _exact_recipe(self) -> "VoyageReplayCropRecipeV1":
        geometry = self.recipe_id[:3]
        expected_ratio = 0.32 if geometry == "g32" else 0.64
        expected_max = 8 if geometry == "g32" else 16
        derived = "blur0p6" in self.recipe_id
        expected_profile = (
            "jpeg-216-q75"
            if "jpeg216" in self.recipe_id
            else "jpeg-150-q75"
            if "jpeg150" in self.recipe_id
            else "png-216"
        )
        if (
            self.horizontal_height_ratio != expected_ratio
            or self.horizontal_max_pixels != expected_max
            or self.source_profile_id != expected_profile
            or derived != (self.degradation == "gaussian0p6-jpeg75-v1")
            or self.gaussian_blur_radius != (0.6 if derived else 0.0)
        ):
            raise ValueError("voyage replay crop recipe is invalid")
        return self


def _recipe(
    recipe_id: VoyageRecipeId,
    *,
    ratio: float,
    profile: VoyageProfileId,
    derived: bool = False,
) -> VoyageReplayCropRecipeV1:
    return VoyageReplayCropRecipeV1(
        recipe_id=recipe_id,
        horizontal_height_ratio=ratio,
        horizontal_max_pixels=8 if ratio == 0.32 else 16,
        source_profile_id=profile,
        degradation=("gaussian0p6-jpeg75-v1" if derived else "actual-raster-v1"),
        gaussian_blur_radius=0.6 if derived else 0.0,
    )


VOYAGE_REPLAY_RECIPES_V1: Final[tuple[VoyageReplayCropRecipeV1, ...]] = (
    _recipe("g32-png216-v1", ratio=0.32, profile="png-216"),
    _recipe("g32-jpeg216-q75-v1", ratio=0.32, profile="jpeg-216-q75"),
    _recipe("g32-jpeg150-q75-v1", ratio=0.32, profile="jpeg-150-q75"),
    _recipe(
        "g32-png216-blur0p6-jpeg75-v1",
        ratio=0.32,
        profile="png-216",
        derived=True,
    ),
    _recipe("g64-png216-v1", ratio=0.64, profile="png-216"),
    _recipe("g64-jpeg216-q75-v1", ratio=0.64, profile="jpeg-216-q75"),
    _recipe("g64-jpeg150-q75-v1", ratio=0.64, profile="jpeg-150-q75"),
    _recipe(
        "g64-png216-blur0p6-jpeg75-v1",
        ratio=0.64,
        profile="png-216",
        derived=True,
    ),
)
_RECIPE_BY_ID: Final = {item.recipe_id: item for item in VOYAGE_REPLAY_RECIPES_V1}


def voyage_replay_recipe_sha256s_v1() -> dict[str, str]:
    """Return canonical hashes for the exact eight public recipe lanes."""

    return {
        item.recipe_id: sha256_bytes(
            canonical_json_bytes(item.model_dump(mode="json", warnings=False))
        )
        for item in VOYAGE_REPLAY_RECIPES_V1
    }


_RECIPE_SHA256S: Final = voyage_replay_recipe_sha256s_v1()
_RECIPE_SET_SHA256: Final = sha256_bytes(canonical_json_bytes(_RECIPE_SHA256S))


class VoyageReplayCropV1(TrainingCropV1):
    """One content-free crop identity plus its deterministic transform lane."""

    recipe_id: VoyageRecipeId
    recipe_sha256: str = Field(pattern=_SHA256)
    source_profile_id: VoyageProfileId
    field_id_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _recipe_binding(self) -> "VoyageReplayCropV1":
        recipe = _RECIPE_BY_ID[self.recipe_id]
        if (
            self.recipe_sha256 != _RECIPE_SHA256S[self.recipe_id]
            or self.source_profile_id != recipe.source_profile_id
            or self.label_length != 5
            or self.sample_kind != "span"
            or self.field_id_sha256 != sha256_bytes(b"bill_of_lading:voyage")
        ):
            raise ValueError("voyage replay crop binding is invalid")
        return self


class VoyageReplayCorpusManifestV1(_WireModel):
    """Content-free optimizer or validation corpus receipt."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_public_voyage_replay_crop_manifest"] = (
        "source_attested_public_voyage_replay_crop_manifest"
    )
    compiler_revision: Literal["source-attested-voyage-replay-family-lanes-v1"] = _COMPILER_REVISION
    dataset_id: Literal["synthetic-raster-v1"] = "synthetic-raster-v1"
    split: Literal["train", "development"]
    purpose: Literal["optimizer", "validation-only"]
    source_receipt_sha256: str = Field(pattern=_SHA256)
    raster_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    parent_selection_receipt_sha256: str = Field(pattern=_SHA256)
    selected_family_count: int = Field(ge=100, le=500)
    selected_family_ids_sha256: str = Field(pattern=_SHA256)
    excluded_family_count: Literal[500] = 500
    excluded_family_ids_sha256: str = Field(pattern=_SHA256)
    holdout_overlap_count: Literal[0] = 0
    independent_audit_family_count: Literal[100] = 100
    independent_audit_family_overlap_count: Literal[0] = 0
    independent_audit_source_overlap_count: Literal[0] = 0
    independent_audit_recipe_sha256: str = Field(pattern=_SHA256)
    independent_audit_family_ids_sha256: str = Field(pattern=_SHA256)
    independent_audit_source_sha256s_sha256: str = Field(pattern=_SHA256)
    voyage_crop_count: int = Field(ge=200, le=500)
    general_crop_count: int = Field(ge=200, le=250)
    total_crop_count: int = Field(ge=400, le=750)
    voyage_recipe_set_sha256: str = Field(pattern=_SHA256)
    voyage_recipe_sha256s: dict[str, str]
    voyage_recipe_lane_counts: dict[str, int]
    voyage_sample_ids_sha256: str = Field(pattern=_SHA256)
    voyage_label_sha256s_sha256: str = Field(pattern=_SHA256)
    voyage_label_pattern_sha256: str = Field(pattern=_SHA256)
    general_sample_ids_sha256: str = Field(pattern=_SHA256)
    general_field_ids_sha256: str = Field(pattern=_SHA256)
    general_field_crop_counts: dict[str, int]
    all_field_key_ids_sha256: str = Field(pattern=_SHA256)
    all_field_key_crop_counts: dict[str, int]
    general_document_type_crop_counts: dict[str, int]
    general_profile_crop_counts: dict[str, int]

    @model_validator(mode="after")
    def _fixed_scope(self) -> "VoyageReplayCorpusManifestV1":
        expected = (
            ("optimizer", 500, 500, 250, 750)
            if self.split == "train"
            else ("validation-only", 100, 200, 200, 400)
        )
        lane_counts = self.voyage_recipe_lane_counts
        expected_lane_values = {62, 63} if self.split == "train" else {25}
        expected_selection_path = (
            f"synthetic-raster-text-v1/fresh-scope-selections/{self.selection_receipt_sha256}.json"
        )
        if (
            (
                self.purpose,
                self.selected_family_count,
                self.voyage_crop_count,
                self.general_crop_count,
                self.total_crop_count,
            )
            != expected
            or self.selection_receipt_relative_path != expected_selection_path
            or self.voyage_recipe_set_sha256 != _RECIPE_SET_SHA256
            or self.voyage_recipe_sha256s != _RECIPE_SHA256S
            or set(lane_counts) != set(_RECIPE_BY_ID)
            or set(lane_counts.values()) != expected_lane_values
            or sum(lane_counts.values()) != self.voyage_crop_count
            or self.voyage_label_pattern_sha256 != _VOYAGE_PATTERN_SHA256
            or sum(self.general_field_crop_counts.values()) != self.general_crop_count
            or any(value < 1 for value in self.general_field_crop_counts.values())
            or sum(self.all_field_key_crop_counts.values()) != self.total_crop_count
            or any(value < 1 for value in self.all_field_key_crop_counts.values())
            or self.all_field_key_ids_sha256 != _hash_list(sorted(self.all_field_key_crop_counts))
            or self.all_field_key_crop_counts.get("bill_of_lading:voyage", 0)
            < self.voyage_crop_count
            or set(self.general_document_type_crop_counts)
            != {"bill_of_lading", "commercial_invoice", "packing_list"}
            or sum(self.general_document_type_crop_counts.values()) != self.general_crop_count
            or any(value < 1 for value in self.general_document_type_crop_counts.values())
            or set(self.general_profile_crop_counts) != set(_PROFILES)
            or sum(self.general_profile_crop_counts.values()) != self.general_crop_count
            or any(value < 1 for value in self.general_profile_crop_counts.values())
        ):
            raise ValueError("voyage replay manifest scope is invalid")
        return self


@dataclass(slots=True)
class VerifiedPublicRasterTextVoyageReplayCorpusV1:
    train_voyage_crops: tuple[VoyageReplayCropV1, ...]
    train_general_crops: tuple[TrainingCropV1, ...]
    development_voyage_crops: tuple[VoyageReplayCropV1, ...]
    development_general_crops: tuple[TrainingCropV1, ...]
    labels_by_sample_id: dict[str, str]
    profile_by_sample_id: dict[str, str]
    field_key_by_sample_id: dict[str, str]
    recipe_by_sample_id: dict[str, VoyageReplayCropRecipeV1]
    train_manifest: VoyageReplayCorpusManifestV1
    development_manifest: VoyageReplayCorpusManifestV1


@dataclass(frozen=True, slots=True)
class _VerifiedVoyageAsset:
    asset_id: str
    family_id: str
    fixture_kind: Literal["clean", "error"]
    source_pdf_sha256: str
    candidates_by_profile: Mapping[str, Any]
    assets_by_profile: Mapping[str, Mapping[str, object]]

    @property
    def label(self) -> str:
        return str(self.candidates_by_profile["png-216"].candidate.label)


def _hash_list(values: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(values)))


def _rank(*parts: str) -> str:
    return sha256_bytes((f"{_COMPILER_REVISION}:" + ":".join(parts)).encode("utf-8"))


def _read_dataset(root: Path) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    try:
        dataset = v2._safe(root, "synthetic-raster-v1", directory=True)
        raster, raster_raw = v2._read_json(dataset / "synthetic-raster-receipt.json")
        source, source_raw = v2._read_json(dataset / "source-receipt.json")
        if (
            raster.get("source_receipt_sha256") != sha256_bytes(source_raw)
            or not isinstance(raster.get("assets"), list)
            or not isinstance(source.get("assets"), list)
        ):
            raise ValueError
        return raster, source, raster_raw, source_raw
    except Exception:
        raise SourceAttestedVoyageReplayCorpusError(
            "voyage_replay_dataset_receipt_invalid"
        ) from None


def _audit_exclusions() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive audit identities from its fixed generator, never from results."""

    try:
        family_ids = tuple(
            f"blra-family-{index:03d}" for index in range(bl_audit.BL_REPAIR_AUDIT_FAMILY_COUNT)
        )
        source_sha256s = tuple(
            sorted(
                hashlib.sha256(
                    bl_audit.build_bl_repair_audit_source_pdf(
                        family_index=index,
                        fixture_kind=fixture_kind,
                    )
                ).hexdigest()
                for index in range(bl_audit.BL_REPAIR_AUDIT_FAMILY_COUNT)
                for fixture_kind in ("clean", "error")
            )
        )
        if len(family_ids) != 100 or len(source_sha256s) != 200:
            raise ValueError
        return family_ids, source_sha256s
    except Exception:
        raise SourceAttestedVoyageReplayCorpusError(
            "voyage_replay_audit_identity_invalid"
        ) from None


def _fixture_kind(asset_id: str) -> Literal["clean", "error"]:
    if asset_id.startswith("clean-"):
        return "clean"
    if asset_id.startswith("error-"):
        return "error"
    raise SourceAttestedVoyageReplayCorpusError("voyage_replay_asset_identity_invalid")


def _verified_voyage_assets(
    *,
    root: Path,
    source_receipt: Mapping[str, object],
    raster_receipt: Mapping[str, object],
    family_ids: set[str],
    split: Literal["train", "development"],
) -> dict[str, tuple[_VerifiedVoyageAsset, _VerifiedVoyageAsset]]:
    try:
        source_root = v2._safe(
            root, source_receipt.get("source_root_relative_path"), directory=True
        )
        raw_sources = source_receipt.get("assets")
        raw_assets = raster_receipt.get("assets")
        if not isinstance(raw_sources, list) or not isinstance(raw_assets, list):
            raise ValueError
        sources: dict[str, Mapping[str, object]] = {}
        for item in raw_sources:
            if not isinstance(item, dict):
                raise ValueError
            digest = v2._sha(item.get("source_pdf_sha256"))
            if digest in sources:
                raise ValueError
            sources[digest] = item
        grouped: dict[str, dict[str, dict[str, Mapping[str, object]]]] = {
            family: {} for family in family_ids
        }
        for item in raw_assets:
            if not isinstance(item, dict):
                raise ValueError
            family = item.get("family_id")
            if family not in family_ids:
                continue
            if item.get("split") != split:
                raise ValueError
            if item.get("document_type") != "bill_of_lading":
                continue
            asset_id = v2._identifier(item.get("asset_id"))
            profile = item.get("profile_id")
            if profile not in _ALL_PROFILES:
                raise ValueError
            profile_map = grouped[str(family)].setdefault(asset_id, {})
            if str(profile) in profile_map:
                raise ValueError
            profile_map[str(profile)] = item

        result: dict[str, tuple[_VerifiedVoyageAsset, _VerifiedVoyageAsset]] = {}
        for family_id in sorted(family_ids):
            asset_groups = grouped[family_id]
            if len(asset_groups) != 2:
                raise ValueError
            verified: list[_VerifiedVoyageAsset] = []
            seen_kinds: set[str] = set()
            for asset_id, by_profile in sorted(asset_groups.items()):
                if set(by_profile) != set(_ALL_PROFILES):
                    raise ValueError
                kind = _fixture_kind(asset_id)
                if kind in seen_kinds:
                    raise ValueError
                seen_kinds.add(kind)
                exemplar = by_profile["png-216"]
                source_sha = v2._sha(exemplar.get("source_pdf_sha256"))
                source = sources.get(source_sha)
                if (
                    source is None
                    or source.get("asset_id") != asset_id
                    or source.get("family_id") != family_id
                ):
                    raise ValueError
                source_path = v2._safe(source_root, source.get("source_relative_path"))
                v2._verify_regular_material(source_path, source, "source_pdf_sha256")
                candidates: dict[str, Any] = {}
                for profile_id in _ALL_PROFILES:
                    asset = by_profile[profile_id]
                    if (
                        asset.get("asset_id") != asset_id
                        or asset.get("family_id") != family_id
                        or asset.get("split") != split
                        or asset.get("document_type") != "bill_of_lading"
                        or asset.get("profile_id") != profile_id
                        or asset.get("source_pdf_sha256") != source_sha
                        or asset.get("media_type") != _EXPECTED_MEDIA[profile_id]
                    ):
                        raise ValueError
                    raster_path = v2._safe(root, asset.get("asset_relative_path"))
                    gold_path = v2._safe(root, asset.get("gold_relative_path"))
                    v2._verify_regular_material(raster_path, asset, "image_sha256")
                    if sha256_file(gold_path) != v2._sha(asset.get("gold_sha256")):
                        raise ValueError
                    gold, _raw = v2._read_json(gold_path)
                    voyage = [
                        item
                        for item in v2._gold_field_candidates_v2(
                            gold,
                            asset_id=asset_id,
                            family_id=family_id,
                            profile_id=profile_id,
                            source_sha256=source_sha,
                        )
                        if item.field_id == "voyage"
                    ]
                    if (
                        len(voyage) != 1
                        or _VOYAGE_PATTERN.fullmatch(voyage[0].candidate.label) is None
                    ):
                        raise ValueError
                    candidates[profile_id] = voyage[0]
                    if profile_id in _PROFILES:
                        v2._decode_rgb(
                            raster_path,
                            profile_id=profile_id,
                            width=v2._integer(asset.get("width"), minimum=1, maximum=20_000),
                            height=v2._integer(asset.get("height"), minimum=1, maximum=20_000),
                        )
                canonical = candidates["png-216"]
                if any(candidates[profile] != canonical for profile in _ALL_PROFILES[1:]):
                    raise ValueError
                verified.append(
                    _VerifiedVoyageAsset(
                        asset_id=asset_id,
                        family_id=family_id,
                        fixture_kind=kind,
                        source_pdf_sha256=source_sha,
                        candidates_by_profile=candidates,
                        assets_by_profile=by_profile,
                    )
                )
            if seen_kinds != {"clean", "error"}:
                raise ValueError
            result[family_id] = (verified[0], verified[1])
        return result
    except SourceAttestedVoyageReplayCorpusError:
        raise
    except Exception:
        raise SourceAttestedVoyageReplayCorpusError("voyage_replay_source_gold_invalid") from None


def _voyage_crop(
    *,
    source: _VerifiedVoyageAsset,
    split: Literal["train", "development"],
    recipe: VoyageReplayCropRecipeV1,
) -> tuple[VoyageReplayCropV1, str]:
    candidate = source.candidates_by_profile[recipe.source_profile_id].candidate
    asset = source.assets_by_profile[recipe.source_profile_id]
    label = str(candidate.label)
    label_sha = sha256_bytes(label.encode("utf-8"))
    payload = {
        "asset_id": source.asset_id,
        "bbox": [round(float(value), 12) for value in candidate.bbox],
        "field_id_sha256": sha256_bytes(b"bill_of_lading:voyage"),
        "label_sha256": label_sha,
        "recipe_sha256": _RECIPE_SHA256S[recipe.recipe_id],
        "span_ids": list(candidate.span_ids),
    }
    crop = VoyageReplayCropV1(
        sample_id=sha256_bytes(canonical_json_bytes(payload)),
        asset_id=source.asset_id,
        family_id=source.family_id,
        split=split,
        sample_kind="span",
        asset_relative_path=v2._relative(asset.get("asset_relative_path")),
        gold_relative_path=v2._relative(asset.get("gold_relative_path")),
        source_pdf_sha256=source.source_pdf_sha256,
        raster_sha256=v2._sha(asset.get("image_sha256")),
        gold_sha256=v2._sha(asset.get("gold_sha256")),
        span_ids=tuple(candidate.span_ids),
        bbox=tuple(candidate.bbox),
        label_sha256=label_sha,
        label_length=len(label),
        page_width=v2._integer(asset.get("width"), minimum=1, maximum=20_000),
        page_height=v2._integer(asset.get("height"), minimum=1, maximum=20_000),
        recipe_id=recipe.recipe_id,
        recipe_sha256=_RECIPE_SHA256S[recipe.recipe_id],
        source_profile_id=recipe.source_profile_id,
        field_id_sha256=sha256_bytes(b"bill_of_lading:voyage"),
    )
    return crop, label


def _ordered_families(family_ids: Sequence[str], *, split: str, seed: int) -> list[str]:
    return sorted(
        family_ids,
        key=lambda family: (_rank(str(seed), split, "family", family), family),
    )


def _compile_voyage_split(
    *,
    assets: Mapping[str, tuple[_VerifiedVoyageAsset, _VerifiedVoyageAsset]],
    split: Literal["train", "development"],
    seed: int,
) -> tuple[
    tuple[VoyageReplayCropV1, ...],
    dict[str, str],
    dict[str, str],
    dict[str, VoyageReplayCropRecipeV1],
]:
    expected_families = 500 if split == "train" else 100
    if len(assets) != expected_families:
        raise SourceAttestedVoyageReplayCorpusError("voyage_replay_family_coverage_invalid")
    crops: list[VoyageReplayCropV1] = []
    labels: dict[str, str] = {}
    profiles: dict[str, str] = {}
    recipes: dict[str, VoyageReplayCropRecipeV1] = {}
    ordered = _ordered_families(tuple(assets), split=split, seed=seed)
    for ordinal, family_id in enumerate(ordered):
        siblings = assets[family_id]
        if split == "train":
            source = min(
                siblings,
                key=lambda item: (
                    _rank(str(seed), split, family_id, item.asset_id, item.source_pdf_sha256),
                    item.asset_id,
                ),
            )
            selected: tuple[tuple[_VerifiedVoyageAsset, VoyageReplayCropRecipeV1], ...] = (
                (source, VOYAGE_REPLAY_RECIPES_V1[ordinal % 8]),
            )
        else:
            by_kind = {item.fixture_kind: item for item in siblings}
            first = (ordinal * 2) % 8
            selected = (
                (by_kind["clean"], VOYAGE_REPLAY_RECIPES_V1[first]),
                (by_kind["error"], VOYAGE_REPLAY_RECIPES_V1[(first + 1) % 8]),
            )
        for source, recipe in selected:
            crop, label = _voyage_crop(source=source, split=split, recipe=recipe)
            if crop.sample_id in labels:
                raise SourceAttestedVoyageReplayCorpusError("voyage_replay_sample_collision")
            crops.append(crop)
            labels[crop.sample_id] = label
            profiles[crop.sample_id] = recipe.source_profile_id
            recipes[crop.sample_id] = recipe
    expected = 500 if split == "train" else 200
    lane_counts = {
        recipe_id: sum(crop.recipe_id == recipe_id for crop in crops) for recipe_id in _RECIPE_BY_ID
    }
    if (
        len(crops) != expected
        or len(labels) != expected
        or set(lane_counts.values()) != ({62, 63} if split == "train" else {25})
        or any(
            sha256_bytes(labels[crop.sample_id].encode("utf-8")) != crop.label_sha256
            for crop in crops
        )
    ):
        raise SourceAttestedVoyageReplayCorpusError("voyage_replay_lane_coverage_invalid")
    return tuple(sorted(crops, key=lambda item: item.sample_id)), labels, profiles, recipes


def _select_general_train_anchors(
    corpus: Any,
    *,
    seed: int,
) -> tuple[TrainingCropV1, ...]:
    candidates = [
        crop
        for crop in corpus.train_crops
        if corpus.field_key_by_sample_id.get(crop.sample_id) != "bill_of_lading:voyage"
    ]
    ranked = sorted(
        candidates,
        key=lambda crop: (
            _rank(str(seed), "general-anchor", crop.family_id, crop.sample_id),
            crop.sample_id,
        ),
    )
    selected: list[TrainingCropV1] = []
    selected_ids: set[str] = set()
    selected_families: set[str] = set()

    def take(predicate: Any) -> bool:
        for crop in ranked:
            if crop.family_id in selected_families or not predicate(crop):
                continue
            selected.append(crop)
            selected_ids.add(crop.sample_id)
            selected_families.add(crop.family_id)
            return True
        return False

    field_keys = sorted({corpus.field_key_by_sample_id[crop.sample_id] for crop in candidates})
    for field_key in field_keys:
        if not take(
            lambda crop, key=field_key: corpus.field_key_by_sample_id[crop.sample_id] == key
        ):
            raise SourceAttestedVoyageReplayCorpusError(
                "voyage_replay_general_anchor_coverage_invalid"
            )
    for document_type in ("bill_of_lading", "commercial_invoice", "packing_list"):
        if not any(
            corpus.document_type_by_sample_id[crop.sample_id] == document_type for crop in selected
        ) and not take(
            lambda crop, value=document_type: (
                corpus.document_type_by_sample_id[crop.sample_id] == value
            )
        ):
            raise SourceAttestedVoyageReplayCorpusError(
                "voyage_replay_general_anchor_coverage_invalid"
            )
    for profile in _PROFILES:
        if not any(corpus.profile_by_sample_id[crop.sample_id] == profile for crop in selected):
            if not take(
                lambda crop, value=profile: corpus.profile_by_sample_id[crop.sample_id] == value
            ):
                raise SourceAttestedVoyageReplayCorpusError(
                    "voyage_replay_general_anchor_coverage_invalid"
                )
    for crop in ranked:
        if len(selected) == 250:
            break
        if crop.family_id not in selected_families:
            selected.append(crop)
            selected_ids.add(crop.sample_id)
            selected_families.add(crop.family_id)
    selected.sort(key=lambda item: item.sample_id)
    if (
        len(selected) != 250
        or len(selected_families) != 250
        or len(selected_ids) != 250
        or any(
            corpus.field_key_by_sample_id[crop.sample_id] == "bill_of_lading:voyage"
            for crop in selected
        )
        or set(field_keys) != {corpus.field_key_by_sample_id[crop.sample_id] for crop in selected}
        or set(_PROFILES) != {corpus.profile_by_sample_id[crop.sample_id] for crop in selected}
    ):
        raise SourceAttestedVoyageReplayCorpusError("voyage_replay_general_anchor_coverage_invalid")
    return tuple(selected)


def _general_statistics(
    crops: Sequence[TrainingCropV1], corpus: Any
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    fields: dict[str, int] = {}
    documents = {
        "bill_of_lading": 0,
        "commercial_invoice": 0,
        "packing_list": 0,
    }
    profiles = {profile: 0 for profile in _PROFILES}
    for crop in crops:
        field = corpus.field_key_by_sample_id[crop.sample_id]
        document = corpus.document_type_by_sample_id[crop.sample_id]
        profile = corpus.profile_by_sample_id[crop.sample_id]
        fields[field] = fields.get(field, 0) + 1
        documents[document] += 1
        profiles[profile] += 1
    return fields, documents, profiles


def _manifest(
    *,
    split: Literal["train", "development"],
    v2_corpus: Any,
    voyage_crops: Sequence[VoyageReplayCropV1],
    general_crops: Sequence[TrainingCropV1],
    audit_family_ids: Sequence[str],
    audit_source_sha256s: Sequence[str],
) -> VoyageReplayCorpusManifestV1:
    fields, documents, profiles = _general_statistics(general_crops, v2_corpus)
    all_fields = dict(fields)
    all_fields["bill_of_lading:voyage"] = all_fields.get("bill_of_lading:voyage", 0) + len(
        voyage_crops
    )
    family_ids = (
        v2_corpus.train_family_ids if split == "train" else v2_corpus.development_family_ids
    )
    return VoyageReplayCorpusManifestV1(
        split=split,
        purpose="optimizer" if split == "train" else "validation-only",
        source_receipt_sha256=v2_corpus.source_receipt_sha256,
        raster_receipt_sha256=v2_corpus.raster_receipt_sha256,
        selection_receipt_relative_path=v2_corpus.selection_receipt_relative_path,
        selection_receipt_sha256=v2_corpus.selection_receipt_sha256,
        parent_selection_receipt_sha256=v2_corpus.parent_selection_receipt_sha256,
        selected_family_count=len(family_ids),
        selected_family_ids_sha256=_hash_list(sorted(family_ids)),
        excluded_family_ids_sha256=v2_corpus.excluded_family_ids_sha256,
        independent_audit_recipe_sha256=bl_audit.BL_REPAIR_AUDIT_RECIPE_SHA256,
        independent_audit_family_ids_sha256=_hash_list(sorted(audit_family_ids)),
        independent_audit_source_sha256s_sha256=_hash_list(sorted(audit_source_sha256s)),
        voyage_crop_count=len(voyage_crops),
        general_crop_count=len(general_crops),
        total_crop_count=len(voyage_crops) + len(general_crops),
        voyage_recipe_set_sha256=_RECIPE_SET_SHA256,
        voyage_recipe_sha256s=dict(_RECIPE_SHA256S),
        voyage_recipe_lane_counts={
            recipe_id: sum(crop.recipe_id == recipe_id for crop in voyage_crops)
            for recipe_id in _RECIPE_BY_ID
        },
        voyage_sample_ids_sha256=_hash_list(sorted(crop.sample_id for crop in voyage_crops)),
        voyage_label_sha256s_sha256=_hash_list(sorted(crop.label_sha256 for crop in voyage_crops)),
        voyage_label_pattern_sha256=_VOYAGE_PATTERN_SHA256,
        general_sample_ids_sha256=_hash_list(sorted(crop.sample_id for crop in general_crops)),
        general_field_ids_sha256=_hash_list(sorted(fields)),
        general_field_crop_counts=fields,
        all_field_key_ids_sha256=_hash_list(sorted(all_fields)),
        all_field_key_crop_counts=all_fields,
        general_document_type_crop_counts=documents,
        general_profile_crop_counts=profiles,
    )


def build_verified_public_raster_text_voyage_replay_corpus_v1(
    document_preview_root: Path,
    *,
    seed: int = _DEFAULT_SEED,
) -> VerifiedPublicRasterTextVoyageReplayCorpusV1:
    """Build the fixed 750 optimizer and 400 validation crop identities."""

    if seed != _DEFAULT_SEED:
        raise SourceAttestedVoyageReplayCorpusError("voyage_replay_seed_invalid")
    try:
        root = v2._root(document_preview_root)
        parent = v2.build_verified_leakage_safe_public_crop_corpus_v2(root, seed=seed)
        raster_receipt, source_receipt, raster_raw, source_raw = _read_dataset(root)
        if (
            parent.source_receipt_sha256 != sha256_bytes(source_raw)
            or parent.raster_receipt_sha256 != sha256_bytes(raster_raw)
            or len(parent.train_family_ids) != 500
            or len(parent.development_family_ids) != 100
            or len(parent.excluded_family_ids) != 500
            or set(parent.train_family_ids) & set(parent.excluded_family_ids)
            or set(parent.train_family_ids) & set(parent.development_family_ids)
            or not set(parent.development_family_ids).issubset(set(parent.excluded_family_ids))
            or parent.holdout_overlap_count != 0
        ):
            raise SourceAttestedVoyageReplayCorpusError("voyage_replay_family_leakage")
        audit_families, audit_sources = _audit_exclusions()
        selected_families = set(parent.train_family_ids) | set(parent.development_family_ids)
        selected_source_sha256s = {
            str(item.get("source_pdf_sha256"))
            for item in source_receipt["assets"]
            if isinstance(item, dict) and item.get("family_id") in selected_families
        }
        if set(audit_families) & selected_families or set(audit_sources) & selected_source_sha256s:
            raise SourceAttestedVoyageReplayCorpusError("voyage_replay_independent_audit_leakage")
        train_assets = _verified_voyage_assets(
            root=root,
            source_receipt=source_receipt,
            raster_receipt=raster_receipt,
            family_ids=set(parent.train_family_ids),
            split="train",
        )
        development_assets = _verified_voyage_assets(
            root=root,
            source_receipt=source_receipt,
            raster_receipt=raster_receipt,
            family_ids=set(parent.development_family_ids),
            split="development",
        )
        (
            train_voyage,
            train_labels,
            train_profiles,
            train_recipes,
        ) = _compile_voyage_split(assets=train_assets, split="train", seed=seed)
        (
            development_voyage,
            development_labels,
            development_profiles,
            development_recipes,
        ) = _compile_voyage_split(
            assets=development_assets,
            split="development",
            seed=seed,
        )
        train_general = _select_general_train_anchors(parent, seed=seed)
        development_general = tuple(
            sorted(parent.development_crops, key=lambda item: item.sample_id)
        )
        if len(development_general) != 200 or {crop.family_id for crop in train_general} & {
            crop.family_id for crop in development_general
        }:
            raise SourceAttestedVoyageReplayCorpusError(
                "voyage_replay_general_anchor_coverage_invalid"
            )
        labels = {
            **train_labels,
            **development_labels,
            **{
                crop.sample_id: parent.labels_by_sample_id[crop.sample_id]
                for crop in (*train_general, *development_general)
            },
        }
        profiles = {
            **train_profiles,
            **development_profiles,
            **{
                crop.sample_id: parent.profile_by_sample_id[crop.sample_id]
                for crop in (*train_general, *development_general)
            },
        }
        field_keys = {
            **{
                crop.sample_id: "bill_of_lading:voyage"
                for crop in (*train_voyage, *development_voyage)
            },
            **{
                crop.sample_id: parent.field_key_by_sample_id[crop.sample_id]
                for crop in (*train_general, *development_general)
            },
        }
        all_crops: tuple[TrainingCropV1, ...] = (
            *train_voyage,
            *train_general,
            *development_voyage,
            *development_general,
        )
        if (
            len(all_crops) != 1_150
            or len({crop.sample_id for crop in all_crops}) != 1_150
            or set(labels) != {crop.sample_id for crop in all_crops}
            or set(profiles) != set(labels)
            or set(field_keys) != set(labels)
            or any(
                sha256_bytes(labels[crop.sample_id].encode("utf-8")) != crop.label_sha256
                for crop in all_crops
            )
        ):
            raise SourceAttestedVoyageReplayCorpusError("voyage_replay_sample_collision")
        train_manifest = _manifest(
            split="train",
            v2_corpus=parent,
            voyage_crops=train_voyage,
            general_crops=train_general,
            audit_family_ids=audit_families,
            audit_source_sha256s=audit_sources,
        )
        development_manifest = _manifest(
            split="development",
            v2_corpus=parent,
            voyage_crops=development_voyage,
            general_crops=development_general,
            audit_family_ids=audit_families,
            audit_source_sha256s=audit_sources,
        )
        return VerifiedPublicRasterTextVoyageReplayCorpusV1(
            train_voyage_crops=train_voyage,
            train_general_crops=train_general,
            development_voyage_crops=development_voyage,
            development_general_crops=development_general,
            labels_by_sample_id=labels,
            profile_by_sample_id=profiles,
            field_key_by_sample_id=field_keys,
            recipe_by_sample_id={**train_recipes, **development_recipes},
            train_manifest=train_manifest,
            development_manifest=development_manifest,
        )
    except SourceAttestedVoyageReplayCorpusError:
        raise
    except Exception:
        raise SourceAttestedVoyageReplayCorpusError("voyage_replay_corpus_invalid") from None


def prepare_voyage_replay_crop_image_v1(
    document_preview_root: Path,
    crop: VoyageReplayCropV1,
) -> Any:
    """Verify, crop, and deterministically degrade one voyage input image."""

    try:
        from PIL import Image, ImageFilter

        root = v2._root(document_preview_root)
        recipe = _RECIPE_BY_ID[crop.recipe_id]
        if (
            recipe.source_profile_id != crop.source_profile_id
            or crop.recipe_sha256 != _RECIPE_SHA256S[crop.recipe_id]
        ):
            raise ValueError
        path = v2._safe(root, crop.asset_relative_path)
        if sha256_file(path) != crop.raster_sha256:
            raise ValueError
        page = v2._decode_rgb(
            path,
            profile_id=crop.source_profile_id,
            width=crop.page_width,
            height=crop.page_height,
        )
        x, y, width, height = crop.bbox
        height_pixels = height * crop.page_height
        horizontal = min(
            recipe.horizontal_max_pixels,
            max(
                recipe.horizontal_min_pixels,
                int(math.ceil(recipe.horizontal_height_ratio * height_pixels)),
            ),
        )
        vertical = int(math.ceil(recipe.vertical_height_ratio * height_pixels))
        left = max(0, math.floor(x * crop.page_width) - horizontal)
        top = max(0, math.floor(y * crop.page_height) - vertical)
        right = min(
            crop.page_width,
            math.ceil((x + width) * crop.page_width) + horizontal,
        )
        bottom = min(
            crop.page_height,
            math.ceil((y + height) * crop.page_height) + vertical,
        )
        if right - left < 2 or bottom - top < 2:
            raise ValueError
        image = page.crop((left, top, right, bottom))
        if recipe.degradation == "gaussian0p6-jpeg75-v1":
            image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
            encoded = BytesIO()
            image.save(
                encoded,
                format="JPEG",
                quality=recipe.jpeg_quality,
                subsampling=recipe.jpeg_subsampling,
                progressive=recipe.jpeg_progressive,
                optimize=recipe.jpeg_optimize,
            )
            encoded.seek(0)
            with Image.open(encoded) as decoded:
                decoded.load()
                image = decoded.convert("RGB")
        resized_width = min(
            recipe.maximum_output_width,
            max(8, int(math.ceil(recipe.output_height * image.width / image.height))),
        )
        return image.resize((resized_width, recipe.output_height), Image.Resampling.BILINEAR)
    except SourceAttestedVoyageReplayCorpusError:
        raise
    except Exception:
        raise SourceAttestedVoyageReplayCorpusError("voyage_replay_crop_material_invalid") from None


__all__ = [
    "SourceAttestedVoyageReplayCorpusError",
    "VOYAGE_REPLAY_RECIPES_V1",
    "VerifiedPublicRasterTextVoyageReplayCorpusV1",
    "VoyageReplayCorpusManifestV1",
    "VoyageReplayCropRecipeV1",
    "VoyageReplayCropV1",
    "build_verified_public_raster_text_voyage_replay_corpus_v1",
    "prepare_voyage_replay_crop_image_v1",
    "voyage_replay_recipe_sha256s_v1",
]
