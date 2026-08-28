from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_voyage_replay as replay,
)
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_voyage_replay_corpus as corpus_source,
)
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_finetuning as parent_source,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    TrainingCropV1,
)
from tools import finetune_source_attested_public_raster_text_voyage_replay as cli


def _crop(index: int, *, voyage: bool) -> TrainingCropV1:
    sample_id = hashlib.sha256(f"{'v' if voyage else 'g'}:{index}".encode()).hexdigest()
    width = 0.02 if voyage else 0.01 + (index % 250) / 1_000
    return TrainingCropV1(
        sample_id=sample_id,
        asset_id=f"asset-{index}",
        family_id=f"family-{index}",
        split="train",
        asset_relative_path=f"assets/{index}.png",
        gold_relative_path=f"gold/{index}.json",
        source_pdf_sha256="1" * 64,
        raster_sha256=hashlib.sha256(f"raster:{index}".encode()).hexdigest(),
        gold_sha256="3" * 64,
        span_ids=(f"span-{index}",),
        bbox=(0.1, 0.1, width, 0.02),
        label_sha256="4" * 64,
        label_length=5,
        page_width=2_000,
        page_height=1_000,
    )


def _metric(crops: int, *, characters: int, errors: int, exact: int) -> replay.ReplayMetricV1:
    return replay.ReplayMetricV1(
        crop_count=crops,
        reference_character_count=characters,
        character_error_count=errors,
        exact_match_count=exact,
        character_error_rate=errors / characters,
        exact_match=exact / crops,
    )


def _evidence(
    *, field_a_exact: int, field_b_exact: int
) -> replay.VoyageReplayDevelopmentEvidenceV1:
    perfect_overall = _metric(200, characters=1_000, errors=0, exact=200)
    perfect_lane = _metric(25, characters=125, errors=0, exact=25)
    general_overall = _metric(200, characters=1_000, errors=20, exact=190)
    return replay.VoyageReplayDevelopmentEvidenceV1(
        voyage_overall=perfect_overall,
        voyage_by_lane={lane: perfect_lane for lane in replay._VOYAGE_LANES},
        general_overall=general_overall,
        general_by_profile={
            "png-216": _metric(67, characters=335, errors=7, exact=64),
            "jpeg-216-q75": _metric(67, characters=335, errors=7, exact=64),
            "jpeg-150-q75": _metric(66, characters=330, errors=6, exact=62),
        },
        general_by_field_key={
            "bill_of_lading:port": _metric(100, characters=500, errors=10, exact=field_a_exact),
            "commercial_invoice:number": _metric(
                100, characters=500, errors=10, exact=field_b_exact
            ),
        },
    )


def test_deterministic_batches_are_exact_and_width_balanced() -> None:
    voyage = tuple(_crop(index, voyage=True) for index in range(500))
    general = tuple(_crop(10_000 + index, voyage=False) for index in range(250))

    first = replay.deterministic_voyage_replay_batches(voyage, general, epoch=0, seed=1_937_429)
    repeated = replay.deterministic_voyage_replay_batches(voyage, general, epoch=0, seed=1_937_429)
    second_epoch = replay.deterministic_voyage_replay_batches(
        voyage, general, epoch=1, seed=1_937_429
    )
    voyage_ids = {crop.sample_id for crop in voyage}
    general_ids = {crop.sample_id for crop in general}

    assert first == repeated
    assert first != second_epoch
    assert len(first) == 25
    assert all(sum(crop.sample_id in voyage_ids for crop in batch) == 20 for batch in first)
    assert all(sum(crop.sample_id in general_ids for crop in batch) == 10 for batch in first)
    assert {crop.sample_id for batch in first for crop in batch} == voyage_ids | general_ids
    general_maxima = [
        max(replay._estimated_replay_width(crop) for crop in batch if crop.sample_id in general_ids)
        for batch in first
    ]
    assert len(set(general_maxima)) == 25


def test_gate_rejects_field_regression_hidden_by_equal_aggregate() -> None:
    baseline = _evidence(field_a_exact=95, field_b_exact=95)
    regressed = _evidence(field_a_exact=90, field_b_exact=100)

    assert not replay.development_gate_passes(baseline=baseline, candidate=regressed)
    assert replay.development_gate_passes(baseline=baseline, candidate=baseline)


def test_gate_rejects_field_cer_regression_hidden_by_equal_exact_match() -> None:
    baseline = _evidence(field_a_exact=95, field_b_exact=95)
    regressed_field = _metric(100, characters=500, errors=11, exact=95)
    regressed = baseline.model_copy(
        update={
            "general_by_field_key": {
                **baseline.general_by_field_key,
                "bill_of_lading:port": regressed_field,
            }
        }
    )

    assert not replay.development_gate_passes(baseline=baseline, candidate=regressed)


def test_gate_rejects_one_imperfect_voyage_recipe_lane() -> None:
    baseline = _evidence(field_a_exact=95, field_b_exact=95)
    lane = replay._VOYAGE_LANES[0]
    candidate = baseline.model_copy(
        update={
            "voyage_by_lane": {
                **baseline.voyage_by_lane,
                lane: _metric(25, characters=125, errors=1, exact=24),
            }
        }
    )

    assert not replay.development_gate_passes(baseline=baseline, candidate=candidate)


def test_fixed_public_contract_exposes_no_epoch_override_or_plaintext_label() -> None:
    parser = cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--document-preview-root",
                "ignored",
                "--maximum-epochs",
                "9",
            ]
        )
    schema = replay.VoyageReplayReceiptV3.model_json_schema()
    encoded = json.dumps(schema, sort_keys=True)
    assert "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3" in encoded
    assert "exact-width-normalized-lru-v1" in encoded
    assert "1234E" not in encoded


def test_prepared_cache_is_fixed_to_512_mib() -> None:
    with pytest.raises(replay.VoyageReplayError) as raised:
        replay.PreparedCropLruCache(limit_bytes=1)
    assert raised.value.code == "voyage_replay_cache_policy_invalid"


def test_actual_corpus_manifest_roundtrip_uses_eight_recipe_lanes() -> None:
    recipe_sha256s = corpus_source.voyage_replay_recipe_sha256s_v1()
    selection_sha = "5" * 64
    fields = {
        "bill_of_lading:port": 100,
        "commercial_invoice:number": 100,
    }
    all_fields = {**fields, "bill_of_lading:voyage": 200}
    manifest = corpus_source.VoyageReplayCorpusManifestV1(
        split="development",
        purpose="validation-only",
        source_receipt_sha256="1" * 64,
        raster_receipt_sha256="2" * 64,
        selection_receipt_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{selection_sha}.json"
        ),
        selection_receipt_sha256=selection_sha,
        parent_selection_receipt_sha256="6" * 64,
        selected_family_count=100,
        selected_family_ids_sha256="7" * 64,
        excluded_family_ids_sha256="8" * 64,
        independent_audit_recipe_sha256="9" * 64,
        independent_audit_family_ids_sha256="a" * 64,
        independent_audit_source_sha256s_sha256="b" * 64,
        voyage_crop_count=200,
        general_crop_count=200,
        total_crop_count=400,
        voyage_recipe_set_sha256=corpus_source._RECIPE_SET_SHA256,
        voyage_recipe_sha256s=recipe_sha256s,
        voyage_recipe_lane_counts={key: 25 for key in recipe_sha256s},
        voyage_sample_ids_sha256="c" * 64,
        voyage_label_sha256s_sha256="d" * 64,
        voyage_label_pattern_sha256=corpus_source._VOYAGE_PATTERN_SHA256,
        general_sample_ids_sha256="e" * 64,
        general_field_ids_sha256="f" * 64,
        general_field_crop_counts=fields,
        all_field_key_ids_sha256=corpus_source._hash_list(sorted(all_fields)),
        all_field_key_crop_counts=all_fields,
        general_document_type_crop_counts={
            "bill_of_lading": 67,
            "commercial_invoice": 67,
            "packing_list": 66,
        },
        general_profile_crop_counts={
            "png-216": 67,
            "jpeg-216-q75": 67,
            "jpeg-150-q75": 66,
        },
    )

    assert (
        corpus_source.VoyageReplayCorpusManifestV1.model_validate_json(manifest.model_dump_json())
        == manifest
    )
    assert set(manifest.voyage_recipe_lane_counts) == set(replay._VOYAGE_LANES)


def test_recipe_aware_crop_has_exact_width_and_content_cache_hits(
    tmp_path: Path,
) -> None:
    recipe = corpus_source.VOYAGE_REPLAY_RECIPES_V1[0]
    label = "1234E"
    root = tmp_path / "_local" / "data" / "document-preview"
    asset = root / "assets" / "voyage.png"
    asset.parent.mkdir(parents=True)
    Image.new("RGB", (2_000, 1_000), "white").save(asset, format="PNG")
    crop_payload = _crop(999, voyage=True).model_dump()
    crop_payload["label_sha256"] = replay.sha256_bytes(label.encode())
    crop_payload["asset_relative_path"] = "assets/voyage.png"
    crop_payload["raster_sha256"] = replay.sha256_file(asset)
    crop = corpus_source.VoyageReplayCropV1(
        **crop_payload,
        recipe_id=recipe.recipe_id,
        recipe_sha256=corpus_source.voyage_replay_recipe_sha256s_v1()[recipe.recipe_id],
        source_profile_id=recipe.source_profile_id,
        field_id_sha256=replay.sha256_bytes(b"bill_of_lading:voyage"),
    )
    prepared = corpus_source.prepare_voyage_replay_crop_image_v1(root, crop)
    assert prepared.size == (replay._estimated_replay_width(crop), 48)

    cache = replay.PreparedCropLruCache()
    key = (crop.sample_id, crop.raster_sha256, crop.recipe_sha256)
    normalized = SimpleNamespace(nbytes=3 * 48 * prepared.width * 4)
    cache.put(key, normalized)
    assert cache.get(key) is normalized
    assert cache.get(key) is normalized


def test_parent_release_binding_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = parent_source.PublicFineTuneReceiptV2.model_construct(
        receipt_sha256=replay._PARENT_RECEIPT_SHA256,
        schema_version="2.0",
        trainer_revision="public-paddle-jit-ctc-leakage-safe-head-v2",
        model_revision="en-ppocrv5-mobile-rec-source-attested-head-v2",
        tuning_scope="head",
        train_holdout_overlap_count=0,
        trainable_tensor_count=2,
        trainable_parameter_count=52_998,
    )
    release = SimpleNamespace(
        release_sha256=replay._PARENT_RELEASE_SHA256,
        receipt=receipt,
    )
    monkeypatch.setattr(
        replay, "load_verified_public_finetuned_recognizer_release_v1", lambda *_: release
    )
    assert replay._verified_parent_release(Path("ignored")) is release

    release.release_sha256 = "0" * 64
    with pytest.raises(replay.VoyageReplayError) as raised:
        replay._verified_parent_release(Path("ignored"))
    assert raised.value.code == "voyage_replay_parent_release_invalid"


def test_parent_prefix_uses_force_short_before_appending_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_path = tmp_path / "한글-긴-경로" / ("release-" * 8)
    release_path.mkdir(parents=True)
    calls: list[tuple[Path, bool]] = []
    short_root = Path("C:/SHORT/RELEASE")

    def operational(path: Path, *, force_short: bool = False) -> Path:
        calls.append((path, force_short))
        return short_root

    monkeypatch.setattr(replay, "_operational_path", operational)
    prefix = replay._parent_recognizer_prefix(SimpleNamespace(release_path=release_path))

    assert calls == [(release_path.resolve(strict=True), True)]
    assert prefix == short_root / "text-rec" / "inference"


class _TopologyParameter:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape
        self.stop_gradient = True
        self.grad: object | None = None

    def clear_gradient(self) -> None:
        self.grad = None

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result


class _TopologyLayer:
    def __init__(self, parameters: tuple[_TopologyParameter, ...]) -> None:
        self.parameters = parameters

    def named_parameters(self):  # type: ignore[no-untyped-def]
        return tuple((f"param_{index}", value) for index, value in enumerate(self.parameters))

    def __call__(self, _value: object) -> object:
        return object()


class _TopologyPaddle:
    @staticmethod
    def zeros(_shape: tuple[int, ...], *, dtype: str) -> object:
        assert dtype == "float32"
        return object()

    @staticmethod
    def to_tensor(_value: object, *, dtype: str) -> object:
        assert dtype in {"int32", "int64"}
        return object()


def _topology_layer(*, suffix: str = "_0") -> _TopologyLayer:
    return _TopologyLayer(
        (
            _TopologyParameter("conv_0.w_0", (3, 3)),
            _TopologyParameter(f"linear_8.b_0_0{suffix}", (438,)),
            _TopologyParameter(f"linear_8.w_0_0{suffix}", (120, 438)),
        )
    )


def _topology_probe(
    monkeypatch: pytest.MonkeyPatch,
    layer: _TopologyLayer,
    *,
    connected: set[str],
) -> None:
    monkeypatch.setattr(replay, "_probability_output", lambda *_args, **_kwargs: object())

    class Loss:
        @staticmethod
        def backward() -> None:
            for parameter in layer.parameters:
                if parameter.name in connected and not parameter.stop_gradient:
                    parameter.grad = object()

    monkeypatch.setattr(replay, "_ctc_loss", lambda *_args, **_kwargs: Loss())


def test_voyage_selector_accepts_only_the_exact_parent_v2_jit_reload_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _topology_layer()
    connected = {
        "linear_8.b_0_0_0",
        "linear_8.w_0_0_0",
        "conv_0.w_0",
    }
    _topology_probe(monkeypatch, layer, connected=connected)

    selected = replay.select_voyage_replay_trainable_parameters_v1(
        _TopologyPaddle(), layer=layer, output_class_count=438
    )

    assert tuple(name for name, _parameter in selected) == (
        "linear_8.b_0_0",
        "linear_8.w_0_0",
    )
    assert tuple(parameter.name for _name, parameter in selected) == (
        "linear_8.b_0_0_0",
        "linear_8.w_0_0_0",
    )
    assert tuple(parameter.shape for _name, parameter in selected) == (
        (438,),
        (120, 438),
    )
    assert sum(parameter.numel() for _name, parameter in selected) == 52_998
    assert [parameter.stop_gradient for parameter in layer.parameters] == [True, False, False]
    assert all(parameter.grad is None for parameter in layer.parameters)


@pytest.mark.parametrize(
    ("suffix", "connected", "output_class_count"),
    (
        ("_1", {"linear_8.b_0_0_1", "linear_8.w_0_0_1"}, 438),
        ("_0", {"linear_8.b_0_0_0"}, 438),
        ("_0", {"linear_8.b_0_0_0", "linear_8.w_0_0_0"}, 437),
    ),
)
def test_voyage_selector_fails_closed_on_alias_connection_or_class_drift(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    connected: set[str],
    output_class_count: int,
) -> None:
    layer = _topology_layer(suffix=suffix)
    _topology_probe(monkeypatch, layer, connected=connected)

    with pytest.raises(parent_source.SourceAttestedPublicRecognizerFineTuneError) as raised:
        replay.select_voyage_replay_trainable_parameters_v1(
            _TopologyPaddle(),
            layer=layer,
            output_class_count=output_class_count,
        )

    assert raised.value.code == "public_finetune_head_topology_invalid"
    assert all(parameter.stop_gradient for parameter in layer.parameters)
    assert all(parameter.grad is None for parameter in layer.parameters)
