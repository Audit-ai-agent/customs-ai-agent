from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace
from typing import Iterator

from PIL import Image
import pytest

from customs_ai_agent.evaluation import (
    source_attested_raster_text_recognizer_training_runtime as training_runtime,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    PageLruCache,
    SourceAttestedRecognizerTrainingError,
    TrainingCropV1,
    build_verified_gold_crop_corpus,
    canonical_json_bytes,
    canonical_json_file_bytes,
    create_recognizer_training_stage,
    deterministic_width_batches,
    estimated_recognition_width,
    export_tiny_paddle_static_recognizer,
    load_verified_source_attested_raster_text_recognizer_release,
    publish_source_attested_raster_text_recognizer_release,
    sha256_bytes,
    sha256_file,
    training_receipt_sha256,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_file_bytes(value))


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _receipt_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_raster_text_recognizer_training_receipt",
        "trainer_revision": "source-attested-tiny-crnn-bigru-ctc-v1",
        "stage": "overfit",
        "training_intent_sha256": "1" * 64,
        "train_manifest_sha256": "2" * 64,
        "development_manifest_sha256": "3" * 64,
        "character_dictionary_sha256": "4" * 64,
        "package_inventory_sha256": "5" * 64,
        "training_runtime_bundle_manifest_sha256": "6" * 64,
        "parent_python_revision": "3.11",
        "training_module_sha256": "7" * 64,
        "seed_sha256": "8" * 64,
        "batch_size": 32,
        "maximum_epochs": 8,
        "include_merged_lines": True,
        "initialization": "deterministic_random_no_pretrained_checkpoint",
        "framework_license": "Apache-2.0",
        "model_revision": "tiny-ds-cnn-bigru-ctc-v1",
        "completed_epochs": 2,
        "best_epoch": 1,
        "early_stopped": False,
        "development_cer": 0.125,
        "development_exact_match": 0.75,
    }


def test_windows_training_bootstrap_requires_short_direct_module_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_root = tmp_path / "bundle" / "sidecar" / "module-root"
    module_root.mkdir(parents=True)
    bundle = SimpleNamespace(
        bundle_root=tmp_path / "bundle",
        manifest=SimpleNamespace(sidecar_module_root_relative_path="sidecar/module-root"),
    )
    calls: list[tuple[Path, bool]] = []
    short_root = Path("C:/DIRECT~1/MODROOT")

    def short_path(path: Path, *, force_short: bool = False) -> Path:
        calls.append((path, force_short))
        return short_root

    monkeypatch.setattr(training_runtime, "_operational_path", short_path)
    monkeypatch.setattr(training_runtime.os, "name", "nt")

    canonical, operational = training_runtime._verified_operational_module_root(bundle)

    assert canonical == module_root.resolve(strict=True)
    assert operational == short_root
    assert calls == [(canonical, True)]


def test_training_network_guard_blocks_all_routes_and_restores() -> None:
    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.socket.sendto,
        socket.create_connection,
        socket.getaddrinfo,
        socket.gethostbyname,
        socket.gethostbyname_ex,
    )
    stream = socket.socket()
    try:
        with training_runtime.network_disabled():
            calls = (
                lambda: stream.connect(("127.0.0.1", 9)),
                lambda: stream.connect_ex(("127.0.0.1", 9)),
                lambda: stream.sendto(b"x", ("127.0.0.1", 9)),
                lambda: socket.create_connection(("127.0.0.1", 9)),
                lambda: socket.getaddrinfo("localhost", 9),
                lambda: socket.gethostbyname("localhost"),
                lambda: socket.gethostbyname_ex("localhost"),
            )
            for call in calls:
                with pytest.raises(OSError, match="network disabled"):
                    call()
    finally:
        stream.close()
    assert (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.socket.sendto,
        socket.create_connection,
        socket.getaddrinfo,
        socket.gethostbyname,
        socket.gethostbyname_ex,
    ) == originals


def _release_stage(root: Path) -> Path:
    stage = create_recognizer_training_stage(root)
    text_rec = stage / "text-rec"
    text_rec.mkdir()
    (text_rec / "config.json").write_bytes(b'{"model":"tiny"}\n')
    (text_rec / "inference.json").write_bytes(b'{"graph":"static"}\n')
    (text_rec / "inference.pdiparams").write_bytes(b"parameter-bytes")
    (text_rec / "inference.yml").write_bytes(b"Global:\n  model_type: rec\n")
    return stage


def _build_dataset(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "synthetic-raster-v1"
    source_root = root / "sources"
    source_root.mkdir()
    source_assets: list[dict[str, object]] = []
    raster_assets: list[dict[str, object]] = []
    paths: dict[str, Path] = {}

    for split, family in (("train", "family-train"), ("development", "family-dev")):
        source = source_root / f"{split}.pdf"
        source.write_bytes(f"source-pdf-{split}".encode("ascii"))
        source_sha = sha256_file(source)
        source_assets.append(
            {
                "family_id": family,
                "source_relative_path": source.name,
                "source_pdf_sha256": source_sha,
                "byte_size": source.stat().st_size,
            }
        )

        image = dataset / "assets" / f"{split}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 50), color="white").save(image, format="PNG")
        asset_id = f"asset-{split}"
        gold = dataset / "gold" / f"{split}.json"
        gold_payload = {
            "asset_id": asset_id,
            "family_id": family,
            "profile_id": "png-216",
            "source_pdf_sha256": source_sha,
            "spans": [
                {
                    "span_id": f"{split}-span-1",
                    "reading_order": 0,
                    "bbox": {"x": 0.10, "y": 0.10, "width": 0.10, "height": 0.10},
                    "text": "AB",
                },
                {
                    "span_id": f"{split}-span-2",
                    "reading_order": 1,
                    "bbox": {"x": 0.22, "y": 0.10, "width": 0.08, "height": 0.10},
                    "text": "12",
                },
            ],
        }
        _write_json(gold, gold_payload)
        raster_assets.append(
            {
                "asset_id": asset_id,
                "family_id": family,
                "split": split,
                "profile_id": "png-216",
                "media_type": "image/png",
                "source_pdf_sha256": source_sha,
                "asset_relative_path": image.relative_to(root).as_posix(),
                "gold_relative_path": gold.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image),
                "gold_sha256": sha256_file(gold),
                "byte_size": image.stat().st_size,
                "width": 100,
                "height": 50,
            }
        )
        paths[f"source_{split}"] = source

    source_receipt = dataset / "source-receipt.json"
    _write_json(
        source_receipt,
        {
            "source_root_relative_path": "sources",
            "assets": source_assets,
        },
    )
    raster_receipt = dataset / "synthetic-raster-receipt.json"
    _write_json(
        raster_receipt,
        {
            "dataset_id": "synthetic-raster-v1",
            "source_receipt_sha256": sha256_file(source_receipt),
            "assets": raster_assets,
        },
    )
    paths["source_receipt"] = source_receipt
    paths["raster_receipt"] = raster_receipt
    return paths


def _crop(index: int, *, width: float) -> TrainingCropV1:
    digest = sha256_bytes(f"sample-{index}".encode("ascii"))
    return TrainingCropV1(
        sample_id=digest,
        asset_id=f"asset-{index}",
        family_id=f"family-{index}",
        split="train",
        asset_relative_path=f"assets/{index}.png",
        gold_relative_path=f"gold/{index}.json",
        source_pdf_sha256="1" * 64,
        raster_sha256="2" * 64,
        gold_sha256="3" * 64,
        span_ids=(f"span-{index}",),
        bbox=(0.0, 0.0, width, 0.1),
        label_sha256="4" * 64,
        label_length=2,
        page_width=100,
        page_height=100,
    )


def test_publish_load_and_idempotent_reuse_are_content_addressed(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)

    first = publish_source_attested_raster_text_recognizer_release(
        root,
        staged_release=_release_stage(root),
        receipt_payload=_receipt_payload(),
    )
    second_stage = _release_stage(root)
    second = publish_source_attested_raster_text_recognizer_release(
        root,
        staged_release=second_stage,
        receipt_payload=_receipt_payload(),
    )

    assert first.release_sha256 == second.release_sha256
    assert first.release_path.name == first.receipt.receipt_sha256
    assert first.receipt.receipt_sha256 == training_receipt_sha256(first.receipt)
    assert not second_stage.exists()
    loaded = load_verified_source_attested_raster_text_recognizer_release(
        root, first.release_sha256
    )
    assert loaded.file_sha256s == first.file_sha256s
    assert set(loaded.file_paths) == {
        "text-rec/config.json",
        "text-rec/inference.json",
        "text-rec/inference.pdiparams",
        "text-rec/inference.yml",
        "training-receipt.json",
    }


def test_release_loader_rejects_an_extra_file(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)
    release = publish_source_attested_raster_text_recognizer_release(
        root,
        staged_release=_release_stage(root),
        receipt_payload=_receipt_payload(),
    )
    (release.release_path / "unexpected.bin").write_bytes(b"not-declared")

    with pytest.raises(SourceAttestedRecognizerTrainingError) as raised:
        load_verified_source_attested_raster_text_recognizer_release(root, release.release_sha256)

    assert raised.value.code == "recognizer_release_tree_invalid"


def test_release_directory_name_must_bind_to_receipt(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)
    release = publish_source_attested_raster_text_recognizer_release(
        root,
        staged_release=_release_stage(root),
        receipt_payload=_receipt_payload(),
    )
    wrong_sha = "f" * 64 if release.release_sha256 != "f" * 64 else "e" * 64
    wrong_path = release.release_path.parent / wrong_sha
    shutil.copytree(release.release_path, wrong_path)

    with pytest.raises(SourceAttestedRecognizerTrainingError) as raised:
        load_verified_source_attested_raster_text_recognizer_release(root, wrong_sha)

    assert raised.value.code == "recognizer_release_binding_invalid"


def test_verified_gold_crop_corpus_is_deterministic_and_keeps_labels_in_memory(
    tmp_path: Path,
) -> None:
    root = _preview_root(tmp_path)
    _build_dataset(root)

    first = build_verified_gold_crop_corpus(
        root, stage="overfit", seed=42, include_merged_lines=True
    )
    second = build_verified_gold_crop_corpus(
        root, stage="overfit", seed=42, include_merged_lines=True
    )

    assert first.train_manifest == second.train_manifest
    assert first.development_manifest == second.development_manifest
    assert first.train_crops == second.train_crops
    assert first.development_crops == second.development_crops
    assert first.train_manifest.crop_count == 3
    assert first.development_manifest.crop_count == 3
    assert sum(crop.sample_kind == "merged_line" for crop in first.train_crops) == 1
    assert sum(crop.sample_kind == "merged_line" for crop in first.development_crops) == 1
    assert set(first.labels_by_sample_id.values()) == {"AB", "12", "AB 12"}
    assert set(first.characters) == set("AB 12")

    public_material = canonical_json_bytes(
        {
            "train_manifest": first.train_manifest.model_dump(mode="json"),
            "development_manifest": first.development_manifest.model_dump(mode="json"),
            "train_crops": [crop.model_dump(mode="json") for crop in first.train_crops],
            "development_crops": [crop.model_dump(mode="json") for crop in first.development_crops],
        }
    )
    assert b'"text"' not in public_material
    assert b"AB" not in public_material
    assert b"AB 12" not in public_material


def test_gold_corpus_rejects_raw_source_receipt_sha_mismatch(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)
    paths = _build_dataset(root)
    raster_receipt = json.loads(paths["raster_receipt"].read_bytes())
    raster_receipt["source_receipt_sha256"] = "0" * 64
    _write_json(paths["raster_receipt"], raster_receipt)

    with pytest.raises(SourceAttestedRecognizerTrainingError) as raised:
        build_verified_gold_crop_corpus(root)

    assert raised.value.code == "training_source_receipt_drift"


def test_gold_corpus_rejects_source_root_traversal(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)
    paths = _build_dataset(root)
    outside = root.parent / "outside"
    outside.mkdir()
    shutil.copyfile(paths["source_train"], outside / "train.pdf")
    shutil.copyfile(paths["source_development"], outside / "development.pdf")
    source_receipt = json.loads(paths["source_receipt"].read_bytes())
    source_receipt["source_root_relative_path"] = "../outside"
    _write_json(paths["source_receipt"], source_receipt)
    raster_receipt = json.loads(paths["raster_receipt"].read_bytes())
    raster_receipt["source_receipt_sha256"] = sha256_file(paths["source_receipt"])
    _write_json(paths["raster_receipt"], raster_receipt)

    with pytest.raises(SourceAttestedRecognizerTrainingError) as raised:
        build_verified_gold_crop_corpus(root)

    assert raised.value.code == "training_source_receipt_invalid"


def test_page_lru_cache_reuses_recent_pages_and_evicts_oldest() -> None:
    cache: PageLruCache[str] = PageLruCache(capacity=2)
    loads: list[str] = []

    def load(value: str) -> str:
        loads.append(value)
        return value

    assert cache.get("a", lambda: load("A")) == "A"
    assert cache.get("b", lambda: load("B")) == "B"
    assert cache.get("a", lambda: load("A-again")) == "A"
    assert cache.get("c", lambda: load("C")) == "C"

    assert loads == ["A", "B", "C"]
    assert cache.keys == ("a", "c")


def test_width_buckets_are_bounded_deterministic_and_lossless() -> None:
    crops = tuple(
        _crop(index, width=width)
        for index, width in enumerate((0.02, 0.04, 0.20, 0.30, 0.50, 0.80, 0.90))
    )

    first = deterministic_width_batches(crops, epoch=3, seed=7, batch_size=2)
    second = deterministic_width_batches(crops, epoch=3, seed=7, batch_size=2)

    assert first == second
    assert all(1 <= len(batch) <= 2 for batch in first)
    assert sorted(crop.sample_id for batch in first for crop in batch) == sorted(
        crop.sample_id for crop in crops
    )
    assert estimated_recognition_width(crops[0]) == 10
    assert estimated_recognition_width(crops[-1]) == 432


def test_static_export_uses_dynamic_width_without_importing_paddle(tmp_path: Path) -> None:
    captures: dict[str, object] = {}

    class FakeInputSpec:
        def __init__(self, *, shape: list[int | None], dtype: str) -> None:
            captures["shape"] = shape
            captures["dtype"] = dtype

    class FakeModel:
        def eval(self) -> None:
            captures["evaluated"] = True

    def to_static(model: object, *, input_spec: list[object], full_graph: bool) -> object:
        captures["input_spec_count"] = len(input_spec)
        captures["full_graph"] = full_graph
        return model

    def save(_model: object, prefix: str) -> None:
        Path(f"{prefix}.json").write_bytes(b"static-graph")
        Path(f"{prefix}.pdiparams").write_bytes(b"static-parameters")

    fake_paddle = SimpleNamespace(
        jit=SimpleNamespace(to_static=to_static, save=save),
        static=SimpleNamespace(InputSpec=FakeInputSpec),
    )
    text_rec = tmp_path / "stage" / "text-rec"

    hashes = export_tiny_paddle_static_recognizer(
        fake_paddle,
        model=FakeModel(),
        text_rec_directory=text_rec,
        characters=(" ", "1", "2", "A", "B"),
    )

    assert captures == {
        "evaluated": True,
        "shape": [None, 3, 48, None],
        "dtype": "float32",
        "input_spec_count": 1,
        "full_graph": True,
    }
    assert set(hashes) == {
        "text-rec/config.json",
        "text-rec/inference.json",
        "text-rec/inference.pdiparams",
        "text-rec/inference.yml",
    }
    assert all(len(value) == 64 for value in hashes.values())


def test_training_runtime_replays_current_intent_before_reusing_a_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    release = SimpleNamespace(release_sha256="a" * 64)
    expected = SimpleNamespace(release=release, reused=True)
    inventory = root / "bundle" / "inventory.json"
    inventory.parent.mkdir()
    inventory.write_bytes(b"{}")
    bundle = SimpleNamespace(manifest_sha256="b" * 64)
    events: list[str] = []
    monkeypatch.setattr(
        training_runtime,
        "discover_verified_source_attested_recognizer_release",
        lambda _root: events.append("historical-release-check") or release,
    )

    @contextmanager
    def verified_imports(_bundle: object) -> Iterator[Path]:
        yield inventory

    monkeypatch.setattr(
        training_runtime,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(training_runtime, "verified_direct_training_imports", verified_imports)
    monkeypatch.setattr(
        training_runtime,
        "train_source_attested_raster_text_recognizer_overfit",
        lambda *_args, **_kwargs: events.append("current-intent-replay") or expected,
    )
    monkeypatch.setattr(
        training_runtime,
        "_verify_release_training_provenance",
        lambda *_args: events.append("published-release-check"),
    )

    result = training_runtime.ensure_source_attested_raster_text_recognizer_overfit(root)

    assert result is expected
    assert events == [
        "historical-release-check",
        "current-intent-replay",
        "published-release-check",
    ]


def test_training_runtime_clean_state_is_locked_offline_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    events: list[str] = []
    inventory = root / "bundle" / "inventories" / "packages.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_bytes(b"{}")
    bundle = SimpleNamespace(manifest_sha256="b" * 64)
    release = SimpleNamespace(release_sha256="c" * 64)
    expected = SimpleNamespace(release=release, reused=False)

    monkeypatch.setattr(
        training_runtime,
        "discover_verified_source_attested_recognizer_release",
        lambda _root: None,
    )

    @contextmanager
    def deny_network() -> Iterator[None]:
        events.append("network-enter")
        yield
        events.append("network-exit")

    @contextmanager
    def verified_imports(value: object) -> Iterator[Path]:
        assert value is bundle
        events.append("imports-enter")
        yield inventory
        events.append("imports-exit")

    def materialize(**kwargs: object) -> object:
        assert kwargs == {"document_preview_root": root}
        events.append("bundle")
        return bundle

    def train(path: Path, **kwargs: object) -> object:
        assert path == root
        assert kwargs == {
            "package_inventory_path": inventory,
            "runtime_bundle_manifest_sha256": "b" * 64,
        }
        events.append("overfit")
        return expected

    monkeypatch.setattr(training_runtime, "network_disabled", deny_network)
    monkeypatch.setattr(training_runtime, "verified_direct_training_imports", verified_imports)
    monkeypatch.setattr(
        training_runtime,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        materialize,
    )
    monkeypatch.setattr(
        training_runtime,
        "train_source_attested_raster_text_recognizer_overfit",
        train,
    )
    monkeypatch.setattr(
        training_runtime,
        "_verify_release_training_provenance",
        lambda *_args: events.append("provenance"),
    )

    result = training_runtime.ensure_source_attested_raster_text_recognizer_overfit(root)

    assert result is expected
    assert events == [
        "network-enter",
        "bundle",
        "imports-enter",
        "overfit",
        "imports-exit",
        "provenance",
        "network-exit",
    ]
    assert (root / ".source-attested-raster-text-recognizer-training.lock").is_file()


def test_training_runtime_rejects_an_undeclared_release_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    releases = root / "raster-text-trained-recognizers-v1" / "releases"
    (releases / "undeclared").mkdir(parents=True)
    monkeypatch.setattr(
        training_runtime,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: pytest.fail("invalid releases must not trigger fallback training"),
    )

    with pytest.raises(SourceAttestedRecognizerTrainingError) as raised:
        training_runtime.ensure_source_attested_raster_text_recognizer_overfit(root)

    assert raised.value.code == "recognizer_release_collection_invalid"


def test_training_runtime_rejects_training_module_drift_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    release_sha256 = "a" * 64
    releases = root / "raster-text-trained-recognizers-v1" / "releases"
    (releases / release_sha256).mkdir(parents=True)
    release = SimpleNamespace(
        release_sha256=release_sha256,
        receipt=SimpleNamespace(
            parent_python_revision=(
                f"{training_runtime.sys.version_info.major}."
                f"{training_runtime.sys.version_info.minor}"
            ),
            training_module_sha256="0" * 64,
            training_runtime_bundle_manifest_sha256="b" * 64,
            package_inventory_sha256="c" * 64,
            development_cer=0.01,
            development_exact_match=0.99,
            stage="overfit",
        ),
    )
    monkeypatch.setattr(
        training_runtime,
        "load_verified_source_attested_raster_text_recognizer_release",
        lambda *_args: release,
    )
    monkeypatch.setattr(
        training_runtime,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: pytest.fail("provenance drift must not trigger fallback training"),
    )

    with pytest.raises(SourceAttestedRecognizerTrainingError) as raised:
        training_runtime.ensure_source_attested_raster_text_recognizer_overfit(root)

    assert raised.value.code == "training_release_provenance_drift"
