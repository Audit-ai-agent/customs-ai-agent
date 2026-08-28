from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from customs_ai_agent.evaluation import raster_text_server_paddle_bundle as server_bundle
from customs_ai_agent.evaluation.raster_text_server_paddle_bundle import (
    RasterTextServerPaddleBundleError,
    load_verified_raster_text_direct_server_paddle_bundle_v1,
    materialize_verified_raster_text_direct_server_paddle_bundle_v1,
    parse_ppocrv5_server_rec_character_dict_v1,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _server_yaml(characters: list[str]) -> bytes:
    def scalar(value: str) -> str:
        if value == "'":
            return "''''"
        if value in {"!", '"', "#", "-", ":", "{", "}", "|"}:
            return "'" + value.replace("'", "''") + "'"
        return value

    return (
        "Global:\n"
        "  model_name: PP-OCRv5_server_rec\n"
        "PostProcess:\n"
        "  name: CTCLabelDecode\n"
        "  character_dict:\n" + "".join(f"  - {scalar(value)}\n" for value in characters)
    ).encode("utf-8")


@dataclass
class _FakeBundle:
    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest_sha256: str
    manifest: Any
    hashes: dict[str, str]

    def file_sha256(self, path: str) -> str:
        return self.hashes[path]


@dataclass
class _Sources:
    root: Path
    mobile: _FakeBundle
    offline: _FakeBundle
    server_yaml: Path


@pytest.fixture
def fake_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Sources:
    root = tmp_path / "_local" / "data" / "document-preview"
    mobile_root = root / "source-mobile"
    offline_root = root / "source-offline"
    (mobile_root / "models" / "text-det").mkdir(parents=True)
    (offline_root / "models" / "text-rec").mkdir(parents=True)

    mobile_hashes: dict[str, str] = {}
    for name in ("config.json", "inference.json", "inference.pdiparams", "inference.yml"):
        path = mobile_root / "models" / "text-det" / name
        path.write_bytes(f"detector-{name}".encode())
        relative = f"models/text-det/{name}"
        mobile_hashes[relative] = _sha(path)

    characters = [f"glyph-{index:05d}" for index in range(18_383)]
    yml = offline_root / "models" / "text-rec" / "inference.yml"
    yml.write_bytes(_server_yaml(characters))
    offline_hashes: dict[str, str] = {}
    for name in ("inference.json", "inference.pdiparams"):
        path = offline_root / "models" / "text-rec" / name
        path.write_bytes(f"server-recognizer-{name}".encode())
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        relative = f"models/text-rec/{name}"
        offline_hashes[relative] = _sha(offline_root / relative)

    mobile = _FakeBundle(
        document_preview_root=root,
        bundle_root=mobile_root,
        bundle_root_relative_path=("raster-text-direct-mobile-bundles-v1/releases/" + "a" * 64),
        manifest_sha256="a" * 64,
        manifest=SimpleNamespace(text_detection_model_name="PP-OCRv5_mobile_det"),
        hashes=mobile_hashes,
    )
    offline = _FakeBundle(
        document_preview_root=root,
        bundle_root=offline_root,
        bundle_root_relative_path="raster-offline-bundles-v1/releases/" + "b" * 64,
        manifest_sha256="b" * 64,
        manifest=SimpleNamespace(local_model_root_relative_path="models"),
        hashes=offline_hashes,
    )

    def load_mobile(*, document_preview_root: Path, manifest_sha256: str) -> _FakeBundle:
        assert document_preview_root.resolve() == root.resolve()
        assert manifest_sha256 == mobile.manifest_sha256
        return mobile

    def load_offline(*, document_preview_root: Path, manifest_sha256: str) -> _FakeBundle:
        assert document_preview_root.resolve() == root.resolve()
        assert manifest_sha256 == offline.manifest_sha256
        return offline

    monkeypatch.setattr(
        server_bundle, "load_verified_raster_text_direct_mobile_paddle_bundle_v1", load_mobile
    )
    monkeypatch.setattr(server_bundle, "load_verified_raster_offline_bundle_release", load_offline)
    return _Sources(root=root, mobile=mobile, offline=offline, server_yaml=yml)


def test_character_dictionary_parser_preserves_quoted_punctuation_and_apostrophe() -> None:
    values = ["!", '"', "#", "'", "-", ":", "\\", "🇩🇪"]

    parsed = parse_ppocrv5_server_rec_character_dict_v1(_server_yaml(values))

    assert parsed == tuple(values)


def test_character_dictionary_parser_rejects_more_than_server_upper_bound() -> None:
    raw = _server_yaml([f"glyph-{index}" for index in range(18_384)])

    with pytest.raises(RasterTextServerPaddleBundleError) as raised:
        parse_ppocrv5_server_rec_character_dict_v1(raw)

    assert raised.value.code == "server_paddle_bundle_character_dictionary_invalid"


def test_materializer_compiles_exact_server_config_and_is_idempotent(
    fake_sources: _Sources,
) -> None:
    first = materialize_verified_raster_text_direct_server_paddle_bundle_v1(
        mobile_bundle=fake_sources.mobile,  # type: ignore[arg-type]
        offline_bundle=fake_sources.offline,  # type: ignore[arg-type]
    )
    second = materialize_verified_raster_text_direct_server_paddle_bundle_v1(
        document_preview_root=fake_sources.root,
        mobile_bundle_manifest_sha256=fake_sources.mobile.manifest_sha256,
        offline_bundle_manifest_sha256=fake_sources.offline.manifest_sha256,
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert os.path.samefile(first.bundle_root, second.bundle_root)
    assert first.manifest.source_recognizer_character_count == 18_383
    assert first.manifest.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert first.manifest.text_recognition_model_name == "PP-OCRv5_server_rec"
    model_files = sorted(
        path.relative_to(first.bundle_root / "models").as_posix()
        for path in (first.bundle_root / "models").rglob("*")
        if path.is_file()
    )
    assert model_files == [
        "text-det/config.json",
        "text-det/inference.json",
        "text-det/inference.pdiparams",
        "text-det/inference.yml",
        "text-rec/config.json",
        "text-rec/inference.json",
        "text-rec/inference.pdiparams",
        "text-rec/inference.yml",
    ]
    config_raw = (first.bundle_root / "models" / "text-rec" / "config.json").read_bytes()
    config = json.loads(config_raw)
    assert config_raw == server_bundle._canonical_json(config)
    assert config["Global"] == {"model_name": "PP-OCRv5_server_rec"}
    assert len(config["PostProcess"]["character_dict"]) == 18_383
    assert (
        hashlib.sha256(config_raw).hexdigest()
        == first.manifest.generated_recognizer_configuration_sha256
    )
    assert not list(fake_sources.root.glob(".raster-text-direct-server-rec-bundles-v1.staging-*"))


def test_materializer_rejects_source_drift_before_reusing_release(
    fake_sources: _Sources,
) -> None:
    materialize_verified_raster_text_direct_server_paddle_bundle_v1(
        mobile_bundle=fake_sources.mobile,  # type: ignore[arg-type]
        offline_bundle=fake_sources.offline,  # type: ignore[arg-type]
    )
    fake_sources.server_yaml.write_bytes(fake_sources.server_yaml.read_bytes() + b"# drift\n")

    with pytest.raises(RasterTextServerPaddleBundleError) as raised:
        materialize_verified_raster_text_direct_server_paddle_bundle_v1(
            mobile_bundle=fake_sources.mobile,  # type: ignore[arg-type]
            offline_bundle=fake_sources.offline,  # type: ignore[arg-type]
        )

    assert raised.value.code == "server_paddle_bundle_source_drift"


def test_loader_rejects_generated_config_drift(fake_sources: _Sources) -> None:
    bundle = materialize_verified_raster_text_direct_server_paddle_bundle_v1(
        mobile_bundle=fake_sources.mobile,  # type: ignore[arg-type]
        offline_bundle=fake_sources.offline,  # type: ignore[arg-type]
    )
    (bundle.bundle_root / "models" / "text-rec" / "config.json").write_bytes(b"{}")

    with pytest.raises(RasterTextServerPaddleBundleError) as raised:
        load_verified_raster_text_direct_server_paddle_bundle_v1(
            document_preview_root=fake_sources.root,
            manifest_sha256=bundle.manifest_sha256,
        )

    assert raised.value.code == "server_paddle_bundle_manifest_invalid"


def test_loader_rejects_extra_file(fake_sources: _Sources) -> None:
    bundle = materialize_verified_raster_text_direct_server_paddle_bundle_v1(
        mobile_bundle=fake_sources.mobile,  # type: ignore[arg-type]
        offline_bundle=fake_sources.offline,  # type: ignore[arg-type]
    )
    (bundle.bundle_root / "unexpected.bin").write_bytes(b"extra")

    with pytest.raises(RasterTextServerPaddleBundleError) as raised:
        load_verified_raster_text_direct_server_paddle_bundle_v1(
            document_preview_root=fake_sources.root,
            manifest_sha256=bundle.manifest_sha256,
        )

    assert raised.value.code == "server_paddle_bundle_manifest_invalid"


def test_loader_rejects_symlink(fake_sources: _Sources) -> None:
    bundle = materialize_verified_raster_text_direct_server_paddle_bundle_v1(
        mobile_bundle=fake_sources.mobile,  # type: ignore[arg-type]
        offline_bundle=fake_sources.offline,  # type: ignore[arg-type]
    )
    source = bundle.bundle_root / "models" / "text-det" / "config.json"
    replacement = bundle.bundle_root / "models" / "text-det" / "linked-config.json"
    try:
        os.symlink(source, replacement)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")

    with pytest.raises(RasterTextServerPaddleBundleError) as raised:
        load_verified_raster_text_direct_server_paddle_bundle_v1(
            document_preview_root=fake_sources.root,
            manifest_sha256=bundle.manifest_sha256,
        )

    assert raised.value.code == "server_paddle_bundle_manifest_invalid"
