from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pytest

import customs_ai_agent.evaluation.raster_text_direct_paddle_bundle as direct_bundle


@pytest.mark.skipif(os.name != "nt", reason="the offline sidecar bundle is Windows-only")
def test_direct_sidecar_environment_redirects_helper_bytecode(tmp_path: Path) -> None:
    temporary_home = tmp_path / "runtime-home"
    temporary_home.mkdir()

    environment = direct_bundle._direct_sidecar_environment(  # noqa: SLF001
        interpreter=tmp_path / "python.exe",
        configuration=tmp_path / "profile.json",
        configuration_sha256="a" * 64,
        models=tmp_path / "models",
        package_inventory=tmp_path / "packages.json",
        package_inventory_sha256="b" * 64,
        model_inventory=tmp_path / "models.json",
        model_inventory_sha256="c" * 64,
        sidecar_sha256="d" * 64,
        temporary_home=temporary_home,
    )

    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPYCACHEPREFIX"] == str(temporary_home.resolve() / "pycache")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _record(root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
    }


def _make_e3_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    work = root / "source-work"
    module_root = work / "sidecar" / "module-root"
    _write(
        work / direct_bundle._SOURCE_PARENT_INVENTORY,
        _canonical(
            {
                "packages": [
                    {"name": "pypdfium2", "version": importlib.metadata.version("pypdfium2")},
                    {"name": "Pillow", "version": importlib.metadata.version("Pillow")},
                ]
            }
        ),
    )
    model_rows: list[dict[str, object]] = []
    for index, relative in enumerate(direct_bundle._DIRECT_MODEL_PATHS, start=1):
        raw = f"model-{index}".encode("ascii")
        _write(work / direct_bundle._SOURCE_DIRECT_MODEL_ROOT / relative, raw)
        model_rows.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_size": len(raw),
            }
        )
    snapshot = hashlib.sha256(_canonical({"files": model_rows})).hexdigest()
    _write(
        work / direct_bundle._SOURCE_DIRECT_MODEL_INVENTORY,
        _canonical(
            {
                "schema_version": "1.0",
                "inventory_kind": "raster-text-direct-models-v1",
                "source_bundle_manifest_sha256": "a" * 64,
                "source_model_root_relative_path": "models",
                "source_model_inventory_sha256": "b" * 64,
                "source_snapshot_sha256": snapshot,
                "artifacts": model_rows,
            }
        ),
    )
    _write(work / direct_bundle._SOURCE_INTERPRETER, b"fake-python")
    for name in direct_bundle._DIRECT_MODULE_DIRECTORIES:
        _write(module_root / name / "__init__.py", f"# {name}\n".encode("ascii"))
    for name in direct_bundle._DIRECT_MODULE_FILES:
        _write(module_root / name, f"# {name}\n".encode("ascii"))
    for package, version, directory in direct_bundle._DIRECT_DISTRIBUTIONS:
        _write(
            module_root / directory / "METADATA",
            f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n".encode("utf-8"),
        )
    files = [_record(work, path) for path in sorted(work.rglob("*")) if path.is_file()]
    source_manifest = {
        "schema_version": "1.0",
        "bundle_id": "raster-offline-bundle-v1",
        "extractor_revision": "fake-e3",
        "raster_dpi": 216,
        "sidecar_timeout_seconds": 420,
        "parent_renderer_package_inventory_relative_path": direct_bundle._SOURCE_PARENT_INVENTORY,
        "raster_text_model_inventory_relative_path": direct_bundle._SOURCE_DIRECT_MODEL_INVENTORY,
        "raster_text_local_model_root_relative_path": direct_bundle._SOURCE_DIRECT_MODEL_ROOT,
        "sidecar_interpreter_root_relative_path": direct_bundle._SOURCE_INTERPRETER_ROOT,
        "sidecar_interpreter_relative_path": direct_bundle._SOURCE_INTERPRETER,
        "sidecar_module_root_relative_path": direct_bundle._SOURCE_MODULE_ROOT,
        "files": sorted(files, key=lambda item: str(item["path"])),
    }
    raw = _canonical(source_manifest)
    source_sha = hashlib.sha256(raw).hexdigest()
    _write(work / "manifest.json", raw)
    release = root / direct_bundle._BASE_RELEASES / source_sha
    release.parent.mkdir(parents=True, exist_ok=True)
    work.rename(release)
    monkeypatch.setattr(direct_bundle, "_E3_DIRECT_BASE_MANIFEST_SHA256", source_sha)
    return root


def _without_real_smoke(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def smoke(bundle: direct_bundle.VerifiedRasterTextDirectPaddleBundleV1) -> None:
        calls.append(bundle.manifest_sha256)

    monkeypatch.setattr(direct_bundle, "_smoke_direct_bundle", smoke)
    return calls


def _windows_os_error(code: int) -> OSError:
    error = OSError("simulated Windows publication failure")
    setattr(error, "winerror", code)
    return error


def test_parent_inventory_ignores_a_sidecar_shadowed_distribution_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = tmp_path / "parent-renderer.json"
    _write(
        inventory,
        _canonical(
            {
                "packages": [
                    {
                        "name": "pypdfium2",
                        "version": importlib.metadata.version("pypdfium2"),
                    },
                    {
                        "name": "Pillow",
                        "version": importlib.metadata.version("Pillow"),
                    },
                ]
            }
        ),
    )
    record = direct_bundle.DirectPaddleBundleFileV1.model_validate(
        _record(tmp_path, inventory)
    )
    monkeypatch.setattr(
        direct_bundle.importlib.metadata,
        "version",
        lambda _name: "sidecar-shadowed-version",
    )

    assert direct_bundle._verify_parent_inventory(
        inventory,
        expected_record=record,
    ) == record.sha256


def _source_root(root: Path) -> Path:
    return direct_bundle._safe_path(
        root, direct_bundle._source_bundle_relative_path(), directory=True
    )


def test_materializes_isolated_direct_bundle_and_reuses_matching_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    smoke_calls = _without_real_smoke(monkeypatch)

    first = direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root
    )
    second = direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert smoke_calls == [first.manifest_sha256]
    assert (first.bundle_root / "sidecar/module-root/paddle").is_dir()
    assert (first.bundle_root / "sidecar/module-root/paddleocr").exists() is False
    assert (first.bundle_root / "sidecar/module-root/paddlex").exists() is False
    assert (first.bundle_root / "models/text-det/inference.pdiparams").is_file()
    loaded = direct_bundle.load_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root, manifest_sha256=first.manifest_sha256
    )
    assert loaded.file_sha256("sidecar/interpreter/python.exe") == first.file_sha256(
        "sidecar/interpreter/python.exe"
    )
    assert loaded.manifest.sidecar_timeout_seconds == 30


def test_prepares_hash_bound_v2_runtime_profile_from_verified_microbundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    bundle = direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root
    )

    prepared = direct_bundle.prepare_raster_text_direct_paddle_runtime_profile_v2(
        bundle,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.42,
    )
    repeated = direct_bundle.prepare_raster_text_direct_paddle_runtime_profile_v2(
        bundle,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.42,
    )

    assert prepared == repeated
    assert prepared.runtime_configuration_relative_path == (
        f"raster-text-runtime-profiles-v1/"
        f"{prepared.raster_text_profile_revision.rsplit(':', 1)[1]}/profile.json"
    )
    raw = (root / prepared.runtime_configuration_relative_path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prepared.runtime_configuration_sha256
    from customs_ai_agent.service.raster_text_preview_runtime import RasterTextOfflineProfileV2

    profile = RasterTextOfflineProfileV2.model_validate_json(raw)
    assert profile.schema_version == "2.0"
    assert profile.engine_kind == "direct_paddle_text"
    assert profile.runtime_api == "paddle-inference-static-v1"
    assert profile.text_detection_limit_side_len == 960
    assert profile.cpu_threads == 6
    assert profile.minimum_page_confidence == 0.42
    assert profile.sidecar_required_packages == [
        "numpy",
        "opencv-contrib-python",
        "paddlepaddle",
        "pyclipper",
    ]


def test_base_rejects_direct_model_drift_before_stage_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    source = _source_root(root)
    (source / "models/raster-text/text-det/inference.json").write_bytes(b"drift")

    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_source_model_invalid"
    assert not (root / "raster-text-direct-bundles-v1/releases").exists()


def test_base_rejects_undeclared_allowlisted_package_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    source = _source_root(root)
    _write(source / "sidecar/module-root/paddle/injected.py", b"not-in-manifest")

    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_source_closure_invalid"


def test_smoke_failure_does_not_publish_release_or_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)

    def fail_smoke(_bundle: direct_bundle.VerifiedRasterTextDirectPaddleBundleV1) -> None:
        raise direct_bundle.RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed")

    monkeypatch.setattr(direct_bundle, "_smoke_direct_bundle", fail_smoke)
    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_smoke_failed"
    assert list((root / "raster-text-direct-bundles-v1/releases").iterdir()) == []
    assert list((root / "raster-text-direct-bundles-v1/intents").iterdir()) == []
    assert not list(root.glob(".raster-text-direct-bundles-v1.staging-*"))


def test_post_smoke_rehash_rejects_an_undeclared_bytecode_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)

    def contaminate_smoke(bundle: direct_bundle.VerifiedRasterTextDirectPaddleBundleV1) -> None:
        _write(
            bundle.bundle_root
            / "sidecar/module-root/paddle/__pycache__/generated.cpython-311.pyc",
            b"unexpected-bytecode-cache",
        )

    monkeypatch.setattr(direct_bundle, "_smoke_direct_bundle", contaminate_smoke)
    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_manifest_invalid"
    assert list((root / "raster-text-direct-bundles-v1/releases").iterdir()) == []
    assert list((root / "raster-text-direct-bundles-v1/intents").iterdir()) == []
    assert not list(root.glob(".raster-text-direct-bundles-v1.staging-*"))


def test_atomic_publish_failure_keeps_release_and_intent_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)

    def fail_rename(_source: object, _destination: object) -> None:
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(direct_bundle.os, "rename", fail_rename)
    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_publish_failed"
    assert list((root / "raster-text-direct-bundles-v1/releases").iterdir()) == []
    assert list((root / "raster-text-direct-bundles-v1/intents").iterdir()) == []
    assert not list(root.glob(".raster-text-direct-bundles-v1.staging-*"))


def test_atomic_publish_retries_transient_windows_lock_after_reverifying_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    original_rename = direct_bundle.os.rename
    original_load = direct_bundle._load_verified_bundle_from_root
    rename_attempts: list[tuple[Path, Path]] = []
    staging_verifications = 0
    delays: list[float] = []

    def transient_once(source: Path, destination: Path) -> None:
        rename_attempts.append((source, destination))
        if len(rename_attempts) == 1:
            raise _windows_os_error(5)
        original_rename(source, destination)

    def track_staging_verification(
        *, root: Path, bundle_root: Path, expected_manifest_sha256: str | None, allow_staging: bool
    ) -> direct_bundle.VerifiedRasterTextDirectPaddleBundleV1:
        nonlocal staging_verifications
        if allow_staging:
            staging_verifications += 1
        return original_load(
            root=root,
            bundle_root=bundle_root,
            expected_manifest_sha256=expected_manifest_sha256,
            allow_staging=allow_staging,
        )

    monkeypatch.setattr(direct_bundle.os, "rename", transient_once)
    monkeypatch.setattr(
        direct_bundle, "_load_verified_bundle_from_root", track_staging_verification
    )
    monkeypatch.setattr(direct_bundle.time, "sleep", lambda seconds: delays.append(seconds))

    bundle = direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root
    )

    assert len(rename_attempts) == 2
    assert delays == [direct_bundle._PUBLISH_TRANSIENT_RETRY_DELAYS_SECONDS[0]]
    # Initial build validation, post-smoke validation, then the retry guard.
    assert staging_verifications == 3
    assert bundle.bundle_root.is_dir()


def test_atomic_publish_exhausts_all_bounded_transient_windows_lock_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    attempts = 0
    delays: list[float] = []

    def always_locked(_source: object, _destination: object) -> None:
        nonlocal attempts
        attempts += 1
        raise _windows_os_error(32)

    monkeypatch.setattr(direct_bundle.os, "rename", always_locked)
    monkeypatch.setattr(direct_bundle.time, "sleep", lambda seconds: delays.append(seconds))

    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_publish_failed"
    assert attempts == len(direct_bundle._PUBLISH_TRANSIENT_RETRY_DELAYS_SECONDS) + 1
    assert delays == list(direct_bundle._PUBLISH_TRANSIENT_RETRY_DELAYS_SECONDS)
    assert list((root / "raster-text-direct-bundles-v1/releases").iterdir()) == []
    assert list((root / "raster-text-direct-bundles-v1/intents").iterdir()) == []


def test_atomic_publish_retry_rejects_staged_material_drift_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    attempts: list[tuple[Path, Path]] = []

    def transient_once(source: Path, destination: Path) -> None:
        attempts.append((source, destination))
        raise _windows_os_error(32)

    def drift_staging_material(_seconds: float) -> None:
        assert attempts
        (attempts[0][0] / "models/text-det/inference.json").write_bytes(b"drift")

    monkeypatch.setattr(direct_bundle.os, "rename", transient_once)
    monkeypatch.setattr(direct_bundle.time, "sleep", drift_staging_material)

    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_manifest_invalid"
    assert len(attempts) == 1
    assert list((root / "raster-text-direct-bundles-v1/releases").iterdir()) == []
    assert list((root / "raster-text-direct-bundles-v1/intents").iterdir()) == []


def test_atomic_publish_does_not_retry_unsupported_windows_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    attempts = 0
    delays: list[float] = []

    def unsupported_error(_source: object, _destination: object) -> None:
        nonlocal attempts
        attempts += 1
        raise _windows_os_error(183)

    monkeypatch.setattr(direct_bundle.os, "rename", unsupported_error)
    monkeypatch.setattr(direct_bundle.time, "sleep", lambda seconds: delays.append(seconds))

    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_publish_failed"
    assert attempts == 1
    assert delays == []


def test_atomic_publish_retry_stops_on_destination_collision_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    attempts: list[tuple[Path, Path]] = []

    def transient_once(source: Path, destination: Path) -> None:
        attempts.append((source, destination))
        raise _windows_os_error(5)

    def create_destination_collision(_seconds: float) -> None:
        assert attempts
        attempts[0][1].mkdir()

    monkeypatch.setattr(direct_bundle.os, "rename", transient_once)
    monkeypatch.setattr(direct_bundle.time, "sleep", create_destination_collision)

    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root
        )

    assert raised.value.code == "direct_paddle_bundle_publish_conflict"
    assert len(attempts) == 1
    destination = attempts[0][1]
    assert destination.is_dir()
    assert not (destination / "manifest.json").exists()
    assert list((root / "raster-text-direct-bundles-v1/intents").iterdir()) == []


def test_atomic_publish_renames_canonical_stage_and_release_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    calls: list[tuple[Path, Path]] = []
    original_rename = direct_bundle.os.rename

    def capture_rename(source: Path, destination: Path) -> None:
        calls.append((source, destination))
        original_rename(source, destination)

    monkeypatch.setattr(direct_bundle.os, "rename", capture_rename)
    bundle = direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root
    )

    assert len(calls) == 1
    stage, destination = calls[0]
    assert stage.parent == root
    assert stage.name.startswith(".raster-text-direct-bundles-v1.staging-")
    assert destination == root / direct_bundle._DIRECT_RELEASES / bundle.manifest_sha256
    assert destination.parent == root / direct_bundle._DIRECT_RELEASES


def test_output_tamper_fails_closed_on_explicit_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_e3_source(tmp_path, monkeypatch)
    _without_real_smoke(monkeypatch)
    bundle = direct_bundle.materialize_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root
    )
    (bundle.bundle_root / "models/text-rec/inference.json").write_bytes(b"tampered")

    with pytest.raises(direct_bundle.RasterTextDirectPaddleBundleError) as raised:
        direct_bundle.load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=root, manifest_sha256=bundle.manifest_sha256
        )

    assert raised.value.code == "direct_paddle_bundle_manifest_invalid"
