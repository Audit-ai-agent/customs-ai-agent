from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
import runpy
import shutil
from dataclasses import replace
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

import customs_ai_agent.evaluation.synthetic_raster_auto_activation as auto_activation
from customs_ai_agent.evaluation.synthetic_raster_auto_activation import (
    _CandidateArtifacts,
    _evaluation_relative_path,
    SyntheticRasterAutoActivationError,
    auto_activate_synthetic_raster_v1,
)
from customs_ai_agent.evaluation.raster_offline_bundle import (
    RasterOfflineBundleError,
    VerifiedRasterOfflineBundleV1,
    VerifiedRasterTextReleaseReferenceV1,
    load_verified_legacy_raster_offline_bundle_for_safe_configuration_migration,
    load_verified_raster_offline_bundle,
    prepare_raster_runtime_profile_v2,
    prepare_raster_text_runtime_profile_v1,
    verify_raster_text_direct_model_material,
)
from customs_ai_agent.evaluation.raster_offline_bundle_bootstrap import (
    RasterOfflineBundleBootstrapError,
    _resolved_comparison_path,
    ensure_raster_offline_bundle_release_v1,
    materialize_verified_unselected_raster_text_release_v1,
    resolve_or_materialize_verified_raster_text_release_reference_v1,
)
from customs_ai_agent.service.ppstructurev3_sidecar import __file__ as sidecar_module_file
from customs_ai_agent.service.raster_document_preview_runtime import (
    _controlled_module_root_tree_sha256,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _write_minimal_bundle(
    root: Path, *, include_safe_engine_detector_configuration: bool = True
) -> None:
    bundle = root / "raster-offline-bundle-v1"
    models = bundle / "models"
    model_directories = (
        "layout",
        "text-det",
        "text-rec",
        "table-classifier",
        "wired-structure",
        "wireless-structure",
        "wired-cells",
        "wireless-cells",
    )
    direct_model_files = {
        "text-det": {
            "inference.json": b"direct-detector-graph",
            "inference.pdiparams": b"direct-detector-parameters",
            "inference.yml": b"direct-detector-metadata",
        },
        "text-rec": {
            "config.json": b"direct-recognizer-config",
            "inference.json": b"direct-recognizer-graph",
            "inference.pdiparams": b"direct-recognizer-parameters",
            "inference.yml": b"direct-recognizer-metadata",
        },
    }
    for directory in model_directories:
        target = models / directory
        target.mkdir(parents=True, exist_ok=True)
        for filename, payload in direct_model_files.get(
            directory, {"weights.bin": directory.encode("ascii")}
        ).items():
            (target / filename).write_bytes(payload)
    interpreter = bundle / "sidecar" / "interpreter" / "python.exe"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_bytes(b"isolated bundle interpreter placeholder")
    module_root = bundle / "sidecar" / "module-root"
    module_path = module_root / "customs_ai_agent" / "service" / "ppstructurev3_sidecar.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    (module_root / "customs_ai_agent" / "__init__.py").write_bytes(b"# bundle package\n")
    (module_root / "customs_ai_agent" / "service" / "__init__.py").write_bytes(
        b"# bundle service package\n"
    )
    # A real target-installed Python closure includes empty marker files such
    # as ``py.typed``.  They must stay part of the exhaustive hash binding.
    (module_root / "customs_ai_agent" / "py.typed").write_bytes(b"")
    module_path.write_bytes(Path(sidecar_module_file).read_bytes())
    sidecar_configuration: dict[str, object] = {
        "schema_version": "1.0",
        "runtime_api": "paddleocr3-ppstructurev3",
        "extractor_revision": "ppstructurev3-local-v1-paddleocr-3.7.0",
        "raster_profile_revision": "template",
        "layout_detection_model_name": "layout",
        "layout_detection_model_directory": "layout",
        "text_detection_model_name": "det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": "rec",
        "text_recognition_model_directory": "text-rec",
        "table_classification_model_name": "classifier",
        "table_classification_model_directory": "table-classifier",
        "wired_table_structure_recognition_model_name": "wired-structure",
        "wired_table_structure_recognition_model_directory": "wired-structure",
        "wireless_table_structure_recognition_model_name": "wireless-structure",
        "wireless_table_structure_recognition_model_directory": "wireless-structure",
        "wired_table_cells_detection_model_name": "wired-cells",
        "wired_table_cells_detection_model_directory": "wired-cells",
        "wireless_table_cells_detection_model_name": "wireless-cells",
        "wireless_table_cells_detection_model_directory": "wireless-cells",
        # The fixed legacy fixture is explicitly CPU-1. The current source
        # derives a separate CPU-6 release from it without changing these
        # legacy bytes.
        "cpu_threads": 1,
    }
    if include_safe_engine_detector_configuration:
        sidecar_configuration.update(
            {
                "inference_engine": "paddle_static",
                "paddle_static_run_mode": "paddle",
                "paddle_static_enable_new_ir": False,
                "paddle_static_enable_cinn": False,
                "text_detection_limit_side_len": 960,
                "text_detection_limit_type": "max",
                "text_detection_threshold": 0.3,
                "text_detection_box_threshold": 0.6,
                "text_detection_unclip_ratio": 1.5,
            }
        )
    (bundle / "sidecar" / "configuration.json").write_bytes(_canonical(sidecar_configuration))
    (bundle / "inventories").mkdir(parents=True, exist_ok=True)
    (bundle / "inventories" / "parent-renderer.json").write_bytes(
        _canonical(
            {
                "packages": [
                    {"name": "Pillow", "version": "10.4.0"},
                    {"name": "pypdfium2", "version": "4.30.0"},
                ]
            }
        )
    )
    (bundle / "inventories" / "sidecar-packages.json").write_bytes(
        _canonical(
            {
                "packages": [
                    {"name": "paddleocr", "version": "3.0.0"},
                    {"name": "paddlepaddle", "version": "3.0.0"},
                    {"name": "paddlex", "version": "3.0.0"},
                ]
            }
        )
    )
    (bundle / "provenance").mkdir(parents=True, exist_ok=True)
    (bundle / "provenance" / "bootstrap-receipt.json").write_bytes(
        _canonical(
            {
                "schema_version": "1.0",
                "sidecar_lock_sha256": hashlib.sha256(
                    (
                        Path(__file__).parents[1]
                        / "requirements"
                        / "raster-sidecar-win-amd64-cp311.lock"
                    ).read_bytes()
                ).hexdigest(),
            }
        )
    )
    artifacts = []
    for file in sorted(models.rglob("*")):
        if file.is_file():
            raw = file.read_bytes()
            artifacts.append(
                {
                    "relative_path": file.relative_to(models).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    (bundle / "inventories" / "models.json").write_bytes(_canonical({"artifacts": artifacts}))
    files = []
    for file in sorted(bundle.rglob("*")):
        if file.is_file():
            raw = file.read_bytes()
            files.append(
                {
                    "path": file.relative_to(bundle).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_size": len(raw),
                }
            )
    manifest = {
        "schema_version": "1.0",
        "bundle_id": "raster-offline-bundle-v1",
        "extractor_revision": "ppstructurev3-local-v1-paddleocr-3.7.0",
        "raster_dpi": 216,
        "sidecar_timeout_seconds": 420,
        "parent_renderer_package_inventory_relative_path": "inventories/parent-renderer.json",
        "parent_renderer_required_packages": ["pypdfium2", "Pillow"],
        "sidecar_package_inventory_relative_path": "inventories/sidecar-packages.json",
        "sidecar_required_packages": ["paddleocr", "paddlepaddle", "paddlex"],
        "sidecar_command_relative_path": (
            "sidecar/module-root/customs_ai_agent/service/ppstructurev3_sidecar.py"
        ),
        "sidecar_interpreter_root_relative_path": "sidecar/interpreter",
        "sidecar_interpreter_relative_path": "sidecar/interpreter/python.exe",
        "sidecar_module_root_relative_path": "sidecar/module-root",
        "sidecar_module_relative_path": (
            "sidecar/module-root/customs_ai_agent/service/ppstructurev3_sidecar.py"
        ),
        "sidecar_configuration_template_relative_path": "sidecar/configuration.json",
        "model_inventory_relative_path": "inventories/models.json",
        "local_model_root_relative_path": "models",
        "files": files,
    }
    (bundle / "manifest.json").write_bytes(_canonical(manifest))


def _publish_minimal_release_selector(root: Path) -> tuple[str, str]:
    """Copy the test legacy bundle into its canonical selected release path."""

    legacy = root / "raster-offline-bundle-v1"
    manifest_sha256 = hashlib.sha256((legacy / "manifest.json").read_bytes()).hexdigest()
    release_relative_path = f"raster-offline-bundles-v1/releases/{manifest_sha256}"
    release = root.joinpath(*release_relative_path.split("/"))
    release.parent.mkdir(parents=True)
    # Match the production publisher: create a sibling stage first, then do
    # a same-volume rename into the content-addressed final location.
    stage = root / ".release-stage"
    shutil.copytree(legacy, stage)
    os.rename(stage, release)
    selector = {
        "schema_version": "1.0",
        "bundle_id": "raster-offline-bundle-v1",
        "release_manifest_sha256": manifest_sha256,
        "release_relative_path": release_relative_path,
    }
    (root / "raster-offline-bundles-v1" / "current.json").write_bytes(_canonical(selector))
    return release_relative_path, manifest_sha256


def _materialize_minimal_raster_text_release(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[VerifiedRasterOfflineBundleV1, bytes]:
    """Build the test-only direct release while keeping the selector frozen."""

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: None)
    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    selector = root / "raster-offline-bundles-v1" / "current.json"
    selector_before = selector.read_bytes()
    candidate = materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)
    assert candidate.manifest_sha256 != selected.manifest_sha256
    assert selector.read_bytes() == selector_before
    return candidate, selector_before


def test_missing_offline_bundle_never_downloads_or_replaces_existing_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    pointer = root / "approvals" / "raster-preview-active.json"
    pointer.parent.mkdir(parents=True)
    original = b'{"legacy":"pointer"}'
    pointer.write_bytes(original)

    calls: list[str] = []

    def _missing_bundle(**_kwargs: object) -> None:
        calls.append("release")
        raise RasterOfflineBundleBootstrapError("offline_bundle_unavailable")

    monkeypatch.setattr(auto_activation, "ensure_raster_offline_bundle_release_v1", _missing_bundle)
    with pytest.raises(SyntheticRasterAutoActivationError, match="offline_bundle_unavailable"):
        auto_activate_synthetic_raster_v1(document_preview_root=root)

    assert calls == ["release"]
    assert pointer.read_bytes() == original


def test_auto_activation_denies_parent_network_before_bundle_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    observed: list[str] = []

    def _attempt_network(**_kwargs: object) -> None:
        with socket.socket() as client:
            with pytest.raises(OSError, match="network access denied"):
                client.connect(("127.0.0.1", 9))
        observed.append("denied")
        raise RasterOfflineBundleBootstrapError("offline_bundle_unavailable")

    monkeypatch.setattr(
        auto_activation, "ensure_raster_offline_bundle_release_v1", _attempt_network
    )

    with pytest.raises(SyntheticRasterAutoActivationError, match="offline_bundle_unavailable"):
        auto_activate_synthetic_raster_v1(document_preview_root=root)

    assert observed == ["denied"]


def test_activation_lock_is_reusable_without_removing_an_inert_marker(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    # The file is an inert target for an OS-owned lock, not a stale ownership
    # marker.  A later automatic retry can acquire it without a person
    # deleting anything.
    lock = root / ".synthetic-raster-v1-auto-activation.lock"
    with auto_activation._activation_lock(root):
        pass
    assert lock.read_bytes() == b"\0"
    with auto_activation._activation_lock(root):
        pass


def test_activation_cli_does_not_expose_human_or_evidence_override_flags() -> None:
    parser = runpy.run_path(
        str(Path(__file__).parents[1] / "tools" / "auto_activate_synthetic_raster_v1.py")
    )["_parser"]()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {"-h", "--help", "--document-preview-root"}


def test_auto_activation_loads_checked_in_tooling_from_a_direct_script_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout_root = Path(auto_activation.__file__).resolve().parents[3]
    for module_name in (
        "tools.evaluate_synthetic_raster",
        "tools.run_ppstructurev3_synthetic_raster_candidates",
        "tools",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [item for item in sys.path if Path(item or ".").resolve() != checkout_root],
    )

    evaluator, runner = auto_activation._load_checked_in_synthetic_raster_tooling()

    assert (
        Path(evaluator.__file__).resolve()
        == checkout_root / "tools" / "evaluate_synthetic_raster.py"
    )
    assert (
        Path(runner.__file__).resolve()
        == checkout_root / "tools" / "run_ppstructurev3_synthetic_raster_candidates.py"
    )
    assert sys.path[0] == str(checkout_root)


def test_evaluation_paths_are_content_addressed_by_candidate_and_evaluator() -> None:
    candidate = _CandidateArtifacts(
        bundle_relative_path="bundles/calibration.json",
        bundle_sha256="b" * 64,
        execution_relative_path="executions/calibration.json",
        execution_sha256="c" * 64,
    )
    calibration = _evaluation_relative_path(
        raster_profile_revision="synthetic-raster-v1:" + "a" * 24,
        runtime_configuration_sha256="d" * 64,
        split="calibration",
        candidate=candidate,
        evaluator_implementation_sha256="e" * 64,
    )
    sealed = _evaluation_relative_path(
        raster_profile_revision="synthetic-raster-v1:" + "a" * 24,
        runtime_configuration_sha256="d" * 64,
        split="sealed",
        candidate=candidate,
        evaluator_implementation_sha256="e" * 64,
        calibration_receipt_sha256="f" * 64,
    )
    assert calibration.startswith(
        "synthetic-raster-v1/activation-evaluations/" + "a" * 24 + "/calibration/"
    )
    assert sealed.startswith("synthetic-raster-v1/activation-evaluations/" + "a" * 24 + "/sealed/")
    assert calibration != sealed
    assert calibration != _evaluation_relative_path(
        raster_profile_revision="synthetic-raster-v1:" + "a" * 24,
        runtime_configuration_sha256="d" * 64,
        split="calibration",
        candidate=_CandidateArtifacts(
            bundle_relative_path="bundles/calibration-next.json",
            bundle_sha256="1" * 64,
            execution_relative_path="executions/calibration.json",
            execution_sha256="c" * 64,
        ),
        evaluator_implementation_sha256="e" * 64,
    )


def test_verified_bundle_generates_one_immutable_v2_profile_without_manual_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    release_relative_path, manifest_sha256 = _publish_minimal_release_selector(root)

    bundle = load_verified_raster_offline_bundle(root)
    profile = prepare_raster_runtime_profile_v2(bundle, pdfium_version="4.30.0")
    runtime_path = root / profile.runtime_configuration_relative_path
    sidecar_path = root / profile.sidecar_configuration_relative_path

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert runtime["schema_version"] == "2.0"
    assert runtime["raster_profile_revision"] == profile.raster_profile_revision
    assert runtime["bundle_root_relative_path"] == release_relative_path
    assert runtime["bundle_manifest_sha256"] == bundle.manifest_sha256
    assert runtime["bundle_manifest_sha256"] == manifest_sha256
    assert runtime["sidecar_interpreter_path"] == (
        f"{release_relative_path}/sidecar/interpreter/python.exe"
    )
    assert runtime["sidecar_interpreter_root"] == f"{release_relative_path}/sidecar/interpreter"
    assert (
        runtime["sidecar_interpreter_sha256"]
        == hashlib.sha256(
            (bundle.bundle_root / "sidecar/interpreter/python.exe").read_bytes()
        ).hexdigest()
    )
    assert runtime["sidecar_module_root"] == f"{release_relative_path}/sidecar/module-root"
    assert runtime["sidecar_module_path"].startswith(runtime["sidecar_module_root"] + "/")
    assert runtime["sidecar_module_root_tree_sha256"] == _controlled_module_root_tree_sha256(
        bundle.bundle_root / "sidecar" / "module-root"
    )
    assert runtime["sidecar_command_path"] == runtime["sidecar_module_path"]
    assert runtime["sidecar_command_sha256"] == runtime["sidecar_module_sha256"]
    assert (
        runtime["sidecar_module_sha256"]
        == hashlib.sha256(
            (
                bundle.bundle_root
                / "sidecar/module-root/customs_ai_agent/service/ppstructurev3_sidecar.py"
            ).read_bytes()
        ).hexdigest()
    )
    assert runtime["sidecar_interpreter_root_tree_sha256"] == _controlled_module_root_tree_sha256(
        bundle.bundle_root / "sidecar" / "interpreter"
    )
    assert sidecar["raster_profile_revision"] == profile.raster_profile_revision
    assert profile.raster_profile_revision.startswith("synthetic-raster-v1:")
    assert profile.runtime_configuration_relative_path.startswith("raster-runtime-profiles-v1/")
    assert profile.sidecar_configuration_relative_path.startswith("raster-runtime-profiles-v1/")
    # A runtime preflight must never reserve the atomically-published dataset
    # namespace before the compiler has produced a complete receipt.
    assert not (root / "synthetic-raster-v1").exists()
    # Reusing the same bundle yields exactly the same immutable config.
    assert prepare_raster_runtime_profile_v2(bundle, pdfium_version="4.30.0") == profile


def test_new_runtime_profile_rejects_a_fixed_legacy_bundle(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)

    legacy = load_verified_raster_offline_bundle(root)

    with pytest.raises(RasterOfflineBundleError, match="offline_bundle_release_selector_invalid"):
        prepare_raster_runtime_profile_v2(legacy, pdfium_version="4.30.0")


def test_release_selector_uses_only_its_content_addressed_release(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    release_relative_path, manifest_sha256 = _publish_minimal_release_selector(root)

    selected = load_verified_raster_offline_bundle(root)

    assert selected.manifest_sha256 == manifest_sha256
    assert selected.bundle_root_relative_path == release_relative_path


def test_invalid_release_selector_never_falls_back_to_legacy_bundle(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    collection = root / "raster-offline-bundles-v1"
    collection.mkdir()
    (collection / "current.json").write_bytes(_canonical({"schema_version": "bad"}))

    with pytest.raises(RasterOfflineBundleError, match="offline_bundle_release_selector_invalid"):
        load_verified_raster_offline_bundle(root)


def test_release_selector_rejects_a_path_whose_basename_is_not_its_manifest_sha(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    release_relative_path, manifest_sha256 = _publish_minimal_release_selector(root)
    selector_path = root / "raster-offline-bundles-v1" / "current.json"
    selector_path.write_bytes(
        _canonical(
            {
                "schema_version": "1.0",
                "bundle_id": "raster-offline-bundle-v1",
                "release_manifest_sha256": manifest_sha256,
                "release_relative_path": release_relative_path.rsplit("/", 1)[0] + "/" + "b" * 64,
            }
        )
    )

    with pytest.raises(RasterOfflineBundleError, match="offline_bundle_release_selector_invalid"):
        load_verified_raster_offline_bundle(root)


def test_selector_with_missing_release_never_falls_back_to_legacy_bundle(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    digest = "a" * 64
    collection = root / "raster-offline-bundles-v1"
    collection.mkdir()
    (collection / "current.json").write_bytes(
        _canonical(
            {
                "schema_version": "1.0",
                "bundle_id": "raster-offline-bundle-v1",
                "release_manifest_sha256": digest,
                "release_relative_path": f"raster-offline-bundles-v1/releases/{digest}",
            }
        )
    )

    with pytest.raises(RasterOfflineBundleError, match="offline_bundle_release_selector_invalid"):
        load_verified_raster_offline_bundle(root)


@pytest.mark.skipif(os.name != "nt", reason="Windows extended paths are platform-specific")
def test_windows_comparison_identity_accepts_extended_copytree_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "sidecar" / "interpreter" / "Lib" / "site-packages"
    nested.mkdir(parents=True)

    extended_nested = Path("\\\\?\\" + str(nested))

    assert (
        _resolved_comparison_path(extended_nested)
        .relative_to(_resolved_comparison_path(source))
        .as_posix()
        == "sidecar/interpreter/Lib/site-packages"
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows extended paths are platform-specific")
def test_release_copy_filter_accepts_extended_callback_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    real_copytree = bootstrap.shutil.copytree

    def copytree_with_extended_callback(
        source: str | Path,
        destination: str | Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        ignore = kwargs.get("ignore", args[1] if len(args) > 1 else None)
        assert callable(ignore)
        if not args:
            resolved_source = str(Path(source).resolve(strict=True))
            extended_source = Path(
                resolved_source
                if resolved_source.startswith("\\\\?\\")
                else "\\\\?\\" + resolved_source
            )
            assert ignore(str(extended_source), ["manifest.json"]) == set()
            assert ignore(str(extended_source / "sidecar"), ["module-root"]) == set()
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(bootstrap.shutil, "copytree", copytree_with_extended_callback)
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: None)

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)

    assert selected.bundle_root_relative_path.startswith("raster-offline-bundles-v1/releases/")


def test_local_release_derivation_preserves_legacy_and_reuses_selected_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    legacy_manifest_before = (root / "raster-offline-bundle-v1" / "manifest.json").read_bytes()
    legacy_sidecar_before = (
        root
        / "raster-offline-bundle-v1"
        / "sidecar"
        / "module-root"
        / "customs_ai_agent"
        / "service"
        / "ppstructurev3_sidecar.py"
    ).read_bytes()

    # The fake bundle models deliberately use short test names.  Change only
    # an accepted, non-model template field so this test exercises the
    # source/config release path without provisioning native packages.
    legacy_template = json.loads(
        (root / "raster-offline-bundle-v1" / "sidecar" / "configuration.json").read_text(
            encoding="utf-8"
        )
    )
    desired_template = dict(legacy_template)
    desired_template["raster_profile_revision"] = "template-next"
    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    monkeypatch.setattr(bootstrap, "_configuration_template", lambda: desired_template)
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))

    first = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    second = ensure_raster_offline_bundle_release_v1(document_preview_root=root)

    assert first == second
    assert first.bundle_root_relative_path.startswith("raster-offline-bundles-v1/releases/")
    assert (
        root / "raster-offline-bundle-v1" / "manifest.json"
    ).read_bytes() == legacy_manifest_before
    assert (
        root
        / "raster-offline-bundle-v1"
        / "sidecar"
        / "module-root"
        / "customs_ai_agent"
        / "service"
        / "ppstructurev3_sidecar.py"
    ).read_bytes() == legacy_sidecar_before
    releases = list((root / "raster-offline-bundles-v1" / "releases").iterdir())
    assert [release.name for release in releases] == [first.manifest_sha256]
    assert smoke_calls == ["smoke"]


def test_text_materialization_recovers_from_clean_predecessor_without_replacing_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken selected structural release is never a text recovery source."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    # A shallow extra is enough to model the real selected-release cache
    # contamination without making this Windows test depend on deep-path
    # creation behaviour.
    selected_cache = root / selected.bundle_root_relative_path / "unmanifested-cache.pyc"
    selected_cache.write_bytes(b"unmanifested cache")
    selector_before = (root / "raster-offline-bundles-v1" / "current.json").read_bytes()

    # Ensure the recovery produces a different hash from the contaminated
    # selected release, so create-only publication cannot reuse that tree.
    recovery_template = dict(bootstrap._configuration_template())
    recovery_template["raster_profile_revision"] = "text-recovery-next"
    monkeypatch.setattr(bootstrap, "_configuration_template", lambda: recovery_template)

    recovered = materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)

    assert recovered.manifest_sha256 != selected.manifest_sha256
    assert (root / "raster-offline-bundles-v1" / "current.json").read_bytes() == selector_before
    assert selected_cache.exists()
    assert (
        "sidecar/module-root/customs_ai_agent/service/raster_text_sidecar.py"
        in recovered.file_sha256_by_relative_path
    )
    assert smoke_calls == ["smoke", "smoke"]
    with pytest.raises(RasterOfflineBundleError, match="offline_bundle_release_selector_invalid"):
        load_verified_raster_offline_bundle(root)


def test_text_materialization_binds_only_the_dedicated_seven_file_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    candidate, selector_before = _materialize_minimal_raster_text_release(root, monkeypatch)
    material = verify_raster_text_direct_model_material(candidate)
    profile = prepare_raster_text_runtime_profile_v1(
        candidate, pdfium_version="4.30.0", minimum_page_confidence=0.8
    )
    profile_payload = json.loads(
        (root / profile.runtime_configuration_relative_path).read_text(encoding="utf-8")
    )
    direct_root = candidate.bundle_root / "models" / "raster-text"
    direct_inventory = json.loads(
        (candidate.bundle_root / "inventories" / "raster-text-models.json").read_text(
            encoding="utf-8"
        )
    )

    assert candidate.manifest.raster_text_local_model_root_relative_path == "models/raster-text"
    assert candidate.manifest.raster_text_model_inventory_relative_path == (
        "inventories/raster-text-models.json"
    )
    assert profile_payload["local_model_root"] == (
        f"{candidate.bundle_root_relative_path}/models/raster-text"
    )
    assert profile_payload["model_inventory_path"] == (
        f"{candidate.bundle_root_relative_path}/inventories/raster-text-models.json"
    )
    assert profile_payload["model_inventory_sha256"] == material.model_inventory_sha256
    assert {
        path.relative_to(direct_root).as_posix()
        for path in direct_root.rglob("*")
        if path.is_file()
    } == {
        "text-det/inference.json",
        "text-det/inference.pdiparams",
        "text-det/inference.yml",
        "text-rec/config.json",
        "text-rec/inference.json",
        "text-rec/inference.pdiparams",
        "text-rec/inference.yml",
    }
    assert len(direct_inventory["artifacts"]) == 7
    assert direct_inventory["source_snapshot_sha256"] == material.source_snapshot_sha256
    derivation = json.loads(
        (candidate.bundle_root / "provenance" / "release-derivation.json").read_text(
            encoding="utf-8"
        )
    )
    assert derivation["raster_text_model_snapshot"] == direct_inventory
    assert candidate.manifest.model_inventory_relative_path == "inventories/models.json"
    assert (
        len(
            json.loads((candidate.bundle_root / "inventories" / "models.json").read_text())[
                "artifacts"
            ]
        )
        > 7
    )
    assert (root / "raster-offline-bundles-v1" / "current.json").read_bytes() == selector_before


def test_text_release_intent_changes_when_the_direct_source_snapshot_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: None)
    base = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    source = bootstrap._snapshot_raster_text_direct_model_source(base)
    changed_first = replace(source.artifacts[0], sha256="b" * 64)
    changed_artifacts = (changed_first, *source.artifacts[1:])
    changed_snapshot_sha256 = hashlib.sha256(
        _canonical(
            {
                "files": [
                    {
                        "relative_path": artifact.relative_path,
                        "sha256": artifact.sha256,
                        "byte_size": artifact.byte_size,
                    }
                    for artifact in changed_artifacts
                ]
            }
        )
    ).hexdigest()
    changed = replace(
        source,
        artifacts=changed_artifacts,
        source_snapshot_sha256=changed_snapshot_sha256,
    )

    original_intent, original_sha256 = bootstrap._raster_text_release_derivation_intent(
        base=base, recognition_model_snapshot=None, raster_text_model_snapshot=source
    )
    changed_intent, changed_sha256 = bootstrap._raster_text_release_derivation_intent(
        base=base, recognition_model_snapshot=None, raster_text_model_snapshot=changed
    )

    assert (
        original_intent["raster_text_model_snapshot"]
        != changed_intent["raster_text_model_snapshot"]
    )
    assert original_sha256 != changed_sha256


@pytest.mark.parametrize("mutation", ("extra", "missing", "symlink", "byte"))
def test_text_direct_model_material_fails_closed_for_every_nonseven_file_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    candidate, selector_before = _materialize_minimal_raster_text_release(root, monkeypatch)
    direct_root = candidate.bundle_root / "models" / "raster-text"
    target = direct_root / "text-det" / "inference.json"
    if mutation == "extra":
        (direct_root / "unexpected.bin").write_bytes(b"unexpected")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "byte":
        target.write_bytes(b"x" * target.stat().st_size)
    else:
        # Windows symlink creation is policy-controlled on some developer
        # machines.  Simulate a reparse-point observation at the exact file
        # boundary so this fail-closed branch is deterministic everywhere.
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: (
                (path.name == "inference.json" and path.parent.name == "text-det")
                or original_is_symlink(path)
            ),
        )

    with pytest.raises(
        RasterOfflineBundleError, match="raster_text_offline_bundle_material_unavailable"
    ):
        verify_raster_text_direct_model_material(candidate)

    assert (root / "raster-offline-bundles-v1" / "current.json").read_bytes() == selector_before


def test_text_materialization_reuses_exact_intent_index_without_stage_or_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second equal request reads one verified index target, never scans/releases."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    (root / selected.bundle_root_relative_path / "unmanifested-cache.pyc").write_bytes(
        b"unmanifested cache"
    )
    selector = root / "raster-offline-bundles-v1" / "current.json"
    selector_before = selector.read_bytes()
    recovery_template = dict(bootstrap._configuration_template())
    recovery_template["raster_profile_revision"] = "text-index-reuse"
    monkeypatch.setattr(bootstrap, "_configuration_template", lambda: recovery_template)

    first = materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)
    intent_directory = root / "raster-offline-bundles-v1" / "raster-text-release-intents-v1"
    index_files = list(intent_directory.glob("*.json"))
    assert len(index_files) == 1
    index_before = index_files[0].read_bytes()

    monkeypatch.setattr(
        bootstrap,
        "_prepare_release_stage",
        lambda **_kwargs: pytest.fail("indexed release must not be staged again"),
    )
    monkeypatch.setattr(
        bootstrap,
        "_sidecar_smoke",
        lambda **_kwargs: pytest.fail("indexed release must not be smoked again"),
    )

    second = materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)

    assert second == first
    assert selector.read_bytes() == selector_before
    assert index_files[0].read_bytes() == index_before
    assert smoke_calls == ["smoke", "smoke"]


def test_fast_text_reference_reuses_current_intent_before_generic_bundle_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The startup fast path returns narrow text evidence, not a full bundle."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)
    candidate, selector_before = _materialize_minimal_raster_text_release(root, monkeypatch)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap,
        "load_verified_raster_offline_bundle",
        lambda *_args, **_kwargs: pytest.fail("fast text reuse must not load a generic bundle"),
    )
    monkeypatch.setattr(
        bootstrap,
        "materialize_verified_unselected_raster_text_release_v1",
        lambda **_kwargs: pytest.fail("matching text intent must not materialize again"),
    )

    reference = resolve_or_materialize_verified_raster_text_release_reference_v1(
        document_preview_root=root
    )

    assert isinstance(reference, VerifiedRasterTextReleaseReferenceV1)
    assert not isinstance(reference, VerifiedRasterOfflineBundleV1)
    assert reference.manifest_sha256 == candidate.manifest_sha256
    assert (root / "raster-offline-bundles-v1" / "current.json").read_bytes() == selector_before
    full_profile = prepare_raster_text_runtime_profile_v1(
        candidate, pdfium_version="4.30.0", minimum_page_confidence=0.8
    )
    fast_profile = prepare_raster_text_runtime_profile_v1(
        reference, pdfium_version="4.30.0", minimum_page_confidence=0.8
    )
    assert fast_profile == full_profile


def test_fast_text_reference_rejects_changed_direct_model_without_generic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching index cannot hide a changed text-model file."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)
    candidate, selector_before = _materialize_minimal_raster_text_release(root, monkeypatch)
    target = candidate.bundle_root / "models" / "raster-text" / "text-det" / "inference.json"
    target.write_bytes(b"changed-direct-model")

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    monkeypatch.setattr(
        bootstrap,
        "materialize_verified_unselected_raster_text_release_v1",
        lambda **_kwargs: pytest.fail("changed matching intent must fail closed, not rebuild"),
    )

    with pytest.raises(
        RasterOfflineBundleBootstrapError,
        match="raster_bundle_unselected_release_intent_candidate_invalid",
    ):
        resolve_or_materialize_verified_raster_text_release_reference_v1(document_preview_root=root)

    assert (root / "raster-offline-bundles-v1" / "current.json").read_bytes() == selector_before


def test_text_materialization_refuses_tampered_intent_index_without_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical but different mapping is fail-closed and never overwritten."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: None)

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    (root / selected.bundle_root_relative_path / "unmanifested-cache.pyc").write_bytes(
        b"unmanifested cache"
    )
    selector = root / "raster-offline-bundles-v1" / "current.json"
    selector_before = selector.read_bytes()
    recovery_template = dict(bootstrap._configuration_template())
    recovery_template["raster_profile_revision"] = "text-index-tamper"
    monkeypatch.setattr(bootstrap, "_configuration_template", lambda: recovery_template)
    materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)

    index_path = next(
        (root / "raster-offline-bundles-v1" / "raster-text-release-intents-v1").glob("*.json")
    )
    tampered = json.loads(index_path.read_text(encoding="utf-8"))
    tampered["release_relative_path"] = "raster-offline-bundles-v1/releases/" + "a" * 64
    index_path.write_bytes(_canonical(tampered))
    index_before = index_path.read_bytes()
    monkeypatch.setattr(
        bootstrap,
        "_prepare_release_stage",
        lambda **_kwargs: pytest.fail("tampered index must not trigger staging"),
    )

    with pytest.raises(
        RasterOfflineBundleBootstrapError,
        match="raster_bundle_unselected_release_intent_index_invalid",
    ):
        materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)

    assert selector.read_bytes() == selector_before
    assert index_path.read_bytes() == index_before


def test_text_materialization_refuses_tampered_indexed_release_without_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index does not make a later modified release trustworthy."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: None)

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    (root / selected.bundle_root_relative_path / "unmanifested-cache.pyc").write_bytes(
        b"unmanifested cache"
    )
    selector = root / "raster-offline-bundles-v1" / "current.json"
    selector_before = selector.read_bytes()
    recovery_template = dict(bootstrap._configuration_template())
    recovery_template["raster_profile_revision"] = "text-index-release-tamper"
    monkeypatch.setattr(bootstrap, "_configuration_template", lambda: recovery_template)
    indexed = materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)
    index_path = next(
        (root / "raster-offline-bundles-v1" / "raster-text-release-intents-v1").glob("*.json")
    )
    index_before = index_path.read_bytes()
    (root / indexed.bundle_root_relative_path / "unmanifested-cache.pyc").write_bytes(
        b"unmanifested cache"
    )
    monkeypatch.setattr(
        bootstrap,
        "_prepare_release_stage",
        lambda **_kwargs: pytest.fail("tampered indexed release must not trigger staging"),
    )

    with pytest.raises(
        RasterOfflineBundleBootstrapError,
        match="raster_bundle_unselected_release_intent_candidate_invalid",
    ):
        materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)

    assert selector.read_bytes() == selector_before
    assert index_path.read_bytes() == index_before


def test_text_materialization_refuses_a_broken_release_without_an_attested_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It must not fall back to scanning arbitrary old release directories."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    _publish_minimal_release_selector(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    (root / selected.bundle_root_relative_path / "unmanifested-cache.pyc").write_bytes(
        b"unmanifested cache"
    )
    selector = root / "raster-offline-bundles-v1" / "current.json"
    selector_before = selector.read_bytes()
    derivation_receipt = (
        root / selected.bundle_root_relative_path / "provenance" / "release-derivation.json"
    )
    tampered = json.loads(derivation_receipt.read_text(encoding="utf-8"))
    tampered["base_bundle_manifest_sha256"] = "b" * 64
    derivation_receipt.write_bytes(_canonical(tampered))

    with pytest.raises(
        RasterOfflineBundleBootstrapError,
        match="raster_bundle_unselected_release_unavailable",
    ):
        materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)

    assert selector.read_bytes() == selector_before
    assert smoke_calls == ["smoke"]


def test_cpu_six_template_derives_a_new_release_without_changing_cpu_one_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    legacy_template_path = root / "raster-offline-bundle-v1" / "sidecar" / "configuration.json"
    legacy_template_before = legacy_template_path.read_bytes()

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    derived_template_path = (
        root / selected.bundle_root_relative_path / "sidecar" / "configuration-template.json"
    )

    assert json.loads(legacy_template_before)["cpu_threads"] == 1
    assert legacy_template_path.read_bytes() == legacy_template_before
    assert json.loads(derived_template_path.read_text(encoding="utf-8"))["cpu_threads"] == 6
    assert smoke_calls == ["smoke"]


def test_only_the_explicit_server_to_local_english_recognition_upgrade_is_allowed() -> None:
    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    desired = bootstrap._configuration_template()
    base = dict(desired)
    base["text_recognition_model_name"] = "PP-OCRv5_server_rec"

    assert bootstrap._is_explicit_local_english_recognition_upgrade(
        base_configuration=base, desired_configuration=desired
    )

    wrong_directory = dict(base)
    wrong_directory["text_recognition_model_directory"] = "outside-text-rec"
    assert not bootstrap._is_explicit_local_english_recognition_upgrade(
        base_configuration=wrong_directory, desired_configuration=desired
    )

    extra_model_change = dict(base)
    extra_model_change["text_detection_model_name"] = "different-detector"
    assert not bootstrap._is_explicit_local_english_recognition_upgrade(
        base_configuration=extra_model_change, desired_configuration=desired
    )


def test_fresh_bootstrap_material_and_smoke_bind_the_downloaded_server_recognizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap
    import customs_ai_agent.service.raster_document_preview_runtime as raster_runtime

    stage = tmp_path / "stage"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    text_rec = stage / "models" / "text-rec"
    text_rec.mkdir(parents=True)
    (text_rec / "inference.yml").write_bytes(b"server-recognizer")
    monkeypatch.setattr(bootstrap, "_required_parent_inventory", lambda: {"packages": []})

    bootstrap._write_bootstrap_material(
        stage=stage,
        model_receipts=[
            {
                "model_name": "PP-OCRv5_server_rec",
                "target_directory": "text-rec",
            }
        ],
        package_inventory={"packages": []},
    )
    configuration_path = stage / "sidecar" / "configuration-template.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    downloaded_recognizer = next(
        item.model_name for item in bootstrap._MODEL_SPECS if item.target_directory == "text-rec"
    )

    assert downloaded_recognizer == "PP-OCRv5_server_rec"
    assert configuration["text_recognition_model_name"] == downloaded_recognizer
    assert bootstrap._configuration_template()["text_recognition_model_name"] == (
        "en_PP-OCRv5_mobile_rec"
    )

    interpreter = stage / "sidecar" / "interpreter" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"interpreter")
    module_root = stage / "sidecar" / "module-root"
    sidecar_path = module_root / "customs_ai_agent" / "service" / "ppstructurev3_sidecar.py"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_bytes(b"sidecar")
    captured_configuration: dict[str, object] = {}

    monkeypatch.setattr(
        raster_runtime,
        "_isolated_v2_sidecar_home",
        lambda: nullcontext(tmp_path / "sidecar-home"),
    )

    def capture_environment(**kwargs: object) -> dict[str, str]:
        smoke_configuration_path = kwargs["configuration_path"]
        assert isinstance(smoke_configuration_path, Path)
        captured_configuration.update(
            json.loads(smoke_configuration_path.read_text(encoding="utf-8"))
        )
        return {}

    monkeypatch.setattr(raster_runtime, "_isolated_v2_sidecar_environment", capture_environment)

    def successful_smoke(*_args: object, **kwargs: object) -> SimpleNamespace:
        payload = kwargs["input"]
        assert isinstance(payload, bytes)
        header = json.loads(payload.split(b"\n", maxsplit=1)[0])
        return SimpleNamespace(
            returncode=0,
            stdout=_canonical(
                {
                    "source_sha256": header["source_sha256"],
                    "raster_profile_revision": "synthetic-raster-v1",
                }
            ),
        )

    monkeypatch.setattr(bootstrap.subprocess, "run", successful_smoke)

    bootstrap._sidecar_smoke(
        stage=stage,
        workspace=workspace,
        interpreter=interpreter,
        module_root=module_root,
        sidecar_path=sidecar_path,
        package_inventory_path=stage / "inventories" / "sidecar-packages.json",
        model_inventory_path=stage / "inventories" / "models.json",
        models_root=stage / "models",
    )

    assert captured_configuration["text_recognition_model_name"] == downloaded_recognizer


def test_release_stage_copies_only_the_attested_local_english_recognition_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    source = root / "models" / "paddleocr3" / "rec"
    source.mkdir(parents=True)
    source_files = {
        "config.json": b"fixed-config",
        "inference.json": b"fixed-graph",
        "inference.pdiparams": b"fixed-parameters",
        "inference.yml": b"fixed-metadata",
    }
    for filename, payload in source_files.items():
        (source / filename).write_bytes(payload)
    monkeypatch.setattr(
        bootstrap,
        "_PINNED_LOCAL_ENGLISH_RECOGNITION_ARTIFACTS",
        tuple(
            SimpleNamespace(
                filename=filename,
                byte_size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                git_blob_sha1=None,
            )
            for filename, payload in sorted(source_files.items())
        ),
    )

    snapshot = bootstrap._snapshot_local_english_recognition_model(root)
    base = load_verified_raster_offline_bundle(root)
    stage = root / ".release-stage"
    bootstrap._prepare_release_stage(
        base=base,
        stage=stage,
        recognition_model_snapshot=snapshot,
    )

    copied = stage / "models" / "text-rec"
    assert {path.name: path.read_bytes() for path in copied.iterdir()} == source_files
    inventory = json.loads((stage / "inventories" / "models.json").read_text(encoding="utf-8"))
    assert {entry["relative_path"] for entry in inventory["artifacts"]} >= {
        f"text-rec/{filename}" for filename in source_files
    }
    assert "text-rec/weights.bin" not in {
        entry["relative_path"] for entry in inventory["artifacts"]
    }
    configuration = json.loads(
        (stage / "sidecar" / "configuration-template.json").read_text(encoding="utf-8")
    )
    assert configuration["text_recognition_model_name"] == "en_PP-OCRv5_mobile_rec"
    derivation = json.loads(
        (stage / "provenance" / "release-derivation.json").read_text(encoding="utf-8")
    )
    assert derivation["recognition_model_snapshot"] == {
        "model_name": "en_PP-OCRv5_mobile_rec",
        "source_relative_path": "models/paddleocr3/rec",
        "target_directory": "text-rec",
        "tree_sha256": snapshot.tree_sha256,
    }


def test_local_english_recognition_snapshot_rejects_unexpected_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    source = root / "models" / "paddleocr3" / "rec"
    source.mkdir(parents=True)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    payload = b"fixed-model"
    (source / "only.bin").write_bytes(payload)
    monkeypatch.setattr(
        bootstrap,
        "_PINNED_LOCAL_ENGLISH_RECOGNITION_ARTIFACTS",
        (
            SimpleNamespace(
                filename="only.bin",
                byte_size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                git_blob_sha1=None,
            ),
        ),
    )
    assert bootstrap._snapshot_local_english_recognition_model(root).source_relative_path == (
        "models/paddleocr3/rec"
    )

    (source / "unexpected.bin").write_bytes(b"not-approved")
    with pytest.raises(
        RasterOfflineBundleBootstrapError, match="raster_bundle_local_english_recognition_invalid"
    ):
        bootstrap._snapshot_local_english_recognition_model(root)


def test_rederivation_replaces_the_copied_release_derivation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))

    first = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    next_template = dict(bootstrap._configuration_template())
    next_template["raster_profile_revision"] = "template-next"
    monkeypatch.setattr(bootstrap, "_configuration_template", lambda: next_template)

    second = ensure_raster_offline_bundle_release_v1(document_preview_root=root)

    assert first.manifest_sha256 != second.manifest_sha256
    assert smoke_calls == ["smoke", "smoke"]
    second_receipt = json.loads(
        (
            root / second.bundle_root_relative_path / "provenance" / "release-derivation.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        second_receipt["configuration_template_sha256"]
        == hashlib.sha256(_canonical(next_template)).hexdigest()
    )


def test_legacy_bundle_is_promoted_to_a_release_before_auto_profile_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    legacy_manifest_before = (root / "raster-offline-bundle-v1" / "manifest.json").read_bytes()

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    # Exercise the promotion condition itself rather than a source/config
    # change: even a byte-identical legacy bundle must become a selected
    # content-addressed release before a newly generated profile can bind it.
    monkeypatch.setattr(bootstrap, "_release_requires_derivation", lambda _bundle: False)
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)

    assert selected.bundle_root_relative_path == (
        f"raster-offline-bundles-v1/releases/{selected.manifest_sha256}"
    )
    selector = json.loads((root / "raster-offline-bundles-v1" / "current.json").read_text())
    assert selector["release_manifest_sha256"] == selected.manifest_sha256
    assert selector["release_relative_path"] == selected.bundle_root_relative_path
    assert (
        root / "raster-offline-bundle-v1" / "manifest.json"
    ).read_bytes() == legacy_manifest_before
    assert smoke_calls == ["smoke"]


def test_historical_legacy_bundle_is_migrated_to_a_safe_release_without_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root, include_safe_engine_detector_configuration=False)

    # Normal loading is the runtime/approval path and must not silently add
    # new settings to an old hash-bound sidecar configuration.
    with pytest.raises(RasterOfflineBundleError, match="offline_bundle_manifest_invalid"):
        load_verified_raster_offline_bundle(root)

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    smoke_calls: list[str] = []
    # The compact fixture lacks the production provenance receipt; isolate
    # that unrelated compatibility check while proving this migration neither
    # invokes package installation nor downloads model bytes.
    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: smoke_calls.append("smoke"))
    monkeypatch.setattr(
        bootstrap,
        "_install_sidecar_packages",
        lambda **_kwargs: pytest.fail("safe legacy migration must not install packages"),
    )
    monkeypatch.setattr(
        bootstrap,
        "_download_models",
        lambda **_kwargs: pytest.fail("safe legacy migration must not download models"),
    )

    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    profile = prepare_raster_runtime_profile_v2(selected, pdfium_version="4.30.0")
    sidecar_path = root / profile.sidecar_configuration_relative_path
    sidecar_raw = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_raw)
    runtime = json.loads(
        (root / profile.runtime_configuration_relative_path).read_text(encoding="utf-8")
    )

    assert selected.bundle_root_relative_path == (
        f"raster-offline-bundles-v1/releases/{selected.manifest_sha256}"
    )
    assert sidecar["inference_engine"] == "paddle_static"
    assert sidecar["paddle_static_enable_new_ir"] is False
    assert sidecar["paddle_static_enable_cinn"] is False
    assert sidecar["evaluation_batch_size"] == 1
    # The input legacy bundle remains CPU-1/960, but source-derived releases
    # bind the current, fixed 2048-pixel detector profile as new material.
    assert sidecar["text_detection_limit_side_len"] == 2048
    assert sidecar["text_detection_limit_type"] == "max"
    assert sidecar["text_detection_threshold"] == 0.3
    assert sidecar["text_detection_box_threshold"] == 0.6
    assert sidecar["text_detection_unclip_ratio"] == 1.5
    # The released profile must bind the freshly written current configuration
    # bytes, not the historical template it used as copy-only source material.
    assert runtime["sidecar_configuration_path"] == profile.sidecar_configuration_relative_path
    assert runtime["sidecar_configuration_sha256"] == hashlib.sha256(sidecar_raw).hexdigest()
    assert profile.sidecar_configuration_sha256 == hashlib.sha256(sidecar_raw).hexdigest()
    assert smoke_calls == ["smoke"]


def test_historical_generated_bytecode_caches_are_discarded_during_safe_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root, include_safe_engine_detector_configuration=False)
    cache = (
        root
        / "raster-offline-bundle-v1"
        / "sidecar"
        / "module-root"
        / "customs_ai_agent"
        / "service"
        / "__pycache__"
        / "ppstructurev3_sidecar.cpython-311.pyc"
    )
    cache.parent.mkdir()
    cache.write_bytes(b"untrusted generated bytecode")

    # The standard resolver remains exact-file-set strict.  Only the fixed
    # legacy-root migration loader may identify this cache as disposable.
    with pytest.raises(RasterOfflineBundleError, match="offline_bundle_manifest_invalid"):
        load_verified_raster_offline_bundle(root)
    legacy = load_verified_legacy_raster_offline_bundle_for_safe_configuration_migration(root)
    assert legacy.historical_generated_bytecode_cache_relative_paths == frozenset(
        {cache.relative_to(root / "raster-offline-bundle-v1").as_posix()}
    )

    import customs_ai_agent.evaluation.raster_offline_bundle_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_verify_release_reuse_compatibility", lambda _bundle: None)
    monkeypatch.setattr(bootstrap, "_sidecar_smoke", lambda **_kwargs: None)
    selected = ensure_raster_offline_bundle_release_v1(document_preview_root=root)

    assert cache.exists()  # The legacy source bundle is never mutated.
    # ``selected`` has just passed the normal exact-file-set verifier, so its
    # manifest is sufficient evidence that the unmanifested cache was not
    # copied into the release (and avoids a long-path recursive walk here).
    assert not any(path.endswith(".pyc") for path in selected.file_sha256_by_relative_path)


def test_historical_migration_rejects_unmatched_generated_bytecode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root, include_safe_engine_detector_configuration=False)
    cache = (
        root
        / "raster-offline-bundle-v1"
        / "sidecar"
        / "module-root"
        / "customs_ai_agent"
        / "service"
        / "__pycache__"
        / "not_a_declared_source.cpython-311.pyc"
    )
    cache.parent.mkdir()
    cache.write_bytes(b"untrusted generated bytecode")

    with pytest.raises(RasterOfflineBundleBootstrapError, match="offline_bundle_manifest_invalid"):
        ensure_raster_offline_bundle_release_v1(document_preview_root=root)
    assert not (root / "raster-offline-bundles-v1").exists()


def test_verified_bundle_requires_the_complete_ppstructure_sidecar_package_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    manifest_path = root / "raster-offline-bundle-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sidecar_required_packages"] = ["paddleocr", "paddlex", "opencv-python"]
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(Exception, match="offline_bundle_manifest_invalid"):
        load_verified_raster_offline_bundle(root)


def test_verified_bundle_rejects_a_sidecar_module_outside_its_controlled_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    manifest_path = root / "raster-offline-bundle-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sidecar_module_relative_path"] = "sidecar/interpreter/python.exe"
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(Exception, match="offline_bundle_manifest_invalid"):
        load_verified_raster_offline_bundle(root)


def test_verified_bundle_rejects_a_separate_sidecar_command_wrapper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    _write_minimal_bundle(root)
    manifest_path = root / "raster-offline-bundle-v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sidecar_command_relative_path"] = "sidecar/module-root/customs_ai_agent/__init__.py"
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(Exception, match="offline_bundle_manifest_invalid"):
        load_verified_raster_offline_bundle(root)


def test_auto_activation_defers_runtime_profile_until_dataset_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    fake_bundle = SimpleNamespace(document_preview_root=root)
    calls: list[str] = []
    monkeypatch.setattr(
        auto_activation, "ensure_raster_offline_bundle_release_v1", lambda **_kwargs: fake_bundle
    )
    monkeypatch.setattr(auto_activation, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(
        auto_activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: SimpleNamespace(
            build_inputs_relative_path="synthetic-raster-build/inputs.json",
            renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
        ),
    )

    def _stop_before_publish(**_kwargs: object) -> None:
        calls.append("compile")
        raise auto_activation.SyntheticRasterCompilerUnavailable("missing")

    monkeypatch.setattr(auto_activation, "compile_synthetic_raster_dataset", _stop_before_publish)
    monkeypatch.setattr(
        auto_activation,
        "prepare_raster_runtime_profile_v2",
        lambda *_args, **_kwargs: pytest.fail("auto activation must wait for dataset publish"),
    )

    with pytest.raises(
        SyntheticRasterAutoActivationError, match="synthetic_raster_build_unavailable"
    ):
        auto_activate_synthetic_raster_v1(document_preview_root=root)

    assert calls == ["compile"]
    assert not (root / "synthetic-raster-v1").exists()


def test_auto_activation_reports_a_sanitized_dataset_build_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    fake_bundle = SimpleNamespace(document_preview_root=root)
    monkeypatch.setattr(
        auto_activation, "ensure_raster_offline_bundle_release_v1", lambda **_kwargs: fake_bundle
    )
    monkeypatch.setattr(auto_activation, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(
        auto_activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: SimpleNamespace(
            build_inputs_relative_path="synthetic-raster-build-v2/inputs.json",
            renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
        ),
    )

    def _fail_without_leaking_source_details(**_kwargs: object) -> None:
        raise ValueError("sensitive local source structure")

    monkeypatch.setattr(
        auto_activation, "compile_synthetic_raster_dataset", _fail_without_leaking_source_details
    )
    with pytest.raises(
        SyntheticRasterAutoActivationError, match="synthetic_raster_dataset_build_failed"
    ):
        auto_activate_synthetic_raster_v1(document_preview_root=root)

    assert not (root / "synthetic-raster-v1").exists()
    assert not (root / "approvals" / "raster-preview-active.json").exists()


def test_sealed_gate_failure_never_calls_pointer_writer_or_replaces_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    pointer = root / "approvals" / "raster-preview-active.json"
    pointer.parent.mkdir(parents=True)
    original = b'{"previous":"activation"}'
    pointer.write_bytes(original)
    fake_bundle = SimpleNamespace(document_preview_root=root)
    fake_inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-build/inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
    )
    fake_profile = SimpleNamespace(
        runtime_configuration_relative_path="raster-runtime-profiles-v1/a/profile.json",
        raster_profile_revision="synthetic-raster-v1:" + "a" * 24,
        runtime_configuration_sha256="c" * 64,
    )
    monkeypatch.setattr(
        auto_activation, "ensure_raster_offline_bundle_release_v1", lambda **_kwargs: fake_bundle
    )
    monkeypatch.setattr(auto_activation, "verify_parent_renderer_inventory", lambda _bundle: None)
    monkeypatch.setattr(
        auto_activation, "ensure_fixed_synthetic_raster_build_inputs", lambda **_kwargs: fake_inputs
    )
    monkeypatch.setattr(auto_activation, "compile_synthetic_raster_dataset", lambda **_kwargs: None)
    monkeypatch.setattr(
        auto_activation, "read_verified_synthetic_raster_dataset", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        auto_activation, "prepare_raster_runtime_profile_v2", lambda *_args, **_kwargs: fake_profile
    )
    import tools.evaluate_synthetic_raster as evaluator
    import tools.run_ppstructurev3_synthetic_raster_candidates as runner

    monkeypatch.setattr(
        runner,
        "run_ppstructurev3_synthetic_raster_candidates",
        lambda **kwargs: {
            "candidate_bundle_relative_path": f"bundles/{kwargs['split']}.json",
            "candidate_bundle_sha256": "d" * 64,
            "candidate_execution_relative_path": f"executions/{kwargs['split']}.json",
            "candidate_execution_sha256": "e" * 64,
            "runtime_configuration_sha256": "c" * 64,
            "raster_profile_revision": "synthetic-raster-v1:" + "a" * 24,
        },
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate_synthetic_raster_split",
        lambda **kwargs: (
            {"status": "calibrated"} if kwargs["split"] == "calibration" else {"status": "failed"},
            "b" * 64,
        ),
    )
    import customs_ai_agent.service.raster_document_preview_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "write_source_attested_synthetic_raster_preview_pointer",
        lambda **_kwargs: pytest.fail("a failed sealed Gate must not write a pointer"),
    )

    with pytest.raises(SyntheticRasterAutoActivationError, match="sealed_raster_gate_failed"):
        auto_activate_synthetic_raster_v1(document_preview_root=root)

    assert pointer.read_bytes() == original
