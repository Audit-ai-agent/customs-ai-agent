from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import ValidationError
import pytest

from customs_ai_agent.evaluation import raster_offline_bundle as bundle_runtime
from customs_ai_agent.evaluation import raster_offline_bundle_bootstrap as bootstrap


def _linux_spec() -> bootstrap._RuntimePlatformSpec:
    return bootstrap._select_runtime_platform(  # noqa: SLF001
        os_name="posix",
        sys_platform="linux",
        machine="x86_64",
        implementation="cpython",
        python_version=(3, 11),
        pointer_bits=64,
    )


def test_linux_amd64_cp311_selects_a_distinct_lock_and_interpreter() -> None:
    selected = _linux_spec()

    assert selected.operating_system == "linux"
    assert selected.architecture == "amd64"
    assert selected.python_abi == "cp311"
    assert selected.requirements_relative_path == (
        "requirements/raster-sidecar-linux-amd64-cp311.lock"
    )
    assert selected.interpreter_relative_path == "sidecar/interpreter/bin/python3.11"


@pytest.mark.parametrize(
    ("machine", "implementation", "version", "pointer_bits"),
    (
        ("aarch64", "cpython", (3, 11), 64),
        ("x86_64", "pypy", (3, 11), 64),
        ("x86_64", "cpython", (3, 12), 64),
        ("x86_64", "cpython", (3, 11), 32),
    ),
)
def test_platform_selection_has_no_native_fallback(
    machine: str,
    implementation: str,
    version: tuple[int, int],
    pointer_bits: int,
) -> None:
    with pytest.raises(
        bootstrap.RasterOfflineBundleBootstrapError,
        match="raster_bundle_bootstrap_platform_unsupported",
    ):
        bootstrap._select_runtime_platform(  # noqa: SLF001
            os_name="posix",
            sys_platform="linux",
            machine=machine,
            implementation=implementation,
            python_version=version,
            pointer_bits=pointer_bits,
        )


def test_schema2_manifest_binds_linux_interpreter_layout(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    interpreter = stage / "sidecar" / "interpreter" / "bin" / "python3.11"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"linux-cpython")

    bootstrap._write_manifest(stage=stage, platform_spec=_linux_spec())  # noqa: SLF001

    value = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    manifest = bundle_runtime.RasterOfflineBundleManifestV1.model_validate(value)
    assert manifest.schema_version == "2.0"
    assert manifest.runtime_operating_system == "linux"
    assert manifest.runtime_architecture == "amd64"
    assert manifest.runtime_python_abi == "cp311"
    assert manifest.sidecar_interpreter_relative_path == ("sidecar/interpreter/bin/python3.11")

    value["sidecar_interpreter_relative_path"] = "sidecar/interpreter/python.exe"
    with pytest.raises(ValidationError):
        bundle_runtime.RasterOfflineBundleManifestV1.model_validate(value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX special files are platform-specific")
def test_base_bundle_tree_rejects_a_posix_fifo(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    os.mkfifo(root / "pipe")

    with pytest.raises(
        bootstrap.RasterOfflineBundleBootstrapError,
        match="raster_bundle_bootstrap_tree_invalid",
    ):
        bootstrap._assert_regular_tree(root)  # noqa: SLF001
