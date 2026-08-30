from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    SyntheticRasterTextActivationError,
    SyntheticRasterTextAutoActivationResultV1,
)
from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from tools import auto_activate_synthetic_raster_text_linux_v1 as cli


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def test_linux_command_exposes_only_the_fixed_root_and_reports_v12_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    observed: list[Path] = []

    def activate(*, document_preview_root: Path) -> SyntheticRasterTextAutoActivationResultV1:
        observed.append(document_preview_root)
        return SyntheticRasterTextAutoActivationResultV1(
            raster_text_profile_revision=f"synthetic-raster-text-v1:{'a' * 24}",
            activation_pointer_sha256="b" * 64,
            sealed_evaluation_relative_path="synthetic-raster-text-v1/sealed-v9/evidence.json",
            sealed_evaluation_sha256="c" * 64,
        )

    monkeypatch.setattr(cli, "auto_activate_synthetic_raster_text_linux_preview97_v1", activate)
    assert cli.main(["--document-preview-root", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert observed == [root.resolve()]
    assert payload == {
        "activation_pointer_sha256": "b" * 64,
        "dataset_id": "synthetic-raster-text-v1",
        "raster_text_profile_revision": f"synthetic-raster-text-v1:{'a' * 24}",
        "runtime_network": "disabled",
        "runtime_platform": "linux/amd64/cp311",
        "status": "activated",
    }


def test_linux_command_returns_only_content_free_failure_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)

    def fail(*, document_preview_root: Path) -> SyntheticRasterTextAutoActivationResultV1:
        assert document_preview_root == root.resolve()
        raise SyntheticRasterTextActivationError("offline_bundle_unavailable")

    monkeypatch.setattr(cli, "auto_activate_synthetic_raster_text_linux_preview97_v1", fail)
    assert cli.main(["--document-preview-root", str(root)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "code": "offline_bundle_unavailable",
    }


def test_linux_activation_rejects_another_native_target_before_bundle_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    monkeypatch.setattr(
        activation,
        "current_raster_runtime_platform_binding",
        lambda: ("windows", "amd64", "cp311"),
    )
    monkeypatch.setattr(
        activation,
        "ensure_raster_offline_bundle_release_v1",
        lambda **_kwargs: pytest.fail("bundle must not be opened on a non-Linux host"),
    )

    with pytest.raises(
        SyntheticRasterTextActivationError,
        match="^raster_text_linux_activation_platform_invalid$",
    ):
        activation.auto_activate_synthetic_raster_text_linux_preview97_v1(
            document_preview_root=root
        )


def test_linux_activation_uses_fixed_materials_and_current_v12_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    offline_manifest_sha256 = "1" * 64
    direct_manifest_sha256 = "2" * 64
    offline = SimpleNamespace(
        document_preview_root=root,
        bundle_root_relative_path="raster-offline-bundle-v1/releases/linux",
        manifest_sha256=offline_manifest_sha256,
        manifest=SimpleNamespace(
            schema_version="2.0",
            runtime_operating_system="linux",
            runtime_architecture="amd64",
            runtime_python_abi="cp311",
        ),
    )
    sidecar = SimpleNamespace(
        document_preview_root=root,
        bundle_root_relative_path="raster-text-direct-bundles-v1/releases/linux",
        manifest_sha256=direct_manifest_sha256,
        manifest=SimpleNamespace(
            schema_version="2.0",
            source_bundle_manifest_sha256=offline_manifest_sha256,
            runtime_operating_system="linux",
            runtime_architecture="amd64",
            runtime_python_abi="cp311",
        ),
    )
    build_inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-v1/build-inputs-v1/build-inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
    )
    profile_info = SimpleNamespace(
        runtime_configuration_relative_path="profiles/linux-v19.json",
        runtime_configuration_sha256="3" * 64,
    )
    profile = activation.RasterTextOfflineProfileV19.model_construct(
        engine_kind=activation._V19_ENGINE_KIND,  # noqa: SLF001
    )
    observed: dict[str, object] = {
        "compile": 0,
        "ensure_release": 0,
        "selection": 0,
        "profiles": [],
    }

    monkeypatch.setattr(
        activation,
        "current_raster_runtime_platform_binding",
        lambda: ("linux", "amd64", "cp311"),
    )
    def ensure_release(*, document_preview_root: Path) -> SimpleNamespace:
        assert document_preview_root == root
        observed["ensure_release"] = int(observed["ensure_release"]) + 1
        return offline

    monkeypatch.setattr(
        activation,
        "ensure_raster_offline_bundle_release_v1",
        ensure_release,
    )
    monkeypatch.setattr(activation, "load_verified_raster_offline_bundle", lambda _root: offline)

    def materialize(**kwargs: object) -> SimpleNamespace:
        observed["materialize"] = kwargs
        return sidecar

    monkeypatch.setattr(
        activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        materialize,
    )
    monkeypatch.setattr(
        activation,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: sidecar,
    )
    monkeypatch.setattr(
        activation, "ensure_fixed_synthetic_raster_build_inputs", lambda **_kwargs: build_inputs
    )
    monkeypatch.setattr(activation, "_existing_synthetic_raster_dataset", lambda _root: True)
    monkeypatch.setattr(
        activation,
        "compile_synthetic_raster_dataset",
        lambda **_kwargs: observed.__setitem__("compile", int(observed["compile"]) + 1),
    )
    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: observed.__setitem__("selection", int(observed["selection"]) + 1),
    )

    recognizer = SimpleNamespace(
        release_sha256=activation._V19_PUBLIC_FINETUNED_RECOGNIZER_RELEASE_SHA256  # noqa: SLF001
    )
    model = SimpleNamespace(
        manifest_sha256=activation._V19_PUBLIC_FINETUNED_MODEL_BUNDLE_SHA256  # noqa: SLF001
    )
    recognizer_hashes: list[str] = []
    model_hashes: list[str] = []

    def load_recognizer(_root: Path, digest: str) -> SimpleNamespace:
        recognizer_hashes.append(digest)
        return recognizer

    def load_model(_root: Path, digest: str) -> SimpleNamespace:
        model_hashes.append(digest)
        return model

    monkeypatch.setattr(
        activation, "load_verified_public_finetuned_recognizer_release_v1", load_recognizer
    )
    monkeypatch.setattr(
        activation, "load_verified_raster_text_public_finetuned_paddle_bundle_v1", load_model
    )
    monkeypatch.setattr(
        activation,
        "_verify_v19_public_finetune_contract",
        lambda **_kwargs: recognizer,
    )

    def prepare_profile(*_args: object, **kwargs: object) -> SimpleNamespace:
        cast_profiles = observed["profiles"]
        assert isinstance(cast_profiles, list)
        cast_profiles.append(kwargs)
        return profile_info

    monkeypatch.setattr(
        activation,
        "prepare_raster_text_public_finetuned_one_pass_runtime_profile_v19",
        prepare_profile,
    )
    monkeypatch.setattr(activation, "_read", lambda *_args, **_kwargs: b"profile")
    monkeypatch.setattr(activation, "parse_raster_text_offline_profile", lambda _raw: profile)

    expected = SyntheticRasterTextAutoActivationResultV1(
        raster_text_profile_revision=f"synthetic-raster-text-v1:{'a' * 24}",
        activation_pointer_sha256="b" * 64,
        sealed_evaluation_relative_path="synthetic-raster-text-v1/sealed-v9/result.json",
        sealed_evaluation_sha256="c" * 64,
    )

    def staged(**kwargs: object) -> SyntheticRasterTextAutoActivationResultV1:
        kwargs["recheck_materials"]()  # type: ignore[operator]
        kwargs["prepare_final_profile"](0.5)  # type: ignore[operator]
        observed["staged"] = kwargs
        return expected

    monkeypatch.setattr(activation, "_run_v2_staged_candidate_activation", staged)

    assert (
        activation.auto_activate_synthetic_raster_text_linux_preview97_v1(
            document_preview_root=root
        )
        == expected
    )
    assert observed["materialize"] == {
        "document_preview_root": root,
        "source_manifest_sha256": offline_manifest_sha256,
    }
    assert recognizer_hashes == [
        activation._V19_PUBLIC_FINETUNED_RECOGNIZER_RELEASE_SHA256,  # noqa: SLF001
        activation._V19_PUBLIC_FINETUNED_RECOGNIZER_RELEASE_SHA256,  # noqa: SLF001
    ]
    assert model_hashes == [
        activation._V19_PUBLIC_FINETUNED_MODEL_BUNDLE_SHA256,  # noqa: SLF001
        activation._V19_PUBLIC_FINETUNED_MODEL_BUNDLE_SHA256,  # noqa: SLF001
    ]
    assert observed["compile"] == 1
    assert observed["ensure_release"] == 1
    assert observed["selection"] == 1
    assert observed["profiles"] == [
        {
            "pdfium_version": "4.30.0",
            "minimum_page_confidence": 0.0,
            "cpu_threads": 6,
        },
        {
            "pdfium_version": "4.30.0",
            "minimum_page_confidence": 0.5,
            "cpu_threads": 6,
        },
    ]
    staged_kwargs = observed["staged"]
    assert isinstance(staged_kwargs, dict)
    assert staged_kwargs["expected_pointer_type"] is activation.RasterTextActivationPointerV12
    assert staged_kwargs["write_sealed"] is (
        activation.write_raster_semantic_text_fresh_sealed_evaluation_v9
    )
    assert staged_kwargs["quality_gate_revision"] == "preview-draft-critical97-cer2-v1"
