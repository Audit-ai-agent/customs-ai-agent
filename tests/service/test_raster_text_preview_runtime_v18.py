"""Focused contracts for the V18 server-recognizer runtime binding."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from pydantic import ValidationError
import pytest

from customs_ai_agent.evaluation import raster_text_server_paddle_bundle as server_bundle
import customs_ai_agent.service.raster_text_preview_runtime as runtime
from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextActivationPointerV6,
    RasterTextCalibrationReceiptV6,
    RasterTextOfflineProfileV18,
    RasterTextRuntimeError,
    RasterTextSealedEvaluationV6,
    RasterTextSealedProfileScoreV6,
)


_SIDECAR_SHA256 = "a" * 64
_SERVER_SHA256 = "b" * 64
_PROFILE_REVISION_PREFIX = "synthetic-raster-text-v1:"
_EVIDENCE_REVISION = "source-attested-fresh-staged-train-holdout-v2"
_ENGINE_KIND = (
    "direct_paddle_mobile_det_server_rec_harmonic_recall_native960_lowbox_"
    "strip2_semantic_repair_v2_onednn"
)


@dataclass(frozen=True)
class _FakeBundle:
    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest_sha256: str
    manifest: Any
    hashes: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        return self.hashes[path]


@dataclass(frozen=True)
class _PreparedFixture:
    root: Path
    sidecar: _FakeBundle
    server: _FakeBundle
    prepared: server_bundle.PreparedRasterTextDirectServerRuntimeProfileV18
    profile: RasterTextOfflineProfileV18
    raw: bytes


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_pointer(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _make_verified_bundle_fakes(root: Path) -> tuple[_FakeBundle, _FakeBundle]:
    sidecar_relative = f"raster-text-direct-bundles-v1/releases/{_SIDECAR_SHA256}"
    server_relative = (
        f"raster-text-direct-server-rec-bundles-v1/releases/{_SERVER_SHA256}"
    )
    sidecar_manifest = SimpleNamespace(
        extractor_revision="direct-paddle-test-v18",
        raster_dpi=216,
        parent_renderer_package_inventory_relative_path="parent/packages.json",
        parent_renderer_required_packages=["pypdfium2", "Pillow"],
        sidecar_interpreter_relative_path="sidecar/interpreter-root/python.exe",
        sidecar_interpreter_root_relative_path="sidecar/interpreter-root",
        sidecar_interpreter_root_tree_sha256="c" * 64,
        sidecar_module_relative_path=(
            "sidecar/module-root/customs_ai_agent/service/"
            "raster_text_direct_paddle_sidecar.py"
        ),
        sidecar_module_root_relative_path="sidecar/module-root",
        sidecar_module_root_tree_sha256="d" * 64,
        sidecar_package_inventory_relative_path="sidecar/packages.json",
        sidecar_required_packages=[
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ],
    )
    sidecar_hashes = {
        sidecar_manifest.parent_renderer_package_inventory_relative_path: "1" * 64,
        sidecar_manifest.sidecar_interpreter_relative_path: "2" * 64,
        sidecar_manifest.sidecar_module_relative_path: "3" * 64,
        sidecar_manifest.sidecar_package_inventory_relative_path: "4" * 64,
    }
    server_manifest = SimpleNamespace(
        local_model_root_relative_path="models",
        model_inventory_relative_path="inventories/direct-paddle-server-rec-models.json",
        provenance_receipt_relative_path="provenance/source-bundles.json",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
    )
    server_hashes = {
        server_manifest.model_inventory_relative_path: "5" * 64,
        server_manifest.provenance_receipt_relative_path: "6" * 64,
    }
    return (
        _FakeBundle(
            document_preview_root=root,
            bundle_root=root / sidecar_relative,
            bundle_root_relative_path=sidecar_relative,
            manifest_sha256=_SIDECAR_SHA256,
            manifest=sidecar_manifest,
            hashes=sidecar_hashes,
        ),
        _FakeBundle(
            document_preview_root=root,
            bundle_root=root / server_relative,
            bundle_root_relative_path=server_relative,
            manifest_sha256=_SERVER_SHA256,
            manifest=server_manifest,
            hashes=server_hashes,
        ),
    )


def _prepare_fixture(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _PreparedFixture:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    sidecar, server = _make_verified_bundle_fakes(root)

    def load_sidecar(*, document_preview_root: Path, manifest_sha256: str) -> _FakeBundle:
        assert document_preview_root == root
        assert manifest_sha256 == _SIDECAR_SHA256
        return sidecar

    def load_server(*, document_preview_root: Path, manifest_sha256: str) -> _FakeBundle:
        assert document_preview_root == root
        assert manifest_sha256 == _SERVER_SHA256
        return server

    monkeypatch.setattr(
        server_bundle,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        load_sidecar,
    )
    monkeypatch.setattr(
        server_bundle,
        "load_verified_raster_text_direct_server_paddle_bundle_v1",
        load_server,
    )
    prepared = server_bundle.prepare_raster_text_direct_server_runtime_profile_v18(
        sidecar,  # type: ignore[arg-type]
        server,  # type: ignore[arg-type]
        pdfium_version="4.30.0",
        minimum_page_confidence=0.42,
        cpu_threads=6,
    )
    raw = (root / prepared.runtime_configuration_relative_path).read_bytes()
    profile = runtime.parse_raster_text_offline_profile(raw)
    assert isinstance(profile, RasterTextOfflineProfileV18)
    return _PreparedFixture(root, sidecar, server, prepared, profile, raw)


def _rebind_profile_fingerprint(payload: dict[str, object]) -> dict[str, object]:
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"profile_fingerprint_sha256", "raster_text_profile_revision"}
    }
    fingerprint = runtime.raster_text_profile_fingerprint_sha256(fingerprint_payload)
    payload["profile_fingerprint_sha256"] = fingerprint
    payload["raster_text_profile_revision"] = _PROFILE_REVISION_PREFIX + fingerprint[:24]
    return payload


def test_v18_preparer_writes_one_canonical_idempotent_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _prepare_fixture(tmp_path=tmp_path, monkeypatch=monkeypatch)

    repeated = server_bundle.prepare_raster_text_direct_server_runtime_profile_v18(
        fixture.sidecar,  # type: ignore[arg-type]
        fixture.server,  # type: ignore[arg-type]
        pdfium_version="4.30.0",
        minimum_page_confidence=0.42,
        cpu_threads=6,
    )

    assert repeated == fixture.prepared
    assert fixture.raw == server_bundle._canonical_json(  # noqa: SLF001
        fixture.profile.model_dump(mode="json", warnings=False)
    )
    assert not fixture.raw.endswith(b"\n")
    assert fixture.profile.engine_kind == _ENGINE_KIND
    assert fixture.profile.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert fixture.profile.text_recognition_model_name == "PP-OCRv5_server_rec"
    assert fixture.profile.mobile_model_bundle_root_relative_path == (
        fixture.server.bundle_root_relative_path
    )
    assert fixture.profile.source_model_inventory_relative_path == (
        f"{fixture.server.bundle_root_relative_path}/provenance/source-bundles.json"
    )
    assert fixture.profile.local_model_root == (
        f"{fixture.server.bundle_root_relative_path}/models"
    )
    assert fixture.profile.ctc_label_policy == "append-space-v1"


def _mobile_root_drift(payload: dict[str, object]) -> None:
    payload["mobile_model_bundle_root_relative_path"] = (
        "raster-text-direct-mobile-bundles-v1/releases/" + _SERVER_SHA256
    )


def _provenance_drift(payload: dict[str, object]) -> None:
    payload["source_model_inventory_relative_path"] = str(
        payload["mobile_model_bundle_root_relative_path"]
    )


def _model_name_drift(payload: dict[str, object]) -> None:
    payload["text_recognition_model_name"] = "en_PP-OCRv5_mobile_rec"


def _control_drift(payload: dict[str, object]) -> None:
    payload["recognition_batch_size"] = 7


@pytest.mark.parametrize(
    "mutate",
    [_mobile_root_drift, _provenance_drift, _model_name_drift, _control_drift],
    ids=["mobile-root", "provenance", "model-name", "runtime-control"],
)
def test_v18_profile_rejects_drift_even_with_a_recomputed_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    fixture = _prepare_fixture(tmp_path=tmp_path, monkeypatch=monkeypatch)
    payload = fixture.profile.model_dump(mode="json", warnings=False)
    mutate(payload)
    _rebind_profile_fingerprint(payload)

    with pytest.raises(ValidationError):
        RasterTextOfflineProfileV18.model_validate(payload)


def _bundle_root_mismatch(bundle: _FakeBundle) -> _FakeBundle:
    return replace(
        bundle,
        bundle_root_relative_path=(
            "raster-text-direct-server-rec-bundles-v1/releases/" + "c" * 64
        ),
    )


def _inventory_hash_mismatch(bundle: _FakeBundle) -> _FakeBundle:
    hashes = dict(bundle.hashes)
    hashes[bundle.manifest.model_inventory_relative_path] = "d" * 64
    return replace(bundle, hashes=hashes)


def _provenance_hash_mismatch(bundle: _FakeBundle) -> _FakeBundle:
    hashes = dict(bundle.hashes)
    hashes[bundle.manifest.provenance_receipt_relative_path] = "e" * 64
    return replace(bundle, hashes=hashes)


def _manifest_model_mismatch(bundle: _FakeBundle) -> _FakeBundle:
    manifest = SimpleNamespace(**vars(bundle.manifest))
    manifest.text_recognition_model_name = "en_PP-OCRv5_mobile_rec"
    return replace(bundle, manifest=manifest)


@pytest.mark.parametrize(
    "mismatch",
    [
        _bundle_root_mismatch,
        _inventory_hash_mismatch,
        _provenance_hash_mismatch,
        _manifest_model_mismatch,
    ],
    ids=["release-root", "inventory-hash", "provenance-hash", "model-name"],
)
def test_v18_runtime_verifier_accepts_exact_loader_and_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: Callable[[_FakeBundle], _FakeBundle],
) -> None:
    fixture = _prepare_fixture(tmp_path=tmp_path, monkeypatch=monkeypatch)
    calls: list[tuple[Path, str]] = []

    def exact_loader(*, document_preview_root: Path, manifest_sha256: str) -> _FakeBundle:
        calls.append((document_preview_root, manifest_sha256))
        return fixture.server

    monkeypatch.setattr(
        runtime,
        "load_verified_raster_text_direct_server_paddle_bundle_v1",
        exact_loader,
    )
    runtime._verify_v18_server_model_bundle(  # noqa: SLF001
        root=fixture.root,
        profile=fixture.profile,
    )
    assert calls == [(fixture.root, _SERVER_SHA256)]

    monkeypatch.setattr(
        runtime,
        "load_verified_raster_text_direct_server_paddle_bundle_v1",
        lambda **_kwargs: mismatch(fixture.server),
    )
    with pytest.raises(RasterTextRuntimeError) as raised:
        runtime._verify_v18_server_model_bundle(  # noqa: SLF001
            root=fixture.root,
            profile=fixture.profile,
        )
    assert str(raised.value) == "raster_text_server_model_bundle_invalid"


def _passing_v6_scores() -> list[RasterTextSealedProfileScoreV6]:
    return [
        RasterTextSealedProfileScoreV6(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=0.99,
        )
        for document_type in (
            "bill_of_lading",
            "commercial_invoice",
            "packing_list",
        )
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]


def test_v6_pointer_calibration_and_evaluation_parsers_share_one_evidence_generation() -> None:
    profile_revision = _PROFILE_REVISION_PREFIX + "f" * 24
    configuration_sha256 = "1" * 64
    pointer = RasterTextActivationPointerV6(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
        raster_text_profile_revision=profile_revision,
        runtime_configuration_path="profiles/final-v18.json",
        runtime_configuration_sha256=configuration_sha256,
        sealed_evaluation_path="fresh-staged-evaluations-v6/final.json",
        sealed_evaluation_sha256="2" * 64,
    )
    calibration = RasterTextCalibrationReceiptV6(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="calibration-10-family-v1",
        family_count=10,
        asset_count=240,
        raster_text_profile_revision=profile_revision,
        runtime_configuration_sha256=configuration_sha256,
        calibration_raster_text_profile_revision=_PROFILE_REVISION_PREFIX + "e" * 24,
        calibration_runtime_configuration_path="profiles/calibration-v18.json",
        calibration_runtime_configuration_sha256="3" * 64,
        extractor_revision="direct-paddle-test-v18",
        selection_receipt_path="fresh-selections-v2/selection.json",
        selection_receipt_sha256="4" * 64,
        projection_receipt_path="fresh-projections-v2/calibration.json",
        projection_receipt_sha256="5" * 64,
        source_gold_receipt_path="fresh-gold-v2/calibration.json",
        source_gold_receipt_sha256="6" * 64,
        candidate_execution_path="fresh-executions-v2/calibration.json",
        candidate_execution_sha256="7" * 64,
        minimum_page_confidence=0.42,
        latency_p95_seconds=9.0,
    )
    evaluation = RasterTextSealedEvaluationV6(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="sealed-stage1-50-family-v1",
        stage1_decision="early_pass",
        family_count=50,
        asset_count=1_200,
        raster_text_profile_revision=profile_revision,
        runtime_configuration_sha256=configuration_sha256,
        extractor_revision="direct-paddle-test-v18",
        selection_receipt_path=calibration.selection_receipt_path,
        selection_receipt_sha256=calibration.selection_receipt_sha256,
        projection_receipt_path="fresh-projections-v2/sealed-stage1.json",
        projection_receipt_sha256="8" * 64,
        stage1_decision_receipt_path="fresh-stage-decisions-v2/stage1.json",
        stage1_decision_receipt_sha256="9" * 64,
        source_gold_receipt_path="fresh-gold-v2/sealed.json",
        source_gold_receipt_sha256="a" * 64,
        calibration_receipt_path="fresh-calibration-receipts-v6/calibration.json",
        calibration_receipt_sha256="b" * 64,
        candidate_execution_path="fresh-executions-v2/sealed-stage1.json",
        candidate_execution_sha256="c" * 64,
        warm_service_smoke_receipt_path="warm-smoke-v18/final.json",
        warm_service_smoke_receipt_sha256="d" * 64,
        latency_p95_seconds=9.0,
        service_warm_latency_p95_seconds=9.0,
        minimum_page_confidence=0.42,
        scores=_passing_v6_scores(),
    )

    assert runtime.parse_raster_text_activation_pointer(
        _canonical_pointer(pointer.model_dump(mode="json", warnings=False))
    ) == pointer
    assert runtime._parse_raster_text_calibration_receipt(  # noqa: SLF001
        _canonical(calibration.model_dump(mode="json", warnings=False))
    ) == calibration
    assert runtime._parse_raster_text_sealed_evaluation(  # noqa: SLF001
        _canonical(evaluation.model_dump(mode="json", warnings=False))
    ) == evaluation
    assert runtime._bound_raster_text_metric_revision(  # noqa: SLF001
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
    ) == "source-attested-semantic-text-v3"

    wrong_generation = pointer.model_dump(mode="json", warnings=False)
    wrong_generation["evidence_revision"] = "source-attested-fresh-staged-holdout-v2"
    with pytest.raises(RasterTextRuntimeError, match="activation_invalid"):
        runtime.parse_raster_text_activation_pointer(_canonical_pointer(wrong_generation))
