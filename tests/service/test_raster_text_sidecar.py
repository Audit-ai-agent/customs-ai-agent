"""Focused protocol tests for the persistent direct-text sidecar child."""

from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import socket
import struct
from types import SimpleNamespace
from typing import BinaryIO, Callable

import pytest

import customs_ai_agent.service.raster_text_sidecar as sidecar


_REVISION = "synthetic-raster-text-v1:" + "a" * 24
_SHA = "a" * 64


def _profile() -> sidecar._Profile:
    return sidecar._Profile(
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v1",
        sidecar_required_packages=("paddleocr", "paddlex", "paddlepaddle"),
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_directory="text-det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_recognition_model_directory="text-rec",
        cpu_threads=6,
        text_detection_limit_side_len=2048,
        text_detection_limit_type="max",
        text_detection_threshold=0.3,
        text_detection_box_threshold=0.6,
        text_detection_unclip_ratio=1.5,
    )


def _request_payload(*, source_sha256: str = _SHA, revision: str = _REVISION) -> bytes:
    header = {
        "protocol": "customs-raster-text-sidecar-v1",
        "source_sha256": source_sha256,
        "raster_text_profile_revision": revision,
        "pages": [{"page_number": 1, "width": 1, "height": 1}],
    }
    return (
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n\x00\x00\x00"
    )


def _frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


def _frames(raw: bytes) -> list[dict[str, object]]:
    stream = io.BytesIO(raw)
    frames: list[dict[str, object]] = []
    while header := stream.read(8):
        assert len(header) == 8
        length = struct.unpack(">Q", header)[0]
        payload = stream.read(length)
        assert len(payload) == length
        frames.append(json.loads(payload))
    return frames


def _install_valid_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[sidecar._Profile, Path, list[str]]:
    profile = _profile()
    model_root = tmp_path / "models"
    model_root.mkdir()
    calls: list[str] = []

    def configuration() -> tuple[sidecar._Profile, Path]:
        calls.append("verify")
        return profile, model_root

    monkeypatch.setattr(sidecar, "configuration_from_environment", configuration)
    return profile, model_root, calls


def _install_service_material_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: sidecar._Profile
) -> tuple[Path, Path]:
    """Install a small real model inventory for warm-request drift tests."""

    configuration = tmp_path / "profile.json"
    configuration_raw = b"profile"
    configuration.write_bytes(configuration_raw)
    package_inventory = tmp_path / "packages.json"
    package_inventory_raw = b"packages"
    package_inventory.write_bytes(package_inventory_raw)
    model_root = tmp_path / "models"
    detection_model = model_root / "text-det" / "model.bin"
    recognition_model = model_root / "text-rec" / "model.bin"
    detection_model.parent.mkdir(parents=True)
    recognition_model.parent.mkdir(parents=True)
    detection_model.write_bytes(b"detector-model")
    recognition_model.write_bytes(b"recognizer-model")
    model_inventory = tmp_path / "models.json"
    model_inventory_raw = json.dumps(
        {
            "artifacts": [
                {
                    "relative_path": "text-det/model.bin",
                    "sha256": hashlib.sha256(detection_model.read_bytes()).hexdigest(),
                },
                {
                    "relative_path": "text-rec/model.bin",
                    "sha256": hashlib.sha256(recognition_model.read_bytes()).hexdigest(),
                },
            ]
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    model_inventory.write_bytes(model_inventory_raw)
    module_path = Path(sidecar.__file__)
    for name, value in {
        "CUSTOMS_AGENT_RASTER_TEXT_CONFIG_PATH": str(configuration),
        "CUSTOMS_AGENT_RASTER_TEXT_CONFIG_SHA256": hashlib.sha256(configuration_raw).hexdigest(),
        "CUSTOMS_AGENT_RASTER_TEXT_SIDECAR_MODULE_SHA256": hashlib.sha256(
            module_path.read_bytes()
        ).hexdigest(),
        "CUSTOMS_AGENT_RASTER_TEXT_REQUIRED_PACKAGES": json.dumps(
            list(profile.sidecar_required_packages)
        ),
        "CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_PATH": str(package_inventory),
        "CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_SHA256": hashlib.sha256(
            package_inventory_raw
        ).hexdigest(),
        "CUSTOMS_AGENT_RASTER_TEXT_MODEL_ROOT": str(model_root),
        "CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_PATH": str(model_inventory),
        "CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_SHA256": hashlib.sha256(
            model_inventory_raw
        ).hexdigest(),
    }.items():
        monkeypatch.setenv(name, value)
    # This helper deliberately exercises the real model verifier while
    # avoiding external Paddle package installation in the parent test venv.
    monkeypatch.setattr(sidecar, "_profile_from_configuration", lambda _raw: profile)
    monkeypatch.setattr(sidecar, "_verify_package_inventory", lambda *_args, **_kwargs: None)
    return model_root, detection_model


def test_service_protocol_ready_then_serially_processes_framed_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile, _model_root, verification_calls = _install_valid_sidecar(monkeypatch, tmp_path)
    engine = object()
    prediction_calls: list[int] = []
    service_checks: list[str] = []
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: engine)
    monkeypatch.setattr(
        sidecar,
        "_verify_service_materials",
        lambda **_kwargs: service_checks.append("checked"),
    )

    def predict(observed_engine: object, pages: object) -> list[dict[str, object]]:
        assert observed_engine is engine
        prediction_calls.append(len(pages))  # type: ignore[arg-type]
        return [{"page_number": 1, "text": "OK", "confidence": 0.9}]

    monkeypatch.setattr(sidecar, "_predict_pages", predict)
    output = io.BytesIO()
    sidecar.run_service_from_stdio(
        io.BytesIO(_frame(_request_payload()) + _frame(_request_payload(source_sha256="b" * 64))),
        output,
    )

    frames = _frames(output.getvalue())
    assert frames[0] == {
        "protocol": "customs-raster-text-sidecar-v1-service",
        "status": "ready",
        "raster_text_profile_revision": profile.raster_text_profile_revision,
        "provider_revision": profile.extractor_revision,
    }
    assert [frame["source_sha256"] for frame in frames[1:]] == [_SHA, "b" * 64]
    assert prediction_calls == [1, 1]
    # The initial configuration check is followed by a second complete check
    # after engine construction, then pre/post checks for each request.
    assert verification_calls == ["verify"]
    assert service_checks == ["checked"] * 5


def test_evaluation_session_ready_precedes_candidates_and_rechecks_materials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile, _model_root, verification_calls = _install_valid_sidecar(monkeypatch, tmp_path)
    engine = object()
    material_checks: list[str] = []
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: engine)
    monkeypatch.setattr(
        sidecar,
        "_verify_service_materials",
        lambda **_kwargs: material_checks.append("checked"),
    )

    def predict(observed_engine: object, _pages: object) -> list[dict[str, object]]:
        assert observed_engine is engine
        return [{"page_number": 1, "text": "OK", "confidence": 0.9}]

    monkeypatch.setattr(sidecar, "_predict_pages", predict)
    output = io.BytesIO()

    sidecar.run_session_from_stdio(io.BytesIO(_frame(_request_payload())), output)

    frames = _frames(output.getvalue())
    assert frames[0] == {
        "protocol": "customs-raster-text-sidecar-v1-session",
        "status": "ready",
        "raster_text_profile_revision": profile.raster_text_profile_revision,
        "provider_revision": profile.extractor_revision,
    }
    assert frames[1]["source_sha256"] == _SHA
    # Configuration/model setup is rechecked after engine construction and
    # before the content-free ready acknowledgement.  Every later candidate
    # has independent pre/post drift checks while the engine is reused.
    assert verification_calls == ["verify"]
    assert material_checks == ["checked", "checked", "checked"]


def test_evaluation_session_post_prediction_drift_withholds_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_valid_sidecar(monkeypatch, tmp_path)
    checks = 0

    def changing_material_check(**_kwargs: object) -> None:
        nonlocal checks
        checks += 1
        # startup post-engine, request pre, request post
        if checks == 3:
            raise sidecar.RasterTextSidecarError("raster_text_sidecar_model_drift")

    monkeypatch.setattr(sidecar, "_verify_service_materials", changing_material_check)
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: object())
    monkeypatch.setattr(
        sidecar,
        "_predict_pages",
        lambda *_args: [{"page_number": 1, "text": "OK", "confidence": 0.9}],
    )
    output = io.BytesIO()

    with pytest.raises(sidecar.RasterTextSidecarError, match="model_drift"):
        sidecar.run_session_from_stdio(io.BytesIO(_frame(_request_payload())), output)

    # The ready acknowledgement has no source content.  The stale OCR result
    # is not emitted once the child detects changed model material after the
    # native prediction.
    assert checks == 3
    assert len(_frames(output.getvalue())) == 1


def test_service_material_check_rehashes_model_bytes_and_rejects_extras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _profile()
    model_root, detection_model = _install_service_material_environment(
        monkeypatch, tmp_path, profile
    )
    observed: list[str] = []
    original = sidecar._verify_model_inventory

    def verify_inventory(**kwargs: object) -> None:
        observed.append("full-tree")
        original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sidecar, "_verify_model_inventory", verify_inventory)

    sidecar._verify_service_materials(profile=profile, model_root=model_root)
    detection_model.write_bytes(b"changed-detector-model")
    with pytest.raises(sidecar.RasterTextSidecarError, match="model_drift"):
        sidecar._verify_service_materials(profile=profile, model_root=model_root)

    detection_model.write_bytes(b"detector-model")
    (model_root / "undeclared.bin").write_bytes(b"extra")
    with pytest.raises(sidecar.RasterTextSidecarError, match="model_drift"):
        sidecar._verify_service_materials(profile=profile, model_root=model_root)

    assert observed == ["full-tree", "full-tree", "full-tree"]


def test_service_post_prediction_model_drift_withholds_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _profile()
    model_root, detection_model = _install_service_material_environment(
        monkeypatch, tmp_path, profile
    )
    material_checks: list[str] = []
    original = sidecar._verify_model_inventory

    def verify_inventory(**kwargs: object) -> None:
        material_checks.append("full-tree")
        original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sidecar, "_verify_model_inventory", verify_inventory)
    monkeypatch.setattr(sidecar, "configuration_from_environment", lambda: (profile, model_root))
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: object())

    def predict_then_change_model(*_args: object) -> list[dict[str, object]]:
        detection_model.write_bytes(b"changed-after-prediction")
        return [{"page_number": 1, "text": "OK", "confidence": 0.9}]

    monkeypatch.setattr(sidecar, "_predict_pages", predict_then_change_model)
    output = io.BytesIO()

    with pytest.raises(sidecar.RasterTextSidecarError, match="model_drift"):
        sidecar.run_service_from_stdio(io.BytesIO(_frame(_request_payload())), output)

    # The complete verifier ran after engine construction and at both request
    # edges.  The stale OCR response was never framed after byte drift.
    assert material_checks == ["full-tree", "full-tree", "full-tree"]
    assert len(_frames(output.getvalue())) == 1


def test_service_ready_is_not_emitted_when_engine_load_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_valid_sidecar(monkeypatch, tmp_path)

    def unavailable(*_args: object) -> object:
        raise sidecar.RasterTextSidecarError("private")

    monkeypatch.setattr(sidecar, "_engine", unavailable)
    output = io.BytesIO()
    with pytest.raises(sidecar.RasterTextSidecarError):
        sidecar.run_service_from_stdio(io.BytesIO(), output)
    assert output.getvalue() == b""


@pytest.mark.parametrize(
    "runner",
    [sidecar.run_session_from_stdio, sidecar.run_service_from_stdio],
    ids=["evaluation", "service"],
)
def test_ready_is_withheld_when_material_drifts_during_engine_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: Callable[[BinaryIO, BinaryIO], None],
) -> None:
    _install_valid_sidecar(monkeypatch, tmp_path)
    order: list[str] = []

    def engine(*_args: object) -> object:
        order.append("engine")
        return object()

    def changed_material(**_kwargs: object) -> None:
        order.append("post-engine-check")
        raise sidecar.RasterTextSidecarError("raster_text_sidecar_model_drift")

    monkeypatch.setattr(sidecar, "_engine", engine)
    monkeypatch.setattr(sidecar, "_verify_service_materials", changed_material)
    output = io.BytesIO()

    with pytest.raises(sidecar.RasterTextSidecarError, match="model_drift"):
        runner(io.BytesIO(), output)

    assert order == ["engine", "post-engine-check"]
    assert output.getvalue() == b""


def test_service_rechecks_material_after_prediction_and_withholds_stale_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_valid_sidecar(monkeypatch, tmp_path)
    observations = 0

    def changing_material_check(**_kwargs: object) -> None:
        nonlocal observations
        observations += 1
        # startup post-engine, request-pre, request-post
        if observations == 3:
            raise sidecar.RasterTextSidecarError("raster_text_sidecar_binding_invalid")

    monkeypatch.setattr(sidecar, "_verify_service_materials", changing_material_check)
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: object())
    monkeypatch.setattr(
        sidecar,
        "_predict_pages",
        lambda *_args: [{"page_number": 1, "text": "OK", "confidence": 0.9}],
    )
    output = io.BytesIO()

    with pytest.raises(sidecar.RasterTextSidecarError, match="binding_invalid"):
        sidecar.run_service_from_stdio(io.BytesIO(_frame(_request_payload())), output)

    # A ready acknowledgement is safe; the stale OCR response was never
    # framed once post-prediction material verification detected a change.
    assert len(_frames(output.getvalue())) == 1


def test_service_malformed_frame_is_a_private_process_failure_after_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_valid_sidecar(monkeypatch, tmp_path)
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: object())
    monkeypatch.setattr(sidecar, "_verify_service_materials", lambda **_kwargs: None)
    output = io.BytesIO()

    with pytest.raises(sidecar.RasterTextSidecarError, match="input_invalid"):
        sidecar.run_service_from_stdio(io.BytesIO(b"\x00"), output)

    assert len(_frames(output.getvalue())) == 1


def test_main_returns_nonzero_for_malformed_service_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_valid_sidecar(monkeypatch, tmp_path)
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: object())
    monkeypatch.setattr(sidecar.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"\x00")))
    monkeypatch.setattr(sidecar.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))

    assert sidecar.main(["--customs-raster-text-sidecar-v1-service"]) == 1


def test_service_engine_load_remains_inside_network_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_valid_sidecar(monkeypatch, tmp_path)

    def engine_with_forbidden_connection(*_args: object) -> object:
        with pytest.raises(sidecar.RasterTextSidecarError, match="network_forbidden"):
            socket.create_connection(("127.0.0.1", 9))
        return object()

    monkeypatch.setattr(sidecar, "_engine", engine_with_forbidden_connection)
    monkeypatch.setattr(sidecar, "_verify_service_materials", lambda **_kwargs: None)
    sidecar.run_service_from_stdio(io.BytesIO(), io.BytesIO())


def test_main_selects_service_protocol_without_replacing_evaluation_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(sidecar, "run_once_from_stdio", lambda *_args: calls.append("once"))
    monkeypatch.setattr(
        sidecar, "run_session_from_stdio", lambda *_args: calls.append("evaluation")
    )
    monkeypatch.setattr(sidecar, "run_service_from_stdio", lambda *_args: calls.append("service"))
    monkeypatch.setattr(sidecar.sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(sidecar.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))

    assert sidecar.main(["--customs-raster-text-sidecar-v1-session"]) == 0
    assert sidecar.main(["--customs-raster-text-sidecar-v1-service"]) == 0
    assert calls == ["evaluation", "service"]
