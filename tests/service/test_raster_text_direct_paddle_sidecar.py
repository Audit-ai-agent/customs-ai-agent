"""Unit coverage for the narrow direct static-inference text POC."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from typing import Any, cast

import pytest

import customs_ai_agent.service.raster_text_direct_paddle_sidecar as sidecar


_REVISION = "synthetic-raster-text-v1:" + "a" * 24
_SHA = "a" * 64


def _profile() -> sidecar.DirectPaddleProfile:
    return sidecar.DirectPaddleProfile(
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-paddle-poc-test-v1",
        text_detection_model_directory="text-det",
        text_recognition_model_directory="text-rec",
        cpu_threads=1,
        text_detection_limit_side_len=2048,
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
    return len(payload).to_bytes(8, byteorder="big", signed=False) + payload


def _frames(value: bytes) -> list[dict[str, object]]:
    stream = io.BytesIO(value)
    result: list[dict[str, object]] = []
    while header := stream.read(8):
        assert len(header) == 8
        length = int.from_bytes(header, byteorder="big", signed=False)
        payload = stream.read(length)
        assert len(payload) == length
        result.append(json.loads(payload))
    return result


def _configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": "paddleocr3_direct_text",
        "runtime_api": "paddleocr3-paddlex-3.2",
        "extractor_revision": "direct-paddle-poc-test-v1",
        "raster_text_profile_revision": _REVISION,
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_directory": "text-rec",
        "cpu_threads": 1,
        "text_detection_limit_side_len": 2048,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.6,
        "text_detection_unclip_ratio": 1.5,
    }


def _write_model_inventory(root: Path, *, model_spec: str = "server_det") -> tuple[bytes, str]:
    expected = (
        sidecar._DIRECT_MODEL_ARTIFACTS
        if model_spec == "server_det"
        else sidecar._DIRECT_MOBILE_MODEL_ARTIFACTS
    )
    artifacts: list[dict[str, str]] = []
    for relative in sorted(expected):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("ascii"))
        artifacts.append(
            {"relative_path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    raw = json.dumps({"artifacts": artifacts}, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return raw, hashlib.sha256(raw).hexdigest()


def test_direct_profile_accepts_only_matching_engine_and_runtime_pair() -> None:
    raw = json.dumps(_configuration()).encode("utf-8")
    assert sidecar._profile_from_configuration(raw) == _profile()
    assert sidecar._profile_from_configuration(raw).row_band_mode == "legacy_max"

    v2 = _configuration()
    v2.update(
        {
            "schema_version": "2.0",
            "engine_kind": "direct_paddle_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_limit_side_len": 960,
        }
    )
    parsed_v2 = sidecar._profile_from_configuration(json.dumps(v2).encode("utf-8"))
    assert parsed_v2.text_detection_limit_side_len == 960
    assert parsed_v2.row_band_mode == "legacy_max"
    assert parsed_v2.pre_ready_warmup

    v2["text_detection_limit_side_len"] = 1024
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(v2).encode("utf-8"))

    bounded = _configuration()
    bounded["text_detection_limit_side_len"] = 1024
    assert (
        sidecar._profile_from_configuration(
            json.dumps(bounded).encode("utf-8")
        ).text_detection_limit_side_len
        == 1024
    )

    invalid = _configuration()
    invalid["runtime_api"] = "paddle-inference-static-v1"
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(invalid).encode("utf-8"))


def test_v3_mobile_profile_accepts_only_the_mobile_detector_pair() -> None:
    v3 = _configuration()
    v3.update(
        {
            "schema_version": "3.0",
            "engine_kind": "direct_paddle_mobile_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v3).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.text_detection_limit_side_len == 960
    assert profile.row_band_mode == "legacy_max"
    assert profile.pre_ready_warmup

    v3_with_harmonic_property = {**v3, "row_band_mode": "harmonic_v1"}
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(v3_with_harmonic_property).encode("utf-8"))

    v3["text_detection_model_name"] = "PP-OCRv5_server_det"
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(v3).encode("utf-8"))

    v2 = _configuration()
    v2.update(
        {
            "schema_version": "2.0",
            "engine_kind": "direct_paddle_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
        }
    )
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(v2).encode("utf-8"))


def test_v4_harmonic_profile_accepts_only_its_exact_mobile_tuple() -> None:
    v4 = _configuration()
    v4.update(
        {
            "schema_version": "4.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v4).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.cpu_threads == 6

    for key, invalid in (
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("engine_kind", "direct_paddle_mobile_text"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v4, key: invalid}).encode("utf-8"))


def test_v5_harmonic_recall_profile_accepts_only_its_lower_threshold_tuple() -> None:
    v5 = _configuration()
    v5.update(
        {
            "schema_version": "5.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.4,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v5).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.4

    for key, invalid in (
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.6),
        ("engine_kind", "direct_paddle_mobile_text_harmonic"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v5, key: invalid}).encode("utf-8"))

    # V1 through V4 cannot opt into V5 semantics merely by adding its engine
    # spelling or the lower threshold to an otherwise old profile tuple.
    old_profiles = [
        _configuration(),
        {
            **_configuration(),
            "schema_version": "2.0",
            "engine_kind": "direct_paddle_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_limit_side_len": 960,
        },
        {
            **_configuration(),
            "schema_version": "3.0",
            "engine_kind": "direct_paddle_mobile_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
        },
        {
            **_configuration(),
            "schema_version": "4.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.6,
        },
    ]
    for old in old_profiles:
        incompatible = {
            **old,
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall",
            "text_detection_box_threshold": 0.4,
        }
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(incompatible).encode("utf-8"))


def test_v6_1024_profile_keeps_v5_harmonic_recall_semantics() -> None:
    v6 = _configuration()
    v6.update(
        {
            "schema_version": "6.0",
            "engine_kind": "direct_paddle_mobile_text_1024",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 1024,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.4,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v6).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.4
    assert profile.text_detection_unclip_ratio == 1.5

    # V6 is V5 plus its 1024 detector limit, not a permissive variant.  A
    # model, timeout, threshold, ordering engine, or detector-scale change is
    # rejected before native Paddle imports.
    for key, invalid in (
        ("text_detection_limit_side_len", 960),
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.6),
        ("text_detection_unclip_ratio", 2.0),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v6, key: invalid}).encode("utf-8"))

    # No V1--V5 schema can acquire V6 semantics by borrowing its otherwise
    # compatible values.  The schema/engine tuple stays the immutable gate.
    old_profiles = [
        _configuration(),
        {
            **_configuration(),
            "schema_version": "2.0",
            "engine_kind": "direct_paddle_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_limit_side_len": 960,
        },
        {
            **_configuration(),
            "schema_version": "3.0",
            "engine_kind": "direct_paddle_mobile_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
        },
        {
            **_configuration(),
            "schema_version": "4.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.6,
        },
        {
            **_configuration(),
            "schema_version": "5.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.4,
        },
    ]
    for old in old_profiles:
        incompatible = {
            **old,
            "engine_kind": "direct_paddle_mobile_text_1024",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_detection_limit_side_len": 1024,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.4,
            "text_detection_unclip_ratio": 1.5,
        }
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(incompatible).encode("utf-8"))


def test_v7_low_box_profile_accepts_only_its_exact_1024_harmonic_tuple() -> None:
    v7 = _configuration()
    v7.update(
        {
            "schema_version": "7.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 1024,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v7).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    for key, invalid in (
        ("text_detection_limit_side_len", 960),
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_box_threshold", 0.6),
        ("text_detection_unclip_ratio", 2.0),
        ("engine_kind", "direct_paddle_mobile_text_1024"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v7, key: invalid}).encode("utf-8"))

    # A .3 box threshold is valid only for V7.  Each otherwise valid legacy
    # tuple rejects it before the sidecar imports native Paddle components.
    legacy_with_low_box = [
        {**_configuration(), "text_detection_box_threshold": 0.3},
        {
            **_configuration(),
            "schema_version": "2.0",
            "engine_kind": "direct_paddle_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_limit_side_len": 960,
            "text_detection_box_threshold": 0.3,
        },
        {
            **_configuration(),
            "schema_version": "3.0",
            "engine_kind": "direct_paddle_mobile_text",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "text_detection_box_threshold": 0.3,
        },
        {
            **_configuration(),
            "schema_version": "4.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
        },
        {
            **_configuration(),
            "schema_version": "5.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
        },
        {
            **_configuration(),
            "schema_version": "6.0",
            "engine_kind": "direct_paddle_mobile_text_1024",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 1024,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
        },
    ]
    for legacy in legacy_with_low_box:
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(legacy).encode("utf-8"))


def test_v8_low_box_profile_accepts_only_its_exact_native_960_harmonic_tuple() -> None:
    v8 = _configuration()
    v8.update(
        {
            "schema_version": "8.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v8).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_box_threshold", 0.6),
        ("text_detection_unclip_ratio", 2.0),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall_1024_lowbox"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v8, key: invalid}).encode("utf-8"))

    # V1--V7 cannot reinterpret the V8 native-960 low-box semantics.  V7
    # retains its own valid .3 value, but must reject V8's 960-side tuple.
    earlier_tuples = (
        ("1.0", "paddleocr3_direct_text", "paddleocr3-paddlex-3.2"),
        ("2.0", "direct_paddle_text", "paddle-inference-static-v1"),
        ("3.0", "direct_paddle_mobile_text", "paddle-inference-static-v1"),
        ("4.0", "direct_paddle_mobile_text_harmonic", "paddle-inference-static-v1"),
        (
            "5.0",
            "direct_paddle_mobile_text_harmonic_recall",
            "paddle-inference-static-v1",
        ),
        ("6.0", "direct_paddle_mobile_text_1024", "paddle-inference-static-v1"),
        (
            "7.0",
            "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
            "paddle-inference-static-v1",
        ),
    )
    for schema_version, engine_kind, runtime_api in earlier_tuples:
        incompatible = {
            **v8,
            "schema_version": schema_version,
            "engine_kind": engine_kind,
            "runtime_api": runtime_api,
        }
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(incompatible).encode("utf-8"))


def test_v9_strip2_profile_accepts_only_its_exact_native_960_tuple() -> None:
    v9 = _configuration()
    v9.update(
        {
            "schema_version": "9.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v9).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.detector_pass_mode == "strip2"
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
        ("detector_pass_mode", "single"),
        ("strip_overlap", 12),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v9, key: invalid}).encode("utf-8"))

    # V1--V8 cannot borrow V9's new engine spelling.  Keeping all V9
    # material and changing only a legacy schema makes every exact tuple fail
    # before package/model imports.
    for schema_version in ("1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(
                json.dumps({**v9, "schema_version": schema_version}).encode("utf-8")
            )


def test_v9_strip_geometry_mapping_and_cross_tile_nms_are_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sidecar._strip2_horizontal_tiles(100) == ((0, 55), (45, 100))
    assert sidecar._strip2_horizontal_tiles(101) == ((0, 56), (45, 101))
    assert sidecar._offset_quad_points_y(
        box=((1.0, 2.0), (5.0, 2.0), (5.0, 6.0), (1.0, 6.0)), y_offset=45
    ) == ((1.0, 47.0), (5.0, 47.0), (5.0, 51.0), (1.0, 51.0))

    duplicate = ((0.0, 46.0), (10.0, 46.0), (10.0, 54.0), (0.0, 54.0))
    unique_same_top_tile = ((20.0, 46.0), (30.0, 46.0), (30.0, 54.0), (20.0, 54.0))
    result = sidecar._strip2_cross_tile_nms(
        (
            sidecar._StripDetectedBox(box=duplicate, score=0.9, tile_index=0, ordinal=0),
            sidecar._StripDetectedBox(box=unique_same_top_tile, score=0.8, tile_index=0, ordinal=1),
            sidecar._StripDetectedBox(box=duplicate, score=0.7, tile_index=1, ordinal=0),
        ),
        overlap_start=45.0,
        overlap_end=55.0,
    )
    assert result == [duplicate, unique_same_top_tile]

    # A cross-tile pair away from the detector overlap cannot be duplicate
    # work.  The fast vertical guards must avoid an expensive polygon clip.
    iou_calls: list[object] = []
    monkeypatch.setattr(
        sidecar,
        "_rotated_quad_iou",
        lambda *_args: iou_calls.append(object()) or 1.0,
    )
    non_overlap = sidecar._strip2_cross_tile_nms(
        (
            sidecar._StripDetectedBox(
                box=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
                score=0.9,
                tile_index=0,
                ordinal=0,
            ),
            sidecar._StripDetectedBox(
                box=((0.0, 80.0), (10.0, 80.0), (10.0, 90.0), (0.0, 90.0)),
                score=0.8,
                tile_index=1,
                ordinal=0,
            ),
        ),
        overlap_start=45.0,
        overlap_end=55.0,
    )
    assert len(non_overlap) == 2
    assert iou_calls == []

    # The production cap is deliberately much larger; a small monkeypatched
    # bound proves overflow fails closed rather than silently dropping a box.
    monkeypatch.setattr(sidecar, "_STRIP2_MAX_IOU_COMPARISONS", 1)
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="output_invalid"):
        sidecar._strip2_cross_tile_nms(
            (
                sidecar._StripDetectedBox(box=duplicate, score=0.9, tile_index=0, ordinal=0),
                sidecar._StripDetectedBox(
                    box=((20.0, 46.0), (30.0, 46.0), (30.0, 54.0), (20.0, 54.0)),
                    score=0.8,
                    tile_index=1,
                    ordinal=0,
                ),
                sidecar._StripDetectedBox(
                    box=((40.0, 46.0), (50.0, 46.0), (50.0, 54.0), (40.0, 54.0)),
                    score=0.7,
                    tile_index=1,
                    ordinal=1,
                ),
            ),
            overlap_start=45.0,
            overlap_end=55.0,
        )


def test_v9_strip2_detector_runs_exact_top_and_bottom_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        shape = (100, 20, 3)

        def __getitem__(self, key: object) -> SimpleNamespace:
            assert isinstance(key, tuple)
            assert isinstance(key[0], slice)
            tile_slices.append((key[0].start, key[0].stop))
            return SimpleNamespace(tile_index=len(tile_slices) - 1)

    tile_slices: list[tuple[int | None, int | None]] = []
    detector_inputs: list[int] = []
    predictor_calls: list[object] = []
    nms_calls: list[tuple[float, float, int]] = []
    engine = SimpleNamespace(detector=object(), np=object())

    def detector_input(*, image: SimpleNamespace, **_kwargs: object) -> tuple[object, int, int]:
        detector_inputs.append(image.tile_index)
        return object(), 20, 55

    monkeypatch.setattr(sidecar, "_detector_input", detector_input)
    monkeypatch.setattr(
        sidecar,
        "_run_predictor",
        lambda **_kwargs: predictor_calls.append(object()) or object(),
    )
    monkeypatch.setattr(sidecar, "_probability_map", lambda **_kwargs: object())
    monkeypatch.setattr(
        sidecar,
        "_detected_scored_boxes",
        lambda *, tile_index, **_kwargs: [
            sidecar._StripDetectedBox(
                box=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)),
                score=0.9,
                tile_index=tile_index,
                ordinal=0,
            )
        ],
    )
    monkeypatch.setattr(sidecar, "_offset_detected_box_y", lambda **kwargs: kwargs["box"])
    monkeypatch.setattr(
        sidecar,
        "_strip2_cross_tile_nms",
        lambda candidates, *, overlap_start, overlap_end: (
            nms_calls.append((overlap_start, overlap_end, len(candidates))) or []
        ),
    )

    assert sidecar._strip2_detected_boxes(image=_Image(), engine=engine) == []
    assert detector_inputs == [0, 1]
    assert tile_slices == [(0, 55), (45, 100)]
    assert len(predictor_calls) == 2
    assert nms_calls == [(45.0, 55.0, 2)]


def test_v9_predict_pages_uses_one_full_page_recognizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        def __getitem__(self, _key: object) -> "_Image":
            return self

    class _Pixels:
        def reshape(self, _shape: tuple[int, int, int]) -> _Image:
            return _Image()

    detector_calls: list[object] = []
    recognizer_calls: list[object] = []
    engine = SimpleNamespace(
        np=SimpleNamespace(frombuffer=lambda *_args, **_kwargs: _Pixels()),
        profile=SimpleNamespace(row_band_mode="harmonic_v1", detector_pass_mode="strip2"),
    )
    recognized = [_line(text="line", left=0, top=0, right=1, bottom=1)]
    monkeypatch.setattr(
        sidecar,
        "_strip2_detected_boxes",
        lambda **_kwargs: detector_calls.append(object()) or [],
    )
    monkeypatch.setattr(
        sidecar,
        "_recognize_crops",
        lambda **_kwargs: recognizer_calls.append(object()) or recognized,
    )
    monkeypatch.setattr(sidecar, "_ordered_lines", lambda lines, **_kwargs: list(lines))

    pages = sidecar._predict_pages(
        engine,
        (sidecar._Page(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00"),),
    )

    assert len(detector_calls) == 1
    assert len(recognizer_calls) == 1
    assert pages == [{"page_number": 1, "text": "line", "confidence": 0.9}]


def test_v10_grid2_profile_accepts_only_its_exact_native_960_tuple() -> None:
    v10 = _configuration()
    v10.update(
        {
            "schema_version": "10.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v10).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.detector_pass_mode == "grid2"
    assert profile.pre_ready_warmup
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
        ("detector_pass_mode", "single"),
        ("grid_overlap_x", 15),
        ("grid_overlap_y", 10),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v10, key: invalid}).encode("utf-8"))

    # V1--V9 cannot borrow V10's new engine spelling.  Changing only the
    # schema leaves V10 material in place and makes every legacy tuple fail.
    for schema_version in ("1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(
                json.dumps({**v10, "schema_version": schema_version}).encode("utf-8")
            )


def test_v11_guarded_grid2_profile_accepts_only_its_exact_native_960_tuple() -> None:
    v11 = _configuration()
    v11.update(
        {
            "schema_version": "11.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v11).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.detector_pass_mode == "guarded_grid2"
    assert profile.pre_ready_warmup
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
        ("detector_pass_mode", "grid2"),
        ("guarded_grid_internal_edge_epsilon", 1.0),
        ("guarded_grid_nms_iou", 0.5),
        ("guarded_grid_intersection_over_min_area", 0.8),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v11, key: invalid}).encode("utf-8"))

    # V1--V10 cannot borrow V11's guarded seam semantics by changing only a
    # schema discriminator.
    for schema_version in (
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "5.0",
        "6.0",
        "7.0",
        "8.0",
        "9.0",
        "10.0",
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(
                json.dumps({**v11, "schema_version": schema_version}).encode("utf-8")
            )


def test_v12_source_attested_recognizer_accepts_only_its_exact_strip2_tuple() -> None:
    v12 = _configuration()
    v12.update(
        {
            "schema_version": "12.0",
            "engine_kind": (
                "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "source_attested_tiny_crnn_bigru_ctc_v1",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v12).encode("utf-8"))
    assert profile.model_spec == "mobile_det_trained_rec"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.detector_pass_mode == "strip2"
    assert profile.pre_ready_warmup
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5
    assert profile.cpu_threads == 6

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_mobile_rec"),
        ("text_detection_model_directory", "text-rec"),
        ("text_recognition_model_directory", "text-det"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
        ("detector_pass_mode", "single"),
        ("strip_overlap", 12),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v12, key: invalid}).encode("utf-8"))

    # Neither side of the recognizer boundary may borrow the other's schema,
    # engine spelling, or model identity.  All mixed forms fail before any
    # package inventory or native Paddle import is reached.
    v9_engine = "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2"
    for historical_schema in (
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "5.0",
        "6.0",
        "7.0",
        "8.0",
        "9.0",
        "10.0",
        "11.0",
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(
                json.dumps({**v12, "schema_version": historical_schema}).encode("utf-8")
            )

    mixed_profiles = (
        {**v12, "engine_kind": v9_engine},
        {
            **v12,
            "schema_version": "9.0",
            "engine_kind": v9_engine,
        },
    )
    for mixed in mixed_profiles:
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(mixed).encode("utf-8"))


def test_v13_semantic_repair_accepts_only_the_exact_public_v9_model_tuple() -> None:
    v13 = _configuration()
    v13.update(
        {
            "schema_version": "13.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v13).encode("utf-8"))
    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v1"
    assert profile.pre_ready_warmup
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5
    assert profile.cpu_threads == 6

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "source_attested_tiny_crnn_bigru_ctc_v1"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
        ("semantic_repair_mode", "none"),
        ("detector_pass_mode", "single"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v13, key: invalid}).encode("utf-8"))

    for historical_schema in (
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "5.0",
        "6.0",
        "7.0",
        "8.0",
        "9.0",
        "10.0",
        "11.0",
        "12.0",
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(
                json.dumps({**v13, "schema_version": historical_schema}).encode("utf-8")
            )


def test_v14_semantic_repair_accepts_only_its_exact_public_v9_model_tuple() -> None:
    v14 = _configuration()
    v14.update(
        {
            "schema_version": "14.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v14).encode("utf-8"))

    assert profile.model_spec == "mobile_det"
    assert profile.row_band_mode == "harmonic_v1"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.pre_ready_warmup
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5
    assert profile.cpu_threads == 6

    for key, invalid in (
        ("schema_version", "13.0"),
        (
            "engine_kind",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
        ),
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("sidecar_timeout_seconds", 60),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "source_attested_tiny_crnn_bigru_ctc_v1"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
        ("semantic_repair_mode", "supported_docs_v1"),
        ("detector_pass_mode", "single"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v14, key: invalid}).encode("utf-8"))


def test_v15_accepts_only_bounded_hash_bound_onednn_controls() -> None:
    v15 = _configuration()
    v15.update(
        {
            "schema_version": "15.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v15).encode("utf-8"))

    assert profile.model_spec == "mobile_det"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.enable_mkldnn
    assert profile.mkldnn_cache_capacity == 10
    assert profile.cpu_threads == 6
    assert profile.recognition_batch_size == 8
    assert profile.opencv_threads == 1
    assert profile.warmup_shape_policy == "detector-736x960-recognizer-48x320-v1"
    assert profile.onednn_model_policy == "recognizer-only-v1"

    for key, invalid in (
        ("enable_mkldnn", False),
        ("mkldnn_cache_capacity", 0),
        ("mkldnn_cache_capacity", 11),
        ("cpu_threads", 8),
        ("recognition_batch_size", 4),
        ("opencv_threads", 0),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v15, key: invalid}).encode("utf-8"))

    historical = dict(v15)
    historical.update(
        {
            "schema_version": "14.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2"
            ),
        }
    )
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(historical).encode("utf-8"))

    for field, value in (
        ("recognition_crop_policy", "detector-quad-v1"),
        ("recognition_resize_interpolation", "cubic-v1"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v15, field: value}).encode("utf-8"))


def test_v16_accepts_only_hash_bound_cubic_recognition_policy() -> None:
    v16 = _configuration()
    v16.update(
        {
            "schema_version": "16.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn_cubic_recognition"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "recognition_crop_policy": "detector-quad-v1",
            "recognition_resize_interpolation": "cubic-v1",
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v16).encode("utf-8"))

    assert profile.model_spec == "mobile_det"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.enable_mkldnn
    assert profile.recognition_crop_policy == "detector-quad-v1"
    assert profile.recognition_resize_interpolation == "cubic-v1"

    for key, invalid in (
        ("recognition_crop_policy", "horizontal-padding"),
        ("recognition_resize_interpolation", "linear-v1"),
        ("recognition_resize_interpolation", "nearest-v1"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v16, key: invalid}).encode("utf-8"))

    for missing in ("recognition_crop_policy", "recognition_resize_interpolation"):
        incomplete = dict(v16)
        del incomplete[missing]
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(incomplete).encode("utf-8"))


def test_v17_accepts_only_hash_bound_horizontal_padding_policy() -> None:
    v17 = _configuration()
    v17.update(
        {
            "schema_version": "17.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn_padded_recognition"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "recognition_crop_policy": "horizontal-height-0p08-max4-v1",
            "recognition_resize_interpolation": "linear-v1",
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v17).encode("utf-8"))

    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.enable_mkldnn
    assert profile.recognition_crop_policy == "horizontal-height-0p08-max4-v1"
    assert profile.recognition_resize_interpolation == "linear-v1"

    for key, invalid in (
        ("recognition_crop_policy", "detector-quad-v1"),
        ("recognition_crop_policy", "horizontal-height-0p10-max8-v1"),
        ("recognition_resize_interpolation", "cubic-v1"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v17, key: invalid}).encode("utf-8"))


def test_v18_accepts_only_the_pinned_server_recognizer_ctc_contract() -> None:
    v18 = _configuration()
    v18.update(
        {
            "schema_version": "18.0",
            "engine_kind": (
                "direct_paddle_mobile_det_server_rec_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_server_rec",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "recognition_crop_policy": "detector-quad-v1",
            "recognition_resize_interpolation": "linear-v1",
            "ctc_label_policy": "append-space-v1",
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v18).encode("utf-8"))

    assert profile.model_spec == "mobile_det_server_rec"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.ctc_label_policy == "append-space-v1"

    for key, invalid in (
        ("text_recognition_model_name", "en_PP-OCRv5_mobile_rec"),
        ("recognition_batch_size", 16),
        ("recognition_crop_policy", "horizontal-height-0p08-max4-v1"),
        ("recognition_resize_interpolation", "cubic-v1"),
        ("ctc_label_policy", "optional-space-v1"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v18, key: invalid}).encode("utf-8"))

    incomplete = dict(v18)
    del incomplete["ctc_label_policy"]
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(incomplete).encode("utf-8"))


def test_v19_accepts_only_the_leakage_safe_public_finetuned_contract() -> None:
    v19 = _configuration()
    v19.update(
        {
            "schema_version": "19.0",
            "engine_kind": (
                "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
                "lowbox_strip2_semantic_repair_v2_onednn"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": ("source_attested_en_PP-OCRv5_mobile_rec_head_v2"),
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "ctc_label_policy": "optional-space-v1",
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v19).encode("utf-8"))

    assert profile.model_spec == "mobile_det_public_finetuned_rec"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.ctc_label_policy == "optional-space-v1"
    assert profile.voyage_retry_mode == "none"

    for key, invalid in (
        ("text_recognition_model_name", "en_PP-OCRv5_mobile_rec"),
        ("recognition_batch_size", 16),
        ("ctc_label_policy", "append-space-v1"),
        ("recognition_crop_policy", "detector-quad-v1"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v19, key: invalid}).encode("utf-8"))

    incomplete = dict(v19)
    del incomplete["ctc_label_policy"]
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(json.dumps(incomplete).encode("utf-8"))


def test_v19_voyage_retry_is_derived_only_from_its_exact_immutable_tuple() -> None:
    v19b = _configuration()
    v19b.update(
        {
            "schema_version": "19.0",
            "engine_kind": (
                "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
                "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "source_attested_en_PP-OCRv5_mobile_rec_head_v2",
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "ctc_label_policy": "optional-space-v1",
        }
    )

    profile = sidecar._profile_from_configuration(json.dumps(v19b).encode("utf-8"))

    assert profile.model_spec == "mobile_det_public_finetuned_rec"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.voyage_retry_mode == "bill-of-lading-padded-v1"

    for changed in (
        {**v19b, "schema_version": "18.0"},
        {**v19b, "voyage_retry_mode": "bill-of-lading-padded-v1"},
        {
            **v19b,
            "engine_kind": (
                "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
                "lowbox_strip2_semantic_repair_v2_onednn"
            ),
            "voyage_retry_mode": "bill-of-lading-padded-v1",
        },
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(changed).encode("utf-8"))


def _v20_configuration() -> dict[str, object]:
    v20 = _configuration()
    v20.update(
        {
            "schema_version": "20.0",
            "engine_kind": (
                "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_"
                "native960_lowbox_strip2_semantic_repair_v2_onednn"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": (
                "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
            ),
            "text_detection_limit_side_len": 960,
            "cpu_threads": 6,
            "sidecar_timeout_seconds": 30,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "recognition_crop_policy": "horizontal-height-0p32-max8-v1",
            "recognition_resize_interpolation": "linear-v1",
            "ctc_label_policy": "optional-space-v1",
        }
    )
    return v20


def test_v20_accepts_only_the_exact_voyage_replay_single_pass_contract() -> None:
    v20 = _v20_configuration()

    profile = sidecar._profile_from_configuration(json.dumps(v20).encode("utf-8"))

    assert profile.model_spec == "mobile_det_public_voyage_replay_rec"
    assert profile.detector_pass_mode == "strip2"
    assert profile.semantic_repair_mode == "supported_docs_v2"
    assert profile.enable_mkldnn
    assert profile.recognition_batch_size == 8
    assert profile.recognition_crop_policy == "horizontal-height-0p32-max8-v1"
    assert profile.recognition_resize_interpolation == "linear-v1"
    assert profile.ctc_label_policy == "optional-space-v1"
    assert profile.voyage_retry_mode == "none"

    for key, invalid in (
        ("schema_version", "19.0"),
        ("text_recognition_model_name", "source_attested_en_PP-OCRv5_mobile_rec_head_v2"),
        ("recognition_batch_size", 16),
        ("recognition_crop_policy", "detector-quad-v1"),
        ("recognition_crop_policy", "horizontal-height-0p16-max8-v1"),
        ("recognition_resize_interpolation", "cubic-v1"),
        ("ctc_label_policy", "append-space-v1"),
    ):
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps({**v20, key: invalid}).encode("utf-8"))

    for missing in (
        "recognition_crop_policy",
        "recognition_resize_interpolation",
        "ctc_label_policy",
    ):
        incomplete = dict(v20)
        del incomplete[missing]
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
            sidecar._profile_from_configuration(json.dumps(incomplete).encode("utf-8"))

    with pytest.raises(sidecar.DirectPaddleSidecarError, match="configuration_invalid"):
        sidecar._profile_from_configuration(
            json.dumps({**v20, "voyage_retry_mode": "none"}).encode("utf-8")
        )


def test_v17_recognition_quad_adds_only_bounded_source_pixel_margin() -> None:
    np = pytest.importorskip("numpy")
    profile = replace(_profile(), recognition_crop_policy="horizontal-height-0p08-max4-v1")
    engine = SimpleNamespace(np=np, profile=profile)
    image = np.zeros((120, 100, 3), dtype="uint8")

    short = np.asarray(((10, 10), (30, 10), (30, 20), (10, 20)), dtype="float32")
    short_expanded = sidecar._recognition_quad(image=image, ordered=short, engine=engine)
    assert short_expanded.tolist() == [
        [9.0, 10.0],
        [31.0, 10.0],
        [31.0, 20.0],
        [9.0, 20.0],
    ]

    tall_at_edges = np.asarray(((1, 10), (98, 10), (98, 110), (1, 110)), dtype="float32")
    tall_expanded = sidecar._recognition_quad(image=image, ordered=tall_at_edges, engine=engine)
    assert tall_expanded.tolist() == [
        [0.0, 10.0],
        [99.0, 10.0],
        [99.0, 110.0],
        [0.0, 110.0],
    ]


def test_v20_recognition_quad_uses_exact_ceil_height_margin_and_2_to_8_clamp() -> None:
    np = pytest.importorskip("numpy")
    profile = replace(_profile(), recognition_crop_policy="horizontal-height-0p32-max8-v1")
    engine = SimpleNamespace(np=np, profile=profile)
    image = np.zeros((120, 100, 3), dtype="uint8")

    cases = (
        (3, [[8.0, 10.0], [32.0, 10.0], [32.0, 13.0], [8.0, 13.0]]),
        (10, [[6.0, 10.0], [34.0, 10.0], [34.0, 20.0], [6.0, 20.0]]),
        (30, [[2.0, 10.0], [38.0, 10.0], [38.0, 40.0], [2.0, 40.0]]),
    )
    for height, expected in cases:
        box = np.asarray(
            ((10, 10), (30, 10), (30, 10 + height), (10, 10 + height)), dtype="float32"
        )
        expanded = sidecar._recognition_quad(image=image, ordered=box, engine=engine)
        assert expanded.tolist() == expected

    at_edges = np.asarray(((1, 10), (98, 10), (98, 40), (1, 40)), dtype="float32")
    expanded_at_edges = sidecar._recognition_quad(image=image, ordered=at_edges, engine=engine)
    assert expanded_at_edges.tolist() == [
        [0.0, 10.0],
        [99.0, 10.0],
        [99.0, 40.0],
        [0.0, 40.0],
    ]


def test_recognition_input_uses_only_profile_bound_resize_interpolation() -> None:
    np = pytest.importorskip("numpy")

    class _Cv2:
        INTER_LINEAR = 1
        INTER_CUBIC = 2

        def __init__(self) -> None:
            self.observed: list[int] = []

        def resize(self, _crop: object, size: tuple[int, int], *, interpolation: int) -> object:
            self.observed.append(interpolation)
            return np.full((size[1], size[0], 3), 128, dtype="uint8")

    crop = np.zeros((10, 20, 3), dtype="uint8")
    cv2 = _Cv2()
    linear_engine = SimpleNamespace(cv2=cv2, np=np, profile=_profile())
    linear = sidecar._recognition_input(crop=crop, engine=linear_engine)
    cubic_engine = SimpleNamespace(
        cv2=cv2,
        np=np,
        profile=replace(_profile(), recognition_resize_interpolation="cubic-v1"),
    )
    cubic = sidecar._recognition_input(crop=crop, engine=cubic_engine)
    padded_linear_engine = SimpleNamespace(
        cv2=cv2,
        np=np,
        profile=replace(_profile(), recognition_crop_policy="horizontal-height-0p08-max4-v1"),
    )
    padded_linear = sidecar._recognition_input(crop=crop, engine=padded_linear_engine)

    assert cv2.observed == [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LINEAR]
    assert linear.shape == cubic.shape == padded_linear.shape == (3, 48, 96)
    assert linear.dtype == cubic.dtype == padded_linear.dtype == np.dtype("float32")


def test_v20_recognize_crops_crops_every_line_once_and_uses_one_native_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    profile = replace(
        _profile(),
        recognition_crop_policy="horizontal-height-0p32-max8-v1",
        recognition_batch_size=8,
    )
    engine = cast(
        Any,
        SimpleNamespace(
            np=np,
            recognizer=object(),
            characters=("A",),
            profile=profile,
        ),
    )
    boxes = tuple(
        np.asarray(((left, 10), (left + 10, 10), (left + 10, 20), (left, 20)))
        for left in (10, 30, 50)
    )
    crop_calls: list[object] = []
    predictor_batches: list[tuple[int, ...]] = []

    def crop_quad(**kwargs: object) -> object:
        crop_calls.append(kwargs["box"])
        assert kwargs["engine"] is engine
        assert "policy_override" not in kwargs
        return object()

    def run_predictor(**kwargs: object) -> object:
        batch = cast(Any, kwargs["batch"])
        predictor_batches.append(batch.shape)
        return np.asarray(
            [
                [[0.1, 0.9], [0.9, 0.1]],
                [[0.1, 0.9], [0.9, 0.1]],
                [[0.1, 0.9], [0.9, 0.1]],
            ],
            dtype="float32",
        )

    monkeypatch.setattr(sidecar, "_crop_quad", crop_quad)
    monkeypatch.setattr(
        sidecar,
        "_recognition_input",
        lambda **_kwargs: np.zeros((3, 48, 32), dtype="float32"),
    )
    monkeypatch.setattr(sidecar, "_run_predictor", run_predictor)

    lines = sidecar._recognize_crops(image=object(), boxes=boxes, engine=engine)

    assert len(crop_calls) == len(boxes)
    assert all(observed is expected for observed, expected in zip(crop_calls, boxes, strict=True))
    assert predictor_batches == [(3, 3, 48, 32)]
    assert [line.text for line in lines] == ["A", "A", "A"]


def test_v10_grid2_geometry_mapping_nms_and_cap_are_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = sidecar._grid2_tiles(100, 100)
    assert tiles == (
        sidecar._Grid2Tile(ordinal=0, x_start=0, x_end=58, y_start=0, y_end=55),
        sidecar._Grid2Tile(ordinal=1, x_start=43, x_end=100, y_start=0, y_end=55),
        sidecar._Grid2Tile(ordinal=2, x_start=0, x_end=58, y_start=45, y_end=100),
        sidecar._Grid2Tile(ordinal=3, x_start=43, x_end=100, y_start=45, y_end=100),
    )
    assert sidecar._grid2_tiles(101, 101) == (
        sidecar._Grid2Tile(ordinal=0, x_start=0, x_end=59, y_start=0, y_end=56),
        sidecar._Grid2Tile(ordinal=1, x_start=43, x_end=101, y_start=0, y_end=56),
        sidecar._Grid2Tile(ordinal=2, x_start=0, x_end=59, y_start=45, y_end=101),
        sidecar._Grid2Tile(ordinal=3, x_start=43, x_end=101, y_start=45, y_end=101),
    )
    assert sidecar._offset_quad_points_xy(
        box=((1.0, 2.0), (5.0, 2.0), (5.0, 6.0), (1.0, 6.0)), x_offset=43, y_offset=45
    ) == ((44.0, 47.0), (48.0, 47.0), (48.0, 51.0), (44.0, 51.0))

    tl = ((44.0, 46.0), (54.0, 46.0), (54.0, 54.0), (44.0, 54.0))
    br = ((45.0, 46.0), (55.0, 46.0), (55.0, 54.0), (45.0, 54.0))
    same_tile = ((20.0, 46.0), (30.0, 46.0), (30.0, 54.0), (20.0, 54.0))
    result = sidecar._grid2_cross_tile_nms(
        (
            sidecar._StripDetectedBox(box=br, score=0.9, tile_index=3, ordinal=0),
            sidecar._StripDetectedBox(box=tl, score=0.9, tile_index=0, ordinal=0),
            sidecar._StripDetectedBox(box=same_tile, score=0.8, tile_index=0, ordinal=1),
        ),
        tiles=tiles,
    )
    assert result == [tl, same_tile]

    # Boxes outside a tile pair's overlap rectangle must not reach expensive
    # polygon IoU even if their tile ordinals differ.
    iou_calls: list[object] = []
    monkeypatch.setattr(
        sidecar,
        "_rotated_quad_iou",
        lambda *_args: iou_calls.append(object()) or 1.0,
    )
    non_overlap = sidecar._grid2_cross_tile_nms(
        (
            sidecar._StripDetectedBox(
                box=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
                score=0.9,
                tile_index=0,
                ordinal=0,
            ),
            sidecar._StripDetectedBox(
                box=((90.0, 90.0), (99.0, 90.0), (99.0, 99.0), (90.0, 99.0)),
                score=0.8,
                tile_index=3,
                ordinal=0,
            ),
        ),
        tiles=tiles,
    )
    assert len(non_overlap) == 2
    assert iou_calls == []

    monkeypatch.setattr(sidecar, "_GRID2_MAX_IOU_COMPARISONS", 1)
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="output_invalid"):
        sidecar._grid2_cross_tile_nms(
            (
                sidecar._StripDetectedBox(box=tl, score=0.9, tile_index=0, ordinal=0),
                sidecar._StripDetectedBox(
                    box=((50.0, 46.0), (58.0, 46.0), (58.0, 54.0), (50.0, 54.0)),
                    score=0.8,
                    tile_index=1,
                    ordinal=0,
                ),
                sidecar._StripDetectedBox(
                    box=((49.0, 46.0), (58.0, 46.0), (58.0, 54.0), (49.0, 54.0)),
                    score=0.7,
                    tile_index=1,
                    ordinal=1,
                ),
            ),
            tiles=tiles,
        )


def test_v11_guarded_grid_marks_only_internal_edges_and_retains_the_flag_on_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = sidecar._grid2_tiles(100, 100)
    outer_top_left = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    internal_right = ((53.0, 10.0), (57.0, 10.0), (57.0, 14.0), (53.0, 14.0))
    outer_right = ((52.0, 10.0), (56.0, 10.0), (56.0, 14.0), (52.0, 14.0))
    near_but_not_touching = ((1.0, 10.0), (5.0, 10.0), (5.0, 14.0), (1.0, 14.0))

    assert not sidecar._guarded_grid2_touches_internal_tile_edge(
        box=outer_top_left, tile=tiles[0], page_width=100, page_height=100
    )
    assert sidecar._guarded_grid2_touches_internal_tile_edge(
        box=outer_top_left, tile=tiles[1], page_width=100, page_height=100
    )
    assert sidecar._guarded_grid2_touches_internal_tile_edge(
        box=outer_top_left, tile=tiles[2], page_width=100, page_height=100
    )
    assert sidecar._guarded_grid2_touches_internal_tile_edge(
        box=internal_right, tile=tiles[0], page_width=100, page_height=100
    )
    assert not sidecar._guarded_grid2_touches_internal_tile_edge(
        box=outer_right, tile=tiles[1], page_width=100, page_height=100
    )
    assert not sidecar._guarded_grid2_touches_internal_tile_edge(
        box=near_but_not_touching, tile=tiles[1], page_width=100, page_height=100
    )

    class _Image:
        shape = (100, 100, 3)

        def __getitem__(self, key: object) -> SimpleNamespace:
            assert isinstance(key, tuple)
            assert isinstance(key[0], slice)
            assert isinstance(key[1], slice)
            return SimpleNamespace(
                tile_index=((0 if key[0].start == 0 else 2) + (0 if key[1].start == 0 else 1))
            )

    engine = SimpleNamespace(detector=object(), np=object())
    captured: list[sidecar._StripDetectedBox] = []
    monkeypatch.setattr(
        sidecar,
        "_detector_input",
        lambda *, image, **_kwargs: (
            object(),
            58 if image.tile_index in {0, 2} else 57,
            55,
        ),
    )
    monkeypatch.setattr(sidecar, "_run_predictor", lambda **_kwargs: object())
    monkeypatch.setattr(sidecar, "_probability_map", lambda **_kwargs: object())
    monkeypatch.setattr(
        sidecar,
        "_detected_scored_boxes",
        lambda *, tile_index, **_kwargs: [
            sidecar._StripDetectedBox(
                box=outer_top_left, score=0.9, tile_index=tile_index, ordinal=0
            )
        ],
    )
    monkeypatch.setattr(
        sidecar,
        "_offset_detected_box_xy",
        lambda *, box, x_offset, y_offset, **_kwargs: tuple(
            (x + x_offset, y + y_offset) for x, y in box
        ),
    )
    monkeypatch.setattr(
        sidecar,
        "_guarded_grid2_cross_tile_nms",
        lambda candidates, *, tiles: captured.extend(candidates) or [],
    )

    assert sidecar._guarded_grid2_detected_boxes(image=_Image(), engine=engine) == []
    assert [candidate.touches_internal_tile_edge for candidate in captured] == [
        False,
        True,
        True,
        True,
    ]
    assert [candidate.box[0] for candidate in captured] == [
        (0.0, 0.0),
        (43.0, 0.0),
        (0.0, 45.0),
        (43.0, 45.0),
    ]


def test_v11_guarded_grid_nms_suppresses_only_a_seam_with_an_intact_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiles = sidecar._grid2_tiles(100, 100)
    intact = ((46.0, 20.0), (52.0, 20.0), (52.0, 26.0), (46.0, 26.0))
    seam = ((43.0, 15.0), (57.0, 15.0), (57.0, 31.0), (43.0, 31.0))
    iou, containment = sidecar._rotated_quad_overlap_metrics(seam, intact)
    assert iou < 0.50
    assert containment == pytest.approx(1.0)

    result = sidecar._guarded_grid2_cross_tile_nms(
        (
            # Its score is higher, but it remains second because it is a
            # seam-marked candidate.  The intact contained box suppresses it.
            sidecar._StripDetectedBox(
                box=seam,
                score=0.99,
                tile_index=1,
                ordinal=0,
                touches_internal_tile_edge=True,
            ),
            sidecar._StripDetectedBox(
                box=intact,
                score=0.10,
                tile_index=0,
                ordinal=0,
                touches_internal_tile_edge=False,
            ),
        ),
        tiles=tiles,
    )
    assert result == [intact]

    # A seam fragment is retained if no non-edge result from an overlapping
    # tile justifies suppressing it.
    assert sidecar._guarded_grid2_cross_tile_nms(
        (
            sidecar._StripDetectedBox(
                box=seam,
                score=0.99,
                tile_index=1,
                ordinal=0,
                touches_internal_tile_edge=True,
            ),
        ),
        tiles=tiles,
    ) == [seam]

    # V10 must retain its historical symmetric score-first NMS even if a
    # caller constructs the new metadata field.  The flag is V11-only.
    duplicate = ((45.0, 20.0), (53.0, 20.0), (53.0, 28.0), (45.0, 28.0))
    offset_duplicate = ((45.5, 20.0), (53.5, 20.0), (53.5, 28.0), (45.5, 28.0))
    assert sidecar._grid2_cross_tile_nms(
        (
            sidecar._StripDetectedBox(
                box=duplicate,
                score=0.9,
                tile_index=1,
                ordinal=0,
                touches_internal_tile_edge=True,
            ),
            sidecar._StripDetectedBox(
                box=offset_duplicate,
                score=0.1,
                tile_index=0,
                ordinal=0,
                touches_internal_tile_edge=False,
            ),
        ),
        tiles=tiles,
    ) == [duplicate]

    # The V11 check still uses the fixed 4,096 comparison cap and fails
    # closed instead of silently dropping dense seam candidates.
    monkeypatch.setattr(sidecar, "_GRID2_MAX_IOU_COMPARISONS", 1)
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="output_invalid"):
        sidecar._guarded_grid2_cross_tile_nms(
            (
                sidecar._StripDetectedBox(
                    box=intact,
                    score=0.1,
                    tile_index=0,
                    ordinal=0,
                    touches_internal_tile_edge=False,
                ),
                sidecar._StripDetectedBox(
                    box=seam,
                    score=0.99,
                    tile_index=1,
                    ordinal=0,
                    touches_internal_tile_edge=True,
                ),
                sidecar._StripDetectedBox(
                    box=((43.0, 14.0), (57.0, 14.0), (57.0, 32.0), (43.0, 32.0)),
                    score=0.98,
                    tile_index=1,
                    ordinal=1,
                    touches_internal_tile_edge=True,
                ),
            ),
            tiles=tiles,
        )


def test_v10_grid2_detector_runs_exact_tl_tr_bl_br_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        shape = (100, 100, 3)

        def __getitem__(self, key: object) -> SimpleNamespace:
            assert isinstance(key, tuple)
            assert isinstance(key[0], slice)
            assert isinstance(key[1], slice)
            tile_slices.append((key[0].start, key[0].stop, key[1].start, key[1].stop))
            return SimpleNamespace(tile_index=len(tile_slices) - 1)

    tile_slices: list[tuple[int | None, int | None, int | None, int | None]] = []
    detector_inputs: list[int] = []
    predictor_calls: list[object] = []
    nms_calls: list[tuple[int, tuple[int, ...]]] = []
    engine = SimpleNamespace(detector=object(), np=object())

    def detector_input(*, image: SimpleNamespace, **_kwargs: object) -> tuple[object, int, int]:
        detector_inputs.append(image.tile_index)
        return object(), 58, 55

    monkeypatch.setattr(sidecar, "_detector_input", detector_input)
    monkeypatch.setattr(
        sidecar,
        "_run_predictor",
        lambda **_kwargs: predictor_calls.append(object()) or object(),
    )
    monkeypatch.setattr(sidecar, "_probability_map", lambda **_kwargs: object())
    monkeypatch.setattr(
        sidecar,
        "_detected_scored_boxes",
        lambda *, tile_index, **_kwargs: [
            sidecar._StripDetectedBox(
                box=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)),
                score=0.9,
                tile_index=tile_index,
                ordinal=0,
            )
        ],
    )
    monkeypatch.setattr(sidecar, "_offset_detected_box_xy", lambda **kwargs: kwargs["box"])
    monkeypatch.setattr(
        sidecar,
        "_grid2_cross_tile_nms",
        lambda candidates, *, tiles: (
            nms_calls.append((len(candidates), tuple(tile.ordinal for tile in tiles))) or []
        ),
    )

    assert sidecar._grid2_detected_boxes(image=_Image(), engine=engine) == []
    assert detector_inputs == [0, 1, 2, 3]
    assert tile_slices == [(0, 55, 0, 58), (0, 55, 43, 100), (45, 100, 0, 58), (45, 100, 43, 100)]
    assert len(predictor_calls) == 4
    assert nms_calls == [(4, (0, 1, 2, 3))]


def test_v10_predict_pages_uses_one_full_page_recognizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        def __getitem__(self, _key: object) -> "_Image":
            return self

    class _Pixels:
        def reshape(self, _shape: tuple[int, int, int]) -> _Image:
            return _Image()

    detector_calls: list[object] = []
    recognizer_calls: list[object] = []
    engine = SimpleNamespace(
        np=SimpleNamespace(frombuffer=lambda *_args, **_kwargs: _Pixels()),
        profile=SimpleNamespace(row_band_mode="harmonic_v1", detector_pass_mode="grid2"),
    )
    recognized = [_line(text="line", left=0, top=0, right=1, bottom=1)]
    monkeypatch.setattr(
        sidecar,
        "_grid2_detected_boxes",
        lambda **_kwargs: detector_calls.append(object()) or [],
    )
    monkeypatch.setattr(
        sidecar,
        "_recognize_crops",
        lambda **_kwargs: recognizer_calls.append(object()) or recognized,
    )
    monkeypatch.setattr(sidecar, "_ordered_lines", lambda lines, **_kwargs: list(lines))

    pages = sidecar._predict_pages(
        engine,
        (sidecar._Page(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00"),),
    )

    assert len(detector_calls) == 1
    assert len(recognizer_calls) == 1
    assert pages == [{"page_number": 1, "text": "line", "confidence": 0.9}]


def test_v20_uses_one_full_page_recognition_and_never_enters_v19b_voyage_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        def __getitem__(self, _key: object) -> "_Image":
            return self

    class _Pixels:
        def reshape(self, _shape: tuple[int, int, int]) -> _Image:
            return _Image()

    recognizer_calls: list[str] = []
    retry_calls: list[str] = []
    profile = SimpleNamespace(
        row_band_mode="harmonic_v1",
        detector_pass_mode="strip2",
        semantic_repair_mode="supported_docs_v2",
        recognition_crop_policy="horizontal-height-0p32-max8-v1",
        voyage_retry_mode="none",
    )
    engine = SimpleNamespace(
        np=SimpleNamespace(frombuffer=lambda *_args, **_kwargs: _Pixels()),
        profile=profile,
    )
    recognized = [_line(text="BILL OF LADING", left=0, top=0, right=100, bottom=10)]
    monkeypatch.setattr(sidecar, "_strip2_detected_boxes", lambda **_kwargs: [object()])
    monkeypatch.setattr(
        sidecar,
        "_recognize_crops",
        lambda **_kwargs: recognizer_calls.append("recognize") or recognized,
    )
    monkeypatch.setattr(sidecar, "_ordered_lines", lambda lines, **_kwargs: list(lines))
    monkeypatch.setattr(
        sidecar,
        "_retry_bill_of_lading_voyage_line",
        lambda **kwargs: retry_calls.append("retry") or list(kwargs["lines"]),
    )
    monkeypatch.setattr(
        sidecar,
        "_semantic_repair_supported_document_lines_v2",
        lambda lines: list(lines),
    )

    page = sidecar._Page(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")
    assert sidecar._predict_pages(cast(Any, engine), (page,)) == [
        {"page_number": 1, "text": "BILL OF LADING", "confidence": 0.9}
    ]
    assert recognizer_calls == ["recognize"]
    assert retry_calls == []

    # The new lane must not alter V19b's separately bound retry semantics.
    engine.profile = SimpleNamespace(
        **{**vars(profile), "voyage_retry_mode": "bill-of-lading-padded-v1"}
    )
    sidecar._predict_pages(cast(Any, engine), (page,))
    assert recognizer_calls == ["recognize", "recognize"]
    assert retry_calls == ["retry"]


def _line(*, text: str, left: float, top: float, right: float, bottom: float) -> sidecar._Line:
    return sidecar._Line(
        text=text,
        confidence=0.9,
        polygon=((left, top), (right, top), (right, bottom), (left, bottom)),
    )


def _bill_of_lading_voyage_lines(*, value: str = "234N") -> list[sidecar._Line]:
    return [
        _line(text="bIlL oF lAdInG", left=20, top=0, right=180, bottom=10),
        _line(text="⑦ Voyage No.", left=90, top=20, right=170, bottom=30),
        _line(text=value, left=100, top=33, right=145, bottom=43),
        _line(text="PANAMA", left=210, top=33, right=270, bottom=43),
    ]


def test_v19b_voyage_value_selection_is_title_label_and_geometry_bound() -> None:
    lines = _bill_of_lading_voyage_lines()

    assert sidecar._unique_bill_of_lading_voyage_value_line(lines) == 2

    assert (
        sidecar._unique_bill_of_lading_voyage_value_line(
            [*lines, _line(text="BILL OF LADING", left=0, top=60, right=100, bottom=70)]
        )
        is None
    )
    assert (
        sidecar._unique_bill_of_lading_voyage_value_line(
            [*lines, _line(text="VOYAGE", left=300, top=20, right=350, bottom=30)]
        )
        is None
    )
    assert (
        sidecar._unique_bill_of_lading_voyage_value_line(
            [*lines, _line(text="12W", left=110, top=34, right=140, bottom=44)]
        )
        is None
    )
    assert (
        sidecar._unique_bill_of_lading_voyage_value_line(
            [
                lines[0],
                lines[1],
                _line(text="234N", left=100, top=90, right=145, bottom=100),
            ]
        )
        is None
    )
    assert (
        sidecar._unique_bill_of_lading_voyage_value_line(
            [
                _line(text="PACKING LIST", left=20, top=0, right=180, bottom=10),
                *lines[1:],
            ]
        )
        is None
    )


@pytest.mark.parametrize(
    (
        "original",
        "original_confidence",
        "first",
        "first_confidence",
        "second",
        "second_confidence",
        "expected",
    ),
    [
        ("234N", 0.80, "1234N", 0.60, "1234N", 0.61, True),
        ("X1234N", 0.80, "1234N", 0.83, "1234N", 0.84, True),
        ("X1234N", 0.80, "1234N", 0.82, "1234N", 0.82, True),
        ("X1234N", 0.80, "1234N", 0.83, "1234N", 0.819, False),
        ("1234N", 0.80, "234N", 0.99, "234N", 0.99, False),
        ("1234N", 0.80, "X1234N", 0.99, "X1234N", 0.99, False),
        ("12X34N", 0.80, "1234N", 0.99, "1234N", 0.99, False),
        ("234N", 0.80, "1234N", 0.60, "X234N", 0.99, False),
        ("234N", 0.80, "2314N", 0.99, "2314N", 0.99, False),
        ("1234N", 0.80, "1234N", 0.99, "1234N", 0.99, False),
        ("12AB", 0.80, "12XB", 0.99, "12XB", 0.99, False),
        ("12 AB", 0.10, "X12 AB", 0.99, "X12 AB", 0.99, False),
        ("ABCD", 0.80, "XABCD", float("nan"), "XABCD", 0.99, False),
    ],
)
def test_v19b_voyage_retry_accepts_only_consensus_prefix_insert_or_confident_delete(
    original: str,
    original_confidence: float,
    first: str,
    first_confidence: float,
    second: str,
    second_confidence: float,
    expected: bool,
) -> None:
    old = replace(
        _line(text=original, left=100, top=33, right=145, bottom=43),
        confidence=original_confidence,
    )
    first_retry = replace(
        _line(text=first, left=100, top=33, right=145, bottom=43),
        confidence=first_confidence,
    )
    second_retry = replace(
        _line(text=second, left=100, top=33, right=145, bottom=43),
        confidence=second_confidence,
    )

    assert (
        sidecar._voyage_retry_consensus_improves_line(
            original=old, candidates=(first_retry, second_retry)
        )
        is expected
    )


def test_v19b_padded_retry_runs_two_fixed_crops_in_one_recognizer_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    old = _bill_of_lading_voyage_lines()[2]
    observed: dict[str, object] = {"policies": []}
    engine = SimpleNamespace(
        np=np,
        recognizer=object(),
        characters=("A",),
        profile=SimpleNamespace(ctc_label_policy="optional-space-v1"),
    )

    def crop_quad(**kwargs: object) -> object:
        cast(list[object], observed["policies"]).append(kwargs["policy_override"])
        observed["box"] = kwargs["box"]
        return kwargs["policy_override"]

    def run_predictor(**kwargs: object) -> object:
        batch = cast(Any, kwargs["batch"])
        observed["batch_shape"] = batch.shape
        return np.zeros((2, 2, 2), dtype="float32")

    monkeypatch.setattr(sidecar, "_crop_quad", crop_quad)
    monkeypatch.setattr(
        sidecar,
        "_recognition_input",
        lambda **kwargs: np.zeros(
            (
                3,
                48,
                37 if kwargs["crop"] == "horizontal-height-0p08-max4-v1" else 41,
            ),
            dtype="float32",
        ),
    )
    monkeypatch.setattr(sidecar, "_run_predictor", run_predictor)
    monkeypatch.setattr(sidecar, "_ctc_decode", lambda *_args, **_kwargs: ("1234N", 0.95))

    retried = sidecar._recognize_padded_voyage_consensus(image=object(), line=old, engine=engine)

    expected = sidecar._Line(text="1234N", confidence=0.95, polygon=old.polygon)
    assert retried == (expected, expected)
    assert observed["policies"] == [
        "horizontal-height-0p08-max4-v1",
        "horizontal-height-0p16-max8-v1",
    ]
    assert cast(Any, observed["box"]).shape == (4, 2)
    assert observed["batch_shape"] == (2, 3, 48, 41)


def test_v19b_padded_retry_rejects_two_effectively_identical_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    old = _bill_of_lading_voyage_lines()[2]
    predictor_called = False
    engine = SimpleNamespace(
        np=np,
        recognizer=object(),
        characters=("A",),
        profile=SimpleNamespace(ctc_label_policy="optional-space-v1"),
    )

    monkeypatch.setattr(sidecar, "_crop_quad", lambda **_kwargs: object())
    monkeypatch.setattr(
        sidecar,
        "_recognition_input",
        lambda **_kwargs: np.zeros((3, 48, 37), dtype="float32"),
    )

    def run_predictor(**_kwargs: object) -> object:
        nonlocal predictor_called
        predictor_called = True
        return np.zeros((2, 2, 2), dtype="float32")

    monkeypatch.setattr(sidecar, "_run_predictor", run_predictor)

    assert (
        sidecar._recognize_padded_voyage_consensus(image=object(), line=old, engine=engine) is None
    )
    assert predictor_called is False


def test_v19b_retry_replaces_only_one_proven_better_voyage_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _bill_of_lading_voyage_lines()
    calls: list[str] = []

    def recognize(**kwargs: object) -> tuple[sidecar._Line, sidecar._Line]:
        calls.append(cast(sidecar._Line, kwargs["line"]).text)
        candidate = replace(cast(sidecar._Line, kwargs["line"]), text="1234N", confidence=0.95)
        return candidate, candidate

    monkeypatch.setattr(sidecar, "_recognize_padded_voyage_consensus", recognize)

    retried = sidecar._retry_bill_of_lading_voyage_line(
        image=object(), lines=lines, engine=cast(Any, object())
    )

    assert calls == ["234N"]
    assert [line.text for line in retried] == [
        "bIlL oF lAdInG",
        "⑦ Voyage No.",
        "1234N",
        "PANAMA",
    ]


def test_v19b_retry_preserves_original_on_ambiguity_or_optional_retry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _bill_of_lading_voyage_lines()
    calls: list[object] = []

    def fail(**_kwargs: object) -> tuple[sidecar._Line, sidecar._Line]:
        calls.append(object())
        raise sidecar.DirectPaddleSidecarError("retry-invalid")

    monkeypatch.setattr(sidecar, "_recognize_padded_voyage_consensus", fail)
    assert (
        sidecar._retry_bill_of_lading_voyage_line(
            image=object(), lines=lines, engine=cast(Any, object())
        )
        == lines
    )
    assert len(calls) == 1

    ambiguous = [*lines, _line(text="12W", left=110, top=34, right=140, bottom=44)]
    assert (
        sidecar._retry_bill_of_lading_voyage_line(
            image=object(), lines=ambiguous, engine=cast(Any, object())
        )
        == ambiguous
    )
    assert len(calls) == 1


@pytest.mark.parametrize("count", ["1", "42", "999"])
def test_v14_text_delta_repairs_one_closed_ascii_carton_count(count: str) -> None:
    texts = ("PACKING LIST", "DESCRIPTION", f"C/T:{count}", "TOTAL")

    assert sidecar.semantic_repair_v14_packing_carton_texts(texts) == (
        "PACKING LIST",
        "DESCRIPTION",
        f"C/T: {count}",
        "TOTAL",
    )


@pytest.mark.parametrize(
    "texts",
    [
        ("PACKING LIST", "C/T:1", "C/T:2"),
        ("PACKING LIST", "PACKING LIST", "C/T:1"),
        ("PACKING LIST", "COMMERCIAL INVOICE", "C/T:1"),
        ("BILL OF LADING", "C/T:1"),
        ("COMMERCIAL INVOICE", "C/T:1"),
        ("PACKING LIST", "C/T: 1"),
        ("PACKING LIST", "CT:1"),
        ("PACKING LIST", "C/T:"),
        ("PACKING LIST", "C/T:1234"),
        ("PACKING LIST", "C/T:١"),
        ("PACKING LIST", "VALUE C/T:1"),
        ("PACKING LIST", "C/T:1 VALUE"),
    ],
    ids=[
        "duplicate-candidate",
        "duplicate-heading",
        "ambiguous-heading",
        "bill-of-lading",
        "invoice",
        "existing-space",
        "different-prefix",
        "missing-count",
        "four-digits",
        "non-ascii-digit",
        "joined-prefix",
        "joined-suffix",
    ],
)
def test_v14_text_delta_preserves_ambiguous_or_non_closed_shapes(
    texts: tuple[str, ...],
) -> None:
    assert sidecar.semantic_repair_v14_packing_carton_texts(texts) == texts


def test_v14_line_wrapper_preserves_v13_behavior_geometry_and_confidence() -> None:
    carton = _line(text="C/T:123", left=10, top=20, right=80, bottom=30)
    ordinal = _line(text="1)", left=10, top=40, right=20, bottom=50)
    lines = (
        _line(text="PACKING LIST", left=20, top=0, right=180, bottom=10),
        carton,
        ordinal,
    )

    v13 = sidecar._semantic_repair_supported_document_lines(lines)
    v14 = sidecar._semantic_repair_supported_document_lines_v2(lines)

    assert [line.text for line in v13] == ["PACKING LIST", "C/T:123", "(1)"]
    assert [line.text for line in v14] == ["PACKING LIST", "C/T: 123", "(1)"]
    assert v14[1].polygon is carton.polygon
    assert v14[1].confidence == carton.confidence
    assert v14[2].polygon is ordinal.polygon
    assert v14[2].confidence == ordinal.confidence


def test_v13_balances_only_a_standalone_numeric_form_ordinal_on_supported_pages() -> None:
    ordinal = _line(text="1)", left=10, top=20, right=22, bottom=30)
    list_item = _line(text="2) ITEM DESCRIPTION", left=10, top=40, right=150, bottom=50)
    already_balanced = _line(text="(3)", left=10, top=60, right=28, bottom=70)
    supported = (
        _line(text="BILL OF LADING", left=20, top=0, right=180, bottom=10),
        ordinal,
        list_item,
        already_balanced,
    )

    repaired = sidecar._semantic_repair_supported_document_lines(supported)

    assert [line.text for line in repaired] == [
        "BILL OF LADING",
        "(1)",
        "2) ITEM DESCRIPTION",
        "(3)",
    ]
    assert repaired[1].polygon is ordinal.polygon
    assert repaired[1].confidence == ordinal.confidence
    assert repaired[2] is list_item
    assert repaired[3] is already_balanced

    unknown = (
        _line(text="DELIVERY NOTE", left=20, top=0, right=180, bottom=10),
        ordinal,
    )
    assert sidecar._semantic_repair_supported_document_lines(unknown) == list(unknown)
    ambiguous_document = (
        _line(text="BILL OF LADING", left=20, top=0, right=180, bottom=10),
        _line(text="COMMERCIAL INVOICE", left=20, top=12, right=180, bottom=22),
        ordinal,
    )
    assert sidecar._semantic_repair_supported_document_lines(ambiguous_document) == list(
        ambiguous_document
    )


def test_v13_packing_slash_requires_exact_heading_geometry_labels_and_same_weight_unit() -> None:
    heading = _line(text="PACKING LIST", left=40, top=0, right=160, bottom=10)
    labels = _line(
        text="TOTAL NET WEIGHT     TOTAL GROSS WEIGHT",
        left=10,
        top=20,
        right=190,
        bottom=30,
    )
    amounts = _line(text="1,234.50 KG 1,300.00 KG", left=20, top=32, right=180, bottom=42)
    item_row = _line(text="ITEM 1 2 KG 3 KG", left=10, top=48, right=190, bottom=58)
    separated = _line(text="4 KG / 5 KG", left=20, top=64, right=180, bottom=74)
    mixed_units = _line(text="6 KG 7 LB", left=20, top=80, right=180, bottom=90)

    repaired = sidecar._semantic_repair_supported_document_lines(
        (heading, labels, amounts, item_row, separated, mixed_units)
    )

    assert [line.text for line in repaired] == [
        "PACKING LIST",
        "TOTAL NET WEIGHT     TOTAL GROSS WEIGHT",
        "1,234.50 KG / 1,300.00 KG",
        "ITEM 1 2 KG 3 KG",
        "4 KG / 5 KG",
        "6 KG 7 LB",
    ]
    assert repaired[2].polygon is amounts.polygon
    assert repaired[3] is item_row
    assert repaired[4] is separated
    assert repaired[5] is mixed_units


def test_v13_packing_slash_preserves_a_value_when_net_gross_labels_are_distant() -> None:
    lines = (
        _line(text="PACKING LIST", left=40, top=0, right=160, bottom=10),
        _line(text="NET WEIGHT", left=10, top=20, right=90, bottom=30),
        _line(text="GROSS WEIGHT", left=100, top=20, right=190, bottom=30),
        _line(text="8 KG 9 KG", left=20, top=200, right=180, bottom=210),
    )

    assert sidecar._semantic_repair_supported_document_lines(lines) == list(lines)


def test_v13_packing_slash_accepts_one_geometry_bound_split_label_pair() -> None:
    heading = _line(text="PACKING LIST", left=40, top=0, right=160, bottom=10)
    split_labels = (
        _line(text="NET WEIGHT", left=10, top=20, right=90, bottom=30),
        _line(text="GROSS WEIGHT", left=100, top=20, right=190, bottom=30),
    )
    amount = _line(text="10 KG 12 KG", left=20, top=32, right=180, bottom=42)
    split_context = (heading, *split_labels, amount)

    repaired = sidecar._semantic_repair_supported_document_lines(split_context)

    assert [line.text for line in repaired] == [
        "PACKING LIST",
        "NET WEIGHT",
        "GROSS WEIGHT",
        "10 KG / 12 KG",
    ]
    assert repaired[-1].polygon is amount.polygon


def test_v13_packing_slash_joins_one_geometry_bound_split_amount_quartet() -> None:
    """The real V9/V13 shape has labels and values on four separate lines."""

    heading = _line(text="PACKING LIST", left=0.40, top=0.02, right=0.60, bottom=0.03)
    net_label = _line(
        text="NET WEIGHT",
        left=0.5666,
        top=0.2930,
        right=0.6184,
        bottom=0.29959,
    )
    gross_label = _line(
        text="GROSS WEIGHT",
        left=0.6859,
        top=0.2930,
        right=0.7508,
        bottom=0.29968,
    )
    net_amount = _line(
        text="1,234.50 KGS",
        left=0.5837,
        top=0.32416,
        right=0.6477,
        bottom=0.33139,
    )
    gross_amount = _line(
        text="1,300.00 KGS",
        left=0.7097,
        top=0.32416,
        right=0.7737,
        bottom=0.33139,
    )
    # A repeated amount pair elsewhere on the form is not evidence for a
    # second quartet because it is well outside the fixed four-height label
    # band.  This mirrors the observed public-mobile output.
    distant_net_amount = _line(
        text="1,234.50 KGS",
        left=0.5837,
        top=0.47382,
        right=0.6477,
        bottom=0.48105,
    )
    distant_gross_amount = _line(
        text="1,300.00 KGS",
        left=0.7097,
        top=0.47382,
        right=0.7737,
        bottom=0.48105,
    )
    lines = (
        heading,
        net_label,
        gross_label,
        net_amount,
        gross_amount,
        distant_net_amount,
        distant_gross_amount,
    )

    repaired = sidecar._semantic_repair_supported_document_lines(lines)

    assert [line.text for line in repaired] == [
        "PACKING LIST",
        "NET WEIGHT",
        "GROSS WEIGHT",
        "1,234.50 KGS /",
        "1,300.00 KGS",
        "1,234.50 KGS",
        "1,300.00 KGS",
    ]
    assert repaired[3].polygon is net_amount.polygon
    assert repaired[3].confidence == net_amount.confidence
    assert repaired[4] is gross_amount
    assert repaired[5] is distant_net_amount
    assert repaired[6] is distant_gross_amount


@pytest.mark.parametrize(
    "extra_label",
    [
        _line(text="NET MASS", left=0.57, top=0.294, right=0.62, bottom=0.301),
        _line(text="GROSS MASS", left=0.69, top=0.294, right=0.75, bottom=0.301),
        _line(
            text="NET WEIGHT GROSS WEIGHT",
            left=0.56,
            top=0.294,
            right=0.76,
            bottom=0.301,
        ),
    ],
    ids=["duplicate-net", "duplicate-gross", "mixed-combined-split"],
)
def test_v13_split_amount_quartet_rejects_duplicate_or_mixed_nearby_labels(
    extra_label: sidecar._Line,
) -> None:
    lines = (
        _line(text="PACKING LIST", left=0.4, top=0.02, right=0.6, bottom=0.03),
        _line(text="NET WEIGHT", left=0.566, top=0.293, right=0.619, bottom=0.300),
        _line(text="GROSS WEIGHT", left=0.686, top=0.293, right=0.751, bottom=0.300),
        extra_label,
        _line(text="10 KGS", left=0.584, top=0.324, right=0.648, bottom=0.331),
        _line(text="12 KGS", left=0.710, top=0.324, right=0.774, bottom=0.331),
    )

    assert sidecar._semantic_repair_supported_document_lines(lines) == list(lines)


def test_v13_split_amount_quartet_rejects_multiple_nearby_amount_pairs() -> None:
    lines = (
        _line(text="PACKING LIST", left=0.4, top=0.02, right=0.6, bottom=0.03),
        _line(text="NET WEIGHT", left=0.566, top=0.293, right=0.619, bottom=0.300),
        _line(text="GROSS WEIGHT", left=0.686, top=0.293, right=0.751, bottom=0.300),
        _line(text="10 KGS", left=0.584, top=0.324, right=0.648, bottom=0.331),
        _line(text="12 KGS", left=0.710, top=0.324, right=0.774, bottom=0.331),
        _line(text="SUMMARY", left=0.40, top=0.332, right=0.48, bottom=0.339),
        _line(text="14 KGS", left=0.584, top=0.326, right=0.648, bottom=0.333),
        _line(text="16 KGS", left=0.710, top=0.326, right=0.774, bottom=0.333),
    )

    assert sidecar._semantic_repair_supported_document_lines(lines) == list(lines)


@pytest.mark.parametrize(
    ("first_text", "second_text", "first_box", "second_box"),
    [
        (
            "10 KGS",
            "12 LBS",
            (0.584, 0.324, 0.648, 0.331),
            (0.710, 0.324, 0.774, 0.331),
        ),
        (
            "10 KGS",
            "12 KGS",
            (0.710, 0.324, 0.774, 0.331),
            (0.584, 0.324, 0.648, 0.331),
        ),
        (
            "10 KGS",
            "12 KGS",
            (0.584, 0.324, 0.648, 0.331),
            (0.710, 0.338, 0.774, 0.345),
        ),
        (
            "10 KGS",
            "12 KGS",
            (0.584, 0.350, 0.648, 0.357),
            (0.710, 0.350, 0.774, 0.357),
        ),
    ],
    ids=["mixed-units", "crossed-x-mapping", "different-value-rows", "distant-values"],
)
def test_v13_split_amount_quartet_rejects_ambiguous_geometry_or_units(
    first_text: str,
    second_text: str,
    first_box: tuple[float, float, float, float],
    second_box: tuple[float, float, float, float],
) -> None:
    lines = (
        _line(text="PACKING LIST", left=0.4, top=0.02, right=0.6, bottom=0.03),
        _line(text="NET WEIGHT", left=0.566, top=0.293, right=0.619, bottom=0.300),
        _line(text="GROSS WEIGHT", left=0.686, top=0.293, right=0.751, bottom=0.300),
        _line(
            text=first_text,
            left=first_box[0],
            top=first_box[1],
            right=first_box[2],
            bottom=first_box[3],
        ),
        _line(
            text=second_text,
            left=second_box[0],
            top=second_box[1],
            right=second_box[2],
            bottom=second_box[3],
        ),
    )

    assert sidecar._semantic_repair_supported_document_lines(lines) == list(lines)


@pytest.mark.parametrize("heading", ["BILL OF LADING", "COMMERCIAL INVOICE"])
def test_v13_split_amount_quartet_is_never_applied_to_other_documents(heading: str) -> None:
    lines = (
        _line(text=heading, left=0.4, top=0.02, right=0.6, bottom=0.03),
        _line(text="NET WEIGHT", left=0.566, top=0.293, right=0.619, bottom=0.300),
        _line(text="GROSS WEIGHT", left=0.686, top=0.293, right=0.751, bottom=0.300),
        _line(text="10 KGS", left=0.584, top=0.324, right=0.648, bottom=0.331),
        _line(text="12 KGS", left=0.710, top=0.324, right=0.774, bottom=0.331),
    )

    assert sidecar._semantic_repair_supported_document_lines(lines) == list(lines)


@pytest.mark.parametrize(
    "label_texts",
    [
        ("NET WEIGHT GROSS WEIGHT", "NET MASS GROSS MASS"),
        ("NET WEIGHT GROSS WEIGHT", "NET WEIGHT"),
        ("NET WEIGHT", "NET MASS", "GROSS WEIGHT"),
        ("NET WEIGHT", "GROSS WEIGHT", "GROSS MASS"),
        ("NET WEIGHT",),
        ("GROSS WEIGHT",),
    ],
    ids=[
        "extra-combined",
        "mixed-combined-split",
        "duplicate-net",
        "duplicate-gross",
        "missing-gross",
        "missing-net",
    ],
)
def test_v13_packing_slash_preserves_duplicate_extra_or_mixed_label_evidence(
    label_texts: tuple[str, ...],
) -> None:
    heading = _line(text="PACKING LIST", left=40, top=0, right=160, bottom=10)
    labels = tuple(
        _line(text=text, left=10, top=20 + index, right=190, bottom=30 + index)
        for index, text in enumerate(label_texts)
    )
    amount = _line(text="10 KG 12 KG", left=20, top=34, right=180, bottom=44)
    ambiguous = (heading, *labels, amount)

    assert sidecar._semantic_repair_supported_document_lines(ambiguous) == list(ambiguous)


def test_v13_packing_slash_preserves_multiple_matching_value_rows() -> None:
    heading = _line(text="PACKING LIST", left=40, top=0, right=160, bottom=10)
    amount = _line(text="10 KG 12 KG", left=20, top=32, right=180, bottom=42)

    combined_label = _line(text="NET WEIGHT GROSS WEIGHT", left=10, top=20, right=190, bottom=30)
    second_amount = _line(text="14 KG 16 KG", left=20, top=44, right=180, bottom=54)
    multiple_rows = (heading, combined_label, amount, second_amount)
    assert sidecar._semantic_repair_supported_document_lines(multiple_rows) == list(multiple_rows)


@pytest.mark.parametrize("heading", ["BILL OF LADING", "COMMERCIAL INVOICE"])
def test_v13_never_applies_the_packing_weight_separator_to_bl_or_invoice(heading: str) -> None:
    lines = (
        _line(text=heading, left=20, top=0, right=180, bottom=10),
        _line(text="NET WEIGHT GROSS WEIGHT", left=10, top=20, right=190, bottom=30),
        _line(text="10 KG 12 KG", left=20, top=32, right=180, bottom=42),
        _line(text="1) ITEM", left=20, top=50, right=180, bottom=60),
    )

    repaired = sidecar._semantic_repair_supported_document_lines(lines)

    assert [line.text for line in repaired] == [
        heading,
        "NET WEIGHT GROSS WEIGHT",
        "10 KG 12 KG",
        "1) ITEM",
    ]


def test_v13_predict_pages_applies_repair_after_geometry_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        def __getitem__(self, _key: object) -> "_Image":
            return self

    class _Pixels:
        def reshape(self, _shape: tuple[int, int, int]) -> _Image:
            return _Image()

    engine = SimpleNamespace(
        np=SimpleNamespace(frombuffer=lambda *_args, **_kwargs: _Pixels()),
        profile=SimpleNamespace(
            row_band_mode="harmonic_v1",
            detector_pass_mode="strip2",
            semantic_repair_mode="supported_docs_v1",
        ),
    )
    recognized = [
        _line(text="PACKING LIST", left=20, top=0, right=180, bottom=10),
        _line(text="NET WEIGHT GROSS WEIGHT", left=10, top=20, right=190, bottom=30),
        _line(text="10 KG 12 KG", left=20, top=32, right=180, bottom=42),
    ]
    monkeypatch.setattr(sidecar, "_strip2_detected_boxes", lambda **_kwargs: [])
    monkeypatch.setattr(sidecar, "_recognize_crops", lambda **_kwargs: recognized)
    monkeypatch.setattr(sidecar, "_ordered_lines", lambda lines, **_kwargs: list(lines))

    pages = sidecar._predict_pages(
        engine,
        (sidecar._Page(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00"),),
    )

    assert pages == [
        {
            "page_number": 1,
            "text": "PACKING LIST\nNET WEIGHT GROSS WEIGHT\n10 KG / 12 KG",
            "confidence": 0.9,
        }
    ]


def test_v14_predict_pages_applies_text_delta_after_v13_ordered_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        def __getitem__(self, _key: object) -> "_Image":
            return self

    class _Pixels:
        def reshape(self, _shape: tuple[int, int, int]) -> _Image:
            return _Image()

    engine = SimpleNamespace(
        np=SimpleNamespace(frombuffer=lambda *_args, **_kwargs: _Pixels()),
        profile=SimpleNamespace(
            row_band_mode="harmonic_v1",
            detector_pass_mode="strip2",
            semantic_repair_mode="supported_docs_v2",
        ),
    )
    recognized = [
        _line(text="PACKING LIST", left=20, top=0, right=180, bottom=10),
        _line(text="C/T:123", left=20, top=20, right=80, bottom=30),
        _line(text="1)", left=20, top=40, right=30, bottom=50),
    ]
    monkeypatch.setattr(sidecar, "_strip2_detected_boxes", lambda **_kwargs: [])
    monkeypatch.setattr(sidecar, "_recognize_crops", lambda **_kwargs: recognized)
    monkeypatch.setattr(sidecar, "_ordered_lines", lambda lines, **_kwargs: list(lines))

    pages = sidecar._predict_pages(
        engine,
        (sidecar._Page(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00"),),
    )

    assert pages == [
        {
            "page_number": 1,
            "text": "PACKING LIST\nC/T: 123\n(1)",
            "confidence": 0.9,
        }
    ]


def test_harmonic_row_band_separates_a_next_row_from_one_inflated_box() -> None:
    """The exact three-box regression: max-height would merge the next row."""

    lines = (
        _line(text="top-left", left=10, top=25, right=80, bottom=35),
        _line(text="top-right-inflated", left=120, top=0, right=200, bottom=60),
        _line(text="next-row-left", left=10, top=45, right=80, bottom=55),
    )

    assert [line.text for line in sidecar._ordered_lines(lines, row_band_mode="legacy_max")] == [
        "top-left",
        "next-row-left",
        "top-right-inflated",
    ]
    assert [line.text for line in sidecar._ordered_lines(lines, row_band_mode="harmonic_v1")] == [
        "top-left",
        "top-right-inflated",
        "next-row-left",
    ]


def test_harmonic_row_band_keeps_ordinary_same_row_ordering() -> None:
    lines = (
        _line(text="left", left=10, top=20, right=60, bottom=30),
        _line(text="right", left=80, top=21, right=130, bottom=31),
        _line(text="next", left=10, top=45, right=60, bottom=55),
    )

    expected = ["left", "right", "next"]
    assert [
        line.text for line in sidecar._ordered_lines(lines, row_band_mode="legacy_max")
    ] == expected
    assert [
        line.text for line in sidecar._ordered_lines(lines, row_band_mode="harmonic_v1")
    ] == expected


def test_predict_pages_uses_the_profile_derived_row_band_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Image:
        def __getitem__(self, _key: object) -> "_Image":
            return self

    class _Pixels:
        def reshape(self, _shape: tuple[int, int, int]) -> _Image:
            return _Image()

    observed_modes: list[str] = []
    engine = SimpleNamespace(
        np=SimpleNamespace(frombuffer=lambda *_args, **_kwargs: _Pixels()),
        detector=object(),
        profile=SimpleNamespace(row_band_mode="harmonic_v1"),
    )
    recognized = [_line(text="line", left=0, top=0, right=1, bottom=1)]
    monkeypatch.setattr(sidecar, "_detector_input", lambda **_kwargs: (object(), 1, 1))
    monkeypatch.setattr(sidecar, "_run_predictor", lambda **_kwargs: object())
    monkeypatch.setattr(sidecar, "_probability_map", lambda **_kwargs: object())
    monkeypatch.setattr(sidecar, "_detected_boxes", lambda **_kwargs: [])
    monkeypatch.setattr(sidecar, "_recognize_crops", lambda **_kwargs: recognized)
    monkeypatch.setattr(
        sidecar,
        "_ordered_lines",
        lambda lines, *, row_band_mode: observed_modes.append(row_band_mode) or list(lines),
    )

    pages = sidecar._predict_pages(
        engine,
        (sidecar._Page(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00"),),
    )

    assert observed_modes == ["harmonic_v1"]
    assert pages == [{"page_number": 1, "text": "line", "confidence": 0.9}]


def test_model_inventory_is_exactly_the_known_seven_files(tmp_path: Path) -> None:
    root = tmp_path / "models"
    raw, digest = _write_model_inventory(root)

    sidecar._verify_model_inventory(raw, expected_sha256=digest, model_root=root)

    (root / "text-rec" / "undeclared.bin").write_bytes(b"extra")
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="model_drift"):
        sidecar._verify_model_inventory(raw, expected_sha256=digest, model_root=root)


def test_v3_mobile_inventory_has_its_own_exact_eight_file_contract(tmp_path: Path) -> None:
    root = tmp_path / "models"
    raw, digest = _write_model_inventory(root, model_spec="mobile_det")

    sidecar._verify_model_inventory(
        raw,
        expected_sha256=digest,
        model_root=root,
        model_spec="mobile_det",
    )
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="model_invalid"):
        sidecar._verify_model_inventory(
            raw,
            expected_sha256=digest,
            model_root=root,
            model_spec="server_det",
        )


def test_v12_trained_recognizer_inventory_reuses_only_the_exact_mobile_eight_file_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    raw, digest = _write_model_inventory(root, model_spec="mobile_det_trained_rec")

    sidecar._verify_model_inventory(
        raw,
        expected_sha256=digest,
        model_root=root,
        model_spec="mobile_det_trained_rec",
    )
    (root / "text-rec" / "undeclared.bin").write_bytes(b"extra")
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="model_drift"):
        sidecar._verify_model_inventory(
            raw,
            expected_sha256=digest,
            model_root=root,
            model_spec="mobile_det_trained_rec",
        )


def test_v19_character_dictionary_is_exactly_the_public_436_class_source(
    tmp_path: Path,
) -> None:
    recognizer = tmp_path / "models" / "text-rec"
    recognizer.mkdir(parents=True)
    characters = [chr(0x1000 + index) for index in range(436)]
    (recognizer / "config.json").write_text(
        json.dumps({"PostProcess": {"character_dict": characters}}), encoding="utf-8"
    )

    assert (
        len(
            sidecar._load_characters(
                model_root=tmp_path / "models",
                relative_directory="text-rec",
                model_spec="mobile_det_public_finetuned_rec",
            )
        )
        == 436
    )

    (recognizer / "config.json").write_text(
        json.dumps({"PostProcess": {"character_dict": characters[:-1]}}), encoding="utf-8"
    )
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="model_invalid"):
        sidecar._load_characters(
            model_root=tmp_path / "models",
            relative_directory="text-rec",
            model_spec="mobile_det_public_finetuned_rec",
        )


def test_ctc_decode_removes_blank_and_consecutive_duplicate_labels() -> None:
    text, confidence = sidecar._ctc_decode(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8],
        ],
        characters=("A", "B"),
    )

    assert text == "AB"
    assert confidence == pytest.approx(0.8)


def test_ctc_decode_accepts_only_the_known_optional_space_surplus() -> None:
    text, confidence = sidecar._ctc_decode(
        [
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.05, 0.8],
            [0.1, 0.05, 0.8, 0.05],
        ],
        characters=("A", "B"),
    )

    assert text == "A B"
    assert confidence == pytest.approx(0.8)


def test_ctc_decode_v18_maps_the_server_exports_appended_space_class() -> None:
    text, confidence = sidecar._ctc_decode(
        [
            [0.1, 0.8, 0.03, 0.03, 0.04],
            [0.1, 0.03, 0.03, 0.04, 0.8],
            [0.1, 0.03, 0.8, 0.03, 0.04],
        ],
        characters=("A", "B", " "),
        label_policy="append-space-v1",
    )

    assert text == "A B"
    assert confidence == pytest.approx(0.8)


def test_ctc_decode_v18_numpy_matches_scalar_probabilities_and_repeats() -> None:
    np = pytest.importorskip("numpy")
    probabilities = np.asarray(
        [
            [0.10, 0.80, 0.04, 0.03, 0.03],
            [0.20, 0.70, 0.04, 0.03, 0.03],
            [0.90, 0.04, 0.02, 0.02, 0.02],
            [0.10, 0.03, 0.03, 0.80, 0.04],
            [0.10, 0.03, 0.03, 0.04, 0.80],
            [0.20, 0.03, 0.03, 0.04, 0.70],
            [0.10, 0.03, 0.80, 0.03, 0.04],
        ],
        dtype="float32",
    )

    vectorized = sidecar._ctc_decode(
        probabilities,
        characters=("A", "B", " "),
        label_policy="append-space-v1",
    )
    scalar = sidecar._ctc_decode(
        probabilities.tolist(),
        characters=("A", "B", " "),
        label_policy="append-space-v1",
    )

    # The dictionary space and the export-appended space are different CTC
    # class indices even though both render as a space.
    assert vectorized[0] == scalar[0] == "A  B"
    assert vectorized[1] == pytest.approx(scalar[1], abs=1e-12)


def test_ctc_decode_v18_numpy_matches_scalar_logits() -> None:
    np = pytest.importorskip("numpy")
    logits = np.asarray(
        [
            [-2.0, 4.0, 1.0, -3.0, -4.0],
            [-2.0, 4.0, 1.0, -3.0, -4.0],
            [5.0, -2.0, -3.0, -4.0, -5.0],
            [-2.0, -3.0, -4.0, -5.0, 4.0],
            [-2.0, -3.0, 4.0, -4.0, -5.0],
        ],
        dtype="float32",
    )

    vectorized = sidecar._ctc_decode(
        logits,
        characters=("A", "B", " "),
        label_policy="append-space-v1",
    )
    scalar = sidecar._ctc_decode(
        logits.tolist(),
        characters=("A", "B", " "),
        label_policy="append-space-v1",
    )

    assert vectorized[0] == scalar[0] == "A B"
    assert vectorized[1] == pytest.approx(scalar[1], rel=1e-12)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(lambda np: np.zeros(5, dtype="float32"), id="one-dimensional"),
        pytest.param(lambda np: np.zeros((1, 1, 5), dtype="float32"), id="three-dimensional"),
        pytest.param(lambda np: np.zeros((0, 5), dtype="float32"), id="no-timesteps"),
        pytest.param(lambda np: np.zeros((1_025, 5), dtype="float32"), id="too-many-timesteps"),
        pytest.param(lambda np: np.zeros((1, 4), dtype="float32"), id="class-count"),
        pytest.param(
            lambda np: np.asarray([[0.1, 0.8, 0.1, float("nan"), 0.0]]),
            id="not-finite-nan",
        ),
        pytest.param(
            lambda np: np.asarray([[0.1, 0.8, 0.1, float("inf"), 0.0]]),
            id="not-finite-inf",
        ),
    ],
)
def test_ctc_decode_v18_numpy_rejects_invalid_tensor(
    invalid: Any,
) -> None:
    np = pytest.importorskip("numpy")

    with pytest.raises(sidecar.DirectPaddleSidecarError, match="output_invalid"):
        sidecar._ctc_decode(
            invalid(np),
            characters=("A", "B", " "),
            label_policy="append-space-v1",
        )


def test_ctc_decode_v18_numpy_never_calls_tensor_tolist() -> None:
    np = pytest.importorskip("numpy")

    def _forbidden_tolist(_self: object) -> list[object]:
        raise AssertionError("the V18 tensor must stay vectorized")

    no_to_list = type("_NoToList", (np.ndarray,), {"tolist": _forbidden_tolist})
    probabilities = np.asarray([[0.10, 0.80, 0.04, 0.03, 0.03]], dtype="float32").view(no_to_list)

    text, confidence = sidecar._ctc_decode(
        probabilities,
        characters=("A", "B", " "),
        label_policy="append-space-v1",
    )

    assert text == "A"
    assert confidence == pytest.approx(0.8)


def test_ctc_decode_rejects_an_unknown_class_count() -> None:
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="output_invalid"):
        sidecar._ctc_decode([[0.2, 0.2, 0.2, 0.2, 0.2]], characters=("A", "B"))


def test_polygon_area_keeps_an_ordinary_detector_rectangle_valid() -> None:
    assert sidecar._polygon_signed_area(((0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0))) == 24.0
    assert sidecar._polygon_signed_area(((0.0, 0.0), (8.0, 0.0), (16.0, 0.0))) == 0.0


def test_predictor_disables_mkldnn_before_native_creation() -> None:
    calls: list[str] = []
    predictor = object()

    class Config:
        def __init__(self, _model: str, _params: str) -> None:
            calls.append("construct")

        def disable_glog_info(self) -> None:
            calls.append("glog")

        def disable_gpu(self) -> None:
            calls.append("gpu")

        def disable_mkldnn(self) -> None:
            calls.append("mkldnn")

        def set_cpu_math_library_num_threads(self, value: int) -> None:
            assert value == 1
            calls.append("threads")

    def create_predictor(_config: Config) -> object:
        calls.append("create")
        return predictor

    paddle = SimpleNamespace(
        inference=SimpleNamespace(Config=Config, create_predictor=create_predictor)
    )

    assert sidecar._new_predictor(paddle=paddle, directory="model", cpu_threads=1) is predictor
    assert calls == ["construct", "glog", "gpu", "mkldnn", "threads", "create"]


def test_recognizer_only_policy_verifies_detector_mkldnn_is_disabled() -> None:
    class Config:
        def __init__(self, _model: str, _params: str) -> None:
            self.enabled = False

        def disable_glog_info(self) -> None: ...

        def disable_gpu(self) -> None: ...

        def disable_mkldnn(self) -> None:
            self.enabled = False

        def mkldnn_enabled(self) -> bool:
            return self.enabled

        def set_cpu_math_library_num_threads(self, _value: int) -> None: ...

    predictor = object()
    paddle = SimpleNamespace(
        inference=SimpleNamespace(
            Config=Config,
            create_predictor=lambda _config: predictor,
        )
    )

    assert (
        sidecar._new_predictor(
            paddle=paddle,
            directory="model",
            cpu_threads=6,
            verify_mkldnn_disabled=True,
        )
        is predictor
    )


def test_v15_predictor_requires_build_support_and_bounds_mkldnn_cache() -> None:
    calls: list[object] = []
    predictor = object()

    class Config:
        def __init__(self, _model: str, _params: str) -> None:
            calls.append("construct")

        def disable_glog_info(self) -> None:
            calls.append("glog")

        def disable_gpu(self) -> None:
            calls.append("gpu")

        def enable_mkldnn(self) -> None:
            calls.append("enable_mkldnn")

        def set_mkldnn_cache_capacity(self, value: int) -> None:
            calls.append(("cache", value))

        def mkldnn_enabled(self) -> bool:
            calls.append("verify_mkldnn")
            return True

        def set_cpu_math_library_num_threads(self, value: int) -> None:
            calls.append(("threads", value))

    paddle = SimpleNamespace(
        base=SimpleNamespace(core=SimpleNamespace(is_compiled_with_mkldnn=lambda: True)),
        inference=SimpleNamespace(
            Config=Config,
            create_predictor=lambda _config: calls.append("create") or predictor,
        ),
    )

    assert (
        sidecar._new_predictor(
            paddle=paddle,
            directory="model",
            cpu_threads=6,
            enable_mkldnn=True,
            mkldnn_cache_capacity=10,
        )
        is predictor
    )
    assert calls == [
        "construct",
        "glog",
        "gpu",
        "enable_mkldnn",
        ("cache", 10),
        "verify_mkldnn",
        ("threads", 6),
        "create",
    ]

    paddle.base.core.is_compiled_with_mkldnn = lambda: False
    with pytest.raises(sidecar.DirectPaddleSidecarError, match="onednn_config_unavailable"):
        sidecar._new_predictor(
            paddle=paddle,
            directory="model",
            cpu_threads=6,
            enable_mkldnn=True,
            mkldnn_cache_capacity=10,
        )


def test_one_shot_response_uses_the_existing_text_sidecar_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _profile()
    monkeypatch.setattr(sidecar, "configuration_from_environment", lambda: (profile, tmp_path))
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: object())
    monkeypatch.setattr(
        sidecar,
        "_predict_pages",
        lambda *_args: [{"page_number": 1, "text": "OK", "confidence": 0.9}],
    )
    output = io.BytesIO()

    sidecar.run_once_from_stdio(io.BytesIO(_request_payload()), output)

    assert json.loads(output.getvalue()) == {
        "protocol": "customs-raster-text-sidecar-v1",
        "source_sha256": _SHA,
        "raster_text_profile_revision": _REVISION,
        "provider_revision": profile.extractor_revision,
        "pages": [{"page_number": 1, "text": "OK", "confidence": 0.9}],
        "warnings": [],
        "needs_review_reasons": [],
    }


def test_framed_service_reuses_engine_and_rechecks_before_and_after_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _profile()
    checks: list[str] = []
    engine = object()
    monkeypatch.setattr(sidecar, "configuration_from_environment", lambda: (profile, tmp_path))
    monkeypatch.setattr(sidecar, "_engine", lambda *_args: engine)
    monkeypatch.setattr(
        sidecar,
        "_verify_live_materials",
        lambda **_kwargs: checks.append("checked"),
    )

    def predict(observed_engine: object, _pages: object) -> list[dict[str, object]]:
        assert observed_engine is engine
        return [{"page_number": 1, "text": "OK", "confidence": 0.9}]

    monkeypatch.setattr(sidecar, "_predict_pages", predict)
    output = io.BytesIO()
    sidecar.run_service_from_stdio(io.BytesIO(_frame(_request_payload())), output)

    frames = _frames(output.getvalue())
    assert frames[0] == {
        "protocol": "customs-raster-text-sidecar-v1-service",
        "status": "ready",
        "raster_text_profile_revision": _REVISION,
        "provider_revision": profile.extractor_revision,
    }
    assert frames[1]["source_sha256"] == _SHA
    assert checks == ["checked", "checked", "checked"]


def test_pre_ready_warmup_uses_only_the_fixed_valid_zero_tensor_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_calls: list[tuple[tuple[int, ...], str]] = []
    predictor_calls: list[tuple[object, tuple[int, ...], object]] = []

    class _Zeros:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class _Numpy:
        def zeros(self, shape: tuple[int, ...], *, dtype: str) -> _Zeros:
            zero_calls.append((shape, dtype))
            return _Zeros(shape)

    numpy = _Numpy()
    detector = object()
    recognizer = object()

    def record_predictor(*, predictor: object, batch: _Zeros, np: object) -> object:
        predictor_calls.append((predictor, batch.shape, np))
        return object()

    monkeypatch.setattr(sidecar, "_run_predictor", record_predictor)

    sidecar._warm_engine(
        engine=cast(
            Any,
            SimpleNamespace(detector=detector, recognizer=recognizer, np=numpy, profile=_profile()),
        )
    )

    assert zero_calls == [
        ((1, 3, 32, 32), "float32"),
        ((1, 3, 48, 160), "float32"),
    ]
    assert predictor_calls == [
        (detector, (1, 3, 32, 32), numpy),
        (recognizer, (1, 3, 48, 160), numpy),
    ]

    zero_calls.clear()
    predictor_calls.clear()
    sidecar._warm_engine(
        engine=cast(
            Any,
            SimpleNamespace(
                detector=detector,
                recognizer=recognizer,
                np=numpy,
                profile=replace(
                    _profile(),
                    enable_mkldnn=True,
                    warmup_shape_policy="detector-736x960-recognizer-48x320-v1",
                ),
            ),
        )
    )
    assert zero_calls == [
        ((1, 3, 736, 960), "float32"),
        ((1, 3, 48, 320), "float32"),
    ]


def test_v19_warmup_requires_the_exact_438_class_native_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Numpy:
        @staticmethod
        def zeros(shape: tuple[int, ...], *, dtype: str) -> SimpleNamespace:
            assert dtype == "float32"
            return SimpleNamespace(shape=shape)

    profile = replace(
        _profile(),
        model_spec="mobile_det_public_finetuned_rec",
        enable_mkldnn=True,
        warmup_shape_policy="detector-736x960-recognizer-48x320-v1",
    )
    engine = cast(
        Any,
        SimpleNamespace(
            detector=object(),
            recognizer=object(),
            np=_Numpy(),
            profile=profile,
            characters=tuple(chr(0x1000 + index) for index in range(436)),
        ),
    )
    outputs = iter(
        (
            SimpleNamespace(ndim=4, shape=(1, 1, 1, 1)),
            SimpleNamespace(ndim=3, shape=(1, 40, 438)),
        )
    )
    monkeypatch.setattr(sidecar, "_run_predictor", lambda **_kwargs: next(outputs))
    sidecar._warm_engine(engine=engine)

    invalid_outputs = iter(
        (
            SimpleNamespace(ndim=4, shape=(1, 1, 1, 1)),
            SimpleNamespace(ndim=3, shape=(1, 40, 437)),
        )
    )
    monkeypatch.setattr(sidecar, "_run_predictor", lambda **_kwargs: next(invalid_outputs))
    with pytest.raises(
        sidecar.DirectPaddleSidecarError,
        match="onednn_recognizer_warmup_unavailable",
    ):
        sidecar._warm_engine(engine=engine)


def test_framed_service_and_session_publish_ready_only_after_pre_ready_warmup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = replace(_profile(), pre_ready_warmup=True)
    events: list[str] = []
    engine = SimpleNamespace(profile=profile)

    def make_engine(*_args: object) -> object:
        events.append("engine")
        return engine

    monkeypatch.setattr(sidecar, "configuration_from_environment", lambda: (profile, tmp_path))
    monkeypatch.setattr(sidecar, "_engine", make_engine)
    monkeypatch.setattr(
        sidecar,
        "_verify_live_materials",
        lambda **_kwargs: events.append("verify"),
    )
    monkeypatch.setattr(sidecar, "_warm_engine", lambda **_kwargs: events.append("warmup"))
    monkeypatch.setattr(
        sidecar,
        "_write_frame",
        lambda *_args: events.append("ready"),
    )

    for runner in (sidecar.run_service_from_stdio, sidecar.run_session_from_stdio):
        runner(io.BytesIO(), io.BytesIO())
        assert events == ["engine", "verify", "warmup", "verify", "ready"]
        events.clear()


def test_pre_ready_warmup_failure_does_not_publish_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = replace(_profile(), pre_ready_warmup=True)
    events: list[str] = []
    engine = SimpleNamespace(profile=profile)
    output = io.BytesIO()

    def make_engine(*_args: object) -> object:
        events.append("engine")
        return engine

    monkeypatch.setattr(sidecar, "configuration_from_environment", lambda: (profile, tmp_path))
    monkeypatch.setattr(sidecar, "_engine", make_engine)
    monkeypatch.setattr(
        sidecar,
        "_verify_live_materials",
        lambda **_kwargs: events.append("verify"),
    )

    def fail_warmup(**_kwargs: object) -> None:
        events.append("warmup")
        raise sidecar.DirectPaddleSidecarError("direct_paddle_sidecar_runtime_unavailable")

    monkeypatch.setattr(sidecar, "_warm_engine", fail_warmup)
    monkeypatch.setattr(
        sidecar,
        "_write_frame",
        lambda *_args: events.append("ready"),
    )

    with pytest.raises(sidecar.DirectPaddleSidecarError, match="runtime_unavailable"):
        sidecar.run_service_from_stdio(io.BytesIO(), output)

    assert events == ["engine", "verify", "warmup"]
    assert output.getvalue() == b""


def test_network_guard_blocks_connections() -> None:
    with sidecar.deny_network_connections():
        with pytest.raises(sidecar.DirectPaddleSidecarError, match="network_forbidden"):
            socket.create_connection(("127.0.0.1", 9))


def test_source_imports_no_high_level_ocr_wrapper_packages() -> None:
    tree = ast.parse(Path(sidecar.__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    assert "paddleocr" not in modules
    assert "paddlex" not in modules
    assert sidecar._RECOGNITION_BATCH_SIZE <= 8
