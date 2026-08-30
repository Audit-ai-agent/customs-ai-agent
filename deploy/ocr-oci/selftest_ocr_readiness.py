"""Network-free checks for exact Raster Text OCR readiness semantics."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("ocr_readiness.py")
SPEC = importlib.util.spec_from_file_location("ocr_readiness", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("readiness_module_unavailable")
READINESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READINESS)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.status = 200
        self._raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _maximum: int) -> bytes:
        return self._raw


class _Opener:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def open(self, _request, timeout: float):
        if timeout != 4.0:
            raise AssertionError("readiness timeout drift")
        return _Response(self._payload)


def main() -> int:
    revision = f"synthetic-raster-text-v1:{'c' * 24}"
    with tempfile.TemporaryDirectory(prefix="ocr-readiness-") as temporary:
        pointer = Path(temporary) / "raster-text-active.json"
        pointer_value = {
            "activation_kind": "source_attested_synthetic_raster_text",
            "current_runner_implementation_sha256": "a" * 64,
            "dataset_id": "synthetic-raster-text-v1",
            "engine_kind": (
                "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
                "lowbox_strip2_semantic_repair_v2_onednn"
            ),
            "evidence_generation_revision": "current-runner-native-full-gate-v1",
            "quality_gate_revision": "preview-draft-critical97-cer2-v1",
            "raster_text_profile_revision": revision,
            "runtime_validation_revision": "activation-full-runtime-compact-v1",
            "schema_version": "12.0",
            "status": "activated",
        }
        pointer.write_bytes(
            json.dumps(pointer_value, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
        )
        environment = {
            "CUSTOMS_AGENT_PORT": "18090",
            "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_PATH": str(pointer),
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": "A" * 43 + "bcd123456789",
            "CUSTOMS_AGENT_SERVICE_MODE": "local-development",
        }
        ready_payload: dict[str, object] = {
            "availability": "available",
            "raster_text_profile_revision": revision,
            "schema_version": "1.0",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(READINESS, "_DEFAULT_POINTER", pointer),
            patch.object(
                READINESS.urllib.request,
                "build_opener",
                return_value=_Opener(ready_payload),
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            if READINESS.main() != 0:
                raise AssertionError("matching OCR binding was not ready")

        wrong_payload = dict(ready_payload)
        wrong_payload["raster_text_profile_revision"] = f"synthetic-raster-text-v1:{'d' * 24}"
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(READINESS, "_DEFAULT_POINTER", pointer),
            patch.object(
                READINESS.urllib.request,
                "build_opener",
                return_value=_Opener(wrong_payload),
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            if READINESS.main() == 0:
                raise AssertionError("superseded OCR binding was accepted")

        pointer_value["schema_version"] = "11.0"
        pointer.write_bytes(
            json.dumps(pointer_value, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
        )
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(READINESS, "_DEFAULT_POINTER", pointer),
            patch.object(
                READINESS.urllib.request,
                "build_opener",
                return_value=_Opener(ready_payload),
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            if READINESS.main() == 0:
                raise AssertionError("legacy V11 OCR pointer was accepted")

    print("ocr_readiness_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
