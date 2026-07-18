from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import customs_ai_agent.ocr_release as release
from customs_ai_agent.adapters import OcrLayoutProviderError
from customs_ai_agent.ocr_identity import (
    OCR_MODEL_ASSETS_LOCK_SHA256,
    OCR_MODEL_SET_IDS,
)
from customs_ai_agent.ocr_release import (
    OcrReleaseGateAttestation,
    OcrReleaseGateAttestationError,
    installed_ocr_environment_lock_sha256,
    load_canonical_ocr_descriptor_identity,
    load_canonical_ocr_release_gate,
    verify_ocr_model_assets,
)
from customs_ai_agent.ocr_worker import _PaddleBackend
from customs_ai_agent.schemas.models import (
    BBox,
    OcrLayoutLedger,
    OcrLayoutPage,
    OcrLayoutSpan,
    stable_ocr_ledger_id,
)

_DUMMY_HASH = "0" * 64
_PADDLE_ENGINE = "PP-StructureV3:customs-paddle-layout-table-v4"
_DOCLING_ENGINE = "DocumentConverter:default-image-pipeline-offline-v2"


def _normalized_requirement_name(value: str) -> str:
    match = re.match(r"^([A-Za-z0-9_.-]+)", value.strip())
    assert match is not None
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def test_ocr_environment_roots_cover_agent_runtime_dependencies_and_lock_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = {
        _normalized_requirement_name(value) for value in project["project"]["dependencies"]
    }

    for provider in ("paddle", "docling"):
        environment_root = root / "evals" / "ocr_environments" / "v1"
        input_lines = (
            (environment_root / f"{provider}.requirements.in")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        input_dependencies = {
            _normalized_requirement_name(line)
            for line in input_lines
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert runtime_dependencies <= input_dependencies

        lock_path = environment_root / f"{provider}.requirements.lock.txt"
        lock_payload = lock_path.read_bytes()
        lock_text = lock_payload.decode("utf-8")
        locked_dependencies = {
            _normalized_requirement_name(line)
            for line in lock_text.splitlines()
            if re.match(r"^[A-Za-z0-9_.-]+==", line)
        }
        assert runtime_dependencies <= locked_dependencies
        assert (
            hashlib.sha256(lock_payload).hexdigest()
            == release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256[provider]
        )


def test_paddle_worker_release_identity_and_model_lock_are_consistent() -> None:
    lock_path = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "ocr_environments"
        / "v1"
        / "model-assets.lock.json"
    )
    payload = lock_path.read_bytes()
    lock = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == OCR_MODEL_ASSETS_LOCK_SHA256
    assert _PaddleBackend.engine_id == _PADDLE_ENGINE
    assert release.CANONICAL_OCR_PROVIDER_ENGINE_IDS["paddle"] == _PADDLE_ENGINE
    assert lock["providers"]["paddle"]["engine_id"] == _PADDLE_ENGINE


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _current_environment_lock() -> str:
    requirements: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        version = distribution.version
        assert isinstance(raw_name, str) and raw_name
        assert isinstance(version, str) and version
        normalized = _normalized_distribution_name(raw_name)
        if normalized == "customs-ai-agent":
            continue
        assert normalized not in requirements
        requirements[normalized] = (raw_name, version)
    return "".join(
        f"{raw_name}=={version} --hash=sha256:{_DUMMY_HASH}\n"
        for _normalized, (raw_name, version) in sorted(requirements.items())
    )


def _tree_identity(root: Path) -> tuple[int, int, str]:
    records: list[tuple[str, int, str]] = []
    for candidate in sorted(root.rglob("*"), key=lambda path: path.as_posix()):
        if candidate.is_file():
            payload = candidate.read_bytes()
            records.append(
                (
                    candidate.relative_to(root).as_posix(),
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )
            )
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_path, size, file_sha256 in records:
        total_bytes += size
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return len(records), total_bytes, digest.hexdigest()


def _model_lock(*, asset: dict[str, object]) -> dict[str, object]:
    placeholder = {
        "asset_id": "fixture/docling",
        "revision": "2" * 40,
        "file_count": 1,
        "total_bytes": 1,
        "tree_sha256": "3" * 64,
    }
    return {
        "schema_version": release.MODEL_ASSETS_LOCK_SCHEMA,
        "tree_hash_algorithm": release.MODEL_ASSETS_TREE_ALGORITHM,
        "providers": {
            "paddle": {
                "assets": [asset],
                "cache_layout": "paddlex_official_models",
                "engine_id": _PADDLE_ENGINE,
            },
            "docling": {
                "assets": [placeholder],
                "cache_layout": "huggingface_hub_snapshot",
                "engine_id": _DOCLING_ENGINE,
            },
        },
    }


@pytest.mark.skipif(
    sys.platform != "win32" or sys.version_info[:2] != (3, 11),
    reason="The release dependency lock is intentionally Windows/Python 3.11 only.",
)
def test_environment_attestation_accepts_only_exact_distribution_inventory(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "requirements.lock.txt"
    lock_path.write_text(_current_environment_lock(), encoding="utf-8")

    observed = installed_ocr_environment_lock_sha256(Path(sys.executable), lock_path)

    assert observed == hashlib.sha256(lock_path.read_bytes()).hexdigest()

    lock_path.write_text(
        lock_path.read_text(encoding="utf-8")
        + f"release-attestation-extra==1.0 --hash=sha256:{_DUMMY_HASH}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        OcrReleaseGateAttestationError,
        match="ocr_release_gate_environment_inventory_mismatch",
    ):
        installed_ocr_environment_lock_sha256(Path(sys.executable), lock_path)


def test_paddle_model_asset_attestation_rejects_drift_and_unlisted_cache_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "paddle-cache"
    model_root = cache_root / "official_models" / "fixture-model"
    model_root.mkdir(parents=True)
    model_file = model_root / "model.bin"
    model_file.write_bytes(b"locked-model")
    revision = "1" * 40
    refs = cache_root / ".cache" / "huggingface" / "hub" / "models--fixture--fixture-model" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision, encoding="ascii")
    file_count, total_bytes, tree_sha256 = _tree_identity(model_root)
    asset = {
        "asset_id": "fixture/fixture-model",
        "revision": revision,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": tree_sha256,
    }
    lock_path = tmp_path / "model-assets.lock.json"
    lock_path.write_text(
        json.dumps(_model_lock(asset=asset), sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "_resolved_cache_root", lambda _provider: cache_root)

    observed = verify_ocr_model_assets("paddle", lock_path, _PADDLE_ENGINE)

    assert observed == hashlib.sha256(lock_path.read_bytes()).hexdigest()

    model_file.write_bytes(b"drifted-model")
    with pytest.raises(
        OcrReleaseGateAttestationError,
        match="ocr_release_gate_model_asset_inventory_mismatch",
    ):
        verify_ocr_model_assets("paddle", lock_path, _PADDLE_ENGINE)

    model_file.write_bytes(b"locked-model")
    unexpected_model = cache_root / "official_models" / "unlisted-model"
    unexpected_model.mkdir()
    (unexpected_model / "model.bin").write_bytes(b"not-attested")
    with pytest.raises(
        OcrReleaseGateAttestationError,
        match="ocr_release_gate_model_cache_inventory_mismatch",
    ):
        verify_ocr_model_assets("paddle", lock_path, _PADDLE_ENGINE)


def test_docling_model_asset_attestation_rejects_unlisted_provider_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "huggingface" / "hub"
    revision = "4" * 40
    repository = cache_root / "models--docling-project--fixture-model"
    model_root = repository / "snapshots" / revision
    model_root.mkdir(parents=True)
    (model_root / "model.bin").write_bytes(b"locked-docling-model")
    refs = repository / "refs"
    refs.mkdir()
    (refs / "main").write_text(revision, encoding="ascii")
    file_count, total_bytes, tree_sha256 = _tree_identity(model_root)
    asset = {
        "asset_id": "docling-project/fixture-model",
        "revision": revision,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": tree_sha256,
    }
    paddle_placeholder = {
        "asset_id": "PaddlePaddle/fixture-paddle",
        "revision": "5" * 40,
        "file_count": 1,
        "total_bytes": 1,
        "tree_sha256": "6" * 64,
    }
    lock = _model_lock(asset=paddle_placeholder)
    providers = cast(dict[str, Any], lock["providers"])
    docling = cast(dict[str, Any], providers["docling"])
    docling["assets"] = [asset]
    lock_path = tmp_path / "model-assets.lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    unrelated = cache_root / "models--BAAI--unrelated"
    unrelated.mkdir(parents=True)
    monkeypatch.setattr(release, "_resolved_cache_root", lambda _provider: cache_root)

    observed = verify_ocr_model_assets("docling", lock_path, _DOCLING_ENGINE)

    assert observed == hashlib.sha256(lock_path.read_bytes()).hexdigest()

    unlisted = cache_root / "models--docling-project--unlisted"
    unlisted.mkdir()
    with pytest.raises(
        OcrReleaseGateAttestationError,
        match="ocr_release_gate_model_cache_inventory_mismatch",
    ):
        verify_ocr_model_assets("docling", lock_path, _DOCLING_ENGINE)


@pytest.mark.parametrize(
    ("environment_sha256", "model_sha256", "expected_code"),
    [
        (
            "f" * 64,
            release.CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256,
            "ocr_release_gate_environment_lock_not_canonical",
        ),
        (
            release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256["paddle"],
            "f" * 64,
            "ocr_release_gate_model_assets_lock_not_canonical",
        ),
    ],
)
def test_canonical_loader_rejects_caller_pinned_noncanonical_locks_before_io(
    environment_sha256: str,
    model_sha256: str,
    expected_code: str,
) -> None:
    missing = Path("must-not-be-read")

    with pytest.raises(OcrReleaseGateAttestationError, match=expected_code):
        load_canonical_ocr_release_gate(
            report_path=missing,
            expected_report_sha256="a" * 64,
            expected_provider="paddle",
            descriptor_path=missing,
            rendered_root=missing,
            expected_agent_wheel_sha256="b" * 64,
            environment_lock_path=missing,
            expected_environment_lock_sha256=environment_sha256,
            model_assets_lock_path=missing,
            expected_model_assets_lock_sha256=model_sha256,
            ocr_python=missing,
        )


def test_canonical_loader_fails_closed_until_replacement_descriptor_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release, "CANONICAL_OCR_DESCRIPTOR_SHA256", None)
    monkeypatch.setattr(release, "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA", None)
    missing = Path("must-not-be-read-before-descriptor-freeze")

    with pytest.raises(
        OcrReleaseGateAttestationError,
        match="ocr_release_gate_descriptor_not_frozen",
    ):
        load_canonical_ocr_release_gate(
            report_path=missing,
            expected_report_sha256="a" * 64,
            expected_provider="paddle",
            descriptor_path=missing,
            rendered_root=missing,
            expected_agent_wheel_sha256="b" * 64,
            environment_lock_path=missing,
            expected_environment_lock_sha256=release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256[
                "paddle"
            ],
            model_assets_lock_path=missing,
            expected_model_assets_lock_sha256=release.CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256,
            ocr_python=missing,
        )


def test_canonical_loader_rejects_report_engine_outside_code_owned_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "descriptor.json"
    authorization_commit_sha = "9" * 40
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": "ocr-benchmark-descriptor/v3",
                "evaluation_version": "evaluation-v3",
                "release_context": {
                    "authorization_commit_sha": authorization_commit_sha,
                    "authorization_receipt_sha256": "7" * 64,
                    "authorization_receipt_registry_key": "8" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rendered_root = tmp_path / "rendered"
    rendered_root.mkdir()
    manifest = rendered_root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_DESCRIPTOR_SHA256",
        hashlib.sha256(descriptor.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA",
        authorization_commit_sha,
    )
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    wrong_engine = OcrReleaseGateAttestation(
        report_sha256="a" * 64,
        descriptor_sha256=release.CANONICAL_OCR_DESCRIPTOR_SHA256,
        manifest_sha256=release.CANONICAL_OCR_MANIFEST_SHA256,
        agent_wheel_sha256="b" * 64,
        environment_lock_sha256=release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256["paddle"],
        model_assets_lock_sha256=release.CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256,
        provider="paddle",
        provider_id="paddle",
        engine_id="tampered-engine",
        engine_version="1.0",
    )
    monkeypatch.setattr(
        release,
        "load_ocr_release_runtime_gate",
        lambda **_kwargs: wrong_engine,
    )

    with pytest.raises(
        OcrReleaseGateAttestationError,
        match="ocr_release_gate_engine_not_canonical",
    ):
        load_canonical_ocr_release_gate(
            report_path=tmp_path / "report.json",
            expected_report_sha256="a" * 64,
            expected_provider="paddle",
            descriptor_path=descriptor,
            rendered_root=rendered_root,
            expected_agent_wheel_sha256="b" * 64,
            environment_lock_path=tmp_path / "requirements.lock.txt",
            expected_environment_lock_sha256=release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256[
                "paddle"
            ],
            model_assets_lock_path=tmp_path / "model-assets.lock.json",
            expected_model_assets_lock_sha256=release.CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256,
            ocr_python=Path(sys.executable),
        )


def test_canonical_loader_seals_service_runtime_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authorization_commit_sha = "9" * 40
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": "ocr-benchmark-descriptor/v3",
                "evaluation_version": "evaluation-v3",
                "release_context": {
                    "authorization_commit_sha": authorization_commit_sha,
                    "authorization_receipt_sha256": "7" * 64,
                    "authorization_receipt_registry_key": "8" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rendered_root = tmp_path / "rendered"
    rendered_root.mkdir()
    manifest = rendered_root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    descriptor_sha256 = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_DESCRIPTOR_SHA256",
        descriptor_sha256,
    )
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA",
        authorization_commit_sha,
    )
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_MANIFEST_SHA256",
        manifest_sha256,
    )
    attestation = OcrReleaseGateAttestation(
        report_sha256="a" * 64,
        descriptor_sha256=descriptor_sha256,
        manifest_sha256=manifest_sha256,
        agent_wheel_sha256="b" * 64,
        environment_lock_sha256=release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256["paddle"],
        model_assets_lock_sha256=release.CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256,
        provider="paddle",
        provider_id="paddle",
        engine_id=_PADDLE_ENGINE,
        engine_version="1.0",
    )
    monkeypatch.setattr(
        release,
        "load_ocr_release_runtime_gate",
        lambda **_kwargs: attestation,
    )

    sealed = load_canonical_ocr_release_gate(
        report_path=tmp_path / "report.json",
        expected_report_sha256="a" * 64,
        expected_provider="paddle",
        descriptor_path=descriptor,
        rendered_root=rendered_root,
        expected_agent_wheel_sha256="b" * 64,
        environment_lock_path=tmp_path / "requirements.lock.txt",
        expected_environment_lock_sha256=release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256["paddle"],
        model_assets_lock_path=tmp_path / "model-assets.lock.json",
        expected_model_assets_lock_sha256=release.CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256,
        ocr_python=Path(sys.executable),
    )
    provider = release.ReleaseLockedOcrProvider(object(), sealed)

    provider.assert_service_ready()
    copied = release.ReleaseLockedOcrProvider(object(), replace(sealed))
    with pytest.raises(
        OcrLayoutProviderError,
        match="ocr_release_gate_identity_mismatch",
    ):
        copied.assert_service_ready()


def test_canonical_descriptor_rejects_authorization_commit_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": "ocr-benchmark-descriptor/v3",
                "evaluation_version": "evaluation-v3",
                "release_context": {
                    "authorization_commit_sha": "1" * 40,
                    "authorization_receipt_sha256": "2" * 64,
                    "authorization_receipt_registry_key": "3" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_DESCRIPTOR_SHA256",
        hashlib.sha256(descriptor.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA",
        "4" * 40,
    )

    with pytest.raises(
        OcrReleaseGateAttestationError,
        match="ocr_release_gate_descriptor_authorization_mismatch",
    ):
        load_canonical_ocr_descriptor_identity(descriptor)


def test_runtime_provider_is_not_service_ready_before_release_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release, "CANONICAL_OCR_DESCRIPTOR_SHA256", None)
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA",
        None,
    )
    attestation = OcrReleaseGateAttestation(
        report_sha256="a" * 64,
        descriptor_sha256="b" * 64,
        manifest_sha256=release.CANONICAL_OCR_MANIFEST_SHA256,
        agent_wheel_sha256="d" * 64,
        environment_lock_sha256=release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256["paddle"],
        model_assets_lock_sha256=OCR_MODEL_ASSETS_LOCK_SHA256,
        provider="paddle",
        provider_id="paddle",
        engine_id=_PADDLE_ENGINE,
        engine_version="1.0",
    )
    provider = release.ReleaseLockedOcrProvider(object(), attestation)

    with pytest.raises(
        OcrLayoutProviderError,
        match="ocr_release_gate_not_frozen",
    ):
        provider.assert_service_ready()


def test_runtime_provider_rejects_handmade_attestation_even_if_hashes_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_sha256 = "b" * 64
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_DESCRIPTOR_SHA256",
        descriptor_sha256,
    )
    monkeypatch.setattr(
        release,
        "CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA",
        "4" * 40,
    )
    attestation = OcrReleaseGateAttestation(
        report_sha256="a" * 64,
        descriptor_sha256=descriptor_sha256,
        manifest_sha256=release.CANONICAL_OCR_MANIFEST_SHA256,
        agent_wheel_sha256="d" * 64,
        environment_lock_sha256=release.CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256["paddle"],
        model_assets_lock_sha256=OCR_MODEL_ASSETS_LOCK_SHA256,
        provider="paddle",
        provider_id="paddle",
        engine_id=_PADDLE_ENGINE,
        engine_version="1.0",
    )
    provider = release.ReleaseLockedOcrProvider(object(), attestation)

    with pytest.raises(
        OcrLayoutProviderError,
        match="ocr_release_gate_identity_mismatch",
    ):
        provider.assert_service_ready()


@pytest.mark.parametrize(
    ("model_id", "model_assets_lock_sha256"),
    [
        ("tampered-model-set", OCR_MODEL_ASSETS_LOCK_SHA256),
        (OCR_MODEL_SET_IDS["paddle"], "f" * 64),
    ],
)
def test_runtime_provider_rejects_model_set_or_asset_lock_drift(
    model_id: str,
    model_assets_lock_sha256: str,
) -> None:
    page = OcrLayoutPage(
        page_number=1,
        image_sha256="1" * 64,
        width_px=10,
        height_px=10,
        spans=[
            OcrLayoutSpan(
                span_id="span-1",
                page=1,
                reading_order=0,
                role="text",
                text="fixture",
                bbox=BBox(page=1, x1=0, y1=0, x2=5, y2=5),
                confidence=1.0,
            )
        ],
    )
    ledger_values = {
        "document_id": "document-1",
        "provider_id": "paddle",
        "engine_id": _PADDLE_ENGINE,
        "engine_version": "1.0",
        "model_id": model_id,
        "model_assets_lock_sha256": model_assets_lock_sha256,
        "pages": [page],
    }
    ledger = OcrLayoutLedger(
        ledger_id=stable_ocr_ledger_id(**ledger_values),
        **ledger_values,
    )

    class Provider:
        def analyze(self, **_kwargs):
            return ledger

    attestation = OcrReleaseGateAttestation(
        report_sha256="a" * 64,
        descriptor_sha256="b" * 64,
        manifest_sha256="c" * 64,
        agent_wheel_sha256="d" * 64,
        environment_lock_sha256="e" * 64,
        model_assets_lock_sha256=OCR_MODEL_ASSETS_LOCK_SHA256,
        provider="paddle",
        provider_id="paddle",
        engine_id=_PADDLE_ENGINE,
        engine_version="1.0",
    )
    provider = release.ReleaseLockedOcrProvider(Provider(), attestation)

    with pytest.raises(
        OcrLayoutProviderError,
        match="ocr_release_gate_runtime_model_mismatch",
    ):
        provider.analyze(document_id="document-1", pages=[])
