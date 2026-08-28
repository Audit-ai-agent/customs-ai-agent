from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionResultV1,
    RasterTextInputV1,
    RasterTextPageV1,
    canonical_raster_text,
    canonical_raster_text_sha256,
)


_PROFILE_REVISION = "synthetic-raster-text-v1:0123456789abcdef01234567"
_EXTRACTOR_REVISION = "direct-text-checkpoint-test-v1"


@dataclass
class _CandidateHarness:
    root: Path
    assets: list[Any]
    state: dict[str, Any]
    extraction_calls: list[str]
    verified_assets: list[str]
    session_starts: list[bool]
    pointer_path: Path
    pointer_raw: bytes


def _install_candidate_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    asset_count: int = 3,
) -> _CandidateHarness:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    pointer_path = root / "approvals" / "raster-text-active.json"
    pointer_path.parent.mkdir()
    pointer_raw = b"existing-pointer-must-not-change"
    pointer_path.write_bytes(pointer_raw)
    assets = [
        SimpleNamespace(
            asset_id=f"asset-{index}",
            profile_id="png-216",
            media_type="image/png",
            image_sha256=f"{index:x}" * 64,
            gold_sha256=f"{index + 8:x}" * 64,
            byte_size=1,
        )
        for index in range(1, asset_count + 1)
    ]
    state: dict[str, Any] = {
        "dataset_sha256": "a" * 64,
        "source_gold_sha256": "b" * 64,
        "runtime_configuration_sha256": "c" * 64,
        "profile_revision": _PROFILE_REVISION,
        "extractor_revision": _EXTRACTOR_REVISION,
        "runner_sha256": "d" * 64,
        "fail_on": None,
        "tick": 0.0,
    }
    extraction_calls: list[str] = []
    verified_assets: list[str] = []
    session_starts: list[bool] = []

    class Session:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Provider:
        def open_evaluation_session(self, *, parent_materials_verified: bool) -> Session:
            session_starts.append(parent_materials_verified)
            return Session()

    provider = Provider()

    def resolve_candidate(**_kwargs: object) -> SimpleNamespace:
        profile = SimpleNamespace(
            raster_text_profile_revision=state["profile_revision"],
            extractor_revision=state["extractor_revision"],
        )
        return SimpleNamespace(
            status="available",
            engine=SimpleNamespace(input_adapter=object(), provider=provider),
            profile=profile,
            runtime_configuration_sha256=state["runtime_configuration_sha256"],
        )

    def verified_asset(
        _root: Path, asset: Any, *, pdfium_version: str
    ) -> bytes:
        assert pdfium_version == "4.30.0"
        verified_assets.append(asset.asset_id)
        return b"x"

    def extract_candidate(
        _engine: object,
        *,
        raster_input: RasterTextInputV1,
        **_kwargs: object,
    ) -> RasterTextExtractionResultV1:
        asset_id = raster_input.metadata.document_asset_id
        extraction_calls.append(asset_id)
        if state["fail_on"] == asset_id:
            raise RuntimeError("content that must never escape")
        asset = next(item for item in assets if item.asset_id == asset_id)
        pages = [RasterTextPageV1(page_number=1, text=f"TEXT-{asset_id}", confidence=0.9)]
        return RasterTextExtractionResultV1(
            document_asset_id=asset.asset_id,
            extraction_request_id=raster_input.metadata.extraction_request_id,
            raster_text_profile_revision=state["profile_revision"],
            status="preview_ready",
            media_type=asset.media_type,
            byte_size=asset.byte_size,
            source_sha256=asset.image_sha256,
            canonical_text=canonical_raster_text(pages),
            canonical_text_sha256=canonical_raster_text_sha256(pages),
            extractor_revision=state["extractor_revision"],
            pages=pages,
            warnings=[],
            needs_review_reasons=[],
        )

    def monotonic() -> float:
        state["tick"] += 1.0
        return state["tick"]

    monkeypatch.setattr(
        activation,
        "_load_dataset_receipt",
        lambda *_args, **_kwargs: (
            SimpleNamespace(pdfium_version="4.30.0"),
            state["dataset_sha256"],
        ),
    )
    monkeypatch.setattr(
        activation,
        "write_source_attested_raster_text_gold_receipt",
        lambda **_kwargs: ("source-gold.json", state["source_gold_sha256"]),
    )
    monkeypatch.setattr(activation, "resolve_paddleocr3_raster_text_candidate", resolve_candidate)
    monkeypatch.setattr(activation, "_scope_assets", lambda *_args: list(assets))
    monkeypatch.setattr(activation, "_verified_asset", verified_asset)
    monkeypatch.setattr(
        activation,
        "inspect_raster_document_structure",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(activation, "extract_raster_text_preview", extract_candidate)
    monkeypatch.setattr(
        activation,
        "_runner_implementation_sha256",
        lambda: state["runner_sha256"],
    )
    monkeypatch.setattr(activation.time, "monotonic", monotonic)
    return _CandidateHarness(
        root=root,
        assets=assets,
        state=state,
        extraction_calls=extraction_calls,
        verified_assets=verified_assets,
        session_starts=session_starts,
        pointer_path=pointer_path,
        pointer_raw=pointer_raw,
    )


def _run(harness: _CandidateHarness) -> activation.RasterTextCandidateRunV1:
    return activation.run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=harness.root,
        runtime_configuration_relative_path="profiles/runtime.json",
        synthetic_raster_receipt_relative_path="synthetic-raster-v1/receipt.json",
        scope="calibration",
    )


def _checkpoints(harness: _CandidateHarness) -> list[Path]:
    return sorted(
        harness.root.glob(
            "synthetic-raster-text-v1/candidate-checkpoints/calibration/*/*.json"
        )
    )


def _expected_checkpoint_path(harness: _CandidateHarness, asset: Any) -> Path:
    identity = activation._candidate_checkpoint_identity(  # noqa: SLF001
        scope="calibration",
        synthetic_raster_receipt_sha256=harness.state["dataset_sha256"],
        source_gold_receipt_sha256=harness.state["source_gold_sha256"],
        runtime_configuration_sha256=harness.state["runtime_configuration_sha256"],
        raster_text_profile_revision=harness.state["profile_revision"],
        extractor_revision=harness.state["extractor_revision"],
        runner_implementation_sha256=harness.state["runner_sha256"],
        asset=asset,
    )
    relative, _digest = activation._candidate_checkpoint_relative_path(identity)  # noqa: SLF001
    return harness.root.joinpath(*relative.split("/"))


def test_candidate_checkpoints_resume_missing_only_and_preserve_output_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _install_candidate_harness(monkeypatch, tmp_path)
    harness.state["fail_on"] = "asset-2"

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_candidate_execution_failed$",
    ):
        _run(harness)

    assert harness.extraction_calls == ["asset-1", "asset-2"]
    assert len(_checkpoints(harness)) == 1
    assert harness.session_starts == [True]

    harness.state["fail_on"] = None
    harness.extraction_calls.clear()
    second = _run(harness)
    assert harness.extraction_calls == ["asset-2", "asset-3"]
    assert len(_checkpoints(harness)) == 3
    assert harness.session_starts == [True, True]

    execution = json.loads(
        harness.root.joinpath(*second.candidate_execution_relative_path.split("/")).read_bytes()
    )
    bundle = json.loads(
        harness.root.joinpath(*execution["candidate_bundle_relative_path"].split("/")).read_bytes()
    )
    assert [item["asset_id"] for item in bundle["candidates"]] == [
        "asset-1",
        "asset-2",
        "asset-3",
    ]

    harness.extraction_calls.clear()
    harness.verified_assets.clear()
    third = _run(harness)
    assert third.candidate_execution_sha256 == second.candidate_execution_sha256
    assert harness.extraction_calls == []
    assert harness.session_starts == [True, True]
    assert harness.verified_assets == ["asset-1", "asset-2", "asset-3"]
    assert harness.pointer_path.read_bytes() == harness.pointer_raw
    captured = capsys.readouterr()
    assert "TEXT-" not in captured.out
    assert "TEXT-" not in captured.err


def test_candidate_runner_ocr_executes_once_for_byte_identical_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_candidate_harness(monkeypatch, tmp_path)
    # Logical clean/error records may have different Gold while their actual
    # raster bytes are identical.  Gold must never participate in OCR reuse.
    harness.assets[1].image_sha256 = harness.assets[0].image_sha256

    result = _run(harness)

    assert harness.extraction_calls == ["asset-1", "asset-3"]
    assert harness.verified_assets == ["asset-1", "asset-2", "asset-3"]
    assert len(_checkpoints(harness)) == 3
    execution = json.loads(
        harness.root.joinpath(*result.candidate_execution_relative_path.split("/")).read_bytes()
    )
    bundle = json.loads(
        harness.root.joinpath(*execution["candidate_bundle_relative_path"].split("/")).read_bytes()
    )
    observations = {
        item["asset_id"]: item for item in bundle["candidates"]
    }
    assert observations["asset-1"]["candidate"]["canonical_text"] == "TEXT-asset-1"
    assert observations["asset-2"]["candidate"]["canonical_text"] == "TEXT-asset-1"
    assert observations["asset-2"]["candidate"]["document_asset_id"] == "asset-2"
    assert observations["asset-2"]["candidate"]["extraction_request_id"] == (
        activation._raster_text_request_id(harness.assets[1])  # noqa: SLF001
    )
    assert observations["asset-2"]["elapsed_seconds"] == observations["asset-1"][
        "elapsed_seconds"
    ]

    harness.extraction_calls.clear()
    harness.verified_assets.clear()
    second = _run(harness)
    assert second.candidate_execution_sha256 == result.candidate_execution_sha256
    assert harness.extraction_calls == []
    assert harness.verified_assets == ["asset-1", "asset-2", "asset-3"]
    assert harness.session_starts == [True]


@pytest.mark.parametrize("tamper_kind", ["digest", "noncanonical"])
def test_candidate_checkpoint_tamper_fails_without_opening_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    harness = _install_candidate_harness(monkeypatch, tmp_path, asset_count=1)
    _run(harness)
    checkpoint_path = _checkpoints(harness)[0]
    value = json.loads(checkpoint_path.read_bytes())
    if tamper_kind == "digest":
        value["observation_sha256"] = "0" * 64
        tampered_raw = activation._canonical_json(value)  # noqa: SLF001
    else:
        tampered_raw = json.dumps(value, indent=2).encode()
    checkpoint_path.write_bytes(tampered_raw)
    starts_before = list(harness.session_starts)

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_candidate_checkpoint_invalid$",
    ):
        _run(harness)

    assert harness.session_starts == starts_before
    assert checkpoint_path.read_bytes() == tampered_raw
    assert harness.pointer_path.read_bytes() == harness.pointer_raw


@pytest.mark.parametrize(
    "changed_binding",
    ["dataset", "source", "runtime", "runner", "image", "gold"],
)
def test_candidate_checkpoint_rejects_material_copied_across_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_binding: str,
) -> None:
    harness = _install_candidate_harness(monkeypatch, tmp_path, asset_count=1)
    _run(harness)
    old_raw = _checkpoints(harness)[0].read_bytes()
    replacements = {
        "dataset": (harness.state, "dataset_sha256", "1" * 64),
        "source": (harness.state, "source_gold_sha256", "2" * 64),
        "runtime": (harness.state, "runtime_configuration_sha256", "3" * 64),
        "runner": (harness.state, "runner_sha256", "4" * 64),
        "image": (harness.assets[0], "image_sha256", "5" * 64),
        "gold": (harness.assets[0], "gold_sha256", "6" * 64),
    }
    target, field, replacement = replacements[changed_binding]
    if isinstance(target, dict):
        target[field] = replacement
    else:
        setattr(target, field, replacement)
    mixed_path = _expected_checkpoint_path(harness, harness.assets[0])
    mixed_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_path.write_bytes(old_raw)
    starts_before = list(harness.session_starts)

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_candidate_checkpoint_invalid$",
    ):
        _run(harness)

    assert harness.session_starts == starts_before
    assert harness.pointer_path.read_bytes() == harness.pointer_raw


def test_candidate_runner_fails_if_implementation_drifts_during_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_candidate_harness(monkeypatch, tmp_path, asset_count=1)
    runner_hashes = iter(("d" * 64, "e" * 64))
    monkeypatch.setattr(
        activation,
        "_runner_implementation_sha256",
        lambda: next(runner_hashes),
    )

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_candidate_runner_drift$",
    ):
        _run(harness)

    assert len(_checkpoints(harness)) == 1
    assert harness.pointer_path.read_bytes() == harness.pointer_raw


def test_candidate_checkpoint_oversize_fails_before_asset_or_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_candidate_harness(monkeypatch, tmp_path, asset_count=1)
    _run(harness)
    checkpoint_path = _checkpoints(harness)[0]
    checkpoint_path.write_bytes(b"x" * (activation._MAX_CANDIDATE_CHECKPOINT_BYTES + 1))  # noqa: SLF001
    harness.verified_assets.clear()
    starts_before = list(harness.session_starts)

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_candidate_checkpoint_invalid$",
    ):
        _run(harness)

    assert harness.verified_assets == []
    assert harness.session_starts == starts_before


def test_candidate_checkpoint_symlink_fails_before_asset_or_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _install_candidate_harness(monkeypatch, tmp_path, asset_count=1)
    _run(harness)
    checkpoint_path = _checkpoints(harness)[0]
    outside = tmp_path / "outside-checkpoint.json"
    outside.write_bytes(checkpoint_path.read_bytes())
    checkpoint_path.unlink()
    try:
        checkpoint_path.symlink_to(outside)
    except OSError:
        # Windows CI accounts may lack symlink privileges.  Preserve the
        # same boundary assertion by making only this exact path report the
        # filesystem state that the loader must reject.
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == checkpoint_path or original_is_symlink(path),
        )
    harness.verified_assets.clear()
    starts_before = list(harness.session_starts)

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^raster_text_candidate_checkpoint_invalid$",
    ):
        _run(harness)

    assert harness.verified_assets == []
    assert harness.session_starts == starts_before


def test_candidate_checkpoint_identity_digest_binds_every_required_input() -> None:
    asset = SimpleNamespace(
        asset_id="asset-1",
        profile_id="png-216",
        image_sha256="8" * 64,
        gold_sha256="9" * 64,
    )
    identity = activation._candidate_checkpoint_identity(  # noqa: SLF001
        scope="calibration",
        synthetic_raster_receipt_sha256="a" * 64,
        source_gold_receipt_sha256="b" * 64,
        runtime_configuration_sha256="c" * 64,
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision=_EXTRACTOR_REVISION,
        runner_implementation_sha256="d" * 64,
        asset=asset,
    )
    variants = [
        identity.model_copy(update={"scope": "sealed"}),
        identity.model_copy(update={"synthetic_raster_receipt_sha256": "1" * 64}),
        identity.model_copy(update={"source_gold_receipt_sha256": "2" * 64}),
        identity.model_copy(update={"runtime_configuration_sha256": "3" * 64}),
        identity.model_copy(
            update={
                "raster_text_profile_revision": (
                    "synthetic-raster-text-v1:111111111111111111111111"
                )
            }
        ),
        identity.model_copy(update={"extractor_revision": "other-extractor"}),
        identity.model_copy(update={"runner_implementation_sha256": "4" * 64}),
        identity.model_copy(update={"asset_id": "asset-2"}),
        identity.model_copy(update={"profile_id": "jpeg-150-q75"}),
        identity.model_copy(update={"image_sha256": "5" * 64}),
        identity.model_copy(update={"gold_sha256": "6" * 64}),
    ]
    baseline_path, baseline_digest = activation._candidate_checkpoint_relative_path(  # noqa: SLF001
        identity
    )
    observed = {
        activation._candidate_checkpoint_relative_path(item)  # noqa: SLF001
        for item in variants
    }

    assert len(observed) == len(variants)
    assert all(path != baseline_path and digest != baseline_digest for path, digest in observed)


def test_candidate_checkpoint_observation_hash_is_canonical() -> None:
    pages = [RasterTextPageV1(page_number=1, text="TEXT", confidence=0.9)]
    candidate = RasterTextExtractionResultV1(
        document_asset_id="asset-1",
        extraction_request_id="raster-text-png-216-aaaaaaaaaaaaaaaaaaaaaaaa",
        raster_text_profile_revision=_PROFILE_REVISION,
        status="preview_ready",
        media_type="image/png",
        byte_size=1,
        source_sha256="a" * 64,
        canonical_text=canonical_raster_text(pages),
        canonical_text_sha256=canonical_raster_text_sha256(pages),
        extractor_revision=_EXTRACTOR_REVISION,
        pages=pages,
        warnings=[],
        needs_review_reasons=[],
    )
    observation = activation.RasterTextCandidateObservationV1(
        asset_id="asset-1",
        profile_id="png-216",
        elapsed_seconds=1.0,
        candidate=candidate,
    )

    assert activation._digest_model(observation) == hashlib.sha256(  # noqa: SLF001
        activation._canonical_json(  # noqa: SLF001
            observation.model_dump(mode="json", warnings=False)
        )
    ).hexdigest()
