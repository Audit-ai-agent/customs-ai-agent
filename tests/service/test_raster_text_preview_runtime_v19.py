"""V19 runtime dispatch regressions for the leakage-safe recognizer lane."""

from __future__ import annotations

from pathlib import Path

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.service import raster_text_preview_runtime as runtime


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_PROFILE_REVISION = f"synthetic-raster-text-v1:{'1' * 24}"
_EVIDENCE_REVISION = "source-attested-public-finetuned-staged-train-holdout-v2"


def _v7_pointer_fields() -> dict[str, object]:
    return {
        "metric_revision": "source-attested-semantic-text-v3",
        "evidence_revision": _EVIDENCE_REVISION,
        "latency_gate_revision": "warm-p95-10s-v1",
        "raster_text_profile_revision": _PROFILE_REVISION,
        "runtime_configuration_path": "runtime/v19.json",
        "runtime_configuration_sha256": _SHA_A,
        "sealed_evaluation_path": "evaluation/v7.json",
        "sealed_evaluation_sha256": _SHA_B,
        "source_recognizer_release_sha256": _SHA_A,
        "source_recognizer_receipt_sha256": _SHA_B,
        "source_recognizer_training_intent_sha256": _SHA_C,
        "source_recognizer_selection_receipt_path": (
            f"synthetic-raster-text-v1/fresh-scope-selections/{_SHA_C}.json"
        ),
        "source_recognizer_selection_receipt_sha256": _SHA_C,
        "source_recognizer_parent_selection_receipt_sha256": _SHA_D,
    }


def test_v19_uses_only_its_combined_bundle_verifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = runtime.RasterTextOfflineProfileV19.model_construct()
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_verify_v19_public_finetuned_model_bundle",
        lambda **_kwargs: calls.append("public-finetuned"),
    )
    monkeypatch.setattr(
        runtime,
        "_verify_v3_mobile_model_bundle",
        lambda **_kwargs: calls.append("historical-mobile"),
    )
    monkeypatch.setattr(
        runtime,
        "_verify_v18_server_model_bundle",
        lambda **_kwargs: calls.append("server-recognizer"),
    )

    runtime._verify_profile_model_bundle(root=tmp_path, profile=profile)

    assert calls == ["public-finetuned"]


def test_v7_pointer_parser_accepts_only_complete_canonical_v19_binding() -> None:
    pointer = runtime.RasterTextActivationPointerV7(**_v7_pointer_fields())
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))

    parsed = runtime.parse_raster_text_activation_pointer(raw)

    assert parsed == pointer
    missing = pointer.model_dump(mode="json", warnings=False)
    del missing["source_recognizer_receipt_sha256"]
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_activation_invalid"):
        runtime.parse_raster_text_activation_pointer(runtime._canonical_json_bytes(missing))
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_activation_invalid"):
        runtime.parse_raster_text_activation_pointer(raw.rstrip(b"\n"))


def test_v7_metric_binding_cannot_mix_v6_evidence() -> None:
    common = {
        "metric_revision": "source-attested-semantic-text-v3",
        "evidence_revision": _EVIDENCE_REVISION,
        "latency_gate_revision": "warm-p95-10s-v1",
    }
    pointer = runtime.RasterTextActivationPointerV7.model_construct(**common)
    evaluation = runtime.RasterTextSealedEvaluationV7.model_construct(**common)
    calibration = runtime.RasterTextCalibrationReceiptV7.model_construct(**common)

    assert (
        runtime._bound_raster_text_metric_revision(
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
        )
        == "source-attested-semantic-text-v3"
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._bound_raster_text_metric_revision(
            pointer=runtime.RasterTextActivationPointerV6.model_construct(
                metric_revision="source-attested-semantic-text-v3",
                evidence_revision="source-attested-fresh-staged-train-holdout-v2",
                latency_gate_revision="warm-p95-10s-v1",
            ),
            evaluation=evaluation,
            calibration=calibration,
        )


def test_pointer_writer_derives_v7_finetune_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document_preview_root = tmp_path / "_local" / "data" / "document-preview"
    document_preview_root.mkdir(parents=True)
    profile = runtime.RasterTextOfflineProfileV19.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19-extractor",
        source_recognizer_release_sha256=_SHA_A,
        source_recognizer_receipt_sha256=_SHA_B,
        source_recognizer_training_intent_sha256=_SHA_C,
        source_recognizer_selection_receipt_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{_SHA_C}.json"
        ),
        source_recognizer_selection_receipt_sha256=_SHA_C,
        source_recognizer_parent_selection_receipt_sha256=_SHA_D,
    )
    evaluation = runtime.RasterTextSealedEvaluationV7.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19-extractor",
        runtime_configuration_sha256="0" * 64,
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
    )
    configuration_raw = b"configuration"
    evaluation_raw = b"evaluation"
    evaluation.runtime_configuration_sha256 = runtime.hashlib.sha256(configuration_raw).hexdigest()
    monkeypatch.setattr(
        runtime,
        "_read_local_file",
        lambda path, **_kwargs: configuration_raw if path.name == "v19.json" else evaluation_raw,
    )
    monkeypatch.setattr(runtime, "parse_raster_text_offline_profile", lambda _raw: profile)
    monkeypatch.setattr(runtime, "_parse_raster_text_sealed_evaluation", lambda _raw: evaluation)
    monkeypatch.setattr(runtime, "_load_binding_from_pointer", lambda **_kwargs: None)
    written: list[bytes] = []
    monkeypatch.setattr(
        runtime,
        "_atomic_replace_raster_text_pointer",
        lambda *, destination, payload: written.append(payload),
    )

    _digest, pointer = runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=document_preview_root,
        runtime_configuration_relative_path="runtime/v19.json",
        sealed_evaluation_relative_path="evaluation/v7.json",
    )

    assert isinstance(pointer, runtime.RasterTextActivationPointerV7)
    assert pointer.source_recognizer_receipt_sha256 == _SHA_B
    assert pointer.source_recognizer_selection_receipt_sha256 == _SHA_C
    assert written


def test_v7_verifies_actual_selection_v2_and_parent_v1_provenance(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    parent = activation.RasterTextFreshScopeSelectionReceiptV1(
        synthetic_raster_receipt_sha256=_SHA_A,
        source_receipt_sha256=_SHA_B,
        selection_seed_sha256=activation._FRESH_HOLDOUT_SELECTION_SEED_SHA256,
        fresh_calibration_family_ids=[f"cal-{index:03d}" for index in range(50)],
        fresh_sealed_family_ids=[f"sealed-{index:03d}" for index in range(100)],
    )
    parent_raw = runtime._canonical_evaluation_json_bytes(
        parent.model_dump(mode="json", warnings=False)
    )
    parent_sha = runtime.hashlib.sha256(parent_raw).hexdigest()
    parent_path = f"synthetic-raster-text-v1/fresh-scope-selections/{parent_sha}.json"
    parent_destination = root.joinpath(*parent_path.split("/"))
    parent_destination.parent.mkdir(parents=True)
    parent_destination.write_bytes(parent_raw)
    selection = activation.RasterTextFreshScopeSelectionReceiptV2(
        synthetic_raster_receipt_sha256=_SHA_A,
        source_receipt_sha256=_SHA_B,
        parent_selection_receipt_relative_path=parent_path,
        parent_selection_receipt_sha256=parent_sha,
        selection_seed_sha256=activation._FRESH_HOLDOUT_SELECTION_SEED_SHA256_V2,
        excluded_family_ids_sha256=_SHA_C,
        eligible_family_ids_sha256=_SHA_D,
        fresh_calibration_family_ids=[f"v2-cal-{index:03d}" for index in range(50)],
        fresh_sealed_family_ids=[f"v2-sealed-{index:03d}" for index in range(100)],
    )
    selection_raw = runtime._canonical_evaluation_json_bytes(
        selection.model_dump(mode="json", warnings=False)
    )
    selection_sha = runtime.hashlib.sha256(selection_raw).hexdigest()
    selection_path = f"synthetic-raster-text-v1/fresh-scope-selections/{selection_sha}.json"
    provenance = {
        "source_recognizer_release_sha256": _SHA_A,
        "source_recognizer_receipt_sha256": _SHA_B,
        "source_recognizer_training_intent_sha256": _SHA_C,
        "source_recognizer_parent_selection_receipt_sha256": parent_sha,
    }
    profile = runtime.RasterTextOfflineProfileV19.model_construct(
        **provenance,
        source_recognizer_selection_receipt_relative_path=selection_path,
        source_recognizer_selection_receipt_sha256=selection_sha,
    )
    pointer = runtime.RasterTextActivationPointerV7.model_construct(
        **provenance,
        source_recognizer_selection_receipt_path=selection_path,
        source_recognizer_selection_receipt_sha256=selection_sha,
    )
    evaluation = runtime.RasterTextSealedEvaluationV7.model_construct(
        **provenance,
        selection_receipt_path=selection_path,
        selection_receipt_sha256=selection_sha,
    )
    calibration = runtime.RasterTextCalibrationReceiptV7.model_construct(
        **provenance,
        selection_receipt_path=selection_path,
        selection_receipt_sha256=selection_sha,
    )

    runtime._verify_v19_finetune_evidence_provenance(
        root=root,
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
        profile=profile,
        calibration_profile=profile,
        selection_raw=selection_raw,
    )

    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._verify_v19_finetune_evidence_provenance(
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
            profile=profile.model_copy(
                update={"source_recognizer_parent_selection_receipt_sha256": _SHA_D}
            ),
            calibration_profile=profile,
            selection_raw=selection_raw,
        )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._verify_v19_finetune_evidence_provenance(
            root=root,
            pointer=pointer,
            evaluation=evaluation,
            calibration=calibration,
            profile=profile,
            calibration_profile=profile,
            selection_raw=parent_raw,
        )
