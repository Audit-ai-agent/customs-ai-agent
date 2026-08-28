"""Independent B/L repair-audit bindings for raster-text pointer V8."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from customs_ai_agent.evaluation import source_attested_bl_repair_audit as audit
from customs_ai_agent.service import raster_text_preview_runtime as runtime


_PROFILE_REVISION = f"synthetic-raster-text-v1:{'1' * 24}"
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_AUDIT_RECEIPT_PATH = "synthetic-bl-repair-audit-v1/receipt.json"
_V19_ENGINE_KIND = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_V19B_ENGINE_KIND = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
)


def _v8_pointer_fields(*, audit_evaluation_sha256: str = _SHA_D) -> dict[str, object]:
    return {
        "metric_revision": "source-attested-semantic-text-v3",
        "evidence_revision": "source-attested-public-finetuned-staged-train-holdout-v2",
        "latency_gate_revision": "warm-p95-10s-v1",
        "engine_kind": _V19B_ENGINE_KIND,
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
        "repair_audit_revision": "source-attested-independent-bl-repair-audit-v1",
        "repair_audit_dataset_receipt_path": _AUDIT_RECEIPT_PATH,
        "repair_audit_dataset_receipt_sha256": _SHA_C,
        "repair_audit_evaluation_path": (
            f"synthetic-bl-repair-audit-v1/evaluations/{audit_evaluation_sha256}.json"
        ),
        "repair_audit_evaluation_sha256": audit_evaluation_sha256,
    }


def _passing_scores() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            fixture_kind=fixture_kind,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=0.99,
            voyage_exact_match=0.99,
            gate_passed=True,
        )
        for fixture_kind in ("clean", "error")
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]


def test_v8_pointer_parser_requires_content_addressed_independent_audit() -> None:
    pointer = runtime.RasterTextActivationPointerV8.model_validate(_v8_pointer_fields())
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))

    parsed = runtime.parse_raster_text_activation_pointer(raw)

    assert isinstance(parsed, runtime.RasterTextActivationPointerV8)
    assert isinstance(parsed, runtime.RasterTextActivationPointerV7)
    invalid = pointer.model_dump(mode="json", warnings=False)
    invalid["repair_audit_evaluation_path"] = (
        "synthetic-bl-repair-audit-v1/evaluations/not-content-addressed.json"
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_activation_invalid"):
        runtime.parse_raster_text_activation_pointer(runtime._canonical_json_bytes(invalid))
    for invalid_engine in (_V19_ENGINE_KIND, "wrong-engine"):
        invalid = pointer.model_dump(mode="json", warnings=False)
        invalid["engine_kind"] = invalid_engine
        with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_activation_invalid"):
            runtime.parse_raster_text_activation_pointer(runtime._canonical_json_bytes(invalid))


def test_v8_runtime_binding_checks_canonical_receipt_evaluation_bundle_and_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_raw = b"receipt-canonical"
    evaluation_raw = b"evaluation-canonical"
    bundle_raw = b"bundle-canonical"
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    evaluation_sha256 = hashlib.sha256(evaluation_raw).hexdigest()
    bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
    receipt = SimpleNamespace()
    scores = _passing_scores()
    evaluation = SimpleNamespace(
        dataset_receipt_sha256=receipt_sha256,
        candidate_profile_revision=_PROFILE_REVISION,
        runtime_configuration_sha256=_SHA_A,
        extractor_revision="v19b-extractor",
        candidate_bundle_relative_path=(
            f"synthetic-bl-repair-audit-v1/candidate-bundles/{bundle_sha256}.json"
        ),
        candidate_bundle_sha256=bundle_sha256,
        scores=scores,
        gate_passed=True,
    )
    bundle = SimpleNamespace(
        dataset_receipt_sha256=receipt_sha256,
        candidate_profile_revision=_PROFILE_REVISION,
        runtime_configuration_sha256=_SHA_A,
        extractor_revision="v19b-extractor",
    )

    class _ReceiptParser:
        @classmethod
        def model_validate_json(cls, raw: bytes) -> SimpleNamespace:
            assert raw == receipt_raw
            return receipt

    class _EvaluationParser:
        @classmethod
        def model_validate_json(cls, raw: bytes) -> SimpleNamespace:
            assert raw == evaluation_raw
            return evaluation

    class _BundleParser:
        @classmethod
        def model_validate_json(cls, raw: bytes) -> SimpleNamespace:
            assert raw == bundle_raw
            return bundle

    monkeypatch.setattr(audit, "BlRepairAuditReceiptV1", _ReceiptParser)
    monkeypatch.setattr(audit, "BlRepairAuditEvaluationV1", _EvaluationParser)
    monkeypatch.setattr(audit, "BlRepairAuditCandidateBundleV1", _BundleParser)
    monkeypatch.setattr(audit, "bl_repair_audit_receipt_sha256", lambda _value: receipt_sha256)
    monkeypatch.setattr(
        audit,
        "bl_repair_audit_evaluation_sha256",
        lambda _value: evaluation_sha256,
    )
    monkeypatch.setattr(
        audit,
        "bl_repair_audit_candidate_bundle_sha256",
        lambda _value: bundle_sha256,
    )
    monkeypatch.setattr(
        runtime,
        "_sidecar_path",
        lambda root, value: root.joinpath(*value.split("/")),
    )

    def read_material(path: Path, *, limit: int) -> bytes:
        if path.name == "receipt.json":
            assert limit == runtime._MAX_EVIDENCE_BYTES
            return receipt_raw
        if path.name == f"{evaluation_sha256}.json":
            assert limit == runtime._MAX_EVIDENCE_BYTES
            return evaluation_raw
        assert path.name == f"{bundle_sha256}.json"
        assert limit == runtime._MAX_REPAIR_AUDIT_BUNDLE_BYTES
        return bundle_raw

    monkeypatch.setattr(runtime, "_read_local_file", read_material)
    pointer_fields = _v8_pointer_fields(audit_evaluation_sha256=evaluation_sha256)
    pointer_fields["repair_audit_dataset_receipt_sha256"] = receipt_sha256
    pointer = runtime.RasterTextActivationPointerV8.model_validate(pointer_fields)
    profile = runtime.RasterTextOfflineProfileV19.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19b-extractor",
        engine_kind=_V19B_ENGINE_KIND,
    )

    runtime._verify_v8_bl_repair_audit_binding(
        root=tmp_path,
        pointer=pointer,
        profile=profile,
    )

    scores[0].voyage_exact_match = 0.97
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._verify_v8_bl_repair_audit_binding(
            root=tmp_path,
            pointer=pointer,
            profile=profile,
        )


def test_pointer_writer_derives_v8_only_from_v7_v19_and_complete_audit_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    configuration_raw = b"configuration"
    sealed_raw = b"sealed"
    receipt_raw = b"audit-receipt"
    audit_evaluation_raw = b"audit-evaluation"
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    audit_evaluation_sha256 = hashlib.sha256(audit_evaluation_raw).hexdigest()
    audit_evaluation_path = (
        f"synthetic-bl-repair-audit-v1/evaluations/{audit_evaluation_sha256}.json"
    )
    profile = runtime.RasterTextOfflineProfileV19.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19b-extractor",
        engine_kind=_V19B_ENGINE_KIND,
        source_recognizer_release_sha256=_SHA_A,
        source_recognizer_receipt_sha256=_SHA_B,
        source_recognizer_training_intent_sha256=_SHA_C,
        source_recognizer_selection_receipt_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{_SHA_C}.json"
        ),
        source_recognizer_selection_receipt_sha256=_SHA_C,
        source_recognizer_parent_selection_receipt_sha256=_SHA_D,
    )
    sealed = runtime.RasterTextSealedEvaluationV7.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19b-extractor",
        runtime_configuration_sha256=configuration_sha256,
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-public-finetuned-staged-train-holdout-v2",
        latency_gate_revision="warm-p95-10s-v1",
    )
    receipt = SimpleNamespace()
    repair_evaluation = SimpleNamespace(
        dataset_receipt_sha256=receipt_sha256,
        extractor_revision="v19b-extractor",
        gate_passed=True,
        scores=_passing_scores(),
    )

    class _ReceiptParser:
        @classmethod
        def model_validate_json(cls, raw: bytes) -> SimpleNamespace:
            assert raw == receipt_raw
            return receipt

    def read_material(path: Path, *, limit: int) -> bytes:
        assert limit in {runtime._MAX_CONFIGURATION_BYTES, runtime._MAX_EVIDENCE_BYTES}
        if path.name == "v19.json":
            return configuration_raw
        if path.name == "v7.json":
            return sealed_raw
        if path.name == "receipt.json":
            return receipt_raw
        assert path.name == f"{audit_evaluation_sha256}.json"
        return audit_evaluation_raw

    monkeypatch.setattr(runtime, "_read_local_file", read_material)
    monkeypatch.setattr(runtime, "parse_raster_text_offline_profile", lambda _raw: profile)
    monkeypatch.setattr(runtime, "_parse_raster_text_sealed_evaluation", lambda _raw: sealed)
    monkeypatch.setattr(audit, "BlRepairAuditReceiptV1", _ReceiptParser)
    monkeypatch.setattr(
        audit,
        "verify_bl_repair_audit_evaluation",
        lambda **_kwargs: repair_evaluation,
    )
    monkeypatch.setattr(audit, "bl_repair_audit_receipt_sha256", lambda _value: receipt_sha256)
    monkeypatch.setattr(
        audit,
        "bl_repair_audit_evaluation_sha256",
        lambda _value: audit_evaluation_sha256,
    )
    monkeypatch.setattr(
        runtime,
        "_sidecar_path",
        lambda root, value: root.joinpath(*value.split("/")),
    )
    monkeypatch.setattr(runtime, "_load_binding_from_pointer", lambda **_kwargs: None)
    written: list[bytes] = []
    monkeypatch.setattr(
        runtime,
        "_atomic_replace_raster_text_pointer",
        lambda *, destination, payload: written.append(payload),
    )

    _digest, pointer = runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path="runtime/v19.json",
        sealed_evaluation_relative_path="evaluation/v7.json",
        repair_audit_dataset_receipt_relative_path=_AUDIT_RECEIPT_PATH,
        repair_audit_evaluation_relative_path=audit_evaluation_path,
    )

    assert isinstance(pointer, runtime.RasterTextActivationPointerV8)
    assert pointer.repair_audit_dataset_receipt_sha256 == receipt_sha256
    assert pointer.repair_audit_evaluation_sha256 == audit_evaluation_sha256
    assert written

    with pytest.raises(
        runtime.RasterTextRuntimeError,
        match="raster_text_activation_arguments_invalid",
    ):
        runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="runtime/v19.json",
            sealed_evaluation_relative_path="evaluation/v7.json",
        )

    old_profile = profile.model_copy(update={"engine_kind": _V19_ENGINE_KIND})
    monkeypatch.setattr(
        runtime,
        "parse_raster_text_offline_profile",
        lambda _raw: old_profile,
    )
    with pytest.raises(
        runtime.RasterTextRuntimeError,
        match="raster_text_activation_arguments_invalid",
    ):
        runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="runtime/v19.json",
            sealed_evaluation_relative_path="evaluation/v7.json",
            repair_audit_dataset_receipt_relative_path=_AUDIT_RECEIPT_PATH,
            repair_audit_evaluation_relative_path=audit_evaluation_path,
        )

    _digest, old_pointer = runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path="runtime/v19.json",
        sealed_evaluation_relative_path="evaluation/v7.json",
    )
    assert type(old_pointer) is runtime.RasterTextActivationPointerV7
    assert "engine_kind" not in old_pointer.model_dump(mode="json", warnings=False)

    with pytest.raises(
        runtime.RasterTextRuntimeError,
        match="raster_text_activation_arguments_invalid",
    ):
        runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="runtime/v19.json",
            sealed_evaluation_relative_path="evaluation/v7.json",
            repair_audit_dataset_receipt_relative_path=_AUDIT_RECEIPT_PATH,
        )

    legacy_profile = runtime.RasterTextOfflineProfileV18.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v18-extractor",
    )
    legacy_evaluation = runtime.RasterTextSealedEvaluationV6.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v18-extractor",
        runtime_configuration_sha256=configuration_sha256,
    )
    monkeypatch.setattr(
        runtime,
        "parse_raster_text_offline_profile",
        lambda _raw: legacy_profile,
    )
    monkeypatch.setattr(
        runtime,
        "_parse_raster_text_sealed_evaluation",
        lambda _raw: legacy_evaluation,
    )
    with pytest.raises(
        runtime.RasterTextRuntimeError,
        match="raster_text_activation_arguments_invalid",
    ):
        runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="runtime/v19.json",
            sealed_evaluation_relative_path="evaluation/v7.json",
            repair_audit_dataset_receipt_relative_path=_AUDIT_RECEIPT_PATH,
            repair_audit_evaluation_relative_path=audit_evaluation_path,
        )


@pytest.mark.parametrize("profile_engine_kind", [_V19_ENGINE_KIND, "wrong-engine"])
def test_v8_load_guard_rejects_old_or_wrong_profile_engine(
    profile_engine_kind: str,
) -> None:
    pointer = runtime.RasterTextActivationPointerV8.model_validate(_v8_pointer_fields())
    profile = runtime.RasterTextOfflineProfileV19.model_construct(
        engine_kind=profile_engine_kind,
    )

    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_profile_mismatch"):
        runtime._verify_activation_pointer_engine_binding(
            pointer=pointer,
            profile=profile,
        )


def test_v7_load_guard_preserves_only_old_v19_engine() -> None:
    pointer = runtime.RasterTextActivationPointerV7.model_validate(
        {
            key: value
            for key, value in _v8_pointer_fields().items()
            if not key.startswith("repair_") and key != "engine_kind"
        }
    )
    old_profile = runtime.RasterTextOfflineProfileV19.model_construct(
        engine_kind=_V19_ENGINE_KIND,
    )

    runtime._verify_activation_pointer_engine_binding(pointer=pointer, profile=old_profile)

    new_profile = old_profile.model_copy(update={"engine_kind": _V19B_ENGINE_KIND})
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_profile_mismatch"):
        runtime._verify_activation_pointer_engine_binding(
            pointer=pointer,
            profile=new_profile,
        )
