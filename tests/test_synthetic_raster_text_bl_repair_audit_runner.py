from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, Iterator, Literal

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.evaluation.source_attested_bl_repair_audit import (
    BlRepairAuditRasterBindingV1,
    BlRepairAuditReceiptV1,
)
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionResultV1,
    RasterTextInputV1,
    RasterTextPageV1,
    canonical_raster_text,
    canonical_raster_text_sha256,
)
from customs_ai_agent.service.raster_text_extraction import MediaAwareRasterTextEngine
from customs_ai_agent.service.raster_text_preview_runtime import RasterTextOfflineProfileV19
from customs_ai_agent.service.raster_text_preview_runtime import RasterTextActivationPointerV8


_PROFILE_REVISION = "synthetic-raster-text-v1:0123456789abcdef01234567"
_EXTRACTOR_REVISION = "direct-paddle-v19b-audit-test"
_RUNTIME_SHA256 = "a" * 64
_RECEIPT_SHA256 = "b" * 64
_RUNNER_SHA256 = "c" * 64


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root.resolve()


def _binding(
    index: int,
    *,
    image_sha256: str | None = None,
) -> BlRepairAuditRasterBindingV1:
    asset_id = f"blra-{index:03d}-clean"
    profile_id: Literal["png-216"] = "png-216"
    return BlRepairAuditRasterBindingV1(
        asset_id=asset_id,
        family_id=f"blra-family-{index:03d}",
        fixture_kind="clean",
        profile_id=profile_id,
        source_pdf_sha256=f"{index + 1:x}" * 64,
        image_sha256=image_sha256 or f"{index + 4:x}" * 64,
        media_type="image/png",
        byte_size=1,
        width=100,
        height=200,
        asset_relative_path=(f"synthetic-bl-repair-audit-v1/assets/{asset_id}-{profile_id}.png"),
        gold_relative_path=(f"synthetic-bl-repair-audit-v1/gold/{asset_id}-{profile_id}.json"),
        gold_sha256=f"{index + 8:x}" * 64,
    )


def _profile() -> RasterTextOfflineProfileV19:
    return RasterTextOfflineProfileV19.model_construct(
        engine_kind=activation._V19B_ENGINE_KIND,  # noqa: SLF001
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision=_EXTRACTOR_REVISION,
    )


def _candidate(
    *, binding: BlRepairAuditRasterBindingV1, raster_input: RasterTextInputV1
) -> RasterTextExtractionResultV1:
    pages = [
        RasterTextPageV1(
            page_number=1,
            text=f"BILL OF LADING {binding.asset_id}",
            confidence=0.99,
        )
    ]
    return RasterTextExtractionResultV1(
        document_asset_id=binding.asset_id,
        extraction_request_id=raster_input.metadata.extraction_request_id,
        raster_text_profile_revision=_PROFILE_REVISION,
        status="preview_ready",
        media_type=binding.media_type,
        byte_size=binding.byte_size,
        source_sha256=binding.image_sha256,
        canonical_text=canonical_raster_text(pages),
        canonical_text_sha256=canonical_raster_text_sha256(pages),
        extractor_revision=_EXTRACTOR_REVISION,
        pages=pages,
        warnings=[],
        needs_review_reasons=[],
    )


def test_bl_repair_audit_checkpoint_resumes_missing_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    bindings = [_binding(index) for index in range(3)]
    receipt = BlRepairAuditReceiptV1.model_construct(
        pdfium_version="test-pdfium",
        rasters=bindings,
    )
    profile = _profile()
    calls: list[str] = []
    sessions: list[bool] = []
    state: dict[str, Any] = {
        "fail_on": bindings[1].asset_id,
        "material_drift": False,
        "tick": 0.0,
    }
    by_asset = {item.asset_id: item for item in bindings}

    class Provider:
        @contextmanager
        def open_evaluation_session(self, *, parent_materials_verified: bool) -> Iterator[object]:
            sessions.append(parent_materials_verified)
            yield object()

    engine = MediaAwareRasterTextEngine(input_adapter=object(), provider=Provider())  # type: ignore[arg-type]

    def extract(
        _engine: object,
        *,
        raster_input: RasterTextInputV1,
        **_kwargs: object,
    ) -> RasterTextExtractionResultV1:
        asset_id = raster_input.metadata.document_asset_id
        calls.append(asset_id)
        if state["fail_on"] == asset_id:
            raise RuntimeError("raw provider diagnostics must not escape")
        return _candidate(binding=by_asset[asset_id], raster_input=raster_input)

    def monotonic() -> float:
        state["tick"] = float(state["tick"]) + 1.0
        return float(state["tick"])

    def verify_material(**_kwargs: object) -> bytes:
        if state["material_drift"]:
            raise activation.SyntheticRasterTextActivationError("bl_repair_audit_material_invalid")
        return b"x"

    monkeypatch.setattr(
        activation,
        "_verify_bl_repair_audit_binding_material",
        verify_material,
    )
    monkeypatch.setattr(
        activation,
        "inspect_raster_document_structure",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(activation, "extract_raster_text_preview", extract)
    monkeypatch.setattr(
        activation,
        "_bl_repair_audit_runner_implementation_sha256",
        lambda: _RUNNER_SHA256,
    )
    monkeypatch.setattr(activation.time, "monotonic", monotonic)

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^bl_repair_audit_candidate_execution_failed$",
    ):
        activation._run_bl_repair_audit_observations(  # noqa: SLF001
            root=root,
            receipt=receipt,
            receipt_sha256=_RECEIPT_SHA256,
            runtime_configuration_sha256=_RUNTIME_SHA256,
            profile=profile,
            engine=engine,
            bindings=bindings,
        )
    assert calls == [bindings[0].asset_id, bindings[1].asset_id]
    assert len(list(root.glob("synthetic-bl-repair-audit-v1/candidate-checkpoints/*/*.json"))) == 1

    state["fail_on"] = None
    calls.clear()
    second = activation._run_bl_repair_audit_observations(  # noqa: SLF001
        root=root,
        receipt=receipt,
        receipt_sha256=_RECEIPT_SHA256,
        runtime_configuration_sha256=_RUNTIME_SHA256,
        profile=profile,
        engine=engine,
        bindings=bindings,
    )
    assert calls == [bindings[1].asset_id, bindings[2].asset_id]
    assert [item.document_asset_id for item in second] == sorted(by_asset)
    assert len(list(root.glob("synthetic-bl-repair-audit-v1/candidate-checkpoints/*/*.json"))) == 3

    calls.clear()
    third = activation._run_bl_repair_audit_observations(  # noqa: SLF001
        root=root,
        receipt=receipt,
        receipt_sha256=_RECEIPT_SHA256,
        runtime_configuration_sha256=_RUNTIME_SHA256,
        profile=profile,
        engine=engine,
        bindings=bindings,
    )
    assert third == second
    assert calls == []
    assert sessions == [True, True]

    state["material_drift"] = True
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^bl_repair_audit_material_invalid$",
    ):
        activation._run_bl_repair_audit_observations(  # noqa: SLF001
            root=root,
            receipt=receipt,
            receipt_sha256=_RECEIPT_SHA256,
            runtime_configuration_sha256=_RUNTIME_SHA256,
            profile=profile,
            engine=engine,
            bindings=bindings,
        )
    state["material_drift"] = False

    checkpoint = next(root.glob("synthetic-bl-repair-audit-v1/candidate-checkpoints/*/*.json"))
    checkpoint.write_bytes(b"{")
    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^bl_repair_audit_candidate_checkpoint_invalid$",
    ):
        activation._run_bl_repair_audit_observations(  # noqa: SLF001
            root=root,
            receipt=receipt,
            receipt_sha256=_RECEIPT_SHA256,
            runtime_configuration_sha256=_RUNTIME_SHA256,
            profile=profile,
            engine=engine,
            bindings=bindings,
        )


def test_bl_repair_audit_reuses_only_identical_media_profile_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    shared_sha256 = "d" * 64
    bindings = [
        _binding(0, image_sha256=shared_sha256),
        _binding(1, image_sha256=shared_sha256),
    ]
    receipt = BlRepairAuditReceiptV1.model_construct(
        pdfium_version="test-pdfium",
        rasters=bindings,
    )
    profile = _profile()
    calls: list[str] = []

    class Provider:
        @contextmanager
        def open_evaluation_session(self, *, parent_materials_verified: bool) -> Iterator[object]:
            assert parent_materials_verified
            yield object()

    engine = MediaAwareRasterTextEngine(input_adapter=object(), provider=Provider())  # type: ignore[arg-type]

    def extract(
        _engine: object,
        *,
        raster_input: RasterTextInputV1,
        **_kwargs: object,
    ) -> RasterTextExtractionResultV1:
        asset_id = raster_input.metadata.document_asset_id
        calls.append(asset_id)
        return _candidate(binding=bindings[0], raster_input=raster_input)

    ticks = iter((1.0, 2.0))
    monkeypatch.setattr(
        activation,
        "_verify_bl_repair_audit_binding_material",
        lambda **_kwargs: b"x",
    )
    monkeypatch.setattr(
        activation,
        "inspect_raster_document_structure",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(activation, "extract_raster_text_preview", extract)
    monkeypatch.setattr(
        activation,
        "_bl_repair_audit_runner_implementation_sha256",
        lambda: _RUNNER_SHA256,
    )
    monkeypatch.setattr(activation.time, "monotonic", lambda: next(ticks))

    observed = activation._run_bl_repair_audit_observations(  # noqa: SLF001
        root=root,
        receipt=receipt,
        receipt_sha256=_RECEIPT_SHA256,
        runtime_configuration_sha256=_RUNTIME_SHA256,
        profile=profile,
        engine=engine,
        bindings=bindings,
    )
    assert calls == [bindings[0].asset_id]
    assert {item.document_asset_id for item in observed} == {
        bindings[0].asset_id,
        bindings[1].asset_id,
    }
    assert len(list(root.glob("synthetic-bl-repair-audit-v1/candidate-checkpoints/*/*.json"))) == 2


def test_bl_repair_audit_gate_failure_is_specific_and_never_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    receipt = SimpleNamespace(rasters=[])
    profile = _profile()
    bundle = SimpleNamespace()
    failed_evaluation = SimpleNamespace(gate_passed=False)

    monkeypatch.setattr(activation, "compile_bl_repair_audit_dataset", lambda **_kwargs: receipt)
    monkeypatch.setattr(activation, "verify_bl_repair_audit_dataset", lambda **_kwargs: receipt)
    monkeypatch.setattr(
        activation, "bl_repair_audit_receipt_sha256", lambda _receipt: _RECEIPT_SHA256
    )
    monkeypatch.setattr(
        activation,
        "resolve_paddleocr3_raster_text_candidate",
        lambda **_kwargs: SimpleNamespace(
            status="available",
            engine=object(),
            profile=profile,
            runtime_configuration_sha256=_RUNTIME_SHA256,
        ),
    )
    monkeypatch.setattr(activation, "_run_bl_repair_audit_observations", lambda **_kwargs: [])
    monkeypatch.setattr(activation, "BlRepairAuditCandidateBundleV1", lambda **_kwargs: bundle)
    monkeypatch.setattr(
        activation,
        "write_bl_repair_audit_candidate_bundle",
        lambda **_kwargs: ("bundle.json", "e" * 64),
    )
    monkeypatch.setattr(
        activation,
        "verify_bl_repair_audit_candidate_bundle",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        activation,
        "evaluate_bl_repair_audit_candidates",
        lambda **_kwargs: failed_evaluation,
    )
    monkeypatch.setattr(
        activation,
        "write_bl_repair_audit_evaluation",
        lambda **_kwargs: ("evaluation.json", "f" * 64),
    )
    monkeypatch.setattr(
        activation,
        "verify_bl_repair_audit_evaluation",
        lambda **_kwargs: failed_evaluation,
    )

    with pytest.raises(
        activation.SyntheticRasterTextActivationError,
        match="^bl_repair_audit_gate_failed$",
    ):
        activation.run_source_attested_bl_repair_audit_candidates(
            document_preview_root=root,
            runtime_configuration_relative_path="profiles/v19b.json",
        )
    assert not (root / "approvals" / "raster-text-active.json").exists()


def test_shared_v2_transaction_runs_audit_after_warm_and_before_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    calls: list[str] = []
    final_profile = _profile()
    calibration_info = SimpleNamespace(
        runtime_configuration_relative_path="profiles/calibration.json",
        runtime_configuration_sha256="1" * 64,
    )
    final_info = SimpleNamespace(
        runtime_configuration_relative_path="profiles/final.json",
        runtime_configuration_sha256=_RUNTIME_SHA256,
    )
    build_inputs = SimpleNamespace(build_inputs_relative_path="build-inputs.json")

    def candidate_run(**kwargs: object) -> SimpleNamespace:
        scope = str(kwargs["scope"])
        calls.append(f"candidate:{scope}")
        return SimpleNamespace(
            source_gold_receipt_relative_path=f"gold/{scope}.json",
            candidate_execution_relative_path=f"execution/{scope}.json",
            latency_p95_seconds=1.0,
        )

    monkeypatch.setattr(
        activation, "run_paddleocr3_synthetic_raster_text_candidates", candidate_run
    )
    monkeypatch.setattr(
        activation,
        "write_raster_semantic_text_smoke_quality_screen_v3",
        lambda **_kwargs: ("smoke.json", "2" * 64, True),
    )
    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: ("selection.json", "3" * 64),
    )
    monkeypatch.setattr(
        activation,
        "verify_source_attested_fresh_raster_text_evidence_v6",
        lambda **_kwargs: (object(), object(), object(), [object()]),
    )
    monkeypatch.setattr(activation, "_calibrated_floor", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(activation, "_read", lambda *_args, **_kwargs: b"profile")
    monkeypatch.setattr(activation, "parse_raster_text_offline_profile", lambda _raw: final_profile)

    def warm(**_kwargs: object) -> tuple[str, str]:
        calls.append("warm")
        return "warm.json", "4" * 64

    monkeypatch.setattr(activation, "run_warm_service_raster_text_smoke", warm)
    monkeypatch.setattr(activation, "_require_auto_activation_warm_latency", lambda **_kwargs: None)
    monkeypatch.setattr(
        activation,
        "write_fresh_raster_text_sealed_stage1_decision",
        lambda **_kwargs: ("stage1.json", "5" * 64, "early_pass"),
    )
    monkeypatch.setattr(
        activation,
        "resolve_paddleocr3_raster_text_candidate",
        lambda **_kwargs: SimpleNamespace(
            status="available",
            profile=final_profile,
            runtime_configuration_sha256=_RUNTIME_SHA256,
        ),
    )
    monkeypatch.setattr(
        activation,
        "_load_fresh_dataset_receipts",
        lambda *_args: (
            SimpleNamespace(build_inputs_relative_path=build_inputs.build_inputs_relative_path),
            "6" * 64,
            object(),
            "7" * 64,
        ),
    )
    pointer_calls: list[dict[str, object]] = []

    def pointer_writer(**kwargs: object) -> tuple[str, RasterTextActivationPointerV8]:
        pointer_calls.append(kwargs)
        calls.append("pointer")
        return (
            "8" * 64,
            RasterTextActivationPointerV8.model_construct(
                raster_text_profile_revision=_PROFILE_REVISION
            ),
        )

    monkeypatch.setattr(
        activation, "write_source_attested_synthetic_raster_text_pointer", pointer_writer
    )
    monkeypatch.setattr(
        activation,
        "_verify_bl_repair_audit_activation_evidence",
        lambda **_kwargs: calls.append("audit-recheck"),
    )
    audit = activation.BlRepairAuditCandidateRunV1(
        dataset_receipt_relative_path="synthetic-bl-repair-audit-v1/receipt.json",
        dataset_receipt_sha256="9" * 64,
        candidate_bundle_relative_path="bundle.json",
        candidate_bundle_sha256="a" * 64,
        evaluation_relative_path="audit-evaluation.json",
        evaluation_sha256="b" * 64,
        raster_text_profile_revision=_PROFILE_REVISION,
        runtime_configuration_sha256=_RUNTIME_SHA256,
    )

    def pre_sealed(*_args: object) -> activation.BlRepairAuditCandidateRunV1:
        calls.append("audit")
        return audit

    result = activation._run_v2_staged_candidate_activation(  # noqa: SLF001
        root=root,
        build_inputs=cast(Any, build_inputs),
        existing_dataset=False,
        dataset_materials_preverified=True,
        calibration_profile_info=calibration_info,
        prepare_final_profile=lambda _floor: final_info,
        expected_profile_type=RasterTextOfflineProfileV19,
        expected_pointer_type=RasterTextActivationPointerV8,
        write_calibration=lambda **_kwargs: ("calibration.json", "c" * 64),
        write_sealed=lambda **_kwargs: ("sealed.json", "d" * 64),
        recheck_materials=lambda: calls.append("materials-recheck"),
        drift_code="drift",
        pre_sealed_activation_check=pre_sealed,
    )

    assert calls.index("warm") < calls.index("audit")
    assert calls.index("audit") < calls.index("candidate:fresh-sealed")
    assert calls.index("materials-recheck") < calls.index("audit-recheck")
    assert calls.index("audit-recheck") < calls.index("pointer")
    assert pointer_calls == [
        {
            "document_preview_root": root,
            "runtime_configuration_relative_path": "profiles/final.json",
            "sealed_evaluation_relative_path": "sealed.json",
            "repair_audit_dataset_receipt_relative_path": (
                "synthetic-bl-repair-audit-v1/receipt.json"
            ),
            "repair_audit_evaluation_relative_path": "audit-evaluation.json",
        }
    ]
    assert result.raster_text_profile_revision == _PROFILE_REVISION
