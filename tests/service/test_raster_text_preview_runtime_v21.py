from __future__ import annotations

import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Callable, Literal, cast

from pydantic import ValidationError
import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as activation
from customs_ai_agent.service import raster_text_preview_runtime as runtime


_PROFILE_REVISION = f"synthetic-raster-text-v1:{'a' * 24}"
_V19_ENGINE_KIND: Literal[
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
] = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_V19B_ENGINE_KIND = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
)
_METRIC_REVISION: Literal["source-attested-semantic-text-v3"] = "source-attested-semantic-text-v3"
_EVIDENCE_REVISION: Literal["source-attested-public-finetuned-staged-train-holdout-v2"] = (
    "source-attested-public-finetuned-staged-train-holdout-v2"
)
_QUALITY_GATE_REVISION: Literal["preview-draft-critical97-cer2-v1"] = (
    "preview-draft-critical97-cer2-v1"
)
_REVALIDATION_REVISION: Literal["historical-runner-current-runtime-smoke-v1"] = (
    "historical-runner-current-runtime-smoke-v1"
)


def _scores(
    critical_value_exact_match: float = 0.97,
) -> list[runtime.RasterTextSealedProfileScoreV8]:
    return [
        runtime.RasterTextSealedProfileScoreV8(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.02,
            critical_value_exact_match=critical_value_exact_match,
        )
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in (
            "png-216",
            "jpeg-216-q75",
            "jpeg-150-q75",
            "scan-pdf-216-q75",
        )
    ]


def _evaluation_v8() -> runtime.RasterTextSealedEvaluationV8:
    return runtime.RasterTextSealedEvaluationV8(
        metric_revision=_METRIC_REVISION,
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
        projection_kind="sealed-stage2-100-family-v1",
        stage1_decision="continue",
        family_count=100,
        asset_count=2_400,
        raster_text_profile_revision=_PROFILE_REVISION,
        runtime_configuration_sha256="1" * 64,
        extractor_revision="v19-extractor",
        selection_receipt_path="selection/v2.json",
        selection_receipt_sha256="2" * 64,
        projection_receipt_path="projection/sealed-stage2.json",
        projection_receipt_sha256="3" * 64,
        stage1_decision_receipt_path="decisions/stage1.json",
        stage1_decision_receipt_sha256="4" * 64,
        source_gold_receipt_path="gold/fresh-sealed.json",
        source_gold_receipt_sha256="5" * 64,
        calibration_receipt_path="evaluations/calibration-v7.json",
        calibration_receipt_sha256="6" * 64,
        candidate_execution_path="executions/historical-sealed.json",
        candidate_execution_sha256="7" * 64,
        warm_service_smoke_receipt_path="smoke/historical.json",
        warm_service_smoke_receipt_sha256="8" * 64,
        source_recognizer_release_sha256="9" * 64,
        source_recognizer_receipt_sha256="a" * 64,
        source_recognizer_training_intent_sha256="b" * 64,
        source_recognizer_parent_selection_receipt_sha256="c" * 64,
        latency_p95_seconds=7.86,
        service_warm_latency_p95_seconds=7.9,
        minimum_page_confidence=0.0,
        quality_gate_revision=_QUALITY_GATE_REVISION,
        evidence_revalidation_revision=_REVALIDATION_REVISION,
        historical_runner_implementation_sha256="d" * 64,
        current_runner_implementation_sha256="e" * 64,
        revalidation_smoke_receipt_path="smoke/current.json",
        revalidation_smoke_receipt_sha256="f" * 64,
        revalidation_candidate_execution_path="executions/current-smoke.json",
        revalidation_candidate_execution_sha256="0" * 64,
        scores=_scores(),
    )


def _pointer_v10(
    evaluation: runtime.RasterTextSealedEvaluationV8,
) -> runtime.RasterTextActivationPointerV10:
    return runtime.RasterTextActivationPointerV10(
        engine_kind=_V19_ENGINE_KIND,
        metric_revision=_METRIC_REVISION,
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
        raster_text_profile_revision=_PROFILE_REVISION,
        runtime_configuration_path="profiles/v19.json",
        runtime_configuration_sha256=evaluation.runtime_configuration_sha256,
        sealed_evaluation_path="evaluations/sealed-v8.json",
        sealed_evaluation_sha256="1" * 64,
        source_recognizer_release_sha256=evaluation.source_recognizer_release_sha256,
        source_recognizer_receipt_sha256=evaluation.source_recognizer_receipt_sha256,
        source_recognizer_training_intent_sha256=(
            evaluation.source_recognizer_training_intent_sha256
        ),
        source_recognizer_selection_receipt_path=evaluation.selection_receipt_path,
        source_recognizer_selection_receipt_sha256=evaluation.selection_receipt_sha256,
        source_recognizer_parent_selection_receipt_sha256=(
            evaluation.source_recognizer_parent_selection_receipt_sha256
        ),
        quality_gate_revision=evaluation.quality_gate_revision,
        evidence_revalidation_revision=evaluation.evidence_revalidation_revision,
        historical_runner_implementation_sha256=(
            evaluation.historical_runner_implementation_sha256
        ),
        current_runner_implementation_sha256=(evaluation.current_runner_implementation_sha256),
        revalidation_smoke_receipt_path=evaluation.revalidation_smoke_receipt_path,
        revalidation_smoke_receipt_sha256=evaluation.revalidation_smoke_receipt_sha256,
        revalidation_candidate_execution_path=(evaluation.revalidation_candidate_execution_path),
        revalidation_candidate_execution_sha256=(
            evaluation.revalidation_candidate_execution_sha256
        ),
    )


def _pointer_v11(
    evaluation: runtime.RasterTextSealedEvaluationV8,
) -> runtime.RasterTextActivationPointerV11:
    values = _pointer_v10(evaluation).model_dump(mode="json", warnings=False)
    values["schema_version"] = "11.0"
    values["runtime_validation_revision"] = "activation-full-runtime-compact-v1"
    return runtime.RasterTextActivationPointerV11.model_validate(values)


class _RuntimeProbe:
    def __init__(self, *, alive: bool = True) -> None:
        self.alive = alive
        self.strict_checks = 0
        self.nonblocking_checks = 0
        self.close_calls = 0
        self.lane = threading.Lock()
        self.probe_entered = threading.Event()
        self.probe_release: threading.Event | None = None

    def verify_runtime_state(self) -> None:
        with self.lane:
            self.strict_checks += 1
            if not self.alive:
                raise runtime.RasterTextRuntimeError("raster_text_service_unavailable")

    def verify_runtime_state_nonblocking(self) -> None:
        self.nonblocking_checks += 1
        self.probe_entered.set()
        if self.probe_release is not None:
            assert self.probe_release.wait(timeout=2)
        if not self.lane.acquire(blocking=False):
            return
        try:
            if not self.alive:
                raise runtime.RasterTextRuntimeError("raster_text_service_unavailable")
        finally:
            self.lane.release()

    def close(self) -> None:
        self.close_calls += 1


def _auto_runtime(
    *,
    tmp_path: Path,
    pointer: runtime.RasterTextActivationPointerV10,
    service_factory: Callable[[], _RuntimeProbe],
    engine: object | None = None,
) -> tuple[
    runtime.AutoActivatingSourceAttestedRasterTextPreviewRuntime,
    list[_RuntimeProbe],
    Path,
]:
    root = tmp_path / "_local" / "data" / "document-preview"
    approval_path = root / "approvals" / "raster-text-active.json"
    approval_path.parent.mkdir(parents=True)
    approval_raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))
    approval_path.write_bytes(approval_raw)
    services: list[_RuntimeProbe] = []
    selected_engine = object() if engine is None else engine

    def loader(**kwargs: object) -> runtime.SourceAttestedRasterTextPreviewBindingV1:
        raw = cast(bytes, kwargs["approval_raw"])
        parsed = runtime.parse_raster_text_activation_pointer(raw)
        service = service_factory()
        services.append(service)
        return runtime.SourceAttestedRasterTextPreviewBindingV1(
            root=root,
            approval_path=approval_path,
            approval_sha256=hashlib.sha256(raw).hexdigest(),
            pointer=parsed,
            profile=_profile_v19(),
            engine=cast(Any, selected_engine),
            runtime_verifier=service.verify_runtime_state,
            runtime_nonblocking_verifier=service.verify_runtime_state_nonblocking,
            runtime_closer=service.close,
        )

    auto = runtime.AutoActivatingSourceAttestedRasterTextPreviewRuntime(
        approval_path=approval_path,
        binding_loader=loader,
        binding_material_validator=lambda **_kwargs: _PROFILE_REVISION,
    )
    return auto, services, approval_path


def _calibration_v7() -> runtime.RasterTextCalibrationReceiptV7:
    return runtime.RasterTextCalibrationReceiptV7.model_construct(
        metric_revision=_METRIC_REVISION,
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
    )


def _profile_v19(*, engine_kind: str = _V19_ENGINE_KIND) -> runtime.RasterTextOfflineProfileV19:
    return runtime.RasterTextOfflineProfileV19.model_construct(
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19-extractor",
        engine_kind=engine_kind,
        source_recognizer_release_sha256="9" * 64,
        source_recognizer_receipt_sha256="a" * 64,
        source_recognizer_training_intent_sha256="b" * 64,
        source_recognizer_selection_receipt_relative_path="selection/v2.json",
        source_recognizer_selection_receipt_sha256="2" * 64,
        source_recognizer_parent_selection_receipt_sha256="c" * 64,
        sidecar_timeout_seconds=30,
    )


def test_v8_lowers_only_critical_exact_gate() -> None:
    assert (
        runtime.RasterTextSealedProfileScoreV8(
            document_type="bill_of_lading",
            profile_id="png-216",
            character_error_rate=0.02,
            critical_value_exact_match=0.97,
        ).critical_value_exact_match
        == 0.97
    )

    with pytest.raises(ValidationError):
        runtime.RasterTextSealedProfileScoreV8(
            document_type="bill_of_lading",
            profile_id="png-216",
            character_error_rate=0.02,
            critical_value_exact_match=0.9699,
        )
    with pytest.raises(ValidationError):
        runtime.RasterTextSealedProfileScoreV7(
            document_type="bill_of_lading",
            profile_id="png-216",
            character_error_rate=0.02,
            critical_value_exact_match=0.97,
        )
    assert (
        runtime.RasterTextSealedProfileScoreV7(
            document_type="bill_of_lading",
            profile_id="png-216",
            character_error_rate=0.02,
            critical_value_exact_match=0.98,
        ).critical_value_exact_match
        == 0.98
    )


def test_v10_parser_and_generation_binding_fail_closed() -> None:
    evaluation = _evaluation_v8()
    pointer = _pointer_v10(evaluation)
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))

    parsed = runtime.parse_raster_text_activation_pointer(raw)
    assert type(parsed) is runtime.RasterTextActivationPointerV10
    assert (
        runtime._bound_raster_text_metric_revision(
            pointer=parsed,
            evaluation=evaluation,
            calibration=_calibration_v7(),
        )
        == _METRIC_REVISION
    )

    bad_hash = pointer.model_copy(update={"revalidation_smoke_receipt_sha256": "a" * 64})
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._bound_raster_text_metric_revision(
            pointer=bad_hash,
            evaluation=evaluation,
            calibration=_calibration_v7(),
        )
    old_pointer = runtime.RasterTextActivationPointerV7.model_construct(
        metric_revision=_METRIC_REVISION,
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._bound_raster_text_metric_revision(
            pointer=old_pointer,
            evaluation=evaluation,
            calibration=_calibration_v7(),
        )


def test_v10_accepts_only_exact_plain_v19_engine() -> None:
    pointer = _pointer_v10(_evaluation_v8())
    runtime._verify_activation_pointer_engine_binding(
        pointer=pointer,
        profile=_profile_v19(),
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_profile_mismatch"):
        runtime._verify_activation_pointer_engine_binding(
            pointer=pointer,
            profile=_profile_v19(engine_kind=_V19B_ENGINE_KIND),
        )


def test_v11_parser_requires_explicit_compact_runtime_revision() -> None:
    pointer = _pointer_v11(_evaluation_v8())
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))

    assert type(runtime.parse_raster_text_activation_pointer(raw)) is (
        runtime.RasterTextActivationPointerV11
    )
    missing_revision = pointer.model_dump(mode="json", warnings=False)
    missing_revision.pop("runtime_validation_revision")
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_activation_invalid"):
        runtime.parse_raster_text_activation_pointer(
            runtime._canonical_json_bytes(missing_revision)
        )


def test_v11_cached_prewarm_restarts_a_dead_idle_child(tmp_path: Path) -> None:
    probes = [_RuntimeProbe(), _RuntimeProbe()]
    auto, loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probes[len(loaded)],
    )

    assert auto.prewarm().availability == "available"
    probes[0].alive = False
    assert auto.prewarm().availability == "available"

    assert loaded == probes
    assert probes[0].strict_checks == 1
    assert probes[0].nonblocking_checks == 1
    assert probes[0].close_calls == 1
    assert probes[1].strict_checks == 1
    assert auto.status().availability == "available"


def test_v11_cached_prewarm_keeps_status_available_during_busy_probe(tmp_path: Path) -> None:
    probe = _RuntimeProbe()
    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probe,
    )
    assert auto.prewarm().availability == "available"
    probe.probe_release = threading.Event()
    result: list[runtime.RasterTextPreviewStatusV1] = []
    worker = threading.Thread(target=lambda: result.append(auto.prewarm()))
    worker.start()
    assert probe.probe_entered.wait(timeout=1)

    # The lifecycle probe runs outside the facade lock.  Backend status must
    # stay stable while the native lane is occupied by an OCR request.
    assert auto.status().availability == "available"
    probe.probe_release.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result[0].availability == "available"
    assert probe.strict_checks == 1
    assert probe.nonblocking_checks == 1


def test_v11_status_preserves_published_revision_while_facade_lock_is_busy(
    tmp_path: Path,
) -> None:
    probe = _RuntimeProbe()
    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probe,
    )
    assert auto.prewarm().availability == "available"
    entered = threading.Event()
    release = threading.Event()

    def hold_facade_lock() -> None:
        with auto._lock:  # noqa: SLF001 - exact status-race regression
            entered.set()
            assert release.wait(timeout=2)

    holder = threading.Thread(target=hold_facade_lock)
    holder.start()
    assert entered.wait(timeout=1)
    status = auto.status()
    release.set()
    holder.join(timeout=1)

    assert not holder.is_alive()
    assert status.availability == "available"
    assert status.raster_text_profile_revision == _PROFILE_REVISION


def test_v10_busy_status_retains_legacy_fail_closed_semantics(tmp_path: Path) -> None:
    probe = _RuntimeProbe()
    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v10(_evaluation_v8()),
        service_factory=lambda: probe,
    )
    assert auto.prewarm().availability == "available"
    entered = threading.Event()
    release = threading.Event()

    def hold_facade_lock() -> None:
        with auto._lock:  # noqa: SLF001 - legacy status contract regression
            entered.set()
            assert release.wait(timeout=2)

    holder = threading.Thread(target=hold_facade_lock)
    holder.start()
    assert entered.wait(timeout=1)
    status = auto.status()
    release.set()
    holder.join(timeout=1)

    assert not holder.is_alive()
    assert status.availability == "unavailable"
    assert status.raster_text_profile_revision is None


def test_v10_cached_prewarm_retains_strict_legacy_liveness(tmp_path: Path) -> None:
    probe = _RuntimeProbe()
    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v10(_evaluation_v8()),
        service_factory=lambda: probe,
    )

    assert auto.prewarm().availability == "available"
    assert auto.prewarm().availability == "available"
    assert probe.strict_checks == 2
    assert probe.nonblocking_checks == 0


def test_v11_prewarm_reloads_a_pointer_replaced_during_probe(tmp_path: Path) -> None:
    probes = [_RuntimeProbe(), _RuntimeProbe()]
    auto, loaded, approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probes[len(loaded)],
    )
    assert auto.prewarm().availability == "available"
    probes[0].probe_release = threading.Event()
    result: list[runtime.RasterTextPreviewStatusV1] = []
    worker = threading.Thread(target=lambda: result.append(auto.prewarm()))
    worker.start()
    assert probes[0].probe_entered.wait(timeout=1)

    replacement = _pointer_v11(_evaluation_v8()).model_copy(
        update={"sealed_evaluation_sha256": "9" * 64}
    )
    approval_path.write_bytes(
        runtime._canonical_json_bytes(replacement.model_dump(mode="json", warnings=False))
    )
    probes[0].probe_release.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result[0].availability == "available"
    assert loaded == probes
    assert probes[0].close_calls == 1
    assert probes[1].strict_checks == 1


def test_v11_cached_prewarm_never_reloads_after_concurrent_shutdown(tmp_path: Path) -> None:
    probe = _RuntimeProbe()
    auto, loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probe,
    )
    assert auto.prewarm().availability == "available"
    probe.probe_release = threading.Event()
    result: list[runtime.RasterTextPreviewStatusV1] = []
    worker = threading.Thread(target=lambda: result.append(auto.prewarm()))
    worker.start()
    assert probe.probe_entered.wait(timeout=1)

    auto.shutdown()
    probe.probe_release.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result[0].availability == "unavailable"
    assert auto.prewarm().availability == "unavailable"
    assert auto.status().availability == "unavailable"
    assert loaded == [probe]
    assert probe.close_calls == 1


def test_v11_timeout_close_remains_recoverable_on_the_next_prewarm(tmp_path: Path) -> None:
    probes = [_RuntimeProbe(), _RuntimeProbe()]
    auto, loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probes[len(loaded)],
    )
    assert auto.prewarm().availability == "available"

    auto.close()
    assert auto.status().availability == "unavailable"
    assert auto.prewarm().availability == "available"
    assert loaded == probes
    assert probes[0].close_calls == 1
    assert probes[1].strict_checks == 1


def test_new_v11_binding_must_be_alive_before_publication(tmp_path: Path) -> None:
    probe = _RuntimeProbe(alive=False)
    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probe,
    )

    assert auto.prewarm().availability == "unavailable"
    assert auto.status().availability == "unavailable"
    assert probe.strict_checks == 1
    assert probe.close_calls == 1


def test_extract_rejects_a_dead_v11_child_before_engine_call(tmp_path: Path) -> None:
    probe = _RuntimeProbe()
    engine_calls: list[str] = []

    class Engine:
        def extract(self, _raster_input: object, _structure: object) -> object:
            engine_calls.append("called")
            return object()

    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probe,
        engine=Engine(),
    )
    assert auto.prewarm().availability == "available"
    probe.alive = False
    raster_input = SimpleNamespace(
        metadata=SimpleNamespace(raster_text_profile_revision=_PROFILE_REVISION)
    )

    with pytest.raises(runtime.RasterTextPreviewUnavailable):
        auto.extract(cast(Any, raster_input), cast(Any, object()))
    assert engine_calls == []
    assert probe.close_calls == 1


def test_stale_revision_request_does_not_clear_current_v11_binding(tmp_path: Path) -> None:
    probe = _RuntimeProbe()
    engine_calls: list[str] = []

    class Engine:
        def extract(self, _raster_input: object, _structure: object) -> object:
            engine_calls.append("called")
            return object()

    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probe,
        engine=Engine(),
    )
    assert auto.prewarm().availability == "available"
    stale_input = SimpleNamespace(
        metadata=SimpleNamespace(
            raster_text_profile_revision=f"synthetic-raster-text-v1:{'f' * 24}"
        )
    )

    with pytest.raises(runtime.RasterTextPreviewUnavailable):
        auto.extract(cast(Any, stale_input), cast(Any, object()))
    assert engine_calls == []
    assert probe.close_calls == 0
    assert auto.status().availability == "available"


def test_extract_suppresses_result_when_v11_child_dies_after_ocr(tmp_path: Path) -> None:
    probe = _RuntimeProbe()
    candidate = runtime.RasterTextPreviewCandidateV1(
        status="preview_ready",
        extractor_revision="v19-extractor",
        pages=[runtime.RasterTextPageV1(page_number=1, text="READY", confidence=0.9)],
    )

    class Engine:
        def extract(self, _raster_input: object, _structure: object) -> object:
            probe.alive = False
            return candidate

    auto, _loaded, _approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probe,
        engine=Engine(),
    )
    assert auto.prewarm().availability == "available"
    raster_input = SimpleNamespace(
        metadata=SimpleNamespace(raster_text_profile_revision=_PROFILE_REVISION)
    )

    with pytest.raises(runtime.RasterTextPreviewUnavailable):
        auto.extract(cast(Any, raster_input), cast(Any, object()))
    assert probe.close_calls == 1


def test_stale_v11_extraction_failure_does_not_clear_replacement_binding(
    tmp_path: Path,
) -> None:
    probes = [_RuntimeProbe(), _RuntimeProbe()]
    extract_entered = threading.Event()
    extract_release = threading.Event()

    class Engine:
        def extract(self, _raster_input: object, _structure: object) -> object:
            extract_entered.set()
            assert extract_release.wait(timeout=2)
            raise runtime.RasterTextPreviewUnavailable("old request failed")

    auto, loaded, approval_path = _auto_runtime(
        tmp_path=tmp_path,
        pointer=_pointer_v11(_evaluation_v8()),
        service_factory=lambda: probes[len(loaded)],
        engine=Engine(),
    )
    assert auto.prewarm().availability == "available"
    raster_input = SimpleNamespace(
        metadata=SimpleNamespace(raster_text_profile_revision=_PROFILE_REVISION)
    )
    failures: list[BaseException] = []

    def extract_old() -> None:
        try:
            auto.extract(cast(Any, raster_input), cast(Any, object()))
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=extract_old)
    worker.start()
    assert extract_entered.wait(timeout=1)
    replacement = _pointer_v11(_evaluation_v8()).model_copy(
        update={"sealed_evaluation_sha256": "8" * 64}
    )
    approval_path.write_bytes(
        runtime._canonical_json_bytes(replacement.model_dump(mode="json", warnings=False))
    )
    assert auto.prewarm().availability == "available"
    assert len(loaded) == 2

    extract_release.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], runtime.RasterTextPreviewUnavailable)
    assert probes[0].close_calls == 1
    assert probes[1].close_calls == 0
    assert auto.status().availability == "available"


@pytest.mark.parametrize(
    ("pointer_factory", "verification_mode", "expected_complete"),
    [
        (_pointer_v11, "runtime", False),
        (_pointer_v11, "activation", True),
        (_pointer_v10, "runtime", True),
    ],
)
def test_only_v11_runtime_uses_compact_profile_material_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pointer_factory: Callable[
        [runtime.RasterTextSealedEvaluationV8], runtime.RasterTextActivationPointerV10
    ],
    verification_mode: Literal["runtime", "activation"],
    expected_complete: bool,
) -> None:
    evaluation = _evaluation_v8()
    pointer = pointer_factory(evaluation)
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration = root / "profiles" / "v19.json"
    configuration.parent.mkdir(parents=True)
    configuration.write_bytes(b"profile")
    observed: list[bool] = []

    def stop_after_profile_resolution(**kwargs: object) -> object:
        observed.append(bool(kwargs["verify_complete_materials"]))
        raise runtime.RasterTextRuntimeError("test_stop")

    monkeypatch.setattr(runtime, "_load_profile_materials", stop_after_profile_resolution)
    with pytest.raises(runtime.RasterTextRuntimeError, match="test_stop"):
        runtime._load_binding_from_pointer(
            root=root,
            approval_path=root / "approvals" / "raster-text-active.json",
            approval_raw=raw,
            approval_sha256=hashlib.sha256(raw).hexdigest(),
            start_service_session=False,
            verification_mode=verification_mode,
        )
    assert observed == [expected_complete]


def test_v11_runtime_rejects_current_runner_drift_before_smoke_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _evaluation_v8()
    pointer = _pointer_v11(evaluation)
    monkeypatch.setattr(
        runtime,
        "_current_raster_text_runner_implementation_sha256",
        lambda: "f" * 64,
    )

    with pytest.raises(
        runtime.RasterTextRuntimeError,
        match="raster_text_runner_implementation_invalid",
    ):
        runtime._verify_runtime_attested_v11_evidence(
            root=Path("unused"),
            pointer=pointer,
            evaluation=evaluation,
            profile=_profile_v19(),
            evidence_by_path={},
        )


def _warm_smoke_raw(
    *,
    profile: runtime.RasterTextOfflineProfileV19,
    source_raw: bytes,
    execution_raw: bytes,
    warm_p95: float,
) -> bytes:
    receipt = runtime.RasterTextWarmServiceSmokeReceiptV1(
        artifact_kind="synthetic_raster_text_warm_service_smoke",
        dataset_id="synthetic-raster-text-v1",
        scope="smoke",
        source_truth_kind="pdf_content_stream_text_v1",
        execution_mode="persistent_service_warm",
        service_protocol="customs-raster-text-sidecar-v1-service",
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_sha256="1" * 64,
        extractor_revision=profile.extractor_revision,
        source_gold_receipt_path="smoke/source.json",
        source_gold_receipt_sha256=hashlib.sha256(source_raw).hexdigest(),
        candidate_execution_path="smoke/execution.json",
        candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        candidate_bundle_sha256="3" * 64,
        equality_mode="exact_canonical_result_v1",
        equality_set_sha256="4" * 64,
        asset_count=24,
        startup_latency_seconds=1.0,
        startup_timeout_seconds=60,
        warm_latency_p95_seconds=warm_p95,
        request_timeout_seconds=30,
    )
    return runtime._canonical_evaluation_json_bytes(receipt.model_dump(mode="json", warnings=False))


def test_revalidation_smoke_binds_current_runner_hash_and_ten_second_p95(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    (root / "smoke").mkdir(parents=True)
    source_raw = b"source-evidence"
    execution_raw = b"execution-evidence"
    (root / "smoke" / "source.json").write_bytes(source_raw)
    (root / "smoke" / "execution.json").write_bytes(execution_raw)
    profile = _profile_v19()
    runner_sha256 = "e" * 64
    bindings = [
        SimpleNamespace(asset_id=f"asset-{index}", profile_id="png-216") for index in range(24)
    ]
    candidates = [
        SimpleNamespace(
            asset_id=f"asset-{index}",
            profile_id="png-216",
            candidate=object(),
        )
        for index in range(24)
    ]
    source = SimpleNamespace(
        scope="smoke",
        bindings=bindings,
        synthetic_raster_receipt_sha256="5" * 64,
    )
    execution = SimpleNamespace(
        scope="smoke",
        candidate_count=24,
        candidate_bundle_sha256="3" * 64,
        synthetic_raster_receipt_sha256="5" * 64,
        runtime_configuration_sha256="1" * 64,
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19-extractor",
        latency_p95_seconds=9.0,
    )
    bundle = SimpleNamespace(
        scope="smoke",
        candidates=candidates,
        synthetic_raster_receipt_sha256="5" * 64,
        runtime_configuration_sha256="1" * 64,
        raster_text_profile_revision=_PROFILE_REVISION,
        extractor_revision="v19-extractor",
        runner_implementation_sha256=runner_sha256,
    )

    monkeypatch.setattr(
        activation,
        "verify_source_attested_raster_text_evidence",
        lambda **_kwargs: (source, execution, bundle, []),
    )
    monkeypatch.setattr(runtime, "raster_text_exact_result_set_sha256", lambda _values: "4" * 64)

    raw = _warm_smoke_raw(
        profile=profile,
        source_raw=source_raw,
        execution_raw=execution_raw,
        warm_p95=9.0,
    )
    runtime._verify_warm_service_smoke_evidence(
        root=root,
        raw=raw,
        profile=profile,
        runtime_configuration_sha256="1" * 64,
        expected_runner_implementation_sha256=runner_sha256,
        expected_candidate_execution_path="smoke/execution.json",
        expected_candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
        latency_p95_limit_seconds=10.0,
    )

    too_slow = _warm_smoke_raw(
        profile=profile,
        source_raw=source_raw,
        execution_raw=execution_raw,
        warm_p95=10.01,
    )
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._verify_warm_service_smoke_evidence(
            root=root,
            raw=too_slow,
            profile=profile,
            runtime_configuration_sha256="1" * 64,
            expected_runner_implementation_sha256=runner_sha256,
            expected_candidate_execution_path="smoke/execution.json",
            expected_candidate_execution_sha256=hashlib.sha256(execution_raw).hexdigest(),
            latency_p95_limit_seconds=10.0,
        )


def test_pointer_writer_emits_v11_with_full_activation_validation_and_preserves_old_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_path = root / "profiles" / "v19.json"
    evaluation_path = root / "evaluations" / "sealed-v8.json"
    configuration_path.parent.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    configuration_raw = b"configuration"
    evaluation_raw = b"evaluation"
    configuration_path.write_bytes(configuration_raw)
    evaluation_path.write_bytes(evaluation_raw)
    evaluation = _evaluation_v8().model_copy(
        update={"runtime_configuration_sha256": hashlib.sha256(configuration_raw).hexdigest()}
    )
    profile = _profile_v19()
    approval_path = root / "approvals" / "raster-text-active.json"
    approval_path.parent.mkdir(parents=True)
    approval_path.write_bytes(b"old-pointer\n")

    monkeypatch.setattr(runtime, "parse_raster_text_offline_profile", lambda _raw: profile)
    monkeypatch.setattr(runtime, "_parse_raster_text_sealed_evaluation", lambda _raw: evaluation)
    observed_pointer_raw: list[bytes] = []

    def accept_pointer(**kwargs: object) -> None:
        raw = kwargs["approval_raw"]
        assert isinstance(raw, bytes)
        assert type(runtime.parse_raster_text_activation_pointer(raw)) is (
            runtime.RasterTextActivationPointerV11
        )
        assert kwargs["verification_mode"] == "activation"
        assert kwargs["start_service_session"] is False
        observed_pointer_raw.append(raw)

    monkeypatch.setattr(runtime, "_load_binding_from_pointer", accept_pointer)
    digest, pointer = runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path="profiles/v19.json",
        sealed_evaluation_relative_path="evaluations/sealed-v8.json",
    )
    assert type(pointer) is runtime.RasterTextActivationPointerV11
    assert pointer.engine_kind == _V19_ENGINE_KIND
    assert pointer.runtime_validation_revision == "activation-full-runtime-compact-v1"
    assert approval_path.read_bytes() == observed_pointer_raw[0]
    assert hashlib.sha256(approval_path.read_bytes()).hexdigest() == digest

    old_pointer = b"still-active\n"
    approval_path.write_bytes(old_pointer)

    def reject_pointer(**_kwargs: object) -> None:
        raise runtime.RasterTextRuntimeError("raster_text_evaluation_invalid")

    monkeypatch.setattr(runtime, "_load_binding_from_pointer", reject_pointer)
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="profiles/v19.json",
            sealed_evaluation_relative_path="evaluations/sealed-v8.json",
        )
    assert approval_path.read_bytes() == old_pointer
