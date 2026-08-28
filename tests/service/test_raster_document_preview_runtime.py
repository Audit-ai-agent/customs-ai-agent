from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from dataclasses import replace
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from pydantic import ValidationError

from customs_ai_agent.evaluation.synthetic_raster import SyntheticRasterReceiptV1
from customs_ai_agent.evaluation.synthetic_raster_candidates import (
    RasterCandidateBundleV1,
    RasterCandidateObservationV1,
    build_raster_candidate_execution,
    candidate_bundle_bytes,
    write_canonical_raster_candidate_bundle,
    write_canonical_raster_candidate_execution,
)
from customs_ai_agent.schemas.raster_document_extraction import (
    RasterDocumentExtractionRequestMetadataV2,
    RasterDocumentSpanV2,
    RasterPreviewInputV2,
)
from customs_ai_agent.service.document_input import DocumentStructure
from customs_ai_agent.service.ppstructurev3_sidecar import (
    __file__ as ppstructurev3_sidecar_module_file,
    _InputPage,
    _FIELD_RULES,
    _header_evidence_matches,
    _native_prediction,
    _table_projection_status,
    _verify_child_model_inventory,
    PpStructureV3SidecarConfigurationV1,
    PpStructureV3SidecarError,
    PpStructureV3StructureProbeResultV1,
    PpStructureV3SidecarResultV1,
    deny_network_connections,
    project_ppstructurev3_predictions,
    run_batch_from_stdio,
    run_evaluation_batch_v2_from_stdio,
    run_service_from_stdio,
    run_structure_probe_from_stdio,
    summarize_ppstructurev3_native_structure,
)
from customs_ai_agent.service.raster_document_extraction import (
    RasterDocumentPreviewInputError,
    RasterizedPageV2,
)
import customs_ai_agent.service.raster_document_preview_runtime as raster_runtime
from customs_ai_agent.service.raster_document_preview_runtime import (
    ApprovedPdfiumRasterPageRenderer,
    AutoActivatingSourceAttestedRasterPreviewRuntime,
    PdfiumRasterInputAdapter,
    PaddleOcr3DirectRasterBaselineProfileV1,
    PpStructureV3CommandExecutor,
    PpStructureV3CommandExecutorV2,
    PpStructureV3CandidateBatchSessionV2,
    PpStructureV3EvaluationBatchInputV2,
    PpStructureV3OfflineProfileV1,
    PpStructureV3OfflineProfileV2,
    PpStructureV3ServiceSessionV1,
    RasterCalibrationThresholdReceiptV2,
    RasterRuntimeError,
    RasterSealedEvaluationResultV2,
    RasterSourceAttestedActivationPointerV2,
    SourceAttestedRasterPreviewBindingV2,
    _controlled_module_root_tree_sha256,
    make_synthetic_raster_profile_revision,
    parse_source_attested_synthetic_raster_activation,
    _verify_model_inventory,
    resolve_paddleocr3_direct_raster_baseline,
    resolve_ppstructurev3_raster_candidate,
    resolve_raster_document_preview_engine,
)


def _input(content: bytes, media_type: str) -> RasterPreviewInputV2:
    return RasterPreviewInputV2(
        metadata=RasterDocumentExtractionRequestMetadataV2(
            document_asset_id="asset-1",
            extraction_request_id="request-1",
            source_sha256="a" * 64,
            media_type=media_type,
            raster_profile_revision="synthetic-raster-v1",
        ),
        byte_size=len(content),
        content=content,
    )


def _batch_frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


def _read_batch_frame(stream: io.BytesIO) -> bytes:
    size = struct.unpack(">Q", stream.read(8))[0]
    return stream.read(size)


def _sidecar_request_payload(
    *, source_sha256: str, raster_profile_revision: str, pixels: bytes
) -> bytes:
    return (
        json.dumps(
            {
                "protocol": "customs-raster-ppstructurev3-v1",
                "source_sha256": source_sha256,
                "raster_profile_revision": raster_profile_revision,
                "pages": [
                    {
                        "page_number": 1,
                        "width": 1,
                        "height": 1,
                        "pixel_length": len(pixels),
                        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                    }
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        + pixels
    )


def _evaluation_batch_payload(items: list[tuple[str, bytes]]) -> bytes:
    header = {
        "protocol": "customs-raster-ppstructurev3-evaluation-batch-v2",
        "items": [
            {
                "item_id": item_id,
                "input_length": len(payload),
                "input_sha256": hashlib.sha256(payload).hexdigest(),
            }
            for item_id, payload in items
        ],
    }
    return (
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
        + b"".join(payload for _item_id, payload in items)
    )


def test_batch_sidecar_constructs_one_pipeline_for_multiple_requests(monkeypatch) -> None:
    """The evaluation-only protocol reuses exactly one already-pinned model."""

    import customs_ai_agent.service.ppstructurev3_sidecar as sidecar

    configuration = SimpleNamespace(
        raster_profile_revision="synthetic-raster-v1:aaaaaaaaaaaaaaaaaaaaaaaa",
        extractor_revision="ppstructurev3-local-v1",
        pipeline_kwargs=lambda _root: {},
        prediction_kwargs=lambda: {},
    )
    constructions: list[object] = []

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            constructions.append(self)

    def fake_result(**kwargs: object) -> PpStructureV3SidecarResultV1:
        return PpStructureV3SidecarResultV1(
            source_sha256=cast(str, kwargs["source_sha256"]),
            raster_profile_revision=cast(str, kwargs["raster_profile_revision"]),
            provider_revision="ppstructurev3-local-v1",
            document_type="unknown",
            classification_confidence=None,
            pages=[],
            field_candidates=[],
            tables=[],
            warnings=[],
            needs_review_reasons=[],
        )

    monkeypatch.setattr(
        sidecar, "configuration_from_environment", lambda: (configuration, Path("."))
    )
    monkeypatch.setattr(sidecar, "_native_prediction", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sidecar, "project_ppstructurev3_predictions", fake_result)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=FakePipeline))

    source_sha256 = "a" * 64
    pixels = b"\xff\x00\x00"
    header = (
        json.dumps(
            {
                "protocol": "customs-raster-ppstructurev3-v1",
                "source_sha256": source_sha256,
                "raster_profile_revision": configuration.raster_profile_revision,
                "pages": [
                    {
                        "page_number": 1,
                        "width": 1,
                        "height": 1,
                        "pixel_length": 3,
                        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        + pixels
    )
    output = io.BytesIO()
    run_batch_from_stdio(io.BytesIO(_batch_frame(header) + _batch_frame(header)), output)

    output.seek(0)
    assert json.loads(_read_batch_frame(output)) == {
        "protocol": "customs-raster-ppstructurev3-batch-v1",
        "status": "ready",
    }
    assert (
        PpStructureV3SidecarResultV1.model_validate_json(_read_batch_frame(output)).source_sha256
        == source_sha256
    )
    assert (
        PpStructureV3SidecarResultV1.model_validate_json(_read_batch_frame(output)).source_sha256
        == source_sha256
    )
    assert output.read() == b""
    assert len(constructions) == 1


def test_service_sidecar_constructs_one_pipeline_for_multiple_requests(monkeypatch) -> None:
    """The approved-service protocol must not be an alias for evaluation."""

    import customs_ai_agent.service.ppstructurev3_sidecar as sidecar

    revision = "synthetic-raster-v1:aaaaaaaaaaaaaaaaaaaaaaaa"
    configuration = SimpleNamespace(
        raster_profile_revision=revision,
        extractor_revision="ppstructurev3-local-v1",
        pipeline_kwargs=lambda _root: {},
        prediction_kwargs=lambda: {},
    )
    constructions: list[object] = []

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            constructions.append(self)

    def fake_result(**kwargs: object) -> PpStructureV3SidecarResultV1:
        return PpStructureV3SidecarResultV1(
            source_sha256=cast(str, kwargs["source_sha256"]),
            raster_profile_revision=cast(str, kwargs["raster_profile_revision"]),
            provider_revision="ppstructurev3-local-v1",
            document_type="unknown",
            classification_confidence=None,
            pages=[],
            field_candidates=[],
            tables=[],
            warnings=[],
            needs_review_reasons=[],
        )

    monkeypatch.setattr(
        sidecar, "configuration_from_environment", lambda: (configuration, Path("."))
    )
    monkeypatch.setattr(sidecar, "_native_prediction", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sidecar, "project_ppstructurev3_predictions", fake_result)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=FakePipeline))

    request = _sidecar_request_payload(
        source_sha256="a" * 64,
        raster_profile_revision=revision,
        pixels=b"\xff\x00\x00",
    )
    output = io.BytesIO()
    run_service_from_stdio(io.BytesIO(_batch_frame(request) + _batch_frame(request)), output)

    output.seek(0)
    assert json.loads(_read_batch_frame(output)) == {
        "protocol": "customs-raster-ppstructurev3-service-v1",
        "status": "ready",
    }
    assert (
        PpStructureV3SidecarResultV1.model_validate_json(_read_batch_frame(output)).source_sha256
        == "a" * 64
    )
    assert (
        PpStructureV3SidecarResultV1.model_validate_json(_read_batch_frame(output)).source_sha256
        == "a" * 64
    )
    assert output.read() == b""
    assert len(constructions) == 1


def test_sidecar_native_structure_probe_reports_only_content_free_counters() -> None:
    native = {
        "overall_ocr_res": {"rec_texts": ["do not disclose this OCR text", "other text"]},
        "layout_det_res": {
            "boxes": [
                {"label": "table"},
                {"label": "text"},
                {"label": "TABLE"},
            ]
        },
        "parsing_res_list": [{}, {}],
        "table_res_list": [
            {
                "cell_box_list": [[0, 0, 10, 10], [10, 0, 20, 10]],
                "table_ocr_pred": {"rec_texts": ["cell text one", "cell text two"]},
            },
            "invalid-native-table-entry",
        ],
    }

    summary = summarize_ppstructurev3_native_structure(native, page_number=1)

    assert summary.model_dump() == {
        "page_number": 1,
        "native_output_mapping": True,
        "has_overall_ocr_result": True,
        "overall_ocr_span_count": 2,
        "has_layout_detection_result": True,
        "layout_box_count": 3,
        "layout_table_box_count": 2,
        "has_parsing_result_list": True,
        "parsing_block_count": 2,
        "native_table_list_valid": True,
        "native_table_count": 2,
        "native_table_mapping_count": 1,
        "table_cell_box_list_present": [True, False],
        "table_cell_counts": [2, 0],
        "table_ocr_prediction_present": [True, False],
        "table_ocr_span_counts": [2, 0],
        "native_table_probe_truncated": False,
    }
    encoded = summary.model_dump_json()
    assert "do not disclose" not in encoded
    assert "cell text" not in encoded
    assert "invalid-native-table-entry" not in encoded

    inconsistent = summary.model_dump()
    inconsistent["native_table_list_valid"] = False
    with pytest.raises(ValidationError, match="table counters are inconsistent"):
        PpStructureV3StructureProbeResultV1.model_validate(
            {"pages": [inconsistent], "table_projection_status": "accepted"}
        )


def test_structure_probe_sidecar_reports_native_and_projection_counters_without_ocr_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.service.ppstructurev3_sidecar as sidecar

    revision = "synthetic-raster-v1:aaaaaaaaaaaaaaaaaaaaaaaa"
    configuration = SimpleNamespace(
        raster_profile_revision=revision,
        pipeline_kwargs=lambda _root: {},
        prediction_kwargs=lambda: {},
    )
    constructions: list[object] = []

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            constructions.append(self)

    native_result = {
        "overall_ocr_res": {"rec_texts": ["unexposed OCR"]},
        "layout_det_res": {"boxes": [{"label": "table"}]},
        "parsing_res_list": [],
        "table_res_list": [
            {
                "cell_box_list": [[0, 0, 1, 1]],
                "table_ocr_pred": {"rec_texts": ["unexposed table OCR"]},
            }
        ],
    }

    monkeypatch.setattr(
        sidecar, "configuration_from_environment", lambda: (configuration, Path("."))
    )
    monkeypatch.setattr(sidecar, "_native_prediction", lambda *_args, **_kwargs: native_result)
    projection_calls: list[object] = []

    def fake_projection(**_kwargs: object) -> SimpleNamespace:
        projection_calls.append(object())
        return SimpleNamespace(
            document_type="commercial_invoice",
            pages=[SimpleNamespace(page_number=1, width=1, height=1, spans=[])],
        )

    monkeypatch.setattr(
        sidecar,
        "project_ppstructurev3_predictions",
        fake_projection,
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=FakePipeline))

    output = io.BytesIO()
    run_structure_probe_from_stdio(
        io.BytesIO(
            _sidecar_request_payload(
                source_sha256="a" * 64,
                raster_profile_revision=revision,
                pixels=b"\xff\x00\x00",
            )
        ),
        output,
    )

    result = PpStructureV3StructureProbeResultV1.model_validate_json(output.getvalue())
    assert result.protocol == "customs-raster-ppstructurev3-structure-probe-v1"
    assert result.pages[0].page_number == 1
    assert result.pages[0].layout_table_box_count == 1
    assert result.pages[0].native_table_count == 1
    assert result.pages[0].table_cell_counts == [1]
    assert result.pages[0].table_ocr_span_counts == [1]
    assert result.table_projection_status == "missing_header_evidence"
    assert b"unexposed" not in output.getvalue()
    assert len(constructions) == 1
    assert len(projection_calls) == 1


def test_evaluation_batch_v2_uses_one_native_list_call_and_rebinds_item_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch-v2 never maps duplicate source SHA values by position alone."""

    import customs_ai_agent.service.ppstructurev3_sidecar as sidecar

    revision = "synthetic-raster-v1:aaaaaaaaaaaaaaaaaaaaaaaa"
    configuration = SimpleNamespace(
        raster_profile_revision=revision,
        extractor_revision="ppstructurev3-local-v1",
        evaluation_batch_size=4,
        pipeline_kwargs=lambda _root: {},
        prediction_kwargs=lambda: {},
    )
    constructions: list[object] = []
    native_batches: list[list[_InputPage]] = []

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            constructions.append(self)

    def fake_predictions(
        _pipeline: object, pages: object, **_kwargs: object
    ) -> list[dict[str, object]]:
        native_batches.append(list(cast(list[_InputPage], pages)))
        return [{}, {}]

    def fake_result(**kwargs: object) -> PpStructureV3SidecarResultV1:
        return PpStructureV3SidecarResultV1(
            source_sha256=cast(str, kwargs["source_sha256"]),
            raster_profile_revision=cast(str, kwargs["raster_profile_revision"]),
            provider_revision="ppstructurev3-local-v1",
            document_type="unknown",
            classification_confidence=None,
            pages=[],
            field_candidates=[],
            tables=[],
            warnings=[],
            needs_review_reasons=[],
        )

    monkeypatch.setattr(
        sidecar, "configuration_from_environment", lambda: (configuration, Path("."))
    )
    monkeypatch.setattr(sidecar, "_native_predictions", fake_predictions)
    monkeypatch.setattr(sidecar, "project_ppstructurev3_predictions", fake_result)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=FakePipeline))

    # Both nested requests intentionally use the same source SHA. Only their
    # outer opaque IDs distinguish clean/error duplicate raster material.
    nested = _sidecar_request_payload(
        source_sha256="a" * 64, raster_profile_revision=revision, pixels=b"\xff\x00\x00"
    )
    output = io.BytesIO()
    run_evaluation_batch_v2_from_stdio(
        io.BytesIO(
            _batch_frame(
                _evaluation_batch_payload([("clean-item", nested), ("error-item", nested)])
            )
        ),
        output,
    )

    output.seek(0)
    assert json.loads(_read_batch_frame(output)) == {
        "protocol": "customs-raster-ppstructurev3-evaluation-batch-v2",
        "status": "ready",
    }
    response = json.loads(_read_batch_frame(output))
    assert [item["item_id"] for item in response["results"]] == ["clean-item", "error-item"]
    assert [item["result"]["source_sha256"] for item in response["results"]] == ["a" * 64] * 2
    assert output.read() == b""
    assert len(constructions) == 1
    assert [len(batch) for batch in native_batches] == [2]


def test_evaluation_batch_v2_rejects_duplicate_outer_item_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.service.ppstructurev3_sidecar as sidecar

    configuration = SimpleNamespace(
        raster_profile_revision="synthetic-raster-v1:aaaaaaaaaaaaaaaaaaaaaaaa",
        extractor_revision="ppstructurev3-local-v1",
        evaluation_batch_size=4,
        pipeline_kwargs=lambda _root: {},
        prediction_kwargs=lambda: {},
    )
    monkeypatch.setattr(
        sidecar, "configuration_from_environment", lambda: (configuration, Path("."))
    )
    monkeypatch.setitem(
        sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=lambda **_kwargs: object())
    )
    nested = _sidecar_request_payload(
        source_sha256="a" * 64,
        raster_profile_revision=configuration.raster_profile_revision,
        pixels=b"\xff\x00\x00",
    )
    with pytest.raises(PpStructureV3SidecarError, match="protocol_invalid"):
        run_evaluation_batch_v2_from_stdio(
            io.BytesIO(
                _batch_frame(
                    _evaluation_batch_payload([("duplicate", nested), ("duplicate", nested)])
                )
            ),
            io.BytesIO(),
        )


def test_parent_evaluation_batch_v2_rebinds_out_of_order_results_by_item_id() -> None:
    """The parent must not associate duplicated source hashes by response order."""

    revision = make_synthetic_raster_profile_revision("a" * 64)
    profile = _automatic_v2_profile(revision).model_copy(update={"evaluation_batch_size": 4})
    page = RasterizedPageV2(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")

    def result(source_sha256: str) -> PpStructureV3SidecarResultV1:
        return PpStructureV3SidecarResultV1(
            source_sha256=source_sha256,
            raster_profile_revision=revision,
            provider_revision=profile.extractor_revision,
            document_type="unknown",
            classification_confidence=None,
            pages=[
                {
                    "page_number": 1,
                    "width": 1,
                    "height": 1,
                    "coordinate_unit": "pixels",
                    "spans": [],
                }
            ],
            field_candidates=[],
            tables=[],
            warnings=[],
            needs_review_reasons=[],
        )

    response = json.dumps(
        {
            "protocol": "customs-raster-ppstructurev3-evaluation-batch-v2",
            # Deliberately reversed to prove IDs, not position, bind results.
            "results": [
                {"item_id": "second", "result": result("b" * 64).model_dump(mode="json")},
                {"item_id": "first", "result": result("a" * 64).model_dump(mode="json")},
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    class FakeProcess:
        def __init__(self) -> None:
            ready = json.dumps(
                {
                    "protocol": "customs-raster-ppstructurev3-evaluation-batch-v2",
                    "status": "ready",
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(_batch_frame(ready) + _batch_frame(response))

        def poll(self) -> None:
            return None

    session = PpStructureV3CandidateBatchSessionV2(
        process=cast(object, FakeProcess()), profile=profile, timeout_seconds=1
    )
    session.verify_ready()
    ledgers = session.extract_layout_batch(
        (
            PpStructureV3EvaluationBatchInputV2(
                item_id="first",
                pages=(page,),
                source_sha256="a" * 64,
                raster_profile_revision=revision,
            ),
            PpStructureV3EvaluationBatchInputV2(
                item_id="second",
                pages=(page,),
                source_sha256="b" * 64,
                raster_profile_revision=revision,
            ),
        ),
        profile=profile,
    )

    assert [ledger.source_sha256 for ledger in ledgers] == ["a" * 64, "b" * 64]


def test_service_session_reuses_one_process_and_reaps_it_on_close() -> None:
    revision = make_synthetic_raster_profile_revision("a" * 64)
    # Evaluation batching is not a serving capability. A batch-4 candidate
    # profile must still use the one-document service protocol after it has
    # been activated.
    profile = _automatic_v2_profile(revision).model_copy(update={"evaluation_batch_size": 4})
    page = RasterizedPageV2(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")

    result = PpStructureV3SidecarResultV1(
        source_sha256="a" * 64,
        raster_profile_revision=revision,
        provider_revision=profile.extractor_revision,
        document_type="unknown",
        classification_confidence=None,
        pages=[
            {
                "page_number": 1,
                "width": 1,
                "height": 1,
                "coordinate_unit": "pixels",
                "spans": [],
            }
        ],
        field_candidates=[],
        tables=[],
        warnings=[],
        needs_review_reasons=[],
    ).model_dump_json().encode("utf-8")

    class FakeProcess:
        def __init__(self) -> None:
            ready = json.dumps(
                {
                    "protocol": "customs-raster-ppstructurev3-service-v1",
                    "status": "ready",
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(_batch_frame(ready) + _batch_frame(result) + _batch_frame(result))
            self.terminated = 0
            self.waited = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated += 1

        def wait(self, *, timeout: int) -> int:
            self.waited += 1
            return 0

    process = FakeProcess()
    cleaned: list[object] = []
    session = PpStructureV3ServiceSessionV1(
        process=cast(object, process),
        profile=profile,
        request_timeout_seconds=1,
        startup_timeout_seconds=1,
        cleanup_home=lambda: cleaned.append(object()),
    )
    session.verify_ready()
    assert session.extract_layout(
        (page,),
        source_sha256="a" * 64,
        raster_profile_revision=revision,
        profile=profile,
    ).source_sha256 == "a" * 64
    assert session.extract_layout(
        (page,),
        source_sha256="a" * 64,
        raster_profile_revision=revision,
        profile=profile,
    ).source_sha256 == "a" * 64
    session.close()

    assert process.terminated == 1
    assert process.waited == 1
    assert len(cleaned) == 1


def test_auto_runtime_closes_the_retired_service_binding_once(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    pointer_path = root / "approvals" / "raster-preview-active.json"
    pointer_path.parent.mkdir(parents=True)
    revision = make_synthetic_raster_profile_revision("a" * 64)
    pointer_path.write_bytes(_automatic_v2_pointer_raw(revision))
    closed: list[str] = []

    class Executor:
        def verify_runtime_state(self) -> None:
            return None

    def loader(**kwargs: object) -> SourceAttestedRasterPreviewBindingV2:
        pointer = parse_source_attested_synthetic_raster_activation(
            cast(bytes, kwargs["approval_raw"])
        )
        return SourceAttestedRasterPreviewBindingV2(
            root=root,
            approval_path=pointer_path,
            approval_sha256=hashlib.sha256(cast(bytes, kwargs["approval_raw"])).hexdigest(),
            pointer=pointer,
            profile=_automatic_v2_profile(pointer.raster_profile_revision),
            calibration=cast(RasterCalibrationThresholdReceiptV2, object()),
            evaluation=cast(RasterSealedEvaluationResultV2, object()),
            engine=cast(object, object()),
            page_renderer=cast(ApprovedPdfiumRasterPageRenderer, object()),
            executor=cast(PpStructureV3CommandExecutorV2, Executor()),
            runtime_closer=lambda: closed.append("closed"),
        )

    runtime = AutoActivatingSourceAttestedRasterPreviewRuntime(
        approval_path=pointer_path,
        binding_loader=loader,
    )
    assert runtime.status().raster_profile_revision == revision
    runtime.close()
    runtime.close()

    assert closed == ["closed"]


def test_strict_image_decoder_fully_decodes_and_cross_checks_dimensions() -> None:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (2, 3), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    adapter = PdfiumRasterInputAdapter(pdfium_version="4.30.0", raster_dpi=216)

    page = adapter.decode_image(
        _input(content, "image/png"),
        DocumentStructure(media_type="image/png", image_width=2, image_height=3),
    )
    assert (page.width, page.height, len(page.pixels)) == (2, 3, 18)

    with pytest.raises(RasterDocumentPreviewInputError, match="decoding failed"):
        adapter.decode_image(
            _input(b"\x89PNG\r\n\x1a\ntruncated", "image/png"),
            DocumentStructure(media_type="image/png", image_width=2, image_height=3),
        )
    with pytest.raises(RasterDocumentPreviewInputError, match="dimensions drifted"):
        adapter.decode_image(
            _input(content, "image/png"),
            DocumentStructure(media_type="image/png", image_width=3, image_height=2),
        )


def test_pdfium_rasterizer_closes_child_handles_before_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    class Image(Resource):
        size = (2, 3)

        def convert(self, _mode: str) -> "Image":
            return converted

        def tobytes(self) -> bytes:
            return b"\x00" * 18

    class Bitmap(Resource):
        def to_pil(self) -> Image:
            return source

    class Page(Resource):
        def render(self, *, scale: float) -> Bitmap:
            assert scale == 3
            return bitmap

    class Document(Resource):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Page:
            assert index == 0
            return page

    converted = Image("converted")
    source = Image("source")
    bitmap = Bitmap("bitmap")
    page = Page("page")
    document = Document("document")
    pdfium = SimpleNamespace(PdfDocument=lambda _content: document)
    monkeypatch.setattr(PdfiumRasterInputAdapter, "_pdfium", lambda _self: pdfium)
    adapter = PdfiumRasterInputAdapter(pdfium_version="4.30.0", raster_dpi=216)

    result = adapter.rasterize_pdf(
        _input(b"%PDF-test", "application/pdf"),
        DocumentStructure(media_type="application/pdf", pdf_page_hint=1),
    )

    assert len(result) == 1
    assert closed == ["converted", "source", "bitmap", "page", "document"]


def test_raster_runtime_resolver_is_disabled_without_an_approved_local_pointer() -> None:
    resolution = resolve_raster_document_preview_engine(
        approval_path=None,
        approval_sha256=None,
    )
    assert resolution.status == "disabled"


def test_direct_paddleocr3_baseline_is_explicitly_non_approval_and_disabled_without_config(
    tmp_path: Path,
) -> None:
    profile = PaddleOcr3DirectRasterBaselineProfileV1(
        extractor_revision="paddleocr3-direct-baseline-v1",
        raster_profile_revision="synthetic-raster-v1",
        pdfium_version="4.30.0",
        raster_dpi=216,
        package_inventory_sha256="a" * 64,
        model_inventory_sha256="b" * 64,
        package_inventory_path="runtime/packages.json",
        model_inventory_path="runtime/models.json",
        required_packages=["paddleocr", "paddlepaddle", "pypdfium2"],
        local_model_root="models",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_directory="det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_recognition_model_directory="rec",
    )
    assert profile.approval_eligible is False
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    resolution = resolve_paddleocr3_direct_raster_baseline(
        document_preview_root=root,
        configuration_path=None,
        configuration_sha256=None,
    )
    assert (resolution.status, resolution.engine) == ("disabled", None)


def test_offline_profile_rejects_remote_or_traversal_model_roots() -> None:
    profile = PpStructureV3OfflineProfileV1(
        extractor_revision="ppstructurev3-local-v1",
        raster_profile_revision="synthetic-raster-v1",
        pdfium_version="4.30.0",
        raster_dpi=216,
        package_inventory_sha256="b" * 64,
        model_inventory_sha256="c" * 64,
        package_inventory_path="runtime/packages.json",
        model_inventory_path="runtime/models.json",
        required_packages=["paddleocr", "paddlepaddle", "pypdfium2"],
        local_model_root="../models",
        local_sidecar_command="bin/ppstructure-sidecar",
        sidecar_command_sha256="d" * 64,
        sidecar_module_sha256="e" * 64,
        sidecar_configuration_path="runtime/ppstructurev3-sidecar.json",
        sidecar_configuration_sha256="f" * 64,
    )
    with pytest.raises(RasterRuntimeError, match="configuration_invalid"):
        profile.model_root_relative_path()


def _write_json(path: Path, value: object) -> bytes:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _sidecar_configuration_payload() -> dict[str, object]:
    return {
        "extractor_revision": "ppstructurev3-local-v1",
        "raster_profile_revision": "synthetic-raster-v1",
        "layout_detection_model_name": "PP-DocLayout-S",
        "layout_detection_model_directory": "layout",
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_directory": "det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_directory": "rec",
        "table_classification_model_name": "PP-LCNet_x1_0_table_cls",
        "table_classification_model_directory": "table-classifier",
        "wired_table_structure_recognition_model_name": "SLANeXt_wired",
        "wired_table_structure_recognition_model_directory": "wired-structure",
        "wireless_table_structure_recognition_model_name": "SLANeXt_wireless",
        "wireless_table_structure_recognition_model_directory": "wireless-structure",
        "wired_table_cells_detection_model_name": "RT-DETR-L_wired_table_cell_det",
        "wired_table_cells_detection_model_directory": "wired-cells",
        "wireless_table_cells_detection_model_name": "RT-DETR-L_wireless_table_cell_det",
        "wireless_table_cells_detection_model_directory": "wireless-cells",
        "cpu_threads": 6,
        "inference_engine": "paddle_static",
        "paddle_static_run_mode": "paddle",
        "paddle_static_enable_new_ir": False,
        "paddle_static_enable_cinn": False,
        "text_detection_limit_side_len": 960,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.6,
        "text_detection_unclip_ratio": 1.5,
    }


def _complete_synthetic_raster_receipt() -> dict[str, object]:
    profiles = [
        {
            "profile_id": "png-216",
            "media_type": "image/png",
            "dpi": 216,
            "jpeg_quality": None,
            "image_only_pdf": False,
        },
        {
            "profile_id": "jpeg-216-q75",
            "media_type": "image/jpeg",
            "dpi": 216,
            "jpeg_quality": 75,
            "image_only_pdf": False,
        },
        {
            "profile_id": "jpeg-150-q75",
            "media_type": "image/jpeg",
            "dpi": 150,
            "jpeg_quality": 75,
            "image_only_pdf": False,
        },
        {
            "profile_id": "scan-pdf-216-q75",
            "media_type": "application/pdf",
            "dpi": 216,
            "jpeg_quality": 75,
            "image_only_pdf": True,
        },
    ]
    suffix_by_profile = {
        "png-216": ".png",
        "jpeg-216-q75": ".jpg",
        "jpeg-150-q75": ".jpg",
        "scan-pdf-216-q75": ".pdf",
    }
    split_counts = (("train", 750), ("development", 100), ("calibration", 50), ("sealed", 100))
    assets: list[dict[str, object]] = []
    family_index = 0
    for split, count in split_counts:
        for _ in range(count):
            family_index += 1
            family_id = f"family-{family_index:04d}"
            for variant in ("clean", "error"):
                for document_type in ("bill_of_lading", "commercial_invoice", "packing_list"):
                    asset_id = f"{family_id}-{variant}-{document_type}"
                    for profile in profiles:
                        assets.append(
                            {
                                "asset_id": asset_id,
                                "family_id": family_id,
                                "document_type": document_type,
                                "split": split,
                                "profile_id": profile["profile_id"],
                                "source_pdf_sha256": "a" * 64,
                                "image_sha256": "b" * 64,
                                "asset_relative_path": (
                                    f"synthetic-raster-v1/assets/{asset_id}-{profile['profile_id']}"
                                    f"{suffix_by_profile[str(profile['profile_id'])]}"
                                ),
                                "media_type": profile["media_type"],
                                "byte_size": 10,
                                "width": 300,
                                "height": 600,
                                "coordinate_transform": "pdf-points-lower-left-to-normalized-top-left-v1",
                                "render_recipe_sha256": "c" * 64,
                                "render_seed_sha256": "d" * 64,
                                "gold_relative_path": (
                                    f"synthetic-raster-v1/gold/{asset_id}-{profile['profile_id']}.json"
                                ),
                                "gold_sha256": "e" * 64,
                            }
                        )
    return {
        "source_dataset_id": "synthetic-customs-pdf-2000",
        "source_dataset_revision": "20260811-r1",
        "source_receipt_sha256": "e" * 64,
        "source_receipt_relative_path": "synthetic-raster-v1/source-receipt.json",
        "build_inputs_relative_path": "runtime/raster-build-inputs.json",
        "build_inputs_sha256": "9" * 64,
        "renderer_revision": "pdfium-raster-v1",
        "renderer_configuration_sha256": "f" * 64,
        "pdfium_version": "4.30.0",
        "font_manifest_sha256": "1" * 64,
        "split_seed_sha256": "2" * 64,
        "render_recipe_sha256": "c" * 64,
        "render_seed_sha256": "d" * 64,
        "profiles": profiles,
        "assets": assets,
    }


def _write_candidate_provenance(
    *,
    root: Path,
    receipt_raw: bytes,
    runtime_configuration_sha256: str,
    extractor_revision: str,
    split: str,
) -> dict[str, object]:
    """Build a contentful fake candidate bundle without OCR/PDFium packages."""

    receipt = SyntheticRasterReceiptV1.model_validate_json(receipt_raw)
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    observations: list[RasterCandidateObservationV1] = []
    for asset in sorted(
        (item for item in receipt.assets if item.split == split),
        key=lambda item: (item.asset_id, item.profile_id),
    ):
        observations.append(
            RasterCandidateObservationV1(
                asset_id=asset.asset_id,
                profile_id=asset.profile_id,
                candidate={
                    "document_asset_id": asset.asset_id,
                    "extraction_request_id": f"candidate-{asset.asset_id}",
                    "raster_profile_revision": "synthetic-raster-v1",
                    "status": "needs_review",
                    "media_type": asset.media_type,
                    "byte_size": asset.byte_size,
                    "source_sha256": asset.image_sha256,
                    "canonical_text_sha256": hashlib.sha256(b"x").hexdigest(),
                    "extractor_revision": extractor_revision,
                    "document_type": "unknown",
                    "classification_confidence": None,
                    "pages": [
                        {
                            "page_number": 1,
                            "width": asset.width,
                            "height": asset.height,
                            "coordinate_unit": "pixels",
                            "spans": [
                                {
                                    "span_id": "span-1",
                                    "ordinal": 1,
                                    "text": "x",
                                    "bbox": {"x": 0, "y": 0, "width": 1, "height": 1},
                                    "polygon": None,
                                    "confidence": 0.5,
                                    "reading_order": 1,
                                }
                            ],
                        }
                    ],
                    "field_candidates": [],
                    "tables": [],
                    "warnings": [],
                    "needs_review_reasons": [{"code": "document_type_unknown"}],
                },
            )
        )
    bundle = RasterCandidateBundleV1(
        split=split,
        synthetic_raster_receipt_sha256=receipt_sha256,
        extractor_revision=extractor_revision,
        candidates=observations,
    )
    bundle_relative_path, bundle_sha256, _bundle_size = write_canonical_raster_candidate_bundle(
        document_preview_root=root, bundle=bundle
    )
    execution = build_raster_candidate_execution(
        bundle=bundle,
        candidate_bundle_relative_path_value=bundle_relative_path,
        candidate_bundle_raw=candidate_bundle_bytes(bundle),
        runtime_configuration_sha256=runtime_configuration_sha256,
    )
    execution_relative_path, execution_sha256 = write_canonical_raster_candidate_execution(
        document_preview_root=root, execution=execution
    )
    return {
        "candidate_bundle_relative_path": bundle_relative_path,
        "candidate_bundle_sha256": bundle_sha256,
        "candidate_execution_relative_path": execution_relative_path,
        "candidate_execution_sha256": execution_sha256,
    }


def test_resolver_builds_the_default_hash_pinned_command_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No optional OCR wheel is needed to verify the normal activation wiring."""

    root = tmp_path / "_local" / "data" / "document-preview"
    approval_path = root / "approvals" / "raster-preview-active.json"
    command = root / "bin" / "ppstructure-sidecar"
    command.parent.mkdir(parents=True)
    command.write_bytes(b"local PP-StructureV3 sidecar placeholder")
    model_root = root / "models"
    configuration_payload = _sidecar_configuration_payload()
    model_artifacts: list[dict[str, str]] = []
    for directory in (
        "layout",
        "det",
        "rec",
        "table-classifier",
        "wired-structure",
        "wireless-structure",
        "wired-cells",
        "wireless-cells",
    ):
        model = model_root / directory / "model.bin"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(f"pinned-{directory}".encode("ascii"))
        model_artifacts.append(
            {
                "relative_path": f"{directory}/model.bin",
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        )
    package_names = ["paddleocr", "paddlepaddle", "pypdfium2"]
    versions = {name: "test-1.0" for name in package_names}
    monkeypatch.setattr(
        "customs_ai_agent.service.raster_document_preview_runtime.importlib.metadata.version",
        lambda name: versions[name],
    )
    package_raw = _write_json(
        root / "runtime" / "packages.json",
        {"packages": [{"name": name, "version": versions[name]} for name in package_names]},
    )
    model_raw = _write_json(
        root / "runtime" / "models.json",
        {"artifacts": model_artifacts},
    )
    sidecar_configuration_raw = _write_json(
        root / "runtime" / "ppstructurev3-sidecar.json", configuration_payload
    )
    profile = {
        "extractor_revision": "ppstructurev3-local-v1",
        "raster_profile_revision": "synthetic-raster-v1",
        "pdfium_version": "4.30.0",
        "raster_dpi": 216,
        "package_inventory_sha256": hashlib.sha256(package_raw).hexdigest(),
        "model_inventory_sha256": hashlib.sha256(model_raw).hexdigest(),
        "package_inventory_path": "runtime/packages.json",
        "model_inventory_path": "runtime/models.json",
        "required_packages": package_names,
        "local_model_root": "models",
        "local_sidecar_command": "bin/ppstructure-sidecar",
        "sidecar_command_sha256": hashlib.sha256(command.read_bytes()).hexdigest(),
        "sidecar_module_sha256": hashlib.sha256(
            Path(ppstructurev3_sidecar_module_file).read_bytes()
        ).hexdigest(),
        "sidecar_configuration_path": "runtime/ppstructurev3-sidecar.json",
        "sidecar_configuration_sha256": hashlib.sha256(sidecar_configuration_raw).hexdigest(),
    }
    profile_raw = _write_json(root / "runtime" / "profile.json", profile)
    synthetic_receipt_path = "synthetic-raster-v1/synthetic-raster-receipt.json"
    synthetic_receipt_raw = _write_json(
        root / "synthetic-raster-v1" / "synthetic-raster-receipt.json",
        _complete_synthetic_raster_receipt(),
    )
    runtime_configuration_sha256 = hashlib.sha256(profile_raw).hexdigest()
    calibration_candidate = _write_candidate_provenance(
        root=root,
        receipt_raw=synthetic_receipt_raw,
        runtime_configuration_sha256=runtime_configuration_sha256,
        extractor_revision="ppstructurev3-local-v1",
        split="calibration",
    )
    sealed_candidate = _write_candidate_provenance(
        root=root,
        receipt_raw=synthetic_receipt_raw,
        runtime_configuration_sha256=runtime_configuration_sha256,
        extractor_revision="ppstructurev3-local-v1",
        split="sealed",
    )
    calibration_raw = _write_json(
        root / "evaluations" / "calibration-thresholds.json",
        {
            "dataset_id": "synthetic-raster-v1",
            "split": "calibration",
            "raster_profile_revision": "synthetic-raster-v1",
            "runtime_configuration_sha256": runtime_configuration_sha256,
            "package_inventory_sha256": hashlib.sha256(package_raw).hexdigest(),
            "model_inventory_sha256": hashlib.sha256(model_raw).hexdigest(),
            "extractor_revision": "ppstructurev3-local-v1",
            "synthetic_raster_receipt_path": synthetic_receipt_path,
            "synthetic_raster_receipt_sha256": hashlib.sha256(synthetic_receipt_raw).hexdigest(),
            **calibration_candidate,
            "calibration_case_manifest_sha256": "d" * 64,
            "thresholds": {
                "classification_confidence": 0.9,
                "span_confidence": 0.9,
                "field_confidence": 0.9,
                "table_cell_confidence": 0.9,
            },
        },
    )
    evaluation_raw = _write_json(
        root / "evaluations" / "sealed.json",
        {
            "dataset_id": "synthetic-raster-v1",
            "split": "sealed",
            "status": "passed",
            "raster_profile_revision": "synthetic-raster-v1",
            "runtime_configuration_sha256": runtime_configuration_sha256,
            "package_inventory_sha256": hashlib.sha256(package_raw).hexdigest(),
            "model_inventory_sha256": hashlib.sha256(model_raw).hexdigest(),
            "extractor_revision": "ppstructurev3-local-v1",
            "synthetic_raster_receipt_path": synthetic_receipt_path,
            "synthetic_raster_receipt_sha256": hashlib.sha256(synthetic_receipt_raw).hexdigest(),
            **sealed_candidate,
            "sealed_case_manifest_sha256": "e" * 64,
            "calibration_case_manifest_sha256": "d" * 64,
            "calibration_threshold_receipt_path": "evaluations/calibration-thresholds.json",
            "calibration_threshold_receipt_sha256": hashlib.sha256(calibration_raw).hexdigest(),
            "gate_scores": [
                {
                    "document_type": document_type,
                    "profile_id": profile_id,
                    "field_exact_match": 0.98,
                    "character_error_rate": 0.02,
                    "table_cell_f1": 0.95,
                    "span_localization_f1": 0.90,
                    "bbox_iou_threshold": 0.5,
                }
                for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
                for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
            ],
        },
    )
    approval_raw = _write_json(
        approval_path,
        {
            "status": "approved",
            "raster_profile_revision": "synthetic-raster-v1",
            "runtime_configuration_path": "runtime/profile.json",
            "runtime_configuration_sha256": runtime_configuration_sha256,
            "sealed_evaluation_path": "evaluations/sealed.json",
            "sealed_evaluation_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
        },
    )

    resolution = resolve_raster_document_preview_engine(
        approval_path=approval_path,
        approval_sha256=hashlib.sha256(approval_raw).hexdigest(),
        # This fixture intentionally has only receipt metadata; production
        # activation uses the default full source/build/raster/Gold verifier.
        receipt_material_verifier=lambda **_kwargs: None,
    )
    assert resolution.status == "approved"
    assert isinstance(resolution.engine.provider.executor, PpStructureV3CommandExecutor)
    assert (
        resolution.engine.provider.executor.configuration_path.name == "ppstructurev3-sidecar.json"
    )
    candidate_resolution = resolve_ppstructurev3_raster_candidate(
        document_preview_root=root,
        configuration_relative_path="runtime/profile.json",
    )
    assert candidate_resolution.status == "available"
    assert candidate_resolution.runtime_configuration_sha256 == runtime_configuration_sha256
    assert candidate_resolution.profile is not None
    assert candidate_resolution.profile.extractor_revision == "ppstructurev3-local-v1"
    assert resolution.page_renderer is not None

    # An old approval/profile must not acquire the new static-engine and
    # detector settings through parser defaults.  Its original configuration
    # SHA is updated consistently here so the resolver reaches the sidecar
    # schema check rather than failing for an unrelated hash mismatch.
    historical_configuration = dict(configuration_payload)
    for field in (
        "inference_engine",
        "paddle_static_run_mode",
        "paddle_static_enable_new_ir",
        "paddle_static_enable_cinn",
        "text_detection_limit_side_len",
        "text_detection_limit_type",
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
    ):
        historical_configuration.pop(field)
    historical_configuration_raw = _write_json(
        root / "runtime" / "ppstructurev3-sidecar.json", historical_configuration
    )
    profile["sidecar_configuration_sha256"] = hashlib.sha256(
        historical_configuration_raw
    ).hexdigest()
    profile_raw = _write_json(root / "runtime" / "profile.json", profile)
    approval_raw = _write_json(
        approval_path,
        {
            "status": "approved",
            "raster_profile_revision": "synthetic-raster-v1",
            "runtime_configuration_path": "runtime/profile.json",
            "runtime_configuration_sha256": hashlib.sha256(profile_raw).hexdigest(),
            "sealed_evaluation_path": "evaluations/sealed.json",
            "sealed_evaluation_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
        },
    )
    assert (
        resolve_raster_document_preview_engine(
            approval_path=approval_path,
            approval_sha256=hashlib.sha256(approval_raw).hexdigest(),
            receipt_material_verifier=lambda **_kwargs: None,
        ).status
        == "unavailable"
    )
    assert (
        resolve_raster_document_preview_engine(
            approval_path=approval_path,
            approval_sha256=hashlib.sha256(approval_raw).hexdigest(),
        ).status
        == "unavailable"
    )
    assert (
        resolve_ppstructurev3_raster_candidate(
            document_preview_root=root,
            configuration_relative_path="runtime/profile.json",
        ).status
        == "unavailable"
    )

    # The remaining executor-drift assertions exercise the originally
    # resolved immutable binding, so restore its exact configuration bytes.
    (root / "runtime" / "ppstructurev3-sidecar.json").write_bytes(sidecar_configuration_raw)
    executor = resolution.engine.provider.executor
    assert isinstance(executor, PpStructureV3CommandExecutor)
    (model_root / "layout" / "model.bin").write_bytes(b"replaced-after-approval")
    with pytest.raises(RasterRuntimeError, match="model_inventory_drift"):
        executor._validate_runtime_bindings()
    (model_root / "layout" / "model.bin").write_bytes(b"pinned-layout")
    versions["paddleocr"] = "drifted-version"
    with pytest.raises(RasterRuntimeError, match="package_inventory_drift"):
        executor._validate_runtime_bindings()


def test_sidecar_model_inventory_rejects_drift_before_native_import(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model = model_root / "layout" / "model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"pinned-model")
    inventory_raw = _write_json(
        tmp_path / "runtime" / "models.json",
        {
            "artifacts": [
                {
                    "relative_path": "layout/model.bin",
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    _verify_child_model_inventory(
        model_root=model_root,
        model_inventory_path=tmp_path / "runtime" / "models.json",
        expected_sha256=hashlib.sha256(inventory_raw).hexdigest(),
    )
    model.write_bytes(b"mutated-model")
    with pytest.raises(PpStructureV3SidecarError, match="model_inventory_drift"):
        _verify_child_model_inventory(
            model_root=model_root,
            model_inventory_path=tmp_path / "runtime" / "models.json",
            expected_sha256=hashlib.sha256(inventory_raw).hexdigest(),
        )


def test_sidecar_configuration_requires_explicit_local_model_directories() -> None:
    payload = _sidecar_configuration_payload()
    payload["wired_table_cells_detection_model_directory"] = "../wired-cells"
    with pytest.raises(ValueError, match="model directory must be local"):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_cpu_threads_are_explicit_and_limited_to_bound_profiles() -> None:
    """CPU parallelism is profile material, never a host-derived default."""

    for cpu_threads in (1, 6):
        payload = _sidecar_configuration_payload()
        payload["cpu_threads"] = cpu_threads
        assert (
            PpStructureV3SidecarConfigurationV1.model_validate(payload).cpu_threads == cpu_threads
        )

    for cpu_threads in (2, 12):
        payload = _sidecar_configuration_payload()
        payload["cpu_threads"] = cpu_threads
        with pytest.raises(ValidationError):
            PpStructureV3SidecarConfigurationV1.model_validate(payload)

    payload = _sidecar_configuration_payload()
    payload.pop("cpu_threads")
    with pytest.raises(ValidationError):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_evaluation_batch_size_is_explicit_release_material() -> None:
    for batch_size in (1, 4):
        payload = _sidecar_configuration_payload()
        payload["evaluation_batch_size"] = batch_size
        assert (
            PpStructureV3SidecarConfigurationV1.model_validate(payload).evaluation_batch_size
            == batch_size
        )

    payload = _sidecar_configuration_payload()
    payload["evaluation_batch_size"] = 2
    with pytest.raises(ValidationError):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_detector_long_side_is_a_small_explicit_profile_allow_list() -> None:
    for long_side in (960, 1536, 2048, 2526):
        payload = _sidecar_configuration_payload()
        payload["text_detection_limit_side_len"] = long_side
        configuration = PpStructureV3SidecarConfigurationV1.model_validate(payload)
        assert configuration.prediction_kwargs() == {
            "use_table_orientation_classify": False,
            "use_ocr_results_with_table_cells": True,
            "use_e2e_wired_table_rec_model": False,
            "use_e2e_wireless_table_rec_model": False,
        }
        assert configuration.text_detection_limit_side_len == long_side

    payload = _sidecar_configuration_payload()
    payload["text_detection_limit_side_len"] = 1200
    with pytest.raises(ValidationError):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_detector_box_threshold_is_a_small_explicit_profile_allow_list() -> None:
    for box_threshold in (0.4, 0.6):
        payload = _sidecar_configuration_payload()
        payload["text_detection_box_threshold"] = box_threshold
        assert (
            PpStructureV3SidecarConfigurationV1.model_validate(payload).text_detection_box_threshold
            == box_threshold
        )

    payload = _sidecar_configuration_payload()
    payload["text_detection_box_threshold"] = 0.5
    with pytest.raises(ValidationError):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_detector_unclip_ratio_is_a_small_explicit_profile_allow_list() -> None:
    for unclip_ratio in (1.5, 2.0):
        payload = _sidecar_configuration_payload()
        payload["text_detection_unclip_ratio"] = unclip_ratio
        assert (
            PpStructureV3SidecarConfigurationV1.model_validate(payload).text_detection_unclip_ratio
            == unclip_ratio
        )

    payload = _sidecar_configuration_payload()
    payload["text_detection_unclip_ratio"] = 1.8
    with pytest.raises(ValidationError):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_keeps_ppstructure_constructor_and_predict_options_separate(
    tmp_path: Path,
) -> None:
    configuration = PpStructureV3SidecarConfigurationV1.model_validate(
        _sidecar_configuration_payload()
    )
    for relative in configuration.model_directories():
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)

    constructor = configuration.pipeline_kwargs(tmp_path)
    prediction = configuration.prediction_kwargs()

    assert "use_table_orientation_classify" not in constructor
    assert "use_ocr_results_with_table_cells" not in constructor
    assert "use_e2e_wired_table_rec_model" not in constructor
    assert "use_e2e_wireless_table_rec_model" not in constructor
    # Do not inherit the upstream PP-Structure YAML's unsafe ``min`` resize
    # mode.  These constructor values are profile material and deliberately
    # stay separate from predict-only table options.
    assert constructor["engine"] == "paddle_static"
    assert constructor["cpu_threads"] == 6
    assert constructor["engine_config"] == {
        "run_mode": "paddle",
        "cpu_threads": 6,
        "enable_new_ir": False,
        "enable_cinn": False,
    }
    assert constructor["text_det_limit_side_len"] == 960
    assert constructor["text_det_limit_type"] == "max"
    assert constructor["text_det_thresh"] == 0.3
    assert constructor["text_det_box_thresh"] == 0.6
    assert constructor["text_det_unclip_ratio"] == 1.5
    assert prediction == {
        "use_table_orientation_classify": False,
        "use_ocr_results_with_table_cells": True,
        "use_e2e_wired_table_rec_model": False,
        "use_e2e_wireless_table_rec_model": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("inference_engine", "paddle"),
        ("paddle_static_enable_new_ir", True),
        ("text_detection_limit_side_len", 736),
        ("text_detection_limit_type", "min"),
    ),
)
def test_sidecar_rejects_unverified_windows_detector_runtime_variants(
    field: str, value: object
) -> None:
    payload = _sidecar_configuration_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_rejects_a_historical_config_without_safe_engine_detector_fields() -> None:
    payload = _sidecar_configuration_payload()
    for field in (
        "inference_engine",
        "paddle_static_run_mode",
        "paddle_static_enable_new_ir",
        "paddle_static_enable_cinn",
        "text_detection_limit_side_len",
        "text_detection_limit_type",
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
    ):
        payload.pop(field)

    with pytest.raises(ValidationError):
        PpStructureV3SidecarConfigurationV1.model_validate(payload)


def test_sidecar_passes_predict_only_options_to_native_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bgr_image = object()

    class _Array:
        def reshape(self, _shape: object) -> object:
            return self

        def __getitem__(self, key: object) -> object:
            assert key == (slice(None), slice(None), slice(None, None, -1))
            return "reversed-rgb-view"

    class _Numpy:
        uint8 = object()

        @staticmethod
        def frombuffer(_pixels: bytes, *, dtype: object) -> _Array:
            assert dtype is _Numpy.uint8
            return _Array()

        @staticmethod
        def ascontiguousarray(value: object) -> object:
            assert value == "reversed-rgb-view"
            return bgr_image

    class _Pipeline:
        received: dict[str, object] | None = None
        received_image: object | None = None

        def predict(self, image: object, **kwargs: object) -> list[dict[str, object]]:
            self.received_image = image
            self.received = kwargs
            return [{"ok": True}]

    pipeline = _Pipeline()
    monkeypatch.setitem(sys.modules, "numpy", _Numpy)
    result = _native_prediction(
        pipeline,
        _InputPage(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00"),
        prediction_kwargs={"use_ocr_results_with_table_cells": True},
    )

    assert result == {"ok": True}
    assert pipeline.received_image is bgr_image
    assert pipeline.received == {"use_ocr_results_with_table_cells": True}


def test_model_inventory_rejects_an_undeclared_file_in_the_explicit_model_root(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    declared = model_root / "layout" / "model.bin"
    extra = model_root / "layout" / "unexpected.bin"
    declared.parent.mkdir(parents=True)
    declared.write_bytes(b"declared")
    extra.write_bytes(b"not-in-inventory")
    raw = json.dumps(
        {
            "artifacts": [
                {
                    "relative_path": "layout/model.bin",
                    "sha256": hashlib.sha256(declared.read_bytes()).hexdigest(),
                }
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")

    with pytest.raises(RasterRuntimeError, match="model_inventory_drift"):
        _verify_model_inventory(
            raw=raw,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            model_root=model_root,
        )


def test_sidecar_denies_network_in_its_own_process_guard() -> None:
    with deny_network_connections():
        with socket.socket() as client:
            with pytest.raises(OSError, match="network access denied"):
                client.connect(("127.0.0.1", 9))


_KNOWN_FORM_TABLE_HEADERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "bill_of_lading": (
        "BILL OF LADING",
        (
            "Container No",
            "Seal No",
            "No. & Kind of Packages/Containers",
            "Description of Goods",
            "Gross Weight",
            "Measurement",
        ),
    ),
    "commercial_invoice": (
        "COMMERCIAL INVOICE",
        ("Marks/No.", "Description of Goods", "Quantity", "Unit Price", "Amount"),
    ),
    "packing_list": (
        "PACKING LIST",
        ("Marks/Numbers/PKGS", "Description of Goods", "Net Weight", "Gross Weight", "Measurement"),
    ),
}


def _known_form_native_result(
    *,
    document_type: str,
    omit_body_column: int | None = None,
    additional_complete_body_row: bool = False,
    include_footer_merge: bool = False,
    obscure_header_text: bool = False,
    reverse_split_header: bool = False,
) -> dict[str, object]:
    """Build observed OCR/cell geometry only; native table OCR is decoy data."""

    title, headers = _KNOWN_FORM_TABLE_HEADERS[document_type]
    boundaries = [100 + (100 * index) for index in range(len(headers) + 1)]
    texts = [title]
    scores = [0.99]
    polygons: list[list[list[int]]] = [[[100, 30], [900, 30], [900, 60], [100, 60]]]
    cell_boxes: list[list[int]] = []
    for index, label in enumerate(headers):
        left, right = boundaries[index], boundaries[index + 1]
        cell_boxes.append([left, 100, right, 140])
        if reverse_split_header and document_type == "commercial_invoice" and index == 1:
            # Provider order is deliberately right-to-left, while the two
            # observed boxes remain a clear single header row.
            midpoint = (left + right) // 2
            texts.extend(("of Goods", "Description"))
            scores.extend((0.98, 0.98))
            polygons.extend(
                (
                    [
                        [midpoint + 2, 110],
                        [right - 8, 110],
                        [right - 8, 130],
                        [midpoint + 2, 130],
                    ],
                    [
                        [left + 8, 110],
                        [midpoint - 2, 110],
                        [midpoint - 2, 130],
                        [left + 8, 130],
                    ],
                )
            )
            continue
        texts.append(f"unrecognized-header-{index}" if obscure_header_text else label)
        scores.append(0.98)
        polygons.append([[left + 8, 110], [right - 8, 110], [right - 8, 130], [left + 8, 130]])
    for index in range(len(headers)):
        if index == omit_body_column:
            continue
        left, right = boundaries[index], boundaries[index + 1]
        cell_boxes.append([left, 140, right, 280])
        texts.append(f"line-item-{index}")
        scores.append(0.97)
        polygons.append([[left + 8, 170], [right - 8, 170], [right - 8, 190], [left + 8, 190]])
    if additional_complete_body_row:
        for index in range(len(headers)):
            left, right = boundaries[index], boundaries[index + 1]
            cell_boxes.append([left, 300, right, 440])
            texts.append(f"second-line-item-{index}")
            scores.append(0.96)
            polygons.append([[left + 8, 330], [right - 8, 330], [right - 8, 350], [left + 8, 350]])
    if include_footer_merge:
        # The observed Invoice probe had 23 native cells.  Model the same
        # high-level condition without carrying any real OCR/document content:
        # 13 merged footer fragments after the complete 10-cell target.  Every
        # fragment overlaps several columns, but all are in later rows and must
        # not make the uniquely evidenced target ambiguous.
        for index in range(13):
            top = 470 + (30 * index)
            cell_boxes.append([boundaries[0], top, boundaries[-1], top + 20])
    return {
        "overall_ocr_res": {"rec_texts": texts, "rec_scores": scores, "rec_polys": polygons},
        "parsing_res_list": (
            [] if reverse_split_header else [{"block_order": 1, "block_bbox": [0, 0, 1_000, 1_000]}]
        ),
        "table_res_list": [
            {
                "cell_box_list": cell_boxes,
                "table_ocr_pred": {"rec_texts": ["private-native-table-ocr-decoy"]},
            }
        ],
    }


def _known_form_projection(
    *,
    document_type: str,
    omit_body_column: int | None = None,
    additional_complete_body_row: bool = False,
    include_footer_merge: bool = False,
    obscure_header_text: bool = False,
    reverse_split_header: bool = False,
) -> tuple[PpStructureV3SidecarResultV1, dict[str, object]]:
    native = _known_form_native_result(
        document_type=document_type,
        omit_body_column=omit_body_column,
        additional_complete_body_row=additional_complete_body_row,
        include_footer_merge=include_footer_merge,
        obscure_header_text=obscure_header_text,
        reverse_split_header=reverse_split_header,
    )
    configuration = PpStructureV3SidecarConfigurationV1.model_validate(
        _sidecar_configuration_payload()
    )
    result = project_ppstructurev3_predictions(
        source_sha256="a" * 64,
        raster_profile_revision="synthetic-raster-v1",
        configuration=configuration,
        pages=[_InputPage(page_number=1, width=1_000, height=1_000, pixels=b"\x00" * 3_000_000)],
        native_results=[native],
    )
    return result, native


@pytest.mark.parametrize(
    ("document_type", "expected_columns"),
    [("bill_of_lading", 6), ("commercial_invoice", 5), ("packing_list", 5)],
)
def test_sidecar_selects_one_complete_known_form_header_body_subgrid(
    document_type: str,
    expected_columns: int,
) -> None:
    result, native = _known_form_projection(
        document_type=document_type,
        include_footer_merge=True,
    )

    assert result.document_type == document_type
    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.table_id == "items"
    assert [(row.row_index, len(row.cells)) for row in table.rows] == [
        (0, expected_columns),
        (1, expected_columns),
    ]
    assert len({cell.cell_id for row in table.rows for cell in row.cells}) == 2 * expected_columns
    native_table = cast(list[object], native["table_res_list"])[0]
    assert isinstance(native_table, dict)
    assert len(cast(list[object], native_table["cell_box_list"])) == (2 * expected_columns) + 13
    # The target must be constructed from page OCR evidence, not native
    # table-recognizer text.  The native decoy is intentionally absent from
    # every external sidecar result field.
    assert "private-native-table-ocr-decoy" not in result.model_dump_json()
    assert "private-native-table-ocr-decoy" in json.dumps(native)


def test_sidecar_rejects_a_known_form_target_with_one_missing_body_cell() -> None:
    result, native = _known_form_projection(
        document_type="commercial_invoice",
        omit_body_column=2,
    )
    preview_page = result.pages[0]

    assert result.tables == []
    assert "table_structure_incomplete" in {reason.code for reason in result.needs_review_reasons}
    assert (
        _table_projection_status(
            document_type="commercial_invoice",
            native_results=[native],
            pages=[preview_page],
        )
        == "body_grid_incomplete"
    )


def test_sidecar_selects_one_unique_known_form_geometry_grid_when_header_ocr_is_missing() -> None:
    result, _native = _known_form_projection(
        document_type="commercial_invoice",
        include_footer_merge=True,
        obscure_header_text=True,
    )

    assert len(result.tables) == 1
    assert [(row.row_index, len(row.cells)) for row in result.tables[0].rows] == [
        (0, 5),
        (1, 5),
    ]


def test_sidecar_uses_native_cell_visual_order_for_one_exact_split_header_only() -> None:
    result, native = _known_form_projection(
        document_type="commercial_invoice",
        reverse_split_header=True,
    )

    assert [(row.row_index, len(row.cells)) for row in result.tables[0].rows] == [
        (0, 5),
        (1, 5),
    ]
    assert "provider_reading_order_unavailable" in {
        reason.code for reason in result.needs_review_reasons
    }
    assert (
        _table_projection_status(
            document_type="commercial_invoice",
            native_results=[native],
            pages=result.pages,
        )
        == "accepted"
    )


def test_sidecar_does_not_visually_join_split_header_text_across_rows() -> None:
    spans = (
        RasterDocumentSpanV2(
            span_id="right-first",
            ordinal=1,
            reading_order=1,
            text="of Goods",
            bbox={"x": 0.60, "y": 0.60, "width": 0.20, "height": 0.10},
            polygon=None,
            confidence=0.98,
        ),
        RasterDocumentSpanV2(
            span_id="left-second",
            ordinal=2,
            reading_order=2,
            text="Description",
            bbox={"x": 0.10, "y": 0.10, "width": 0.20, "height": 0.10},
            polygon=None,
            confidence=0.98,
        ),
    )

    assert _header_evidence_matches(spans=spans, aliases=("description of goods",)) == ()


def test_sidecar_rejects_ambiguous_geometry_grid_when_header_ocr_is_missing() -> None:
    result, _native = _known_form_projection(
        document_type="commercial_invoice",
        additional_complete_body_row=True,
        obscure_header_text=True,
    )

    assert result.tables == []
    assert "table_structure_incomplete" in {reason.code for reason in result.needs_review_reasons}


def test_sidecar_rejects_two_complete_known_form_body_rows_as_ambiguous() -> None:
    result, native = _known_form_projection(
        document_type="commercial_invoice",
        additional_complete_body_row=True,
    )
    preview_page = result.pages[0]

    assert result.tables == []
    assert "table_structure_ambiguous" in {reason.code for reason in result.needs_review_reasons}
    assert (
        _table_projection_status(
            document_type="commercial_invoice",
            native_results=[native],
            pages=[preview_page],
        )
        == "ambiguous_body_evidence"
    )


def test_sidecar_rejects_multiple_native_known_form_targets_instead_of_guessing() -> None:
    result, native = _known_form_projection(document_type="commercial_invoice")
    configuration = PpStructureV3SidecarConfigurationV1.model_validate(
        _sidecar_configuration_payload()
    )
    native_tables = cast(list[object], native["table_res_list"])
    ambiguous = project_ppstructurev3_predictions(
        source_sha256="a" * 64,
        raster_profile_revision="synthetic-raster-v1",
        configuration=configuration,
        pages=[_InputPage(page_number=1, width=1_000, height=1_000, pixels=b"\x00" * 3_000_000)],
        native_results=[{**native, "table_res_list": [native_tables[0], native_tables[0]]}],
    )

    assert result.tables
    assert ambiguous.tables == []
    assert "table_structure_ambiguous" in {reason.code for reason in ambiguous.needs_review_reasons}


def test_sidecar_does_not_fill_missing_table_structure_from_ocr_text() -> None:
    configuration = PpStructureV3SidecarConfigurationV1.model_validate(
        _sidecar_configuration_payload()
    )
    result = project_ppstructurev3_predictions(
        source_sha256="a" * 64,
        raster_profile_revision="synthetic-raster-v1",
        configuration=configuration,
        pages=[_InputPage(page_number=1, width=100, height=100, pixels=b"\x00" * 30_000)],
        native_results=[
            {
                "overall_ocr_res": {
                    "rec_texts": ["PACKING LIST"],
                    "rec_scores": [0.99],
                    "rec_polys": [[[5, 5], [90, 5], [90, 15], [5, 15]]],
                },
                "parsing_res_list": [{"block_order": 1, "block_bbox": [0, 0, 100, 100]}],
            }
        ],
    )

    assert result.tables == []
    assert [reason.code for reason in result.needs_review_reasons] == [
        "field_mapping_incomplete",
        "table_structure_incomplete",
    ]


def test_sidecar_classifies_only_an_exact_contiguous_split_supported_title() -> None:
    configuration = PpStructureV3SidecarConfigurationV1.model_validate(
        _sidecar_configuration_payload()
    )
    result = project_ppstructurev3_predictions(
        source_sha256="a" * 64,
        raster_profile_revision="synthetic-raster-v1",
        configuration=configuration,
        pages=[_InputPage(page_number=1, width=1_000, height=1_000, pixels=b"\x00" * 3_000_000)],
        native_results=[
            {
                "overall_ocr_res": {
                    "rec_texts": ["COMMERCIAL", "INVOICE"],
                    "rec_scores": [0.99, 0.98],
                    "rec_polys": [
                        [[100, 30], [410, 30], [410, 60], [100, 60]],
                        [[420, 30], [700, 30], [700, 60], [420, 60]],
                    ],
                },
                "parsing_res_list": [{"block_order": 1, "block_bbox": [0, 0, 1_000, 1_000]}],
                "table_res_list": [],
            }
        ],
    )

    assert result.document_type == "commercial_invoice"


def test_sidecar_rejects_disconnected_or_extra_title_text() -> None:
    configuration = PpStructureV3SidecarConfigurationV1.model_validate(
        _sidecar_configuration_payload()
    )
    result = project_ppstructurev3_predictions(
        source_sha256="a" * 64,
        raster_profile_revision="synthetic-raster-v1",
        configuration=configuration,
        pages=[_InputPage(page_number=1, width=1_000, height=1_000, pixels=b"\x00" * 3_000_000)],
        native_results=[
            {
                "overall_ocr_res": {
                    "rec_texts": ["COMMERCIAL", "INVOICE", "NUMBER"],
                    "rec_scores": [0.99, 0.98, 0.97],
                    "rec_polys": [
                        [[100, 30], [410, 30], [410, 60], [100, 60]],
                        [[800, 30], [980, 30], [980, 60], [800, 60]],
                        [[100, 100], [350, 100], [350, 130], [100, 130]],
                    ],
                },
                "parsing_res_list": [{"block_order": 1, "block_bbox": [0, 0, 1_000, 1_000]}],
                "table_res_list": [],
            }
        ],
    )

    assert result.document_type == "unknown"


def _native_field_projection(
    *, title: str, pairs: list[tuple[str, str]]
) -> PpStructureV3SidecarResultV1:
    texts = [title]
    scores = [0.99]
    polygons = [[[30, 15], [970, 15], [970, 35], [30, 35]]]
    for index, (label, value) in enumerate(pairs, start=1):
        top = 45 + (index * 55)
        texts.extend((label, value))
        scores.extend((0.98, 0.97))
        polygons.extend(
            (
                [[30, top], [430, top], [430, top + 20], [30, top + 20]],
                [[520, top], [950, top], [950, top + 20], [520, top + 20]],
            )
        )
    return project_ppstructurev3_predictions(
        source_sha256="a" * 64,
        raster_profile_revision="synthetic-raster-v1",
        configuration=PpStructureV3SidecarConfigurationV1.model_validate(
            _sidecar_configuration_payload()
        ),
        pages=[_InputPage(page_number=1, width=1_000, height=1_000, pixels=b"\x00" * 3_000_000)],
        native_results=[
            {
                "overall_ocr_res": {
                    "rec_texts": texts,
                    "rec_scores": scores,
                    "rec_polys": polygons,
                },
                "parsing_res_list": [{"block_order": 1, "block_bbox": [0, 0, 1_000, 1_000]}],
                "table_res_list": [],
            }
        ],
    )


def _one_known_form_field_projection(
    *,
    split_label: bool = False,
    include_value: bool = True,
    duplicate_first_slot: bool = False,
    include_later_same_row_span: bool = False,
    include_table_ocr_decoy: bool = False,
) -> PpStructureV3SidecarResultV1:
    """Build only observed candidate OCR/layout inputs for field-map tests."""

    texts = ["COMMERCIAL INVOICE"]
    scores = [0.99]
    polygons = [[[30, 15], [970, 15], [970, 35], [30, 35]]]
    if split_label:
        texts.extend(("Invoice", "Number"))
        scores.extend((0.98, 0.98))
        polygons.extend(
            (
                [[30, 70], [180, 70], [180, 90], [30, 90]],
                [[185, 70], [430, 70], [430, 90], [185, 90]],
            )
        )
    else:
        texts.append("Invoice Number")
        scores.append(0.98)
        polygons.append([[30, 70], [430, 70], [430, 90], [30, 90]])
    if include_value:
        texts.append("VALUE-A")
        scores.append(0.97)
        polygons.append([[520, 70], [680, 70], [680, 90], [520, 90]])
    if duplicate_first_slot:
        texts.append("VALUE-B")
        scores.append(0.96)
        polygons.append([[521, 70], [681, 70], [681, 90], [521, 90]])
    if include_later_same_row_span:
        texts.append("LATER")
        scores.append(0.95)
        polygons.append([[820, 70], [950, 70], [950, 90], [820, 90]])
    table_result: list[object] = []
    if include_table_ocr_decoy:
        table_result = [
            {
                "cell_box_list": [],
                "table_ocr_pred": {
                    "rec_texts": ["NOT-USED"],
                    "rec_scores": [0.99],
                    "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                },
            }
        ]
    return project_ppstructurev3_predictions(
        source_sha256="a" * 64,
        raster_profile_revision="synthetic-raster-v1",
        configuration=PpStructureV3SidecarConfigurationV1.model_validate(
            _sidecar_configuration_payload()
        ),
        pages=[_InputPage(page_number=1, width=1_000, height=1_000, pixels=b"\x00" * 3_000_000)],
        native_results=[
            {
                "overall_ocr_res": {
                    "rec_texts": texts,
                    "rec_scores": scores,
                    "rec_polys": polygons,
                },
                "parsing_res_list": [{"block_order": 1, "block_bbox": [0, 0, 1_000, 1_000]}],
                "table_res_list": table_result,
            }
        ],
    )


def _field_value(result: PpStructureV3SidecarResultV1, field_id: str) -> str | None:
    return next(
        (field.value for field in result.field_candidates if field.field_id == field_id),
        None,
    )


def test_sidecar_known_form_field_mapper_selects_the_first_unique_right_slot() -> None:
    result = _one_known_form_field_projection(include_later_same_row_span=True)

    assert _field_value(result, "inv_no") == "VALUE-A"
    assert "field_mapping_ambiguous" not in {reason.code for reason in result.needs_review_reasons}


def test_sidecar_known_form_field_mapper_accepts_one_exact_split_label_join() -> None:
    result = _one_known_form_field_projection(split_label=True)

    assert _field_value(result, "inv_no") == "VALUE-A"


@pytest.mark.parametrize(
    ("include_value", "duplicate_first_slot", "expected_reason"),
    (
        (False, False, "field_mapping_incomplete"),
        (True, True, "field_mapping_ambiguous"),
    ),
)
def test_sidecar_known_form_field_mapper_rejects_missing_or_ambiguous_slots(
    include_value: bool,
    duplicate_first_slot: bool,
    expected_reason: str,
) -> None:
    result = _one_known_form_field_projection(
        include_value=include_value,
        duplicate_first_slot=duplicate_first_slot,
    )

    assert _field_value(result, "inv_no") is None
    assert expected_reason in {reason.code for reason in result.needs_review_reasons}


def test_sidecar_known_form_field_mapper_never_uses_table_ocr_text() -> None:
    without_table_ocr = _one_known_form_field_projection()
    with_table_ocr = _one_known_form_field_projection(include_table_ocr_decoy=True)

    assert _field_value(without_table_ocr, "inv_no") == _field_value(with_table_ocr, "inv_no")


def test_sidecar_retains_the_legacy_generic_field_mapper_when_no_known_slot_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.service.ppstructurev3_sidecar as sidecar

    monkeypatch.setattr(sidecar, "_known_form_field_slots", lambda _document_type: ())
    result = _native_field_projection(
        title="COMMERCIAL INVOICE",
        pairs=[("Invoice Number", "VALUE-A")],
    )

    assert _field_value(result, "inv_no") == "VALUE-A"


@pytest.mark.parametrize(
    ("document_type", "title", "pairs", "expected_values"),
    [
        (
            "bill_of_lading",
            "BILL OF LADING",
            [
                ("Shipper", "ACME EXPORT"),
                ("To Applicant / Consignee", "BUYER INC"),
                ("Port of Loading", "BUSAN"),
                ("Port of Discharge", "LOS ANGELES"),
                ("Final Destination", "LONG BEACH"),
                ("Vessel / Flight", "V-1"),
                ("Voyage No", "001"),
                ("B/L No", "BL-001"),
                ("Container No", "CONT-001"),
                ("Seal No", "SEAL-001"),
                ("Goods Description", "WIDGETS"),
                ("Gross Weight", "100 KG"),
                ("No of Containers", "1"),
            ],
            {},
        ),
        (
            "commercial_invoice",
            "COMMERCIAL INVOICE",
            [
                ("Exporter", "ACME EXPORT"),
                ("Importer", "BUYER INC"),
                ("Consignee", "BUYER INC"),
                ("Port of Loading", "BUSAN"),
                ("Port of Discharge", "LOS ANGELES"),
                ("Vessel / Flight", "V-1"),
                ("Invoice Number", "INV-001"),
                ("Invoice Date", "2026-08-01"),
                ("L/C No", "LC-001"),
                ("Goods Description", "WIDGETS"),
                ("Quantity", "2 PCS"),
                ("Unit Price", "10.00"),
                ("Amount", "20.00"),
                ("Total", "20.00"),
            ],
            {"qty": "2", "unit": "PCS"},
        ),
        (
            "packing_list",
            "PACKING LIST",
            [
                ("Exporter", "ACME EXPORT"),
                ("Consignee", "BUYER INC"),
                ("No. of Invoice", "INV-001"),
                ("No. of L / C", "LC-001"),
                ("Goods Description", "WIDGETS"),
                ("Unit", "PCS"),
                ("Quantity", "2"),
                ("No of Cartons / PKGS", "3 CTNS"),
                ("Net Weight", "10 KG"),
                ("Gross Weight", "12 KG"),
                ("CBM", "0.20"),
            ],
            {"net_gross_weight": "10 KG / 12 KG"},
        ),
    ],
)
def test_sidecar_field_projector_uses_the_exact_source_attested_field_vocabulary(
    document_type: Literal["bill_of_lading", "commercial_invoice", "packing_list"],
    title: str,
    pairs: list[tuple[str, str]],
    expected_values: dict[str, str],
) -> None:
    result = _native_field_projection(title=title, pairs=pairs)
    fields = {field.field_id: field.value for field in result.field_candidates}

    assert tuple(fields) == tuple(rule.field_id for rule in _FIELD_RULES[document_type])
    assert all(fields[field_id] == value for field_id, value in expected_values.items())
    assert [reason.code for reason in result.needs_review_reasons] == ["table_structure_incomplete"]


def _automatic_v2_pointer_raw(revision: str) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": "2.0",
                "activation_kind": "source_attested_synthetic_raster",
                "status": "activated",
                "dataset_id": "synthetic-raster-v1",
                "raster_profile_revision": revision,
                "runtime_configuration_path": "runtime/profile-v2.json",
                "runtime_configuration_sha256": "a" * 64,
                "sealed_evaluation_path": "evaluations/sealed-v2.json",
                "sealed_evaluation_sha256": "b" * 64,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _automatic_v2_profile(revision: str) -> PpStructureV3OfflineProfileV2:
    return PpStructureV3OfflineProfileV2(
        extractor_revision="ppstructurev3-local-v2",
        raster_profile_revision=revision,
        pdfium_version="4.30.0",
        raster_dpi=216,
        parent_renderer_package_inventory_sha256="c" * 64,
        parent_renderer_package_inventory_path="runtime/parent-renderer-packages.json",
        parent_renderer_required_packages=["pypdfium2", "Pillow"],
        sidecar_package_inventory_sha256="d" * 64,
        sidecar_package_inventory_path="runtime/sidecar-packages.json",
        sidecar_required_packages=["paddleocr", "paddlex", "paddlepaddle"],
        model_inventory_sha256="e" * 64,
        model_inventory_path="runtime/models.json",
        local_model_root="models",
        sidecar_interpreter_root="raster-offline-bundle-v1/sidecar/interpreter",
        sidecar_interpreter_root_tree_sha256="9" * 64,
        sidecar_interpreter_path="raster-offline-bundle-v1/sidecar/interpreter/python.exe",
        sidecar_interpreter_sha256="f" * 64,
        sidecar_module_root="raster-offline-bundle-v1/sidecar/module-root",
        sidecar_module_root_tree_sha256="0" * 64,
        sidecar_command_path=(
            "raster-offline-bundle-v1/sidecar/module-root/customs_ai_agent/service/"
            "ppstructurev3_sidecar.py"
        ),
        sidecar_command_sha256="1" * 64,
        sidecar_module_path=(
            "raster-offline-bundle-v1/sidecar/module-root/customs_ai_agent/service/"
            "ppstructurev3_sidecar.py"
        ),
        sidecar_module_sha256="1" * 64,
        sidecar_configuration_path="runtime/ppstructurev3-sidecar.json",
        sidecar_configuration_sha256="2" * 64,
    )


def test_v2_profile_splits_parent_renderer_from_sidecar_packages() -> None:
    revision = make_synthetic_raster_profile_revision("a" * 64)
    profile = _automatic_v2_profile(revision)

    assert profile.parent_renderer_required_packages == ["pypdfium2", "Pillow"]
    assert "paddleocr" not in profile.parent_renderer_required_packages
    assert profile.sidecar_required_packages == ["paddleocr", "paddlex", "paddlepaddle"]
    invalid = profile.model_dump(mode="json")
    invalid["parent_renderer_required_packages"] = ["pypdfium2", "paddleocr"]
    with pytest.raises(ValidationError, match="package inventories"):
        PpStructureV3OfflineProfileV2.model_validate(invalid)
    invalid = profile.model_dump(mode="json")
    invalid["sidecar_required_packages"] = ["paddleocr", "paddlex", "opencv-python"]
    with pytest.raises(ValidationError, match="package inventories"):
        PpStructureV3OfflineProfileV2.model_validate(invalid)
    invalid = profile.model_dump(mode="json")
    invalid["sidecar_interpreter_path"] = "bin/python"
    with pytest.raises(ValidationError, match="sidecar module root"):
        PpStructureV3OfflineProfileV2.model_validate(invalid)
    invalid = profile.model_dump(mode="json")
    invalid["sidecar_module_path"] = "raster-offline-bundle-v1/sidecar/not-under-root.py"
    with pytest.raises(ValidationError, match="sidecar module root"):
        PpStructureV3OfflineProfileV2.model_validate(invalid)
    invalid = profile.model_dump(mode="json")
    invalid["sidecar_command_path"] = "raster-offline-bundle-v1/sidecar/module-root/entry.py"
    with pytest.raises(ValidationError, match="sidecar module root"):
        PpStructureV3OfflineProfileV2.model_validate(invalid)


def test_new_release_profile_and_pointer_bind_path_basename_to_manifest_sha() -> None:
    revision = make_synthetic_raster_profile_revision("a" * 64)
    release_sha256 = "a" * 64
    release_root = f"raster-offline-bundles-v1/releases/{release_sha256}"
    profile_payload = _automatic_v2_profile(revision).model_dump(mode="json")
    profile_payload.update(
        {
            "bundle_root_relative_path": release_root,
            "bundle_manifest_sha256": release_sha256,
            "parent_renderer_package_inventory_path": f"{release_root}/inventories/parent.json",
            "sidecar_package_inventory_path": f"{release_root}/inventories/sidecar.json",
            "model_inventory_path": f"{release_root}/inventories/models.json",
            "local_model_root": f"{release_root}/models",
            "sidecar_interpreter_root": f"{release_root}/sidecar/interpreter",
            "sidecar_interpreter_path": f"{release_root}/sidecar/interpreter/python.exe",
            "sidecar_module_root": f"{release_root}/sidecar/module-root",
            "sidecar_command_path": (
                f"{release_root}/sidecar/module-root/customs_ai_agent/service/"
                "ppstructurev3_sidecar.py"
            ),
            "sidecar_module_path": (
                f"{release_root}/sidecar/module-root/customs_ai_agent/service/"
                "ppstructurev3_sidecar.py"
            ),
        }
    )
    profile = PpStructureV3OfflineProfileV2.model_validate(profile_payload)
    assert profile.bundle_root_relative_path == release_root

    invalid_profile = dict(profile_payload)
    invalid_profile["bundle_manifest_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="bundle binding"):
        PpStructureV3OfflineProfileV2.model_validate(invalid_profile)

    pointer_payload = {
        "raster_profile_revision": revision,
        "bundle_root_relative_path": release_root,
        "bundle_manifest_sha256": release_sha256,
        "runtime_configuration_path": "raster-runtime-profiles-v1/a/profile.json",
        "runtime_configuration_sha256": "c" * 64,
        "sealed_evaluation_path": "synthetic-raster-v1/evaluations/sealed.json",
        "sealed_evaluation_sha256": "d" * 64,
    }
    pointer = RasterSourceAttestedActivationPointerV2.model_validate(pointer_payload)
    assert pointer.bundle_manifest_sha256 == release_sha256

    invalid_pointer = dict(pointer_payload)
    invalid_pointer["bundle_root_relative_path"] = "raster-offline-bundle-v1"
    with pytest.raises(ValidationError, match="bundle binding"):
        RasterSourceAttestedActivationPointerV2.model_validate(invalid_pointer)

    # Existing v2 records had neither field; keep this narrow compatibility
    # path so the new invariant does not invalidate an already-active legacy
    # pointer/profile just because the software was upgraded.
    legacy_pointer = dict(pointer_payload)
    legacy_pointer.pop("bundle_root_relative_path")
    legacy_pointer.pop("bundle_manifest_sha256")
    assert (
        RasterSourceAttestedActivationPointerV2.model_validate(
            legacy_pointer
        ).bundle_root_relative_path
        is None
    )


def test_v2_loader_allows_an_exhaustive_bundle_manifest_larger_than_sidecar_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release manifest has its own bounded input cap.

    A real isolated Paddle interpreter has tens of thousands of declared
    files, so its exhaustive manifest can legitimately exceed the 1 MiB cap
    reserved for sidecar configuration and runtime-profile payloads.  This
    isolates that wider cap to the manifest binding; it does not relax the
    sidecar input boundary.
    """

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    manifest_raw = b"m" * (raster_runtime._MAX_SIDECAR_BYTES + 1)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    release_root = f"raster-offline-bundles-v1/releases/{manifest_sha256}"
    revision = make_synthetic_raster_profile_revision(manifest_sha256)
    payload = _automatic_v2_profile(revision).model_dump(mode="json")
    command_path = (
        f"{release_root}/sidecar/module-root/customs_ai_agent/service/ppstructurev3_sidecar.py"
    )
    payload.update(
        {
            "bundle_root_relative_path": release_root,
            "bundle_manifest_sha256": manifest_sha256,
            "parent_renderer_package_inventory_path": f"{release_root}/inventories/parent.json",
            "sidecar_package_inventory_path": f"{release_root}/inventories/sidecar.json",
            "model_inventory_path": f"{release_root}/inventories/models.json",
            "local_model_root": f"{release_root}/models",
            "sidecar_interpreter_root": f"{release_root}/sidecar/interpreter",
            "sidecar_interpreter_path": f"{release_root}/sidecar/interpreter/python.exe",
            "sidecar_module_root": f"{release_root}/sidecar/module-root",
            "sidecar_command_path": command_path,
            "sidecar_module_path": command_path,
            "parent_renderer_package_inventory_sha256": "a" * 64,
            "sidecar_package_inventory_sha256": "b" * 64,
            "model_inventory_sha256": "c" * 64,
            "sidecar_interpreter_root_tree_sha256": "d" * 64,
            "sidecar_interpreter_sha256": "0" * 64,
            "sidecar_module_root_tree_sha256": "f" * 64,
            "sidecar_command_sha256": "0" * 64,
            "sidecar_module_sha256": "0" * 64,
            "sidecar_configuration_path": "runtime/sidecar.json",
        }
    )
    sidecar_configuration = _sidecar_configuration_payload()
    sidecar_configuration["extractor_revision"] = payload["extractor_revision"]
    sidecar_configuration["raster_profile_revision"] = revision
    sidecar_configuration_raw = _write_json(
        root / "runtime" / "sidecar.json", sidecar_configuration
    )
    payload["sidecar_configuration_sha256"] = hashlib.sha256(sidecar_configuration_raw).hexdigest()
    profile_raw = _write_json(root / "runtime" / "profile.json", payload)
    (root / release_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / release_root / "manifest.json").write_bytes(manifest_raw)
    (root / release_root / "models").mkdir(parents=True)
    for relative in (
        f"{release_root}/inventories/parent.json",
        f"{release_root}/inventories/sidecar.json",
        f"{release_root}/inventories/models.json",
    ):
        _write_json(root / relative, {"fixture": True})

    original_read = raster_runtime._read_local_file
    manifest_limits: list[int] = []

    def recording_read(path: Path, *, limit: int) -> bytes:
        if path.name == "manifest.json":
            manifest_limits.append(limit)
        return original_read(path, limit=limit)

    monkeypatch.setattr(raster_runtime, "_read_local_file", recording_read)
    # The test intentionally uses a 64-character release directory below
    # pytest's deep temporary path.  Its manifest-cap assertion is unrelated
    # to the Windows 8.3 operational-path fallback.
    monkeypatch.setattr(raster_runtime, "_windows_operational_sidecar_path", lambda path: path)
    monkeypatch.setattr(raster_runtime, "_verify_package_inventory", lambda **_kwargs: None)
    monkeypatch.setattr(raster_runtime, "_verify_model_inventory", lambda **_kwargs: None)
    monkeypatch.setattr(raster_runtime, "_validate_controlled_child_file", lambda **_kwargs: None)
    monkeypatch.setattr(raster_runtime, "_controlled_directory_tree_sha256", lambda _path: "d" * 64)
    monkeypatch.setattr(
        raster_runtime, "_controlled_module_root_tree_sha256", lambda _path: "f" * 64
    )
    monkeypatch.setattr(raster_runtime, "_sha256_regular_file", lambda *_args, **_kwargs: "0" * 64)
    monkeypatch.setattr(PpStructureV3CommandExecutorV2, "verify_runtime_state", lambda _self: None)
    monkeypatch.setattr(
        PpStructureV3SidecarConfigurationV1,
        "validate_model_directories",
        lambda _self, _model_root: None,
    )

    materials = raster_runtime._load_ppstructurev3_v2_runtime_materials(
        root=root,
        configuration_path=root / "runtime" / "profile.json",
        configuration_raw=profile_raw,
        expected_sha256=hashlib.sha256(profile_raw).hexdigest(),
    )

    assert materials.profile.bundle_manifest_sha256 == manifest_sha256
    assert manifest_limits == [raster_runtime._MAX_OFFLINE_BUNDLE_MANIFEST_BYTES]
    assert raster_runtime._MAX_OFFLINE_BUNDLE_MANIFEST_BYTES > raster_runtime._MAX_SIDECAR_BYTES


def _isolated_v2_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[PpStructureV3CommandExecutorV2, Path, Path]:
    """Build a minimal fully hash-bound executor without native Paddle imports."""

    root = tmp_path / "bundle"
    interpreter = root / "interpreter" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"pinned interpreter")
    module_root = root / "module-root"
    module_path = module_root / "customs_ai_agent" / "service" / "ppstructurev3_sidecar.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(b"pinned module")
    command_path = module_path
    configuration_raw = _write_json(root / "runtime" / "sidecar.json", {"pinned": True})
    parent_inventory_raw = _write_json(
        root / "runtime" / "parent.json",
        {
            "packages": [
                {"name": "pypdfium2", "version": "test-1.0"},
                {"name": "Pillow", "version": "test-1.0"},
            ]
        },
    )
    sidecar_inventory_raw = _write_json(
        root / "runtime" / "sidecar.json.packages",
        {
            "packages": [
                {"name": "paddleocr", "version": "test-1.0"},
                {"name": "paddlex", "version": "test-1.0"},
                {"name": "paddlepaddle", "version": "test-1.0"},
            ]
        },
    )
    model = root / "models" / "layout" / "weights.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"pinned model")
    model_inventory_raw = _write_json(
        root / "runtime" / "models.json",
        {
            "artifacts": [
                {
                    "relative_path": "layout/weights.bin",
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    monkeypatch.setattr(
        raster_runtime.importlib.metadata,
        "version",
        lambda _name: "test-1.0",
    )
    return (
        PpStructureV3CommandExecutorV2(
            interpreter_path=interpreter,
            interpreter_sha256=hashlib.sha256(interpreter.read_bytes()).hexdigest(),
            interpreter_root=interpreter.parent,
            interpreter_root_tree_sha256=_controlled_module_root_tree_sha256(interpreter.parent),
            timeout_seconds=420,
            configuration_path=root / "runtime" / "sidecar.json",
            configuration_sha256=hashlib.sha256(configuration_raw).hexdigest(),
            model_root=root / "models",
            parent_renderer_package_inventory_path=root / "runtime" / "parent.json",
            parent_renderer_package_inventory_sha256=hashlib.sha256(
                parent_inventory_raw
            ).hexdigest(),
            parent_renderer_required_packages=("pypdfium2", "Pillow"),
            sidecar_package_inventory_path=root / "runtime" / "sidecar.json.packages",
            sidecar_package_inventory_sha256=hashlib.sha256(sidecar_inventory_raw).hexdigest(),
            sidecar_required_packages=("paddleocr", "paddlex", "paddlepaddle"),
            model_inventory_path=root / "runtime" / "models.json",
            model_inventory_sha256=hashlib.sha256(model_inventory_raw).hexdigest(),
            sidecar_module_root=module_root,
            sidecar_module_root_tree_sha256=_controlled_module_root_tree_sha256(module_root),
            sidecar_command_path=command_path,
            sidecar_command_sha256=hashlib.sha256(command_path.read_bytes()).hexdigest(),
            sidecar_module_path=module_path,
            sidecar_module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        ),
        interpreter,
        module_root,
    )


def test_v2_executor_rejects_an_injected_module_root_package_before_sidecar_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, _interpreter, module_root = _isolated_v2_executor(tmp_path, monkeypatch)
    executor.verify_runtime_state()

    injected = module_root / "paddleocr" / "__init__.py"
    injected.parent.mkdir()
    injected.write_bytes(b"shadow package")

    with pytest.raises(RasterRuntimeError, match="native_sidecar_drift"):
        executor.verify_runtime_state()


def test_v2_executor_rejects_interpreter_tree_and_source_module_drift_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, interpreter, _module_root = _isolated_v2_executor(tmp_path, monkeypatch)
    executor.verify_runtime_state()

    # A profile cannot introduce a wrapper with a different binding: the
    # command and the source module must be the same launch artifact.
    with pytest.raises(RasterRuntimeError, match="runtime_sidecar_invalid"):
        replace(executor, sidecar_command_sha256="0" * 64).verify_runtime_state()

    executor.sidecar_module_path.write_bytes(b"changed source module")
    with pytest.raises(RasterRuntimeError, match="native_sidecar_drift"):
        executor.verify_runtime_state()

    # Use a fresh bound executor so the next assertion proves interpreter-tree
    # drift rather than merely reobserving the changed source-module tree.
    interpreter_executor, interpreter, _module_root = _isolated_v2_executor(
        tmp_path / "interpreter-tree", monkeypatch
    )
    interpreter_executor.verify_runtime_state()
    # A DLL/stdlib file supplied beside the exact interpreter is part of the
    # interpreter-root digest, not an untracked executable dependency.
    (interpreter.parent / "stdlib-injected.py").write_bytes(b"interpreter shadow")
    with pytest.raises(RasterRuntimeError, match="native_sidecar_drift"):
        interpreter_executor.verify_runtime_state()


def test_v2_executor_uses_the_pinned_interpreter_and_never_inherits_python_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, interpreter, module_root = _isolated_v2_executor(tmp_path, monkeypatch)
    revision = make_synthetic_raster_profile_revision("a" * 64)
    captured: dict[str, object] = {}
    monkeypatch.setenv("PATH", "outside-path")
    monkeypatch.setenv("PYTHONPATH", "outside-pythonpath")
    monkeypatch.setenv("USERPROFILE", "outside-user-profile")
    monkeypatch.setenv("HOME", "outside-home")
    monkeypatch.setattr(PpStructureV3CommandExecutorV2, "verify_runtime_state", lambda _self: None)

    def fake_run(arguments: list[str], **kwargs: object) -> SimpleNamespace:
        captured["arguments"] = arguments
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=1, stdout=b"")

    monkeypatch.setattr(raster_runtime.subprocess, "run", fake_run)
    with pytest.raises(RasterRuntimeError, match="native_sidecar_unavailable"):
        executor.extract_layout(
            [RasterizedPageV2(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")],
            source_sha256="a" * 64,
            raster_profile_revision=revision,
            profile=_automatic_v2_profile(revision),
        )

    arguments = cast(list[str], captured["arguments"])
    environment = cast(dict[str, str], captured["environment"])
    assert arguments[0] == str(interpreter)
    assert arguments[1:4] == ["-I", "-B", "-c"]
    assert arguments[5:8] == [
        str(interpreter.parent),
        str(module_root),
        str(executor.sidecar_command_path),
    ]
    assert executor.sidecar_command_path == executor.sidecar_module_path
    assert arguments[-1] == "--customs-raster-ppstructurev3-v1"
    assert "PYTHONPATH" not in environment
    assert "outside-path" not in environment["PATH"]
    assert environment["PATH"] == os.pathsep.join((str(interpreter.parent), str(module_root)))
    # ModelScope calls Path.home() while PP-Structure starts.  The sidecar
    # gets a disposable child-owned home/cache, never the real user profile
    # and never the immutable bundle/module root.
    assert environment["USERPROFILE"] != "outside-user-profile"
    assert environment["HOME"] == environment["USERPROFILE"]
    assert environment["HOMEDRIVE"] + environment["HOMEPATH"] == environment["HOME"]
    assert environment["MODELSCOPE_CACHE"].startswith(environment["HOME"])
    assert environment["PADDLE_HOME"].startswith(environment["HOME"])
    assert environment["TEMP"].startswith(environment["HOME"])


def test_v2_executor_structure_probe_uses_the_same_pinned_sidecar_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diagnostic changes only the sidecar mode, not its trust boundary."""

    executor, interpreter, module_root = _isolated_v2_executor(tmp_path, monkeypatch)
    revision = make_synthetic_raster_profile_revision("a" * 64)
    captured: dict[str, object] = {}
    monkeypatch.setenv("PATH", "outside-path")
    monkeypatch.setenv("PYTHONPATH", "outside-pythonpath")
    monkeypatch.setattr(PpStructureV3CommandExecutorV2, "verify_runtime_state", lambda _self: None)

    def fake_run(arguments: list[str], **kwargs: object) -> SimpleNamespace:
        captured["arguments"] = arguments
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=PpStructureV3StructureProbeResultV1.model_validate(
                {
                    "table_projection_status": "accepted",
                    "pages": [
                        {
                            "page_number": 1,
                            "native_output_mapping": True,
                            "has_overall_ocr_result": True,
                            "overall_ocr_span_count": 1,
                            "has_layout_detection_result": True,
                            "layout_box_count": 1,
                            "layout_table_box_count": 1,
                            "has_parsing_result_list": True,
                            "parsing_block_count": 1,
                            "native_table_list_valid": True,
                            "native_table_count": 1,
                            "native_table_mapping_count": 1,
                            "table_cell_box_list_present": [True],
                            "table_cell_counts": [1],
                            "table_ocr_prediction_present": [True],
                            "table_ocr_span_counts": [1],
                            "native_table_probe_truncated": False,
                        }
                    ],
                }
            )
            .model_dump_json()
            .encode("utf-8"),
        )

    monkeypatch.setattr(raster_runtime.subprocess, "run", fake_run)
    result = executor.probe_native_structure(
        [RasterizedPageV2(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")],
        source_sha256="a" * 64,
        raster_profile_revision=revision,
        profile=_automatic_v2_profile(revision),
    )

    arguments = cast(list[str], captured["arguments"])
    environment = cast(dict[str, str], captured["environment"])
    assert result.pages[0].layout_table_box_count == 1
    assert arguments[0] == str(interpreter)
    assert arguments[1:4] == ["-I", "-B", "-c"]
    assert arguments[5:8] == [
        str(interpreter.parent),
        str(module_root),
        str(executor.sidecar_command_path),
    ]
    assert arguments[-1] == "--customs-raster-ppstructurev3-structure-probe-v1"
    assert "PYTHONPATH" not in environment
    assert "outside-path" not in environment["PATH"]

    with pytest.raises(RasterRuntimeError, match="native_sidecar_binding_invalid"):
        executor.probe_native_structure(
            [],
            source_sha256="a" * 64,
            raster_profile_revision=revision,
            profile=_automatic_v2_profile(revision),
        )

    wrong_page = result.model_copy(
        update={"pages": [result.pages[0].model_copy(update={"page_number": 2})]}
    )
    monkeypatch.setattr(
        raster_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=wrong_page.model_dump_json().encode("utf-8"),
        ),
    )
    with pytest.raises(RasterRuntimeError, match="native_sidecar_binding_invalid"):
        executor.probe_native_structure(
            [RasterizedPageV2(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")],
            source_sha256="a" * 64,
            raster_profile_revision=revision,
            profile=_automatic_v2_profile(revision),
        )


def test_v2_executor_structure_probe_rejects_page_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, _interpreter, _module_root = _isolated_v2_executor(tmp_path, monkeypatch)
    revision = make_synthetic_raster_profile_revision("a" * 64)
    monkeypatch.setattr(PpStructureV3CommandExecutorV2, "verify_runtime_state", lambda _self: None)
    monkeypatch.setattr(
        raster_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=PpStructureV3StructureProbeResultV1.model_validate(
                {
                    "table_projection_status": "not_attempted_unknown_document",
                    "pages": [
                        {
                            "page_number": 1,
                            "native_output_mapping": False,
                            "has_overall_ocr_result": False,
                            "overall_ocr_span_count": 0,
                            "has_layout_detection_result": False,
                            "layout_box_count": 0,
                            "layout_table_box_count": 0,
                            "has_parsing_result_list": False,
                            "parsing_block_count": 0,
                            "native_table_list_valid": False,
                            "native_table_count": 0,
                            "native_table_mapping_count": 0,
                            "table_cell_box_list_present": [],
                            "table_cell_counts": [],
                            "table_ocr_prediction_present": [],
                            "table_ocr_span_counts": [],
                            "native_table_probe_truncated": False,
                        },
                        {
                            "page_number": 2,
                            "native_output_mapping": False,
                            "has_overall_ocr_result": False,
                            "overall_ocr_span_count": 0,
                            "has_layout_detection_result": False,
                            "layout_box_count": 0,
                            "layout_table_box_count": 0,
                            "has_parsing_result_list": False,
                            "parsing_block_count": 0,
                            "native_table_list_valid": False,
                            "native_table_count": 0,
                            "native_table_mapping_count": 0,
                            "table_cell_box_list_present": [],
                            "table_cell_counts": [],
                            "table_ocr_prediction_present": [],
                            "table_ocr_span_counts": [],
                            "native_table_probe_truncated": False,
                        },
                    ],
                }
            )
            .model_dump_json()
            .encode("utf-8"),
        ),
    )

    with pytest.raises(RasterRuntimeError, match="native_sidecar_binding_invalid"):
        executor.probe_native_structure(
            [RasterizedPageV2(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")],
            source_sha256="a" * 64,
            raster_profile_revision=revision,
            profile=_automatic_v2_profile(revision),
        )


def test_auto_runtime_clears_old_binding_for_missing_invalid_and_replaced_pointer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    pointer_path = root / "approvals" / "raster-preview-active.json"
    pointer_path.parent.mkdir(parents=True)
    first_revision = make_synthetic_raster_profile_revision("a" * 64)
    second_revision = make_synthetic_raster_profile_revision("b" * 64)

    class Executor:
        calls = 0

        def verify_runtime_state(self) -> None:
            self.calls += 1

    class Engine:
        calls = 0

        def extract(self, _raster_input: object, _structure: object) -> str:
            self.calls += 1
            return "first-engine" if self.calls == 1 else "second-engine"

    class Renderer:
        def render_pdf_page(
            self, _raster_input: object, _structure: object, *, page_number: int
        ) -> str:
            return f"page-{page_number}"

    executor = Executor()
    engine = Engine()
    renderer = Renderer()
    loader_calls: list[str] = []

    def loader(**kwargs: object) -> SourceAttestedRasterPreviewBindingV2:
        raw = cast(bytes, kwargs["approval_raw"])
        pointer = parse_source_attested_synthetic_raster_activation(raw)
        loader_calls.append(pointer.raster_profile_revision)
        return SourceAttestedRasterPreviewBindingV2(
            root=root,
            approval_path=pointer_path,
            approval_sha256="0" * 64,
            pointer=pointer,
            profile=_automatic_v2_profile(pointer.raster_profile_revision),
            calibration=cast(RasterCalibrationThresholdReceiptV2, object()),
            evaluation=cast(RasterSealedEvaluationResultV2, object()),
            engine=cast(object, engine),
            page_renderer=cast(ApprovedPdfiumRasterPageRenderer, renderer),
            executor=cast(PpStructureV3CommandExecutorV2, executor),
        )

    runtime = AutoActivatingSourceAttestedRasterPreviewRuntime(
        approval_path=pointer_path,
        binding_loader=loader,
    )

    assert runtime.status().availability == "unavailable"
    pointer_path.write_bytes(_automatic_v2_pointer_raw(first_revision))
    assert runtime.status().raster_profile_revision == first_revision
    assert loader_calls == [first_revision]

    pointer_path.write_bytes(b"{}\n")
    assert runtime.status().availability == "unavailable"
    # A valid replacement must create a fresh binding rather than retain the
    # earlier one that was cleared when the malformed pointer appeared.
    pointer_path.write_bytes(_automatic_v2_pointer_raw(second_revision))
    assert runtime.status().raster_profile_revision == second_revision
    assert loader_calls == [first_revision, second_revision]
    assert executor.calls >= 2


def test_auto_runtime_rejects_pointer_changed_during_binding_load(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    pointer_path = root / "approvals" / "raster-preview-active.json"
    pointer_path.parent.mkdir(parents=True)
    first_revision = make_synthetic_raster_profile_revision("a" * 64)
    second_revision = make_synthetic_raster_profile_revision("b" * 64)
    pointer_path.write_bytes(_automatic_v2_pointer_raw(first_revision))
    changed = False

    class Executor:
        def verify_runtime_state(self) -> None:
            return None

    def loader(**kwargs: object) -> SourceAttestedRasterPreviewBindingV2:
        nonlocal changed
        pointer = parse_source_attested_synthetic_raster_activation(
            cast(bytes, kwargs["approval_raw"])
        )
        if not changed:
            changed = True
            pointer_path.write_bytes(_automatic_v2_pointer_raw(second_revision))
        # The facade rejects the changed second snapshot before this dummy
        # binding could become active, so its contents are immaterial.
        return SourceAttestedRasterPreviewBindingV2(
            root=root,
            approval_path=pointer_path,
            approval_sha256="0" * 64,
            pointer=pointer,
            profile=_automatic_v2_profile(pointer.raster_profile_revision),
            calibration=cast(RasterCalibrationThresholdReceiptV2, object()),
            evaluation=cast(RasterSealedEvaluationResultV2, object()),
            engine=cast(object, object()),
            page_renderer=cast(ApprovedPdfiumRasterPageRenderer, object()),
            executor=cast(PpStructureV3CommandExecutorV2, Executor()),
        )

    runtime = AutoActivatingSourceAttestedRasterPreviewRuntime(
        approval_path=pointer_path,
        binding_loader=loader,
    )

    assert runtime.status().availability == "unavailable"
    assert runtime.status().raster_profile_revision == second_revision


def test_auto_runtime_rejects_pointer_changed_during_cached_drift_check(tmp_path: Path) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    pointer_path = root / "approvals" / "raster-preview-active.json"
    pointer_path.parent.mkdir(parents=True)
    first_revision = make_synthetic_raster_profile_revision("a" * 64)
    second_revision = make_synthetic_raster_profile_revision("b" * 64)
    pointer_path.write_bytes(_automatic_v2_pointer_raw(first_revision))

    class Executor:
        calls = 0

        def verify_runtime_state(self) -> None:
            self.calls += 1
            # The first check belongs to initial activation.  Mutate the
            # pointer during the later cached-binding check to exercise the
            # otherwise easy-to-miss verification race.
            if self.calls == 2:
                pointer_path.write_bytes(_automatic_v2_pointer_raw(second_revision))

    executor = Executor()

    def loader(**kwargs: object) -> SourceAttestedRasterPreviewBindingV2:
        pointer = parse_source_attested_synthetic_raster_activation(
            cast(bytes, kwargs["approval_raw"])
        )
        return SourceAttestedRasterPreviewBindingV2(
            root=root,
            approval_path=pointer_path,
            approval_sha256=hashlib.sha256(cast(bytes, kwargs["approval_raw"])).hexdigest(),
            pointer=pointer,
            profile=_automatic_v2_profile(pointer.raster_profile_revision),
            calibration=cast(RasterCalibrationThresholdReceiptV2, object()),
            evaluation=cast(RasterSealedEvaluationResultV2, object()),
            engine=cast(object, object()),
            page_renderer=cast(ApprovedPdfiumRasterPageRenderer, object()),
            executor=cast(PpStructureV3CommandExecutorV2, executor),
        )

    runtime = AutoActivatingSourceAttestedRasterPreviewRuntime(
        approval_path=pointer_path,
        binding_loader=loader,
    )

    assert runtime.status().raster_profile_revision == first_revision
    assert runtime.status().availability == "unavailable"
    # The old binding was discarded rather than silently returning it after
    # a replacement raced its drift validation.  A later call can load only
    # the current pointer as a new immutable binding.
    assert runtime.status().raster_profile_revision == second_revision
