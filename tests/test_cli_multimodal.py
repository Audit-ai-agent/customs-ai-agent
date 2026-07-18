from __future__ import annotations

import json
import sys
import types
from io import BytesIO
from datetime import date
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image
from pydantic import ValidationError
from reportlab.pdfgen import canvas

import customs_ai_agent.adapters.images as image_adapters
from customs_ai_agent.adapters import image_page_from_file
from customs_ai_agent.adapters.images import (
    _MAX_PDF_WORKER_RESPONSE_BYTES,
    _invoke_pdf_worker,
    _parse_pdf_worker_response,
    _pdf_worker_command,
    _pdf_worker_cwd,
    _pdf_worker_environment,
)
from customs_ai_agent.cli import (
    ContractValidationError,
    LocalReviewInputError,
    _Pdfium2Rasterizer,
    _agent_contract_schema,
    _cmd_export_schema,
    _cmd_review,
    _load_manifest,
    _prepare_manifest,
    _serialize_public_result,
    _validate_contract_payload,
    build_parser,
    main,
)
from customs_ai_agent.ocr_release import (
    OcrReleaseGateAttestation,
    OcrReleaseGateAttestationError,
    ReleaseLockedOcrProvider,
)
from customs_ai_agent.ocr_identity import (
    OCR_MODEL_ASSETS_LOCK_SHA256,
    OCR_MODEL_SET_IDS,
)
from customs_ai_agent.runner import run_review
from customs_ai_agent.adapters import PdfRasterizerError
from customs_ai_agent import pdf_worker
from customs_ai_agent.schemas.models import (
    BBox,
    ImagePageInput,
    OcrLayoutLedger,
    OcrLayoutPage,
    OcrLayoutSpan,
    ReviewDocumentResultV2,
    ReviewRunRequest,
    ReviewRunRequestV2,
    stable_ocr_ledger_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPOSITORY_ROOT / "contracts" / "agent"


def _clear_evidence_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "EVIDENCE_SERVICE_BASE_URL",
        "EVIDENCE_SERVICE_TOKEN",
        "EVIDENCE_SERVICE_TOKEN_HEADER",
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_manifest(
    directory: Path,
    *,
    contract_version: str,
    llm_mode: str,
    documents: list[dict[str, object]],
) -> Path:
    manifest = directory / "review-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "customs-agent-local-review-manifest/v1",
                "contract_version": contract_version,
                "llm_mode": llm_mode,
                "case_id": "case-local-cli",
                "review_run_id": "run-local-cli",
                "review_as_of": "2026-07-10",
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _minimal_completed_state(
    request: ReviewRunRequest | ReviewRunRequestV2,
) -> dict[str, object]:
    case_id = request.case_id or "case-manual"
    return {
        "case_id": case_id,
        "review_run_id": request.review_run_id,
        "review_as_of": request.review_as_of.isoformat(),
        "trade_type": request.trade_type,
        "jurisdiction": request.jurisdiction,
        "final_status": "needs_review",
        "requires_human_review": True,
        # This deliberately minimal state has no Evidence request or item.
        # V2 therefore requires a failed/unknown, human-review-only result;
        # claiming an OK snapshot without item-level Evidence would violate
        # the public fail-closed contract.
        "retrieval_status": "failed",
        "retrieval_coverage": "unknown",
        "knowledge_snapshot_id": None,
        "retrieval_request": {},
        "retrieval_results": [],
        "retrieval_errors": [],
        "validation_issues": [],
        "regulatory_findings": [],
        "rag_evidence": [],
        "audit_trail": [],
        "model_calls": [],
        "model_errors": [],
        "model_conflicts": [],
        "report_json": {
            "case_id": case_id,
            "overall_status": "needs_review",
            "summary": {
                "total_documents": 0,
                "total_issues": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "average_extraction_confidence": 0.0,
                "retrieval_status": "failed",
                "retrieval_coverage": "unknown",
                "knowledge_snapshot_id": None,
                "outcome_reasons": [],
            },
            "document_summary": [],
            "issues": [],
            "regulatory_findings": [],
            "rag_evidence": [],
            "disclaimer": (
                "This result is a pre-clearance review aid. Final legal decisions "
                "must be made by a responsible human reviewer."
            ),
        },
    }


def _write_image_and_ledger(directory: Path) -> tuple[Path, Path]:
    image_path = directory / "invoice.png"
    Image.new("RGB", (240, 160), color="white").save(image_path)
    page = image_page_from_file(image_path)
    ledger_page = OcrLayoutPage(
        page_number=1,
        image_sha256=page.sha256,
        width_px=page.width_px,
        height_px=page.height_px,
        spans=[
            OcrLayoutSpan(
                span_id="span-1",
                page=1,
                reading_order=0,
                role="text",
                text="COMMERCIAL INVOICE",
                bbox=BBox(page=1, x1=10, y1=10, x2=180, y2=35),
                confidence=0.99,
            )
        ],
    )
    values = {
        "document_id": "doc-image",
        "provider_id": "static-eval-ledger",
        "engine_id": "fixture-layout",
        "engine_version": "1.0",
        "pages": [ledger_page],
    }
    ledger = OcrLayoutLedger(
        ledger_id=stable_ocr_ledger_id(**values),
        **values,
    )
    ledger_path = directory / "invoice.ocr.json"
    ledger_path.write_text(
        json.dumps(ledger.model_dump(mode="json")),
        encoding="utf-8",
    )
    return image_path, ledger_path


def test_existing_run_parser_and_v1_text_contract_remain_compatible() -> None:
    parsed = build_parser().parse_args(["run", "--scenario", "happy_path"])
    payload = {
        "scenario_id": "happy_path",
        "llm_mode": "mock",
        "review_as_of": "2026-07-10",
    }

    assert parsed.command == "run"
    assert parsed.scenario == "happy_path"
    assert _validate_contract_payload("v1", "request", payload)


def test_review_text_manifest_writes_only_v1_public_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_evidence_environment(monkeypatch)
    source = tmp_path / "invoice.txt"
    source.write_text(
        "COMMERCIAL INVOICE\nInvoice No: INV-001\nTotal Amount: USD 100.00\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path,
        contract_version="v1",
        llm_mode="mock",
        documents=[
            {
                "document_id": "doc-text",
                "filename": "invoice.txt",
                "declared_type": "invoice",
                "source": {"kind": "text", "path": source.name},
            }
        ],
    )
    output = tmp_path / "result.json"

    assert main(["review", "--manifest", str(manifest), "--out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    console = capsys.readouterr().out
    serialized = json.dumps(payload)

    assert payload["schema_version"] == "1.0"
    assert payload["case_id"] == "case-local-cli"
    assert str(manifest) not in console
    assert str(source) not in console
    assert "COMMERCIAL INVOICE" not in console
    for forbidden in (
        "documents",
        "content_text",
        "image_pages",
        "data_base64",
        "source_path",
        "manifest_path",
    ):
        assert f'"{forbidden}"' not in serialized


def test_review_parser_has_no_implicit_ocr_provider() -> None:
    parsed = build_parser().parse_args(
        [
            "review",
            "--manifest",
            "review-manifest.json",
            "--out",
            "result.json",
        ]
    )

    assert parsed.ocr_provider is None


def test_review_visual_manifest_requires_explicit_provider_before_io_or_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path,
        contract_version="v2",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-image",
                "filename": "missing.png",
                "source": {"kind": "image", "path": "missing.png"},
            }
        ],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "customs_ai_agent.cli.ensure_llm_mode_available",
        lambda *_args, **_kwargs: calls.append("claude_preflight"),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.build_ocr_layout_provider",
        lambda *_args, **_kwargs: calls.append("provider"),
    )

    with pytest.raises(LocalReviewInputError, match="ocr_provider_required"):
        _cmd_review(
            build_parser().parse_args(
                [
                    "review",
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(tmp_path / "result.json"),
                ]
            )
        )

    assert calls == []


def test_review_real_ocr_requires_complete_release_gate_before_io_or_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path,
        contract_version="v2",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-image",
                "filename": "missing.png",
                "source": {"kind": "image", "path": "missing.png"},
            }
        ],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "customs_ai_agent.cli.ensure_llm_mode_available",
        lambda *_args, **_kwargs: calls.append("claude_preflight"),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.build_ocr_layout_provider",
        lambda *_args, **_kwargs: calls.append("provider"),
    )

    with pytest.raises(
        LocalReviewInputError,
        match="ocr_release_gate_arguments_required",
    ):
        _cmd_review(
            build_parser().parse_args(
                [
                    "review",
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(tmp_path / "result.json"),
                    "--ocr-provider",
                    "paddle",
                    "--ocr-python",
                    sys.executable,
                ]
            )
        )

    assert calls == []


def test_review_rejects_failed_release_attestation_before_io_provider_or_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path,
        contract_version="v2",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-image",
                "filename": "missing.png",
                "source": {"kind": "image", "path": "missing.png"},
            }
        ],
    )
    calls: list[str] = []

    def reject_attestation(**_kwargs):
        calls.append("attestation")
        raise OcrReleaseGateAttestationError("ocr_release_gate_status_not_passing")

    monkeypatch.setattr(
        "customs_ai_agent.cli.load_canonical_ocr_release_gate",
        reject_attestation,
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.ensure_llm_mode_available",
        lambda *_args, **_kwargs: calls.append("claude_preflight"),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.build_ocr_layout_provider",
        lambda *_args, **_kwargs: calls.append("provider"),
    )

    with pytest.raises(
        LocalReviewInputError,
        match="ocr_release_gate_status_not_passing",
    ):
        _cmd_review(
            build_parser().parse_args(
                [
                    "review",
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(tmp_path / "result.json"),
                    "--ocr-provider",
                    "paddle",
                    "--ocr-python",
                    sys.executable,
                    "--ocr-release-gate-report",
                    str(tmp_path / "combined.json"),
                    "--ocr-release-gate-report-sha256",
                    "a" * 64,
                    "--ocr-release-descriptor",
                    str(tmp_path / "descriptor.json"),
                    "--ocr-release-rendered-root",
                    str(tmp_path / "rendered"),
                    "--expected-agent-wheel-sha256",
                    "d" * 64,
                    "--ocr-environment-lock",
                    str(tmp_path / "paddle.requirements.lock.txt"),
                    "--ocr-environment-lock-sha256",
                    "e" * 64,
                    "--ocr-model-assets-lock",
                    str(tmp_path / "model-assets.lock.json"),
                    "--ocr-model-assets-lock-sha256",
                    "f" * 64,
                ]
            )
        )

    assert calls == ["attestation"]


def test_review_ledger_rejects_real_provider_release_arguments_before_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path,
        contract_version="v2",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-image",
                "filename": "missing.png",
                "source": {
                    "kind": "image",
                    "path": "missing.png",
                    "ocr_ledger": "missing.ocr.json",
                },
            }
        ],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "customs_ai_agent.cli.ensure_llm_mode_available",
        lambda *_args, **_kwargs: calls.append("claude_preflight"),
    )

    with pytest.raises(LocalReviewInputError, match="ocr_release_gate_forbidden"):
        _cmd_review(
            build_parser().parse_args(
                [
                    "review",
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(tmp_path / "result.json"),
                    "--ocr-provider",
                    "ledger",
                    "--ocr-release-gate-report",
                    str(tmp_path / "combined.json"),
                ]
            )
        )

    assert calls == []


def test_review_real_ocr_attests_before_provider_and_wraps_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_evidence_environment(monkeypatch)
    image_path = tmp_path / "invoice.png"
    Image.new("RGB", (240, 160), color="white").save(image_path)
    manifest = _write_manifest(
        tmp_path,
        contract_version="v2",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-image",
                "filename": "invoice.png",
                "source": {"kind": "image", "path": image_path.name},
            }
        ],
    )
    output = tmp_path / "result.json"
    report_path = tmp_path / "combined.json"
    descriptor_path = tmp_path / "descriptor.json"
    rendered_root = tmp_path / "rendered"
    environment_lock_path = tmp_path / "paddle.requirements.lock.txt"
    model_assets_lock_path = tmp_path / "model-assets.lock.json"
    attestation = OcrReleaseGateAttestation(
        report_sha256="a" * 64,
        descriptor_sha256="b" * 64,
        manifest_sha256="c" * 64,
        agent_wheel_sha256="d" * 64,
        environment_lock_sha256="e" * 64,
        model_assets_lock_sha256="f" * 64,
        provider="paddle",
        provider_id="paddle",
        engine_id="paddle-test-engine",
        engine_version="1.0",
    )
    events: list[str] = []
    captured: dict[str, object] = {}

    class MatchingProvider:
        def analyze(
            self,
            *,
            document_id: str,
            pages: list[ImagePageInput],
        ) -> OcrLayoutLedger:
            page = pages[0]
            ledger_page = OcrLayoutPage(
                page_number=1,
                image_sha256=page.sha256,
                width_px=page.width_px,
                height_px=page.height_px,
                spans=[
                    OcrLayoutSpan(
                        span_id="span-release-lock",
                        page=1,
                        reading_order=0,
                        role="text",
                        text="COMMERCIAL INVOICE",
                        bbox=BBox(page=1, x1=10, y1=10, x2=180, y2=35),
                        confidence=0.99,
                    )
                ],
            )
            values = {
                "document_id": document_id,
                "provider_id": attestation.provider_id,
                "engine_id": attestation.engine_id,
                "engine_version": attestation.engine_version,
                "model_id": OCR_MODEL_SET_IDS["paddle"],
                "model_assets_lock_sha256": OCR_MODEL_ASSETS_LOCK_SHA256,
                "pages": [ledger_page],
            }
            return OcrLayoutLedger(
                ledger_id=stable_ocr_ledger_id(**values),
                **values,
            )

    def fake_attestation(**kwargs):
        events.append("attestation")
        assert kwargs == {
            "report_path": report_path,
            "expected_report_sha256": "a" * 64,
            "expected_provider": "paddle",
            "descriptor_path": descriptor_path,
            "rendered_root": rendered_root,
            "expected_agent_wheel_sha256": "d" * 64,
            "environment_lock_path": environment_lock_path,
            "expected_environment_lock_sha256": "e" * 64,
            "model_assets_lock_path": model_assets_lock_path,
            "expected_model_assets_lock_sha256": "f" * 64,
            "ocr_python": Path(sys.executable),
        }
        return attestation

    def fake_provider_factory(provider_name, *, python_executable):
        events.append("provider")
        assert provider_name == "paddle"
        assert python_executable == Path(sys.executable)
        return MatchingProvider()

    def fake_build_analyzer(_env_file=None, *, ocr_layout_provider=None):
        events.append("analyzer")
        captured["ocr_provider"] = ocr_layout_provider
        return object()

    def fake_run_review(request, **_options):
        events.append("run")
        captured["request"] = request
        return _minimal_completed_state(request)

    monkeypatch.setattr(
        "customs_ai_agent.cli.load_canonical_ocr_release_gate",
        fake_attestation,
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.ensure_llm_mode_available",
        lambda *_args, **_kwargs: events.append("claude_preflight"),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.build_ocr_layout_provider",
        fake_provider_factory,
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.build_document_analyzer",
        fake_build_analyzer,
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr("customs_ai_agent.cli.run_review_v2", fake_run_review)

    assert (
        main(
            [
                "review",
                "--manifest",
                str(manifest),
                "--out",
                str(output),
                "--rag-url",
                "http://127.0.0.1:8010",
                "--ocr-provider",
                "paddle",
                "--ocr-python",
                sys.executable,
                "--ocr-release-gate-report",
                str(report_path),
                "--ocr-release-gate-report-sha256",
                "a" * 64,
                "--ocr-release-descriptor",
                str(descriptor_path),
                "--ocr-release-rendered-root",
                str(rendered_root),
                "--expected-agent-wheel-sha256",
                "d" * 64,
                "--ocr-environment-lock",
                str(environment_lock_path),
                "--ocr-environment-lock-sha256",
                "e" * 64,
                "--ocr-model-assets-lock",
                str(model_assets_lock_path),
                "--ocr-model-assets-lock-sha256",
                "f" * 64,
            ]
        )
        == 0
    )

    provider = captured["ocr_provider"]
    request = captured["request"]
    assert isinstance(provider, ReleaseLockedOcrProvider)
    assert isinstance(request, ReviewRunRequestV2)
    ledger = provider.analyze(
        document_id="doc-image",
        pages=request.documents[0].image_pages,
    )
    assert ledger.engine_id == attestation.engine_id
    assert events == [
        "attestation",
        "claude_preflight",
        "provider",
        "analyzer",
        "run",
    ]


def test_review_image_v2_ledger_injects_bound_provider_and_redacts_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _clear_evidence_environment(monkeypatch)
    image_path, ledger_path = _write_image_and_ledger(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        contract_version="v2",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-image",
                "filename": "invoice.png",
                "source": {
                    "kind": "image",
                    "path": image_path.name,
                    "ocr_ledger": ledger_path.name,
                },
            }
        ],
    )
    output = tmp_path / "result.json"
    captured: dict[str, object] = {}
    retriever = object()
    analyzer = object()

    monkeypatch.setattr(
        "customs_ai_agent.cli.ensure_llm_mode_available",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.HttpRegulationRetriever",
        lambda *_args, **_kwargs: retriever,
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )

    def fake_build_analyzer(_env_file=None, *, ocr_layout_provider=None):
        captured["ocr_provider"] = ocr_layout_provider
        return analyzer

    def fake_run_review(request, **options):
        captured["request"] = request
        captured["options"] = options
        return _minimal_completed_state(request)

    monkeypatch.setattr(
        "customs_ai_agent.cli.build_document_analyzer",
        fake_build_analyzer,
    )
    monkeypatch.setattr("customs_ai_agent.cli.run_review_v2", fake_run_review)

    assert (
        main(
            [
                "review",
                "--manifest",
                str(manifest),
                "--out",
                str(output),
                "--rag-url",
                "http://127.0.0.1:8010",
                "--ocr-provider",
                "ledger",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    console = capsys.readouterr().out
    request = captured["request"]
    assert isinstance(request, ReviewRunRequestV2)
    page = request.documents[0].image_pages[0]
    provider = captured["ocr_provider"]

    assert provider.analyze(document_id="doc-image", pages=[page]).document_id == "doc-image"
    assert captured["options"] == {"retriever": retriever, "analyzer": analyzer}
    assert payload["schema_version"] == "2.0"
    assert "data_base64" not in json.dumps(payload)
    assert str(manifest) not in console
    assert str(image_path) not in console
    assert page.data_base64 not in console
    Draft202012Validator(_agent_contract_schema("v2", "result")).validate(payload)


def test_pdf_manifest_is_rasterized_with_bounded_page_provenance(tmp_path: Path) -> None:
    pdf_path = tmp_path / "packing-list.pdf"
    document = canvas.Canvas(str(pdf_path))
    document.drawString(72, 720, "PACKING LIST")
    document.save()
    manifest_path = _write_manifest(
        tmp_path,
        contract_version="v2",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-pdf",
                "filename": "packing-list.pdf",
                "source": {"kind": "pdf", "path": pdf_path.name},
            }
        ],
    )
    manifest, root = _load_manifest(manifest_path)

    prepared = _prepare_manifest(
        manifest,
        root,
        ocr_provider="paddle",
        review_as_of=None,
    )
    request = ReviewRunRequestV2.model_validate(prepared.request_values)
    assert request.documents[0].image_pages == []
    assert request.documents[0].pdf_pages is not None
    page = request.documents[0].pdf_pages.pages[0]

    assert request.documents[0].pdf_pages.page_count == 1
    assert request.documents[0].pdf_pages.source_pdf_sha256 == page.provenance.source_sha256
    assert page.provenance.source_media_type == "application/pdf"
    assert page.provenance.transforms == ["pdf_rasterize"]
    assert page.provenance.rasterizer_id == "pypdfium2"
    assert page.provenance.raster_dpi == 180
    assert page.data_base64


def test_pdf_rasterizer_rejects_all_huge_media_boxes_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def __init__(self, size: tuple[float, float]) -> None:
            self.size = size
            self.closed = False
            self.rendered = False

        def get_size(self) -> tuple[float, float]:
            return self.size

        def get_mediabox(self) -> tuple[float, float, float, float]:
            return (0.0, 0.0, self.size[0], self.size[1])

        def render(self, **_kwargs):
            self.rendered = True
            raise AssertionError("preflight must reject before rendering")

        def close(self) -> None:
            self.closed = True

    safe_page = FakePage((612.0, 792.0))
    huge_page = FakePage((100_000.0, 100_000.0))

    class FakeDocument:
        def __init__(self) -> None:
            self.pages = [safe_page, huge_page]
            self.closed = False
            self.raw = object()

        def __len__(self) -> int:
            return len(self.pages)

        def __getitem__(self, index: int) -> FakePage:
            return self.pages[index]

        def close(self) -> None:
            self.closed = True

    document = FakeDocument()
    pdfium = types.ModuleType("pypdfium2")
    pdfium.PdfDocument = lambda _payload: document  # type: ignore[attr-defined]
    pdfium.PYPDFIUM_INFO = "test-version"  # type: ignore[attr-defined]
    pdfium_raw = types.ModuleType("pypdfium2.raw")
    pdfium_raw.FPDF_GetSecurityHandlerRevision = lambda _document: -1  # type: ignore[attr-defined]
    pdfium.raw = pdfium_raw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", pdfium)
    monkeypatch.setitem(sys.modules, "pypdfium2.raw", pdfium_raw)

    with pytest.raises(RuntimeError) as captured:
        pdf_worker._render_pdf(
            b"%PDF-1.7\nsynthetic\nstartxref\n1\n%%EOF\n",
            dpi=180,
            max_pages=5,
        )

    assert "pdf_page_dimensions_invalid" in str(captured.value)
    assert all(page.closed for page in document.pages)
    assert not any(page.rendered for page in document.pages)
    assert document.closed


def test_pdf_worker_command_cwd_and_environment_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "ANTHROPIC_API_KEY": "should-not-cross",
        "LAW_API_OC": "should-not-cross",
        "DATABASE_URL": "should-not-cross",
        "PYTHONPATH": "should-not-cross",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)

    executable = Path(sys.executable).resolve()
    command = _pdf_worker_command(executable)
    environment = _pdf_worker_environment()
    cwd = Path(_pdf_worker_cwd()).resolve()

    assert command == [
        str(executable),
        "-I",
        "-m",
        "customs_ai_agent.pdf_worker",
    ]
    assert not any(value in command for value in secrets.values())
    assert not any(name in environment for name in secrets)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert cwd.is_dir()
    assert cwd != Path.cwd().resolve()


def test_pdf_worker_process_uses_isolated_command_and_suppressed_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross")
    monkeypatch.setenv("DATABASE_URL", "must-not-cross")
    captured: dict[str, object] = {}

    class OpenInput(BytesIO):
        def close(self) -> None:
            self.flush()

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = OpenInput()
            self.stdout = BytesIO(b'{"status":"test"}\n')

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            return 0

    process = FakeProcess()

    def fake_popen(command, **options):
        captured["command"] = command
        captured["options"] = options
        return process

    monkeypatch.setattr(image_adapters.subprocess, "Popen", fake_popen)
    executable = Path(sys.executable).resolve()

    response = _invoke_pdf_worker(
        b'{"request":"bounded"}\n',
        python_executable=executable,
        timeout_seconds=2,
    )

    options = captured["options"]
    assert isinstance(options, dict)
    assert captured["command"] == _pdf_worker_command(executable)
    assert options["stderr"] is image_adapters.subprocess.DEVNULL
    assert Path(options["cwd"]).resolve() == Path(_pdf_worker_cwd()).resolve()
    assert "shell" not in options
    assert "ANTHROPIC_API_KEY" not in options["env"]
    assert "DATABASE_URL" not in options["env"]
    assert process.stdin.getvalue() == b'{"request":"bounded"}\n'
    assert response == b'{"status":"test"}\n'


@pytest.mark.parametrize(
    "raw_response",
    [
        b"not-json\n",
        b"{}\n{}\n",
    ],
)
def test_pdf_worker_rejects_malformed_responses(
    raw_response: bytes,
) -> None:
    with pytest.raises(PdfRasterizerError):
        _parse_pdf_worker_response(
            raw_response,
            request_id="pdfreq_" + "a" * 32,
            source_hash="b" * 64,
            dpi=180,
            max_pages=5,
        )


def test_pdf_worker_rejects_oversized_response() -> None:
    with pytest.raises(PdfRasterizerError):
        _parse_pdf_worker_response(
            b"x" * (_MAX_PDF_WORKER_RESPONSE_BYTES + 1),
            request_id="pdfreq_" + "a" * 32,
            source_hash="b" * 64,
            dpi=180,
            max_pages=5,
        )


def test_pdf_worker_rejects_polyglot_and_corrupt_containers() -> None:
    valid = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\nstartxref\n9\n%%EOF\n"

    pdf_worker._validate_pdf_container(valid)
    with pytest.raises(RuntimeError, match="pdf_polyglot_rejected"):
        pdf_worker._validate_pdf_container(valid + b"MZ")
    with pytest.raises(RuntimeError, match="pdf_structure_invalid"):
        pdf_worker._validate_pdf_container(valid.replace(b"%%EOF", b""))


def test_pdf_worker_rejects_encrypted_pdf_without_rendering() -> None:
    buffer = BytesIO()
    document = canvas.Canvas(buffer, encrypt="test-password")
    document.drawString(72, 720, "CONFIDENTIAL")
    document.save()

    with pytest.raises(PdfRasterizerError):
        _Pdfium2Rasterizer().rasterize(
            buffer.getvalue(),
            dpi=180,
            max_pages=5,
        )


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.txt", "C:\\private\\outside.txt"],
)
def test_manifest_rejects_unsafe_source_paths_without_echoing_them(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        contract_version="v1",
        llm_mode="mock",
        documents=[
            {
                "document_id": "doc-text",
                "filename": "invoice.txt",
                "source": {"kind": "text", "path": unsafe_path},
            }
        ],
    )

    with pytest.raises(LocalReviewInputError) as caught:
        _load_manifest(manifest_path)

    assert unsafe_path not in str(caught.value)


def test_v1_manifest_rejects_visual_input_before_file_loading(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        contract_version="v1",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-image",
                "filename": "missing.png",
                "source": {"kind": "image", "path": "missing.png"},
            }
        ],
    )
    manifest, root = _load_manifest(manifest_path)

    with pytest.raises(LocalReviewInputError, match="v2_required_for_visual_input"):
        _prepare_manifest(
            manifest,
            root,
            ocr_provider="paddle",
            review_as_of=None,
        )


def test_static_agent_contracts_match_exporter_and_validate_v1_v2_samples(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (40, 40), color="white").save(image_path)
    page = image_page_from_file(image_path).model_dump(mode="json")
    v1_request = {
        "documents": [
            {
                "document_id": "doc-text",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE",
            }
        ],
        "llm_mode": "mock",
        "review_as_of": "2026-07-10",
    }
    v2_request = {
        "documents": [
            {
                "document_id": "doc-image",
                "filename": "invoice.png",
                "content_text": None,
                "image_pages": [page],
            }
        ],
        "llm_mode": "llm",
        "review_as_of": "2026-07-10",
    }

    for version, sample in (("v1", v1_request), ("v2", v2_request)):
        for kind in ("request", "result"):
            exported = _agent_contract_schema(version, kind)
            checked_in = json.loads(
                (CONTRACT_ROOT / version / f"review_{kind}.schema.json").read_text(encoding="utf-8")
            )
            assert checked_in == exported
            Draft202012Validator.check_schema(exported)
        Draft202012Validator(_agent_contract_schema(version, "request")).validate(sample)
        _validate_contract_payload(version, "request", sample)

    with pytest.raises(ContractValidationError, match="v1_visual_input_forbidden"):
        _validate_contract_payload("v1", "request", v2_request)
    assert list(
        Draft202012Validator(_agent_contract_schema("v1", "request")).iter_errors(v2_request)
    )


def test_public_result_contract_versions_are_explicit_and_raw_keys_fail_closed() -> None:
    request = ReviewRunRequest(
        scenario_id="happy_path",
        review_run_id="run-contract",
        review_as_of=date(2026, 7, 10),
    )
    state = _minimal_completed_state(request)

    v1_result = _serialize_public_result(state, contract_version="v1")
    v2_result = _serialize_public_result(state, contract_version="v2")

    assert v1_result["schema_version"] == "1.0"
    assert v2_result["schema_version"] == "2.0"
    Draft202012Validator(_agent_contract_schema("v1", "result")).validate(v1_result)
    Draft202012Validator(_agent_contract_schema("v2", "result")).validate(v2_result)
    unsafe = dict(v2_result)
    unsafe["data_base64"] = "sensitive"
    with pytest.raises(ContractValidationError):
        _validate_contract_payload("v2", "result", unsafe)


def test_v2_result_contains_redacted_extracted_normalized_and_ocr_provenance() -> None:
    request = ReviewRunRequest(
        scenario_id="happy_path",
        review_run_id="run-v2-provenance",
        review_as_of=date(2026, 7, 10),
    )
    state = _minimal_completed_state(request)
    ledger_id = "ocr_" + "a" * 64
    evidence = {
        "document_id": "doc-image",
        "page": 1,
        "bbox": {"page": 1, "x1": 10, "y1": 20, "x2": 180, "y2": 50},
        "raw_text": "INVOICE NO INV-001",
        "text_start": None,
        "text_end": None,
        "confidence": 0.99,
        "source_modality": "image",
        "image_sha256": "b" * 64,
        "image_width_px": 1200,
        "image_height_px": 1800,
        "coordinate_space": "original_image_pixels_top_left",
        "ocr_ledger_id": ledger_id,
        "ocr_span_ids": ["span-invoice-no"],
    }
    field = {
        "name": "invoice_no",
        "value": "INV-001",
        "normalized_value": None,
        "confidence": 0.70,
        "confidence_basis": "model_mapped_to_independent_ocr_uncalibrated",
        "evidence": evidence,
    }
    state["classified_documents"] = {
        "doc-image": {
            "document_id": "doc-image",
            "document_type": "invoice",
            "provider": "anthropic",
            "input_modality": "image",
        }
    }
    state["extracted_documents"] = {
        "doc-image": {
            "document_id": "doc-image",
            "filename": "private-local-name.png",
            "document_type": "invoice",
            "fields": {"invoice_no": field},
            "line_items": [],
            "extraction_status": "completed",
            "extraction_confidence": 0.70,
            "extraction_provider": "anthropic",
            "input_modality": "image",
            "model_id": "claude-sonnet-5",
            "prompt_version": "1.2.0",
            "ocr_layout": {
                "schema_version": "1.0",
                "ledger_id": ledger_id,
                "provider_id": "paddle-pp-structure-v3",
                "engine_id": "paddleocr",
                "engine_version": "3.7.0",
                "model_id": "test-paddle-model-set",
                "model_assets_lock_sha256": "a" * 64,
            },
        }
    }
    state["normalized_documents"] = {
        "doc-image": {
            "document_id": "doc-image",
            "document_type": "invoice",
            "fields": {"invoice_no": "INV-001"},
            "line_items": [],
        }
    }
    state["report_json"]["summary"]["total_documents"] = 1
    state["report_json"]["document_summary"] = [
        {
            "document_id": "doc-image",
            "document_type": "invoice",
            "extraction_status": "completed",
            "extraction_confidence": 0.70,
        }
    ]

    payload = _serialize_public_result(state, contract_version="v2")
    document = payload["document_results"][0]
    result_field = document["fields"][0]
    serialized = json.dumps(payload)

    assert document["document_id"] == "doc-image"
    assert document["ocr_layout"]["ledger_id"] == ledger_id
    assert result_field["value"] == "INV-001"
    assert result_field["normalized_value"] == "INV-001"
    assert result_field["source"]["ocr_span_ids"] == ["span-invoice-no"]
    assert result_field["source"]["bbox"]["x1"] == 10.0
    assert "private-local-name.png" not in serialized
    assert "INVOICE NO INV-001" not in serialized
    assert "raw_text" not in serialized
    assert "data_base64" not in serialized


def test_real_workflow_public_result_recursively_excludes_local_filename() -> None:
    private_name = "customer-acme-private-invoice-2026.txt"
    request = ReviewRunRequest(
        case_id="case-filename-leak-regression",
        review_run_id="run-filename-leak-regression",
        review_as_of=date(2026, 7, 10),
        documents=[
            {
                "document_id": "doc-invoice",
                "filename": private_name,
                "declared_type": "invoice",
                "content_text": (
                    "COMMERCIAL INVOICE\n"
                    "INVOICE NO: INV-001\n"
                    "SELLER: ACME\n"
                    "BUYER: BUYER\n"
                    "CURRENCY: USD\n"
                    "TOTAL AMOUNT: 100\n"
                    "ITEM: 1 | GOODS | 731815 | 1 | EA | 100\n"
                ),
            }
        ],
    )

    state = run_review(request)
    payload = _serialize_public_result(state, contract_version="v2")
    serialized = json.dumps(payload, sort_keys=True)

    assert private_name not in serialized
    assert '"filename"' not in serialized
    assert all("filename" not in document for document in payload["report"]["document_summary"])


def test_recursive_public_result_guard_rejects_filename_at_any_depth() -> None:
    with pytest.raises(RuntimeError, match="public_result_safety_violation"):
        _serialize_public_result(
            {
                **_minimal_completed_state(
                    ReviewRunRequest(
                        scenario_id="happy_path",
                        review_run_id="run-recursive-filename-guard",
                        review_as_of=date(2026, 7, 10),
                    )
                ),
                "report_json": {
                    "case_id": "case-manual",
                    "overall_status": "needs_review",
                    "summary": {"total_issues": 0},
                    "document_summary": [
                        {
                            "document_id": "doc-private",
                            "filename": "private-name.pdf",
                        }
                    ],
                    "issues": [],
                    "regulatory_findings": [],
                    "rag_evidence": [],
                },
            },
            contract_version="v2",
        )


def test_failed_v2_document_contract_forbids_business_values() -> None:
    common = {
        "document_id": "doc-failed",
        "document_type": "unknown",
        "input_modality": "image",
        "extraction_status": "failed",
        "extraction_confidence": 0,
        "extraction_provider": "ocr_layout",
    }

    assert ReviewDocumentResultV2.model_validate(common).ocr_layout is None
    with pytest.raises(ValidationError, match="cannot contain fields"):
        ReviewDocumentResultV2.model_validate(
            {
                **common,
                "fields": [
                    {
                        "name": "invoice_no",
                        "value": "SHOULD-NOT-EXIST",
                        "confidence": 0,
                        "confidence_basis": "rule_heuristic",
                        "source": {
                            "document_id": "doc-failed",
                            "page": 1,
                            "source_modality": "text",
                            "text_start": 0,
                            "text_end": 1,
                        },
                    }
                ],
            }
        )


def test_validate_contract_cli_reports_only_contract_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = tmp_path / "request-with-private-path-name.json"
    request_path.write_text(
        json.dumps(
            {
                "scenario_id": "happy_path",
                "llm_mode": "mock",
                "review_as_of": "2026-07-10",
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "validate-contract",
                "--version",
                "v1",
                "--kind",
                "request",
                "--input",
                str(request_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert output.strip() == "valid: agent v1 request"
    assert str(request_path) not in output


@pytest.mark.parametrize(
    ("scope", "kind", "expected_suffix"),
    [
        ("internal", "ocr-layout-ledger", "/internal/v1/ocr-layout-ledger.schema.json"),
        (
            "internal",
            "claude-vision-candidate",
            "/internal/v1/claude-vision-candidate.schema.json",
        ),
        ("local", "review-manifest", "/local/v1/review-manifest.schema.json"),
    ],
)
def test_export_schema_cli_exposes_packaged_non_backend_contracts(
    tmp_path: Path,
    scope: str,
    kind: str,
    expected_suffix: str,
) -> None:
    output = tmp_path / f"{scope}-{kind}.json"

    assert (
        main(
            [
                "export-schema",
                "--scope",
                scope,
                "--version",
                "v1",
                "--kind",
                kind,
                "--out",
                str(output),
            ]
        )
        == 0
    )
    schema = json.loads(output.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith(expected_suffix)


def test_export_schema_cli_rejects_unreleased_scope_version_kind_combination() -> None:
    with pytest.raises(
        ContractValidationError,
        match="contract_schema_selection_invalid",
    ):
        _cmd_export_schema(
            build_parser().parse_args(
                [
                    "export-schema",
                    "--scope",
                    "local",
                    "--version",
                    "v2",
                    "--kind",
                    "review-manifest",
                ]
            )
        )
