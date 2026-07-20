from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from customs_ai_agent.adapters.llm import (
    CLAUDE_CANDIDATE_SCHEMA_VERSION,
    DEFAULT_CLAUDE_MODEL,
    LLMProviderError,
)
from customs_ai_agent.adapters.ocr import OcrLayoutProviderError
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.schemas.llm import (
    ClaudeCropReadCandidate,
    ClaudeVisionDocumentCandidate,
)
from customs_ai_agent.schemas.models import ReviewRunResultV2
from evals import run_image_runtime_diagnostic as diagnostic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, tuple[diagnostic.FixedDocument, ...]]:
    root = tmp_path / "_artifacts" / "datasets" / "trade_documents_images" / "source-v1" / "raw"
    specifications = (
        ("ci:test", "commercial_invoice", "invoice", "commercial_invoice/CI_test.jpg", "red"),
        ("pl:test", "packing_list", "packing_list", "packing_list/PL_test.jpg", "green"),
        (
            "bl:test",
            "bill_of_lading",
            "bill_of_lading",
            "bill_of_lading/BL_test.jpg",
            "blue",
        ),
    )
    fixed: list[diagnostic.FixedDocument] = []
    for document_id, source_type, expected_type, relative_name, color in specifications:
        image = root / relative_name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 48), color=color).save(image, format="JPEG")
        fixed.append(
            diagnostic.FixedDocument(
                document_id=document_id,
                source_document_type=source_type,
                expected_document_type=expected_type,
                relative_name=relative_name,
                source_sha256=_sha256(image),
            )
        )

    gold_file = root.parent / "derived" / "adjudicated-gold" / "v1" / "adjudicated-gold.json"
    gold_file.parent.mkdir(parents=True, exist_ok=True)
    gold = {
        "schema_version": "trade-image-adjudicated-gold/v1",
        "gold_set_id": diagnostic.EXPECTED_GOLD_SET_ID,
        "selection_id": diagnostic.EXPECTED_SELECTION_ID,
        "gold_eligible": True,
        "cases": [
            {
                "case_id": "test-case",
                "adjudication_status": "adjudicated",
                "gold_eligible": True,
                "triplet_fingerprint_sha256": "a" * 64,
                "annotation_references": ["review-a", "review-b"],
                "documents": [
                    {
                        "document_id": document.document_id,
                        "source_document_type": document.source_document_type,
                        "expected_document_type": document.expected_document_type,
                        "relative_path": document.relative_name,
                        "source_image_sha256": document.source_sha256,
                        # This business value must never reach a report.
                        "expected_fields": {"secret_field": "PRIVATE-GOLD-VALUE"},
                    }
                    for document in fixed
                ],
            }
        ],
    }
    gold_file.write_text(json.dumps(gold), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "CANONICAL_DATASET_ROOT", root)
    monkeypatch.setattr(diagnostic, "CANONICAL_GOLD_FILE", gold_file)
    monkeypatch.setattr(diagnostic, "EXPECTED_GOLD_SHA256", _sha256(gold_file))
    monkeypatch.setattr(diagnostic, "FIXED_CASE_ID", "test-case")
    monkeypatch.setattr(diagnostic, "FIXED_TRIPLET_SHA256", "a" * 64)
    monkeypatch.setattr(diagnostic, "FIXED_DOCUMENTS", tuple(fixed))
    return root, tuple(fixed)


def _env_file(tmp_path: Path) -> Path:
    env_file = tmp_path / "private.env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=private-anthropic-secret\n"
        "EVIDENCE_SERVICE_TOKEN=private-evidence-secret\n",
        encoding="utf-8",
    )
    return env_file


def _call(
    *,
    document_id: str,
    ledger_number: int,
    provider: str = "paddle",
    request_id: str = "PRIVATE-PROVIDER-REQUEST-ID",
) -> SimpleNamespace:
    return SimpleNamespace(
        document_id=document_id,
        provider="anthropic",
        model_id=DEFAULT_CLAUDE_MODEL,
        prompt_id=diagnostic.DOCUMENT_VISION_ANALYSIS_PROMPT_ID,
        prompt_version=diagnostic.DOCUMENT_VISION_ANALYSIS_PROMPT_VERSION,
        prompt_sha256=diagnostic._prompt_sha256(
            diagnostic.load_document_vision_analysis_system_prompt
        ),
        output_schema_id="customs_claude_vision_document_candidate",
        output_schema_version=CLAUDE_CANDIDATE_SCHEMA_VERSION,
        output_schema_sha256=diagnostic._canonical_sha256(
            ClaudeVisionDocumentCandidate.model_json_schema()
        ),
        input_modality="image",
        ocr_ledger_id=f"ocr_{ledger_number:064x}",
        ocr_provider_id=provider,
        ocr_engine_id="PP-StructureV3:test",
        ocr_engine_version="3.7.0",
        ocr_model_id="paddle-test-model-v1",
        ocr_model_assets_lock_sha256="b" * 64,
        request_id=request_id,
    )


def _result(
    fixed: tuple[diagnostic.FixedDocument, ...],
    *,
    provider: str = "paddle",
    partial: bool = False,
) -> ReviewRunResultV2:
    calls = [
        _call(
            document_id=document.document_id,
            ledger_number=index,
            provider=provider,
        )
        for index, document in enumerate(fixed, start=1)
    ]
    documents = [
        SimpleNamespace(
            document_id=document.document_id,
            input_modality="image",
            extraction_status="completed",
            ocr_layout=SimpleNamespace(ledger_id=call.ocr_ledger_id),
            fields=[SimpleNamespace(value="PRIVATE-EXTRACTED-VALUE")],
        )
        for document, call in zip(fixed, calls, strict=True)
    ]
    retrieval_errors = (
        [SimpleNamespace(code="PARTIAL_RESULTS", detail="PRIVATE-EVIDENCE-BODY")] if partial else []
    )
    return ReviewRunResultV2.model_construct(
        document_results=documents,
        model_calls=calls,
        model_errors=[],
        model_conflicts=[],
        retrieval_status="partial" if partial else "ok",
        retrieval_coverage="partial" if partial else "complete",
        knowledge_snapshot_id=None if partial else "private-snapshot-id",
        retrieval_results=[
            SimpleNamespace(
                coverage="partial" if partial else "complete",
                errors=[SimpleNamespace(code="PARTIAL_RESULTS")] if partial else [],
            )
        ],
        retrieval_errors=retrieval_errors,
        status="needs_review",
        requires_human_review=True,
        rag_evidence=[] if partial else [SimpleNamespace()],
        validation_issues=[],
        regulatory_findings=[],
    )


class _FakeProvider:
    def __init__(self) -> None:
        self.closed = False

    def analyze(self, **_options: Any) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _patch_live(
    monkeypatch: pytest.MonkeyPatch,
    result: ReviewRunResultV2,
    captured: dict[str, Any],
    *,
    readiness: date | Exception = date(2026, 7, 10),
) -> None:
    def ready(*_args: Any, **kwargs: Any) -> date:
        captured.setdefault("order", []).append("readiness")
        captured["readiness_options"] = kwargs
        if isinstance(readiness, Exception):
            raise readiness
        return readiness

    provider = _FakeProvider()

    def build_provider(name: str, **options: Any) -> _FakeProvider:
        captured.setdefault("order", []).append("provider")
        captured["provider_name"] = name
        captured["provider_options"] = options
        captured["provider"] = provider
        return provider

    def run_nonrelease_diagnostic(**options: Any) -> ReviewRunResultV2:
        captured.setdefault("order", []).append("diagnostic_review")
        captured["request"] = options["request"]
        captured["diagnostic_options"] = options
        return result

    monkeypatch.setattr(diagnostic, "fetch_supported_review_as_of", ready)
    monkeypatch.setattr(diagnostic, "build_ocr_layout_provider", build_provider)
    monkeypatch.setattr(
        diagnostic,
        "_run_nonrelease_diagnostic_review",
        run_nonrelease_diagnostic,
    )


def _evaluate(
    *,
    env_file: Path,
    allow_unready: bool = False,
) -> tuple[int, dict[str, Any]]:
    return diagnostic.evaluate(
        env_file=env_file,
        provider="paddle",
        # POSIX virtual environments normally expose ``sys.executable`` as a
        # symlink. The production diagnostic requires the exact regular-file
        # target so the executable identity can be hashed and frozen.
        ocr_python=Path(sys.executable).resolve(strict=True),
        evidence_base_url="http://127.0.0.1:8010",
        review_date=date(2026, 7, 10),
        live=True,
        nonrelease_ocr_diagnostic=True,
        allow_unready_diagnostic=allow_unready,
    )


def test_explicit_flags_are_required_before_any_preflight_or_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        diagnostic,
        "_contract_preflight",
        lambda: pytest.fail("contract preflight must not run"),
    )

    exit_code, report = diagnostic.evaluate(
        env_file=tmp_path / "missing.env",
        provider="paddle",
        ocr_python=tmp_path / "missing-python",
        evidence_base_url="http://127.0.0.1:8010",
        review_date=None,
    )

    assert exit_code == 1
    assert report["status"] == "PREFLIGHT_BLOCKED"
    assert report["phase"] == "PREFLIGHT_BLOCKED"
    assert report["release_ready"] is False
    assert set(report["error_codes"]) == {
        "live_flag_required",
        "nonrelease_ocr_diagnostic_flag_required",
        "review_date_required",
    }


def test_ready_service_runs_public_runtime_and_stays_nonrelease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, fixed = _canonical_fixture(monkeypatch, tmp_path)
    env_file = _env_file(tmp_path)
    captured: dict[str, Any] = {"order": []}
    _patch_live(monkeypatch, _result(fixed), captured)

    exit_code, report = _evaluate(env_file=env_file)

    assert exit_code == 0
    assert report["status"] == "DIAGNOSTIC_PASS"
    assert report["release_ready"] is False
    assert report["release_decision_eligible"] is False
    assert report["diagnostic_execution_passed"] is True
    assert report["counts"]["input_document_count"] == 3
    assert report["counts"]["completed_document_count"] == 3
    assert report["counts"]["claude_main_call_count"] == 3
    assert report["counts"]["ocr_ledger_count"] == 3
    assert report["ocr"]["release_selected_provider_claimed"] is False
    assert captured["order"] == [
        "readiness",
        "provider",
        "diagnostic_review",
    ]
    assert captured["readiness_options"]["contract_identity_mode"] == "required"
    assert captured["provider"].closed is True
    assert len(captured["request"].documents) == 3
    assert all(document.image_pages for document in captured["request"].documents)


def test_unready_data_requires_override_before_provider_or_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, fixed = _canonical_fixture(monkeypatch, tmp_path)
    env_file = _env_file(tmp_path)
    captured: dict[str, Any] = {"order": []}
    _patch_live(
        monkeypatch,
        _result(fixed, partial=True),
        captured,
        readiness=RetrievalError("Evidence Service readiness request failed: HTTP status 503"),
    )

    exit_code, report = _evaluate(env_file=env_file)

    assert exit_code == 1
    assert captured["order"] == ["readiness"]
    assert report["status"] == "BLOCKED_DATA_READINESS"
    assert report["data_readiness_status"] == "UNREADY_503"
    assert report["error_codes"] == ["evidence_readiness_unready_503"]


def test_missing_evidence_contract_identity_fails_before_provider_or_claude(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, fixed = _canonical_fixture(monkeypatch, tmp_path)
    captured: dict[str, Any] = {"order": []}
    _patch_live(
        monkeypatch,
        _result(fixed),
        captured,
        readiness=RetrievalError(
            "Evidence Service readiness request failed: contract identity missing"
        ),
    )

    exit_code, report = _evaluate(env_file=_env_file(tmp_path))

    assert exit_code == 1
    assert captured["order"] == ["readiness"]
    assert captured["readiness_options"]["contract_identity_mode"] == "required"
    assert report["status"] == "BLOCKED_DATA_READINESS"
    assert report["data_readiness_status"] == "FAILED"
    assert report["error_codes"] == ["evidence_contract_identity_missing"]


def test_unready_override_runs_fail_closed_and_never_becomes_release_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, fixed = _canonical_fixture(monkeypatch, tmp_path)
    env_file = _env_file(tmp_path)
    captured: dict[str, Any] = {"order": []}
    _patch_live(
        monkeypatch,
        _result(fixed, partial=True),
        captured,
        readiness=RetrievalError("Evidence Service readiness request failed: HTTP status 503"),
    )

    exit_code, report = _evaluate(env_file=env_file, allow_unready=True)

    assert exit_code == 1
    assert report["status"] == "BLOCKED_DATA_READINESS"
    assert report["release_ready"] is False
    assert report["diagnostic_execution_passed"] is True
    assert report["agent_result"]["status"] == "needs_review"
    assert report["agent_result"]["requires_human_review"] is True
    assert "evidence_data_incomplete" in report["error_codes"]
    assert "PARTIAL_RESULTS" in report["error_codes"]
    assert captured["order"][-1] == "diagnostic_review"


def test_tampered_canonical_image_fails_before_network_or_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, fixed = _canonical_fixture(monkeypatch, tmp_path)
    (root / fixed[0].relative_name).write_bytes(b"tampered")
    monkeypatch.setattr(
        diagnostic,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: pytest.fail("network must not run"),
    )
    monkeypatch.setattr(
        diagnostic,
        "build_ocr_layout_provider",
        lambda *_args, **_kwargs: pytest.fail("provider must not run"),
    )

    exit_code, report = _evaluate(env_file=_env_file(tmp_path))

    assert exit_code == 1
    assert report["status"] == "BLOCKED_EVALUATION_ASSET"
    assert report["phase"] == "PREFLIGHT_BLOCKED"
    assert report["error_codes"] == ["fixed_source_image_hash_mismatch"]


def test_missing_canonical_tree_never_falls_back_to_root_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root_reference = tmp_path / "trade_documents_images"
    root_reference.mkdir()
    monkeypatch.setattr(
        diagnostic,
        "CANONICAL_DATASET_ROOT",
        tmp_path / "_artifacts" / "datasets" / "missing" / "raw",
    )
    monkeypatch.setattr(
        diagnostic,
        "CANONICAL_GOLD_FILE",
        root_reference / "adjudicated-gold.json",
    )
    monkeypatch.setattr(
        diagnostic,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: pytest.fail("network must not run"),
    )

    exit_code, report = _evaluate(env_file=_env_file(tmp_path))

    assert exit_code == 1
    assert report["status"] == "BLOCKED_EVALUATION_ASSET"
    assert report["error_codes"] == ["canonical_dataset_or_gold_missing"]


def test_explicitly_invalidated_gold_never_reaches_provider_or_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _canonical_fixture(monkeypatch, tmp_path)
    observed_hash = _sha256(diagnostic.CANONICAL_GOLD_FILE)
    monkeypatch.setattr(
        diagnostic,
        "image_gold_invalidation_code",
        lambda digest: (
            "adjudicated_gold_invalidated_by_label_error" if digest == observed_hash else None
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: pytest.fail("network must not run"),
    )
    monkeypatch.setattr(
        diagnostic,
        "build_ocr_layout_provider",
        lambda *_args, **_kwargs: pytest.fail("provider must not run"),
    )

    exit_code, report = _evaluate(env_file=_env_file(tmp_path))

    assert exit_code == 1
    assert report["status"] == "BLOCKED_EVALUATION_ASSET"
    assert report["error_codes"] == ["adjudicated_gold_invalidated_by_label_error"]


def test_report_omits_values_locations_destinations_credentials_and_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, fixed = _canonical_fixture(monkeypatch, tmp_path)
    env_file = _env_file(tmp_path)
    captured: dict[str, Any] = {"order": []}
    _patch_live(monkeypatch, _result(fixed), captured)

    exit_code, report = _evaluate(env_file=env_file)
    serialized = json.dumps(report)

    assert exit_code == 0
    for forbidden in (
        "PRIVATE-GOLD-VALUE",
        "PRIVATE-EXTRACTED-VALUE",
        "private-anthropic-secret",
        "private-evidence-secret",
        "PRIVATE-PROVIDER-REQUEST-ID",
        str(env_file),
        str(diagnostic.CANONICAL_DATASET_ROOT),
        "CI_test.jpg",
        "http://",
        "127.0.0.1",
    ):
        assert forbidden not in serialized


def test_provider_failure_is_reduced_to_safe_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _canonical_fixture(monkeypatch, tmp_path)
    private_detail = "PRIVATE-OCR-TEXT-AND-LOCAL-PATH"
    monkeypatch.setattr(
        diagnostic,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        diagnostic,
        "build_ocr_layout_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OcrLayoutProviderError("ocr_worker_failed", private_detail)
        ),
    )

    exit_code, report = _evaluate(env_file=_env_file(tmp_path))
    serialized = json.dumps(report)

    assert exit_code == 1
    assert report["status"] == "DIAGNOSTIC_EXECUTION_FAILED"
    assert report["phase"] == "EXECUTION_BLOCKED"
    assert report["error_codes"] == ["ocr_worker_failed"]
    assert private_detail not in serialized


def test_claude_failure_is_reduced_to_safe_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _canonical_fixture(monkeypatch, tmp_path)
    private_detail = "PRIVATE-CLAUDE-RESPONSE-AND-REQUEST-ID"
    provider = _FakeProvider()
    monkeypatch.setattr(
        diagnostic,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        diagnostic,
        "build_ocr_layout_provider",
        lambda *_args, **_kwargs: provider,
    )

    def failing_review(**_options: Any) -> ReviewRunResultV2:
        raise LLMProviderError(
            "provider_unavailable",
            private_detail,
            provider_request_id="PRIVATE-PROVIDER-ID",
        )

    monkeypatch.setattr(
        diagnostic,
        "_run_nonrelease_diagnostic_review",
        failing_review,
    )

    exit_code, report = _evaluate(env_file=_env_file(tmp_path))
    serialized = json.dumps(report)

    assert exit_code == 1
    assert report["status"] == "DIAGNOSTIC_EXECUTION_FAILED"
    assert report["error_codes"] == ["provider_unavailable"]
    assert private_detail not in serialized
    assert "PRIVATE-PROVIDER-ID" not in serialized
    assert provider.closed is True


def test_cli_requires_every_mutating_or_network_gate() -> None:
    parser = diagnostic.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(
        [
            "--env-file",
            "private.env",
            "--provider",
            "docling",
            "--ocr-python",
            "python.exe",
            "--evidence-base-url",
            "http://127.0.0.1:8010",
            "--review-date",
            "2026-07-10",
            "--live",
            "--nonrelease-ocr-diagnostic",
        ]
    )
    assert args.live is True
    assert args.nonrelease_ocr_diagnostic is True
    assert not hasattr(args, "dataset_root")


def test_crop_prompt_and_schema_are_supported_without_exposing_crop_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, fixed = _canonical_fixture(monkeypatch, tmp_path)
    result = _result(fixed)
    crop_call = _call(
        document_id=fixed[0].document_id,
        ledger_number=1,
        request_id="PRIVATE-CROP-REQUEST-ID",
    )
    crop_call.input_modality = "image_crop"
    crop_call.prompt_id = diagnostic.DOCUMENT_CROP_RECONCILIATION_PROMPT_ID
    crop_call.prompt_version = diagnostic.DOCUMENT_CROP_RECONCILIATION_PROMPT_VERSION
    crop_call.prompt_sha256 = diagnostic._prompt_sha256(
        diagnostic.load_document_crop_reconciliation_system_prompt
    )
    crop_call.output_schema_id = "customs_claude_crop_read_candidate"
    crop_call.output_schema_sha256 = diagnostic._canonical_sha256(
        ClaudeCropReadCandidate.model_json_schema()
    )
    result.model_calls.append(crop_call)
    captured: dict[str, Any] = {"order": []}
    _patch_live(monkeypatch, result, captured)

    exit_code, report = _evaluate(env_file=_env_file(tmp_path))

    assert exit_code == 0
    assert report["counts"]["claude_crop_call_count"] == 1
    assert "PRIVATE-CROP-REQUEST-ID" not in json.dumps(report)
