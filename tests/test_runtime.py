from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from typing import Any

import pytest
from PIL import Image
from pydantic import SecretStr

import customs_ai_agent
import customs_ai_agent.runtime as runtime_module
from customs_ai_agent.adapters.llm import ClaudeSettings
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.runtime import (
    AgentRuntime,
    RuntimeConfig,
    RuntimeConfigurationError,
    RuntimeExecutionError,
    RuntimeInputError,
)
from customs_ai_agent.schemas.models import (
    ImagePageInput,
    ImagePageProvenance,
    ReviewRunRequestV2,
    ReviewRunResultV2,
)


class _FakeOcrProvider:
    def analyze(self, **_kwargs: Any) -> Any:
        raise AssertionError("OCR must not run in runtime composition tests")


class _FakeReleaseOcrProvider(_FakeOcrProvider):
    def assert_service_ready(self) -> None:
        return None


def test_top_level_package_exports_only_service_runtime_api() -> None:
    assert set(customs_ai_agent.__all__) == {
        "AgentRuntime",
        "AgentRuntimeError",
        "RuntimeConfig",
        "RuntimeConfigurationError",
        "RuntimeExecutionError",
        "RuntimeInputError",
    }
    for internal_name in (
        "run_review",
        "run_review_v2",
        "run_review_result",
        "run_review_result_v2",
    ):
        assert not hasattr(customs_ai_agent, internal_name)


def _settings(secret: str = "unit-test-anthropic-secret") -> ClaudeSettings:
    return ClaudeSettings(api_key=SecretStr(secret))


def _public_result(
    request: ReviewRunRequestV2 | None = None,
) -> ReviewRunResultV2:
    request = request or _text_request()
    documents = [
        {
            "document_id": document.document_id,
            "filename": f"{document.document_id}.txt",
            "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-001",
        }
        for document in request.documents or []
    ]
    return run_review_result_v2(
        ReviewRunRequestV2(
            case_id=request.case_id,
            documents=documents,
            llm_mode="mock",
            review_run_id=request.review_run_id,
            review_as_of=request.review_as_of,
            trade_type=request.trade_type,
            jurisdiction=request.jurisdiction,
        )
    )


def _image_page() -> ImagePageInput:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    return ImagePageInput(
        page_number=1,
        media_type="image/png",
        data_base64=encoded,
        byte_length=len(raw),
        sha256=digest,
        width_px=8,
        height_px=8,
        provenance=ImagePageProvenance(
            source_media_type="image/png",
            source_sha256=digest,
            source_page_number=1,
            transforms=["metadata_strip"],
            original_exif_orientation=1,
        ),
    )


def _text_request() -> ReviewRunRequestV2:
    return ReviewRunRequestV2(
        case_id="case-runtime-text",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-runtime-text",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-001",
            }
        ],
    )


def _visual_request() -> ReviewRunRequestV2:
    return ReviewRunRequestV2(
        case_id="case-runtime-image",
        llm_mode="llm",
        documents=[
            {
                "document_id": "doc-runtime-image",
                "filename": "invoice.png",
                "image_pages": [_image_page()],
            }
        ],
    )


def _patch_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_result: ReviewRunResultV2 | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    retriever = object()
    analyzer = object()

    def fake_retriever(base_url: str, **options: Any) -> object:
        captured["base_url"] = base_url
        captured["retriever_options"] = options
        return retriever

    def fake_analyzer(
        settings: ClaudeSettings,
        ocr_layout_provider: Any = None,
    ) -> object:
        captured["settings"] = settings
        captured["ocr_layout_provider"] = ocr_layout_provider
        return analyzer

    def fake_run(request: ReviewRunRequestV2, **options: Any) -> ReviewRunResultV2:
        captured["request"] = request
        captured["run_options"] = options
        return expected_result or _public_result(request)

    monkeypatch.setattr(runtime_module, "HttpRegulationRetriever", fake_retriever)
    monkeypatch.setattr(
        runtime_module.ClaudeDocumentAnalyzer,
        "from_settings",
        fake_analyzer,
    )
    monkeypatch.setattr(runtime_module, "run_review_result_v2", fake_run)
    captured["retriever"] = retriever
    captured["analyzer"] = analyzer
    return captured


def test_runtime_config_and_composition_wire_claude_ocr_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "ReleaseLockedOcrProvider",
        _FakeReleaseOcrProvider,
    )
    provider = _FakeReleaseOcrProvider()
    settings = _settings()
    token = SecretStr("unit-test-evidence-secret")
    config = RuntimeConfig(
        claude_settings=settings,
        evidence_base_url="http://127.0.0.1:8010/",
        ocr_layout_provider=provider,
        evidence_service_token=token,
        evidence_token_header="X-Agent-Token",
        evidence_timeout_seconds=12.0,
        evidence_max_retries=2,
    )

    agent = AgentRuntime(config)
    result = agent.review(_text_request())

    assert result.schema_version == "2.0"
    assert config.evidence_base_url == "http://127.0.0.1:8010"
    assert captured["base_url"] == "http://127.0.0.1:8010"
    assert captured["settings"] is settings
    assert captured["ocr_layout_provider"] is provider
    assert captured["retriever_options"] == {
        "service_token": "unit-test-evidence-secret",
        "token_header": "X-Agent-Token",
        "allowed_origins": frozenset(),
        "timeout_seconds": 12.0,
        "max_retries": 2,
        "contract_identity_mode": "required",
    }
    assert captured["run_options"] == {
        "retriever": captured["retriever"],
        "analyzer": captured["analyzer"],
    }
    rendered_config = repr(config)
    assert "unit-test-anthropic-secret" not in rendered_config
    assert "unit-test-evidence-secret" not in rendered_config


def test_runtime_config_rejects_raw_ocr_provider_for_service_use() -> None:
    with pytest.raises(
        RuntimeConfigurationError,
        match="ocr_release_gate_required",
    ):
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
            ocr_layout_provider=_FakeOcrProvider(),
        )


def test_runtime_contract_identity_handshake_is_required_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)

    AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    assert captured["retriever_options"]["contract_identity_mode"] == "required"


def test_service_runtime_rejects_optional_contract_identity_mode() -> None:
    with pytest.raises(
        RuntimeConfigurationError,
        match="evidence_contract_identity_mode_invalid",
    ):
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
            evidence_contract_identity_mode="optional",  # type: ignore[arg-type]
        )


def test_runtime_rejects_disabled_or_unknown_contract_identity_mode() -> None:
    with pytest.raises(
        RuntimeConfigurationError,
        match="evidence_contract_identity_mode_invalid",
    ):
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
            evidence_contract_identity_mode="disabled",  # type: ignore[arg-type]
        )


def test_visual_review_without_ocr_provider_fails_before_workflow_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(
        RuntimeConfigurationError,
        match="ocr_layout_provider_required",
    ):
        agent.review(_visual_request())

    assert "request" not in captured


def test_service_runtime_rejects_mock_mode_without_running_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeInputError, match="real_llm_mode_required"):
        agent.review(_text_request().model_copy(update={"llm_mode": "mock"}))

    assert "request" not in captured


@pytest.mark.parametrize(
    "service_input",
    [
        ReviewRunRequestV2(
            scenario_id="happy_path",
            llm_mode="llm",
        ),
        ReviewRunRequestV2(
            documents=[
                {
                    "document_id": "doc-without-case",
                    "filename": "invoice.txt",
                    "content_text": "COMMERCIAL INVOICE",
                }
            ],
            llm_mode="llm",
        ),
    ],
)
def test_service_runtime_requires_explicit_case_and_documents(
    monkeypatch: pytest.MonkeyPatch,
    service_input: ReviewRunRequestV2,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(
        RuntimeInputError,
        match="service_request_identity_required",
    ):
        agent.review(service_input)

    assert "request" not in captured


def test_runtime_rejects_result_from_different_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _text_request()
    payload = _public_result(request).model_dump(mode="python")
    payload["case_id"] = "case-other"
    payload["report"]["case_id"] = "case-other"
    result = ReviewRunResultV2.model_validate(payload)
    _patch_runtime_dependencies(monkeypatch, expected_result=result)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(
        RuntimeExecutionError,
        match="review_result_correlation_failed",
    ):
        agent.review(request)


def test_runtime_rejects_result_document_set_from_different_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _text_request()
    other_request = request.model_copy(
        update={"documents": [request.documents[0].model_copy(update={"document_id": "doc-other"})]}
    )
    result = _public_result(other_request)
    _patch_runtime_dependencies(monkeypatch, expected_result=result)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(
        RuntimeExecutionError,
        match="review_result_correlation_failed",
    ):
        agent.review(request)


def test_invalid_dict_error_does_not_echo_raw_base64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    marker = "private-base64-marker-that-must-not-escape"

    with pytest.raises(RuntimeInputError) as captured:
        agent.review(
            {
                "case_id": "case-invalid",
                "llm_mode": "llm",
                "documents": [
                    {
                        "document_id": "doc-invalid",
                        "filename": "invalid.png",
                        "image_pages": [{"data_base64": marker}],
                    }
                ],
            }
        )

    assert marker not in str(captured.value)


def test_runtime_revalidates_bypassed_request_model_before_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    bypassed = ReviewRunRequestV2.model_construct(
        schema_version="2.0",
        case_id="case-bypassed",
        llm_mode="llm",
        documents=[{"document_id": "invalid"}],
    )

    with pytest.raises(RuntimeInputError, match="review_request_invalid"):
        agent.review(bypassed)

    assert "request" not in captured


def test_runtime_revalidates_mutated_result_before_public_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _public_result(_text_request())
    object.__setattr__(result, "schema_version", "invalid")
    _patch_runtime_dependencies(monkeypatch, expected_result=result)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeExecutionError, match="review_result_invalid"):
        agent.review(_text_request())


@pytest.mark.parametrize(
    "marker_kind",
    ["claude_secret", "evidence_secret"],
)
def test_runtime_blocks_sensitive_markers_in_public_result(
    monkeypatch: pytest.MonkeyPatch,
    marker_kind: str,
) -> None:
    request = _text_request()
    result = _public_result(request)
    claude_secret = "unit-test-anthropic-secret"
    evidence_secret = "unit-test-evidence-secret"
    marker = {
        "claude_secret": claude_secret,
        "evidence_secret": evidence_secret,
    }[marker_kind]
    result.document_results[0].fields[0].value = marker
    _patch_runtime_dependencies(monkeypatch, expected_result=result)
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(claude_secret),
            evidence_base_url="http://127.0.0.1:8010",
            evidence_service_token=SecretStr(evidence_secret),
        )
    )

    with pytest.raises(
        RuntimeExecutionError,
        match="public_result_safety_violation",
    ) as captured:
        agent.review(request)

    assert marker not in str(captured.value)


def test_public_safety_blocks_raw_image_marker() -> None:
    marker = _image_page().data_base64

    with pytest.raises(
        RuntimeExecutionError,
        match="public_result_safety_violation",
    ):
        runtime_module._assert_public_result_safe(
            {"value": marker},
            sensitive_markers=(marker,),
        )


def test_unknown_execution_exception_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-exception-detail"
    _patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "run_review_result_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    agent = AgentRuntime(
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeExecutionError, match="review_execution_failed") as captured:
        agent.review(_text_request())

    assert secret not in str(captured.value)


def test_invalid_external_evidence_url_error_does_not_echo_credentials() -> None:
    secret_url = "https://user:private-password@example.invalid"

    with pytest.raises(RuntimeConfigurationError) as captured:
        RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url=secret_url,
        )

    assert secret_url not in str(captured.value)
    assert "private-password" not in str(captured.value)
