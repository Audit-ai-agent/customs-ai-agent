from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from typing import Any

import pytest
from PIL import Image
from pydantic import SecretStr, ValidationError

import customs_ai_agent
import customs_ai_agent.runtime as runtime_module
import customs_ai_agent.schemas.models as models_module
from customs_ai_agent.adapters.llm import ClaudeSettings, LLMError
from customs_ai_agent.adapters.ocr import OcrLayoutProviderError
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.runtime import (
    AgentRuntime,
    RuntimeConfig,
    RuntimeConfigurationError,
    RuntimeExecutionError,
    RuntimeInputError,
)
from customs_ai_agent.schemas.models import (
    DocumentInputV2,
    ImagePageInput,
    ImagePageProvenance,
    ReviewRunRequestV2,
    ReviewRunResultV2,
    ServiceReviewRunRequestV2,
)


class _FakeOcrProvider:
    def analyze(self, **_kwargs: Any) -> Any:
        raise AssertionError("OCR must not run in runtime composition tests")


class _FakeReleaseOcrProvider(_FakeOcrProvider):
    def __init__(self) -> None:
        self.close_calls = 0

    def assert_service_ready(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1


class _FakeOwnedAnalyzer:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeProductionRetriever:
    def __init__(self) -> None:
        self.preflight_calls: list[Any] = []

    def preflight(self, review_as_of: Any = None) -> object:
        self.preflight_calls.append(review_as_of)
        return object()


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


def _runtime_config(**options: Any) -> RuntimeConfig:
    options.setdefault(
        "evidence_service_token",
        SecretStr("Abcdefghijklmnopqrstuvwxyz0123456789"),
    )
    return runtime_module.RuntimeConfig(**options)


def _public_result(
    request: ServiceReviewRunRequestV2 | ReviewRunRequestV2 | None = None,
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


def _text_request() -> ServiceReviewRunRequestV2:
    return ServiceReviewRunRequestV2(
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


def _service_text_payload() -> dict[str, Any]:
    return {
        "case_id": "case-runtime-service-text",
        "documents": [
            {
                "document_id": "doc-runtime-service-text",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-001",
            }
        ],
        "llm_mode": "llm",
    }


def _service_text_payload_with_document_codes(
    codes: list[str],
) -> dict[str, Any]:
    payload = _service_text_payload()
    payload["documents"][0]["requirement_document_codes"] = codes
    return payload


def _visual_request() -> ServiceReviewRunRequestV2:
    return ServiceReviewRunRequestV2(
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


def _raw_image_page(*, byte_length: int = 3, encoded: str = "AAAA") -> dict[str, Any]:
    return {
        "page_number": 1,
        "media_type": "image/png",
        "data_base64": encoded,
        "byte_length": byte_length,
        "sha256": "a" * 64,
        "width_px": 1,
        "height_px": 1,
        "provenance": {
            "source_media_type": "image/png",
            "source_sha256": "a" * 64,
            "source_page_number": 1,
        },
    }


def _raw_image_document(index: int, page: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": f"doc-image-{index}",
        "filename": f"invoice-{index}.png",
        "image_pages": [page],
    }


def _patch_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_result: ReviewRunResultV2 | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    retriever = _FakeProductionRetriever()
    analyzer = _FakeOwnedAnalyzer()

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

    monkeypatch.setattr(
        runtime_module,
        "ProductionHttpRegulationRetriever",
        fake_retriever,
    )
    monkeypatch.setattr(
        runtime_module,
        "load_released_evidence_runtime_identity",
        lambda: object(),
    )
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
    token = SecretStr("Abcdefghijklmnopqrstuvwxyz0123456789")
    config = _runtime_config(
        claude_settings=settings,
        evidence_base_url="http://127.0.0.1:8010/",
        ocr_layout_provider=provider,
        evidence_service_token=token,
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
        "service_token": "Abcdefghijklmnopqrstuvwxyz0123456789",
        "allowed_origins": frozenset(),
        "timeout_seconds": 12.0,
        "max_retries": 2,
    }
    assert captured["run_options"] == {
        "retriever": captured["retriever"],
        "analyzer": captured["analyzer"],
    }
    rendered_config = repr(config)
    assert "unit-test-anthropic-secret" not in rendered_config
    assert "Abcdefghijklmnopqrstuvwxyz0123456789" not in rendered_config


def test_unreleased_production_lock_prevents_claude_client_and_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def no_analyzer(*_args: Any, **_kwargs: Any) -> object:
        calls.append("claude")
        raise AssertionError

    def no_retriever(*_args: Any, **_kwargs: Any) -> object:
        calls.append("network")
        raise AssertionError

    monkeypatch.setattr(
        runtime_module.ClaudeDocumentAnalyzer,
        "from_settings",
        no_analyzer,
    )
    monkeypatch.setattr(
        runtime_module,
        "ProductionHttpRegulationRetriever",
        no_retriever,
    )
    with pytest.raises(
        RuntimeConfigurationError,
        match="evidence_runtime_release_unreleased",
    ):
        AgentRuntime(
            _runtime_config(
                claude_settings=_settings(),
                evidence_base_url="http://127.0.0.1:8010",
            )
        )
    assert calls == []


def test_missing_evidence_bearer_fails_before_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.ClaudeDocumentAnalyzer,
        "from_settings",
        lambda *_args, **_kwargs: pytest.fail("Claude client must not be built"),
    )
    with pytest.raises(
        RuntimeConfigurationError,
        match="evidence_service_token_required",
    ):
        runtime_module.RuntimeConfig(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )


def test_data_preflight_failure_prevents_any_claude_workflow_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)

    def unready(_review_as_of: Any = None) -> object:
        raise RetrievalError("private data detail")

    captured["retriever"].preflight = unready
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    with pytest.raises(RuntimeExecutionError, match="review_execution_failed"):
        agent.review(_text_request())
    assert "request" not in captured


def test_internal_v2_request_preserves_scenario_and_mock_compatibility() -> None:
    scenario_request = ReviewRunRequestV2(scenario_id="happy_path")
    document_request = ReviewRunRequestV2(
        documents=[
            {
                "document_id": "doc-internal-mock",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE",
            }
        ],
        llm_mode="mock",
    )

    assert scenario_request.scenario_id == "happy_path"
    assert scenario_request.llm_mode == "mock"
    assert document_request.documents is not None
    assert document_request.llm_mode == "mock"


def test_service_runtime_rejects_internal_workflow_request_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    internal_request = ReviewRunRequestV2(
        documents=[
            {
                "document_id": "doc-internal-real",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE",
            }
        ],
        case_id="case-internal-real",
        llm_mode="llm",
    )

    with pytest.raises(RuntimeInputError, match="review_request_invalid"):
        agent.review(internal_request)  # type: ignore[arg-type]

    assert "request" not in captured


def test_schema_valid_service_text_payload_reaches_internal_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _service_text_payload()
    service_request = ServiceReviewRunRequestV2.model_validate(payload)
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    result = agent.review(payload)

    assert result.case_id == service_request.case_id
    assert isinstance(captured["request"], ReviewRunRequestV2)
    assert captured["request"].scenario_id is None
    assert captured["request"].case_id == service_request.case_id
    assert captured["request"].llm_mode == "llm"


def test_service_request_rejects_aggregate_page_count_before_nested_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models_module.ImagePageInput,
        "decoded_bytes",
        lambda _self: pytest.fail("nested image decode must not run"),
    )
    payload = {
        "case_id": "case-aggregate-pages",
        "llm_mode": "llm",
        "documents": [_raw_image_document(index, _raw_image_page()) for index in range(11)],
    }

    with pytest.raises(
        ValidationError,
        match="combined image pages",
    ) as captured:
        ServiceReviewRunRequestV2.model_validate(payload)

    assert captured.value.errors()[0]["loc"] == ()


def test_service_request_rejects_aggregate_base64_before_nested_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models_module.ImagePageInput,
        "decoded_bytes",
        lambda _self: pytest.fail("nested image decode must not run"),
    )
    monkeypatch.setattr(models_module, "MAX_IMAGE_BYTES_PER_RUN", 100)
    monkeypatch.setattr(models_module, "MAX_IMAGE_BASE64_CHARS_PER_RUN", 10)
    payload = {
        "case_id": "case-aggregate-base64",
        "llm_mode": "llm",
        "documents": [_raw_image_document(index, _raw_image_page()) for index in range(3)],
    }

    with pytest.raises(
        ValidationError,
        match="combined image base64",
    ) as captured:
        ServiceReviewRunRequestV2.model_validate(payload)

    assert captured.value.errors()[0]["loc"] == ()


def test_service_request_rejects_aggregate_declared_bytes_before_nested_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models_module.ImagePageInput,
        "decoded_bytes",
        lambda _self: pytest.fail("nested image decode must not run"),
    )
    monkeypatch.setattr(models_module, "MAX_IMAGE_BYTES_PER_RUN", 8)
    monkeypatch.setattr(models_module, "MAX_IMAGE_BASE64_CHARS_PER_RUN", 100)
    payload = {
        "case_id": "case-aggregate-bytes",
        "llm_mode": "llm",
        "documents": [
            _raw_image_document(index, _raw_image_page(byte_length=5, encoded="AAAAAAAA"))
            for index in range(2)
        ],
    }

    with pytest.raises(
        ValidationError,
        match="combined image bytes",
    ) as captured:
        ServiceReviewRunRequestV2.model_validate(payload)

    assert captured.value.errors()[0]["loc"] == ()


def test_service_request_rejects_aggregate_text_before_nested_image_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        models_module.ImagePageInput,
        "decoded_bytes",
        lambda _self: pytest.fail("nested image decode must not run"),
    )
    documents = [_raw_image_document(0, _raw_image_page())]
    documents.extend(
        {
            "document_id": f"doc-text-{index}",
            "filename": f"invoice-{index}.txt",
            "content_text": "X" * 200_000,
        }
        for index in range(3)
    )

    with pytest.raises(
        ValidationError,
        match="combined document text",
    ) as captured:
        ServiceReviewRunRequestV2.model_validate(
            {
                "case_id": "case-aggregate-text",
                "llm_mode": "llm",
                "documents": documents,
            }
        )

    assert captured.value.errors()[0]["loc"] == ()


def test_review_json_rejects_duplicate_keys_without_running_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeInputError, match="review_request_invalid") as error:
        agent.review_json(
            b'{"case_id":"case-a","case_id":"case-b","documents":[],"llm_mode":"llm"}'
        )

    assert error.value.__context__ is None
    assert "request" not in captured


def test_review_json_valid_service_payload_reaches_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    result = agent.review_json(
        json.dumps(
            _service_text_payload(),
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    )

    assert result.case_id == "case-runtime-service-text"
    assert captured["request"].case_id == result.case_id


@pytest.mark.parametrize(
    "serialized",
    [
        b'{"case_id":NaN}',
        ('{"case_id":' + ("1" * 5_000) + "}").encode(),
    ],
)
def test_review_json_normalizes_noncanonical_json_values(
    monkeypatch: pytest.MonkeyPatch,
    serialized: bytes,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeInputError) as error:
        agent.review_json(serialized)

    assert error.value.code == "review_request_invalid"
    assert "request" not in captured


def test_review_json_rejects_oversized_body_before_json_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(runtime_module, "_MAX_SERVICE_REQUEST_JSON_BYTES", 16)
    monkeypatch.setattr(
        runtime_module.json,
        "loads",
        lambda *_args, **_kwargs: pytest.fail("JSON decode must not run"),
    )
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeInputError, match="review_request_too_large"):
        agent.review_json(b"{" + (b" " * 16) + b"}")

    assert "request" not in captured


def test_review_json_counts_memoryview_bytes_not_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(runtime_module, "_MAX_SERVICE_REQUEST_JSON_BYTES", 16)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    wide_view = memoryview(bytearray(20)).cast("I")
    assert len(wide_view) == 5
    assert wide_view.nbytes == 20

    with pytest.raises(RuntimeInputError) as error:
        agent.review_json(wide_view)

    assert error.value.code == "review_request_too_large"
    assert "request" not in captured


def test_review_json_normalizes_released_memoryview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    released = memoryview(b"{}")
    released.release()

    with pytest.raises(RuntimeInputError) as error:
        agent.review_json(released)

    assert error.value.code == "review_request_invalid"
    assert "request" not in captured


def test_review_json_rejects_bytes_subclass_with_untrusted_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LyingBytes(bytes):
        def __len__(self) -> int:
            return 0

    captured = _patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(runtime_module, "_MAX_SERVICE_REQUEST_JSON_BYTES", 4)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeInputError) as error:
        agent.review_json(LyingBytes(b"{}      "))

    assert error.value.code == "review_request_invalid"
    assert "request" not in captured


def test_runtime_rejects_nested_constructed_models_before_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    bypassed_page = ImagePageInput.model_construct(
        page_number=1,
        media_type="image/png",
        data_base64="private-unvalidated-base64",
        byte_length=1,
        sha256="a" * 64,
        width_px=1,
        height_px=1,
        provenance={
            "source_media_type": "image/png",
            "source_sha256": "a" * 64,
            "source_page_number": 1,
        },
    )
    bypassed_document = DocumentInputV2.model_construct(
        document_id="doc-bypassed",
        filename=r"C:\private\secret.png",
        image_pages=[bypassed_page],
        requirement_document_codes=[],
    )
    payload = {
        "case_id": "case-bypassed-nested",
        "documents": [bypassed_document],
        "llm_mode": "llm",
    }

    with pytest.raises(RuntimeInputError, match="review_request_invalid"):
        agent.review(payload)
    with pytest.raises(ValidationError):
        ServiceReviewRunRequestV2.model_validate(payload)

    assert "request" not in captured


def test_runtime_rejects_custom_mapping_subclass_before_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomDict(dict[str, Any]):
        pass

    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    payload = CustomDict(_service_text_payload())

    with pytest.raises(RuntimeInputError, match="review_request_invalid"):
        agent.review(payload)

    assert "request" not in captured


def test_runtime_normalizes_constructed_model_serialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    circular_documents: list[Any] = []
    circular_documents.append(circular_documents)
    bypassed = ServiceReviewRunRequestV2.model_construct(
        case_id="case-circular",
        documents=circular_documents,
        llm_mode="llm",
    )

    with pytest.raises(RuntimeInputError) as error:
        agent.review(bypassed)

    assert error.value.code == "review_request_invalid"
    assert error.value.__cause__ is None
    assert "request" not in captured


@pytest.mark.parametrize(
    "review_as_of",
    [
        "2026-01-01T00:00:00",
        0,
    ],
)
def test_runtime_applies_json_schema_date_format_before_pydantic_coercion(
    monkeypatch: pytest.MonkeyPatch,
    review_as_of: Any,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    payload = _service_text_payload()
    payload["review_as_of"] = review_as_of

    with pytest.raises(RuntimeInputError, match="review_request_invalid"):
        agent.review(payload)

    assert "request" not in captured


@pytest.mark.parametrize(
    "field_name",
    [
        "page_number",
        "byte_length",
        "width_px",
        "height_px",
    ],
)
def test_runtime_rejects_image_numeric_strings_before_pydantic_coercion(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    page = _image_page().model_dump(mode="json")
    page[field_name] = str(page[field_name])
    payload = {
        "case_id": "case-numeric-string",
        "llm_mode": "llm",
        "documents": [
            {
                "document_id": "doc-numeric-string",
                "filename": "invoice.png",
                "image_pages": [page],
            }
        ],
    }

    with pytest.raises(RuntimeInputError, match="review_request_invalid"):
        agent.review(payload)

    assert "request" not in captured


def test_runtime_config_rejects_raw_ocr_provider_for_service_use() -> None:
    with pytest.raises(
        RuntimeConfigurationError,
        match="ocr_release_gate_required",
    ):
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
            ocr_layout_provider=_FakeOcrProvider(),
        )


def test_runtime_uses_only_production_v2_release_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)

    AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    assert captured["retriever_options"] == {
        "service_token": "Abcdefghijklmnopqrstuvwxyz0123456789",
        "allowed_origins": frozenset(),
        "timeout_seconds": 60.0,
        "max_retries": 1,
    }


def test_runtime_context_closes_owned_anthropic_analyzer_but_not_injected_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr(
        runtime_module,
        "ReleaseLockedOcrProvider",
        _FakeReleaseOcrProvider,
    )
    provider = _FakeReleaseOcrProvider()
    config = _runtime_config(
        claude_settings=_settings(),
        evidence_base_url="http://127.0.0.1:8010",
        ocr_layout_provider=provider,
    )

    with AgentRuntime(config) as runtime:
        assert runtime.review(_text_request()).schema_version == "2.0"

    analyzer = captured["analyzer"]
    assert analyzer.close_calls == 1
    assert provider.close_calls == 0

    runtime.close()
    assert analyzer.close_calls == 1
    assert provider.close_calls == 0
    with pytest.raises(RuntimeExecutionError) as exc_info:
        runtime.review(_text_request())
    assert exc_info.value.code == "runtime_closed"


def test_runtime_close_failure_is_content_free_and_stays_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    private_detail = "private client close detail"

    def fail_close() -> None:
        raise RuntimeError(private_detail)

    captured["analyzer"].close = fail_close
    runtime = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeExecutionError) as captured_error:
        runtime.close()

    assert captured_error.value.code == "runtime_close_failed"
    assert private_detail not in str(captured_error.value)
    runtime.close()
    with pytest.raises(RuntimeExecutionError, match="runtime_closed"):
        runtime.review_json(b"{}")


def test_service_runtime_has_no_optional_v1_contract_identity_mode() -> None:
    with pytest.raises(TypeError, match="evidence_contract_identity_mode"):
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
            evidence_contract_identity_mode="optional",  # type: ignore[arg-type]
        )


def test_service_runtime_has_no_disabled_v1_contract_identity_mode() -> None:
    with pytest.raises(TypeError, match="evidence_contract_identity_mode"):
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
            evidence_contract_identity_mode="disabled",  # type: ignore[arg-type]
        )


def test_visual_review_without_ocr_provider_fails_before_workflow_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
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
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    invalid = _service_text_payload()
    invalid["llm_mode"] = "mock"
    with pytest.raises(RuntimeInputError, match="review_request_invalid"):
        agent.review(invalid)

    assert "request" not in captured


@pytest.mark.parametrize(
    "service_input",
    [
        {
            "scenario_id": "happy_path",
            "llm_mode": "llm",
        },
        {
            "documents": [
                {
                    "document_id": "doc-without-case",
                    "filename": "invoice.txt",
                    "content_text": "COMMERCIAL INVOICE",
                }
            ],
            "llm_mode": "llm",
        },
        {
            **_service_text_payload(),
            "scenario_id": None,
        },
        {
            **_service_text_payload(),
            "unknown_policy": "allow",
        },
        {
            **_service_text_payload(),
            "uploaded_requirement_document_codes": [f"REQ-{index}" for index in range(101)],
        },
        _service_text_payload_with_document_codes(["REQ VALID WITH SPACES"]),
    ],
)
def test_service_runtime_rejects_invalid_policy_before_workflow(
    monkeypatch: pytest.MonkeyPatch,
    service_input: dict[str, Any],
) -> None:
    captured = _patch_runtime_dependencies(monkeypatch)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(
        RuntimeInputError,
        match="review_request_invalid",
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
        _runtime_config(
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
        _runtime_config(
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
        _runtime_config(
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
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )
    bypassed = ServiceReviewRunRequestV2.model_construct(
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
        _runtime_config(
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
    evidence_secret = "Abcdefghijklmnopqrstuvwxyz0123456789"
    marker = {
        "claude_secret": claude_secret,
        "evidence_secret": evidence_secret,
    }[marker_kind]
    result.document_results[0].fields[0].value = marker
    _patch_runtime_dependencies(monkeypatch, expected_result=result)
    agent = AgentRuntime(
        _runtime_config(
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


@pytest.mark.parametrize("position", ["prefix", "middle", "suffix"])
def test_public_safety_blocks_substantial_raw_image_fragments(position: str) -> None:
    marker = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/" * 700)[:40_000]
    starts = {
        "prefix": 0,
        "middle": (len(marker) - 500) // 2,
        "suffix": len(marker) - 500,
    }
    fragment = marker[starts[position] : starts[position] + 500]

    with pytest.raises(
        RuntimeExecutionError,
        match="public_result_safety_violation",
    ):
        runtime_module._assert_public_result_safe(
            {"value": f"model-output:{fragment}:end"},
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
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeExecutionError, match="review_execution_failed") as captured:
        agent.review(_text_request())

    assert secret not in str(captured.value)
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    "provider_error",
    [
        LLMError("provider_unavailable", "private-provider-detail"),
        OcrLayoutProviderError("ocr_worker_failed", "private-provider-detail"),
        RetrievalError("private-provider-detail"),
    ],
)
def test_runtime_normalizes_provider_errors_at_public_boundary(
    monkeypatch: pytest.MonkeyPatch,
    provider_error: Exception,
) -> None:
    _patch_runtime_dependencies(monkeypatch)

    def fail_run(*_args: Any, **_kwargs: Any) -> ReviewRunResultV2:
        raise provider_error

    monkeypatch.setattr(runtime_module, "run_review_result_v2", fail_run)
    agent = AgentRuntime(
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url="http://127.0.0.1:8010",
        )
    )

    with pytest.raises(RuntimeExecutionError) as captured:
        agent.review(_text_request())

    assert captured.value.code == "review_execution_failed"
    assert str(captured.value) == (
        "review_execution_failed: "
        "The Agent review failed without exposing private execution details."
    )
    assert "private-provider-detail" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert captured.value.__suppress_context__ is False


def test_public_safety_blocks_camel_case_sensitive_key() -> None:
    with pytest.raises(
        RuntimeExecutionError,
        match="public_result_safety_violation",
    ):
        runtime_module._assert_public_result_safe(
            {"dataBase64": "private-image-content"},
            sensitive_markers=(),
        )


def test_invalid_external_evidence_url_error_does_not_echo_credentials() -> None:
    secret_url = "https://user:private-password@example.invalid"

    with pytest.raises(RuntimeConfigurationError) as captured:
        _runtime_config(
            claude_settings=_settings(),
            evidence_base_url=secret_url,
        )

    assert secret_url not in str(captured.value)
    assert "private-password" not in str(captured.value)
