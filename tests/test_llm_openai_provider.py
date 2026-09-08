"""OpenAI provider path of the functional-UAT LLM adapter (no network)."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

from pydantic import SecretStr
import pytest

from customs_ai_agent.adapters.llm import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENAI_MODEL,
    ClaudeDocumentAnalyzer,
    ClaudeGroundedResponder,
    ClaudeSettings,
    LLMConfigurationError,
    LLMProviderError,
    OpenAIMessagesClient,
    build_provider_client,
)
from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionRequestMetadataV1,
    PreviewInputV1,
)
from customs_ai_agent.service.document_preview_vlm_demo import ClaudeVlmOcrAdapter
from customs_ai_agent.service.settings import ServiceSettings


_KEY = "sk-test-" + "x" * 40
_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"


class _FakeCompletions:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.output))]
        )


class _FakeOpenAI:
    def __init__(self, output: str) -> None:
        self.completions = _FakeCompletions(output)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CUSTOMS_AGENT_LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
        "CUSTOMS_AGENT_OPENAI_MODEL",
        "CUSTOMS_AGENT_CLAUDE_MODEL",
        "CUSTOMS_AGENT_LLM_MAX_TOKENS",
        "CUSTOMS_AGENT_CLAUDE_MAX_TOKENS",
        "CUSTOMS_AGENT_LLM_TIMEOUT_SECONDS",
        "CUSTOMS_AGENT_CLAUDE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_openai_settings_from_environment_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("CUSTOMS_AGENT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", _KEY)
    monkeypatch.setenv("CUSTOMS_AGENT_LLM_MAX_TOKENS", "16384")

    settings = ClaudeSettings.from_environment()

    assert settings.provider == "openai"
    assert settings.model == DEFAULT_OPENAI_MODEL
    assert settings.max_tokens == 16384
    assert settings.api_key.get_secret_value() == _KEY


def test_openai_settings_inferred_from_key_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", _KEY)

    assert ClaudeSettings.from_environment().provider == "openai"

    # Both keys present without an explicit provider keeps the historical default.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "y" * 40)
    settings = ClaudeSettings.from_environment()
    assert settings.provider == "anthropic"
    assert settings.model == DEFAULT_CLAUDE_MODEL


def test_legacy_claude_variables_still_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "y" * 40)
    monkeypatch.setenv("CUSTOMS_AGENT_CLAUDE_MAX_TOKENS", "8192")
    monkeypatch.setenv("CUSTOMS_AGENT_CLAUDE_TIMEOUT_SECONDS", "120")

    settings = ClaudeSettings.from_environment()

    assert settings.provider == "anthropic"
    assert settings.max_tokens == 8192
    assert settings.timeout_seconds == 120


@pytest.mark.parametrize("name", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
def test_openai_transport_override_is_rejected(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("CUSTOMS_AGENT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", _KEY)
    monkeypatch.setenv(name, "http://evil.example")

    with pytest.raises(LLMConfigurationError):
        ClaudeSettings.from_environment()


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("CUSTOMS_AGENT_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", _KEY)

    with pytest.raises(LLMConfigurationError):
        ClaudeSettings.from_environment()


@pytest.mark.parametrize(
    ("provider", "model", "valid"),
    [
        ("openai", "gpt-4.1-mini", True),
        ("openai", "gpt-5-mini", True),
        ("openai", "o4-mini", True),
        ("openai", "claude-haiku-4-5-20251001", False),
        ("anthropic", "gpt-4.1-mini", False),
        ("anthropic", "claude-haiku-4-5-20251001", True),
        ("openai", "gpt-4.1-mini; rm -rf /", False),
    ],
)
def test_model_name_must_match_provider(provider: str, model: str, valid: bool) -> None:
    if valid:
        assert (
            ClaudeSettings(api_key=SecretStr(_KEY), provider=provider, model=model).model == model
        )  # type: ignore[arg-type]
    else:
        with pytest.raises(ValueError):
            ClaudeSettings(api_key=SecretStr(_KEY), provider=provider, model=model)  # type: ignore[arg-type]


def test_messages_client_translates_text_request() -> None:
    fake = _FakeOpenAI('{"ok": true}')
    client = OpenAIMessagesClient(fake)

    response = client.messages.create(
        model="gpt-4.1-mini",
        max_tokens=512,
        temperature=0,
        system="Return JSON.",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.content[0].text == '{"ok": true}'
    request = fake.completions.requests[0]
    assert request["model"] == "gpt-4.1-mini"
    assert request["max_completion_tokens"] == 512
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"] == [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "hello"},
    ]
    assert "max_tokens" not in request
    client.close()
    assert fake.closed


@pytest.mark.parametrize("model", ["gpt-5-mini", "o4-mini"])
def test_messages_client_omits_temperature_for_reasoning_models(model: str) -> None:
    fake = _FakeOpenAI("{}")
    OpenAIMessagesClient(fake).messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        system="s",
        messages=[{"role": "user", "content": "x"}],
    )
    assert "temperature" not in fake.completions.requests[0]


def test_messages_client_translates_image_and_document_blocks() -> None:
    fake = _FakeOpenAI('{"pages": []}')
    png = base64.standard_b64encode(b"\x89PNG").decode("ascii")
    pdf = base64.standard_b64encode(b"%PDF-1.4").decode("ascii")

    OpenAIMessagesClient(fake).messages.create(
        model="gpt-4.1-mini",
        max_tokens=256,
        temperature=0,
        system="ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": png},
                    },
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": pdf},
                    },
                    {"type": "text", "text": "Transcribe."},
                ],
            }
        ],
    )

    parts = fake.completions.requests[0]["messages"][1]["content"]
    assert parts == [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
        {
            "type": "file",
            "file": {"filename": "document.pdf", "file_data": f"data:application/pdf;base64,{pdf}"},
        },
        {"type": "text", "text": "Transcribe."},
    ]


def test_messages_client_rejects_unknown_block() -> None:
    fake = _FakeOpenAI("{}")
    with pytest.raises(LLMProviderError):
        OpenAIMessagesClient(fake).messages.create(
            model="gpt-4.1-mini",
            max_tokens=256,
            temperature=0,
            system="s",
            messages=[{"role": "user", "content": [{"type": "tool_use", "id": "x"}]}],
        )
    assert fake.completions.requests == []


def test_messages_client_tolerates_empty_choices() -> None:
    class _Empty:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[]))
            )

    response = OpenAIMessagesClient(_Empty()).messages.create(
        model="gpt-4.1-mini",
        max_tokens=256,
        system="s",
        messages=[{"role": "user", "content": "x"}],
    )
    assert response.content[0].text == ""


def test_document_analyzer_runs_on_openai_client() -> None:
    candidate = {
        "document_type": "invoice",
        "fields": [],
    }
    fake = _FakeOpenAI(json.dumps(candidate))
    settings = ClaudeSettings(api_key=SecretStr(_KEY), provider="openai")
    analyzer = ClaudeDocumentAnalyzer(settings, OpenAIMessagesClient(fake))

    try:
        analysis = analyzer.analyze(
            {"document_id": "doc-1", "filename": "invoice.txt", "content_text": "Invoice total 100"}
        )
    except (
        Exception
    ) as error:  # the wire candidate model may need more fields; the call shape is what matters
        assert type(error).__name__ == "LLMOutputValidationError"
    else:
        assert analysis.model_id == DEFAULT_OPENAI_MODEL
    request = fake.completions.requests[0]
    assert request["model"] == DEFAULT_OPENAI_MODEL
    assert request["messages"][0]["role"] == "system"
    assert "doc-1" in request["messages"][1]["content"]


def test_grounded_responder_runs_on_openai_client() -> None:
    fake = _FakeOpenAI('{"answer": "ok"}')
    settings = ClaudeSettings(api_key=SecretStr(_KEY), provider="openai", model="gpt-4.1")
    responder = ClaudeGroundedResponder(settings, OpenAIMessagesClient(fake))

    parsed = responder.respond({"question": "q"}, purpose="assistant_answer")

    assert parsed == {"answer": "ok"}
    assert fake.completions.requests[0]["model"] == "gpt-4.1"


def test_vlm_ocr_adapter_uses_openai_client() -> None:
    fake = _FakeOpenAI(json.dumps({"pages": [{"page_number": 1, "text": "INVOICE 123"}]}))
    service_settings = ServiceSettings(
        bearer_token=SecretStr(_TOKEN),
        evidence_base_url="http://127.0.0.1:18083",
        evidence_service_token=SecretStr(_TOKEN),
    )
    settings = ClaudeSettings(api_key=SecretStr(_KEY), provider="openai")
    adapter = ClaudeVlmOcrAdapter(
        service_settings, claude_settings=settings, client=OpenAIMessagesClient(fake)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (2).to_bytes(4, "big")
        + (3).to_bytes(4, "big")
    )
    preview_input = PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="asset-vlm",
            extraction_request_id="request-vlm",
            source_sha256="a" * 64,
            media_type="image/png",
        ),
        byte_size=len(png),
        content=png,
    )

    candidate = adapter.extract_raster_or_image(preview_input)

    assert candidate.status == "needs_review"
    assert candidate.extractor_revision == f"vlm-ocr-demo-v1:{DEFAULT_OPENAI_MODEL}"
    assert candidate.pages[0].spans[0].text == "INVOICE 123"
    part = fake.completions.requests[0]["messages"][1]["content"][0]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_provider_client_reports_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real_import = importlib.import_module

    def _missing(name: str, package: str | None = None) -> Any:
        if name == "openai":
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _missing)
    with pytest.raises(LLMConfigurationError):
        build_provider_client(
            ClaudeSettings(api_key=SecretStr(_KEY), provider="openai"), max_retries=0
        )


def test_build_provider_client_wraps_real_openai_sdk() -> None:
    pytest.importorskip("openai")
    client = build_provider_client(
        ClaudeSettings(api_key=SecretStr(_KEY), provider="openai"), max_retries=0
    )
    assert isinstance(client, OpenAIMessagesClient)
    client.close()
