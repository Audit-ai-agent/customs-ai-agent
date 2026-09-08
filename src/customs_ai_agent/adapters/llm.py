"""Text-only LLM adapter (Anthropic Claude or OpenAI), loaded only for explicit functional UAT.

Both providers are driven through one narrow call surface modelled on the
Anthropic ``client.messages.create`` API.  The OpenAI path wraps the official
SDK in :class:`OpenAIMessagesClient`, which translates that request shape to
Chat Completions and returns a response with the same ``content[].text``
reading, so the analyzer, grounded responder, and VLM OCR demo stay identical.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from customs_ai_agent.prompts import (
    load_document_analysis_system_prompt,
    load_grounded_assistant_system_prompt,
    load_evidence_relevance_system_prompt,
    load_grounded_report_system_prompt,
    load_official_guidance_system_prompt,
)
from customs_ai_agent.scenario_transcript import ScenarioTranscriptRun
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate


# Use a current, generally available Haiku revision for the default local
# Functional-UAT path. Operators may still select another permitted Claude
# model through the local-only profile value.
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
# Default OpenAI model: a current, generally available small model that
# accepts text, image, and PDF file inputs and honours temperature 0.
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
LLMProvider = Literal["anthropic", "openai"]
_SAFE_MODEL = re.compile(r"^claude-[A-Za-z0-9._-]{2,120}$")
_SAFE_OPENAI_MODEL = re.compile(r"^(?:gpt-|chatgpt-|o[1-9])[A-Za-z0-9._-]{1,120}$")
# gpt-5 and o-series reasoning models reject a non-default sampling temperature.
_OPENAI_FIXED_TEMPERATURE = re.compile(r"^(?:gpt-5|o[1-9])")
_SINGLE_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>\{.*\})\r?\n?```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMOutputValidationError(LLMError):
    pass


def _json_text_from_model_output(output: str) -> str:
    """Accept one complete JSON code fence without accepting surrounding prose."""

    normalized = output.strip()
    match = _SINGLE_JSON_FENCE.fullmatch(normalized)
    return match.group("body").strip() if match is not None else normalized


class ClaudeSettings(BaseModel):
    """Provider-neutral LLM settings (the class name is kept for compatibility)."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr = Field(repr=False)
    provider: LLMProvider = "anthropic"
    model: str = ""
    max_tokens: int = Field(default=4096, ge=256, le=16_384)
    timeout_seconds: float = Field(default=90, ge=1, le=300)

    @field_validator("api_key")
    @classmethod
    def _key_present(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().strip()) < 16:
            raise ValueError("LLM API key is missing or invalid")
        return value

    @model_validator(mode="after")
    def _model_matches_provider(self) -> "ClaudeSettings":
        model = self.model.strip()
        if not model:
            model = DEFAULT_OPENAI_MODEL if self.provider == "openai" else DEFAULT_CLAUDE_MODEL
        pattern = _SAFE_OPENAI_MODEL if self.provider == "openai" else _SAFE_MODEL
        if pattern.fullmatch(model) is None:
            raise ValueError(f"{self.provider} model name is invalid")
        if model != self.model:
            object.__setattr__(self, "model", model)
        return self

    @staticmethod
    def provider_from_environment() -> str:
        """The provider the environment selects (explicitly, or by which key is present)."""

        provider = os.environ.get("CUSTOMS_AGENT_LLM_PROVIDER", "").strip().casefold()
        if provider:
            return provider
        # Explicit selection wins; otherwise follow whichever key is present.
        return (
            "openai"
            if os.environ.get("OPENAI_API_KEY", "").strip()
            and not os.environ.get("ANTHROPIC_API_KEY", "").strip()
            else "anthropic"
        )

    @classmethod
    def from_environment(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | str | None = None,
        timeout_seconds: float | str | None = None,
    ) -> "ClaudeSettings":
        """Build settings from the environment; keyword overrides replace the non-secret fields."""

        provider = (provider or cls.provider_from_environment()).strip().casefold()
        unsafe: tuple[str, ...]
        if provider == "openai":
            unsafe = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
            key_name, model_name, default_model = (
                "OPENAI_API_KEY",
                "CUSTOMS_AGENT_OPENAI_MODEL",
                DEFAULT_OPENAI_MODEL,
            )
        elif provider == "anthropic":
            unsafe = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS")
            key_name, model_name, default_model = (
                "ANTHROPIC_API_KEY",
                "CUSTOMS_AGENT_CLAUDE_MODEL",
                DEFAULT_CLAUDE_MODEL,
            )
        else:
            raise LLMConfigurationError("functional UAT LLM provider is invalid")
        for name in unsafe:
            if os.environ.get(name, "").strip():
                raise LLMConfigurationError("unsafe LLM transport override")
        if max_tokens is None:
            max_tokens = os.environ.get(
                "CUSTOMS_AGENT_LLM_MAX_TOKENS", ""
            ).strip() or os.environ.get("CUSTOMS_AGENT_CLAUDE_MAX_TOKENS", "4096")
        if timeout_seconds is None:
            timeout_seconds = os.environ.get(
                "CUSTOMS_AGENT_LLM_TIMEOUT_SECONDS", ""
            ).strip() or os.environ.get("CUSTOMS_AGENT_CLAUDE_TIMEOUT_SECONDS", "90")
        try:
            return cls.model_validate(
                {
                    "api_key": os.environ.get(key_name, "").strip(),
                    "provider": provider,
                    "model": model
                    if model is not None
                    else os.environ.get(model_name, default_model),
                    "max_tokens": max_tokens,
                    "timeout_seconds": timeout_seconds,
                }
            )
        except ValidationError:
            raise LLMConfigurationError("functional UAT LLM configuration is invalid") from None


LLMSettings = ClaudeSettings

_PROVIDER_KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def provider_credential_present(provider: str) -> bool:
    """Whether the environment carries a plausible credential for ``provider``."""

    key_name = _PROVIDER_KEY_NAMES.get(provider)
    return bool(key_name) and len(os.environ.get(key_name or "", "").strip()) >= 16


def _data_url(source: Any) -> str:
    if not isinstance(source, Mapping):
        raise LLMProviderError("The provider request block is invalid.")
    media_type = source.get("media_type")
    data = source.get("data")
    if (
        source.get("type") != "base64"
        or not isinstance(media_type, str)
        or not isinstance(data, str)
    ):
        raise LLMProviderError("The provider request block is invalid.")
    return f"data:{media_type};base64,{data}"


def _openai_content(content: Any) -> Any:
    """Translate Anthropic-style message content into Chat Completions parts."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise LLMProviderError("The provider request content is invalid.")
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise LLMProviderError("The provider request block is invalid.")
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image":
            parts.append(
                {"type": "image_url", "image_url": {"url": _data_url(block.get("source"))}}
            )
        elif kind == "document":
            parts.append(
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": _data_url(block.get("source")),
                    },
                }
            )
        else:
            raise LLMProviderError("The provider request block is invalid.")
    return parts


class _OpenAIMessagesNamespace:
    def __init__(self, owner: "OpenAIMessagesClient") -> None:
        self._owner = owner

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[Mapping[str, Any]],
        temperature: float | None = 0,
    ) -> Any:
        return self._owner.create_message(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            temperature=temperature,
        )


class OpenAIMessagesClient:
    """Anthropic ``messages.create`` call surface over the OpenAI Chat Completions API."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.messages = _OpenAIMessagesNamespace(self)

    def create_message(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[Mapping[str, Any]],
        temperature: float | None,
    ) -> Any:
        chat_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            chat_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": _openai_content(message.get("content")),
                }
            )
        request: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_completion_tokens": max_tokens,
            # Every prompt asks for a single JSON object; JSON mode enforces that.
            "response_format": {"type": "json_object"},
        }
        if temperature is not None and _OPENAI_FIXED_TEMPERATURE.match(model) is None:
            request["temperature"] = temperature
        response = self._client.chat.completions.create(**request)
        choices = getattr(response, "choices", None)
        text = ""
        if isinstance(choices, list) and choices:
            choice_message = getattr(choices[0], "message", None)
            value = getattr(choice_message, "content", None)
            if isinstance(value, str):
                text = value
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def build_provider_client(settings: ClaudeSettings, *, max_retries: int) -> Any:
    """Construct the SDK client for ``settings.provider`` behind the shared call surface.

    Raises :class:`LLMConfigurationError` when the SDK is not installed or cannot start.
    """

    if settings.provider == "openai":
        try:
            module = importlib.import_module("openai")
            client = module.OpenAI(
                api_key=settings.api_key.get_secret_value(),
                timeout=settings.timeout_seconds,
                max_retries=max_retries,
            )
        except ModuleNotFoundError:
            raise LLMConfigurationError(
                "Install the functional-uat extra before OpenAI UAT."
            ) from None
        except Exception:
            raise LLMConfigurationError(
                "The local OpenAI client could not be initialized."
            ) from None
        return OpenAIMessagesClient(client)
    try:
        module = importlib.import_module("anthropic")
        return module.Anthropic(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
            max_retries=max_retries,
        )
    except ModuleNotFoundError:
        raise LLMConfigurationError("Install the functional-uat extra before Claude UAT.") from None
    except Exception:
        raise LLMConfigurationError("The local Claude client could not be initialized.") from None


@dataclass(frozen=True)
class DocumentAnalysis:
    candidate: ClaudeDocumentCandidate
    model_id: str


class DocumentAnalyzer(Protocol):
    model_id: str

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: ScenarioTranscriptRun | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis: ...


class ClaudeDocumentAnalyzer:
    """One direct Claude request for one text document."""

    def __init__(self, settings: ClaudeSettings, client: Any) -> None:
        self._settings = settings
        self._client = client
        self.model_id = settings.model
        self._system_prompt = load_document_analysis_system_prompt()

    @classmethod
    def from_settings(cls, settings: ClaudeSettings) -> "ClaudeDocumentAnalyzer":
        return cls(settings, build_provider_client(settings, max_retries=1))

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: ScenarioTranscriptRun | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis:
        document_id = document.get("document_id")
        filename = document.get("filename")
        text = document.get("content_text")
        if (
            not isinstance(document_id, str)
            or not isinstance(filename, str)
            or not isinstance(text, str)
        ):
            raise LLMProviderError("The text document is invalid.")
        if not text.strip() or len(text) > 200_000:
            raise LLMProviderError("The text document exceeds the profile limit.")
        # This is deliberately a fixed, non-document value.  A Functional-UAT
        # verifier may distinguish the bounded retry degradation from an
        # initial extraction failure without learning a document id, requested
        # field, prompt, or provider output.
        analysis_phase = "focused_reextraction" if focus_fields else "initial_extraction"
        focused_instruction = (
            "\nPerform one focused re-check for only these missing or ambiguous extraction fields: "
            + ", ".join(focus_fields)
            + ". Keep all output source-grounded; do not infer a value that is absent."
            if focus_fields
            else ""
        )
        user_message = (
            "Return only the requested JSON for this document.\n"
            f"document_id: {document_id}\nfilename: {filename}{focused_instruction}\ntext:\n{text}"
        )
        request_payload = {
            "provider": self._settings.provider,
            "model": self.model_id,
            "analysis_phase": analysis_phase,
            "max_tokens": self._settings.max_tokens,
            "temperature": 0,
            "system_prompt": self._system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        if transcript is not None:
            # The adapter is also directly unit-testable.  Add the live key at
            # this boundary even when a caller supplied a bare transcript run.
            transcript = transcript.with_sensitive_values(
                (self._settings.api_key.get_secret_value(),)
            )
            transcript.write("claude", "request_started", request_payload)
        try:
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=self._settings.max_tokens,
                temperature=0,
                system=self._system_prompt,
                messages=request_payload["messages"],
            )
        except Exception:
            if transcript is not None:
                transcript.write(
                    "claude",
                    "request_failed",
                    {
                        "provider": self._settings.provider,
                        "model": self.model_id,
                        "analysis_phase": analysis_phase,
                        "error_code": "provider_request_failed",
                    },
                )
            raise LLMProviderError("Claude document analysis is unavailable.") from None
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            if transcript is not None:
                transcript.write(
                    "claude",
                    "response_invalid",
                    {
                        "provider": self._settings.provider,
                        "model": self.model_id,
                        "analysis_phase": analysis_phase,
                        "error_code": "provider_response_invalid",
                    },
                )
            raise LLMOutputValidationError("Claude did not return a text result.")
        output = "".join(
            part
            for part in (getattr(item, "text", None) for item in content)
            if isinstance(part, str)
        ).strip()
        if transcript is not None:
            transcript.write(
                "claude",
                "response_received",
                {
                    "provider": self._settings.provider,
                    "model": self.model_id,
                    "analysis_phase": analysis_phase,
                    "output_text": output,
                },
            )
        try:
            candidate = ClaudeDocumentCandidate.model_validate_json(
                _json_text_from_model_output(output)
            )
        except (ValidationError, ValueError):
            if transcript is not None:
                transcript.write(
                    "claude",
                    "response_invalid",
                    {
                        "provider": self._settings.provider,
                        "model": self.model_id,
                        "analysis_phase": analysis_phase,
                        "error_code": "provider_output_invalid",
                    },
                )
            raise LLMOutputValidationError(
                "Claude returned an invalid document analysis."
            ) from None
        return DocumentAnalysis(candidate=candidate, model_id=self.model_id)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class ClaudeGroundedResponder:
    """One bounded Claude response for a review report or assistant answer.

    The caller validates the returned JSON against the public wire model and
    uses a deterministic fallback for any provider or grounding failure.
    """

    def __init__(self, settings: ClaudeSettings, client: Any) -> None:
        self._settings = settings
        self._client = client
        self.model_id = settings.model
        self._system_prompt = load_grounded_assistant_system_prompt()
        self._report_system_prompt = load_grounded_report_system_prompt()
        self._guidance_system_prompt = load_official_guidance_system_prompt()
        self._evidence_relevance_system_prompt = load_evidence_relevance_system_prompt()

    @classmethod
    def from_settings(cls, settings: ClaudeSettings) -> "ClaudeGroundedResponder":
        # Every grounded response has a deterministic caller fallback.  Keep
        # this one provider attempt inside Backend's response budget instead
        # of allowing the SDK to consume it with a second full timeout.
        return cls(settings, build_provider_client(settings, max_retries=0))

    def respond(
        self,
        payload: Mapping[str, Any],
        *,
        transcript: ScenarioTranscriptRun | None = None,
        purpose: str,
    ) -> dict[str, Any]:
        """Make exactly one direct provider call and return its JSON object."""

        if purpose not in {
            "assistant_answer",
            "review_report",
            "knowledge_answer",
            "evidence_relevance",
        }:
            raise LLMOutputValidationError("Claude response purpose is invalid.")
        system_prompt = {
            "review_report": self._report_system_prompt,
            "assistant_answer": self._system_prompt,
            "knowledge_answer": self._guidance_system_prompt,
            "evidence_relevance": self._evidence_relevance_system_prompt,
        }[purpose]
        user_message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        request_payload = {
            "provider": self._settings.provider,
            "purpose": purpose,
            "model": self.model_id,
            "max_tokens": self._settings.max_tokens,
            "temperature": 0,
            "system_prompt": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        if transcript is not None:
            transcript = transcript.with_sensitive_values(
                (self._settings.api_key.get_secret_value(),)
            )
            transcript.write("claude", f"{purpose}_request_started", request_payload)
        try:
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=self._settings.max_tokens,
                temperature=0,
                system=system_prompt,
                messages=request_payload["messages"],
            )
        except Exception:
            if transcript is not None:
                transcript.write(
                    "claude",
                    f"{purpose}_request_failed",
                    {
                        "provider": self._settings.provider,
                        "model": self.model_id,
                        "error_code": "provider_request_failed",
                    },
                )
            raise LLMProviderError("Claude grounded response is unavailable.") from None
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            if transcript is not None:
                transcript.write(
                    "claude",
                    f"{purpose}_response_invalid",
                    {
                        "provider": self._settings.provider,
                        "model": self.model_id,
                        "error_code": "provider_response_invalid",
                    },
                )
            raise LLMOutputValidationError("Claude did not return a text result.")
        output = "".join(
            part
            for part in (getattr(item, "text", None) for item in content)
            if isinstance(part, str)
        ).strip()
        if transcript is not None:
            transcript.write(
                "claude",
                f"{purpose}_response_received",
                {
                    "provider": self._settings.provider,
                    "model": self.model_id,
                    "output_text": output,
                },
            )
        try:
            parsed = json.loads(_json_text_from_model_output(output))
        except (TypeError, ValueError):
            if transcript is not None:
                transcript.write(
                    "claude",
                    f"{purpose}_response_invalid",
                    {
                        "provider": self._settings.provider,
                        "model": self.model_id,
                        "error_code": "provider_output_invalid",
                    },
                )
            raise LLMOutputValidationError("Claude returned invalid JSON.") from None
        if not isinstance(parsed, dict):
            if transcript is not None:
                transcript.write(
                    "claude",
                    f"{purpose}_response_invalid",
                    {
                        "provider": self._settings.provider,
                        "model": self.model_id,
                        "error_code": "provider_output_invalid",
                    },
                )
            raise LLMOutputValidationError("Claude returned an invalid JSON object.")
        return parsed

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


__all__ = [
    "ClaudeDocumentAnalyzer",
    "ClaudeGroundedResponder",
    "ClaudeSettings",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "LLMProvider",
    "LLMSettings",
    "OpenAIMessagesClient",
    "build_provider_client",
    "provider_credential_present",
    "DocumentAnalysis",
    "DocumentAnalyzer",
    "LLMConfigurationError",
    "LLMError",
    "LLMOutputValidationError",
    "LLMProviderError",
]
