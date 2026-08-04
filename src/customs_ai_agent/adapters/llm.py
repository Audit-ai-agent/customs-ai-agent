"""Text-only Claude adapter, loaded only for explicit functional UAT."""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from customs_ai_agent.prompts import load_document_analysis_system_prompt
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_SAFE_MODEL = re.compile(r"^claude-[A-Za-z0-9._-]{2,120}$")


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMOutputValidationError(LLMError):
    pass


class ClaudeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: SecretStr = Field(repr=False)
    model: str = DEFAULT_CLAUDE_MODEL
    max_tokens: int = Field(default=4096, ge=256, le=16_384)
    timeout_seconds: float = Field(default=90, ge=1, le=300)

    @field_validator("api_key")
    @classmethod
    def _key_present(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().strip()) < 16:
            raise ValueError("Claude API key is missing or invalid")
        return value

    @field_validator("model")
    @classmethod
    def _safe_model(cls, value: str) -> str:
        value = value.strip()
        if _SAFE_MODEL.fullmatch(value) is None:
            raise ValueError("Claude model name is invalid")
        return value

    @classmethod
    def from_environment(cls) -> "ClaudeSettings":
        for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS"):
            if os.environ.get(name, "").strip():
                raise LLMConfigurationError("unsafe Claude transport override")
        try:
            return cls.model_validate(
                {
                    "api_key": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
                    "model": os.environ.get("CUSTOMS_AGENT_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
                    "max_tokens": os.environ.get("CUSTOMS_AGENT_CLAUDE_MAX_TOKENS", "4096"),
                    "timeout_seconds": os.environ.get("CUSTOMS_AGENT_CLAUDE_TIMEOUT_SECONDS", "90"),
                }
            )
        except ValidationError:
            raise LLMConfigurationError("functional UAT Claude configuration is invalid") from None


@dataclass(frozen=True)
class DocumentAnalysis:
    candidate: ClaudeDocumentCandidate
    model_id: str


class DocumentAnalyzer(Protocol):
    model_id: str

    def analyze(self, document: Mapping[str, Any]) -> DocumentAnalysis: ...


class ClaudeDocumentAnalyzer:
    """One direct Claude request for one text document."""

    def __init__(self, settings: ClaudeSettings, client: Any) -> None:
        self._settings = settings
        self._client = client
        self.model_id = settings.model
        self._system_prompt = load_document_analysis_system_prompt()

    @classmethod
    def from_settings(cls, settings: ClaudeSettings) -> "ClaudeDocumentAnalyzer":
        try:
            module = importlib.import_module("anthropic")
            client = module.Anthropic(
                api_key=settings.api_key.get_secret_value(),
                timeout=settings.timeout_seconds,
                max_retries=1,
            )
        except ModuleNotFoundError:
            raise LLMConfigurationError(
                "Install the functional-uat extra before Claude UAT."
            ) from None
        except Exception:
            raise LLMConfigurationError(
                "The local Claude client could not be initialized."
            ) from None
        return cls(settings, client)

    def analyze(self, document: Mapping[str, Any]) -> DocumentAnalysis:
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
        try:
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=self._settings.max_tokens,
                temperature=0,
                system=self._system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Return only the requested JSON for this document.\n"
                            f"document_id: {document_id}\nfilename: {filename}\ntext:\n{text}"
                        ),
                    }
                ],
            )
        except Exception:
            raise LLMProviderError("Claude document analysis is unavailable.") from None
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            raise LLMOutputValidationError("Claude did not return a text result.")
        output = "".join(
            part
            for part in (getattr(item, "text", None) for item in content)
            if isinstance(part, str)
        ).strip()
        try:
            candidate = ClaudeDocumentCandidate.model_validate_json(output)
        except (ValidationError, ValueError):
            raise LLMOutputValidationError(
                "Claude returned an invalid document analysis."
            ) from None
        return DocumentAnalysis(candidate=candidate, model_id=self.model_id)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


__all__ = [
    "ClaudeDocumentAnalyzer",
    "ClaudeSettings",
    "DEFAULT_CLAUDE_MODEL",
    "DocumentAnalysis",
    "DocumentAnalyzer",
    "LLMConfigurationError",
    "LLMError",
    "LLMOutputValidationError",
    "LLMProviderError",
]
