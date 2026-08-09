"""Text-only Claude adapter, loaded only for explicit functional UAT."""

from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from customs_ai_agent.prompts import (
    load_document_analysis_system_prompt,
    load_grounded_assistant_system_prompt,
    load_grounded_report_system_prompt,
    load_official_guidance_system_prompt,
)
from customs_ai_agent.scenario_transcript import ScenarioTranscriptRun
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate


# Use a current, generally available Haiku revision for the default local
# Functional-UAT path. Operators may still select another permitted Claude
# model through the local-only profile value.
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_SAFE_MODEL = re.compile(r"^claude-[A-Za-z0-9._-]{2,120}$")
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
        analysis_phase = (
            "focused_reextraction" if focus_fields else "initial_extraction"
        )
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
            "provider": "anthropic",
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
                        "provider": "anthropic",
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
                        "provider": "anthropic",
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
                    "provider": "anthropic",
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
                        "provider": "anthropic",
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

    @classmethod
    def from_settings(cls, settings: ClaudeSettings) -> "ClaudeGroundedResponder":
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

    def respond(
        self,
        payload: Mapping[str, Any],
        *,
        transcript: ScenarioTranscriptRun | None = None,
        purpose: str,
    ) -> dict[str, Any]:
        """Make exactly one direct provider call and return its JSON object."""

        if purpose not in {"assistant_answer", "review_report", "knowledge_answer"}:
            raise LLMOutputValidationError("Claude response purpose is invalid.")
        system_prompt = {
            "review_report": self._report_system_prompt,
            "assistant_answer": self._system_prompt,
            "knowledge_answer": self._guidance_system_prompt,
        }[purpose]
        user_message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        request_payload = {
            "provider": "anthropic",
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
                        "provider": "anthropic",
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
                        "provider": "anthropic",
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
                {"provider": "anthropic", "model": self.model_id, "output_text": output},
            )
        try:
            parsed = json.loads(_json_text_from_model_output(output))
        except (TypeError, ValueError):
            if transcript is not None:
                transcript.write(
                    "claude",
                    f"{purpose}_response_invalid",
                    {
                        "provider": "anthropic",
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
                        "provider": "anthropic",
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
    "DocumentAnalysis",
    "DocumentAnalyzer",
    "LLMConfigurationError",
    "LLMError",
    "LLMOutputValidationError",
    "LLMProviderError",
]
