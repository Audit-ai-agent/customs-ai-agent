"""Process-wide, operator-adjustable LLM settings for the functional-UAT profile.

The environment still supplies every credential and the baseline provider,
model, and limits.  An operator may override the non-secret fields (provider,
model, output limit, timeout) at runtime through the Agent's internal
``/internal/agent/v3/llm-settings`` surface; the override is validated with the
same rules as the environment, optionally persisted to one JSON file, and
pushed to every subscriber (the profile runtime and the VLM OCR demo adapter),
so a change applies to the next request without a restart.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from customs_ai_agent.adapters.llm import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENAI_MODEL,
    ClaudeSettings,
    LLMConfigurationError,
    LLMProvider,
    build_provider_client,
    provider_credential_present,
)


_LOGGER = logging.getLogger("customs_ai_agent.llm_settings")
_PROVIDERS: tuple[LLMProvider, ...] = ("anthropic", "openai")
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": DEFAULT_CLAUDE_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
}


class LLMSettingsUpdate(BaseModel):
    """Operator-supplied override; every field is optional and non-secret."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: LLMProvider | None = None
    model: str | None = Field(default=None, min_length=1, max_length=128)
    max_tokens: int | None = Field(default=None, ge=256, le=16_384)
    timeout_seconds: float | None = Field(default=None, ge=1, le=300)

    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (self.provider, self.model, self.max_tokens, self.timeout_seconds)
        )


class LLMSettingsError(RuntimeError):
    """Operator input rejected; ``code`` is a stable, content-free identifier."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LLMSettingsStore:
    """Current LLM settings = environment baseline + optional persisted override."""

    def __init__(self, *, override_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._override_path = override_path
        self._override: LLMSettingsUpdate | None = None
        self._updated_at: str | None = None
        self._version = 1
        self._subscribers: list[Callable[[ClaudeSettings], None]] = []
        self._load_override()

    # ---- construction -------------------------------------------------

    @classmethod
    def from_environment(cls) -> "LLMSettingsStore":
        raw = os.environ.get("CUSTOMS_AGENT_LLM_SETTINGS_PATH", "").strip()
        return cls(override_path=Path(raw) if raw else None)

    def _load_override(self) -> None:
        if self._override_path is None or not self._override_path.is_file():
            return
        try:
            data = json.loads(self._override_path.read_text(encoding="utf-8"))
            update = LLMSettingsUpdate.model_validate(
                {
                    key: data.get(key)
                    for key in ("provider", "model", "max_tokens", "timeout_seconds")
                }
            )
            # The override must still be a valid configuration against the
            # credentials of this process; otherwise it is ignored, not fatal.
            ClaudeSettings.from_environment(**update.model_dump(exclude_none=True))
        except (OSError, ValueError, ValidationError, LLMConfigurationError):
            _LOGGER.warning("llm_settings_override_ignored")
            return
        self._override = update
        updated_at = data.get("updated_at") if isinstance(data, dict) else None
        self._updated_at = updated_at if isinstance(updated_at, str) else None

    def _persist(self) -> None:
        if self._override_path is None:
            return
        try:
            if self._override is None:
                if self._override_path.exists():
                    self._override_path.unlink()
                return
            self._override_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **self._override.model_dump(exclude_none=True),
                "updated_at": self._updated_at,
            }
            tmp = self._override_path.with_suffix(self._override_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._override_path)
        except OSError:
            _LOGGER.warning("llm_settings_persist_failed")

    # ---- reading ------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    def current(self) -> ClaudeSettings:
        """Effective settings; raises LLMConfigurationError when the environment is unusable."""

        with self._lock:
            overrides = self._override.model_dump(exclude_none=True) if self._override else {}
        return ClaudeSettings.from_environment(**overrides)

    def describe(self) -> dict[str, Any]:
        """Non-secret public view for the operator panel."""

        with self._lock:
            override = self._override
            updated_at = self._updated_at
            version = self._version
        configured: dict[str, Any]
        error_code: str | None = None
        try:
            settings = self.current()
            configured = {
                "provider": settings.provider,
                "model": settings.model,
                "max_tokens": settings.max_tokens,
                "timeout_seconds": settings.timeout_seconds,
            }
        except LLMConfigurationError:
            error_code = "llm_configuration_invalid"
            fallback_provider = (override.provider if override else None) or "anthropic"
            configured = {
                "provider": fallback_provider,
                "model": (override.model if override else None)
                or _DEFAULT_MODELS[fallback_provider],
                "max_tokens": (override.max_tokens if override else None) or 4096,
                "timeout_seconds": (override.timeout_seconds if override else None) or 90,
            }
        return {
            **configured,
            "source": "override" if override is not None else "environment",
            "version": version,
            "updated_at": updated_at,
            "error_code": error_code,
            "providers": [
                {
                    "provider": provider,
                    "credential_present": provider_credential_present(provider),
                    "default_model": _DEFAULT_MODELS[provider],
                }
                for provider in _PROVIDERS
            ],
        }

    # ---- writing ------------------------------------------------------

    def subscribe(self, callback: Callable[[ClaudeSettings], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def resolve(self, update: LLMSettingsUpdate) -> ClaudeSettings:
        """Validate ``update`` layered on the current override without applying it."""

        with self._lock:
            base = self._override.model_dump(exclude_none=True) if self._override else {}
        merged = {**base, **update.model_dump(exclude_none=True)}
        provider = merged.get("provider") or ClaudeSettings.provider_from_environment()
        if not provider_credential_present(provider):
            raise LLMSettingsError(
                "llm_credential_missing", "No credential is configured for that provider."
            )
        # A provider switch without an explicit model falls back to that provider's default.
        if update.provider is not None and update.model is None:
            merged.pop("model", None)
        try:
            return ClaudeSettings.from_environment(**merged)
        except LLMConfigurationError:
            raise LLMSettingsError(
                "llm_settings_invalid", "The LLM settings are not valid."
            ) from None

    def apply(self, update: LLMSettingsUpdate) -> ClaudeSettings:
        if update.is_empty():
            raise LLMSettingsError("llm_settings_invalid", "No settings were supplied.")
        settings = self.resolve(update)
        # Instantiating the SDK client proves the provider package is installed.
        build_provider_client(settings, max_retries=0).close()
        with self._lock:
            base = self._override.model_dump(exclude_none=True) if self._override else {}
            merged = {**base, **update.model_dump(exclude_none=True)}
            if update.provider is not None and update.model is None:
                merged["model"] = settings.model
            self._override = LLMSettingsUpdate.model_validate(merged)
            self._updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._version += 1
            self._persist()
            subscribers = list(self._subscribers)
        self._notify(subscribers, settings)
        return settings

    def reset(self) -> None:
        with self._lock:
            if self._override is None:
                return
            self._override = None
            self._updated_at = None
            self._version += 1
            self._persist()
            subscribers = list(self._subscribers)
        try:
            settings = self.current()
        except LLMConfigurationError:
            return
        self._notify(subscribers, settings)

    @staticmethod
    def _notify(
        subscribers: list[Callable[[ClaudeSettings], None]], settings: ClaudeSettings
    ) -> None:
        for callback in subscribers:
            try:
                callback(settings)
            except Exception:
                _LOGGER.warning("llm_settings_subscriber_failed")

    # ---- connection test ---------------------------------------------

    def probe(self, update: LLMSettingsUpdate | None = None) -> dict[str, Any]:
        """Make one minimal provider request and report success without provider text."""

        try:
            settings = self.resolve(update) if update is not None else self.current()
        except LLMConfigurationError:
            return {"ok": False, "error_code": "llm_configuration_invalid"}
        except LLMSettingsError as error:
            return {"ok": False, "error_code": error.code}
        try:
            client = build_provider_client(settings, max_retries=0)
        except LLMConfigurationError:
            return {
                "ok": False,
                "provider": settings.provider,
                "model": settings.model,
                "error_code": "provider_sdk_unavailable",
            }
        started = time.monotonic()
        try:
            response = client.messages.create(
                model=settings.model,
                max_tokens=256,
                temperature=0,
                system='Reply with exactly this JSON object and nothing else: {"ok": true}',
                messages=[{"role": "user", "content": "ping"}],
            )
        except Exception:
            return {
                "ok": False,
                "provider": settings.provider,
                "model": settings.model,
                "latency_ms": round((time.monotonic() - started) * 1000),
                "error_code": "provider_request_failed",
            }
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        content = getattr(response, "content", None)
        text = ""
        if isinstance(content, list):
            text = "".join(
                part
                for part in (getattr(item, "text", None) for item in content)
                if isinstance(part, str)
            )
        return {
            "ok": bool(text.strip()),
            "provider": settings.provider,
            "model": settings.model,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "error_code": None if text.strip() else "provider_response_invalid",
        }


LLM_SETTINGS = LLMSettingsStore.from_environment()


__all__ = [
    "LLM_SETTINGS",
    "LLMSettingsError",
    "LLMSettingsStore",
    "LLMSettingsUpdate",
]
