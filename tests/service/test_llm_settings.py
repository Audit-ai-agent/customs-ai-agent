"""Operator-adjustable LLM settings: store semantics and the internal HTTP surface."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from customs_ai_agent.adapters import llm as llm_module
from customs_ai_agent.adapters.llm import ClaudeSettings, LLMConfigurationError
from customs_ai_agent.service import llm_settings as store_module
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.llm_settings import (
    LLMSettingsError,
    LLMSettingsStore,
    LLMSettingsUpdate,
)
from customs_ai_agent.service.profiles import SERVICE_MODE_HEADER
from customs_ai_agent.service.settings import ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
_OPENAI_KEY = "sk-test-" + "x" * 40
_ANTHROPIC_KEY = "sk-ant-" + "y" * 40


class _FakeClient:
    """Stands in for both SDK clients behind ``build_provider_client``."""

    instances: list["_FakeClient"] = []

    def __init__(self, settings: ClaudeSettings, output: str = '{"ok": true}') -> None:
        self.settings = settings
        self.output = output
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        _FakeClient.instances.append(self)

    def _create(self, **request: Any) -> Any:
        self.calls.append(request)
        if self.output == "RAISE":
            raise RuntimeError("provider down")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.output)])

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in (
        "CUSTOMS_AGENT_LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CUSTOMS_AGENT_OPENAI_MODEL",
        "CUSTOMS_AGENT_CLAUDE_MODEL",
        "CUSTOMS_AGENT_LLM_MAX_TOKENS",
        "CUSTOMS_AGENT_CLAUDE_MAX_TOKENS",
        "CUSTOMS_AGENT_LLM_TIMEOUT_SECONDS",
        "CUSTOMS_AGENT_CLAUDE_TIMEOUT_SECONDS",
        "CUSTOMS_AGENT_LLM_SETTINGS_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_KEY)
    monkeypatch.setenv("CUSTOMS_AGENT_LLM_PROVIDER", "openai")
    _FakeClient.instances.clear()
    monkeypatch.setattr(
        llm_module,
        "build_provider_client",
        lambda settings, *, max_retries: _FakeClient(settings),
    )
    monkeypatch.setattr(
        store_module,
        "build_provider_client",
        lambda settings, *, max_retries: _FakeClient(settings),
    )
    return monkeypatch


def _settings(**changes: Any) -> ServiceSettings:
    return replace(
        ServiceSettings(
            bearer_token=SecretStr(_TOKEN),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(_TOKEN),
        ),
        **changes,
    )


# ---------------------------------------------------------------- store


def test_describe_reports_environment_baseline(env: pytest.MonkeyPatch) -> None:
    store = LLMSettingsStore()

    view = store.describe()

    assert view["provider"] == "openai"
    assert view["model"] == llm_module.DEFAULT_OPENAI_MODEL
    assert view["source"] == "environment"
    assert view["version"] == 1
    assert view["error_code"] is None
    providers = {item["provider"]: item for item in view["providers"]}
    assert providers["openai"]["credential_present"] is True
    assert providers["anthropic"]["credential_present"] is False


def test_apply_overrides_persist_and_notify(env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "llm-settings.json"
    store = LLMSettingsStore(override_path=path)
    seen: list[ClaudeSettings] = []
    store.subscribe(seen.append)

    applied = store.apply(LLMSettingsUpdate(model="gpt-5.6-luna", max_tokens=8192))

    assert applied.model == "gpt-5.6-luna"
    assert applied.max_tokens == 8192
    assert seen and seen[-1].model == "gpt-5.6-luna"
    assert store.version == 2
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["model"] == "gpt-5.6-luna"
    assert saved["max_tokens"] == 8192
    assert "api_key" not in saved and _OPENAI_KEY not in path.read_text(encoding="utf-8")
    # The probe client created to validate the SDK was closed again.
    assert all(client.closed for client in _FakeClient.instances)

    reloaded = LLMSettingsStore(override_path=path)
    assert reloaded.describe()["model"] == "gpt-5.6-luna"
    assert reloaded.describe()["source"] == "override"


def test_provider_switch_uses_provider_default_model(env: pytest.MonkeyPatch) -> None:
    env.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    store = LLMSettingsStore()
    store.apply(LLMSettingsUpdate(model="gpt-5.6-luna"))

    switched = store.apply(LLMSettingsUpdate(provider="anthropic"))

    assert switched.provider == "anthropic"
    assert switched.model == llm_module.DEFAULT_CLAUDE_MODEL
    assert switched.api_key.get_secret_value() == _ANTHROPIC_KEY


def test_provider_without_credential_is_rejected(env: pytest.MonkeyPatch) -> None:
    store = LLMSettingsStore()

    with pytest.raises(LLMSettingsError) as error:
        store.apply(LLMSettingsUpdate(provider="anthropic"))

    assert error.value.code == "llm_credential_missing"
    assert store.version == 1


def test_invalid_model_name_is_rejected(env: pytest.MonkeyPatch) -> None:
    store = LLMSettingsStore()

    with pytest.raises(LLMSettingsError) as error:
        store.apply(LLMSettingsUpdate(model="claude-haiku-4-5-20251001"))

    assert error.value.code == "llm_settings_invalid"


def test_reset_returns_to_environment(env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "llm-settings.json"
    store = LLMSettingsStore(override_path=path)
    store.apply(LLMSettingsUpdate(model="gpt-5.6-luna"))

    store.reset()

    assert store.describe()["source"] == "environment"
    assert store.describe()["model"] == llm_module.DEFAULT_OPENAI_MODEL
    assert not path.exists()
    assert store.version == 3


def test_stale_override_file_is_ignored(env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "llm-settings.json"
    path.write_text(json.dumps({"provider": "anthropic", "model": "claude-x"}), encoding="utf-8")

    store = LLMSettingsStore(override_path=path)

    assert store.describe()["source"] == "environment"


def test_probe_reports_success_without_provider_text(env: pytest.MonkeyPatch) -> None:
    store = LLMSettingsStore()

    result = store.probe(LLMSettingsUpdate(model="gpt-5.6-luna"))

    assert result["ok"] is True
    assert result["model"] == "gpt-5.6-luna"
    assert result["error_code"] is None
    assert "text" not in result and "output" not in result
    client = _FakeClient.instances[-1]
    assert client.calls[0]["model"] == "gpt-5.6-luna"
    assert client.closed
    # A probe never changes the active configuration.
    assert store.describe()["model"] == llm_module.DEFAULT_OPENAI_MODEL


def test_probe_maps_provider_failure(env: pytest.MonkeyPatch) -> None:
    env.setattr(
        store_module,
        "build_provider_client",
        lambda settings, *, max_retries: _FakeClient(settings, output="RAISE"),
    )
    store = LLMSettingsStore()

    result = store.probe()

    assert result["ok"] is False
    assert result["error_code"] == "provider_request_failed"


def test_current_raises_when_environment_unusable(env: pytest.MonkeyPatch) -> None:
    env.delenv("OPENAI_API_KEY")
    store = LLMSettingsStore()

    with pytest.raises(LLMConfigurationError):
        store.current()
    assert store.describe()["error_code"] == "llm_configuration_invalid"


# ---------------------------------------------------------------- HTTP surface


class _Runtime:
    def review_v3_json(self, _payload: bytes) -> object:
        raise AssertionError("review is not part of this test")

    def close(self) -> None:
        return None


def _app(mode: str, store: LLMSettingsStore, monkeypatch: pytest.MonkeyPatch) -> Any:
    from customs_ai_agent.service import app as app_module

    monkeypatch.setattr(app_module, "LLM_SETTINGS", store)
    return create_app(
        _settings(service_mode=mode),  # type: ignore[arg-type]
        runtime_factory=lambda _settings: _Runtime(),  # type: ignore[arg-type,return-value]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )


def _headers(mode: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKEN}", SERVICE_MODE_HEADER: mode}


def test_http_read_requires_bearer(env: pytest.MonkeyPatch) -> None:
    app = _app("functional-uat", LLMSettingsStore(), env)
    with TestClient(app, raise_server_exceptions=False) as client:
        denied = client.get(
            "/internal/agent/v3/llm-settings", headers={SERVICE_MODE_HEADER: "functional-uat"}
        )
        accepted = client.get("/internal/agent/v3/llm-settings", headers=_headers("functional-uat"))

    assert denied.status_code == 401
    assert accepted.status_code == 200
    body = accepted.json()
    assert body["provider"] == "openai"
    assert body["adjustable"] is True
    assert body["service_mode"] == "functional-uat"
    assert "api_key" not in json.dumps(body) and _OPENAI_KEY not in json.dumps(body)


def test_http_update_reset_and_test_round_trip(env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = LLMSettingsStore(override_path=tmp_path / "llm.json")
    app = _app("functional-uat", store, env)
    headers = _headers("functional-uat")
    with TestClient(app, raise_server_exceptions=False) as client:
        updated = client.put(
            "/internal/agent/v3/llm-settings",
            headers=headers,
            json={"model": "gpt-5.6-luna", "timeout_seconds": 120},
        )
        tested = client.post(
            "/internal/agent/v3/llm-settings/test", headers=headers, json={"max_tokens": 1024}
        )
        tested_current = client.post(
            "/internal/agent/v3/llm-settings/test", headers=headers, json={}
        )
        reset = client.delete("/internal/agent/v3/llm-settings", headers=headers)

    assert updated.status_code == 200
    assert updated.json()["model"] == "gpt-5.6-luna"
    assert updated.json()["timeout_seconds"] == 120
    assert updated.json()["source"] == "override"
    assert tested.status_code == 200 and tested.json()["ok"] is True
    assert tested.json()["model"] == "gpt-5.6-luna"
    assert tested_current.status_code == 200 and tested_current.json()["ok"] is True
    assert reset.status_code == 200 and reset.json()["source"] == "environment"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({}, "llm_settings_invalid"),
        ({"model": "gpt-4.1-mini", "unexpected": 1}, "llm_settings_invalid"),
        ({"max_tokens": 10}, "llm_settings_invalid"),
        ({"provider": "anthropic"}, "llm_credential_missing"),
        ({"model": "claude-haiku-4-5-20251001"}, "llm_settings_invalid"),
    ],
)
def test_http_update_rejects_bad_input(
    env: pytest.MonkeyPatch, body: dict[str, Any], code: str
) -> None:
    app = _app("functional-uat", LLMSettingsStore(), env)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.put(
            "/internal/agent/v3/llm-settings", headers=_headers("functional-uat"), json=body
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


def test_http_update_is_refused_in_local_development(env: pytest.MonkeyPatch) -> None:
    app = _app("local-development", LLMSettingsStore(), env)
    headers = _headers("local-development")
    with TestClient(app, raise_server_exceptions=False) as client:
        read = client.get("/internal/agent/v3/llm-settings", headers=headers)
        update = client.put(
            "/internal/agent/v3/llm-settings", headers=headers, json={"model": "gpt-4.1"}
        )
        test = client.post("/internal/agent/v3/llm-settings/test", headers=headers, json={})

    assert read.status_code == 200 and read.json()["adjustable"] is False
    assert update.status_code == 409
    assert update.json()["error"]["code"] == "llm_settings_unavailable"
    assert test.status_code == 409
