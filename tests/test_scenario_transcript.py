from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import URLError

import pytest
from pydantic import SecretStr

import customs_ai_agent.service.profiles as profile_module
import customs_ai_agent.scenario_transcript as transcript_module
from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeSettings,
    LLMProviderError,
)
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.scenario_transcript import (
    ScenarioTranscriptError,
    ScenarioTranscriptRoot,
    TRANSCRIPT_ROOT_ENV,
)
from customs_ai_agent.service.profiles import ProfileEvidenceRetriever, ProfileRuntime
from customs_ai_agent.service.settings import ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz" + "0123456789ABCDEFGHIJK"
_SECOND_TOKEN = "Zyxwvutsrqponmlkjihgfedcba" + "9876543210ABCDEFGHIJK"
_CLAUDE_KEY = "sk" + "-ant-" + "functional-uat-live-key-abcdefghijklmnop"


def _root(tmp_path: Path) -> Path:
    return tmp_path / "_local" / "state" / "scenario-runs"


def _settings(**changes: Any) -> ServiceSettings:
    return replace(
        ServiceSettings(
            bearer_token=SecretStr(_TOKEN),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(_TOKEN),
        ),
        **changes,
    )


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(value) for value in path.read_text(encoding="utf-8").splitlines()]


def test_transcript_requires_local_root_and_redacts_secret_values(tmp_path: Path) -> None:
    root = ScenarioTranscriptRoot.from_environment(
        {TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))}, sensitive_values=(_TOKEN,)
    )
    run = root.for_run("run-transcript")
    assert run is not None
    synthetic_key = "sk" + "-ant-" + "abcdefghijklmnop"
    run.write(
        "agent-data",
        "http_exchange",
        {
            "access_token": "not-to-be-recorded",
            "content_text": (
                f"synthetic document ANTHROPIC_API_KEY={synthetic_key}; literal={_TOKEN}"
            ),
        },
    )

    payload = _lines(run.directory / "agent-data.jsonl")[0]["payload"]
    assert payload["access_token"] == "[REDACTED]"
    assert (
        payload["content_text"]
        == "synthetic document ANTHROPIC_API_KEY=[REDACTED]; literal=[REDACTED]"
    )

    with pytest.raises(ScenarioTranscriptError, match="stream_invalid"):
        run.write("backend-agent", "review_exchange", {})

    with pytest.raises(ScenarioTranscriptError, match="root_invalid"):
        ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(tmp_path / "outside")})


def test_functional_uat_uses_only_the_fixed_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _root(tmp_path)
    monkeypatch.setattr(transcript_module, "_functional_uat_directory", lambda: expected)

    root = ScenarioTranscriptRoot.for_service_mode("functional-uat", environ={})
    assert root.directory == expected.resolve()
    same_root = ScenarioTranscriptRoot.for_service_mode(
        "functional-uat", {TRANSCRIPT_ROOT_ENV: str(expected)}
    )
    assert same_root.directory == root.directory

    with pytest.raises(ScenarioTranscriptError, match="root_invalid"):
        ScenarioTranscriptRoot.for_service_mode(
            "functional-uat",
            {TRANSCRIPT_ROOT_ENV: str(tmp_path / "other" / "scenario-runs")},
        )


def test_functional_runtime_automatically_redacts_all_live_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    calls: list[str] = []

    def functional_root(
        cls: type[ScenarioTranscriptRoot],
        service_mode: str,
        environ: dict[str, str] | None = None,
        *,
        sensitive_values: tuple[str, ...] = (),
    ) -> ScenarioTranscriptRoot:
        del cls, environ
        calls.append(service_mode)
        return root.with_sensitive_values(sensitive_values)

    monkeypatch.setattr(
        ScenarioTranscriptRoot,
        "for_service_mode",
        classmethod(functional_root),
    )
    monkeypatch.setattr(
        ClaudeSettings,
        "from_environment",
        classmethod(lambda cls: ClaudeSettings(api_key=SecretStr(_CLAUDE_KEY))),
    )
    monkeypatch.setattr(
        ClaudeDocumentAnalyzer,
        "from_settings",
        classmethod(lambda cls, settings: profile_module._DeterministicDocumentAnalyzer()),
    )

    settings = _settings(
        service_mode="functional-uat",
        evidence_service_token=SecretStr(_SECOND_TOKEN),
    )
    runtime = ProfileRuntime(settings)
    assert calls == ["functional-uat"]
    transcript = runtime._transcript_root.for_run("run-functional-secrets")
    assert transcript is not None
    transcript.write(
        "claude",
        "test_event",
        {"content_text": f"{_TOKEN} {_SECOND_TOKEN} {_CLAUDE_KEY}"},
    )
    record = _lines(transcript.directory / "claude.jsonl")[0]
    assert record["payload"]["content_text"] == "[REDACTED] [REDACTED] [REDACTED]"


def test_retriever_records_actual_data_request_and_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    class Response:
        headers = Headers({"X-Customs-Service-Mode": "local-development"})

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return b'{"status":"ready"}'

    monkeypatch.setattr(profile_module._HTTP, "open", lambda *_args, **_kwargs: Response())
    retriever = ProfileEvidenceRetriever(_settings(), transcript_root=root)
    assert retriever._request(
        retriever.readiness_path,
        method="GET",
        review_run_id="run-data-trace",
    ) == {"status": "ready"}

    records = _lines(_root(tmp_path) / "run-data-trace" / "agent-data.jsonl")
    assert records[0]["event"] == "http_exchange"
    assert records[0]["payload"]["path"] == retriever.readiness_path
    assert records[0]["payload"]["response_body"] == {"status": "ready"}


def test_retriever_records_safe_failure_without_provider_error_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ScenarioTranscriptRoot.from_environment(
        {TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))}, sensitive_values=(_TOKEN,)
    )

    def failing_open(*_args: object, **_kwargs: object) -> object:
        raise URLError(f"provider error {_TOKEN}")

    monkeypatch.setattr(profile_module._HTTP, "open", failing_open)
    retriever = ProfileEvidenceRetriever(_settings(), transcript_root=root)
    with pytest.raises(RetrievalError):
        retriever._request(
            retriever.retrieve_path,
            method="POST",
            payload={"document": _TOKEN},
            review_run_id="run-data-failure",
        )

    record = _lines(_root(tmp_path) / "run-data-failure" / "agent-data.jsonl")[0]
    assert record["event"] == "http_failed"
    assert record["payload"] == {
        "error_code": "data_request_failed",
        "method": "POST",
        "path": retriever.retrieve_path,
    }
    assert _TOKEN not in json.dumps(record)


def test_deterministic_runtime_marks_claude_not_called_without_writing_backend_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})

    def data_request(
        _self: ProfileEvidenceRetriever,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        review_run_id: str | None = None,
    ) -> dict[str, Any]:
        assert review_run_id == "run-local-trace"
        if path == ProfileEvidenceRetriever.readiness_path:
            assert method == "GET"
            return {"status": "ready"}
        assert path == ProfileEvidenceRetriever.retrieve_path
        assert method == "POST"
        assert payload is not None
        return {
            "schema_version": "3.0",
            "request_id": payload["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        }

    monkeypatch.setattr(ProfileEvidenceRetriever, "_request", data_request)
    runtime = ProfileRuntime(_settings(), transcript_root=root)
    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-local-trace",
                "documents": [
                    {
                        "document_id": "doc-local-trace",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nHS CODE: 0303540000",
                    }
                ],
                "review_run_id": "run-local-trace",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )
    assert result.engine is not None and result.engine.kind == "deterministic"
    directory = _root(tmp_path) / "run-local-trace"
    claude = _lines(directory / "claude.jsonl")
    assert claude[0]["event"] == "not_called"
    assert claude[0]["payload"]["called"] is False
    assert not (directory / "backend-agent.jsonl").exists()


def test_claude_trace_contains_actual_prompt_and_full_text_response(tmp_path: Path) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    transcript = root.for_run("run-claude-trace")
    assert transcript is not None
    output = json.dumps(
        {
            "schema_version": "1.0",
            "document_type": "invoice",
            "classification_source_quote": "COMMERCIAL INVOICE",
            "fields": [],
            "line_items": [],
        }
    )

    class Messages:
        def create(self, **_kwargs: Any) -> object:
            return SimpleNamespace(
                content=[SimpleNamespace(text=output)],
                model_dump=lambda **_kwargs: {"id": "msg-test", "content": [{"text": output}]},
            )

    analyzer = ClaudeDocumentAnalyzer(
        ClaudeSettings(api_key=SecretStr(_TOKEN)), SimpleNamespace(messages=Messages())
    )
    result = analyzer.analyze(
        {
            "document_id": "doc-claude-trace",
            "filename": "invoice.txt",
            "content_text": "COMMERCIAL INVOICE\nHS CODE: 0303540000",
        },
        transcript=transcript,
    )
    assert result.candidate.document_type == "invoice"
    records = _lines(_root(tmp_path) / "run-claude-trace" / "claude.jsonl")
    assert [record["event"] for record in records] == ["request_started", "response_received"]
    request_payload = records[0]["payload"]
    assert request_payload["system_prompt"].startswith("You are a transcription")
    assert request_payload["messages"][0]["content"].endswith("HS CODE: 0303540000")
    assert records[1]["payload"]["output_text"] == output
    assert "provider_response" not in records[1]["payload"]


def test_claude_failure_records_safe_code_and_redacts_live_key(tmp_path: Path) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    transcript = root.for_run("run-claude-failure")
    assert transcript is not None

    class Messages:
        def create(self, **_kwargs: Any) -> object:
            raise RuntimeError(_TOKEN)

    analyzer = ClaudeDocumentAnalyzer(
        ClaudeSettings(api_key=SecretStr(_TOKEN)), SimpleNamespace(messages=Messages())
    )
    with pytest.raises(LLMProviderError):
        analyzer.analyze(
            {
                "document_id": "doc-claude-failure",
                "filename": "invoice.txt",
                "content_text": f"COMMERCIAL INVOICE\nliteral {_TOKEN}",
            },
            transcript=transcript,
        )

    records = _lines(_root(tmp_path) / "run-claude-failure" / "claude.jsonl")
    assert [record["event"] for record in records] == ["request_started", "request_failed"]
    assert records[1]["payload"] == {
        "error_code": "provider_request_failed",
        "model": analyzer.model_id,
        "provider": "anthropic",
    }
    assert _TOKEN not in json.dumps(records)
