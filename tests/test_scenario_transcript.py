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
    ClaudeGroundedResponder,
    ClaudeSettings,
    LLMOutputValidationError,
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
_GUIDANCE_TOKEN = "Mnbvcxzlkjhgfdsapoiuytrewq" + "0987654321ABCDEFGHIJK"
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


def test_review_transcript_cannot_claim_reserved_knowledge_namespace(tmp_path: Path) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})

    with pytest.raises(ScenarioTranscriptError, match="run_reserved"):
        root.for_run("knowledge")
    assert not (_root(tmp_path) / "knowledge").exists()


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
    monkeypatch.setattr(
        ClaudeGroundedResponder,
        "from_settings",
        classmethod(lambda cls, settings: None),
    )

    settings = _settings(
        service_mode="functional-uat",
        evidence_service_token=SecretStr(_SECOND_TOKEN),
        guidance_base_url="http://127.0.0.1:18093",
        guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
        guidance_registry_version="1.0",
    )
    runtime = ProfileRuntime(settings)
    assert calls == ["functional-uat"]
    transcript = runtime._transcript_root.for_run("run-functional-secrets")
    assert transcript is not None
    transcript.write(
        "claude",
        "test_event",
        {"content_text": f"{_TOKEN} {_SECOND_TOKEN} {_GUIDANCE_TOKEN} {_CLAUDE_KEY}"},
    )
    record = _lines(transcript.directory / "claude.jsonl")[0]
    assert record["payload"]["content_text"] == "[REDACTED] [REDACTED] [REDACTED] [REDACTED]"


def test_functional_uat_records_safe_data_skip_without_calling_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty bounded retrieval plan still owns the required Data stream."""

    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
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
    monkeypatch.setattr(
        ClaudeGroundedResponder,
        "from_settings",
        classmethod(lambda cls, settings: None),
    )
    calls: list[dict[str, Any]] = []

    def unexpected_retrieve(
        _self: ProfileEvidenceRetriever, request: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append(request)
        raise AssertionError("empty retrieval plans must not call Data")

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", unexpected_retrieve)
    runtime = ProfileRuntime(
        _settings(service_mode="functional-uat"),
        transcript_root=root,
    )
    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-functional-skip",
                "documents": [
                    {
                        "document_id": "document-functional-skip",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\\nInvoice No: INV-SKIP",
                    }
                ],
                "review_run_id": "run-functional-skip",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )

    assert calls == []
    retrieval = next(
        stage for stage in result.workflow.stages if stage.stage == "retrieve_evidence"
    )
    assert retrieval.status == "skipped"
    records = _lines(_root(tmp_path) / "run-functional-skip" / "agent-data.jsonl")
    assert len(records) == 1
    assert records[0]["event"] == "retrieve_skipped"
    assert records[0]["payload"] == {
        "reason_code": "no_hs_qualified_commodity",
        "retrieve_called": False,
    }
    assert _TOKEN not in json.dumps(records)
    assert _CLAUDE_KEY not in json.dumps(records)


def test_explicit_development_transcript_records_safe_data_skip_without_calling_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded fixture run has the required stream without a fake Data call."""

    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    calls: list[dict[str, Any]] = []

    def unexpected_retrieve(
        _self: ProfileEvidenceRetriever, request: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append(request)
        raise AssertionError("empty retrieval plans must not call Data")

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", unexpected_retrieve)
    runtime = ProfileRuntime(_settings(), transcript_root=root)
    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-development-skip",
                "documents": [
                    {
                        "document_id": "document-development-skip",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\\nInvoice No: INV-SKIP",
                    }
                ],
                "review_run_id": "run-development-skip",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )

    assert calls == []
    retrieval = next(
        stage for stage in result.workflow.stages if stage.stage == "retrieve_evidence"
    )
    assert retrieval.status == "skipped"
    records = _lines(_root(tmp_path) / "run-development-skip" / "agent-data.jsonl")
    assert len(records) == 1
    assert records[0]["schema_version"] == "scenario-transcript/v1"
    assert records[0]["event"] == "retrieve_skipped"
    assert records[0]["payload"] == {
        "reason_code": "no_hs_qualified_commodity",
        "retrieve_called": False,
    }
    assert _TOKEN not in json.dumps(records)


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
    output = (
        "```json\n"
        + json.dumps(
            {
                "schema_version": "1.0",
                "document_type": "invoice",
                "classification_source_quote": "COMMERCIAL INVOICE",
                "fields": [],
                "line_items": [],
            }
        )
        + "\n```"
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
    assert request_payload["analysis_phase"] == "initial_extraction"
    assert request_payload["system_prompt"].startswith("You are a transcription")
    assert request_payload["messages"][0]["content"].endswith("HS CODE: 0303540000")
    assert records[1]["payload"]["output_text"] == output
    assert records[1]["payload"]["analysis_phase"] == "initial_extraction"
    assert "provider_response" not in records[1]["payload"]


def test_initial_extraction_failure_is_not_marked_as_focused_reextraction(
    tmp_path: Path,
) -> None:
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
        "analysis_phase": "initial_extraction",
        "error_code": "provider_request_failed",
        "model": analyzer.model_id,
        "provider": "anthropic",
    }
    assert _TOKEN not in json.dumps(records)


def test_initial_invalid_output_is_not_marked_as_focused_reextraction(
    tmp_path: Path,
) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    transcript = root.for_run("run-initial-invalid")
    assert transcript is not None

    class Messages:
        def create(self, **_kwargs: Any) -> object:
            return SimpleNamespace(content=[SimpleNamespace(text="not a JSON document analysis")])

    analyzer = ClaudeDocumentAnalyzer(
        ClaudeSettings(api_key=SecretStr(_TOKEN)), SimpleNamespace(messages=Messages())
    )
    with pytest.raises(LLMOutputValidationError, match="invalid document analysis"):
        analyzer.analyze(
            {
                "document_id": "doc-initial-invalid",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\\nInvoice No: INV-1",
            },
            transcript=transcript,
        )

    records = _lines(_root(tmp_path) / "run-initial-invalid" / "claude.jsonl")
    assert records[-1]["event"] == "response_invalid"
    assert records[-1]["payload"] == {
        "analysis_phase": "initial_extraction",
        "error_code": "provider_output_invalid",
        "model": analyzer.model_id,
        "provider": "anthropic",
    }


def test_focused_reextraction_invalid_output_has_only_safe_phase_metadata(
    tmp_path: Path,
) -> None:
    """A verifier can identify this bounded retry without document contents."""

    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    transcript = root.for_run("run-focused-reextract-invalid")
    assert transcript is not None

    class Messages:
        def create(self, **_kwargs: Any) -> object:
            return SimpleNamespace(content=[SimpleNamespace(text="not a JSON document analysis")])

    analyzer = ClaudeDocumentAnalyzer(
        ClaudeSettings(api_key=SecretStr(_TOKEN)), SimpleNamespace(messages=Messages())
    )
    with pytest.raises(LLMOutputValidationError, match="invalid document analysis"):
        analyzer.analyze(
            {
                "document_id": "doc-focused-reextract-invalid",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\\nInvoice No: INV-1",
            },
            transcript=transcript,
            focus_fields=("quantity",),
        )

    records = _lines(_root(tmp_path) / "run-focused-reextract-invalid" / "claude.jsonl")
    assert [record["event"] for record in records] == [
        "request_started",
        "response_received",
        "response_invalid",
    ]
    assert records[0]["payload"]["analysis_phase"] == "focused_reextraction"
    assert records[1]["payload"]["analysis_phase"] == "focused_reextraction"
    assert records[2]["payload"] == {
        "analysis_phase": "focused_reextraction",
        "error_code": "provider_output_invalid",
        "model": analyzer.model_id,
        "provider": "anthropic",
    }
    assert "content_text" not in records[2]["payload"]
    assert "focus_fields" not in records[2]["payload"]


def test_grounded_responder_records_one_redacted_chat_exchange(tmp_path: Path) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    transcript = root.for_run("run-grounded-chat")
    assert transcript is not None
    output = (
        "```JSON\n"
        + json.dumps(
            {
                "answer_scope": "case_grounded",
                "direct_answer": "근거를 확인했습니다.",
                "citations": [{"kind": "comparison", "reference_id": "comparison-1"}],
                "suggested_questions": [],
                "correction_draft": None,
            }
        )
        + "\n```"
    )

    class Messages:
        def create(self, **_kwargs: Any) -> object:
            return SimpleNamespace(content=[SimpleNamespace(text=output)])

    responder = ClaudeGroundedResponder(
        ClaudeSettings(api_key=SecretStr(_TOKEN)), SimpleNamespace(messages=Messages())
    )
    assert (
        responder.respond(
            {"question": f"literal {_TOKEN}", "review_snapshot": {"case_id": "case-1"}},
            transcript=transcript,
            purpose="assistant_answer",
        )["answer_scope"]
        == "case_grounded"
    )

    records = _lines(_root(tmp_path) / "run-grounded-chat" / "claude.jsonl")
    assert [record["event"] for record in records] == [
        "assistant_answer_request_started",
        "assistant_answer_response_received",
    ]
    assert _TOKEN not in json.dumps(records)


def test_grounded_responder_records_safe_invalid_event_for_non_object_json(tmp_path: Path) -> None:
    root = ScenarioTranscriptRoot.from_environment({TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))})
    transcript = root.for_run("run-grounded-non-object")
    assert transcript is not None
    output = json.dumps([f"untrusted provider output {_TOKEN}"])

    class Messages:
        def create(self, **_kwargs: Any) -> object:
            return SimpleNamespace(content=[SimpleNamespace(text=output)])

    responder = ClaudeGroundedResponder(
        ClaudeSettings(api_key=SecretStr(_TOKEN)), SimpleNamespace(messages=Messages())
    )
    with pytest.raises(LLMOutputValidationError, match="invalid JSON object"):
        responder.respond(
            {"question": "현재 사건의 법령 근거를 알려 주세요.", "review_snapshot": {}},
            transcript=transcript,
            purpose="assistant_answer",
        )

    records = _lines(_root(tmp_path) / "run-grounded-non-object" / "claude.jsonl")
    assert [record["event"] for record in records] == [
        "assistant_answer_request_started",
        "assistant_answer_response_received",
        "assistant_answer_response_invalid",
    ]
    assert records[-1]["payload"] == {
        "error_code": "provider_output_invalid",
        "model": responder.model_id,
        "provider": "anthropic",
    }
    assert "output_text" not in records[-1]["payload"]
    assert output not in json.dumps(records[-1])
    assert _TOKEN not in json.dumps(records)
