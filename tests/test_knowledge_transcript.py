"""Local-only transcript coverage for the standalone knowledge assistant."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

import customs_ai_agent.scenario_transcript as transcript_module
import customs_ai_agent.service.profiles as profile_module
from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    ClaudeGroundedResponder,
    ClaudeSettings,
)
from customs_ai_agent.scenario_transcript import (
    KNOWLEDGE_TRANSCRIPT_ROOT_ENV,
    KnowledgeTranscriptRoot,
    KnowledgeTranscriptTurn,
    ScenarioTranscriptError,
    ScenarioTranscriptRun,
)
from customs_ai_agent.service.profiles import ProfileRuntime
from customs_ai_agent.service.runtime_errors import RuntimeInputError
from customs_ai_agent.service.settings import ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
_SECOND_TOKEN = "Zyxwvutsrqponmlkjihgfedcba" + "9876543210ABCDEFGHIJK"
_GUIDANCE_TOKEN = "Mnbvcxzlkjhgfdsapoiuytrewq" + "0987654321ABCDEFGHIJK"
_CLAUDE_KEY = "sk" + "-ant-" + "knowledge-transcript-key-abcdefghijklmnop"
_REGISTRY_SHA256 = "6fed779ee0ec1cd2358972925fe1b2788838a197d4f33e62e461fcfed2bb7d6c"
# Keep this list aligned with the documented set recognised by
# ``str.splitlines()``.  The boundary must not be limited to CR/LF.
_SPLITLINES_BOUNDARIES = (
    "\n",
    "\r",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
)


def _root(tmp_path: Path) -> Path:
    """Preserved explicit-development root; Functional-UAT uses a new namespace."""

    return tmp_path / "_local" / "state" / "knowledge-runs"


def _functional_root(tmp_path: Path) -> Path:
    return tmp_path / "_local" / "state" / "scenario-runs" / "knowledge"


def _settings(**changes: Any) -> ServiceSettings:
    return replace(
        ServiceSettings(
            bearer_token=SecretStr(_TOKEN),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(_SECOND_TOKEN),
        ),
        **changes,
    )


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(value) for value in path.read_text(encoding="utf-8").splitlines()]


def _response(
    request_id: str,
    *,
    secret: str = "",
    official_law: bool = False,
) -> dict[str, Any]:
    source = {
        "source_id": "guide_" + "a" * 64,
        "registry_source_id": "law_001556",
        "source_kind": "official_law" if official_law else "development_fixture",
        "title": "관세법 제1조" if official_law else "통관 절차의 기본 안내",
        "uri": (
            "https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EA%B4%80%EC%84%B8%EB%B2%95/%EC%A0%9C1%EC%A1%B0"
            if official_law
            else "https://customs.local.invalid/dev-fixture/guidance/customs_clearance"
        ),
        "topic": "customs_clearance",
        "direction": "both",
        "excerpt": f"통관은 신고 내용과 관련 서류를 확인하는 절차입니다. {secret}".strip(),
        "version_hash": "a" * 64,
        "retrieval_methods": ["registry"],
    }
    if official_law:
        source.update(
            {
                "law_id": "001556",
                "law_name": "관세법",
                "article_no": "제1조",
            }
        )
    return {
        "schema_version": "3.0",
        "request_id": request_id,
        "status": "covered",
        "snapshot_id": "dev-fixture-guidance",
        "as_of": "2026-08-05",
        "coverage": "covered",
        "registry_version": "1.0",
        "registry_sha256": _REGISTRY_SHA256,
        "topic": "customs_clearance",
        "results": [source],
        "errors": [],
    }


def _knowledge_payload(
    *,
    question: str = "수입 통관 절차를 안내해 주세요.",
    recent_turns: list[dict[str, str]] | None = None,
) -> bytes:
    return json.dumps(
        {
            "schema_version": "1.0",
            "conversation_id": "knowledge-conversation",
            "question": question,
            "trade_type": "both",
            "as_of": "2026-08-05",
            "recent_turns": recent_turns or [],
        }
    ).encode("utf-8")


def test_knowledge_transcript_uses_an_isolated_safe_turn_layout_and_redacts(
    tmp_path: Path,
) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))},
        sensitive_values=(_TOKEN,),
    )
    turn = root.for_turn("conversation-safe", "turn-" + "a" * 32)
    assert turn is not None
    assert turn.directory == _root(tmp_path) / "conversation-safe" / ("turn-" + "a" * 32)

    turn.write(
        "agent-data",
        "http_exchange",
        {
            "query": f"통관 절차와 literal={_TOKEN}",
            "authorization": "must-not-be-recorded",
        },
    )
    record = _lines(turn.directory / "agent-data.jsonl")[0]
    assert record["payload"]["query"] == "통관 절차와 literal=[REDACTED]"
    assert record["payload"]["authorization"] == "[REDACTED]"

    with pytest.raises(ScenarioTranscriptError, match="payload_forbidden"):
        turn.write("agent-data", "http_exchange", {"review_result": {"case_id": "case-1"}})
    with pytest.raises(ScenarioTranscriptError, match="conversation_id_invalid"):
        root.for_turn("../outside", "turn-" + "b" * 32)
    with pytest.raises(ScenarioTranscriptError, match="turn_id_invalid"):
        root.for_turn("conversation-safe", "run-unsafe")


def test_knowledge_functional_uat_requires_the_fixed_local_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _functional_root(tmp_path)
    monkeypatch.setattr(
        transcript_module,
        "_functional_uat_knowledge_directory",
        lambda: expected,
    )

    root = KnowledgeTranscriptRoot.for_service_mode("functional-uat", environ={})
    assert root.directory == expected.resolve()
    same = KnowledgeTranscriptRoot.for_service_mode(
        "functional-uat", {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(expected)}
    )
    assert same.directory == root.directory
    with pytest.raises(ScenarioTranscriptError, match="root_invalid"):
        KnowledgeTranscriptRoot.for_service_mode(
            "functional-uat",
            {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(tmp_path / "wrong" / "knowledge-runs")},
        )


def test_ordinary_development_keeps_standalone_knowledge_transcript_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def guidance(_self: object, request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        calls.append(request)
        return _response(request["request_id"])

    monkeypatch.setattr(profile_module.ProfileEvidenceRetriever, "guidance_query", guidance)
    runtime = ProfileRuntime(
        _settings(),
        knowledge_transcript_root=KnowledgeTranscriptRoot(),
    )
    answer = runtime.knowledge_answer_json(_knowledge_payload())

    assert answer.answer_scope == "general_customs_guidance"
    assert len(calls) == 1
    assert not _root(tmp_path).exists()


@pytest.mark.parametrize(
    "payload_update",
    [
        {"question": "상업송장\n송장번호: INV-001"},
        {"question": "comparison-1515e58을 설명해 주세요."},
        {"question": "원산지증명서.txt의 내용을 확인해 주세요."},
        {
            "recent_turns": [
                {
                    "role": "user",
                    "content": "원산지증명서\nHS 코드: 0303540000",
                }
            ]
        },
        *[
            {"question": f"일반 통관 절차를 알려 주세요.{boundary}"}
            for boundary in _SPLITLINES_BOUNDARIES
        ],
        *[
            {"recent_turns": [{"role": "user", "content": f"이전 일반 질문{boundary}"}]}
            for boundary in _SPLITLINES_BOUNDARIES
        ],
    ],
)
def test_unsafe_standalone_input_is_rejected_before_data_or_transcript_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_update: dict[str, Any],
) -> None:
    """A direct runtime caller cannot bypass the HTTP knowledge boundary."""

    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))}
    )
    data_calls: list[dict[str, Any]] = []

    def guidance(_self: object, request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        data_calls.append(request)
        raise AssertionError("rejected knowledge input must not reach Data")

    monkeypatch.setattr(profile_module.ProfileEvidenceRetriever, "guidance_query", guidance)
    runtime = ProfileRuntime(_settings(), knowledge_transcript_root=root)
    payload = json.loads(_knowledge_payload())
    payload.update(payload_update)

    with pytest.raises(RuntimeInputError) as raised:
        runtime.knowledge_answer_json(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    assert raised.value.code == "knowledge_input_boundary"
    assert "상업송장" not in str(raised.value)
    assert "원산지증명서" not in str(raised.value)
    assert data_calls == []
    assert list(_root(tmp_path).iterdir()) == []


def test_explicit_development_transcript_records_one_guidance_exchange_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))}, sensitive_values=(_SECOND_TOKEN,)
    )
    calls: list[str] = []

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
            return json.dumps(_response("placeholder", secret=_SECOND_TOKEN)).encode("utf-8")

    def open_request(request: Any, **_kwargs: Any) -> Response:
        calls.append(request.full_url)
        body = json.loads(request.data.decode("utf-8"))
        response = _response(body["request_id"], secret=_SECOND_TOKEN)

        class BoundResponse(Response):
            def read(self, _limit: int) -> bytes:
                return json.dumps(response).encode("utf-8")

        return BoundResponse()

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    runtime = ProfileRuntime(_settings(), knowledge_transcript_root=root)
    answer = runtime.knowledge_answer_json(
        _knowledge_payload(question=f"수입 통관 절차 {_SECOND_TOKEN}")
    )

    assert answer.answer_scope == "general_customs_guidance"
    assert len(calls) == 1
    turn_directories = list((_root(tmp_path) / "knowledge-conversation").iterdir())
    assert len(turn_directories) == 1
    data_records = _lines(turn_directories[0] / "agent-data.jsonl")
    assert len(data_records) == 1
    assert data_records[0]["event"] == "http_exchange"
    assert data_records[0]["payload"]["path"] == "/internal/rag/v3/guidance/query"
    serialized = json.dumps(data_records, ensure_ascii=False)
    assert _SECOND_TOKEN not in serialized
    assert "case_id" not in serialized
    assert "review_run_id" not in serialized
    assert "document_id" not in serialized
    assert not (turn_directories[0] / "claude.jsonl").exists()


def test_functional_knowledge_turn_records_one_data_and_one_claude_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_functional_root(tmp_path))}
    )
    claude_settings = ClaudeSettings(api_key=SecretStr(_CLAUDE_KEY))
    claude_calls: list[dict[str, Any]] = []

    class Messages:
        def create(self, **kwargs: Any) -> object:
            claude_calls.append(kwargs)
            output = json.dumps(
                {
                    "presentation_style": "source_focus",
                    "source_slots": [0],
                    "suggestion_topics": ["trade_documents"],
                },
                ensure_ascii=False,
            )
            return SimpleNamespace(content=[SimpleNamespace(text=output)])

    responder = ClaudeGroundedResponder(claude_settings, SimpleNamespace(messages=Messages()))
    monkeypatch.setattr(
        ClaudeSettings,
        "from_environment",
        classmethod(lambda _cls: claude_settings),
    )
    monkeypatch.setattr(
        ClaudeDocumentAnalyzer,
        "from_settings",
        classmethod(lambda _cls, _settings: profile_module._DeterministicDocumentAnalyzer()),
    )
    monkeypatch.setattr(
        ClaudeGroundedResponder,
        "from_settings",
        classmethod(lambda _cls, _settings: responder),
    )

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    data_calls: list[dict[str, Any]] = []

    def open_request(request: Any, **_kwargs: Any) -> object:
        body = json.loads(request.data.decode("utf-8"))
        data_calls.append({"url": request.full_url, "body": body})
        response = _response(body["request_id"], secret=_GUIDANCE_TOKEN, official_law=True)

        class Response:
            headers = Headers({"X-Customs-Service-Mode": "functional-uat"})

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return json.dumps(response, ensure_ascii=False).encode("utf-8")

        return Response()

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    runtime = ProfileRuntime(
        _settings(
            service_mode="functional-uat",
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="1.0",
        ),
        knowledge_transcript_root=root,
    )
    answer = runtime.knowledge_answer_json(
        _knowledge_payload(
            question=f"수입 통관 절차를 안내해 주세요. {_TOKEN} {_GUIDANCE_TOKEN} {_CLAUDE_KEY}",
            recent_turns=[
                {"role": "assistant", "content": "이전 일반 안내의 고유 문장: 이전-통관-범위-문구"},
                {"role": "user", "content": "추가 일반 질문의 고유 문장: 사용자-최근-질문-문구"},
            ],
        )
    )

    assert answer.engine.kind == "anthropic"
    assert "공식 자료 발췌" in answer.direct_answer
    assert len(data_calls) == 1
    assert len(claude_calls) == 1
    raw_question = f"수입 통관 절차를 안내해 주세요. {_TOKEN} {_GUIDANCE_TOKEN} {_CLAUDE_KEY}"
    assert raw_question not in json.dumps(data_calls, ensure_ascii=False)
    provider_serialized = json.dumps(claude_calls, ensure_ascii=False)
    assert raw_question not in provider_serialized
    assert "recent_turns" not in provider_serialized
    assert "이전-통관-범위-문구" not in provider_serialized
    assert "사용자-최근-질문-문구" not in provider_serialized
    assert "guidance_presentation" in provider_serialized
    turn_directory = next((_functional_root(tmp_path) / "knowledge-conversation").iterdir())
    data_records = _lines(turn_directory / "agent-data.jsonl")
    claude_records = _lines(turn_directory / "claude.jsonl")
    assert [item["event"] for item in data_records] == ["http_exchange"]
    assert [item["event"] for item in claude_records] == [
        "knowledge_answer_request_started",
        "knowledge_answer_response_received",
    ]
    serialized = json.dumps([*data_records, *claude_records], ensure_ascii=False)
    for marker in (
        _TOKEN,
        _SECOND_TOKEN,
        _GUIDANCE_TOKEN,
        _CLAUDE_KEY,
        "case_id",
        "review_run_id",
        "document_id",
        "이전-통관-범위-문구",
        "사용자-최근-질문-문구",
    ):
        assert marker not in serialized


@pytest.mark.parametrize(
    "unsafe_payload",
    [
        _knowledge_payload(question="COMMERCIAL INVOICE\nInvoice No: INV-001"),
        _knowledge_payload(question="일반 통관 절차\u2028추가 내용"),
        _knowledge_payload(
            recent_turns=[{"role": "user", "content": "이전 일반 질문\u2029추가 내용"}]
        ),
    ],
    ids=["document", "unicode-line-separator-question", "unicode-paragraph-separator-history"],
)
def test_functional_unsafe_input_reaches_neither_data_nor_claude_or_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_payload: bytes
) -> None:
    """Functional-UAT must reject before any external capability is considered."""

    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_functional_root(tmp_path))}
    )
    claude_settings = ClaudeSettings(api_key=SecretStr(_CLAUDE_KEY))
    claude_calls: list[dict[str, Any]] = []
    data_calls: list[dict[str, Any]] = []

    class Messages:
        def create(self, **kwargs: Any) -> object:
            claude_calls.append(kwargs)
            raise AssertionError("rejected knowledge input must not reach Claude")

    responder = ClaudeGroundedResponder(claude_settings, SimpleNamespace(messages=Messages()))
    monkeypatch.setattr(
        ClaudeSettings,
        "from_environment",
        classmethod(lambda _cls: claude_settings),
    )
    monkeypatch.setattr(
        ClaudeDocumentAnalyzer,
        "from_settings",
        classmethod(lambda _cls, _settings: profile_module._DeterministicDocumentAnalyzer()),
    )
    monkeypatch.setattr(
        ClaudeGroundedResponder,
        "from_settings",
        classmethod(lambda _cls, _settings: responder),
    )

    def guidance(_self: object, request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        data_calls.append(request)
        raise AssertionError("rejected knowledge input must not reach Data")

    monkeypatch.setattr(profile_module.ProfileEvidenceRetriever, "guidance_query", guidance)
    runtime = ProfileRuntime(
        _settings(service_mode="functional-uat"), knowledge_transcript_root=root
    )

    with pytest.raises(RuntimeInputError) as raised:
        runtime.knowledge_answer_json(unsafe_payload)

    assert raised.value.code == "knowledge_input_boundary"
    assert data_calls == []
    assert claude_calls == []
    assert list(_functional_root(tmp_path).iterdir()) == []


def test_functional_knowledge_turn_records_safe_claude_non_call_for_no_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_functional_root(tmp_path))}
    )
    claude_settings = ClaudeSettings(api_key=SecretStr(_CLAUDE_KEY))
    monkeypatch.setattr(
        ClaudeSettings,
        "from_environment",
        classmethod(lambda _cls: claude_settings),
    )
    monkeypatch.setattr(
        ClaudeDocumentAnalyzer,
        "from_settings",
        classmethod(lambda _cls, _settings: profile_module._DeterministicDocumentAnalyzer()),
    )
    monkeypatch.setattr(
        ClaudeGroundedResponder,
        "from_settings",
        classmethod(lambda _cls, _settings: None),
    )

    def no_coverage(
        _self: object,
        request: dict[str, Any],
        *,
        transcript: KnowledgeTranscriptTurn | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert transcript is not None
        transcript.write(
            "agent-data",
            "http_exchange",
            {"path": "/internal/rag/v3/guidance/query", "status": "no_coverage"},
        )
        response = _response(request["request_id"])
        response.update({"status": "no_coverage", "coverage": "no_coverage", "results": []})
        return response

    monkeypatch.setattr(profile_module.ProfileEvidenceRetriever, "guidance_query", no_coverage)
    runtime = ProfileRuntime(
        _settings(
            service_mode="functional-uat",
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="1.0",
        ),
        knowledge_transcript_root=root,
    )
    answer = runtime.knowledge_answer_json(_knowledge_payload())

    assert answer.answer_scope == "insufficient_evidence"
    turn_directory = next((_functional_root(tmp_path) / "knowledge-conversation").iterdir())
    claude_records = _lines(turn_directory / "claude.jsonl")
    assert claude_records == [
        {
            "schema_version": "scenario-transcript/v1",
            "recorded_at": claude_records[0]["recorded_at"],
            "event": "knowledge_answer_not_called",
            "payload": {
                "provider": "claude",
                "called": False,
                "reason_code": "official_guidance_coverage_unavailable",
            },
        }
    ]


def test_knowledge_transcript_write_failure_fails_closed_before_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))}
    )

    def failed_write(
        _self: ScenarioTranscriptRun, _stream: str, _event: str, _payload: dict[str, Any]
    ) -> None:
        raise ScenarioTranscriptError("scenario_transcript_write_failed")

    monkeypatch.setattr(ScenarioTranscriptRun, "write", failed_write)
    turn = root.for_turn("conversation-failure", "turn-" + "f" * 32)
    assert turn is not None
    with pytest.raises(ScenarioTranscriptError, match="knowledge_transcript_write_failed"):
        turn.write("agent-data", "http_exchange", {"query": "통관 절차"})


def test_runtime_stops_when_an_enabled_knowledge_transcript_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_root(tmp_path))}
    )
    calls: list[dict[str, Any]] = []

    def guidance(
        _self: object,
        request: dict[str, Any],
        *,
        transcript: KnowledgeTranscriptTurn | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(request)
        assert transcript is not None
        transcript.write("agent-data", "http_exchange", {"query": request["query"]})
        return _response(request["request_id"])

    def failed_write(
        _self: ScenarioTranscriptRun, _stream: str, _event: str, _payload: dict[str, Any]
    ) -> None:
        raise ScenarioTranscriptError("scenario_transcript_write_failed")

    monkeypatch.setattr(profile_module.ProfileEvidenceRetriever, "guidance_query", guidance)
    monkeypatch.setattr(ScenarioTranscriptRun, "write", failed_write)
    runtime = ProfileRuntime(_settings(), knowledge_transcript_root=root)

    with pytest.raises(ScenarioTranscriptError, match="knowledge_transcript_write_failed"):
        runtime.knowledge_answer_json(_knowledge_payload())
    assert len(calls) == 1
