"""Strict local verification coverage for standalone knowledge transcripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customs_ai_agent.knowledge_transcript_verifier import (
    KnowledgeTranscriptVerificationError,
    main,
    verify_knowledge_transcript_turn,
)
from customs_ai_agent.scenario_transcript import (
    KNOWLEDGE_TRANSCRIPT_ROOT_ENV,
    KnowledgeTranscriptRoot,
)


def _functional_root(tmp_path: Path) -> Path:
    return tmp_path / "_local" / "state" / "scenario-runs" / "knowledge"


def _legacy_root(tmp_path: Path) -> Path:
    return tmp_path / "_local" / "state" / "knowledge-runs"


def _safe_record(*, payload: dict[str, object]) -> str:
    return (
        json.dumps(
            {
                "schema_version": "scenario-transcript/v1",
                "recorded_at": "2026-08-08T00:00:00+00:00",
                "event": "knowledge_answer_response_received",
                "payload": payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _functional_turn(tmp_path: Path) -> Path:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_functional_root(tmp_path))}
    )
    turn = root.for_turn("knowledge-conversation", "turn-" + "a" * 32)
    assert turn is not None
    turn.write("agent-data", "http_exchange", {"query": "통관 절차 안내"})
    turn.write("claude", "knowledge_answer_response_received", {"answer": "공식 근거 안내"})
    return turn.directory


def test_verifier_accepts_only_complete_functional_uat_turns(tmp_path: Path) -> None:
    turn = _functional_turn(tmp_path)

    verify_knowledge_transcript_turn(turn, service_mode="functional-uat")
    assert main(["--turn-directory", str(turn), "--service-mode", "functional-uat"]) == 0


def test_verifier_requires_claude_stream_for_functional_uat(tmp_path: Path) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_functional_root(tmp_path))}
    )
    turn = root.for_turn("knowledge-conversation", "turn-" + "b" * 32)
    assert turn is not None
    turn.write("agent-data", "http_exchange", {"query": "통관 절차 안내"})

    with pytest.raises(KnowledgeTranscriptVerificationError, match="required_stream_missing"):
        verify_knowledge_transcript_turn(turn.directory, service_mode="functional-uat")


@pytest.mark.parametrize(
    "payload",
    [
        {"document_text": "원문을 넣으면 안 됩니다."},
        {"original_document_text": "원문을 넣으면 안 됩니다."},
        {"api_key": "not-redacted-credential"},
        {"answer": "Bearer Abcdefghijklmnopqrstuvwxyz0123456789"},
    ],
)
def test_verifier_rejects_case_data_and_secret_patterns(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    turn = _functional_turn(tmp_path)
    unsafe = _safe_record(payload=payload)
    (turn / "claude.jsonl").write_text(unsafe, encoding="utf-8", newline="\n")

    with pytest.raises(KnowledgeTranscriptVerificationError, match="sensitive_or_case_data"):
        verify_knowledge_transcript_turn(turn, service_mode="functional-uat")


def test_verifier_rejects_legacy_root_for_functional_uat_but_preserves_local_checks(
    tmp_path: Path,
) -> None:
    root = KnowledgeTranscriptRoot.from_environment(
        {KNOWLEDGE_TRANSCRIPT_ROOT_ENV: str(_legacy_root(tmp_path))}
    )
    turn = root.for_turn("legacy-conversation", "turn-" + "c" * 32)
    assert turn is not None
    turn.write("agent-data", "http_exchange", {"query": "통관 절차 안내"})

    verify_knowledge_transcript_turn(turn.directory, service_mode="local-development")
    with pytest.raises(KnowledgeTranscriptVerificationError, match="root_invalid"):
        verify_knowledge_transcript_turn(turn.directory, service_mode="functional-uat")


def test_verifier_cli_never_echoes_supplied_path_or_transcript_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    turn = _functional_turn(tmp_path)
    status = main(["--turn-directory", str(turn), "--service-mode", "functional-uat"])
    captured = capsys.readouterr()

    assert status == 0
    assert captured.out == '{"status":"passed"}\n'
    assert str(turn) not in captured.out
    assert "공식 근거 안내" not in captured.out
