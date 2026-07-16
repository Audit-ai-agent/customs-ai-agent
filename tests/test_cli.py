import json
from datetime import date

import pytest

from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.cli import build_parser, main


def test_run_cli_accepts_optional_real_rag_url():
    parser = build_parser()

    real = parser.parse_args(
        ["run", "--scenario", "happy_path", "--rag-url", "http://localhost:8010"]
    )
    mock = parser.parse_args(["run", "--scenario", "happy_path"])

    assert real.rag_url == "http://localhost:8010"
    assert mock.rag_url is None


def test_run_cli_accepts_explicit_review_date():
    parsed = build_parser().parse_args(
        ["run", "--scenario", "happy_path", "--review-as-of", "2026-07-10"]
    )

    assert parsed.review_as_of == date(2026, 7, 10)


@pytest.mark.parametrize("value", ["07/10/2026", "20260710", "2026-W28-5"])
def test_run_cli_rejects_non_iso_review_date(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["run", "--scenario", "happy_path", "--review-as-of", value]
        )


def test_run_cli_injects_http_retriever_and_preserves_retrieval_output(
    monkeypatch, tmp_path
):
    sentinel = object()
    captured = {}

    def fake_http_retriever(url, **options):
        captured["url"] = url
        captured["options"] = options
        return sentinel

    def fake_run_review(request, retriever=None):
        captured["retriever"] = retriever
        return {
            "case_id": "case_happy_path",
            "final_status": "completed",
            "requires_human_review": False,
            "retrieval_status": "ok",
            "retrieval_coverage": "complete",
            "knowledge_snapshot_id": "snapshot-1",
            "retrieval_results": [{"item_id": "item-1"}],
            "report_json": {
                "overall_status": "completed",
                "summary": {"total_issues": 0},
            },
        }

    monkeypatch.setattr("customs_ai_agent.cli.HttpRegulationRetriever", fake_http_retriever)
    monkeypatch.setattr("customs_ai_agent.cli.run_review", fake_run_review)
    output = tmp_path / "review.json"

    exit_code = main(
        [
            "run",
            "--scenario",
            "happy_path",
            "--rag-url",
            "http://localhost:8010",
            "--review-as-of",
            "2026-07-10",
            "--out",
            str(output),
        ]
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured == {
        "url": "http://localhost:8010",
        "options": {"service_token": None, "token_header": "X-Demo-Token"},
        "retriever": sentinel,
    }
    assert written["retrieval_status"] == "ok"
    assert written["knowledge_snapshot_id"] == "snapshot-1"
    assert written["report_json"]["overall_status"] == "completed"


def test_run_cli_uses_env_url_token_and_provider_review_date(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_http_retriever(url, **options):
        captured["retriever"] = (url, options)
        return sentinel

    def fake_supported_date(url, **options):
        captured["readiness"] = (url, options)
        return date(2026, 7, 10)

    def fake_run_review(request, retriever=None):
        captured["request_date"] = request.review_as_of
        captured["injected"] = retriever
        return {
            "case_id": "case_happy_path",
            "final_status": "completed",
            "requires_human_review": False,
            "report_json": {
                "overall_status": "completed",
                "summary": {"total_issues": 0},
            },
        }

    monkeypatch.setenv("EVIDENCE_SERVICE_BASE_URL", "http://127.0.0.1:8010")
    monkeypatch.setenv("EVIDENCE_SERVICE_TOKEN", "local-test-secret")
    monkeypatch.setenv("EVIDENCE_SERVICE_TOKEN_HEADER", "X-Demo-Token")
    monkeypatch.setattr("customs_ai_agent.cli.HttpRegulationRetriever", fake_http_retriever)
    monkeypatch.setattr(
        "customs_ai_agent.cli.fetch_supported_review_as_of", fake_supported_date
    )
    monkeypatch.setattr("customs_ai_agent.cli.run_review", fake_run_review)

    assert main(["run", "--scenario", "happy_path"]) == 0
    options = {
        "service_token": "local-test-secret",
        "token_header": "X-Demo-Token",
    }
    assert captured == {
        "retriever": ("http://127.0.0.1:8010", options),
        "readiness": ("http://127.0.0.1:8010", options),
        "request_date": date(2026, 7, 10),
        "injected": sentinel,
    }


def test_run_cli_explicit_url_and_date_override_environment(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_http_retriever(url, **_options):
        captured["url"] = url
        return sentinel

    def fail_if_readiness_is_called(*_args, **_kwargs):
        raise AssertionError("explicit review date must bypass readiness discovery")

    def fake_run_review(request, retriever=None):
        captured["date"] = request.review_as_of
        captured["retriever"] = retriever
        return {
            "case_id": "case_happy_path",
            "final_status": "completed",
            "requires_human_review": False,
            "report_json": {
                "overall_status": "completed",
                "summary": {"total_issues": 0},
            },
        }

    monkeypatch.setenv("EVIDENCE_SERVICE_BASE_URL", "http://127.0.0.1:9999")
    monkeypatch.setattr("customs_ai_agent.cli.HttpRegulationRetriever", fake_http_retriever)
    monkeypatch.setattr(
        "customs_ai_agent.cli.fetch_supported_review_as_of",
        fail_if_readiness_is_called,
    )
    monkeypatch.setattr("customs_ai_agent.cli.run_review", fake_run_review)

    assert main(
        [
            "run",
            "--scenario",
            "happy_path",
            "--rag-url",
            "http://localhost:8010",
            "--review-as-of",
            "2026-07-09",
        ]
    ) == 0
    assert captured == {
        "url": "http://localhost:8010",
        "date": date(2026, 7, 9),
        "retriever": sentinel,
    }


def test_run_cli_does_not_fall_back_to_mock_when_readiness_fails(monkeypatch):
    monkeypatch.setenv("EVIDENCE_SERVICE_BASE_URL", "http://127.0.0.1:8010")
    monkeypatch.setattr(
        "customs_ai_agent.cli.HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.fetch_supported_review_as_of",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RetrievalError("Evidence Service readiness request failed: network error")
        ),
    )
    monkeypatch.setattr(
        "customs_ai_agent.cli.run_review",
        lambda *_args, **_kwargs: pytest.fail("review must not run after readiness failure"),
    )

    with pytest.raises(RetrievalError, match="readiness request failed"):
        main(["run", "--scenario", "happy_path"])


def test_unavailable_llm_mode_fails_before_live_rag_network(monkeypatch):
    monkeypatch.delenv("CUSTOMS_AGENT_ENABLE_LLM", raising=False)
    monkeypatch.setenv("EVIDENCE_SERVICE_BASE_URL", "http://127.0.0.1:8010")
    monkeypatch.setattr(
        "customs_ai_agent.cli.fetch_supported_review_as_of",
        lambda *_args, **_kwargs: pytest.fail("unavailable LLM mode must not call Data"),
    )

    with pytest.raises(RuntimeError, match="CUSTOMS_AGENT_ENABLE_LLM is not set"):
        main(["run", "--scenario", "happy_path", "--llm-mode", "llm"])
