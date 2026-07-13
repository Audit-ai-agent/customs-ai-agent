import json

from customs_ai_agent.cli import build_parser, main


def test_run_cli_accepts_optional_real_rag_url():
    parser = build_parser()

    real = parser.parse_args(
        ["run", "--scenario", "happy_path", "--rag-url", "http://localhost:8010"]
    )
    mock = parser.parse_args(["run", "--scenario", "happy_path"])

    assert real.rag_url == "http://localhost:8010"
    assert mock.rag_url is None


def test_run_cli_injects_http_retriever_and_preserves_retrieval_output(
    monkeypatch, tmp_path
):
    sentinel = object()
    captured = {}

    def fake_http_retriever(url):
        captured["url"] = url
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
            "--out",
            str(output),
        ]
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured == {"url": "http://localhost:8010", "retriever": sentinel}
    assert written["retrieval_status"] == "ok"
    assert written["knowledge_snapshot_id"] == "snapshot-1"
    assert written["report_json"]["overall_status"] == "completed"
