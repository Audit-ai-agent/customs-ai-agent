import json
from pathlib import Path

from evals import run_claude_live
from customs_ai_agent.adapters.llm import LLMError


GOLD_PATH = Path(__file__).resolve().parents[1] / "evals" / "claude_text_gold_v1.json"


def _forbid_analyzer_creation(*_args, **_kwargs):
    raise AssertionError("tampered evaluation must fail before provider setup")


def test_tampered_gold_fails_before_provider_setup(monkeypatch, tmp_path):
    tampered = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    tampered["documents"]["doc_bl"]["fields"]["bl_no"] = "TAMPERED"
    tampered_path = tmp_path / "tampered-gold.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(
        run_claude_live.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_live.evaluate(
        env_file=tmp_path / "unused.env",
        gold_path=tampered_path,
        max_documents=1,
        run_workflow=False,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["failures"] == [{"document_id": "preflight", "code": "gold_set_hash_mismatch"}]


def test_tampered_prompt_hash_fails_before_provider_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(run_claude_live, "_prompt_sha256", lambda: "0" * 64)
    monkeypatch.setattr(
        run_claude_live.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_live.evaluate(
        env_file=tmp_path / "unused.env",
        gold_path=GOLD_PATH,
        max_documents=1,
        run_workflow=False,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["failures"] == [{"document_id": "preflight", "code": "prompt_hash_mismatch"}]


def test_valid_gold_without_live_flag_never_creates_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_claude_live.ClaudeDocumentAnalyzer,
        "from_settings",
        _forbid_analyzer_creation,
    )

    exit_code, report = run_claude_live.evaluate(
        env_file=tmp_path / "unused.env",
        gold_path=GOLD_PATH,
        max_documents=1,
        run_workflow=False,
        live=False,
        max_calls=1,
        max_total_input_tokens=100_000,
        max_total_output_tokens=8_192,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["provider_called"] is False
    assert report["failures"] == [{"document_id": "preflight", "code": "live_flag_required"}]


def test_failed_provider_call_consumes_budget_without_sdk_retry(
    monkeypatch,
    tmp_path,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=unit-test-placeholder\n"
        "CUSTOMS_AGENT_CLAUDE_MAX_TOKENS=1024\n"
        "CUSTOMS_AGENT_CLAUDE_MAX_RETRIES=3\n",
        encoding="utf-8",
    )
    captured_settings = []

    class FailingAnalyzer:
        def analyze(self, _document):
            raise LLMError("provider_unavailable", "simulated provider failure")

    def analyzer_factory(settings):
        captured_settings.append(settings)
        return FailingAnalyzer()

    monkeypatch.setattr(
        run_claude_live.ClaudeDocumentAnalyzer,
        "from_settings",
        staticmethod(analyzer_factory),
    )

    exit_code, report = run_claude_live.evaluate(
        env_file=env_file,
        gold_path=GOLD_PATH,
        max_documents=1,
        run_workflow=False,
        live=True,
        max_calls=1,
        max_total_input_tokens=100_000,
        max_total_output_tokens=1_024,
    )

    assert exit_code == 1
    assert report["failures"] == [{"document_id": "doc_bl", "code": "provider_unavailable"}]
    assert report["call_accounting"] == {
        "attempted": 1,
        "succeeded": 0,
        "retried": 0,
    }
    assert report["cost_budget"]["configured_max_retries"] == 0
    assert len(captured_settings) == 1
    assert captured_settings[0].max_retries == 0
