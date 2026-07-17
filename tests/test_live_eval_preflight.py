import json
from pathlib import Path

from evals import run_claude_live


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
        "from_environment",
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
    assert report["failures"] == [
        {"document_id": "preflight", "code": "gold_set_hash_mismatch"}
    ]


def test_tampered_prompt_hash_fails_before_provider_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(run_claude_live, "_prompt_sha256", lambda: "0" * 64)
    monkeypatch.setattr(
        run_claude_live.ClaudeDocumentAnalyzer,
        "from_environment",
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
    assert report["failures"] == [
        {"document_id": "preflight", "code": "prompt_hash_mismatch"}
    ]
