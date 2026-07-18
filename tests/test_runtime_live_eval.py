from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from evals import run_runtime_live
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.schemas.models import ReviewRunRequestV2, ReviewRunResultV2


def _env_file(tmp_path: Path, *, allowed_origins: str = "") -> Path:
    path = tmp_path / "private-runtime.env"
    path.write_text(
        "ANTHROPIC_API_KEY=private-anthropic-test-secret\n"
        "EVIDENCE_SERVICE_TOKEN=private-evidence-test-secret\n"
        "EVIDENCE_SERVICE_TOKEN_HEADER=X-Private-Evidence-Token\n"
        f"EVIDENCE_SERVICE_ALLOWED_ORIGINS={allowed_origins}\n",
        encoding="utf-8",
    )
    return path


def _live_result(
    scenario_id: str,
    *,
    prompt_sha256: str | None = None,
) -> ReviewRunResultV2:
    result = run_review_result_v2(ReviewRunRequestV2(scenario_id=scenario_id))
    payload = result.model_dump(mode="json")
    prompt_hash = prompt_sha256 or run_runtime_live._text_prompt_sha256()
    output_schema_hash = run_runtime_live._text_output_schema_sha256()
    payload["model_calls"] = [
        {
            "document_id": document.document_id,
            "provider": "anthropic",
            "model_id": run_runtime_live.DEFAULT_CLAUDE_MODEL,
            "prompt_id": run_runtime_live.DOCUMENT_ANALYSIS_PROMPT_ID,
            "prompt_version": run_runtime_live.DOCUMENT_ANALYSIS_PROMPT_VERSION,
            "prompt_sha256": prompt_hash,
            "output_schema_id": run_runtime_live.TEXT_OUTPUT_SCHEMA_ID,
            "output_schema_version": run_runtime_live.CLAUDE_CANDIDATE_SCHEMA_VERSION,
            "output_schema_sha256": output_schema_hash,
            "request_id": f"request-{index}",
            "message_id": f"message-{index}",
            "stop_reason": "end_turn",
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "latency_ms": 25,
            "input_modality": "text",
        }
        for index, document in enumerate(result.document_results, start=1)
    ]
    return ReviewRunResultV2.model_validate(payload)


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    result: ReviewRunResultV2,
    captured: dict[str, Any],
) -> None:
    class FakeConfig:
        @classmethod
        def from_environment(cls, **options: Any) -> object:
            captured.setdefault("order", []).append("runtime_config")
            captured["config_options"] = options
            return object()

    class FakeRuntime:
        def __init__(self, config: object) -> None:
            captured["config"] = config

        def review(self, request: ReviewRunRequestV2) -> ReviewRunResultV2:
            captured.setdefault("order", []).append("agent_runtime")
            captured["request"] = request
            return result

    monkeypatch.setattr(run_runtime_live, "RuntimeConfig", FakeConfig)
    monkeypatch.setattr(run_runtime_live, "AgentRuntime", FakeRuntime)


def _forbid_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: pytest.fail("readiness must not run"),
    )

    class ForbiddenConfig:
        @classmethod
        def from_environment(cls, **_kwargs: Any) -> object:
            pytest.fail("runtime configuration must not run")

    class ForbiddenRuntime:
        def __init__(self, _config: object) -> None:
            pytest.fail("AgentRuntime must not be constructed")

    monkeypatch.setattr(run_runtime_live, "RuntimeConfig", ForbiddenConfig)
    monkeypatch.setattr(run_runtime_live, "AgentRuntime", ForbiddenRuntime)


def test_without_live_flag_never_reads_env_or_calls_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _forbid_live_dependencies(monkeypatch)
    missing_env = tmp_path / "does-not-exist.env"

    exit_code, report = run_runtime_live.evaluate(
        env_file=missing_env,
        live=False,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["readiness_status"] == "NOT_ATTEMPTED"
    assert report["agent_execution_passed"] is False
    assert report["release_ready"] is False
    assert report["errors"] == [{"code": "live_flag_required"}]


def test_ready_allowlisted_service_runs_public_runtime_after_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_url = "https://evidence.example"
    env_file = _env_file(tmp_path, allowed_origins=base_url)
    result = _live_result("happy_path")
    captured: dict[str, Any] = {"order": []}

    def fake_readiness(url: str, **options: Any) -> date:
        captured["order"].append("readiness")
        captured["readiness"] = (url, options)
        return date(2026, 7, 10)

    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        fake_readiness,
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        base_url=base_url,
        live=True,
    )

    request = captured["request"]
    config_options = captured["config_options"]
    readiness_url, readiness_options = captured["readiness"]
    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["readiness_status"] == "READY"
    assert report["agent_execution_passed"] is True
    assert report["release_ready"] is True
    assert report["errors"] == []
    assert report["counts"]["document_count"] == 4
    assert report["counts"]["document_completed_count"] == 4
    assert report["counts"]["model_call_count"] == 4
    assert report["models"] == [
        {
            "provider": "anthropic",
            "model_id": run_runtime_live.DEFAULT_CLAUDE_MODEL,
            "call_count": 4,
        }
    ]
    assert captured["order"] == ["readiness", "runtime_config", "agent_runtime"]
    assert readiness_url == base_url
    assert readiness_options["allowed_origins"] == frozenset({base_url})
    assert readiness_options["service_token"] == "private-evidence-test-secret"
    assert readiness_options["contract_identity_mode"] == "required"
    assert config_options["evidence_base_url"] == base_url
    assert config_options["env_file"] == env_file
    assert config_options["evidence_allowed_origins"] == frozenset({base_url})
    assert (
        config_options["evidence_service_token"].get_secret_value()
        == "private-evidence-test-secret"
    )
    assert isinstance(request, ReviewRunRequestV2)
    assert request.scenario_id is None
    assert request.case_id == "case_evidence_live_exact_hsk10"
    assert request.documents is not None
    assert len(request.documents) == run_runtime_live.EXPECTED_DOCUMENT_COUNT
    assert request.llm_mode == "llm"
    assert request.review_as_of == date(2026, 7, 10)


def test_unready_503_diagnostic_executes_agent_and_stays_non_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    result = _live_result("evidence_live_exact_hsk10")
    captured: dict[str, Any] = {"order": []}

    def unready(*_args: Any, **_kwargs: Any) -> date:
        captured["order"].append("readiness")
        raise RetrievalError("Evidence Service readiness request failed: HTTP status 503")

    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        unready,
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
        allow_unready_diagnostic=True,
        diagnostic_review_as_of=date(2026, 7, 10),
    )

    assert exit_code == 1
    assert report["status"] == "BLOCKED_DATA_READINESS"
    assert report["readiness_status"] == "UNREADY_503_DIAGNOSTIC"
    assert report["agent_execution_passed"] is True
    assert report["release_ready"] is False
    assert report["result_status"] == "needs_review"
    assert report["requires_human_review"] is True
    assert report["errors"] == [{"code": "evidence_readiness_incomplete"}]
    assert captured["order"] == ["readiness", "runtime_config", "agent_runtime"]
    request = captured["request"]
    assert request.scenario_id is None
    assert request.case_id == "case_evidence_live_exact_hsk10"
    assert request.documents is not None
    assert len(request.documents) == run_runtime_live.EXPECTED_DOCUMENT_COUNT
    assert request.review_as_of == date(2026, 7, 10)
    assert request.llm_mode == "llm"


def test_unready_503_requires_explicit_date_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)

    def unready(*_args: Any, **_kwargs: Any) -> date:
        raise RetrievalError("Evidence Service readiness request failed: HTTP status 503")

    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        unready,
    )

    class ForbiddenConfig:
        @classmethod
        def from_environment(cls, **_kwargs: Any) -> object:
            pytest.fail("runtime must not run without an explicit diagnostic date")

    monkeypatch.setattr(run_runtime_live, "RuntimeConfig", ForbiddenConfig)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
        allow_unready_diagnostic=True,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["readiness_status"] == "UNREADY_503"
    assert report["errors"] == [{"code": "evidence_readiness_unready_503"}]


def test_non_503_readiness_failure_cannot_use_diagnostic_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RetrievalError("Evidence Service readiness request failed: network error")
        ),
    )

    class ForbiddenConfig:
        @classmethod
        def from_environment(cls, **_kwargs: Any) -> object:
            pytest.fail("runtime must not run after non-503 readiness failure")

    monkeypatch.setattr(run_runtime_live, "RuntimeConfig", ForbiddenConfig)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
        allow_unready_diagnostic=True,
        diagnostic_review_as_of=date(2026, 7, 10),
    )

    assert exit_code == 1
    assert report["readiness_status"] == "FAILED"
    assert report["errors"] == [{"code": "evidence_readiness_failed"}]


def test_ready_but_incomplete_evidence_is_fail_closed_not_release_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    result = _live_result("evidence_live_exact_hsk10")
    captured: dict[str, Any] = {"order": []}
    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
    )
    failure_codes = {item["code"] for item in report["errors"]}

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["agent_execution_passed"] is True
    assert report["release_ready"] is False
    assert "ready_retrieval_status_not_ok" in failure_codes
    assert "ready_retrieval_coverage_incomplete" in failure_codes


def test_prompt_identity_mismatch_fails_agent_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    result = _live_result("happy_path", prompt_sha256="0" * 64)
    captured: dict[str, Any] = {"order": []}
    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
    )

    assert exit_code == 1
    assert report["agent_execution_passed"] is False
    assert report["release_ready"] is False
    assert {"code": "prompt_identity_invalid"} in report["errors"]


def test_unsafe_external_destination_fails_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    _forbid_live_dependencies(monkeypatch)
    unsafe_url = "https://not-allowlisted.example/private-path"

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        base_url=unsafe_url,
        live=True,
    )
    serialized = json.dumps(report)

    assert exit_code == 1
    assert report["network_status"] == "DISABLED_CONFIGURATION_PREFLIGHT_FAILED"
    assert report["errors"] == [{"code": "unsafe_evidence_destination"}]
    assert unsafe_url not in serialized


def test_report_omits_source_values_credentials_paths_and_provider_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    result = _live_result("happy_path")
    captured: dict[str, Any] = {"order": []}
    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
    )
    serialized = json.dumps(report)

    assert exit_code == 0
    assert report["hashes"]["result_sha256"]
    for forbidden in (
        "Stainless bolts",
        "7318151000",
        "Blue Ocean Export Co.",
        "private-anthropic-test-secret",
        "private-evidence-test-secret",
        str(env_file),
        "request-1",
        "message-1",
        "https://",
        'source_uri": "http',
    ):
        assert forbidden not in serialized


def test_runtime_exception_is_reduced_to_safe_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    private_detail = "private-provider-body-and-document-text"
    monkeypatch.setattr(
        run_runtime_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )

    class FakeConfig:
        @classmethod
        def from_environment(cls, **_kwargs: Any) -> object:
            return object()

    class FailingRuntime:
        def __init__(self, _config: object) -> None:
            pass

        def review(self, _request: ReviewRunRequestV2) -> ReviewRunResultV2:
            raise RuntimeError(private_detail)

    monkeypatch.setattr(run_runtime_live, "RuntimeConfig", FakeConfig)
    monkeypatch.setattr(run_runtime_live, "AgentRuntime", FailingRuntime)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
    )
    serialized = json.dumps(report)

    assert exit_code == 1
    assert report["errors"] == [{"code": "agent_runtime_execution_failed"}]
    assert private_detail not in serialized
