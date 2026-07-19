from __future__ import annotations

import hashlib
import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals import run_runtime_live
from customs_ai_agent.adapters import retrieval as retrieval_module
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseIdentity,
    canonical_json_bytes,
)
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.schemas.models import (
    ReviewRunRequestV2,
    ReviewRunResultV2,
    ServiceReviewRunRequestV2,
)


def _env_file(
    tmp_path: Path,
    *,
    allowed_origins: str = "",
    token_header: str = "Authorization",
) -> Path:
    path = tmp_path / "private-runtime.env"
    path.write_text(
        "ANTHROPIC_API_KEY=private-anthropic-test-secret\n"
        "EVIDENCE_SERVICE_TOKEN=private-evidence-test-secret\n"
        f"EVIDENCE_SERVICE_TOKEN_HEADER={token_header}\n"
        f"EVIDENCE_SERVICE_ALLOWED_ORIGINS={allowed_origins}\n",
        encoding="utf-8",
    )
    return path


def _readiness_identity(review_as_of: date = date(2026, 7, 10)) -> SimpleNamespace:
    return SimpleNamespace(supported_review_as_of=review_as_of)


def _sealed_runtime_identity() -> EvidenceRuntimeReleaseIdentity:
    payload: dict[str, str] = {
        "contract": "evidence-runtime-release",
        "schema_version": "2.0",
        "retrieval_schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": "1" * 40,
        "provider_build_sha256": "2" * 64,
        "retrieval_schema_sha256": (
            "531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a"
        ),
        "provider_source_manifest_sha256": "3" * 64,
        "knowledge_generation_id": "4" * 64,
        "knowledge_generation_attestation_sha256": "5" * 64,
        "knowledge_source_manifest_sha256": "6" * 64,
        "model_manifest_sha256": "7" * 64,
        "index_config_sha256": "8" * 64,
        "evaluation_report_sha256": "9" * 64,
        "seal_attestation_sha256": "a" * 64,
        "knowledge_snapshot_id": "test-only-runtime-snapshot",
        "supported_review_as_of": "2026-07-10",
    }
    payload["runtime_release_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return EvidenceRuntimeReleaseIdentity.from_mapping(payload)


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

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            captured["runtime_exit_count"] = int(captured.get("runtime_exit_count", 0)) + 1

        def review(self, request: ServiceReviewRunRequestV2) -> ReviewRunResultV2:
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
        "fetch_evidence_runtime_readiness",
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


def test_ready_allowlisted_service_without_quality_attestation_stays_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_url = "https://evidence.example"
    env_file = _env_file(tmp_path, allowed_origins=base_url)
    result = _live_result("happy_path")
    captured: dict[str, Any] = {"order": []}

    def fake_readiness(url: str, **options: Any) -> SimpleNamespace:
        captured["order"].append("readiness")
        captured["readiness"] = (url, options)
        return _readiness_identity()

    monkeypatch.setattr(
        run_runtime_live,
        "fetch_evidence_runtime_readiness",
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
    assert exit_code == 1
    assert report["status"] == "BLOCKED_RAG_RELEASE_ATTESTATION"
    assert report["readiness_status"] == "READY"
    assert report["agent_execution_passed"] is True
    assert report["release_ready"] is False
    assert report["implementation_candidate_ready"] is False
    assert report["scope"] == run_runtime_live.LIVE_PROVIDER_SCOPE
    assert report["errors"] == [{"code": "rag_quality_attestation_missing"}]
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
    assert captured["runtime_exit_count"] == 1
    assert readiness_url == base_url
    assert readiness_options["allowed_origins"] == frozenset({base_url})
    assert readiness_options["service_token"] == "private-evidence-test-secret"
    assert "token_header" not in readiness_options
    assert "contract_identity_mode" not in readiness_options
    assert config_options["evidence_base_url"] == base_url
    assert config_options["env_file"] == env_file
    assert config_options["evidence_allowed_origins"] == frozenset({base_url})
    assert "evidence_token_header" not in config_options
    assert (
        config_options["evidence_service_token"].get_secret_value()
        == "private-evidence-test-secret"
    )
    assert isinstance(request, ServiceReviewRunRequestV2)
    assert request.case_id == "case_evidence_live_exact_hsk10"
    assert request.documents is not None
    assert len(request.documents) == run_runtime_live.EXPECTED_DOCUMENT_COUNT
    assert request.llm_mode == "llm"
    assert request.review_as_of == date(2026, 7, 10)


def test_provider_conformance_pass_is_never_reported_as_rag_release_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_url = "https://evidence.example"
    env_file = _env_file(tmp_path, allowed_origins=base_url)
    result = _live_result("happy_path")
    captured: dict[str, Any] = {"order": []}
    monkeypatch.setattr(
        run_runtime_live,
        "fetch_evidence_runtime_readiness",
        lambda *_args, **_kwargs: _readiness_identity(),
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        base_url=base_url,
        live=True,
        provider_conformance=True,
    )

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["scope"] == "PROVIDER_CONFORMANCE_NOT_RAG_QUALITY"
    assert report["agent_execution_passed"] is True
    assert report["release_ready"] is False
    assert report["implementation_candidate_ready"] is True
    assert report["errors"] == []
    assert captured["runtime_exit_count"] == 1


def test_live_preflight_uses_real_v2_loopback_readiness_and_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _sealed_runtime_identity()
    observed: list[tuple[str, str | None]] = []
    response_bytes = json.dumps(
        {
            "status": "ready",
            "evidence_runtime_release": identity.to_wire(),
        },
        separators=(",", ":"),
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
            observed.append((self.path, self.headers.get("Authorization")))
            if self.path != "/internal/rag/v2/health/ready":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = int(server.server_address[1])
            env_file = _env_file(tmp_path)
            result = _live_result("happy_path")
            captured: dict[str, Any] = {"order": []}
            monkeypatch.setattr(
                retrieval_module,
                "load_released_evidence_runtime_identity",
                lambda: identity,
            )
            _patch_runtime(monkeypatch, result, captured)

            exit_code, report = run_runtime_live.evaluate(
                env_file=env_file,
                base_url=f"http://127.0.0.1:{port}",
                live=True,
                provider_conformance=True,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["scope"] == run_runtime_live.PROVIDER_CONFORMANCE_SCOPE
    assert report["release_ready"] is False
    assert observed == [
        (
            "/internal/rag/v2/health/ready",
            "Bearer private-evidence-test-secret",
        )
    ]
    assert captured["request"].review_as_of == identity.supported_review_as_of
    assert "evidence_token_header" not in captured["config_options"]


def test_custom_evidence_auth_header_is_rejected_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _forbid_live_dependencies(monkeypatch)
    env_file = _env_file(
        tmp_path,
        token_header="X-Private-Evidence-Token",
    )

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
    )

    assert exit_code == 1
    assert report["network_status"] == "DISABLED_CONFIGURATION_PREFLIGHT_FAILED"
    assert report["errors"] == [{"code": "evidence_token_header_invalid"}]


def test_provider_conformance_scope_is_preserved_when_live_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _forbid_live_dependencies(monkeypatch)

    exit_code, report = run_runtime_live.evaluate(
        env_file=tmp_path / "must-not-be-read.env",
        live=False,
        provider_conformance=True,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["scope"] == "PROVIDER_CONFORMANCE_NOT_RAG_QUALITY"
    assert report["release_ready"] is False
    assert report["implementation_candidate_ready"] is False


def test_cli_exposes_explicit_provider_conformance_flag(tmp_path: Path) -> None:
    args = run_runtime_live.build_parser().parse_args(
        [
            "--env-file",
            str(tmp_path / "runtime.env"),
            "--live",
            "--provider-conformance",
        ]
    )

    assert args.live is True
    assert args.provider_conformance is True


def test_unready_503_diagnostic_executes_agent_and_stays_non_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = _env_file(tmp_path)
    result = _live_result("evidence_live_exact_hsk10")
    captured: dict[str, Any] = {"order": []}

    def unready(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        captured["order"].append("readiness")
        raise RetrievalError("Evidence Service readiness request failed: HTTP status 503")

    monkeypatch.setattr(
        run_runtime_live,
        "fetch_evidence_runtime_readiness",
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
    assert isinstance(request, ServiceReviewRunRequestV2)
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

    def unready(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        raise RetrievalError("Evidence Service readiness request failed: HTTP status 503")

    monkeypatch.setattr(
        run_runtime_live,
        "fetch_evidence_runtime_readiness",
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
        "fetch_evidence_runtime_readiness",
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
        "fetch_evidence_runtime_readiness",
        lambda *_args, **_kwargs: _readiness_identity(),
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
        provider_conformance=True,
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
        "fetch_evidence_runtime_readiness",
        lambda *_args, **_kwargs: _readiness_identity(),
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
        "fetch_evidence_runtime_readiness",
        lambda *_args, **_kwargs: _readiness_identity(),
    )
    _patch_runtime(monkeypatch, result, captured)

    exit_code, report = run_runtime_live.evaluate(
        env_file=env_file,
        live=True,
        provider_conformance=True,
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
        "fetch_evidence_runtime_readiness",
        lambda *_args, **_kwargs: _readiness_identity(),
    )

    class FakeConfig:
        @classmethod
        def from_environment(cls, **_kwargs: Any) -> object:
            return object()

    exit_calls = 0

    class FailingRuntime:
        def __init__(self, _config: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            nonlocal exit_calls
            exit_calls += 1

        def review(self, _request: ServiceReviewRunRequestV2) -> ReviewRunResultV2:
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
    assert exit_calls == 1
