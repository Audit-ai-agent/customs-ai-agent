from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from customs_ai_agent.agentic import to_review_run_result_v3
from customs_ai_agent.contracts import get_schema
from customs_ai_agent.fixtures.scenarios import get_scenario
from customs_ai_agent.runner import run_review_result_v2
from customs_ai_agent.schemas.models import ReviewRunRequestV2
from customs_ai_agent.schemas.v3 import ServiceReviewRunRequestV3, canonical_json_sha256
from tools.agent_execution_inspector import run_v3_conformance as runner


_REAL_TRACKED_SOURCE_INPUT_IDENTITY = runner._tracked_source_input_identity


@pytest.fixture(autouse=True)
def _tracked_source_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    def identity(_path: Path, raw: bytes, request: ServiceReviewRunRequestV3):
        return {
            "repository_id": "customs-ai-agent",
            "repository_history_roots_sha256": "3" * 64,
            "repository_path_base": "agent-checkout-root",
            "tracked_relative_path": "tests/unit-generated.request.json",
            "git_object_format": "sha1",
            "repository_head_oid": "1" * 40,
            "git_blob_oid": "2" * 40,
            "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_request_sha256": canonical_json_sha256(request),
            "canonicalization": "customs-ai-agent/canonical-json-v1",
        }

    monkeypatch.setattr(runner, "_tracked_source_input_identity", identity)


def _result():
    scenario = get_scenario("happy_path")
    request = ServiceReviewRunRequestV3(
        case_id=scenario["case_id"],
        documents=scenario["documents"],
        llm_mode="llm",
        review_run_id="conformance_runner_test",
        review_as_of="2026-07-14",
        trade_type="import",
        jurisdiction="KR",
        document_language="en",
        assertion_cutoff_at="2026-07-22T00:00:00Z",
    )
    result_v2 = run_review_result_v2(
        ReviewRunRequestV2(
            scenario_id="happy_path",
            review_run_id=request.review_run_id,
            review_as_of=request.review_as_of,
        )
    )
    return request, to_review_run_result_v3(
        result_v2,
        request,
        evidence_runtime_v3_sha256="e" * 64,
    )


def test_checked_in_v3_positive_control_is_schema_valid_hash_frozen_and_test_only() -> None:
    path = Path(runner.__file__).parent / "clean_text_production_candidate_v3.request.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    request = ServiceReviewRunRequestV3.model_validate(payload)

    Draft202012Validator(get_schema("backend", "v3", "review-request")).validate(payload)
    assert hashlib.sha256(raw).hexdigest() == (
        "bb5c87b544a9a46c6261832fd24b25b988b9f1fd56ba07f3bdb815c00e674567"
    )
    assert canonical_json_sha256(request) == (
        "7c68df85e2d38a42a094c89df14f328a189beeb736b40900de963cd865152e4f"
    )
    assert runner._request_is_test_only_positive_control(request)
    assertions = [
        assertion
        for document in request.supporting_documents
        for assertion in document.verification_assertions
    ]
    assert assertions and all(assertion.method == "test_issuer" for assertion in assertions)

    source = _REAL_TRACKED_SOURCE_INPUT_IDENTITY(path, raw, request)
    repository = Path(runner.__file__).resolve().parents[2]
    expected_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert source["repository_id"] == "customs-ai-agent"
    assert source["repository_path_base"] == "agent-checkout-root"
    assert source["repository_head_oid"] == expected_commit
    assert source["git_object_format"] == "sha1"
    assert source["git_blob_oid"] == "d1ae078bfd7fbab9f655d9e2b241e2f6ae80fd7a"
    assert len(source["repository_history_roots_sha256"]) == 64
    assert source["tracked_relative_path"] == (
        "tools/agent_execution_inspector/clean_text_production_candidate_v3.request.json"
    )
    assert source["raw_bytes_sha256"] == (
        "bb5c87b544a9a46c6261832fd24b25b988b9f1fd56ba07f3bdb815c00e674567"
    )
    assert source["canonical_request_sha256"] == (
        "7c68df85e2d38a42a094c89df14f328a189beeb736b40900de963cd865152e4f"
    )
    assert source["canonicalization"] == "customs-ai-agent/canonical-json-v1"


def test_tracked_source_identity_rejects_untracked_or_modified_request(
    tmp_path: Path,
) -> None:
    source = Path(runner.__file__).parent / "clean_text_production_candidate_v3.request.json"
    raw = source.read_bytes()
    request = ServiceReviewRunRequestV3.model_validate_json(raw)
    copied = tmp_path / "untracked-request.json"
    copied.write_bytes(raw)
    with pytest.raises(ValueError, match="outside_expected_agent_repository"):
        _REAL_TRACKED_SOURCE_INPUT_IDENTITY(copied, raw, request)


def test_source_identity_rejects_same_fixture_in_arbitrary_non_ascii_git_repository(
    tmp_path: Path,
) -> None:
    source = Path(runner.__file__).parent / "clean_text_production_candidate_v3.request.json"
    raw = source.read_bytes()
    request = ServiceReviewRunRequestV3.model_validate_json(raw)
    arbitrary = tmp_path / "임의-저장소"
    arbitrary.mkdir()
    copied = arbitrary / source.name
    copied.write_bytes(raw)
    subprocess.run(["git", "init", "-q", str(arbitrary)], check=True)
    subprocess.run(["git", "-C", str(arbitrary), "add", copied.name], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(arbitrary),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="outside_expected_agent_repository"):
        _REAL_TRACKED_SOURCE_INPUT_IDENTITY(copied, raw, request)


def test_arbitrary_repository_source_is_rejected_before_credential_or_network_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request, _result_value = _result()
    arbitrary = tmp_path / "other-repository"
    arbitrary.mkdir()
    request_path = arbitrary / "request.json"
    request_path.write_text(
        json.dumps(request.model_dump(mode="json")),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(arbitrary)], check=True)
    subprocess.run(["git", "-C", str(arbitrary), "add", request_path.name], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(arbitrary),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "request",
        ],
        check=True,
    )
    identity_path = tmp_path / "identity.json"
    identity_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_conformance_identity",
        lambda _path: SimpleNamespace(
            base_runtime_release=SimpleNamespace(supported_review_as_of=request.review_as_of)
        ),
    )
    monkeypatch.setattr(
        runner,
        "_tracked_source_input_identity",
        _REAL_TRACKED_SOURCE_INPUT_IDENTITY,
    )
    monkeypatch.setattr(
        runner,
        "_read_token",
        lambda _path: pytest.fail("credential must not be read for an arbitrary repository"),
    )
    monkeypatch.setattr(
        runner,
        "_non_release_runtime",
        lambda **_kwargs: pytest.fail("network runtime must not be constructed"),
    )

    report = runner.run_conformance(
        base_url="http://127.0.0.1:9999",
        identity_path=identity_path,
        request_path=request_path,
        token_path=tmp_path / "token.txt",
        env_file=tmp_path / ".env",
        output_dir=tmp_path / "out",
    )

    assert report["failure_stage"] == "source_input_validation"
    assert report["error_code"] == "source_input_validation"
    assert "source_input" not in report


def test_conformance_runner_uses_serialized_agent_v3_boundary_and_redacts_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request, result = _result()
    raw_request = json.dumps(request.model_dump(mode="json")).encode("utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_bytes(raw_request)
    for name in ("identity.json", "token.txt", ".env"):
        (tmp_path / name).write_text("placeholder", encoding="utf-8")
    identity = SimpleNamespace(
        runtime_release_v3_sha256="e" * 64,
        binding_headers=lambda: {f"X-Test-{index}": str(index) for index in range(17)},
        base_runtime_release=SimpleNamespace(supported_review_as_of=request.review_as_of),
    )
    observed: dict[str, object] = {}

    class Retriever:
        def preflight(self, review_as_of):
            observed["preflight_review_as_of"] = review_as_of

    class Runtime:
        _retriever_v3 = Retriever()

        def review_v3_json(self, payload: bytes):
            observed["payload"] = payload
            return result

        def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(runner, "load_conformance_identity", lambda _path: identity)
    monkeypatch.setattr(runner, "_read_token", lambda _path: "private-token-value")
    monkeypatch.setattr(runner, "_non_release_runtime", lambda **_kwargs: Runtime())
    monkeypatch.setattr(
        runner,
        "_evaluate_machine_assertions",
        lambda **_kwargs: (
            {"unit_test_machine_assertion": True},
            {
                "successful_readiness_get_count": 1,
                "successful_retrieval_post_count": 1,
            },
        ),
    )
    report = runner.run_conformance(
        base_url="http://127.0.0.1:9999",
        identity_path=tmp_path / "identity.json",
        request_path=request_path,
        token_path=tmp_path / "token.txt",
        env_file=tmp_path / ".env",
        output_dir=tmp_path / "out",
    )

    assert report["status"] == "PASS"
    assert report["entrypoint"] == "AgentRuntime.review_v3_json"
    assert observed == {
        "preflight_review_as_of": request.review_as_of,
        "payload": raw_request,
        "closed": True,
    }
    encoded = (tmp_path / "out" / "report.json").read_text(encoding="utf-8")
    assert "private-token-value" not in encoded
    assert "COMMERCIAL INVOICE" not in encoded
    assert report["source_input"]["raw_bytes_sha256"] == hashlib.sha256(raw_request).hexdigest()
    assert report["checks"]["tracked_source_canonical_hash_matches_result"] is True
    assert (tmp_path / "out" / "junit.xml").is_file()


def test_non_release_runtime_explicitly_disables_assertion_trust(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        api_key=SimpleNamespace(get_secret_value=lambda: "private-anthropic-key")
    )
    analyzer = SimpleNamespace()
    retriever = SimpleNamespace()
    monkeypatch.setattr(runner.ClaudeSettings, "from_environment", lambda _path: settings)
    monkeypatch.setattr(
        runner.ClaudeDocumentAnalyzer,
        "from_settings",
        lambda _settings: analyzer,
    )
    monkeypatch.setattr(
        runner,
        "LoopbackEvidenceV3ConformanceRetriever",
        lambda *_args, **_kwargs: retriever,
    )
    runtime = runner._non_release_runtime(
        base_url="http://127.0.0.1:9999",
        identity=SimpleNamespace(),
        service_token="private-evidence-token",
        env_file=tmp_path / ".env",
    )
    assert runtime._analyzer is analyzer
    assert runtime._retriever_v3 is retriever
    assert runtime._assertion_verifier is None


def test_conformance_runner_rejects_review_date_not_bound_to_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request, _result_value = _result()
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(request.model_dump(mode="json")),
        encoding="utf-8",
    )
    for name in ("identity.json", "token.txt", ".env"):
        (tmp_path / name).write_text("private-canary", encoding="utf-8")
    identity = SimpleNamespace(
        base_runtime_release=SimpleNamespace(
            supported_review_as_of=request.review_as_of.replace(day=request.review_as_of.day - 1)
        )
    )
    monkeypatch.setattr(runner, "load_conformance_identity", lambda _path: identity)
    monkeypatch.setattr(
        runner,
        "_non_release_runtime",
        lambda **_kwargs: pytest.fail("runtime must not be constructed after date mismatch"),
    )

    report = runner.run_conformance(
        base_url="http://127.0.0.1:9999",
        identity_path=tmp_path / "identity.json",
        request_path=request_path,
        token_path=tmp_path / "token.txt",
        env_file=tmp_path / ".env",
        output_dir=tmp_path / "out",
    )

    assert report == {
        "schema_version": "agent-v3-http-conformance-report/v1",
        "status": "FAIL",
        "non_release": True,
        "provider_scope": "TEST_ONLY_LOOPBACK_CONFORMANCE_PROVIDER",
        "production_release_evidence": False,
        "entrypoint": "AgentRuntime.review_v3_json",
        "failure_stage": "evidence_preflight",
        "error_code": "evidence_review_date_mismatch",
        "error_type": "contract_validation",
        "source_input": {
            "repository_id": "customs-ai-agent",
            "repository_history_roots_sha256": "3" * 64,
            "repository_path_base": "agent-checkout-root",
            "tracked_relative_path": "tests/unit-generated.request.json",
            "git_object_format": "sha1",
            "repository_head_oid": "1" * 40,
            "git_blob_oid": "2" * 40,
            "raw_bytes_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "canonical_request_sha256": canonical_json_sha256(request),
            "canonicalization": "customs-ai-agent/canonical-json-v1",
        },
    }
    assert "private-canary" not in (tmp_path / "out" / "report.json").read_text(encoding="utf-8")


def test_conformance_runner_reports_allowlisted_claude_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request, _result_value = _result()
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(request.model_dump(mode="json")),
        encoding="utf-8",
    )
    for name in ("identity.json", "token.txt", ".env"):
        (tmp_path / name).write_text("private-canary", encoding="utf-8")
    identity = SimpleNamespace(
        base_runtime_release=SimpleNamespace(supported_review_as_of=request.review_as_of)
    )
    monkeypatch.setattr(runner, "load_conformance_identity", lambda _path: identity)
    monkeypatch.setattr(runner, "_read_token", lambda _path: "private-token")

    def fail_runtime(**_kwargs):
        from customs_ai_agent.adapters.llm import LLMConfigurationError

        raise LLMConfigurationError("api_key_invalid", "private-canary")

    monkeypatch.setattr(runner, "_non_release_runtime", fail_runtime)

    report = runner.run_conformance(
        base_url="http://127.0.0.1:9999",
        identity_path=tmp_path / "identity.json",
        request_path=request_path,
        token_path=tmp_path / "token.txt",
        env_file=tmp_path / ".env",
        output_dir=tmp_path / "out",
    )

    assert report["failure_stage"] == "claude_configuration"
    assert report["error_code"] == "claude_configuration"
    assert report["error_type"] == "configuration"
    encoded = (tmp_path / "out" / "report.json").read_text(encoding="utf-8")
    assert "private-canary" not in encoded
    assert "api_key_invalid" not in encoded


def test_machine_assertions_do_not_accept_mock_only_result_as_live_conformance() -> None:
    request, result = _result()
    identity = SimpleNamespace(
        runtime_release_v3_sha256="e" * 64,
        binding_headers=lambda: {f"X-Test-{index}": str(index) for index in range(17)},
    )
    retriever = SimpleNamespace(
        successful_readiness_get_count=1,
        successful_retrieval_post_count=1,
    )

    checks, observations = runner._evaluate_machine_assertions(
        request=request,
        result=result,
        retriever=retriever,
        identity=identity,
    )

    assert checks["actual_anthropic_document_model_calls_observed"] is False
    assert checks["request_is_test_only_four_document_positive_control"] is False
    assert observations["anthropic_model_call_count"] == 0


def test_conformance_cli_requires_explicit_non_release_acknowledgement() -> None:
    with pytest.raises(SystemExit, match="acknowledge"):
        runner.main(
            [
                "--base-url",
                "http://127.0.0.1:9999",
                "--identity",
                "identity.json",
                "--request",
                "request.json",
                "--token-file",
                "token.txt",
                "--env-file",
                ".env",
                "--output-dir",
                "out",
            ]
        )


def test_documented_direct_script_help_runs_outside_repository_cwd(tmp_path: Path) -> None:
    script = Path(runner.__file__).resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--acknowledge-non-release" in completed.stdout
