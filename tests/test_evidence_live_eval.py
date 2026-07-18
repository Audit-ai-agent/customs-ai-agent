from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.request import Request

import pytest

from evals import run_evidence_live
from customs_ai_agent.contracts.evidence import load_pinned_evidence_contract_identity
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalResponse,
    RetrievalItemResult,
    ReviewRunRequest,
)


def _clear_evidence_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVIDENCE_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("EVIDENCE_SERVICE_TOKEN_HEADER", raising=False)


def _state(
    *,
    retrieval_status: str = "ok",
    retrieval_coverage: str = "complete",
    hs_resolution: str = "exact",
    result_coverage: str = "complete",
    final_status: str = "completed",
    requires_human_review: bool = False,
    snapshot_id: str | None = "knowledge-snapshot-private-id",
    errors: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "retrieval_status": retrieval_status,
        "retrieval_coverage": retrieval_coverage,
        "knowledge_snapshot_id": snapshot_id,
        "retrieval_results": [
            {
                "item_id": "sensitive-request-item",
                "hs_resolution": hs_resolution,
                "coverage": result_coverage,
                "errors": errors or [],
                "evidence": [
                    {
                        "text": "PRIVATE EVIDENCE TEXT",
                        "source_uri": "https://private.example/evidence/1",
                    }
                ],
            }
        ],
        "retrieval_errors": errors or [],
        "validation_issues": [],
        "regulatory_findings": [],
        "rag_evidence": [
            {
                "text": "PRIVATE EVIDENCE TEXT",
                "source_uri": "https://private.example/evidence/1",
            }
        ],
        "final_status": final_status,
        "requires_human_review": requires_human_review,
        "retrieval_request": {
            "description": "PRIVATE REQUEST FIELD VALUE",
            "hs_code": "7318151000",
        },
    }


def _patch_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: pytest.fail("HTTP client must not be constructed"),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: pytest.fail("readiness must not be called"),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "run_review",
        lambda *_args, **_kwargs: pytest.fail("workflow must not run"),
    )


class _HttpBody:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]


def _readiness_payload(*, include_identity: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ready",
        "knowledge_snapshot_id": "snapshot-evidence-live-test",
        "supported_review_as_of": "2026-07-10",
    }
    if include_identity:
        identity = load_pinned_evidence_contract_identity()
        payload["evidence_contract"] = {
            "contract": identity.contract,
            "schema_version": identity.schema_version,
            "provider_release": identity.provider_release,
            "provider_revision": identity.provider_revision,
            "schema_sha256": identity.schema_sha256,
        }
    return payload


def test_without_live_flag_is_preflight_only_and_never_calls_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    _patch_no_network(monkeypatch)

    exit_code, report = run_evidence_live.evaluate(live=False)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["network_mode"] == "preflight_only_network_disabled"
    assert report["case_count"] == 0
    assert report["agent_workflow_count"] == 0
    assert report["readiness_get_count"] == 0
    assert report["retrieval_post_count"] == 0
    assert "api_workflow_count" not in report
    assert report["failures"] == [{"code": "live_flag_required"}]
    assert report["contract"]["consumer_matches_pinned_schema"] is True


def test_current_data_readiness_without_identity_can_never_be_a_release_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    methods: list[str] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int = -1) -> bytes:
            return json.dumps(
                {
                    "status": "ready",
                    "knowledge_snapshot_id": "current-data-snapshot",
                    "supported_review_as_of": "2026-07-10",
                    "law_corpus": {"status": "ready"},
                    "law_index": {"status": "ready"},
                }
            ).encode("utf-8")

    def current_data_readiness(request, **_kwargs):
        methods.append(request.get_method())
        return Response()

    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        current_data_readiness,
    )
    monkeypatch.setattr(
        run_evidence_live,
        "run_review",
        lambda *_args, **_kwargs: pytest.fail("workflow must not run"),
    )

    exit_code, report = run_evidence_live.evaluate(live=True)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["readiness_status"] == "failed"
    assert report["failures"] == [{"code": "evidence_readiness_failed"}]
    assert methods == ["GET", "GET"]
    assert report["agent_workflow_count"] == 0
    assert report["readiness_get_count"] == 2
    assert report["retrieval_post_count"] == 0


def test_stubbed_complete_path_cannot_claim_live_http_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_evidence_environment(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EVIDENCE_SERVICE_TOKEN=private-live-token\n"
        "EVIDENCE_SERVICE_TOKEN_HEADER=X-Private-Test-Token\n",
        encoding="utf-8",
    )
    retriever = object()
    captured: dict[str, object] = {"workflow_calls": 0}

    def fake_retriever(base_url: str, **options):
        captured["retriever"] = (base_url, options)
        return retriever

    def fake_ready(base_url: str, **options):
        captured["readiness"] = (base_url, options)
        return date(2026, 7, 10)

    def fake_run(request: ReviewRunRequest, *, retriever=None):
        captured["workflow_calls"] = int(captured["workflow_calls"]) + 1
        captured["request"] = request
        captured["injected_retriever"] = retriever
        return _state()

    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        fake_retriever,
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        fake_ready,
    )
    monkeypatch.setattr(run_evidence_live, "run_review", fake_run)

    exit_code, report = run_evidence_live.evaluate(
        base_url="http://127.0.0.1:8010",
        env_file=env_file,
        live=True,
    )
    options = {
        "service_token": "private-live-token",
        "token_header": "X-Private-Test-Token",
        "contract_identity_mode": "required",
        "max_retries": 1,
    }
    request = captured["request"]

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["network_mode"] == "live"
    assert report["case_count"] == 1
    assert report["agent_workflow_count"] == 1
    assert report["readiness_get_count"] == 0
    assert report["retrieval_post_count"] == 0
    assert report["failures"] == [
        {"code": "evidence_readiness_http_count_invalid"},
        {"code": "evidence_retrieval_http_count_invalid"},
    ]
    assert report["workflow"]["fail_closed_policy_passed"] is True
    assert report["workflow"]["release_readiness_passed"] is False
    assert report["workflow"]["retrieval_decision_class"] == "complete_contract"
    assert report["workflow"]["supported_review_as_of"] == "2026-07-10"
    assert report["workflow"]["final_status"] == "completed"
    assert captured["retriever"] == ("http://127.0.0.1:8010", options)
    assert captured["readiness"] == ("http://127.0.0.1:8010", options)
    assert captured["workflow_calls"] == 1
    assert captured["injected_retriever"] is retriever
    assert isinstance(request, ReviewRunRequest)
    assert request.scenario_id == "happy_path"
    assert request.review_as_of == date(2026, 7, 10)
    assert request.llm_mode == "mock"


def test_nested_live_evaluation_is_blocked_before_retriever_or_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    _patch_no_network(monkeypatch)

    with run_evidence_live._observe_evidence_http_calls():
        exit_code, report = run_evidence_live.evaluate(live=True)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["network_mode"] == "disabled_http_observer_busy"
    assert report["failures"] == [{"code": "evidence_http_observer_busy"}]
    assert report["agent_workflow_count"] == 0
    assert report["readiness_get_count"] == 0
    assert report["retrieval_post_count"] == 0
    assert report["other_http_count"] == 0


def test_ready_http_path_counts_two_readiness_gets_and_one_retrieval_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    methods: list[str] = []

    def fake_urlopen(request, **_kwargs):
        method = request.get_method()
        methods.append(method)
        if method == "GET":
            return _HttpBody(_readiness_payload())

        request_payload = json.loads(request.data)
        results = []
        for item in request_payload["items"]:
            hs_code = item["hs_code"]
            resolved_hsk10 = hs_code if len(hs_code) == 10 else "7318151000"
            results.append(
                RetrievalItemResult(
                    item_id=item["item_id"],
                    hs_resolution="exact" if len(hs_code) == 10 else "unique_descendant",
                    resolved_hsk10=resolved_hsk10,
                    candidate_hsk10s=[resolved_hsk10],
                    coverage="complete",
                    requirements_found=False,
                    evidence=[],
                    errors=[],
                )
            )
        response = EvidenceRetrievalResponse(
            request_id=request_payload["request_id"],
            status="ok",
            knowledge_snapshot_id="snapshot-evidence-live-test",
            results=results,
            errors=[],
        )
        return _HttpBody(response.model_dump(mode="json"))

    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        fake_urlopen,
    )

    exit_code, report = run_evidence_live.evaluate(live=True)

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert methods == ["GET", "GET", "POST"]
    assert report["case_count"] == 1
    assert report["agent_workflow_count"] == 1
    assert report["readiness_get_count"] == 2
    assert report["retrieval_post_count"] == 1
    assert report["other_http_count"] == 0
    assert report["http_count_unit"] == "adapter_urlopen_attempt_including_retries"
    assert report["workflow"]["fail_closed_policy_passed"] is True
    assert report["workflow"]["release_readiness_passed"] is True


def test_excessive_expected_endpoint_attempts_prevent_live_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: object(),
    )

    def fake_run(*_args, **_kwargs):
        for _ in range(5):
            run_evidence_live.retrieval_adapter.urlopen(
                Request(
                    "http://127.0.0.1:8010/internal/rag/v1/health/ready",
                    method="GET",
                )
            )
        for _ in range(3):
            run_evidence_live.retrieval_adapter.urlopen(
                Request(
                    "http://127.0.0.1:8010/internal/rag/v1/retrieve",
                    data=b"{}",
                    method="POST",
                )
            )
        return _state()

    monkeypatch.setattr(run_evidence_live, "run_review", fake_run)

    exit_code, report = run_evidence_live.evaluate(live=True)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["readiness_get_count"] == 5
    assert report["retrieval_post_count"] == 3
    assert report["workflow"]["fail_closed_policy_passed"] is True
    assert report["workflow"]["release_readiness_passed"] is False
    assert report["failures"] == [
        {"code": "evidence_readiness_http_count_invalid"},
        {"code": "evidence_retrieval_http_count_invalid"},
    ]


def test_unexpected_http_attempt_prevents_an_otherwise_complete_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: object(),
    )

    def fake_run(*_args, **_kwargs):
        run_evidence_live.retrieval_adapter.urlopen(
            Request(
                "http://127.0.0.1:8010/internal/rag/v1/unexpected",
                method="GET",
            )
        )
        return _state()

    monkeypatch.setattr(run_evidence_live, "run_review", fake_run)

    exit_code, report = run_evidence_live.evaluate(live=True)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["agent_workflow_count"] == 1
    assert report["readiness_get_count"] == 0
    assert report["retrieval_post_count"] == 0
    assert report["other_http_count"] == 1
    assert report["failures"] == [
        {"code": "unexpected_evidence_http_attempt"},
        {"code": "evidence_readiness_http_count_invalid"},
        {"code": "evidence_retrieval_http_count_invalid"},
    ]


def test_exact_hsk10_diagnostic_uses_the_explicit_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )

    def fake_run(request: ReviewRunRequest, **_kwargs):
        captured["request"] = request
        return _state(final_status="needs_review", requires_human_review=True)

    monkeypatch.setattr(run_evidence_live, "run_review", fake_run)

    exit_code, report = run_evidence_live.evaluate(
        live=True,
        scenario_id="evidence_live_exact_hsk10",
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["failures"] == [
        {"code": "evidence_readiness_http_count_invalid"},
        {"code": "evidence_retrieval_http_count_invalid"},
    ]
    assert report["scenario_id"] == "evidence_live_exact_hsk10"
    request = captured["request"]
    assert isinstance(request, ReviewRunRequest)
    assert request.scenario_id == "evidence_live_exact_hsk10"


@pytest.mark.parametrize(
    (
        "retrieval_status",
        "retrieval_coverage",
        "hs_resolution",
        "result_coverage",
    ),
    [
        ("partial", "partial", "exact", "partial"),
        ("failed", "unknown", "invalid", "unknown"),
        ("ok", "complete", "ambiguous", "complete"),
    ],
)
def test_incomplete_or_ambiguous_http_result_passes_only_when_needs_review(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_status: str,
    retrieval_coverage: str,
    hs_resolution: str,
    result_coverage: str,
) -> None:
    _clear_evidence_environment(monkeypatch)
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "run_review",
        lambda *_args, **_kwargs: _state(
            retrieval_status=retrieval_status,
            retrieval_coverage=retrieval_coverage,
            hs_resolution=hs_resolution,
            result_coverage=result_coverage,
            final_status="needs_review",
            requires_human_review=True,
            errors=[{"code": "CORPUS_INCOMPLETE", "message": "PRIVATE MESSAGE"}],
        ),
    )

    exit_code, report = run_evidence_live.evaluate(live=True)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["workflow"]["fail_closed_policy_passed"] is True
    assert report["workflow"]["release_readiness_passed"] is False
    assert report["failures"] == [
        {"code": "evidence_readiness_http_count_invalid"},
        {"code": "evidence_retrieval_http_count_invalid"},
    ]
    assert report["workflow"]["retrieval_decision_class"] == "incomplete_fail_closed"
    assert report["workflow"]["final_status"] == "needs_review"
    assert report["workflow"]["requires_human_review"] is True
    assert report["workflow"]["retrieval_error_codes"] == ["CORPUS_INCOMPLETE"]


def test_incomplete_http_result_marked_completed_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "run_review",
        lambda *_args, **_kwargs: _state(
            retrieval_status="partial",
            retrieval_coverage="partial",
            result_coverage="partial",
            final_status="completed",
            requires_human_review=False,
        ),
    )

    exit_code, report = run_evidence_live.evaluate(live=True)
    failure_codes = {item["code"] for item in report["failures"]}

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert "incomplete_retrieval_completed" in failure_codes
    assert "incomplete_retrieval_not_routed_to_review" in failure_codes


def test_unready_diagnostic_proves_fail_closed_but_never_becomes_release_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    retriever = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: retriever,
    )

    def fail_readiness(*_args, **_kwargs):
        raise RuntimeError("private readiness detail")

    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        fail_readiness,
    )

    def fake_run(request: ReviewRunRequest, *, retriever=None):
        captured["request"] = request
        captured["retriever"] = retriever
        return _state(
            retrieval_status="partial",
            retrieval_coverage="partial",
            result_coverage="partial",
            final_status="needs_review",
            requires_human_review=True,
            errors=[{"code": "CORPUS_INCOMPLETE", "message": "PRIVATE MESSAGE"}],
        )

    monkeypatch.setattr(run_evidence_live, "run_review", fake_run)

    exit_code, report = run_evidence_live.evaluate(
        live=True,
        allow_unready_diagnostic=True,
        diagnostic_review_as_of=date(2026, 7, 10),
    )

    assert exit_code == 1
    assert report["status"] == "BLOCKED"
    assert report["network_mode"] == "live_unready_diagnostic"
    assert report["readiness_status"] == "failed_diagnostic_override"
    assert report["failures"] == [{"code": "evidence_readiness_incomplete"}]
    assert report["workflow"]["fail_closed_policy_passed"] is True
    assert report["workflow"]["release_readiness_passed"] is False
    assert report["workflow"]["retrieval_decision_class"] == "incomplete_fail_closed"
    request = captured["request"]
    assert isinstance(request, ReviewRunRequest)
    assert request.review_as_of == date(2026, 7, 10)
    assert captured["retriever"] is retriever


def test_unready_required_identity_counts_rechecks_but_zero_retrieval_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    methods: list[str] = []

    def fake_urlopen(request, **_kwargs):
        methods.append(request.get_method())
        return _HttpBody(_readiness_payload(include_identity=False))

    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        fake_urlopen,
    )

    exit_code, report = run_evidence_live.evaluate(
        live=True,
        allow_unready_diagnostic=True,
        diagnostic_review_as_of=date(2026, 7, 10),
    )

    assert exit_code == 1
    assert report["status"] == "BLOCKED"
    assert report["readiness_status"] == "failed_diagnostic_override"
    assert report["workflow"]["fail_closed_policy_passed"] is True
    assert report["workflow"]["final_status"] == "needs_review"
    assert methods == ["GET", "GET", "GET", "GET"]
    assert report["case_count"] == 1
    assert report["agent_workflow_count"] == 1
    assert report["readiness_get_count"] == 4
    assert report["retrieval_post_count"] == 0
    assert report["other_http_count"] == 0


def test_unready_diagnostic_requires_explicit_date_and_does_not_run_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("not ready")),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "run_review",
        lambda *_args, **_kwargs: pytest.fail("workflow must not run"),
    )

    exit_code, report = run_evidence_live.evaluate(
        live=True,
        allow_unready_diagnostic=True,
    )

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["failures"] == [{"code": "evidence_readiness_failed"}]
    assert report["agent_workflow_count"] == 0
    assert report["readiness_get_count"] == 0
    assert report["retrieval_post_count"] == 0


def test_report_redacts_request_evidence_uri_token_env_and_snapshot_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_evidence_environment(monkeypatch)
    env_file = tmp_path / "private-evidence.env"
    env_file.write_text(
        "EVIDENCE_SERVICE_TOKEN=unique-private-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_evidence_live,
        "HttpRegulationRetriever",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "fetch_supported_review_as_of",
        lambda *_args, **_kwargs: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        run_evidence_live,
        "run_review",
        lambda *_args, **_kwargs: _state(
            final_status="needs_review",
            requires_human_review=True,
        ),
    )

    exit_code, report = run_evidence_live.evaluate(
        env_file=env_file,
        live=True,
    )
    serialized = json.dumps(report)
    snapshot_id = "knowledge-snapshot-private-id"

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert (
        report["workflow"]["knowledge_snapshot_id_sha256"]
        == hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()
    )
    for forbidden in (
        "PRIVATE REQUEST FIELD VALUE",
        "PRIVATE EVIDENCE TEXT",
        "https://private.example/evidence/1",
        "unique-private-token",
        str(env_file),
        snapshot_id,
        "7318151000",
    ):
        assert forbidden not in serialized
    assert "source_uri" in report["redaction"]
    assert report["redaction"]["source_uri"] == "OMITTED"


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "https://evidence.example:443",
        "http://127.0.0.1:8010/path",
        "http://user:password@127.0.0.1:8010",
        "http://127.0.0.1:8010?token=private",
    ],
)
def test_unsafe_or_non_loopback_url_fails_before_http(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_url: str,
) -> None:
    _clear_evidence_environment(monkeypatch)
    _patch_no_network(monkeypatch)

    exit_code, report = run_evidence_live.evaluate(
        base_url=unsafe_url,
        live=True,
    )
    serialized = json.dumps(report)

    assert exit_code == 1
    assert report["network_mode"] == "disabled_unsafe_url"
    assert report["failures"] == [{"code": "unsafe_base_url"}]
    assert unsafe_url not in serialized


def test_contract_drift_blocks_live_network_and_keeps_release_pin_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_evidence_environment(monkeypatch)
    _patch_no_network(monkeypatch)
    monkeypatch.setattr(
        run_evidence_live,
        "EXPECTED_EVIDENCE_SCHEMA_SHA256",
        "0" * 64,
    )

    exit_code, report = run_evidence_live.evaluate(live=True)
    failure_codes = {item["code"] for item in report["failures"]}
    contract = report["contract"]

    assert exit_code == 1
    assert report["network_mode"] == "disabled_contract_preflight_failed"
    assert "lock_schema_sha256_unexpected" in failure_codes
    assert contract["consumer_matches_pinned_schema"] is False
    assert (
        contract["pinned_provider_release_revision"]
        == run_evidence_live.EXPECTED_PINNED_PROVIDER_REVISION
    )
    assert contract["provider_repository_head_status"] == "not_observed_http_only"
