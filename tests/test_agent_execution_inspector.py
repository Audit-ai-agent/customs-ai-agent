from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import customs_ai_agent.adapters.retrieval as retrieval_module
from customs_ai_agent.contracts.evidence_runtime_release import canonical_json_bytes
from tools.agent_execution_inspector import server


TOKEN = "ThreeLawDiagnosticToken-0123456789-ABCDEFGHIJK"


def _identity_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract": "evidence-runtime-release",
        "schema_version": "2.0",
        "retrieval_schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": "a" * 40,
        "provider_build_sha256": "1" * 64,
        "retrieval_schema_sha256": "2" * 64,
        "provider_source_manifest_sha256": "3" * 64,
        "knowledge_generation_id": "4" * 64,
        "knowledge_generation_attestation_sha256": "5" * 64,
        "knowledge_source_manifest_sha256": "6" * 64,
        "model_manifest_sha256": "7" * 64,
        "index_config_sha256": "8" * 64,
        "evaluation_report_sha256": "9" * 64,
        "seal_attestation_sha256": "b" * 64,
        "knowledge_snapshot_id": "diagnostic-snapshot",
        "supported_review_as_of": "2026-07-10",
    }
    payload["runtime_release_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": server.MANIFEST_SCHEMA_VERSION,
        "scope": server.SCOPE,
        "release_ready": False,
        "rag_quality_evaluated": False,
        "expected_production_law_count": 58,
        "expected_diagnostic_law_count": 3,
        "observed_target_law_count": 3,
        "observed_total_law_count": 3,
        "observed_article_count": 1007,
        "qdrant_point_count": 1007,
        "neo4j_article_count": 1007,
        "law_article_manifest_sha256": "c" * 64,
        "postgres_qdrant_ids_sha256": "d" * 64,
        "qdrant_point_ids_sha256": "d" * 64,
        "postgres_graph_ids_sha256": "e" * 64,
        "neo4j_article_ids_sha256": "e" * 64,
        "runtime_stability": {
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "tokenizers_parallelism": "false",
            "hf_hub_offline": "1",
        },
        "data_revision": "a" * 40,
        "evidence_runtime_release": _identity_payload(),
    }


def _write_manifest(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    path = (tmp_path / "diagnostic-manifest.json").resolve()
    path.write_text(
        json.dumps(payload or _manifest_payload(), sort_keys=True),
        encoding="utf-8",
    )
    return path


def _request_payload(*, document_count: int = 1) -> dict[str, Any]:
    return {
        "case_id": "case-inspector",
        "documents": [
            {
                "document_id": f"invoice-{index}",
                "filename": f"invoice-{index}.txt",
                "declared_type": "invoice",
                "content_text": (
                    "COMMERCIAL INVOICE\n"
                    f"INVOICE NO: INV-{index}\n"
                    "ITEM: STEEL BOLT | HS: 7318151000 | QTY: 10 | AMOUNT: 100"
                ),
                "requirement_document_codes": [],
                "image_pages": [],
                "pdf_pages": None,
            }
            for index in range(document_count)
        ],
        "llm_mode": "llm",
        "review_run_id": "run-inspector",
        "review_as_of": "2026-07-10",
        "trade_type": "import",
        "jurisdiction": "KR",
        "uploaded_requirement_document_codes": [],
    }


class _FakeExecutor:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.requests: list[Any] = []
        self.result = result or {
            "schema_version": server.ENVELOPE_SCHEMA_VERSION,
            "scope": server.SCOPE,
            "release_ready": False,
            "summary": {"status": "needs_review"},
            "result": {"schema_version": "2.0", "status": "needs_review"},
        }

    def health(self) -> dict[str, Any]:
        return {
            "schema_version": server.ENVELOPE_SCHEMA_VERSION,
            "status": "ready",
            "release_ready": False,
        }

    def execute(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return self.result


def _app(
    tmp_path: Path,
    executor: _FakeExecutor | None = None,
    *,
    run_timeout_seconds: float = 300.0,
    max_run_records: int = server.MAX_RUN_RECORDS,
    run_record_ttl_seconds: float = server.RUN_RECORD_TTL_SECONDS,
):
    html = (tmp_path / "inspector.html").resolve()
    html.write_text("<!doctype html><title>Inspector</title>", encoding="utf-8")
    selected = executor or _FakeExecutor()
    app = server.create_app(
        settings=server.InspectorSettings(
            host="127.0.0.1",
            port=18021,
            run_timeout_seconds=run_timeout_seconds,
            max_run_records=max_run_records,
            run_record_ttl_seconds=run_record_ttl_seconds,
        ),
        executor=selected,
        html_path=html,
    )
    return app, selected


def test_manifest_requires_explicit_three_of_fifty_eight_nonrelease_scope(
    tmp_path: Path,
) -> None:
    manifest = server.load_diagnostic_manifest(_write_manifest(tmp_path))

    assert manifest.observed_target_law_count == 3
    assert manifest.expected_production_law_count == 58
    assert manifest.identity.knowledge_snapshot_id == "diagnostic-snapshot"
    assert manifest.public_metadata()["release_ready"] is False

    invalid = _manifest_payload()
    invalid["release_ready"] = True
    with pytest.raises(server.InspectorConfigurationError):
        server.load_diagnostic_manifest(_write_manifest(tmp_path, invalid))


def test_manifest_rejects_store_count_drift_and_secret_like_fields(tmp_path: Path) -> None:
    drifted = _manifest_payload()
    drifted["qdrant_point_count"] = 1006
    with pytest.raises(server.InspectorConfigurationError):
        server.load_diagnostic_manifest(_write_manifest(tmp_path, drifted))

    secret_bearing = _manifest_payload()
    secret_bearing["service_token"] = TOKEN
    with pytest.raises(server.InspectorExecutionError):
        server.load_diagnostic_manifest(_write_manifest(tmp_path, secret_bearing))


def test_identity_overlay_is_process_local_and_always_restored() -> None:
    original = retrieval_module.load_released_evidence_runtime_identity
    identity = server.EvidenceRuntimeReleaseIdentity.from_mapping(_identity_payload())

    with server.diagnostic_runtime_identity(identity):
        assert retrieval_module.load_released_evidence_runtime_identity() == identity

    assert retrieval_module.load_released_evidence_runtime_identity is original
    with pytest.raises(RuntimeError):
        with server.diagnostic_runtime_identity(identity):
            raise RuntimeError("expected")
    assert retrieval_module.load_released_evidence_runtime_identity is original


def test_local_app_accepts_only_explicit_bounded_text_requests(tmp_path: Path) -> None:
    app, executor = _app(tmp_path)
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-store"
        assert page.headers["x-frame-options"] == "DENY"

        missing_header = client.post("/api/reviews", json=_request_payload())
        assert missing_header.status_code == 403

        accepted = client.post(
            "/api/reviews",
            json=_request_payload(),
            headers={server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE},
        )
        assert accepted.status_code == 200
        assert accepted.headers["cache-control"] == "no-store"
        accepted_run = accepted.json()
        assert accepted_run["schema_version"] == server.RUN_SCHEMA_VERSION
        assert accepted_run["status"] == "succeeded"
        assert accepted_run["terminal"] is True
        assert accepted_run["result"]["summary"]["status"] == "needs_review"
        assert len(accepted_run["run_id"]) == 32

        too_many = client.post(
            "/api/reviews",
            json=_request_payload(document_count=7),
            headers={server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE},
        )
        assert too_many.status_code == 413

        mock = _request_payload()
        mock["llm_mode"] = "mock"
        rejected_mock = client.post(
            "/api/reviews",
            json=mock,
            headers={server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE},
        )
        assert rejected_mock.status_code == 422

    assert len(executor.requests) == 1


def test_local_app_rejects_duplicate_json_and_nonloopback_clients(tmp_path: Path) -> None:
    app, _executor = _app(tmp_path)
    headers = {
        "Content-Type": "application/json",
        server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE,
    }
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        duplicate = client.post(
            "/api/reviews",
            content=b'{"case_id":"a","case_id":"b"}',
            headers=headers,
        )
        assert duplicate.status_code == 422

    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("192.0.2.10", 51001),
    ) as client:
        blocked = client.get("/api/health")
        assert blocked.status_code == 403


def test_local_app_rejects_forged_host_and_cross_origin(tmp_path: Path) -> None:
    app, _executor = _app(tmp_path)
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        forged_host = client.get("/api/health", headers={"Host": "attacker.example"})
        cross_origin = client.post(
            "/api/reviews",
            json=_request_payload(),
            headers={
                server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE,
                "Origin": "https://attacker.example",
            },
        )
        same_origin = client.post(
            "/api/reviews",
            json=_request_payload(),
            headers={
                server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE,
                "Origin": "http://127.0.0.1:18021",
            },
        )

    assert forged_host.status_code == 403
    assert cross_origin.status_code == 403
    assert same_origin.status_code == 200


def test_timeout_run_can_be_retrieved_after_worker_finishes_and_keeps_single_flight(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    finish = threading.Event()

    class _BlockingExecutor(_FakeExecutor):
        def execute(self, request: Any) -> dict[str, Any]:
            self.requests.append(request)
            started.set()
            finish.wait(timeout=5)
            return self.result

    executor = _BlockingExecutor()
    app, _ = _app(tmp_path, executor, run_timeout_seconds=1.0)
    headers = {server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE}
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        first = client.post("/api/reviews", json=_request_payload(), headers=headers)
        assert started.is_set()
        assert first.status_code == 202
        first_run = first.json()
        assert first_run["status"] == "running"
        assert first_run["terminal"] is False
        run_id = first_run["run_id"]

        pending = client.get(f"/api/runs/{run_id}", headers=headers)
        assert pending.status_code == 200
        assert pending.json()["status"] == "running"
        assert "result" not in pending.json()

        while_worker_lives = client.post(
            "/api/reviews",
            json=_request_payload(),
            headers=headers,
        )
        assert while_worker_lives.status_code == 503
        assert while_worker_lives.json()["error"]["retryable"] is True

        finish.set()
        for _ in range(50):
            if not app.state.execution_admission.locked():
                break
            time.sleep(0.02)
        assert not app.state.execution_admission.locked()

        completed = client.get(f"/api/runs/{run_id}", headers=headers)
        assert completed.status_code == 200
        completed_run = completed.json()
        assert completed_run["status"] == "succeeded"
        assert completed_run["terminal"] is True
        assert completed_run["result"]["summary"]["status"] == "needs_review"


def test_run_status_requires_header_and_terminal_records_expire(
    tmp_path: Path,
) -> None:
    app, _executor = _app(tmp_path)
    headers = {server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE}
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        created = client.post("/api/reviews", json=_request_payload(), headers=headers)
        run_id = created.json()["run_id"]

        missing_header = client.get(f"/api/runs/{run_id}")
        assert missing_header.status_code == 403

        record = app.state.run_records[run_id]
        assert record.completed_monotonic is not None
        record.completed_monotonic -= server.RUN_RECORD_TTL_SECONDS + 1
        expired = client.get(f"/api/runs/{run_id}", headers=headers)
        assert expired.status_code == 404


def test_run_registry_persists_only_bounded_public_state_not_request_input(
    tmp_path: Path,
) -> None:
    app, _executor = _app(tmp_path)
    payload = _request_payload()
    payload["documents"][0]["content_text"] += "\nUNIQUE-RAW-INPUT-MARKER"
    headers = {server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE}
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        created = client.post("/api/reviews", json=payload, headers=headers)
        run_id = created.json()["run_id"]
        record = app.state.run_records[run_id]

        assert set(record.__slots__) == {
            "run_id",
            "created_monotonic",
            "completion",
            "status",
            "completed_monotonic",
            "result",
            "error_code",
        }
        public_json = json.dumps(record.public_payload(), sort_keys=True)
        assert "UNIQUE-RAW-INPUT-MARKER" not in public_json
        assert TOKEN not in public_json
        assert len(public_json.encode("utf-8")) <= server.MAX_PUBLIC_RESULT_BYTES


def test_server_refuses_forbidden_output_even_from_executor(tmp_path: Path) -> None:
    app, _executor = _app(
        tmp_path,
        _FakeExecutor(
            {
                "schema_version": server.ENVELOPE_SCHEMA_VERSION,
                "scope": server.SCOPE,
                "content_text": "must-not-leak",
            }
        ),
    )
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        response = client.post(
            "/api/reviews",
            json=_request_payload(),
            headers={server.INSPECTOR_HEADER: server.INSPECTOR_HEADER_VALUE},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "result" not in response.json()
    assert "must-not-leak" not in response.text


def test_ui_clears_previous_result_before_reading_or_submitting_next_input() -> None:
    html = Path(server.__file__).with_name("inspector.html").read_text(encoding="utf-8")
    reset_start = html.index("function resetExecutionView()")
    submit_start = html.index("async function submitReview(event)")
    submit_end = html.index("function newRunId()", submit_start)
    reset_source = html[reset_start:submit_start]
    submit_source = html[submit_start:submit_end]

    assert '$("results").classList.remove("on")' in reset_source
    assert "lastResponseEnvelope = null" in reset_source
    assert '"panel-rag"' in reset_source
    assert '"panel-findings"' in reset_source
    assert '"panel-xai"' in reset_source
    assert "replaceChildren()" in reset_source
    assert submit_source.index("resetExecutionView();") < submit_source.index(
        "const payload = readForm(true);"
    )
    assert "await awaitCompletedRun(run)" in submit_source


def test_actual_executor_preflights_data_before_constructing_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = (tmp_path / ".env").resolve()
    env_file.write_text("ANTHROPIC_API_KEY=not-read-by-test\n", encoding="utf-8")
    identity = server.EvidenceRuntimeReleaseIdentity.from_mapping(_identity_payload())
    manifest = server.DiagnosticManifest(
        identity=identity,
        data_revision="a" * 40,
        expected_production_law_count=58,
        observed_target_law_count=3,
        observed_article_count=1007,
        qdrant_point_count=1007,
        neo4j_article_count=1007,
        law_article_manifest_sha256="c" * 64,
        postgres_qdrant_ids_sha256="d" * 64,
        qdrant_point_ids_sha256="d" * 64,
        postgres_graph_ids_sha256="e" * 64,
        neo4j_article_ids_sha256="e" * 64,
    )
    events: list[str] = []

    class _Retriever:
        def preflight(self, _review_as_of: object) -> object:
            events.append("preflight")
            return identity

    class _Analyzer:
        def close(self) -> None:
            events.append("close")

    class _Result:
        def model_dump(self, **_kwargs: object) -> dict[str, Any]:
            return {
                "schema_version": "2.0",
                "status": "needs_review",
                "requires_human_review": True,
                "model_calls": [{"provider": "anthropic"}],
                "model_errors": [],
                "retrieval_results": [],
                "rag_evidence": [],
                "regulatory_findings": [],
                "validation_issues": [],
                "document_results": [{"extraction_provider": "anthropic"}],
            }

    monkeypatch.setattr(
        server,
        "ensure_llm_mode_available",
        lambda *_args, **_kwargs: SimpleNamespace(model="claude-test"),
    )
    monkeypatch.setattr(
        server.ActualInspectorExecutor,
        "_retriever",
        lambda _self: _Retriever(),
    )
    monkeypatch.setattr(
        server,
        "build_document_analyzer",
        lambda _env: events.append("analyzer") or _Analyzer(),
    )
    monkeypatch.setattr(
        server,
        "run_review_v2",
        lambda *_args, **_kwargs: (
            events.append("workflow")
            or {"pairing_result": {"document_set_status": "deterministic"}}
        ),
    )
    monkeypatch.setattr(server, "to_review_run_result_v2", lambda _state: _Result())

    executor = server.ActualInspectorExecutor(
        env_file=env_file,
        evidence_base_url="http://127.0.0.1:18012",
        service_token=TOKEN,
        diagnostic_manifest=manifest,
    )
    request = server.ServiceReviewRunRequestV2.model_validate(_request_payload())
    envelope = executor.execute(request)

    assert events == ["preflight", "analyzer", "workflow", "close"]
    assert envelope["scope"] == server.SCOPE
    assert envelope["release_ready"] is False
    assert envelope["summary"]["model_call_count"] == 1
    assert envelope["summary"]["mock_fallback_count"] == 0


def test_diagnostic_identity_overlay_is_absent_while_claude_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = (tmp_path / ".env").resolve()
    env_file.write_text("ANTHROPIC_API_KEY=not-read-by-test\n", encoding="utf-8")
    identity = server.EvidenceRuntimeReleaseIdentity.from_mapping(_identity_payload())
    original_loader = retrieval_module.load_released_evidence_runtime_identity
    events: list[str] = []

    def fake_init(
        self: object,
        _base_url: str,
        **_kwargs: object,
    ) -> None:
        assert retrieval_module.load_released_evidence_runtime_identity() == identity
        events.append("init")

    def fake_preflight(
        self: object,
        _review_as_of: object = None,
        **_kwargs: object,
    ) -> object:
        assert retrieval_module.load_released_evidence_runtime_identity() == identity
        events.append("preflight")
        return identity

    monkeypatch.setattr(server.ProductionHttpRegulationRetriever, "__init__", fake_init)
    monkeypatch.setattr(
        server.ProductionHttpRegulationRetriever,
        "preflight",
        fake_preflight,
    )
    retriever = server.DiagnosticProductionHttpRegulationRetriever(
        "http://127.0.0.1:18012",
        diagnostic_identity=identity,
        service_token=TOKEN,
        timeout_seconds=1,
        max_retries=0,
    )
    assert retrieval_module.load_released_evidence_runtime_identity is original_loader
    assert retriever.preflight() == identity
    assert retrieval_module.load_released_evidence_runtime_identity is original_loader
    assert events == ["init", "preflight"]
