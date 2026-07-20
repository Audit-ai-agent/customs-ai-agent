from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

import customs_ai_agent.adapters.retrieval as retrieval_module
from customs_ai_agent.contracts.evidence_runtime_release import canonical_json_bytes
from customs_ai_agent.runner import run_review_v2
from customs_ai_agent.schemas.models import ReviewRunRequestV2
from tools.agent_execution_inspector import server


TOKEN = "ThreeLawDiagnosticToken-0123456789-ABCDEFGHIJK"


def _scenario_catalog_paths() -> tuple[Path, Path]:
    directory = Path(server.__file__).parent
    return (
        directory / "scenario_catalog.json",
        directory / "scenario_catalog.schema.json",
    )


def _scenario_catalog() -> dict[str, Any]:
    catalog_path, _schema_path = _scenario_catalog_paths()
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def _scenario_by_id(catalog: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    return next(scenario for scenario in catalog["scenarios"] if scenario["id"] == scenario_id)


def _documents_by_id(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {document["document_id"]: document for document in scenario["documents"]}


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


def test_synthetic_scenario_catalog_matches_schema_and_service_contract() -> None:
    catalog_path, schema_path = _scenario_catalog_paths()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(catalog)
    loaded = server.load_scenario_catalog(
        catalog_path.resolve(),
        schema_path.resolve(),
    )

    assert loaded == catalog
    assert catalog["schema_version"] == server.SCENARIO_CATALOG_SCHEMA_VERSION
    assert "Claude API" in catalog["catalog_notice"]
    assert "정답 라벨" in catalog["expectation_disclaimer"]

    scenarios = catalog["scenarios"]
    identifiers = [scenario["id"] for scenario in scenarios]
    orders = [scenario["order"] for scenario in scenarios]
    assert identifiers == [
        "exact_hsk_missing_requirement",
        "exact_hsk_requirement_submitted",
        "quantity_mismatch",
        "invoice_total_mismatch",
        "origin_mismatch",
        "missing_hs_code",
        "prompt_injection_document_data",
        "duplicate_invoice_document_set",
    ]
    assert len(identifiers) == len(set(identifiers))
    assert orders == sorted(orders)
    assert len(orders) == len(set(orders))

    for scenario in scenarios:
        document_ids = [document["document_id"] for document in scenario["documents"]]
        assert 1 <= len(document_ids) <= server.MAX_DOCUMENTS
        assert len(document_ids) == len(set(document_ids))
        assert all(document["content_text"].strip() for document in scenario["documents"])
        assert (
            sum(len(document["content_text"]) for document in scenario["documents"])
            <= server.MAX_TOTAL_TEXT_CHARS
        )
        server.ServiceReviewRunRequestV2.model_validate(
            {
                "case_id": scenario["case_id"],
                "documents": scenario["documents"],
                "llm_mode": "llm",
                "review_run_id": "catalog_test",
                "review_as_of": "2026-07-10",
                "trade_type": scenario["trade_type"],
                "jurisdiction": "KR",
                "uploaded_requirement_document_codes": scenario[
                    "uploaded_requirement_document_codes"
                ],
            }
        )


def test_synthetic_scenario_mutations_are_explicit_and_minimal() -> None:
    catalog = _scenario_catalog()
    baseline = _scenario_by_id(catalog, "exact_hsk_missing_requirement")
    baseline_documents = _documents_by_id(baseline)
    assert list(baseline_documents) == ["doc_inv", "doc_pl", "doc_bl", "doc_co"]
    assert all(
        "0303540000" in baseline_documents[document_id]["content_text"]
        for document_id in ("doc_inv", "doc_co")
    )
    assert baseline["uploaded_requirement_document_codes"] == []

    submitted = _scenario_by_id(catalog, "exact_hsk_requirement_submitted")
    assert submitted["documents"] == baseline["documents"]
    assert submitted["uploaded_requirement_document_codes"] == ["CCCT_DOC_46E819BB87B27371"]
    assert "metadata-only" in submitted["injected_variation"]

    quantity = _documents_by_id(_scenario_by_id(catalog, "quantity_mismatch"))
    assert {
        document_id
        for document_id in baseline_documents
        if quantity[document_id] != baseline_documents[document_id]
    } == {"doc_pl"}
    assert "QTY: 1190 KG" in quantity["doc_pl"]["content_text"]
    assert "QTY: 1200 KG" in baseline_documents["doc_pl"]["content_text"]

    invoice_total = _documents_by_id(_scenario_by_id(catalog, "invoice_total_mismatch"))
    assert {
        document_id
        for document_id in baseline_documents
        if invoice_total[document_id] != baseline_documents[document_id]
    } == {"doc_inv"}
    assert "TOTAL AMOUNT: 11900" in invoice_total["doc_inv"]["content_text"]
    assert "AMOUNT: 12000" in invoice_total["doc_inv"]["content_text"]

    origin = _documents_by_id(_scenario_by_id(catalog, "origin_mismatch"))
    assert {
        document_id
        for document_id in baseline_documents
        if origin[document_id] != baseline_documents[document_id]
    } == {"doc_co"}
    assert "ORIGIN COUNTRY: Vietnam" in origin["doc_co"]["content_text"]
    assert "COUNTRY OF ORIGIN: Norway" in origin["doc_inv"]["content_text"]

    missing_hs = _documents_by_id(_scenario_by_id(catalog, "missing_hs_code"))
    assert {
        document_id
        for document_id in baseline_documents
        if missing_hs[document_id] != baseline_documents[document_id]
    } == {"doc_inv", "doc_co"}
    assert not any("0303540000" in document["content_text"] for document in missing_hs.values())
    assert "HS:" not in missing_hs["doc_inv"]["content_text"]
    assert "HS CODE:" not in missing_hs["doc_co"]["content_text"]

    injection = _documents_by_id(_scenario_by_id(catalog, "prompt_injection_document_data"))
    assert {
        document_id
        for document_id in baseline_documents
        if injection[document_id] != baseline_documents[document_id]
    } == {"doc_inv"}
    assert injection["doc_inv"]["content_text"].startswith(
        baseline_documents["doc_inv"]["content_text"]
    )
    assert "UNTRUSTED DOCUMENT NOTE" in injection["doc_inv"]["content_text"]

    duplicate = _documents_by_id(_scenario_by_id(catalog, "duplicate_invoice_document_set"))
    assert set(duplicate) == {*baseline_documents, "doc_inv_duplicate"}
    assert all(
        duplicate[document_id] == baseline_documents[document_id]
        for document_id in baseline_documents
    )
    assert duplicate["doc_inv_duplicate"]["declared_type"] == "invoice"
    assert "INVOICE NO: INV-2026-002" in duplicate["doc_inv_duplicate"]["content_text"]


def test_synthetic_scenarios_preserve_deterministic_workflow_meaning() -> None:
    expected = {
        "exact_hsk_missing_requirement": ("deterministic", set(), "partial"),
        "exact_hsk_requirement_submitted": ("deterministic", set(), "partial"),
        "quantity_mismatch": (
            "deterministic",
            {"QTY_MISMATCH_INVOICE_PACKING"},
            "partial",
        ),
        "invoice_total_mismatch": (
            "deterministic",
            {"INV_TOTAL_MISMATCH"},
            "partial",
        ),
        "origin_mismatch": ("deterministic", {"ORIGIN_MISMATCH"}, "partial"),
        "missing_hs_code": (
            "deterministic",
            {"HS_CODE_MISSING", "VALIDATION_INPUT_MISSING"},
            "partial",
        ),
        "prompt_injection_document_data": ("deterministic", set(), "partial"),
        "duplicate_invoice_document_set": (
            "ambiguous_document_set",
            {"DUPLICATE_DOCUMENT_TYPE"},
            "failed",
        ),
    }
    for scenario in _scenario_catalog()["scenarios"]:
        request = ReviewRunRequestV2.model_validate(
            {
                "case_id": scenario["case_id"],
                "documents": scenario["documents"],
                "llm_mode": "mock",
                "review_run_id": f"semantic_{scenario['id']}",
                "review_as_of": "2026-07-10",
                "trade_type": scenario["trade_type"],
                "jurisdiction": "KR",
                "uploaded_requirement_document_codes": scenario[
                    "uploaded_requirement_document_codes"
                ],
            }
        )
        state = run_review_v2(request)
        pairing_status, issue_codes, retrieval_status = expected[scenario["id"]]
        assert state["pairing_result"]["document_set_status"] == pairing_status
        assert {issue["issue_code"] for issue in state["validation_issues"]} == issue_codes
        assert state["retrieval_status"] == retrieval_status
        assert state["final_status"] != "completed"

        classified_types = {
            value["document_type"] for value in state["classified_documents"].values()
        }
        assert classified_types == {document["declared_type"] for document in scenario["documents"]}

        if scenario["id"] == "duplicate_invoice_document_set":
            assert state["rag_evidence"] == []
            assert {error["code"] for error in state["retrieval_errors"]} == {
                "AMBIGUOUS_INVOICE_RETRIEVAL_BLOCKED"
            }


def test_synthetic_scenario_catalog_has_no_credentials_paths_or_external_urls() -> None:
    serialized = json.dumps(
        _scenario_catalog(),
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()
    forbidden_literals = (
        "anthropic_api_key=",
        "authorization: bearer ",
        "c:\\users\\",
        "file://",
        "http://",
        "https://",
        "neo4j://",
        "postgresql://",
        "qdrant_api_key=",
        "sk-ant-",
    )
    assert not any(marker in serialized for marker in forbidden_literals)


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "https://example.invalid/source",
        "file:///tmp/source.json",
        "C:\\Users\\someone\\source.json",
    ],
)
def test_scenario_catalog_startup_rejects_external_provenance_and_paths(
    tmp_path: Path,
    unsafe_value: str,
) -> None:
    catalog = _scenario_catalog()
    catalog["scenarios"][0]["purpose"] += f" {unsafe_value}"
    tampered_path = (tmp_path / "tampered-scenario-catalog.json").resolve()
    tampered_path.write_text(
        json.dumps(catalog, ensure_ascii=False),
        encoding="utf-8",
    )
    _catalog_path, schema_path = _scenario_catalog_paths()

    with pytest.raises(
        server.InspectorConfigurationError,
        match="scenario_catalog_non_synthetic_marker_detected",
    ):
        server.load_scenario_catalog(tampered_path, schema_path.resolve())


def test_scenario_catalog_startup_rejects_secret_markers(tmp_path: Path) -> None:
    catalog = _scenario_catalog()
    catalog["scenarios"][0]["purpose"] += " sk-ant-" + "synthetic-marker"
    tampered_path = (tmp_path / "secret-bearing-scenario-catalog.json").resolve()
    tampered_path.write_text(
        json.dumps(catalog, ensure_ascii=False),
        encoding="utf-8",
    )
    _catalog_path, schema_path = _scenario_catalog_paths()

    with pytest.raises(
        server.InspectorConfigurationError,
        match="scenario_catalog_secret_marker_detected",
    ):
        server.load_scenario_catalog(tampered_path, schema_path.resolve())


def test_scenario_catalog_endpoint_is_loopback_no_store_and_never_executes(
    tmp_path: Path,
) -> None:
    app, executor = _app(tmp_path)
    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("127.0.0.1", 51001),
    ) as client:
        response = client.get("/api/scenarios")
        forged_host = client.get(
            "/api/scenarios",
            headers={"Host": "attacker.example"},
        )
        cross_origin = client.get(
            "/api/scenarios",
            headers={"Origin": "https://attacker.example"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert response.json()["schema_version"] == server.SCENARIO_CATALOG_SCHEMA_VERSION
    assert forged_host.status_code == 403
    assert cross_origin.status_code == 403
    assert executor.requests == []

    with TestClient(
        app,
        base_url="http://127.0.0.1:18021",
        client=("192.0.2.10", 51001),
    ) as nonlocal_client:
        blocked = nonlocal_client.get("/api/scenarios")
    assert blocked.status_code == 403


def test_scenario_picker_selects_without_mutation_or_network_and_loads_explicitly() -> None:
    html = Path(server.__file__).with_name("inspector.html").read_text(encoding="utf-8")
    selection_handler = '$("scenario-select").addEventListener("change", renderScenarioDetails);'
    render_start = html.index("function renderScenarioDetails()")
    render_end = html.index("function markLoadedScenarioModified()", render_start)
    load_start = html.index("function loadSelectedScenario()")
    load_end = html.index("async function loadScenarioCatalog()", load_start)
    render_source = html[render_start:render_end]
    load_source = html[load_start:load_end]

    assert selection_handler in html
    assert 'id="load-scenario" type="button"' in html
    assert "apiJson(" not in render_source
    assert "fetch(" not in render_source
    assert '$("case-id").value =' not in render_source
    assert '$("document-editors").innerHTML =' not in render_source

    assert load_source.index("if (submissionLocked)") < load_source.index("resetExecutionView();")
    assert load_source.index("resetExecutionView();") < load_source.index(
        '$("case-id").value = payload.case_id;'
    )
    assert "newRunId();" in load_source
    assert "supportedReviewAsOf()" in load_source
    assert "submissionGeneration" not in load_source
    assert "activeSubmissionGeneration" not in load_source
    assert "apiJson(" not in load_source
    assert "fetch(" not in load_source
    assert "/api/reviews" not in load_source
    assert "requestSubmit" not in load_source
    assert ".submit(" not in load_source

    busy_start = html.index("function setBusy(busy)")
    busy_end = html.index("function updateElapsed()", busy_start)
    busy_source = html[busy_start:busy_end]
    assert '$("scenario-select").disabled = busy' in busy_source
    assert '$("load-scenario").disabled =' in busy_source
    assert "submissionLocked" in load_source
    assert "markLoadedScenarioModified();" in html
    assert "loadedScenarioModified = true;" in html


def test_scenario_picker_is_memory_only_escapes_metadata_and_has_valid_javascript(
    tmp_path: Path,
) -> None:
    html = Path(server.__file__).with_name("inspector.html").read_text(encoding="utf-8")
    lower = html.casefold()
    assert "localstorage" not in lower
    assert "sessionstorage" not in lower
    assert "indexeddb" not in lower
    assert "${escapeHtml(scenario.purpose)}" in html
    assert "${escapeHtml(scenario.injected_variation)}" in html
    assert "${escapeHtml(scenario.expected_checkpoint)}" in html
    assert '$("scenario-catalog-notice").textContent = catalog.catalog_notice;' in html
    assert (
        '$("scenario-expectation-disclaimer").textContent = catalog.expectation_disclaimer;'
    ) in html

    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    script_path = tmp_path / "inspector.js"
    script_path.write_text(script, encoding="utf-8")
    node = shutil.which("node")
    assert node is not None, "Node.js is required to syntax-check the Inspector JavaScript"
    completed = subprocess.run(
        [node, "--check", str(script_path)],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


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
    assert "await awaitCompletedRun(run, generation)" in submit_source


def test_ui_serializes_overlapping_submissions_and_generation_guards_every_await() -> None:
    html = Path(server.__file__).with_name("inspector.html").read_text(encoding="utf-8")
    await_run_start = html.index("async function awaitCompletedRun(")
    await_run_end = html.index("async function loadHealth()", await_run_start)
    submit_start = html.index("async function submitReview(event)")
    submit_end = html.index("function newRunId()", submit_start)
    await_run_source = html[await_run_start:await_run_end]
    submit_source = html[submit_start:submit_end]

    lock_guard = submit_source.index("if (submissionLocked)")
    lock_set = submit_source.index("submissionLocked = true;")
    generation_set = submit_source.index("const generation = ++submissionGeneration;")
    busy_set = submit_source.index("setBusy(true);")
    first_await = submit_source.index("await ")
    reset = submit_source.index("resetExecutionView();")
    assert lock_guard < lock_set < generation_set < busy_set < first_await
    assert lock_guard < reset

    assert submit_source.count("await ") == 3
    assert submit_source.count("requireCurrentSubmission(generation);") >= 4
    assert submit_source.index(
        "requireCurrentSubmission(generation);",
        submit_source.index("await captureInputDigests(payload)"),
    ) < submit_source.index("lastInputDigests = inputDigests;")
    assert submit_source.index(
        "requireCurrentSubmission(generation);",
        submit_source.index('await apiJson("/api/reviews"'),
    ) < submit_source.index("const envelope = await awaitCompletedRun")
    assert "if (isCurrentSubmission(generation) && error !== STALE_SUBMISSION)" in submit_source
    assert "finally {\n    if (isCurrentSubmission(generation)) {" in submit_source

    assert await_run_source.count("await ") == 2
    assert await_run_source.count("requireCurrentSubmission(generation);") >= 3
    assert await_run_source.index(
        "requireCurrentSubmission(generation);",
        await_run_source.index("await delay(pollDelay);"),
    ) < await_run_source.index("const nextRun = await apiJson")
    assert await_run_source.index(
        "requireCurrentSubmission(generation);",
        await_run_source.index("const nextRun = await apiJson"),
    ) < await_run_source.index("run = nextRun;")


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
