"""Run controlled Functional-UAT dependency faults without persisting raw input.

This is deliberately separate from the real end-to-end matrix.  It uses the
same Functional-UAT settings and internal HTTP surface, but replaces exactly
one dependency at a time so provider/Data outages can be verified without
stopping the live local services or pretending that an outage was real.

It never writes a scenario transcript, prints a secret, or retains document
bodies.  Its one-line JSON output contains only check names and safe counts.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol, cast
from uuid import uuid4

from fastapi.testclient import TestClient

from customs_ai_agent.adapters.llm import DocumentAnalysis, LLMProviderError
from customs_ai_agent.scenario_transcript import ScenarioTranscriptRoot
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate, ClaudeFieldCandidate
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.profiles import (
    SERVICE_MODE_HEADER,
    _DeterministicDocumentAnalyzer,
    ProfileRuntime,
)
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings


class FaultMatrixError(RuntimeError):
    """A controlled check did not show its defined safe behavior."""


class _Analyzer(Protocol):
    model_id: str

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: object | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis: ...


class _FailingAnalyzer:
    model_id = "controlled-extraction-failure"

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: object | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis:
        del document, transcript, focus_fields
        raise LLMProviderError("controlled dependency fault")


class _DeterministicAnalyzer:
    """Use the bounded extractor to avoid an extra provider call in fault lanes."""

    model_id = "controlled-deterministic-analyzer"

    def __init__(self) -> None:
        self._delegate = _DeterministicDocumentAnalyzer()
        self.calls = 0

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: object | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis:
        self.calls += 1
        return self._delegate.analyze(
            document,
            transcript=cast(Any, transcript),
            focus_fields=focus_fields,
        )


class _InvalidQuoteAnalyzer:
    model_id = "controlled-invalid-source-quote"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: object | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis:
        del transcript
        document_id = str(document["document_id"])
        self.calls.append((document_id, focus_fields))
        return DocumentAnalysis(
            candidate=ClaudeDocumentCandidate(
                document_type="invoice",
                classification_source_quote="COMMERCIAL INVOICE",
                fields=[
                    ClaudeFieldCandidate(
                        name="invoice_no",
                        value="INV-NOT-IN-QUOTE",
                        source_quote="Invoice No: INV-FAULT-001",
                    )
                ],
            ),
            model_id=self.model_id,
        )


class _ReextractFailureAnalyzer:
    model_id = "controlled-reextract-failure"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: object | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis:
        del transcript, focus_fields
        self.calls += 1
        if self.calls > 1:
            raise LLMProviderError("controlled dependency fault")
        return DocumentAnalysis(
            candidate=ClaudeDocumentCandidate(
                document_type="invoice",
                classification_source_quote="COMMERCIAL INVOICE",
                fields=[
                    ClaudeFieldCandidate(
                        name="invoice_no",
                        value="INV-FAULT-001",
                        source_quote="Invoice No: INV-FAULT-001",
                    )
                ],
            ),
            model_id=self.model_id,
        )


class _FailingResponder:
    def __init__(self) -> None:
        self.calls = 0

    def respond(self, *_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        self.calls += 1
        raise LLMProviderError("controlled dependency fault")


@dataclass(slots=True)
class _StaticRetriever:
    status: str
    calls: int = 0

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": self.status,
            "knowledge_snapshot_id": "controlled-fault",
            "results": [],
        }

    def record_retrieve_skipped(self, *, review_run_id: str) -> None:
        """Match the production retriever without creating a transcript.

        The fault matrix deliberately has no Functional-UAT transcript root.
        A no-HS retrieval plan must therefore remain a no-op here while still
        exercising the same ProfileRuntime branch as the service.
        """

        del review_run_id


def _headers(settings: ServiceSettings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.bearer_token.get_secret_value()}",
        SERVICE_MODE_HEADER: "functional-uat",
    }


def _review_payload(run_suffix: str, *, paired_documents: bool = False) -> dict[str, Any]:
    documents = [
        {
            "document_id": "invoice",
            "filename": "invoice.txt",
            "content_text": (
                "COMMERCIAL INVOICE\n"
                "Invoice No: INV-FAULT-001\n"
                "Total Amount: USD 20\n"
                "Country of Origin: KR\n"
                "Description: Widget | HS Code: 123456 | Quantity: 2 PCS | Unit: PCS | "
                "Unit Price: USD 10 | Amount: USD 20"
            ),
        }
    ]
    if paired_documents:
        documents.append(
            {
                "document_id": "packing",
                "filename": "packing.txt",
                "content_text": (
                    "PACKING LIST\n"
                    "Packing No: PL-FAULT-001\n"
                    "Total Gross Weight: 100 KG\n"
                    "Description: Widget | HS Code: 123456 | Quantity: 2 PCS | Unit: PCS | "
                    "Gross Weight: 100 KG"
                ),
            }
        )
    return {
        "case_id": f"case-fault-{run_suffix}",
        "review_run_id": f"run-fault-{run_suffix}",
        "review_as_of": "2026-08-05",
        "trade_type": "import",
        "jurisdiction": "KR",
        "documents": documents,
    }


def _new_runtime(settings: ServiceSettings) -> tuple[ProfileRuntime, tuple[object, object | None]]:
    # An explicit disabled root prevents these controlled, synthetic fault
    # payloads from entering Functional-UAT transcripts.
    runtime = ProfileRuntime(settings, transcript_root=ScenarioTranscriptRoot())
    return runtime, (runtime._analyzer, runtime._grounded_responder)


def _close_originals(originals: tuple[object, object | None]) -> None:
    for item in originals:
        close = getattr(item, "close", None)
        if callable(close):
            close()


def _review_http(
    settings: ServiceSettings, runtime: ProfileRuntime, payload: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    app = create_app(
        settings,
        runtime_factory=lambda _settings: runtime,
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/reviews", headers=_headers(settings), json=payload
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise FaultMatrixError("agent_response_not_json") from exc
    if not isinstance(body, dict):
        raise FaultMatrixError("agent_response_not_object")
    return response.status_code, body


def _make_result(
    settings: ServiceSettings, *, analyzer: _Analyzer, retriever: _StaticRetriever
) -> tuple[ProfileRuntime, tuple[object, object | None], dict[str, Any]]:
    runtime, originals = _new_runtime(settings)
    runtime._analyzer = analyzer
    runtime._grounded_responder = None
    runtime._retriever = retriever
    status, body = _review_http(
        settings, runtime, _review_payload(uuid4().hex, paired_documents=True)
    )
    if status != 200:
        _close_originals(originals)
        raise FaultMatrixError("controlled_result_creation_failed")
    return runtime, originals, body


@contextmanager
def _data_stub(*, status: str, response_mode: str) -> Iterator[str]:
    """A loopback-only Data contract stub which discards all request content."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return None

        def _send(self, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header(SERVICE_MODE_HEADER, response_mode)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            self._send({"status": "ready"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            try:
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                request_id = request.get("request_id") if isinstance(request, Mapping) else None
            except (UnicodeDecodeError, ValueError):
                request_id = None
            self._send(
                {
                    "schema_version": "3.0",
                    "request_id": request_id,
                    "status": status,
                    "knowledge_snapshot_id": "controlled-fault",
                    "results": [],
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        if isinstance(host, bytes):
            host = host.decode("ascii")
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _unused_loopback_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        _host, port = probe.getsockname()
    return f"http://127.0.0.1:{port}"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise FaultMatrixError(code)


def _check_extraction_failure(settings: ServiceSettings) -> dict[str, Any]:
    runtime, originals = _new_runtime(settings)
    runtime._analyzer = _FailingAnalyzer()
    try:
        status, body = _review_http(settings, runtime, _review_payload(uuid4().hex))
    finally:
        _close_originals(originals)
    _require(status == 502, "extraction_failure_status")
    _require(
        body.get("error", {}).get("code") == "review_execution_failed", "extraction_failure_code"
    )
    return {"check": "claude_extraction_safe_failure", "status": "passed"}


def _check_report_and_assistant_fallback(settings: ServiceSettings) -> dict[str, Any]:
    runtime, originals = _new_runtime(settings)
    analyzer = _DeterministicAnalyzer()
    responder = _FailingResponder()
    runtime._analyzer = analyzer
    runtime._grounded_responder = responder  # type: ignore[assignment]
    runtime._retriever = _StaticRetriever("ok")
    payload = _review_payload(uuid4().hex, paired_documents=True)
    app = create_app(
        settings,
        runtime_factory=lambda _settings: runtime,
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            review = client.post(
                "/internal/agent/v2/reviews", headers=_headers(settings), json=payload
            )
            _require(review.status_code == 200, "report_fallback_status")
            result = review.json()
            _require(
                isinstance(result.get("report", {}).get("sections"), list)
                and len(result["report"]["sections"]) == 8,
                "report_fallback_sections",
            )
            answer = client.post(
                "/internal/agent/v2/assistant/answers",
                headers=_headers(settings),
                json={
                    "schema_version": "1.0",
                    "case_id": result["case_id"],
                    "review_run_id": result["review_run_id"],
                    "conversation_id": f"conversation-fault-{uuid4().hex}",
                    "review_result": result,
                    "question": "현재 사건의 수량 비교 결과를 설명해 주세요.",
                    "recent_turns": [],
                },
            )
            _require(answer.status_code == 200, "assistant_fallback_status")
            body = answer.json()
            _require(
                body.get("answer_scope") == "case_grounded" and bool(body.get("citations")),
                "assistant_fallback_grounding",
            )
    finally:
        _close_originals(originals)
    _require(responder.calls == 2, "fallback_responder_call_count")
    return {
        "check": "claude_report_and_assistant_fallback",
        "status": "passed",
        "responder_calls": responder.calls,
    }


def _check_invalid_quote_and_reextract_limit(settings: ServiceSettings) -> dict[str, Any]:
    runtime, originals = _new_runtime(settings)
    analyzer = _InvalidQuoteAnalyzer()
    runtime._analyzer = analyzer
    runtime._grounded_responder = None
    runtime._retriever = _StaticRetriever("ok")
    try:
        status, body = _review_http(settings, runtime, _review_payload(uuid4().hex))
    finally:
        _close_originals(originals)
    workflow = body.get("workflow", {})
    reextractions = workflow.get("reextractions", []) if isinstance(workflow, Mapping) else []
    warnings = body.get("document_analyses", [])
    warning_codes = {
        warning.get("code")
        for analysis in warnings
        if isinstance(analysis, Mapping)
        for warning in analysis.get("warnings", [])
        if isinstance(warning, Mapping)
    }
    _require(status == 200, "invalid_quote_status")
    _require("field_source_invalid" in warning_codes, "invalid_quote_warning")
    _require(len(reextractions) == 1 and len(analyzer.calls) == 2, "reextract_limit")
    _require(
        workflow.get("planner", {}).get("cycle_count", 3) <= 2
        and len(workflow.get("planner", {}).get("actions", [])) <= 6,
        "planner_limit",
    )
    return {
        "check": "invalid_source_quote_and_single_reextract",
        "status": "passed",
        "analyzer_calls": len(analyzer.calls),
    }


def _check_reextract_failure_status(settings: ServiceSettings) -> dict[str, Any]:
    runtime, originals = _new_runtime(settings)
    analyzer = _ReextractFailureAnalyzer()
    runtime._analyzer = analyzer
    runtime._grounded_responder = None
    runtime._retriever = _StaticRetriever("ok")
    try:
        status, body = _review_http(settings, runtime, _review_payload(uuid4().hex))
    finally:
        _close_originals(originals)
    _require(status == 200, "reextract_failure_status")
    _require(
        body.get("execution_status") == "partial"
        and body.get("workflow", {}).get("status") == "partial",
        "reextract_failure_propagation",
    )
    _require(analyzer.calls == 2, "reextract_failure_limit")
    return {
        "check": "reextract_failure_partial_status",
        "status": "passed",
        "analyzer_calls": analyzer.calls,
    }


def _check_data_statuses(settings: ServiceSettings) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for status, expected_execution, expected_status in (
        ("partial", "partial", "needs_review"),
        ("failed", "failed", "blocked"),
    ):
        with _data_stub(status=status, response_mode="functional-uat") as origin:
            fault_settings = replace(settings, evidence_base_url=origin)
            runtime, originals = _new_runtime(fault_settings)
            runtime._analyzer = _DeterministicAnalyzer()
            runtime._grounded_responder = None
            try:
                http_status, body = _review_http(
                    fault_settings, runtime, _review_payload(uuid4().hex)
                )
            finally:
                _close_originals(originals)
        _require(http_status == 200, f"data_{status}_http_status")
        _require(
            body.get("execution_status") == expected_execution
            and body.get("status") == expected_status,
            f"data_{status}_status_propagation",
        )
        results.append({"check": f"data_{status}_propagation", "status": "passed"})
    return results


def _check_data_dependency_failures(settings: ServiceSettings) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    cases: list[tuple[str, str]] = []
    with _data_stub(status="ok", response_mode="local-development") as origin:
        cases.append(("data_mode_mismatch_safe_failure", origin))
        for label, target in cases:
            fault_settings = replace(settings, evidence_base_url=target)
            runtime, originals = _new_runtime(fault_settings)
            runtime._analyzer = _DeterministicAnalyzer()
            runtime._grounded_responder = None
            try:
                status, body = _review_http(fault_settings, runtime, _review_payload(uuid4().hex))
            finally:
                _close_originals(originals)
            _require(status == 502, f"{label}_status")
            _require(
                body.get("error", {}).get("code") == "review_execution_failed", f"{label}_code"
            )
            checks.append({"check": label, "status": "passed"})
    fault_settings = replace(settings, evidence_base_url=_unused_loopback_url())
    runtime, originals = _new_runtime(fault_settings)
    runtime._analyzer = _DeterministicAnalyzer()
    runtime._grounded_responder = None
    try:
        status, body = _review_http(fault_settings, runtime, _review_payload(uuid4().hex))
    finally:
        _close_originals(originals)
    _require(status == 502, "data_transport_safe_failure_status")
    _require(
        body.get("error", {}).get("code") == "review_execution_failed",
        "data_transport_safe_failure_code",
    )
    checks.append({"check": "data_transport_safe_failure", "status": "passed"})
    return checks


def main() -> int:
    # The checks intentionally exercise 502 paths.  Keep their service-log
    # entries out of this compact command result; each result is asserted and
    # summarized below instead.
    logging.getLogger("customs_ai_agent.service").disabled = True
    try:
        settings = ServiceSettings.from_environment()
        _require(settings.service_mode == "functional-uat", "functional_uat_mode_required")
        checks = [
            _check_extraction_failure(settings),
            _check_report_and_assistant_fallback(settings),
            _check_invalid_quote_and_reextract_limit(settings),
            _check_reextract_failure_status(settings),
            *_check_data_statuses(settings),
            *_check_data_dependency_failures(settings),
        ]
    except ServiceConfigurationError:
        # This tool deliberately does not load ``_local`` inputs itself.
        # Requiring the normal, dot-sourced profile importer keeps the secret
        # boundary in one place and makes a missing profile actionable without
        # echoing an environment variable name or value.
        print(
            json.dumps(
                {
                    "functional_uat_fault_matrix": "failed",
                    "reason": "functional_uat_profile_not_loaded_or_invalid",
                }
            )
        )
        return 2
    except FaultMatrixError as exc:
        print(json.dumps({"functional_uat_fault_matrix": "failed", "reason": str(exc)}))
        return 1
    except Exception:
        print(json.dumps({"functional_uat_fault_matrix": "failed", "reason": "unexpected_fault"}))
        return 1
    print(
        json.dumps(
            {
                "functional_uat_fault_matrix": "passed",
                "controlled_dependency_faults": True,
                "checks": checks,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
