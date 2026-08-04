from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import customs_ai_agent.runtime as runtime_module
from customs_ai_agent.adapters.retrieval import MockRegulationRetriever
from customs_ai_agent.agentic import runtime_release_identity, to_review_run_result_v3
from customs_ai_agent.schemas.models import UserCorrection
from customs_ai_agent.schemas.v3 import (
    HumanDecisionV1,
    ReviewRevisionRequestV1,
    stable_object_id,
)
from tests.test_llm_workflow import StaticAnalyzer
from tests.test_v3_bounded_agent import _request


class _CountingRetriever(MockRegulationRetriever):
    expected_identity = SimpleNamespace(runtime_release_v3_sha256="e" * 64)

    def __init__(self) -> None:
        self.retrieve_calls = 0
        self.preflight_calls = 0

    def preflight(self, *_args, **_kwargs):
        self.preflight_calls += 1
        return self.expected_identity

    def policy_bindings(self, _request_id):
        return {}

    def retrieve(self, request):
        self.retrieve_calls += 1
        return super().retrieve(request)


class _ReuseAnalyzer(StaticAnalyzer):
    model_id = "claude-sonnet-5"

    def __init__(self) -> None:
        super().__init__()
        identity = runtime_release_identity(
            model_id=self.model_id,
            evidence_snapshot_id="mock-knowledge-v1",
            evidence_runtime_v3_sha256="e" * 64,
        )
        self._prompt_sha256 = identity.extraction_prompt_sha256
        self._schema_sha256 = identity.extraction_schema_sha256

    def analyze(self, document):
        analysis = super().analyze(document)
        return replace(
            analysis,
            metadata=replace(
                analysis.metadata,
                model_id=self.model_id,
                prompt_sha256=self._prompt_sha256,
                output_schema_sha256=self._schema_sha256,
            ),
        )


def _runtime(analyzer: _ReuseAnalyzer, retriever: _CountingRetriever):
    runtime = object.__new__(runtime_module.AgentRuntime)
    runtime._closed = False
    runtime._config = SimpleNamespace()
    runtime._analyzer = analyzer
    runtime._retriever_v3 = retriever
    runtime._assertion_verifier = None
    runtime._uat_capability = None
    runtime._secret_markers = ()
    return runtime


def _revision(runtime, analyzer, retriever):
    parent_request = _request(review_run_id="reuse-parent")
    service_request = runtime_module._project_service_request_v3_to_v2(parent_request)
    v2_result, rejections, _state = runtime._execute_review_v3(
        service_request,
        request=parent_request,
        retriever=retriever,
    )
    assert rejections == []
    parent_result = to_review_run_result_v3(
        v2_result,
        parent_request,
        evidence_runtime_v3_sha256="e" * 64,
    )
    invoice = next(
        document
        for document in parent_result.document_results
        if document.document_type == "invoice"
    )
    invoice_no = next(field for field in invoice.fields if field.name == "invoice_no")
    correction = UserCorrection(
        document_id=invoice.document_id,
        field_path="fields.invoice_no",
        value=f"{invoice_no.value}-R1",
    )
    decision_payload = {
        "decision": "correct_field",
        "actor_id": "reviewer-1",
        "actor_role": "reviewer",
        "decided_at": "2026-07-14T00:00:00Z",
        "reason": "Confirmed against the submitted source.",
        "confirmed_reference_id": None,
        "correction": correction,
        "supporting_document": None,
    }
    decision = HumanDecisionV1(
        decision_id=stable_object_id("decision", decision_payload),
        **decision_payload,
    )
    revised = parent_request.model_copy(deep=True)
    revised.review_run_id = "reuse-child"
    return ReviewRevisionRequestV1(
        parent_result_sha256=parent_result.result_sha256,
        parent_input_manifest_sha256=parent_result.input_manifest_sha256,
        parent_result=parent_result,
        parent_request=parent_request,
        revised_request=revised,
        decisions=[decision],
    )


def test_child_reuses_parent_extraction_and_retrieves_evidence_once(monkeypatch):
    analyzer = _ReuseAnalyzer()
    retriever = _CountingRetriever()
    runtime = _runtime(analyzer, retriever)
    revision = _revision(runtime, analyzer, retriever)
    analyzer.calls.clear()
    retriever.retrieve_calls = 0

    original_projection = runtime_module.to_review_run_result_v3

    def deterministic_projection(*args, **kwargs):
        kwargs["analyzer"] = None
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "to_review_run_result_v3", deterministic_projection)
    result = runtime.revision_v3_json(
        json.dumps(revision.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
    )

    assert analyzer.calls == []
    assert retriever.retrieve_calls == 1
    assert len(result.child_result.model_calls) == len(revision.parent_result.model_calls) == 4
    reuse = next(
        event
        for event in result.child_result.audit_trail
        if event.event == "parent_extraction_reused"
    )
    assert reuse.stage == "extract_documents"
    assert reuse.metadata == {
        "parent_result_sha256": revision.parent_result_sha256,
        "aggregate_reuse_key_sha256": reuse.metadata["aggregate_reuse_key_sha256"],
        "reused_document_count": 4,
        "inherited_model_call_count": 4,
        "actual_document_model_call_count": 0,
    }
    assert len(reuse.metadata["aggregate_reuse_key_sha256"]) == 64
    assert (
        sum(event.stage == "normalize_documents" for event in result.child_result.audit_trail) == 1
    )
    assert (
        sum(event.stage == "retrieve_regulations" for event in result.child_result.audit_trail) == 1
    )
    assert sum(event.stage == "compliance_review" for event in result.child_result.audit_trail) == 1
    assert (
        sum(event.stage == "generate_xai_report" for event in result.child_result.audit_trail) == 1
    )
    assert (
        sum(
            event.stage == "route_to_human_or_complete" for event in result.child_result.audit_trail
        )
        == 1
    )


def test_revision_falls_back_when_parent_extraction_identity_changes(monkeypatch):
    analyzer = _ReuseAnalyzer()
    retriever = _CountingRetriever()
    runtime = _runtime(analyzer, retriever)
    revision = _revision(runtime, analyzer, retriever)
    analyzer.model_id = "claude-sonnet-5-changed"
    fallback_child = revision.parent_result
    observed = {"calls": 0}

    def fallback(*_args, **_kwargs):
        observed["calls"] += 1
        return fallback_child

    monkeypatch.setattr(runtime_module.AgentRuntime, "_review_v3_with_corrections", fallback)
    monkeypatch.setattr(runtime_module, "_assert_public_corrections_applied", lambda *_args: None)
    result = runtime.revision_v3_json(
        json.dumps(revision.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
    )

    assert observed["calls"] == 1
    assert result.child_result == fallback_child
