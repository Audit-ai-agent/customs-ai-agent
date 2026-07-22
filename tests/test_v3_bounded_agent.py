from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from customs_ai_agent.agentic import (
    AGENT_POLICY_ID,
    ActiveBoundedRuntimeOutcome,
    BoundedToolBroker,
    BoundedToolPolicyError,
    BoundedToolResult,
    DeterministicReevaluationResult,
    _deterministic_grounded_answer_plan,
    _render_grounded_claim,
    _rewrite_grounded_answer,
    _rewrite_report,
    activate_bounded_plan,
    answer_grounded_question,
    build_fact_ledger,
    claude_narrative,
    compile_shadow_plan,
    deterministic_narrative,
    execute_active_bounded_plan,
    to_review_run_result_v3,
    verified_requirement_codes_by_item,
)
from customs_ai_agent.adapters.llm import StructuredAgenticResult
from customs_ai_agent.fixtures.scenarios import get_scenario
from customs_ai_agent.runner import run_review_result_v2, run_review_v2, to_review_run_result_v2
from customs_ai_agent.schemas.models import (
    ReviewRunRequestV2,
    ServiceReviewRunRequestV2,
    UserCorrection,
)
from customs_ai_agent.schemas.v3 import (
    GroundedQuestionV1,
    HumanDecisionV1,
    ReviewRevisionRequestV1,
    ReviewGapV1,
    ServiceReviewRunRequestV3,
    SupportingDocumentV1,
    VerificationAssertionV1,
    stable_object_id,
)


@pytest.fixture(autouse=True)
def _fixed_test_build_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    import customs_ai_agent.agentic as agentic

    monkeypatch.setattr(
        agentic,
        "load_packaged_build_manifest",
        lambda **_kwargs: SimpleNamespace(
            git_sha="0" * 40,
            source_snapshot_sha256="0" * 64,
        ),
    )
    monkeypatch.setattr(agentic, "package_code_snapshot_sha256", lambda: "0" * 64)
    monkeypatch.setattr(
        agentic,
        "load_assertion_trust_store",
        lambda: SimpleNamespace(release_state="active", trust_store_sha256="b" * 64),
    )


def _request(*, review_run_id: str = "v3_parent_run") -> ServiceReviewRunRequestV3:
    scenario = get_scenario("happy_path")
    return ServiceReviewRunRequestV3(
        case_id=scenario["case_id"],
        documents=scenario["documents"],
        llm_mode="llm",
        review_run_id=review_run_id,
        review_as_of="2026-07-14",
        trade_type="import",
        jurisdiction="KR",
        document_language="en",
        assertion_cutoff_at="2026-07-22T00:00:00Z",
    )


def _v3_result():
    request = _request()
    result_v2 = run_review_result_v2(
        ReviewRunRequestV2(
            scenario_id="happy_path",
            review_run_id=request.review_run_id,
            review_as_of=request.review_as_of,
        )
    )
    return (
        request,
        result_v2,
        to_review_run_result_v3(
            result_v2,
            request,
            evidence_runtime_v3_sha256="e" * 64,
        ),
    )


def _assertion(
    *,
    method: str = "trusted_connector",
    source_sha256: str = "a" * 64,
    asserted_at: str = "2026-06-02T00:00:00Z",
    authority_id: str | None = None,
    authority_release: str | None = None,
) -> VerificationAssertionV1:
    if method == "human_approver":
        asserted_by = "reviewer-1"
        default_authority_id = (
            "backend:principal-sha256:" + hashlib.sha256(asserted_by.encode("utf-8")).hexdigest()
        )
        default_authority_release = "backend-human-decision-policy/v1"
    else:
        asserted_by = "connector-service"
        default_authority_id = "connector:issuer-registry"
        default_authority_release = "connector-policy:issuer-registry/v1"
    payload = {
        "method": method,
        "subject_document_id": "support-1",
        "asserted_by": asserted_by,
        "asserted_at": asserted_at,
        "authority_id": authority_id or default_authority_id,
        "authority_release": (
            default_authority_release if authority_release is None else authority_release
        ),
        "source_reference_sha256": source_sha256,
    }
    return VerificationAssertionV1(
        assertion_id=stable_object_id("assert", payload),
        **payload,
    )


def _supporting_document(
    *,
    hs_codes: list[str] | None = None,
    assertion: VerificationAssertionV1 | None = None,
) -> SupportingDocumentV1:
    return SupportingDocumentV1(
        document_id="support-1",
        document_code="REQ_CERT",
        issuer="Synthetic issuer",
        document_number="SYN-001",
        issued_on="2026-06-01",
        valid_until="2026-12-31",
        subject_item_ids=["item-1"],
        hs_codes=hs_codes or ["0303540000"],
        source_sha256="a" * 64,
        verification_assertions=[assertion or _assertion()],
    )


def test_v3_forbids_both_legacy_requirement_code_shortcuts() -> None:
    request = _request()
    payload = request.model_dump(mode="python")
    payload["uploaded_requirement_document_codes"] = ["REQ_CERT"]
    with pytest.raises(ValidationError):
        ServiceReviewRunRequestV3.model_validate(payload)


def test_v3_contract_and_model_share_the_six_primary_document_limit() -> None:
    from customs_ai_agent.contracts import get_schema

    request = _request()
    schema = ServiceReviewRunRequestV3.model_json_schema(mode="validation")
    assert schema["properties"]["documents"]["maxItems"] == 6
    packaged = get_schema("backend", "v3", "review-request")
    packaged_document = packaged["$defs"]["TextDocumentInputV3"]
    assert packaged["properties"]["documents"]["maxItems"] == 6
    assert packaged_document["required"] == ["document_id", "filename", "content_text"]
    assert packaged_document["properties"]["image_pages"]["maxItems"] == 0
    assert packaged_document["properties"]["pdf_pages"]["type"] == "null"
    payload = request.model_dump(mode="python", warnings=False)
    template = payload["documents"][0]
    payload["documents"] = [
        {**template, "document_id": f"primary-{index}", "filename": f"primary-{index}.txt"}
        for index in range(7)
    ]
    with pytest.raises(ValidationError, match="document count"):
        ServiceReviewRunRequestV3.model_validate(payload)


def test_supporting_metadata_never_becomes_a_fifth_primary_claude_document() -> None:
    request, result_v2, _result_v3 = _v3_result()
    request.supporting_documents = [_supporting_document()]
    from customs_ai_agent.runtime import _project_service_request_v3_to_v2

    projected = _project_service_request_v3_to_v2(request)
    assert len(projected.documents) == 4
    assert {document.document_id for document in projected.documents}.isdisjoint(
        {document.document_id for document in request.supporting_documents}
    )
    assert result_v2.validation_issues == []


def test_v3_to_v2_projection_drops_every_v3_only_field() -> None:
    import customs_ai_agent.runtime as runtime_module

    request = _request()
    projected = runtime_module._project_service_request_v3_to_v2(request)
    assert isinstance(projected, ServiceReviewRunRequestV2)
    payload = projected.model_dump(mode="python", warnings=False)
    assert set(payload) == set(ServiceReviewRunRequestV2.model_fields)
    assert "supporting_documents" not in payload
    assert "document_language" not in payload
    assert "assertion_cutoff_at" not in payload
    assert payload["uploaded_requirement_document_codes"] == []

    payload = request.model_dump(mode="python")
    payload["documents"][0]["requirement_document_codes"] = ["REQ_CERT"]
    with pytest.raises(ValidationError):
        ServiceReviewRunRequestV3.model_validate(payload)


def test_supporting_document_requires_exact_item_hsk_source_and_released_policy() -> None:
    policy = {
        "REQ_CERT": {
            "required_document_code": "REQ_CERT",
            "verification_policy": "external_verification_required",
            "policy_source_sha256": "b" * 64,
            "policy_release": "requirements-v1",
            "approved": True,
        }
    }
    verified, rejected = verified_requirement_codes_by_item(
        [_supporting_document()],
        case_id=_request().case_id,
        review_as_of=_request().review_as_of,
        assertion_cutoff_at=_request().assertion_cutoff_at,
        retrieval_results=[{"item_id": "item-1", "resolved_hsk10": "0303540000"}],
        policy_bindings=policy,
        assertion_verifier=lambda **_kwargs: True,
    )
    assert verified == {"item-1": ["REQ_CERT"]}
    assert rejected == []

    wrong_hsk, rejected_hsk = verified_requirement_codes_by_item(
        [_supporting_document(hs_codes=["0303550000"])],
        case_id=_request().case_id,
        review_as_of=_request().review_as_of,
        assertion_cutoff_at=_request().assertion_cutoff_at,
        retrieval_results=[{"item_id": "item-1", "resolved_hsk10": "0303540000"}],
        policy_bindings=policy,
        assertion_verifier=lambda **_kwargs: True,
    )
    assert wrong_hsk == {}
    assert any("scope_mismatch" in reason for reason in rejected_hsk)

    wrong_source, rejected_source = verified_requirement_codes_by_item(
        [_supporting_document(assertion=_assertion(source_sha256="c" * 64))],
        case_id=_request().case_id,
        review_as_of=_request().review_as_of,
        assertion_cutoff_at=_request().assertion_cutoff_at,
        retrieval_results=[{"item_id": "item-1", "resolved_hsk10": "0303540000"}],
        policy_bindings=policy,
        assertion_verifier=lambda **_kwargs: True,
    )
    assert wrong_source == {}
    assert any("assertion_missing" in reason for reason in rejected_source)


def test_human_assertion_after_legal_review_date_is_valid_until_immutable_job_cutoff() -> None:
    policy = {
        "REQ_CERT": {
            "required_document_code": "REQ_CERT",
            "verification_policy": "human_only",
            "policy_source_sha256": "b" * 64,
            "policy_release": "requirements-v1",
            "approved": True,
        }
    }
    document = _supporting_document(
        assertion=_assertion(
            method="human_approver",
            asserted_at="2026-07-20T09:00:00Z",
        )
    )
    verified, rejected = verified_requirement_codes_by_item(
        [document],
        case_id=_request().case_id,
        review_as_of=_request().review_as_of,
        assertion_cutoff_at=_request().assertion_cutoff_at,
        retrieval_results=[{"item_id": "item-1", "resolved_hsk10": "0303540000"}],
        policy_bindings=policy,
        assertion_verifier=lambda **_kwargs: True,
    )
    assert verified == {"item-1": ["REQ_CERT"]}
    assert rejected == []

    future_document = _supporting_document(
        assertion=_assertion(
            method="human_approver",
            asserted_at="2026-07-23T00:00:00Z",
        )
    )
    future_verified, future_rejected = verified_requirement_codes_by_item(
        [future_document],
        case_id=_request().case_id,
        review_as_of=_request().review_as_of,
        assertion_cutoff_at=_request().assertion_cutoff_at,
        retrieval_results=[{"item_id": "item-1", "resolved_hsk10": "0303540000"}],
        policy_bindings=policy,
        assertion_verifier=lambda **_kwargs: True,
    )
    assert future_verified == {}
    assert any("assertion_missing" in reason for reason in future_rejected)


def test_test_issuer_assertion_can_never_verify_a_production_requirement() -> None:
    payload = {
        "method": "test_issuer",
        "subject_document_id": "support-1",
        "asserted_by": "synthetic-fixture",
        "asserted_at": "2026-06-02T00:00:00Z",
        "authority_id": "test:fixture-issuer",
        "authority_release": "synthetic-policy-v1",
        "source_reference_sha256": "a" * 64,
    }
    assertion = VerificationAssertionV1(
        assertion_id=stable_object_id("assert", payload),
        **payload,
    )
    policy = {
        "REQ_CERT": {
            "required_document_code": "REQ_CERT",
            "verification_policy": "presence_only",
            "policy_source_sha256": "b" * 64,
            "policy_release": "requirements-v1",
            "approved": True,
        }
    }
    verified, rejected = verified_requirement_codes_by_item(
        [_supporting_document(assertion=assertion)],
        case_id=_request().case_id,
        review_as_of=_request().review_as_of,
        assertion_cutoff_at=_request().assertion_cutoff_at,
        retrieval_results=[{"item_id": "item-1", "resolved_hsk10": "0303540000"}],
        policy_bindings=policy,
        assertion_verifier=lambda **_kwargs: True,
    )
    assert verified == {}
    assert rejected == ["support-1:trusted_verification_assertion_missing"]


@pytest.mark.parametrize(
    ("authority_id", "authority_release"),
    [
        (f"backend-user:sha256:{'b' * 64}", "backend-human-decision-policy/v1"),
        (f"backend:principal-sha256:{'b' * 64}", ""),
        (f"backend:principal-sha256:{'b' * 64}", "caller-selected-policy/v9"),
    ],
)
def test_human_assertion_namespace_and_policy_release_are_exact(
    authority_id: str,
    authority_release: str,
) -> None:
    with pytest.raises(ValidationError):
        _assertion(
            method="human_approver",
            authority_id=authority_id,
            authority_release=authority_release,
        )


def test_unreleased_profile_keeps_clean_result_human_review_required() -> None:
    _request_value, _result_v2, result = _v3_result()
    assert result.assessment == "no_issue_detected"
    assert result.review_disposition == "human_review_required"
    assert result.assurance_level == "production_candidate"
    assert result.completion_eligibility.eligible is False
    assert result.runtime_release.release_profile_state == "unreleased"
    assert result.runtime_release.kill_switch_active is True
    assert result.required_actions
    assert len(result.action_trace) <= 6
    assert len(result.action_trace) == len(result.observations)
    assert all(0 <= item.cycle <= 1 for item in result.action_trace)


def test_narrative_renderer_rejects_model_authored_text_and_citations() -> None:
    _request_value, result_v2, _result = _v3_result()
    facts = build_fact_ledger(result_v2)
    canonical = deterministic_narrative(result_v2, facts)
    rewrites = [
        {
            "section": section,
            "source_claim_id": claim.claim_id,
            "frame": "plain",
        }
        for section in (
            "summary",
            "document_consistency",
            "regulatory_review",
            "required_actions",
        )
        for claim in getattr(canonical, section)
    ]
    rewrites[0]["text"] = "세관이 통관을 승인했습니다."
    rewrites[0]["citations"] = []
    call = StructuredAgenticResult(
        value={"rewrites": rewrites},
        model_id="claude-test",
        prompt_sha256="a" * 64,
        output_schema_sha256="b" * 64,
        request_id="request-1",
        message_id="message-1",
        input_tokens=10,
        output_tokens=10,
        latency_ms=1,
    )
    with pytest.raises(ValueError, match="frame selection is invalid"):
        _rewrite_report(canonical, rewrites, call=call)


def test_narrative_renderer_accepts_only_typed_code_owned_frame_and_exact_citations() -> None:
    _request_value, result_v2, _result = _v3_result()
    canonical = deterministic_narrative(result_v2, build_fact_ledger(result_v2))
    values = [
        {
            "section": section,
            "source_claim_id": claim.claim_id,
            "frame": "summary_brief" if section == "summary" else "plain",
        }
        for section in (
            "summary",
            "document_consistency",
            "regulatory_review",
            "required_actions",
        )
        for claim in getattr(canonical, section)
    ]
    call = StructuredAgenticResult(
        value={"claims": values},
        model_id="claude-test",
        prompt_sha256="a" * 64,
        output_schema_sha256="b" * 64,
        request_id="request-1",
        message_id="message-1",
        input_tokens=10,
        output_tokens=10,
        latency_ms=1,
    )
    report = _rewrite_report(canonical, values, call=call)
    assert report.generation_mode == "claude"
    assert report.summary[0].text.startswith("요약하면, ")

    invalid_cases = []
    unknown = [dict(item) for item in values]
    unknown[0]["source_claim_id"] = "claim_" + "f" * 64
    invalid_cases.append((unknown, "unknown typed claim"))
    wrong_section = [dict(item) for item in values]
    wrong_section[0]["section"] = "required_actions"
    invalid_cases.append((wrong_section, "unknown typed claim"))
    wrong_frame = [dict(item) for item in values]
    wrong_frame[0]["frame"] = "action_owner"
    invalid_cases.append((wrong_frame, "not allowed"))
    duplicate = [dict(item) for item in values]
    duplicate[1] = dict(duplicate[0])
    invalid_cases.append((duplicate, "identity is invalid"))
    missing = [dict(item) for item in values[1:]]
    invalid_cases.append((missing, "cover every canonical claim"))
    extra_field = [dict(item) for item in values]
    extra_field[0]["canonical_text"] = canonical.summary[0].text
    invalid_cases.append((extra_field, "frame selection is invalid"))
    for invalid, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _rewrite_report(canonical, invalid, call=call)


def test_claude_narrative_boundary_accepts_only_frame_selections() -> None:
    _request_value, result_v2, _result = _v3_result()
    facts = build_fact_ledger(result_v2)
    canonical = deterministic_narrative(result_v2, facts)

    class FrameSelectionAnalyzer:
        def run_bounded_agentic_task(self, *_args, **kwargs):
            assert kwargs["task_id"] == "customs_narrative_report"
            item_schema = kwargs["output_schema"]["properties"]["claims"]["items"]
            assert item_schema["required"] == ["section", "source_claim_id", "frame"]
            assert "text" not in item_schema["properties"]
            assert "citations" not in item_schema["properties"]
            canonical_claims = kwargs["untrusted_context"]["canonical_claims"]
            assert all("citations" not in item for item in canonical_claims)
            claims = [
                {
                    "section": item["section"],
                    "source_claim_id": item["source_claim_id"],
                    "frame": item["allowed_frames"][-1],
                }
                for item in canonical_claims
            ]
            return StructuredAgenticResult(
                value={"claims": claims},
                model_id="claude-test",
                prompt_sha256="a" * 64,
                output_schema_sha256="b" * 64,
                request_id="request-narrative-frame",
                message_id="message-narrative-frame",
                input_tokens=10,
                output_tokens=10,
                latency_ms=1,
            )

    report, _call = claude_narrative(  # type: ignore[arg-type]
        FrameSelectionAnalyzer(),
        result_v2,
        facts,
    )
    assert report.generation_mode == "claude"
    for section in ("summary", "document_consistency", "regulatory_review", "required_actions"):
        canonical_claims = getattr(canonical, section)
        rendered_claims = getattr(report, section)
        assert [item.citations for item in rendered_claims] == [
            item.citations for item in canonical_claims
        ]
        assert all(
            rendered.text.endswith(source.text)
            for rendered, source in zip(rendered_claims, canonical_claims, strict=True)
        )


def test_invalid_narrative_and_chat_each_make_exactly_two_claude_calls_then_fallback() -> None:
    request, result_v2, baseline = _v3_result()

    class InvalidTypedOutputAnalyzer:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def run_bounded_agentic_task(self, *_args, **kwargs):
            task_id = kwargs["task_id"]
            self.calls[task_id] = self.calls.get(task_id, 0) + 1
            value = (
                {"suggestions": []} if task_id == "customs_review_coordinator" else {"claims": []}
            )
            return StructuredAgenticResult(
                value=value,
                model_id=baseline.runtime_release.model_id,
                prompt_sha256=(
                    baseline.runtime_release.chat_prompt_sha256
                    if task_id == "customs_case_answer"
                    else "a" * 64
                ),
                output_schema_sha256=(
                    baseline.runtime_release.chat_schema_sha256
                    if task_id == "customs_case_answer"
                    else "b" * 64
                ),
                request_id=f"request-{task_id}-{self.calls[task_id]}",
                message_id=f"message-{task_id}-{self.calls[task_id]}",
                input_tokens=10,
                output_tokens=10,
                latency_ms=1,
            )

    narrative_analyzer = InvalidTypedOutputAnalyzer()
    result = to_review_run_result_v3(
        result_v2,
        request,
        evidence_runtime_v3_sha256="e" * 64,
        analyzer=narrative_analyzer,  # type: ignore[arg-type]
    )
    assert narrative_analyzer.calls["customs_review_coordinator"] == 1
    assert narrative_analyzer.calls["customs_narrative_report"] == 2
    assert result.narrative_report.generation_mode == "deterministic_fallback"

    chat_analyzer = InvalidTypedOutputAnalyzer()
    answer = answer_grounded_question(
        GroundedQuestionV1(
            question_id="question_" + "9" * 64,
            question="현재 심사 결과 상태는 무엇인가요?",
            review_result=result,
        ),
        analyzer=chat_analyzer,  # type: ignore[arg-type]
    )
    assert chat_analyzer.calls == {"customs_case_answer": 2}
    assert answer.status == "answered"
    assert answer.generation_mode == "deterministic_fallback"


def test_frame_renderer_uses_only_code_owned_prefix_and_canonical_text() -> None:
    canonical = "수출자는 수입자에게 100 USD를 지급했습니다."
    assert _render_grounded_claim(canonical, "plain", section="answer") == canonical
    assert (
        _render_grounded_claim(
            canonical,
            "answer_grounded",
            section="answer",
        )
        == f"확인된 Case 근거에 따르면, {canonical}"
    )
    with pytest.raises(ValueError, match="not allowed"):
        _render_grounded_claim(canonical, "summary_brief", section="answer")
    with pytest.raises(ValueError, match="not allowed"):
        _render_grounded_claim(
            canonical,
            "수입자는 수출자에게 100 USD를 지급했습니다.",
            section="answer",
        )


def test_case_answer_renderer_selects_frames_but_code_owns_text_and_citations() -> None:
    _request_value, _result_v2, result = _v3_result()
    question = GroundedQuestionV1(
        question_id="question_" + "8" * 64,
        question="현재 심사 결과 상태는 무엇인가요?",
        review_result=result,
    )
    plan = _deterministic_grounded_answer_plan(question)
    assert plan.status == "answered"
    values = [
        {"source_claim_id": claim.claim_id, "frame": "answer_grounded"} for claim in plan.claims
    ]
    call = StructuredAgenticResult(
        value={"claims": values},
        model_id=result.runtime_release.model_id,
        prompt_sha256=result.runtime_release.chat_prompt_sha256,
        output_schema_sha256=result.runtime_release.chat_schema_sha256,
        request_id="request-answer-frame",
        message_id="message-answer-frame",
        input_tokens=10,
        output_tokens=10,
        latency_ms=1,
    )
    answer = _rewrite_grounded_answer(
        plan,
        values,
        question_id=question.question_id,
        call=call,
    )
    assert answer.generation_mode == "claude"
    assert answer.answer.startswith("확인된 Case 근거에 따르면, ")
    expected_citations = {
        (citation.kind, citation.reference_id)
        for claim in plan.claims
        for citation in claim.citations
    }
    assert {(item.kind, item.reference_id) for item in answer.citations} == expected_citations

    invalid_cases = []
    unknown = [dict(item) for item in values]
    unknown[0]["source_claim_id"] = "answer_claim_" + "f" * 64
    invalid_cases.append((unknown, "unknown canonical claim"))
    wrong_frame = [dict(item) for item in values]
    wrong_frame[0]["frame"] = "summary_brief"
    invalid_cases.append((wrong_frame, "not allowed"))
    duplicate = [dict(item) for item in values]
    duplicate.append(dict(duplicate[0]))
    invalid_cases.append((duplicate, "identity is invalid"))
    missing = [dict(item) for item in values[1:]]
    invalid_cases.append((missing, "cover every canonical claim"))
    extra_field = [dict(item) for item in values]
    extra_field[0]["canonical_text"] = plan.claims[0].text
    invalid_cases.append((extra_field, "frame selection is invalid"))
    for invalid, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _rewrite_grounded_answer(
                plan,
                invalid,
                question_id=question.question_id,
                call=call,
            )


def test_case_answer_boundary_accepts_only_frame_selections() -> None:
    _request_value, _result_v2, result = _v3_result()
    question = GroundedQuestionV1(
        question_id="question_" + "7" * 64,
        question="현재 심사 결과 상태는 무엇인가요?",
        review_result=result,
    )

    class FrameSelectionAnalyzer:
        def run_bounded_agentic_task(self, *_args, **kwargs):
            assert kwargs["task_id"] == "customs_case_answer"
            item_schema = kwargs["output_schema"]["properties"]["claims"]["items"]
            assert item_schema["required"] == ["source_claim_id", "frame"]
            assert "text" not in item_schema["properties"]
            assert "citations" not in item_schema["properties"]
            canonical_claims = kwargs["untrusted_context"]["canonical_claims"]
            assert all("citations" not in item for item in canonical_claims)
            return StructuredAgenticResult(
                value={
                    "claims": [
                        {
                            "source_claim_id": item["source_claim_id"],
                            "frame": "answer_evidence",
                        }
                        for item in canonical_claims
                    ]
                },
                model_id=result.runtime_release.model_id,
                prompt_sha256=result.runtime_release.chat_prompt_sha256,
                output_schema_sha256=result.runtime_release.chat_schema_sha256,
                request_id="request-answer-boundary",
                message_id="message-answer-boundary",
                input_tokens=10,
                output_tokens=10,
                latency_ms=1,
            )

    answer = answer_grounded_question(question, analyzer=FrameSelectionAnalyzer())  # type: ignore[arg-type]
    assert answer.generation_mode == "claude"
    assert answer.answer.startswith("근거 확인 결과: ")
    assert answer.citations


def _active_profile() -> SimpleNamespace:
    return SimpleNamespace(
        release_state="approved",
        kill_switch_active=False,
        auto_completion_enabled=True,
        bounded_adaptive_active_enabled=True,
        agent_code_snapshot_sha256="0" * 64,
        profile_id="signed-active-test-profile",
        model_id="claude-test-release-20260722",
        model_revision="claude-test-release-20260722",
        extraction_prompt_sha256="1" * 64,
        coordinator_prompt_sha256="2" * 64,
        narrative_prompt_sha256="3" * 64,
        chat_prompt_sha256="4" * 64,
        extraction_schema_sha256="c" * 64,
        coordinator_schema_sha256="d" * 64,
        narrative_schema_sha256="e" * 64,
        chat_schema_sha256="f" * 64,
        request_schema_sha256="5" * 64,
        result_schema_sha256="6" * 64,
        policy_id="bounded-adaptive-customs-review/v1",
        tool_registry_id="customs-agent-tools/text-v1",
        profile_sha256="7" * 64,
        review_subject_sha256="8" * 64,
        evaluation_report_sha256="9" * 64,
        reviewer_trust_store_sha256="a" * 64,
        assertion_trust_store_sha256="b" * 64,
        approved_languages=["en"],
        approved_document_types=[
            "bill_of_lading",
            "certificate_of_origin",
            "invoice",
            "packing_list",
        ],
        approved_hsk10=["0303540000"],
        approved_trade_types=["import"],
    )


def _test_gap(kind: str, index: int, reference: str) -> ReviewGapV1:
    payload = {
        "gap_type": kind,
        "reason_code": f"test_gap_{index}",
        "source_reference_ids": [reference],
        "status": "open",
    }
    return ReviewGapV1(
        gap_id=stable_object_id("gap", payload),
        **payload,
    )


def test_active_bounded_executor_is_disabled_by_current_packaged_release() -> None:
    plan = compile_shadow_plan([_test_gap("field_uncertainty", 1, "field-1")])
    with pytest.raises(BoundedToolPolicyError, match="not release-authorized"):
        activate_bounded_plan(plan, policy_id=AGENT_POLICY_ID)


def test_active_bounded_executor_honors_approved_profile_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.agentic as agentic

    profile = _active_profile()
    profile.kill_switch_active = True
    monkeypatch.setattr(agentic, "load_release_profile", lambda: profile)
    plan = compile_shadow_plan([_test_gap("field_uncertainty", 1, "field-1")])
    with pytest.raises(BoundedToolPolicyError, match="not release-authorized"):
        activate_bounded_plan(plan, policy_id=AGENT_POLICY_ID)


def test_active_budget_exhaustion_forces_human_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.agentic as agentic

    request, result_v2, _ = _v3_result()
    monkeypatch.setattr(agentic, "load_release_profile", _active_profile)
    monkeypatch.setattr(
        agentic,
        "automatic_completion_unmet_reason_codes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(agentic, "_coordinator_identity_blockers", lambda *_args: [])
    monkeypatch.setattr(agentic, "_narrative_identity_blockers", lambda *_args: [])

    class Exhausted:
        def execute(self, *, plan, fact_ledger):
            assert plan.mode == "active"
            assert fact_ledger.entries
            return ActiveBoundedRuntimeOutcome(
                result=result_v2,
                supporting_document_rejections=(),
                action_trace=(),
                observations=(),
                resolved_gap_ids=(),
                failure_reason_codes=("agent_budget_exhausted",),
            )

    result = to_review_run_result_v3(
        result_v2,
        request,
        evidence_runtime_v3_sha256="e" * 64,
        active_executor=Exhausted(),
    )
    assert result.review_disposition == "human_review_required"
    assert result.execution_status == "partial"
    assert "agent_budget_exhausted" in {item.action_code for item in result.required_actions}
    assert any(
        gap.reason_code == "agent_budget_exhausted" and gap.status == "open"
        for gap in result.review_gaps
    )


def test_active_bounded_executor_runs_allowlisted_observation_then_code_reevaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.agentic as agentic

    gaps = [
        _test_gap("field_uncertainty", 1, "field-1"),
        _test_gap("evidence_incomplete", 2, "evidence-1"),
        _test_gap("validation_issue", 3, "issue-1"),
    ]
    monkeypatch.setattr(agentic, "load_release_profile", _active_profile)
    active = activate_bounded_plan(
        compile_shadow_plan(gaps),
        policy_id=AGENT_POLICY_ID,
    )
    observed_stages: list[tuple[str, ...]] = []

    def handler(reference_ids: tuple[str, ...]) -> BoundedToolResult:
        return BoundedToolResult(
            status="succeeded",
            reason_code="trusted_observation",
            output_reference_ids=(f"observed:{reference_ids[0]}",),
        )

    def reevaluate(**values):
        action = values["action"]
        observed_stages.append(values["affected_stages"])
        return DeterministicReevaluationResult(
            status="succeeded",
            reason_code="deterministic_stages_reexecuted",
            resolved_gap_ids=(action.gap_id,),
            output_reference_ids=(f"rerun:{action.gap_id}",),
        )

    traces, observations, resolved = execute_active_bounded_plan(
        active,
        broker=BoundedToolBroker(span_reread=handler, evidence_retry=handler),
        reevaluate=reevaluate,
        policy_id=AGENT_POLICY_ID,
    )
    assert active.mode == "active"
    assert len(traces) == len(observations) <= 6
    assert set(resolved) == {gaps[0].gap_id, gaps[1].gap_id}
    assert (
        "extraction",
        "normalization",
        "pairing",
        "validation",
        "compliance",
        "outcome",
    ) in observed_stages
    assert ("evidence", "compliance", "outcome") in observed_stages
    # A validation issue is not mutable by an adaptive Tool and compiles to HITL.
    assert any(
        item.tool_name == "human.request/v1" and item.status == "succeeded" for item in traces
    )


def test_release_calibration_cannot_clear_pairing_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy

    import customs_ai_agent.runtime as runtime_module

    request = _request(review_run_id="v3_pairing_hold")
    state = run_review_v2(
        ReviewRunRequestV2(
            scenario_id="happy_path",
            review_run_id=request.review_run_id,
            review_as_of=request.review_as_of,
        )
    )
    state = copy.deepcopy(state)
    state["pairing_result"]["document_type_conflicts"] = [
        {"document_type": "invoice", "document_ids": ["doc_inv", "doc_inv_duplicate"]}
    ]
    state["requires_human_review"] = True
    monkeypatch.setattr(
        runtime_module,
        "automatic_completion_unmet_reason_codes",
        lambda *_args, **_kwargs: [],
    )

    executor = object.__new__(runtime_module._RuntimeBoundedAdaptiveExecutor)
    executor._request = request
    executor._retriever = SimpleNamespace(
        policy_bindings=lambda _request_id: {},
        expected_identity=SimpleNamespace(runtime_release_v3_sha256="e" * 64),
    )
    executor._analyzer = SimpleNamespace(model_id="claude-test-release-20260722")
    executor._assertion_verifier = None
    _next_state, result, _rejections = executor._finalize_state(state)

    assert result.status == "needs_review"
    assert result.requires_human_review is True
    assert "pairing_document_type_conflict" in runtime_module.extraction_completion_blockers(state)


def test_agent_runtime_service_entry_executes_authorized_evidence_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy

    import customs_ai_agent.agentic as agentic
    import customs_ai_agent.runtime as runtime_module
    from customs_ai_agent.adapters.llm import LLMError
    from customs_ai_agent.nodes.compliance import compliance_review
    from customs_ai_agent.nodes.report import generate_xai_report
    from customs_ai_agent.nodes.route import route_to_human_or_complete

    request = _request(review_run_id="v3_active_service_entry")
    workflow_request = ReviewRunRequestV2(
        scenario_id="happy_path",
        review_run_id=request.review_run_id,
        review_as_of=request.review_as_of,
    )
    released_state = run_review_v2(workflow_request)
    initial_state = copy.deepcopy(released_state)
    initial_state.update(
        {
            "rag_evidence": [],
            "retrieval_results": [],
            "retrieval_errors": [
                {
                    "code": "injected_retryable_failure",
                    "message": "Injected retryable Evidence failure.",
                }
            ],
            "retrieval_status": "failed",
            "retrieval_coverage": "unknown",
            "knowledge_snapshot_id": None,
            "requires_human_review": True,
        }
    )
    for node in (compliance_review, generate_xai_report, route_to_human_or_complete):
        initial_state.update(node(initial_state))
    initial_result = to_review_run_result_v2(initial_state)
    active_profile = SimpleNamespace(
        release_state="approved",
        kill_switch_active=False,
        auto_completion_enabled=True,
        bounded_adaptive_active_enabled=True,
        agent_code_snapshot_sha256="0" * 64,
        profile_id="signed-active-test-profile",
        model_id="claude-sonnet-5",
        model_revision="claude-sonnet-5-20260722",
        extraction_prompt_sha256="1" * 64,
        coordinator_prompt_sha256="2" * 64,
        narrative_prompt_sha256="3" * 64,
        chat_prompt_sha256="4" * 64,
        extraction_schema_sha256="c" * 64,
        coordinator_schema_sha256="d" * 64,
        narrative_schema_sha256="e" * 64,
        chat_schema_sha256="f" * 64,
        request_schema_sha256="5" * 64,
        result_schema_sha256="6" * 64,
        policy_id="bounded-adaptive-customs-review/v1",
        tool_registry_id="customs-agent-tools/text-v1",
        profile_sha256="7" * 64,
        review_subject_sha256="8" * 64,
        evaluation_report_sha256="9" * 64,
        reviewer_trust_store_sha256="a" * 64,
        assertion_trust_store_sha256="b" * 64,
        approved_languages=["en"],
        approved_document_types=[
            "bill_of_lading",
            "certificate_of_origin",
            "invoice",
            "packing_list",
        ],
        approved_hsk10=["0303540000"],
        approved_trade_types=["import"],
    )
    monkeypatch.setattr(agentic, "load_release_profile", lambda: active_profile)

    def release_reasons(result, *_args, require_completed_outcome=True, **_kwargs):
        reasons = []
        if result.retrieval_status != "ok" or result.retrieval_coverage != "complete":
            reasons.append("evidence_not_complete")
        if require_completed_outcome and (
            result.status != "completed" or result.requires_human_review
        ):
            reasons.append("deterministic_review_requires_human")
        return reasons

    monkeypatch.setattr(agentic, "automatic_completion_unmet_reason_codes", release_reasons)
    monkeypatch.setattr(
        runtime_module,
        "automatic_completion_unmet_reason_codes",
        release_reasons,
    )
    monkeypatch.setattr(agentic, "_coordinator_identity_blockers", lambda *_args: [])
    monkeypatch.setattr(agentic, "_narrative_identity_blockers", lambda *_args: [])
    original_compile = agentic.compile_shadow_plan
    monkeypatch.setattr(
        agentic,
        "compile_shadow_plan",
        lambda gaps: original_compile(
            [gap for gap in gaps if gap.gap_type == "evidence_incomplete"]
        ),
    )

    class Analyzer:
        model_id = "claude-sonnet-5"

        def run_bounded_agentic_task(self, *_args, **_kwargs):
            raise LLMError("injected_planner_unavailable", "test")

    class Retriever:
        expected_identity = SimpleNamespace(runtime_release_v3_sha256="e" * 64)

        def preflight(self, *_args, **_kwargs):
            return self.expected_identity

        def policy_bindings(self, _request_id):
            return {}

    retriever = Retriever()
    analyzer = Analyzer()

    def fake_execute(_self, *_args, **_kwargs):
        return initial_result, [], copy.deepcopy(initial_state)

    monkeypatch.setattr(runtime_module.AgentRuntime, "_execute_review_v3", fake_execute)
    monkeypatch.setattr(
        runtime_module.AgentRuntime,
        "_get_retriever_v3",
        lambda _self: retriever,
    )

    retrieval_keys = (
        "rag_evidence",
        "retrieval_request_id",
        "retrieval_request",
        "retrieval_status",
        "retrieval_coverage",
        "knowledge_snapshot_id",
        "retrieval_results",
        "retrieval_errors",
        "audit_trail",
    )
    monkeypatch.setattr(
        runtime_module,
        "retrieve_regulations",
        lambda *_args, **_kwargs: {
            key: copy.deepcopy(released_state[key]) for key in retrieval_keys
        },
    )

    runtime = object.__new__(runtime_module.AgentRuntime)
    runtime._analyzer = analyzer
    runtime._closed = False
    runtime._config = None
    runtime._retriever = None
    runtime._retriever_v3 = retriever
    runtime._secret_markers = ()
    runtime._assertion_verifier = None
    result = runtime.review_v3(request)

    evidence_actions = [
        item for item in result.action_trace if item.tool_name == "evidence.retrieve/v3"
    ]
    assert evidence_actions and all(item.status == "succeeded" for item in evidence_actions), [
        item.model_dump(mode="json") for item in result.observations
    ]
    assert result.retrieval_status == "ok"
    assert result.retrieval_coverage == "complete"
    assert result.validation_issues == initial_result.validation_issues
    assert result.regulatory_findings == initial_result.regulatory_findings
    assert any(
        gap.gap_type == "evidence_incomplete" and gap.status == "resolved"
        for gap in result.review_gaps
    )
    assert result.review_disposition == "auto_review_complete", {
        "eligibility": result.completion_eligibility.model_dump(mode="json"),
        "gaps": [(item.reason_code, item.status) for item in result.review_gaps],
        "observations": [(item.reason_code, item.status) for item in result.observations],
        "status": result.status,
        "human": result.requires_human_review,
    }


def test_actual_llm_uncalibrated_state_bootstraps_only_through_signed_code_owned_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import customs_ai_agent.agentic as agentic
    import customs_ai_agent.runtime as runtime_module
    from customs_ai_agent.adapters.retrieval import MockRegulationRetriever
    from customs_ai_agent.runtime import _project_service_request_v3_to_v2
    from tests.test_llm_workflow import StaticAnalyzer

    exact_model = "claude-sonnet-5-20260722"
    profile = _active_profile()
    profile.model_id = exact_model
    profile.model_revision = exact_model
    profile.approved_hsk10 = ["7318151000"]
    monkeypatch.setattr(agentic, "load_release_profile", lambda: profile)
    identity = agentic.runtime_release_identity(
        model_id=exact_model,
        evidence_snapshot_id="mock-knowledge-v1",
        evidence_runtime_v3_sha256="e" * 64,
    )
    for name in (
        "agent_code_snapshot_sha256",
        "extraction_prompt_sha256",
        "coordinator_prompt_sha256",
        "narrative_prompt_sha256",
        "chat_prompt_sha256",
        "extraction_schema_sha256",
        "coordinator_schema_sha256",
        "narrative_schema_sha256",
        "chat_schema_sha256",
        "result_schema_sha256",
        "policy_id",
        "tool_registry_id",
    ):
        setattr(profile, name, getattr(identity, name))
    from customs_ai_agent.contracts import schema_sha256

    profile.request_schema_sha256 = schema_sha256("backend", "v3", "review-request")

    class ReleaseAnalyzer(StaticAnalyzer):
        model_id = exact_model

        def analyze(self, document):
            analysis = super().analyze(document)
            return replace(
                analysis,
                metadata=replace(
                    analysis.metadata,
                    model_id=exact_model,
                    prompt_sha256=identity.extraction_prompt_sha256,
                    output_schema_sha256=identity.extraction_schema_sha256,
                ),
            )

    class ReleaseRetriever(MockRegulationRetriever):
        expected_identity = SimpleNamespace(runtime_release_v3_sha256="e" * 64)

        def preflight(self, *_args, **_kwargs):
            return self.expected_identity

        def policy_bindings(self, _request_id):
            return {}

    request = _request(review_run_id="v3_llm_release_bootstrap")
    service_request = _project_service_request_v3_to_v2(request)
    runtime = object.__new__(runtime_module.AgentRuntime)
    runtime._analyzer = ReleaseAnalyzer()
    runtime._secret_markers = ()
    runtime._assertion_verifier = None
    retriever = ReleaseRetriever()

    v2_result, rejections, state = runtime._execute_review_v3(
        service_request,
        request=request,
        retriever=retriever,
    )

    assert v2_result.model_calls
    bootstrap_blockers = agentic.automatic_completion_unmet_reason_codes(
        v2_result,
        request,
        agentic.runtime_release_identity(
            model_id=exact_model,
            evidence_snapshot_id=v2_result.knowledge_snapshot_id,
            evidence_runtime_v3_sha256="e" * 64,
        ),
        supporting_rejections=rejections,
        require_completed_outcome=False,
    )
    assert v2_result.status == "completed", {
        "release_blockers": bootstrap_blockers,
        "outcome_reasons": v2_result.report.summary.outcome_reasons,
        "extraction_blockers": runtime_module.extraction_completion_blockers(state),
    }
    assert v2_result.requires_human_review is False
    assert "llm_extraction_uncalibrated" not in v2_result.report.summary.outcome_reasons
    assert rejections == []
    for altered_calls in (
        v2_result.model_calls[:-1],
        [*v2_result.model_calls, v2_result.model_calls[0]],
    ):
        incomplete = v2_result.model_copy(update={"model_calls": altered_calls})
        reasons = agentic.automatic_completion_unmet_reason_codes(
            incomplete,
            request,
            agentic.runtime_release_identity(
                model_id=exact_model,
                evidence_snapshot_id=incomplete.knowledge_snapshot_id,
                evidence_runtime_v3_sha256="e" * 64,
            ),
            supporting_rejections=[],
            require_completed_outcome=False,
        )
        assert "claude_model_call_identity_incomplete" in reasons

    # Coordinator/narrative exact-call enforcement has separate tests.  Keep
    # this regression focused on the formerly impossible calibration bootstrap.
    monkeypatch.setattr(agentic, "_coordinator_identity_blockers", lambda *_args: [])
    monkeypatch.setattr(agentic, "_narrative_identity_blockers", lambda *_args: [])
    result = to_review_run_result_v3(
        v2_result,
        request,
        evidence_runtime_v3_sha256="e" * 64,
    )
    assert result.completion_eligibility.eligible is True
    assert result.review_disposition == "auto_review_complete"


def test_active_broker_rejects_unknown_tool_and_repeated_field_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.agentic as agentic

    broker = BoundedToolBroker(
        span_reread=lambda refs: BoundedToolResult(
            status="succeeded",
            reason_code="observed",
            output_reference_ids=refs,
        )
    )
    with pytest.raises(BoundedToolPolicyError, match="outside"):
        broker.execute("shell.execute/v1", ("field-1",))
    monkeypatch.setattr(agentic, "load_release_profile", _active_profile)
    gaps = [
        _test_gap("field_uncertainty", 1, "field-1"),
        _test_gap("field_uncertainty", 2, "field-1"),
    ]
    active = activate_bounded_plan(compile_shadow_plan(gaps), policy_id=AGENT_POLICY_ID)

    def reevaluate(**values):
        return DeterministicReevaluationResult(
            status="succeeded",
            reason_code="rerun",
            resolved_gap_ids=(values["action"].gap_id,),
        )

    traces, _observations, _resolved = execute_active_bounded_plan(
        active,
        broker=broker,
        reevaluate=reevaluate,
        policy_id=AGENT_POLICY_ID,
    )
    rereads = [item for item in traces if item.tool_name == "document.reread_span/v1"]
    assert [item.status for item in rereads] == ["succeeded", "deferred"]
    assert len(traces) <= active.tool_call_limit <= 6


def test_bounded_recovery_evaluation_meets_ninety_five_percent_without_policy_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.agentic as agentic

    monkeypatch.setattr(agentic, "load_release_profile", _active_profile)
    recovered = 0
    for index in range(20):
        gap = _test_gap("evidence_incomplete", index, f"evidence-{index}")
        active = activate_bounded_plan(
            compile_shadow_plan([gap]),
            policy_id=AGENT_POLICY_ID,
        )

        def retry(
            refs: tuple[str, ...],
            *,
            should_fail: bool = index == 19,
        ) -> BoundedToolResult:
            return BoundedToolResult(
                status="failed" if should_fail else "succeeded",
                reason_code="injected_failure" if should_fail else "retrieved",
                output_reference_ids=refs,
            )

        def reevaluate(**values):
            return DeterministicReevaluationResult(
                status="succeeded",
                reason_code="evidence_compliance_outcome_rerun",
                resolved_gap_ids=(values["action"].gap_id,),
            )

        _traces, _observations, resolved = execute_active_bounded_plan(
            active,
            broker=BoundedToolBroker(evidence_retry=retry),
            reevaluate=reevaluate,
            policy_id=AGENT_POLICY_ID,
        )
        recovered += bool(resolved)
    assert recovered / 20 >= 0.95


def test_revision_compiler_rejects_every_unapproved_request_change() -> None:
    parent_request, _result_v2, parent_result = _v3_result()
    first_document = parent_result.document_results[0]
    first_field = first_document.fields[0]
    correction = UserCorrection(
        document_id=first_document.document_id,
        field_path=f"fields.{first_field.name}",
        value=first_field.value,
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
    revised.review_run_id = "v3_child_run"
    revision = ReviewRevisionRequestV1(
        parent_result_sha256=parent_result.result_sha256,
        parent_input_manifest_sha256=parent_result.input_manifest_sha256,
        parent_result=parent_result,
        parent_request=parent_request,
        revised_request=revised,
        decisions=[decision],
    )
    assert revision.revised_request.review_run_id == "v3_child_run"

    tampered = revised.model_copy(deep=True)
    tampered.trade_type = "export"
    with pytest.raises(ValidationError, match="not compiled"):
        ReviewRevisionRequestV1(
            parent_result_sha256=parent_result.result_sha256,
            parent_input_manifest_sha256=parent_result.input_manifest_sha256,
            parent_result=parent_result,
            parent_request=parent_request,
            revised_request=tampered,
            decisions=[decision],
        )


def test_revision_never_claims_decision_applied_when_public_child_value_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.runtime as runtime_module

    parent_request, _result_v2, parent_result = _v3_result()
    first_document = parent_result.document_results[0]
    first_field = first_document.fields[0]
    correction = UserCorrection(
        document_id=first_document.document_id,
        field_path=f"fields.{first_field.name}",
        value=f"{first_field.value}-human-corrected",
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
    revised.review_run_id = "v3_child_public_value_mismatch"
    revision = ReviewRevisionRequestV1(
        parent_result_sha256=parent_result.result_sha256,
        parent_input_manifest_sha256=parent_result.input_manifest_sha256,
        parent_result=parent_result,
        parent_request=parent_request,
        revised_request=revised,
        decisions=[decision],
    )
    runtime = object.__new__(runtime_module.AgentRuntime)
    runtime._closed = False
    runtime._secret_markers = ()
    monkeypatch.setattr(
        runtime_module.AgentRuntime,
        "_review_v3_with_corrections",
        lambda *_args, **_kwargs: parent_result,
    )

    with pytest.raises(runtime_module.RuntimeConflictError) as captured:
        runtime.revision_v3_json(
            json.dumps(revision.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
        )

    assert captured.value.code == "correction_application_failed"


def test_revision_rejects_conflicting_terminal_human_decisions() -> None:
    parent_request, _result_v2, parent_result = _v3_result()
    revised = parent_request.model_copy(deep=True)
    revised.review_run_id = "v3_terminal_conflict_child"

    def terminal(kind: str, role: str, suffix: str) -> HumanDecisionV1:
        payload = {
            "decision": kind,
            "actor_id": f"reviewer-{suffix}",
            "actor_role": role,
            "decided_at": "2026-07-14T00:00:00Z",
            "reason": "Explicit terminal review decision.",
            "confirmed_reference_id": None,
            "correction": None,
            "supporting_document": None,
        }
        return HumanDecisionV1(
            decision_id=stable_object_id("decision", payload),
            **payload,
        )

    with pytest.raises(ValidationError, match="at most one terminal"):
        ReviewRevisionRequestV1(
            parent_result_sha256=parent_result.result_sha256,
            parent_input_manifest_sha256=parent_result.input_manifest_sha256,
            parent_result=parent_result,
            parent_request=parent_request,
            revised_request=revised,
            decisions=[
                terminal("approve_review", "approver", "approve"),
                terminal("reject_review", "reviewer", "reject"),
            ],
        )


def test_case_answer_is_grounded_or_explicitly_refused() -> None:
    _request_value, _result_v2, result = _v3_result()
    outcome = answer_grounded_question(
        GroundedQuestionV1(
            question_id="question_" + "1" * 64,
            question="현재 심사 결과 상태는 무엇인가요?",
            review_result=result,
        )
    )
    assert outcome.status == "answered"
    assert outcome.citations

    refused = answer_grounded_question(
        GroundedQuestionV1(
            question_id="question_" + "2" * 64,
            question="오늘 날씨를 알려줘",
            review_result=result,
        )
    )
    assert refused.status == "out_of_scope"
    assert refused.citations == []
